"""
The agent: an MCP client wired to Gemini's OpenAI-compatible API.

It ties together two protocols that were designed independently:

  1. **MCP** — how the agent discovers and invokes tools on `server.py`.
     The agent spawns the server as a subprocess and speaks JSON-RPC over its
     stdin/stdout.

  2. **OpenAI-style tool calling** — how the agent asks the LLM *which* tool to
     use. Gemini exposes an OpenAI-compatible endpoint, so we use the plain
     `openai` SDK against Google's base URL.

The bridge between them is small and lives in `mcp_tool_to_openai_schema()`:
an MCP tool declaration is already a name + description + JSON Schema, which is
exactly the shape OpenAI's `tools` parameter wants.

Usage:
    python agent.py "what is 17 * 23 + 4?"   # one-shot
    python agent.py                          # interactive chat
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, Tool
from openai import OpenAI, RateLimitError

# --- Configuration ---------------------------------------------------------

PROJECT_DIR = Path(__file__).parent
SERVER_SCRIPT = PROJECT_DIR / "server.py"

# Gemini speaks OpenAI's wire format at this URL, so the official `openai` SDK
# works unchanged — only the base URL and key differ.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MODEL = "gemini-flash-latest"

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools provided by an MCP server. "
    "Use a tool when it genuinely helps answer the question, and answer directly "
    "when it does not. Base your final answer on the tool results you receive."
)

# Safety net for the tool-calling loop: a confused model could otherwise ping-pong
# between tool calls forever.
MAX_TURNS = 8


# --- MCP -> OpenAI schema bridge -------------------------------------------


def mcp_tool_to_openai_schema(tool: Tool) -> dict[str, Any]:
    """Convert one MCP tool declaration into an OpenAI `tools` entry.

    MCP gives us `name`, `description` and `input_schema` (JSON Schema, derived
    from the server function's type hints). OpenAI wants the same three things
    nested under a "function" key. That is the entire translation.
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.input_schema,
        },
    }


def _serialize_tool_call(call: Any) -> dict[str, Any]:
    """Turn a tool call from the response back into a request-shaped dict.

    We rebuild it by hand instead of dumping the SDK object so that only fields
    the API accepts are sent back — with one Gemini-specific exception.

    Gemini's newer models attach an opaque `extra_content.google.thought_signature`
    to each tool call and *require* it to be echoed back in the follow-up request;
    omitting it is a 400. It is not part of the OpenAI spec, so the SDK parks it
    in `model_extra`. Against real OpenAI this key is simply absent.
    """
    serialized: dict[str, Any] = {
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.function.name,
            "arguments": call.function.arguments,
        },
    }

    extra_content = (call.model_extra or {}).get("extra_content")
    if extra_content:
        serialized["extra_content"] = extra_content

    return serialized


def render_tool_result(result: CallToolResult) -> str:
    """Flatten an MCP tool result into the plain text the LLM expects back.

    An MCP result is a list of content blocks (text, images, ...) plus an
    optional machine-readable `structured_content`. Our tools return strings, so
    the text blocks hold the answer; the structured fallback covers tools that
    return only JSON.
    """
    parts = [block.text for block in result.content if getattr(block, "text", None)]
    if parts:
        return "\n".join(parts)

    if result.structured_content is not None:
        return json.dumps(result.structured_content)

    return "(tool returned no output)"


# --- The tool-calling loop --------------------------------------------------


async def run_turn(
    llm: OpenAI,
    model: str,
    mcp_client: Client,
    openai_tools: list[dict[str, Any]],
    messages: list[dict[str, Any]],
) -> str:
    """Run one user turn to completion and return the assistant's final text.

    A "turn" may involve several round trips: the model can ask for tools, read
    the results, and then ask for more tools before it is ready to answer.
    `messages` is mutated in place so the conversation carries across turns.
    """
    for _ in range(MAX_TURNS):
        # 1. Ask the model what to do, advertising every MCP tool.
        try:
            response = llm.chat.completions.create(
                model=model,
                messages=messages,
                tools=openai_tools,
            )
        except RateLimitError:
            # The free Gemini tier allows only a handful of requests per minute,
            # and one question can cost several (one per tool-calling round).
            return "(rate limited by the Gemini API - wait a minute and try again)"

        message = response.choices[0].message

        # 2. Record the assistant's reply. We rebuild the dict by hand rather
        #    than dumping the SDK object, so only fields the API accepts go back.
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": message.content,
        }
        if message.tool_calls:
            assistant_message["tool_calls"] = [
                _serialize_tool_call(call) for call in message.tool_calls
            ]
        messages.append(assistant_message)

        # 3. No tool calls means the model answered directly — we are done.
        if not message.tool_calls:
            return message.content or "(no answer)"

        # 4. Otherwise dispatch each requested call to the MCP server and append
        #    one "tool" message per call. Every tool_call_id must be answered.
        for call in message.tool_calls:
            name = call.function.name
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}

            print(f"  [tool] {name}({json.dumps(arguments)})")

            try:
                result = await mcp_client.call_tool(name, arguments)
                # `is_error` means the tool itself failed (bad arguments, an
                # exception in the server). That is not fatal: we feed the error
                # text back and let the model correct itself.
                output = render_tool_result(result)
                if result.is_error:
                    output = f"Tool error: {output}"
            except Exception as exc:  # transport/protocol failure, unknown tool
                output = f"Tool error: {type(exc).__name__}: {exc}"

            print(f"  [tool] -> {output}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": output,
                }
            )

    return f"(stopped after {MAX_TURNS} tool-calling rounds without a final answer)"


# --- Wiring -----------------------------------------------------------------


async def main() -> None:
    load_dotenv(PROJECT_DIR / ".env")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in.")
    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)

    llm = OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)

    # Launch `server.py` as a subprocess and talk MCP over its stdio pipes.
    # `sys.executable` is this venv's Python, so the child gets the same deps.
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_SCRIPT)],
    )

    # Entering the client performs the MCP handshake; leaving it shuts the
    # server subprocess down.
    async with Client(stdio_client(server_params)) as mcp_client:
        # Tool *discovery*: we never hardcode the tool list. Add a tool to
        # server.py and the agent picks it up on the next run.
        tools = (await mcp_client.list_tools()).tools
        openai_tools = [mcp_tool_to_openai_schema(tool) for tool in tools]

        print(f"Connected to MCP server '{mcp_client.server_info.name}'.")
        print(f"Model: {model}")
        print("Tools:", ", ".join(tool.name for tool in tools), "\n")

        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        # One-shot mode: everything after the script name is the question.
        if len(sys.argv) > 1:
            query = " ".join(sys.argv[1:])
            print(f"You: {query}")
            messages.append({"role": "user", "content": query})
            answer = await run_turn(llm, model, mcp_client, openai_tools, messages)
            print(f"\nAssistant: {answer}")
            return

        # Interactive mode. History (and the server's in-memory notes) persist
        # across questions because both the message list and the subprocess live
        # for the whole session.
        print("Type a question, or 'exit' to quit.\n")
        while True:
            try:
                query = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not query:
                continue
            if query.lower() in {"exit", "quit"}:
                break

            messages.append({"role": "user", "content": query})
            answer = await run_turn(llm, model, mcp_client, openai_tools, messages)
            print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    asyncio.run(main())
