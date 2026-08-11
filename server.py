"""
MCP server exposing a few small, self-contained tools.

This is one half of the demo. It knows nothing about LLMs, Gemini, or the agent
that will call it — that is the whole point of MCP. A server just declares
"here are the tools I have, here is the shape of their inputs", and any MCP
client can discover and invoke them.

The `mcp` SDK does the protocol work for us:
  * `@mcp.tool()` registers a function as an MCP tool.
  * The function's *type hints* become the tool's JSON Schema (`input_schema`),
    which is exactly what an LLM needs to decide how to call it.
  * The function's *docstring* becomes the tool description the LLM reads.

Transport is stdio: this script talks JSON-RPC over stdin/stdout, and the agent
launches it as a subprocess. That means **nothing may be printed to stdout**
except protocol messages — use stderr if you need to debug.

Run directly (`python server.py`) and it will sit waiting for an MCP client on
stdin. Normally you don't run it yourself; `agent.py` spawns it for you.
"""

from __future__ import annotations

import ast
import operator
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from mcp.server import MCPServer

# The server object. `instructions` is optional prose shown to clients that ask
# what this server is for.
mcp = MCPServer(
    name="demo-tools",
    instructions=(
        "A tiny demo toolbox: arithmetic, current time in a timezone, "
        "and an in-memory notepad."
    ),
)


# ---------------------------------------------------------------------------
# Tool 1: calculate
# ---------------------------------------------------------------------------

# We deliberately do NOT use eval(). A tool's arguments come from an LLM, which
# in turn is influenced by untrusted user text, so the expression is parsed into
# an AST and only these node types are allowed through.
_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _evaluate(node: ast.AST) -> float:
    """Recursively evaluate a whitelisted arithmetic AST node."""
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)

    # A plain number literal, e.g. `42` or `3.5`.
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"only numbers are allowed, got {node.value!r}")
        return node.value

    # `a + b`, `a * b`, ...
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        # Keep `2 ** 10000000` from hanging the server.
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large (max 100)")
        return _BINARY_OPS[type(node.op)](left, right)

    # `-a`, `+a`
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_evaluate(node.operand))

    raise ValueError(f"unsupported syntax: {type(node).__name__}")


@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression and return the result.

    Supports + - * / // % ** and parentheses on numbers only. No variables,
    no function calls. Example: "(17 * 23) + 4".
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        # Raising turns into an MCP tool error, which the agent hands back to
        # the model so it can correct itself and retry.
        raise ValueError(f"could not parse {expression!r}: {exc.msg}") from exc

    try:
        result = _evaluate(tree)
    except ZeroDivisionError as exc:
        raise ValueError("division by zero") from exc

    # Render 6.0 as "6" so the model sees a clean number.
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return f"{expression} = {result}"


# ---------------------------------------------------------------------------
# Tool 2: get_current_time
# ---------------------------------------------------------------------------


@mcp.tool()
def get_current_time(timezone: str = "UTC") -> str:
    """Get the current date and time in an IANA timezone.

    `timezone` is a name like "UTC", "Europe/Berlin", or "Asia/Karachi".
    Defaults to UTC.
    """
    try:
        zone = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(
            f"unknown timezone {timezone!r}; use an IANA name such as 'Europe/Berlin'"
        ) from exc

    now = datetime.now(zone)
    return f"{now:%Y-%m-%d %H:%M:%S %Z} ({timezone})"


# ---------------------------------------------------------------------------
# Tools 3 & 4: a two-function notepad
# ---------------------------------------------------------------------------

# Intentionally in-memory: no database, no file, no setup. The notes live as
# long as this server subprocess does, i.e. for one `agent.py` session. Swapping
# this list for SQLite would not change anything the client sees, which is a
# nice illustration of the boundary MCP draws.
_notes: list[str] = []


@mcp.tool()
def add_note(text: str) -> str:
    """Save a short note. Notes are kept in memory for this session only."""
    text = text.strip()
    if not text:
        raise ValueError("note text must not be empty")

    _notes.append(text)
    return f"Saved note #{len(_notes)}: {text}"


@mcp.tool()
def list_notes() -> str:
    """List every note saved so far in this session."""
    if not _notes:
        return "No notes saved yet."

    return "\n".join(f"{i}. {note}" for i, note in enumerate(_notes, start=1))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Blocks, serving MCP over stdin/stdout until the client closes the pipe.
    mcp.run(transport="stdio")
