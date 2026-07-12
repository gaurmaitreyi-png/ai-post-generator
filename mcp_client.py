"""
MCP client bridge.

Connects the bot (as an MCP client) to the Agentic News MCP server over stdio,
discovers whatever tools the server advertises, and adapts them for Gemini:

    MCP tool schema  ->  Gemini function declaration     (so the LLM knows what exists)
    Gemini function_call  ->  MCP tools/call             (so the LLM can actually run it)

The point of doing it this way: the bot no longer hard-codes its own tools. It asks
the server what it can do. Any other MCP client could connect to the same server and
get the same capabilities.
"""

import json
import logging
import sys
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

# JSON-Schema keys that Gemini's function-declaration parser does not accept.
_STRIP_KEYS = {"title", "$schema", "additionalProperties", "definitions", "$defs", "default"}


def _clean_schema(node):
    """Recursively drop schema keys Gemini rejects."""
    if isinstance(node, dict):
        return {k: _clean_schema(v) for k, v in node.items() if k not in _STRIP_KEYS}
    if isinstance(node, list):
        return [_clean_schema(v) for v in node]
    return node


class MCPToolbox:
    """Owns the MCP session and exposes it in the shape the agent loop expects."""

    def __init__(self, server_script: str = "mcp_server.py"):
        self.server_script = server_script
        self._stack: AsyncExitStack | None = None
        self.session: ClientSession | None = None
        self.declarations: list[dict] = []

    async def connect(self) -> "MCPToolbox":
        """Start the MCP server as a subprocess and discover its tools."""
        self._stack = AsyncExitStack()
        params = StdioServerParameters(command=sys.executable, args=[self.server_script])
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()

        listed = await self.session.list_tools()
        self.declarations = [
            {
                "name": t.name,
                "description": (t.description or "").strip(),
                "parameters": _clean_schema(t.inputSchema),
            }
            for t in listed.tools
        ]
        logger.info(f"[mcp] connected; discovered tools: {[d['name'] for d in self.declarations]}")
        return self

    async def call_tool(self, name: str, args: dict) -> dict:
        """Invoke a tool on the MCP server and return its result as a dict."""
        if self.session is None:
            return {"error": "MCP session is not connected"}
        result = await self.session.call_tool(name, args or {})
        if getattr(result, "isError", False):
            return {"error": f"tool {name} failed on the server"}

        # FastMCP returns structured output when it can; otherwise text content.
        structured = getattr(result, "structuredContent", None)
        if structured:
            return structured.get("result", structured)
        for block in (result.content or []):
            text = getattr(block, "text", None)
            if text:
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {"result": text}
        return {"result": "done"}

    async def close(self):
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
            self.session = None
