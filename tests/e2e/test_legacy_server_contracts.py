"""Retained minimal SDK transport and legacy secure registration remain callable."""

import os
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@pytest.mark.asyncio
async def test_minimal_entry_uses_actual_sdk_tool_objects_and_handlers(tmp_path):
    environment = dict(
        os.environ,
        OPENAI_API_KEY="sk-controlled-legacy-fixture",
        DATA_PATH=str(tmp_path / "data"),
        PYTHONPATH=str(Path(__file__).parents[2]),
    )
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "servers.mcp_server_minimal"], env=environment, cwd=str(tmp_path)
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        assert [tool.name for tool in tools.tools] == ["test_connection"]
        result = await session.call_tool("test_connection", {})
        assert not result.isError
        assert "connection successful" in result.content[0].text
        missing = await session.call_tool("unknown-tool", {})
        assert missing.isError


def test_secure_entry_uses_registered_legacy_server_instead_of_assigning_sdk_methods():
    from mcp.types import CallToolRequest, ListToolsRequest

    from servers import mcp_server

    assert ListToolsRequest in mcp_server.server.request_handlers
    assert CallToolRequest in mcp_server.server.request_handlers
