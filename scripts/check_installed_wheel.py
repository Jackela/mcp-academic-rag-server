#!/usr/bin/env python3
"""Run installed console and real stdio protocol checks outside the source tree.

Run with the wheel installed in this Python environment. No live API requests.
"""

import asyncio
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def check_protocol(executable, directory):
    env = dict(os.environ)
    env.update(OPENAI_API_KEY="sk-offline-fixture-key-only", DATA_PATH=str(Path(directory) / "data"))
    params = StdioServerParameters(command=str(executable), args=[], env=env, cwd=directory)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            assert {t.name for t in tools.tools} == {
                "test_connection",
                "process_document",
                "query_documents",
                "validate_system",
            }
            result = await session.call_tool("test_connection", {"message": "installed-wheel-fixture"})
            assert "installed-wheel-fixture" in result.content[0].text
            bad = await session.call_tool("process_document", {"file_path": "/nonexistent/fixture.pdf"})
            assert "not found" in bad.content[0].text


def main():
    bin_dir = Path(sys.executable).resolve().parent
    # Virtualenv Python is often a symlink; console scripts live beside argv[0].
    bin_dir = Path(sys.executable).parent
    suffix = ".exe" if os.name == "nt" else ""
    with tempfile.TemporaryDirectory(
        prefix="installed-wheel-", dir=Path(__file__).resolve().parents[1] / ".wheel-smoke"
    ) as directory:
        for name in [
            "mcp-academic-rag-server",
            "academic-rag",
            "academic-rag-server",
            "mcp-academic-rag-server-secure",
            "mcp-academic-rag-server-dev",
        ]:
            executable = bin_dir / (name + suffix)
            subprocess.run([str(executable), "--help"], cwd=directory, check=True, capture_output=True)
        asyncio.run(check_protocol(bin_dir / ("mcp-academic-rag-server" + suffix), directory))
    print("Installed wheel console and MCP stdio contracts passed")


if __name__ == "__main__":
    (Path(__file__).resolve().parents[1] / ".wheel-smoke").mkdir(exist_ok=True)
    main()
