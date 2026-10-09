#!/usr/bin/env python3
"""Exercise an installed Docker image with real MCP stdio and no external network."""

import argparse
import asyncio
import subprocess
import uuid
from datetime import timedelta

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def check_protocol(image: str) -> None:
    name = "mcp-contract-" + uuid.uuid4().hex
    params = StdioServerParameters(
        command="docker",
        args=[
            "run",
            "--rm",
            "-i",
            "--network",
            "none",
            "--name",
            name,
            "-e",
            "OPENAI_API_KEY=sk-offline-fixture-only",
            image,
        ],
    )
    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=30)) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert {tool.name for tool in tools.tools} == {
                    "test_connection",
                    "validate_system",
                    "process_document",
                    "query_documents",
                }
                echo = await session.call_tool("test_connection", {"message": "container-offline-contract"})
                assert not echo.isError and "container-offline-contract" in echo.content[0].text
                missing = await session.call_tool("process_document", {"file_path": "/missing-fixture.pdf"})
                assert missing.isError and "not found" in missing.content[0].text
                invalid = await session.call_tool("process_document", {})
                assert invalid.isError
    finally:
        # Own this exact unique test container only. Normal stdio exit removes it.
        found = subprocess.run(["docker", "container", "inspect", name], capture_output=True, check=False, timeout=15)
        if found.returncode == 0:
            subprocess.run(["docker", "rm", "-f", name], check=True, capture_output=True, timeout=15)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    args = parser.parse_args()
    native_check = (
        "import os, torch, faiss, numpy as np, core; "
        "assert os.getuid() != 0; "
        "assert torch.tensor([1, 2]).sum().item() == 3; "
        "index = faiss.IndexFlatL2(2); index.add(np.array([[1, 0]], dtype='float32')); "
        "assert index.search(np.array([[1, 0]], dtype='float32'), 1)[1].tolist() == [[0]]; "
        "assert '/opt/venv/' in core.__file__"
    )
    subprocess.run(
        ["docker", "run", "--rm", "--network", "none", "--entrypoint", "python", args.image, "-c", native_check],
        check=True,
        capture_output=True,
        timeout=60,
    )
    subprocess.run(
        ["docker", "run", "--rm", "--network", "none", "--entrypoint", "python", args.image, "-m", "pip", "check"],
        check=True,
        capture_output=True,
        timeout=30,
    )
    subprocess.run(
        ["docker", "run", "--rm", "--network", "none", args.image, "--help"],
        check=True,
        capture_output=True,
        timeout=30,
    )
    rejected = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", args.image, "--validate-only"],
        check=False,
        capture_output=True,
        timeout=30,
    )
    assert rejected.returncode != 0 and b"OPENAI_API_KEY" in rejected.stderr
    asyncio.run(check_protocol(args.image))
    print("Container installed entry, real MCP stdio and negative contracts passed (network disabled)")


if __name__ == "__main__":
    main()
