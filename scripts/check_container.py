#!/usr/bin/env python3
"""Exercise an installed Docker image with real MCP stdio and no external network."""

import argparse
import asyncio
import json
import os
import subprocess
import uuid
from datetime import timedelta
from pathlib import Path

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


def check_dependencies(image: str, *, installer_present: bool) -> list[dict]:
    """Inspect both actual Python installations, retaining real pip check before cleanup."""
    inventories = []
    inventory_code = (
        "import importlib.metadata as m, importlib.util as u, json, pathlib, sys, sysconfig, zipfile; "
        "root = pathlib.Path(sysconfig.get_path('purelib')); "
        "present = u.find_spec('pip') is not None; "
        "bom = root / 'pip/_vendor/bom.cdx.json'; "
        "bundled = pathlib.Path(sysconfig.get_path('stdlib')) / 'ensurepip/_bundled'; "
        "assert present == " + repr(installer_present) + "; "
        "assert not " + repr(not installer_present) + " or not list(root.glob('pip-*.dist-info')); "
        "assert not " + repr(not installer_present) + " or not (root / 'pip').exists(); "
        "print(json.dumps({'python': sys.executable, 'site_packages': str(root), "
        "'pip_present': present, 'pip_vendor_exists': (root / 'pip/_vendor').exists(), "
        "'packages': sorted((d.metadata['Name'], d.version) for d in m.distributions()), "
        "'bom_path': str(bom), 'bom_components': json.loads(bom.read_text()).get('components', []) if bom.exists() else [], "
        "'ensurepip_wheels': [{'path': str(p), 'pip_vendor_files': [n for n in zipfile.ZipFile(p).namelist() if n in ('pip/_vendor/bom.cdx.json', 'pip/_vendor/msgpack/__init__.py', 'pip/_vendor/urllib3/_version.py', 'pip/_vendor/pkg_resources/__init__.py')]} for p in bundled.glob('pip-*.whl')]}))"
    )
    for python in ("/usr/local/bin/python", "/opt/venv/bin/python"):
        command = ["docker", "run", "--rm", "--network", "none", "--entrypoint", python, image]
        if installer_present:
            checked = subprocess.run(
                command + ["-m", "pip", "check"], check=True, capture_output=True, text=True, timeout=30
            )
            print(python + ": " + checked.stdout.strip())
        result = subprocess.run(
            command + ["-c", inventory_code], check=True, capture_output=True, text=True, timeout=30
        )
        inventories.append(json.loads(result.stdout))
    return inventories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument(
        "--dependency-image", required=True, help="Pre-cleanup runtime-validation image for real pip check"
    )
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    dependency_inventory = check_dependencies(args.dependency_image, installer_present=True)
    # The same final absence assertion must reject the genuine pre-cleanup image.
    try:
        check_dependencies(args.dependency_image, installer_present=False)
    except subprocess.CalledProcessError as failure:
        assert "AssertionError" in failure.stderr
        print("Pre-cleanup image correctly fails the final pip-absence contract")
    else:
        raise AssertionError("Pip-absence check accepted the pre-cleanup image")
    runtime_inventory = check_dependencies(args.image, installer_present=False)
    for before, after in zip(dependency_inventory, runtime_inventory):
        expected = [package for package in before["packages"] if package[0].lower() != "pip"]
        assert after["packages"] == expected, "Cleanup changed packages other than pip"
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
    receipt = {
        "checkout_sha": os.environ.get("GITHUB_SHA"),
        "dependency_image": json.loads(
            subprocess.check_output(["docker", "image", "inspect", args.dependency_image], text=True)
        )[0]["Id"],
        "runtime_image": json.loads(subprocess.check_output(["docker", "image", "inspect", args.image], text=True))[0][
            "Id"
        ],
        "dependency_inventory": dependency_inventory,
        "runtime_inventory": runtime_inventory,
        "before_absence_contract": "rejected genuine pre-cleanup image with AssertionError",
        "real_pip_check": "passed in both pre-cleanup Python installations",
        "runtime_contracts": "native Torch/FAISS, non-root, help, rejected missing key, official MCP stdio/errors passed with network disabled",
    }
    if args.receipt:
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
    print("Container installed entry, real MCP stdio and negative contracts passed (network disabled)")


if __name__ == "__main__":
    main()
