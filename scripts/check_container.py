#!/usr/bin/env python3
"""Exercise an installed Docker image with real MCP stdio and no external network."""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import uuid
from datetime import timedelta
from pathlib import Path
from textwrap import dedent

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def run_checked(command: list[str], *, timeout: float, text: bool = False) -> subprocess.CompletedProcess:
    """Keep controlled container failure diagnostics while propagating the original error."""
    try:
        return subprocess.run(command, check=True, capture_output=True, text=text, timeout=timeout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as failure:
        for output in (failure.stdout, failure.stderr):
            if output:
                diagnostic = output.decode("utf-8", errors="replace") if isinstance(output, bytes) else output
                print(diagnostic, end="", file=sys.stderr)
        raise


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
            run_checked(["docker", "rm", "-f", name], timeout=15)


def check_dependencies(image: str, *, installer_present: bool) -> list[dict]:
    """Inspect both actual Python installations, retaining real pip check before cleanup."""
    inventories = []
    inventory_code = (
        "import importlib.metadata as m, importlib.util as u, json, pathlib, sys, sysconfig, zipfile; "
        "root = pathlib.Path(sysconfig.get_path('purelib')); "
        "present = u.find_spec('pip') is not None; "
        "bom = root / 'pip/_vendor/bom.cdx.json'; "
        "ensurepip = pathlib.Path(sysconfig.get_path('stdlib')) / 'ensurepip'; "
        "bundled = ensurepip / '_bundled'; "
        "assert ensurepip.exists() == " + repr(installer_present) + "; "
        "assert (u.find_spec('ensurepip') is not None) == " + repr(installer_present) + "; "
        "assert present == " + repr(installer_present) + "; "
        "assert not " + repr(not installer_present) + " or not list(root.glob('pip-*.dist-info')); "
        "assert not " + repr(not installer_present) + " or not (root / 'pip').exists(); "
        "print(json.dumps({'python': sys.executable, 'site_packages': str(root), "
        "'pip_present': present, 'pip_vendor_exists': (root / 'pip/_vendor').exists(), 'ensurepip_exists': ensurepip.exists(), "
        "'packages': sorted((d.metadata['Name'], d.version) for d in m.distributions()), "
        "'bom_path': str(bom), 'bom_components': json.loads(bom.read_text()).get('components', []) if bom.exists() else [], "
        "'ensurepip_wheels': [{'path': str(p), 'pip_vendor_files': [n for n in zipfile.ZipFile(p).namelist() if n in ('pip/_vendor/bom.cdx.json', 'pip/_vendor/msgpack/__init__.py', 'pip/_vendor/urllib3/_version.py', 'pip/_vendor/pkg_resources/__init__.py')]} for p in bundled.glob('pip-*.whl')]}))"
    )
    for python in ("/usr/local/bin/python", "/opt/venv/bin/python"):
        command = ["docker", "run", "--rm", "--network", "none", "--entrypoint", python, image]
        if installer_present:
            checked = run_checked(command + ["-m", "pip", "check"], text=True, timeout=30)
            print(python + ": " + checked.stdout.strip())
        result = run_checked(command + ["-c", inventory_code], text=True, timeout=30)
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
    print("Checking the pre-cleanup image rejects final installer absence (expected failure)", flush=True)
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
    run_checked(
        ["docker", "run", "--rm", "--network", "none", "--entrypoint", "python", args.image, "-c", native_check],
        timeout=60,
    )
    persistence_check = dedent("""
        import ctypes, ctypes.util, json, pathlib, sqlite3, tempfile
        from core.config_manager import ConfigManager
        from core.config_validator import generate_default_config
        from document_stores.implementations.memory_vector_store import InMemoryVectorStore
        from haystack.dataclasses import Document
        from rag.chat_session import ChatSessionManager

        magic_path = ctypes.util.find_library('magic')
        assert magic_path, 'System libmagic is missing'
        magic_library = ctypes.CDLL(magic_path)
        magic_library.magic_version.argtypes = []
        magic_library.magic_version.restype = ctypes.c_int
        magic_version = magic_library.magic_version()
        assert magic_version > 0
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            with sqlite3.connect(root / 'native.db') as database:
                database.execute('CREATE TABLE source (id INTEGER PRIMARY KEY, text TEXT)')
                database.execute('INSERT INTO source VALUES (?, ?)', (1, 'controlled source'))
            with sqlite3.connect(root / 'native.db') as database:
                assert database.execute('SELECT text FROM source WHERE id = ?', (1,)).fetchone() == ('controlled source',)
                database.execute('INSERT INTO source VALUES (?, ?)', (2, 'rolled back'))
                database.rollback()
                assert database.execute('SELECT count(*) FROM source').fetchone() == (1,)
            config = generate_default_config()
            config['storage']['base_path'] = str(root)
            config['storage']['output_path'] = str(root / 'output')
            path = root / 'config.json'
            path.write_text(json.dumps(config))
            manager = ConfigManager(str(path))
            assert manager.is_config_valid(), manager.get_validation_report()
            assert manager.set_value('app.name', 'container-fixture')
            assert manager.save_config()
            assert ConfigManager(str(path)).get_value('app.name') == 'container-fixture'
            store = InMemoryVectorStore({'vector_dimension': 2, 'similarity': 'cosine'})
            assert store.initialize()
            document = Document(id='container-source', content='controlled source', embedding=[1.0, 0.0])
            assert store.add_documents([document])
            assert store.get_document_by_id(document.id).content == document.content
            assert store.search([1.0, 0.0], top_k=1)[0][0].id == document.id
            sessions = ChatSessionManager()
            session = sessions.create_session(session_id='container-session')
            session.add_message('user', 'controlled question')
            session_path = root / 'sessions.json'
            assert sessions.save_sessions(str(session_path))
            restored = ChatSessionManager()
            assert restored.load_sessions(str(session_path))
            assert restored.get_session(session.session_id).get_messages() == session.get_messages()
            assert store.delete_document(document.id)
            assert store.get_document_count() == 0
        print(json.dumps({'libmagic_library': magic_path, 'libmagic_version': magic_version, 'sqlite_version': sqlite3.sqlite_version, 'sqlite_write_read_rollback': 'passed',
                          'configuration_roundtrip': 'passed', 'document_store_write_read_search_delete': 'passed',
                          'session_roundtrip': 'passed'}))
        """)
    persistence = run_checked(
        ["docker", "run", "--rm", "--network", "none", "--entrypoint", "python", args.image, "-c", persistence_check],
        text=True,
        timeout=60,
    )
    persistence_receipt = json.loads(persistence.stdout)
    run_checked(["docker", "run", "--rm", "--network", "none", args.image, "--help"], timeout=30)
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
        "runtime_tag": args.image,
        "runtime_manifest": os.environ.get("CONTAINER_IMAGE_MANIFEST"),
        "dependency_image": json.loads(
            subprocess.check_output(["docker", "image", "inspect", args.dependency_image], text=True)
        )[0]["Id"],
        "runtime_image": json.loads(subprocess.check_output(["docker", "image", "inspect", args.image], text=True))[0][
            "Id"
        ],
        "persistence_contracts": persistence_receipt,
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
