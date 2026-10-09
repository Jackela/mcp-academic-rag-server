"""CLI process replacement is explicit, bounded, and preserves actual process I/O."""

import importlib.util
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from utils import native_runtime

ROOT = Path(__file__).resolve().parents[2]


def test_startup_parser_skips_counted_argv_and_selects_only_native_keys():
    data = struct.pack("=i", 2) + b"/python\0\0/python\0DYLD_LIBRARY_PATH=/argv-is-not-env\0"
    data += b"SECRET=must-not-be-returned\0DYLD_LIBRARY_PATH=/torch/lib\0KMP_DUPLICATE_LIB_OK=FALSE\0\0"
    assert native_runtime._parse_startup_native_environment(data) == {
        "DYLD_LIBRARY_PATH": "/torch/lib",
        "KMP_DUPLICATE_LIB_OK": "FALSE",
    }


def test_startup_parser_accepts_many_empty_arguments():
    data = struct.pack("=i", 100) + b"/python\0\0/python\0" + b"\0" * 99
    data += b"DYLD_LIBRARY_PATH=/torch/lib\0KMP_DUPLICATE_LIB_OK=FALSE\0\0"
    assert native_runtime._parse_startup_native_environment(data) == {
        "DYLD_LIBRARY_PATH": "/torch/lib",
        "KMP_DUPLICATE_LIB_OK": "FALSE",
    }


@pytest.mark.parametrize(
    "data",
    [
        b"",
        struct.pack("=i", -1) + b"/python\0\0",
        struct.pack("=i", 1) + b"missing-termination",
        struct.pack("=i", 2) + b"/python\0\0/python\0",
        struct.pack("=i", 1) + b"/python\0\0/python\0DYLD_LIBRARY_PATH=/lib",
        struct.pack("=i", 1) + b"/python\0\0/python\0DYLD_LIBRARY_PATH=\xff\0\0",
        struct.pack("=i", 1) + b"/python\0\0/python\0KMP_DUPLICATE_LIB_OK=FALSE\0KMP_DUPLICATE_LIB_OK=TRUE\0\0",
    ],
)
def test_startup_parser_rejects_invalid_layout_without_exposing_values(data):
    with pytest.raises(RuntimeError, match="Darwin"):
        native_runtime._parse_startup_native_environment(data)


def test_unavailable_darwin_startup_query_is_an_explicit_error(monkeypatch):
    monkeypatch.setattr(native_runtime.sys, "platform", "darwin")
    monkeypatch.setattr(native_runtime, "_torch_library_directory", lambda: Path("/torch/lib"))

    def unavailable():
        raise RuntimeError("Cannot read Darwin native startup environment")

    monkeypatch.setattr(native_runtime, "_startup_native_environment", unavailable)
    with pytest.raises(RuntimeError, match="Cannot read Darwin"):
        native_runtime.require_native_runtime()


def test_non_darwin_does_not_find_libraries_or_replace_host(monkeypatch):
    monkeypatch.setattr(native_runtime.sys, "platform", "linux")
    monkeypatch.setattr(native_runtime, "_torch_library_directory", lambda: pytest.fail("native discovery on Linux"))
    monkeypatch.setattr(native_runtime.os, "execve", lambda *args: pytest.fail("host replaced"))
    before = dict(os.environ)
    assert native_runtime.native_subprocess_environment({"KEEP": "value"}) == {"KEEP": "value"}
    native_runtime.prepare_cli_native_runtime()
    native_runtime.require_native_runtime()
    assert dict(os.environ) == before


def test_missing_torch_has_explicit_error(monkeypatch):
    monkeypatch.setattr(native_runtime.sys, "platform", "darwin")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="installed Torch"):
        native_runtime.native_subprocess_environment({})


def test_library_import_does_not_replace_host_or_load_native():
    code = """
import os, sys
def forbidden(*args): raise AssertionError('library import replaced host')
os.execve = forbidden
import utils.native_runtime
import servers.mcp_server_sdk
assert not {'torch', 'faiss', 'sklearn'}.intersection(sys.modules)
print('import only')
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "import only"


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin dyld process-start contract")
def test_actual_exec_keeps_options_argv_stdio_and_exit_status(tmp_path):
    code = """
import json, os, sys
from pathlib import Path
from utils.native_runtime import prepare_cli_native_runtime
with Path(sys.argv[1]).open('a') as trace: trace.write(str(os.getpid()) + '\\n')
prepare_cli_native_runtime()
print(json.dumps({'argv':sys.argv[2:], 'input':sys.stdin.readline().strip(), 'KMP':os.environ['KMP_DUPLICATE_LIB_OK']}))
print('retained stderr',file=sys.stderr)
sys.exit(7)
"""
    trace = tmp_path / "starts"
    env = dict(os.environ)
    for name in ("DYLD_LIBRARY_PATH", "MCP_NATIVE_RUNTIME_EXEC", "KMP_DUPLICATE_LIB_OK"):
        env.pop(name, None)
    result = subprocess.run(
        [sys.executable, "-u", "-c", code, str(trace), "literal $argument", "中文参数"],
        cwd=ROOT,
        env=env,
        input="owned stdin\n",
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 7, result.stderr
    assert json.loads(result.stdout) == {
        "argv": ["literal $argument", "中文参数"],
        "input": "owned stdin",
        "KMP": "FALSE",
    }
    assert result.stderr.strip() == "retained stderr"
    starts = trace.read_text().splitlines()
    assert len(starts) == 2 and starts[0] == starts[1], "one exec retains its original process ID"


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin reentry negative contract")
def test_reentry_with_missing_startup_environment_fails_without_loop(tmp_path):
    directory = native_runtime._torch_library_directory()
    env = dict(os.environ, MCP_NATIVE_RUNTIME_EXEC=str(directory), OPENAI_API_KEY="sk-offline-fixture-only")
    env.pop("DYLD_LIBRARY_PATH", None)
    result = subprocess.run(
        [sys.executable, "-m", "servers.mcp_server_sdk"],
        cwd=ROOT,
        env=env,
        input="",
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 1
    assert result.stdout == "" and "Native runtime setup failed" in result.stderr and "DYLD" in result.stderr


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin SDK embedding negative contract")
def test_embedded_sdk_does_not_replace_unconfigured_host():
    env = dict(os.environ, OPENAI_API_KEY="sk-offline-fixture-only")
    env.pop("DYLD_LIBRARY_PATH", None)
    code = """
import asyncio, os
def forbidden(*args): raise AssertionError('SDK replaced embedded host')
os.execve = forbidden
from servers.mcp_server_sdk import main
asyncio.run(main())
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 1 and result.stdout == ""
    assert "DYLD" in result.stderr and "SDK replaced" not in result.stderr


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin late environment negative contract")
@pytest.mark.parametrize("preload_torch", [False, True])
def test_late_environment_update_cannot_pass_embedded_startup(preload_torch):
    env = dict(os.environ)
    for name in ("DYLD_LIBRARY_PATH", "MCP_NATIVE_RUNTIME_EXEC", "KMP_DUPLICATE_LIB_OK"):
        env.pop(name, None)
    code = """
import os, sys
from utils.native_runtime import native_subprocess_environment, require_native_runtime
if sys.argv[1] == 'True': import torch
os.environ.update(native_subprocess_environment())
try:
    require_native_runtime()
except RuntimeError as error:
    assert 'DYLD must precede Python' in str(error)
    assert 'faiss' not in sys.modules and 'sklearn' not in sys.modules
    print('late environment rejected before native operations')
else:
    raise AssertionError('late environment was incorrectly accepted')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(preload_torch)], cwd=ROOT, env=env, text=True, capture_output=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "late environment rejected before native operations"
