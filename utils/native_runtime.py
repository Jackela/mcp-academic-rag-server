"""Prepare a Darwin CLI process to share the installed Torch OpenMP runtime.

Library imports never replace their host process. Embedded consumers must start
their Python subprocess with native_subprocess_environment() before native imports.
"""

import ctypes
import importlib.util
import os
import struct
import sys
from pathlib import Path
from typing import Dict, Mapping, Optional, Set

_EXEC_MARKER = "MCP_NATIVE_RUNTIME_EXEC"


def _parse_startup_native_environment(data: bytes) -> Dict[str, str]:
    """Read only two native keys after the executable and counted argv fields."""
    if len(data) <= 4:
        raise RuntimeError("Invalid Darwin startup environment size")
    count = struct.unpack_from("=i", data)[0]
    if not 0 < count <= len(data) - 4:
        raise RuntimeError("Invalid Darwin startup argument count")
    position = data.find(b"\0", 4)
    if position < 0:
        raise RuntimeError("Invalid Darwin startup executable layout")
    position += 1
    while position < len(data) and data[position] == 0:
        position += 1
    for _ in range(count):
        end = data.find(b"\0", position)
        if end < 0:
            raise RuntimeError("Truncated Darwin startup arguments")
        position = end + 1
    return _selected_startup_native_fields(data, position)


def _selected_startup_native_fields(data: bytes, position: int) -> Dict[str, str]:
    """Extract only recognized keys from the bounded environment fields."""
    prefixes = {key: key.encode() + b"=" for key in ("DYLD_LIBRARY_PATH", "KMP_DUPLICATE_LIB_OK")}
    selected: Dict[str, str] = {}
    while position < len(data) and data[position] != 0:
        end = data.find(b"\0", position)
        if end < 0:
            raise RuntimeError("Truncated Darwin startup environment")
        for key, prefix in prefixes.items():
            if data.startswith(prefix, position, end):
                if key in selected:
                    raise RuntimeError("Duplicate Darwin native startup key")
                try:
                    selected[key] = data[position + len(prefix) : end].decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise RuntimeError("Invalid UTF-8 in Darwin native startup key") from exc
        position = end + 1
    if position >= len(data):
        raise RuntimeError("Unterminated Darwin startup environment")
    return selected


def _startup_native_environment() -> Dict[str, str]:
    """Query own-pid KERN_PROCARGS2; never log or retain raw argv/environment."""
    loader = ctypes.CDLL(None, use_errno=True)
    query = loader.sysctl
    query.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    query.restype = ctypes.c_int
    mib = (ctypes.c_int * 3)(1, 49, os.getpid())  # CTL_KERN, KERN_PROCARGS2, own pid
    size = ctypes.c_size_t()
    if query(mib, 3, None, ctypes.byref(size), None, 0):
        raise RuntimeError("Cannot size Darwin native startup environment")
    if not 4 < size.value <= 4 * 1024 * 1024:
        raise RuntimeError("Invalid Darwin native startup environment size")
    capacity = size.value
    buffer = ctypes.create_string_buffer(capacity)
    if query(mib, 3, buffer, ctypes.byref(size), None, 0):
        raise RuntimeError("Cannot read Darwin native startup environment")
    if not 4 < size.value <= capacity:
        raise RuntimeError("Invalid Darwin native startup environment length")
    return _parse_startup_native_environment(buffer.raw[: size.value])


def _torch_library_directory() -> Path:
    spec = importlib.util.find_spec("torch")
    if spec is None or spec.origin is None:
        raise RuntimeError("Darwin native runtime requires the installed Torch package")
    directory = (Path(spec.origin).resolve().parent / "lib").resolve()
    if not (directory / "libomp.dylib").is_file():
        raise RuntimeError("Installed Torch is missing lib/libomp.dylib")
    return directory


def native_subprocess_environment(environ: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """Return startup environment for a new Python process; leave the host untouched."""
    environment = dict(os.environ if environ is None else environ)
    if sys.platform != "darwin":
        return environment
    directory = str(_torch_library_directory())
    existing = environment.get("DYLD_LIBRARY_PATH", "").split(os.pathsep)
    paths = [directory] + [path for path in existing if path and path != directory]
    environment["DYLD_LIBRARY_PATH"] = os.pathsep.join(paths)
    environment["KMP_DUPLICATE_LIB_OK"] = "FALSE"
    return environment


def _loaded_openmp_paths() -> Set[Path]:
    loader = ctypes.CDLL(None)
    count = loader._dyld_image_count
    count.argtypes = []
    count.restype = ctypes.c_uint32
    name = loader._dyld_get_image_name
    name.argtypes = [ctypes.c_uint32]
    name.restype = ctypes.c_char_p
    paths: Set[Path] = set()
    for index in range(count()):
        value = name(index)
        if value:
            path = Path(os.fsdecode(value)).resolve()
            if any(prefix in path.name.lower() for prefix in ("libomp", "libiomp", "libgomp")):
                paths.add(path)
    return paths


def require_native_runtime() -> None:
    """Validate explicit Darwin SDK startup without changing or reexecuting its host."""
    if sys.platform != "darwin":
        return
    directory = _torch_library_directory()
    startup = _startup_native_environment()
    for environment in (startup, os.environ):
        paths = environment.get("DYLD_LIBRARY_PATH", "").split(os.pathsep)
        if not paths[0] or Path(paths[0]).resolve() != directory:
            raise RuntimeError(
                "Start the Python process with native_subprocess_environment(); DYLD must precede Python"
            )
        if environment.get("KMP_DUPLICATE_LIB_OK", "").upper() != "FALSE":
            raise RuntimeError("Darwin native runtime requires startup KMP_DUPLICATE_LIB_OK=FALSE")
    loaded = _loaded_openmp_paths()
    if loaded - {(directory / "libomp.dylib").resolve()}:
        raise RuntimeError(
            "Darwin process already loaded a different OpenMP runtime; restart with the startup environment"
        )


def prepare_cli_native_runtime() -> None:
    """Replace only an explicitly invoked Darwin CLI once, retaining argv and stdio."""
    if sys.platform != "darwin":
        return
    directory = str(_torch_library_directory())
    marker = os.environ.get(_EXEC_MARKER)
    if marker is not None:
        if marker != directory:
            raise RuntimeError("Native runtime reentry marker does not match the installed Torch package")
        require_native_runtime()
        return
    environment = native_subprocess_environment()
    environment[_EXEC_MARKER] = directory
    os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]], environment)
    raise RuntimeError("Native runtime process replacement returned unexpectedly")
