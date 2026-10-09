"""Bind only the standalone wheel's native Windows libmagic and database."""

import ctypes
import os
import struct
from importlib import metadata
from pathlib import Path
from typing import Tuple


def _owned_library_files() -> Tuple[Path, Path]:
    distribution = metadata.distribution("python-magic-standalone")
    for incompatible in ("python-magic", "python-magic-bin"):
        try:
            metadata.distribution(incompatible)
        except metadata.PackageNotFoundError:
            continue
        raise ImportError(
            "Windows MIME backend requires an isolated standalone installation without competing magic packages"
        )
    directory = Path(str(distribution.locate_file("magic"))).resolve()
    library, database = ((directory / name).resolve() for name in ("libmagic-1.dll", "magic.mgc"))
    if not all(path.is_relative_to(directory) and path.is_file() for path in (library, database)):
        raise ImportError("Windows MIME backend is missing its owned DLL or magic database")
    with library.open("rb") as stream:
        header = stream.read(64)
        if len(header) != 64 or header[:2] != b"MZ":
            raise ImportError("Windows MIME backend has an invalid native DLL")
        stream.seek(struct.unpack_from("<I", header, 60)[0])
        pe = stream.read(26)
    machine, optional_magic = (0x8664, 0x20B) if ctypes.sizeof(ctypes.c_void_p) == 8 else (0x14C, 0x10B)
    if len(pe) != 26 or pe[:4] != b"PE\0\0" or struct.unpack_from("<H", pe, 4)[0] != machine:
        raise ImportError("Windows MIME backend DLL architecture does not match Python")
    if struct.unpack_from("<H", pe, 24)[0] != optional_magic:
        raise ImportError("Windows MIME backend DLL has an incompatible pointer ABI")
    return library, database


class WindowsMimeBackend:
    """Keep libmagic classification; each call owns and closes a separate cookie."""

    def __init__(self) -> None:
        self.library_path, self.database_path = _owned_library_files()
        # Search dependent DLLs only beside this DLL and in System32, never PATH/CWD.
        self._library = ctypes.CDLL(str(self.library_path), winmode=0x100 | 0x800)
        for name, arguments, result in (
            ("magic_version", [], ctypes.c_int),
            ("magic_open", [ctypes.c_int], ctypes.c_void_p),
            ("magic_load", [ctypes.c_void_p, ctypes.c_char_p], ctypes.c_int),
            ("magic_buffer", [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t], ctypes.c_char_p),
            ("magic_error", [ctypes.c_void_p], ctypes.c_char_p),
            ("magic_close", [ctypes.c_void_p], None),
        ):
            function = getattr(self._library, name)
            function.argtypes, function.restype = arguments, result
        self.version = int(self._library.magic_version())
        if self.version < 546:
            raise ImportError("Windows MIME backend requires the maintained libmagic 5.46 or newer")

    def from_file(self, file_path: str, mime: bool = False, *, max_bytes: int) -> str:
        if max_bytes <= 0:
            raise ValueError("MIME inspection requires a positive file size limit")
        cookie = self._library.magic_open(0x10 if mime else 0)
        if not cookie:
            raise RuntimeError("Windows libmagic could not initialize a file inspection handle")
        try:
            if self._library.magic_load(cookie, os.fsencode(self.database_path)) != 0:
                raise RuntimeError("Windows libmagic could not load its owned database")
            # Python opens Unicode filenames; the native library inspects their bytes.
            with open(file_path, "rb") as stream:
                content = stream.read(max_bytes + 1)
            if len(content) > max_bytes:
                raise RuntimeError("File too large during MIME inspection")
            result = self._library.magic_buffer(cookie, content, len(content))
            if not isinstance(result, bytes):
                error = self._library.magic_error(cookie)
                message = (
                    error.decode("utf-8", errors="replace") if isinstance(error, bytes) else "file inspection failed"
                )
                raise RuntimeError(message)
            return result.decode("utf-8")
        finally:
            self._library.magic_close(cookie)
