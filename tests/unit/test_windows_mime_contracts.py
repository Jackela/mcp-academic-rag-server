"""Fail-closed DLL selection locally; real native MIME on the Windows matrix."""

import hashlib
import json
import struct
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from importlib import metadata
from types import SimpleNamespace

import pytest
from PIL import Image
from pypdf import PdfWriter

from utils import windows_mime
from utils.security_utils import InputValidator, SecurityConfig, magic


@pytest.mark.parametrize("failure", ["package", "competing", "dll", "database", "header", "architecture", "abi"])
def test_missing_or_incompatible_owned_dll_never_uses_path_or_cwd(tmp_path, monkeypatch, failure):
    directory = tmp_path / "magic"
    directory.mkdir()
    header = bytearray(90)
    header[:2] = b"MZ"
    struct.pack_into("<I", header, 60, 64)
    header[64:68] = b"PE\0\0"
    struct.pack_into("<H", header, 68, 0x8664 if failure != "architecture" else 0x14C)
    struct.pack_into("<H", header, 88, 0x20B if failure != "abi" else 0x10B)
    if failure != "dll":
        (directory / "libmagic-1.dll").write_bytes(header if failure != "header" else b"not a DLL")
    if failure != "database":
        (directory / "magic.mgc").write_bytes(b"owned database fixture")
    (tmp_path / "msys-magic-1.dll").write_bytes(b"foreign DLL must never be loaded")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path))

    def distribution(name):
        if name == "python-magic-standalone" and failure != "package":
            return SimpleNamespace(locate_file=lambda path: tmp_path / path)
        if name == "python-magic" and failure == "competing":
            return SimpleNamespace()
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(windows_mime.metadata, "distribution", distribution)
    monkeypatch.setattr(windows_mime.ctypes, "CDLL", lambda *args, **kwargs: pytest.fail("foreign/native DLL loaded"))
    with pytest.raises((ImportError, metadata.PackageNotFoundError)):
        windows_mime.WindowsMimeBackend()


@pytest.mark.parametrize("failure", ["load", "file", "query"])
def test_each_open_handle_is_closed_on_inspection_failure(tmp_path, failure):
    closed = []
    backend = object.__new__(windows_mime.WindowsMimeBackend)
    backend.database_path = tmp_path / "magic.mgc"
    backend._library = SimpleNamespace(
        magic_open=lambda flags: 123,
        magic_load=lambda cookie, database: -1 if failure == "load" else 0,
        magic_buffer=lambda cookie, data, size: None,
        magic_error=lambda cookie: b"controlled native query failure",
        magic_close=lambda cookie: closed.append(cookie),
    )
    file = tmp_path / "file.txt"
    if failure != "file":
        file.write_text("owned content")
    with pytest.raises((RuntimeError, OSError)):
        backend.from_file(str(file), mime=True, max_bytes=SecurityConfig.MAX_FILE_SIZE)
    assert closed == [123]


def test_size_growth_after_validation_is_bounded_and_rejected(tmp_path, monkeypatch):
    import utils.security_utils as security

    closed = []
    backend = object.__new__(windows_mime.WindowsMimeBackend)
    backend.database_path = tmp_path / "magic.mgc"
    backend._library = SimpleNamespace(
        magic_open=lambda flags: 123,
        magic_load=lambda cookie, database: 0,
        magic_buffer=lambda *args: pytest.fail("oversize content reached native inspection"),
        magic_close=lambda cookie: closed.append(cookie),
    )
    path = tmp_path / "grown.txt"
    path.write_bytes(b"content grew after the size check")
    monkeypatch.setattr(security, "magic", backend)
    monkeypatch.setattr(security.SecurityConfig, "MAX_FILE_SIZE", 4)
    monkeypatch.setattr(security.os.path, "getsize", lambda path: 1)
    valid, error = security.InputValidator.validate_file_content(str(path))
    assert not valid and "File too large during MIME inspection" in error
    assert closed == [123]


def _write_real_allowed_file(path, kind):
    if kind == "pdf":
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        with path.open("wb") as stream:
            writer.write(stream)
    elif kind in ("png", "jpeg", "tiff"):
        Image.new("RGB", (3, 3), color="red").save(path, format=kind.upper())
    elif kind in ("utf8", "utf16"):
        path.write_text("Controlled text 文献内容\n", encoding="utf-8" if kind == "utf8" else "utf-16")
    elif kind == "docx":
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "[Content_Types].xml",
                '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                '<Default Extension="xml" ContentType="application/xml"/>'
                '<Override PartName="/word/document.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>',
            )
            archive.writestr(
                "_rels/.rels",
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
                'Target="word/document.xml"/></Relationships>',
            )
            archive.writestr(
                "word/document.xml",
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                "<w:body><w:p><w:r><w:t>Controlled document</w:t></w:r></w:p></w:body></w:document>",
            )


@pytest.mark.skipif(sys.platform != "win32", reason="Actual Windows standalone DLL ABI and content contract")
def test_actual_windows_native_mime_and_threaded_handles(tmp_path, capsys):
    assert isinstance(magic, windows_mime.WindowsMimeBackend), "the maintained standalone backend must actually load"
    expected = {
        "pdf": "application/pdf",
        "png": "image/png",
        "jpeg": "image/jpeg",
        "tiff": "image/tiff",
        "utf8": "text/plain",
        "utf16": "text/plain",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
    files = []
    for kind, mime in expected.items():
        path = tmp_path / ("文献-" + kind + (".txt" if kind.startswith("utf") else "." + kind))
        _write_real_allowed_file(path, kind)
        assert magic.from_file(str(path), mime=True, max_bytes=SecurityConfig.MAX_FILE_SIZE) == mime
        assert InputValidator.validate_file_content(str(path)) == (True, None)
        files.append(path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(
            valid for valid, _ in pool.map(lambda path: InputValidator.validate_file_content(str(path)), files * 3)
        )
    for kind, content in {
        "unknown": b"\0\xff\x01\x80" * 20,
        "executable-disguised-pdf": b"MZ" + b"\0" * 510,
        "html": b"<!DOCTYPE html><html><body>controlled</body></html>",
        "json": b'{"controlled":true,"value":123}',
        "shell": b"#!/bin/sh\necho controlled\n",
    }.items():
        path = tmp_path / (kind + ".pdf")
        path.write_bytes(content)
        valid, error = InputValidator.validate_file_content(str(path))
        assert not valid and error is not None, (
            kind,
            magic.from_file(str(path), mime=True, max_bytes=SecurityConfig.MAX_FILE_SIZE),
        )
    with capsys.disabled():
        print(
            json.dumps(
                {
                    "native_library": str(magic.library_path),
                    "native_library_sha256": hashlib.sha256(magic.library_path.read_bytes()).hexdigest(),
                    "magic_version": magic.version,
                    "allowed_contents": expected,
                    "rejected_contents": ["unknown", "executable-disguised-pdf", "html", "json", "shell"],
                    "threaded_inspections": 21,
                }
            )
        )
