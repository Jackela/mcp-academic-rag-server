"""Retained debug servers terminate at EOF and report real protocol/parser failures."""

import io
import json
from unittest.mock import patch

import pytest
from pypdf import PdfWriter

from servers.jsonrpc_stdio import serve_requests
from servers.mcp_server_enhanced import EnhancedMCPServer
from servers.mcp_server_standalone import MCPServer, SimpleDocumentProcessor


@pytest.mark.asyncio
@pytest.mark.parametrize("server_class", [EnhancedMCPServer, MCPServer])
async def test_shared_transport_handles_discovery_errors_and_eof(server_class, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    server = server_class()
    inputs = [
        {"jsonrpc": "2.0", "method": "initialize", "params": {}, "id": 1},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "method": "tools/list", "id": 2},
        {"jsonrpc": "2.0", "method": "unknown", "id": 3},
        {"jsonrpc": "2.0", "method": "tools/call", "params": [], "id": 4},
    ]
    reader = io.StringIO("\n".join(json.dumps(item) for item in inputs) + "\ninvalid-json\n[]\n")
    writer = io.StringIO()
    await serve_requests(server._dispatch_request, reader, writer)
    messages = [json.loads(line) for line in writer.getvalue().splitlines()]
    assert [message["id"] for message in messages] == [1, 2, 3, 4, None, None]
    assert messages[0]["result"]["serverInfo"]["name"]
    assert any(tool["name"] == "process_document" for tool in messages[1]["result"]["tools"])
    assert [message["error"]["code"] for message in messages[2:]] == [-32601, -32602, -32700, -32600]


@pytest.mark.asyncio
async def test_handler_failure_uses_current_id_and_recovers():
    async def controlled_handler(request):
        if request["id"] == 2:
            raise ValueError("controlled handler failure")
        return {"jsonrpc": "2.0", "result": "controlled", "id": request["id"]}

    source = io.StringIO("\n".join(json.dumps({"id": number, "method": "controlled"}) for number in [1, 2, 3]))
    output = io.StringIO()
    await serve_requests(controlled_handler, source, output)
    messages = [json.loads(line) for line in output.getvalue().splitlines()]
    assert messages[1]["id"] == 2 and messages[1]["error"]["code"] == -32603
    assert messages[2]["id"] == 3 and messages[2]["result"] == "controlled"


def test_blank_pdf_and_absent_optional_backend_are_not_extraction_success(tmp_path):
    path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(path)
    processor = SimpleDocumentProcessor()
    real_extract = processor._extract_pdf_text

    def controlled_extract(file_path, backend):
        if backend != "pypdf":
            raise ImportError("Optional backend is unavailable")
        return real_extract(file_path, backend)

    with patch.object(processor, "_extract_pdf_text", side_effect=controlled_extract):
        with pytest.raises(ValueError, match="未提取到文本"):
            processor.process_pdf(str(path))
    assert processor.documents == {}


def test_preserved_rag_format_reports_only_controlled_source_content():
    text = MCPServer._format_rag_answer(
        {
            "answer": "Controlled answer",
            "documents": [{"content": "Controlled source", "metadata": {"file_name": "fixture.txt"}}],
        },
        "Controlled question",
        1,
        "fixture-session",
    )
    assert "Controlled answer" in text and "fixture.txt" in text
    assert "fixture-session" in text and "前1个" in text
