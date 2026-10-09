"""Current MCP boundary and processing workflow, using controlled offline data.

Historical HaystackRAGPipeline/memory_store fixtures were never public classes.
Real native retrieval, prompt generation and sessions run in the integration suite;
this suite verifies MCP stdio, handler injection, processing and failure recovery.
"""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from core.pipeline import Pipeline
from models.document import Document
from models.process_result import ProcessResult
from processors.base_processor import BaseProcessor
from servers import mcp_server_sdk as server
from tests.integration.test_rag_integration import rag_system


class FixtureProcessor(BaseProcessor):
    def __init__(self, fail=False):
        super().__init__(name="FixtureProcessor")
        self.fail = fail

    def process(self, document):
        if self.fail:
            return ProcessResult.error_result("controlled processing failure")
        document.store_content(self.get_stage(), "controlled fixture text")
        return ProcessResult.success_result()


@pytest.mark.asyncio
async def test_real_stdio_discovery_and_errors(tmp_path):
    env = dict(os.environ)
    env.update(
        OPENAI_API_KEY="sk-offline-fixture-only",
        DATA_PATH=str(tmp_path / "data"),
        PYTHONPATH=str(Path(__file__).resolve().parents[2]),
    )
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "servers.mcp_server_sdk"], env=env, cwd=str(tmp_path)
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            assert {tool.name for tool in tools.tools} == {
                "test_connection",
                "validate_system",
                "process_document",
                "query_documents",
            }
            echo = await session.call_tool("test_connection", {"message": "offline-boundary"})
            assert "offline-boundary" in echo.content[0].text
            missing = await session.call_tool("process_document", {"file_path": str(tmp_path / "missing.pdf")})
            assert "not found" in missing.content[0].text
            bad_args = await session.call_tool("process_document", {})
            assert bad_args.isError


@pytest.mark.parametrize("key", [None, "invalid-key"])
def test_entry_rejects_missing_or_invalid_key(key, tmp_path):
    env = dict(os.environ)
    env.pop("OPENAI_API_KEY", None)
    if key is not None:
        env["OPENAI_API_KEY"] = key
    env.update(PYTHONPATH=str(Path(__file__).resolve().parents[2]), DATA_PATH=str(tmp_path / "data"))
    result = subprocess.run(
        [sys.executable, "-m", "servers.mcp_server_sdk", "--validate-only"],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=20,
    )
    assert result.returncode != 0
    assert "OPENAI_API_KEY" in result.stderr


@pytest.mark.asyncio
async def test_processing_tool_reports_actual_pipeline_result(monkeypatch, tmp_path):
    file = tmp_path / "fixture.txt"
    file.write_text("controlled input")
    pipeline = Pipeline("fixture-pipeline")
    processor = FixtureProcessor()
    pipeline.add_processor(processor)
    context = SimpleNamespace(is_initialized=True, document_pipeline=pipeline)
    monkeypatch.setattr(server, "server_context", context)
    success = await server.handle_process_document({"file_path": str(file)})
    assert "处理成功" in success[0].text
    assert "FixtureProcessor" in success[0].text
    processor.fail = True
    failed = await server.handle_process_document({"file_path": str(file)})
    assert "处理失败" in failed[0].text
    assert "controlled processing failure" in failed[0].text
    processor.fail = False
    recovered = await server.handle_process_document({"file_path": str(file)})
    assert "处理成功" in recovered[0].text
    pipeline._executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_query_tool_uses_real_session_pipeline(monkeypatch, rag_system):
    store, processor, pipeline, generator, manager = rag_system
    document = Document("fixture.txt")
    document.store_content("OCRProcessor", "controlled query context")
    assert processor.process(document).is_successful()
    context = SimpleNamespace(is_initialized=True, session_manager=manager, rag_pipeline=pipeline)
    monkeypatch.setattr(server, "server_context", context)
    result = await server.handle_query_documents({"query": "fixture question", "session_id": "fixture-session"})
    assert "Controlled fixture answer" in result[0].text
    assert manager.get_session("fixture-session") is not None
    assert "controlled query context" in generator.calls[-1][-1].text
    generator.fail = True
    failed = await server.handle_query_documents({"query": "failure fixture", "session_id": "fixture-session"})
    assert "查询处理失败" in failed[0].text


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments,expected", [({}, "query"), ({"query": ""}, "query")])
async def test_query_rejects_empty_arguments(arguments, expected):
    result = await server.handle_query_documents(arguments)
    assert expected in result[0].text.lower()


@pytest.mark.asyncio
async def test_concurrent_processing_uses_unique_documents():
    documents = [Document(f"fixture-{i}.txt") for i in range(20)]
    pipeline = Pipeline("batch-fixture")
    pipeline.add_processor(FixtureProcessor())
    try:
        result = await pipeline.process_documents(documents, max_concurrent=5)
        assert len(result) == 20
        assert all(outcome.is_successful() for outcome in result.values())
        assert all(doc.get_content("FixtureProcessor") == "controlled fixture text" for doc in documents)
    finally:
        pipeline._executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_batch_failure_preserves_each_document_outcome():
    documents = [Document(f"failure-{i}.txt") for i in range(3)]
    pipeline = Pipeline("batch-failure")
    pipeline.add_processor(FixtureProcessor(fail=True))
    try:
        result = await pipeline.process_documents(documents, max_concurrent=2)
        assert len(result) == 3
        assert all(not outcome.is_successful() for outcome in result.values())
        assert all("controlled processing failure" in outcome.get_message() for outcome in result.values())
    finally:
        pipeline._executor.shutdown(wait=True)
