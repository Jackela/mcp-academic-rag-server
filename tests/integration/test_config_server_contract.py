"""Config-center tools bind one actual validated snapshot to the maintained document/RAG path."""

import json
from unittest.mock import patch

import pytest
from pydantic import AnyUrl

from core.config_center import ConfigCenter
from core.config_validator import generate_default_config
from models.process_result import ProcessResult
from processors.base_processor import BaseProcessor
from servers import mcp_server_config_center as server
from tests.integration.test_shared_document_store import (
    ControlledDocumentEmbedder,
    ControlledGenerator,
    ControlledQueryEmbedder,
)


class ControlledInputProcessor(BaseProcessor):
    """Only the external text-extraction boundary is substituted with declared fixture text."""

    def __init__(self, config=None):
        super().__init__(name="ControlledInputProcessor", config=config)

    def process(self, document):
        document.store_content("ocr", {"text": "Controlled uploaded source text."})
        document.add_metadata("title", "Controlled source")
        return ProcessResult.success_result()


@pytest.mark.asyncio
async def test_config_center_source_pipeline_resources_and_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HAYSTACK_TELEMETRY_ENABLED", "False")
    config = generate_default_config()
    config["storage"]["base_path"] = str(tmp_path)
    config["storage"]["output_path"] = str(tmp_path / "output")
    config["llm"] = {"provider": "openai", "model": "gpt-3.5-turbo", "api_key": "controlled-fixture"}
    config["processors"] = {
        name: {"enabled": name in {"pre_processor", "embedding_processor"}, "config": {}}
        for name in ["pre_processor", "ocr_processor", "structure_processor", "embedding_processor"]
    }
    config["processor_mappings"] = {
        "pre_processor": {"module": __name__, "class": "ControlledInputProcessor"},
        "embedding_processor": {
            "module": "processors.haystack_embedding_processor",
            "class": "HaystackEmbeddingProcessor",
        },
    }
    directory = tmp_path / "configuration"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(config))
    center = ConfigCenter(str(directory), environment="test", watch_changes=False)
    monkeypatch.setattr(server, "config_center", center)
    monkeypatch.setattr(server, "server_context", None)
    generator = ControlledGenerator()
    with (
        patch("socket.socket.connect", side_effect=AssertionError("No external services in this contract")),
        patch("haystack.telemetry._telemetry.telemetry", None),
        patch("connectors.openai_connector.OpenAIChatGenerator", return_value=generator),
        patch("rag.haystack_pipeline.SentenceTransformersTextEmbedder", return_value=ControlledQueryEmbedder()),
        patch(
            "processors.haystack_embedding_processor.SentenceTransformersDocumentEmbedder",
            return_value=ControlledDocumentEmbedder(),
        ),
    ):
        try:
            assert await server.initialize_server_context()
            context = server.server_context
            assert context.config_manager is center.config_manager
            resources = await server.handle_list_resources()
            assert "config://current" in {str(resource.uri) for resource in resources}
            actual = json.loads(await server.handle_read_resource(AnyUrl("config://current")))
            assert actual == center.get_config()
            tools = {tool.name: tool for tool in await server.handle_list_tools()}
            assert tools["process_document"].inputSchema["required"] == ["file_path"]
            legacy = await server.handle_process_document({"content": "obsolete phantom input"})
            assert "file_path is required" in legacy[0].text
            path = tmp_path / "controlled.pdf"
            path.write_text("Fixture input; no PDF extraction claim")
            processed = await server.handle_process_document({"file_path": str(path)})
            assert "处理成功" in processed[0].text
            assert context.rag_pipeline.document_store.count_documents() == 1
            answer = await server.handle_query_documents({"query": "source", "top_k": 1})
            assert "Controlled answer" in answer[0].text
            assert any("Controlled uploaded source text" in (message.text or "") for message in generator.messages)
            missing = await server.handle_process_document({"file_path": str(tmp_path / "missing.pdf")})
            assert "not found" in missing[0].text
        finally:
            if server.server_context is not None:
                server.server_context.cleanup()
            center.close()


@pytest.mark.asyncio
async def test_uninitialized_configuration_center_reports_failure(monkeypatch):
    monkeypatch.setattr(server, "config_center", None)
    monkeypatch.setattr(server, "server_context", None)
    assert not await server.initialize_server_context()
    reply = await server.handle_process_document({"file_path": "controlled.pdf"})
    assert "未初始化" in reply[0].text
