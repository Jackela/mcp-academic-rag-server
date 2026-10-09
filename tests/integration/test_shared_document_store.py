"""Upload processing and RAG retrieval must use one real Haystack document store."""

import json
from dataclasses import replace
from typing import List
from unittest.mock import patch

from haystack import component
from haystack.dataclasses import ChatMessage
from haystack.dataclasses import Document as HaystackDocument

from core.config_manager import ConfigManager
from core.config_validator import generate_default_config
from core.server_context import ServerContext
from models.document import Document
from processors.haystack_embedding_processor import HaystackEmbeddingProcessor

VECTOR = [1.0] + [0.0] * 383


@component
class ControlledDocumentEmbedder:
    @component.output_types(documents=List[HaystackDocument])
    def run(self, documents: List[HaystackDocument]):
        return {"documents": [replace(document, embedding=VECTOR) for document in documents]}


@component
class ControlledQueryEmbedder:
    @component.output_types(embedding=List[float])
    def run(self, text: str):
        return {"embedding": VECTOR}


@component
class ControlledGenerator:
    def __init__(self):
        self.messages = []

    @component.output_types(replies=List[ChatMessage])
    def run(self, messages: List[ChatMessage], generation_kwargs: dict = None):
        self.messages = messages
        return {"replies": [ChatMessage.from_assistant("Controlled answer")]}


def test_processed_document_is_retrieved_with_source_metadata(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HAYSTACK_TELEMETRY_ENABLED", "False")
    config = generate_default_config()
    config["storage"]["base_path"] = str(tmp_path)
    config["storage"]["output_path"] = str(tmp_path / "output")
    config["llm"] = {"provider": "openai", "model": "gpt-3.5-turbo", "api_key": "controlled-fixture"}
    config["processors"] = {
        name: {"enabled": name == "embedding_processor", "config": {"chunk_size": 8, "chunk_overlap": 0}}
        for name in ["pre_processor", "ocr_processor", "structure_processor", "embedding_processor"]
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    manager = ConfigManager(str(config_path))
    assert manager.is_config_valid(), manager.get_validation_report()
    context = ServerContext()
    context._config_manager = manager
    early_session = context.session_manager.create_session()
    generator = ControlledGenerator()
    with (
        patch("socket.socket.connect", side_effect=AssertionError("No external service in this contract")),
        patch("haystack.telemetry._telemetry.telemetry", None),
        patch("connectors.openai_connector.OpenAIChatGenerator", return_value=generator),
        patch("rag.haystack_pipeline.SentenceTransformersTextEmbedder", return_value=ControlledQueryEmbedder()),
        patch(
            "processors.haystack_embedding_processor.SentenceTransformersDocumentEmbedder",
            return_value=ControlledDocumentEmbedder(),
        ),
    ):
        try:
            context.initialize()
            assert context.rag_pipeline is not None
            assert early_session.rag_pipeline is context.rag_pipeline
            processor = context.processors[0]
            assert isinstance(processor, HaystackEmbeddingProcessor)
            assert processor.document_store.document_store is context.rag_pipeline.document_store
            source = Document(tmp_path / "controlled.pdf")
            source.add_metadata("title", "Controlled source")
            source.store_content("OCRProcessor", {"text": "Controlled retrieved text from the uploaded source."})
            processed = processor.process(source)
            assert processed.success, processed.message
            assert context.rag_pipeline.document_store.count_documents() == 1
            result = context.rag_pipeline.run("What does the source say?")
            assert "error" not in result, result
            assert result["answer"] == "Controlled answer"
            assert len(result["documents"]) == 1
            retrieved = result["documents"][0]
            assert retrieved["metadata"]["original_id"] == source.document_id
            assert retrieved["metadata"]["metadata"]["title"] == "Controlled source"
            assert "Controlled retrieved text" in retrieved["content"]
            assert any("Controlled retrieved text" in (message.text or "") for message in generator.messages)
        finally:
            context.cleanup()
        assert context.rag_pipeline is None
        assert context.processors == []
