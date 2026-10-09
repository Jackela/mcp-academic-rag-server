"""Exercise existing processor, real Haystack pipeline and sessions offline.

Text and embeddings are controlled fixtures, not PDF extraction or model results.
The former fixture referenced DocumentProcessor/EmbeddingConnector factories that
never existed. Only the current public APIs are exercised here.
"""

from dataclasses import replace
from typing import List
from unittest.mock import patch

import pytest
from haystack import component
from haystack.dataclasses import ChatMessage
from haystack.dataclasses import Document as HaystackDocument

from connectors.haystack_llm_connector import HaystackLLMFactory
from document_stores.implementations.haystack_store import HaystackDocumentStore
from models.document import Document
from processors.haystack_embedding_processor import HaystackEmbeddingProcessor
from rag.chat_session import ChatSessionManager
from rag.haystack_pipeline import RAGPipelineFactory
from rag.prompt_builder import PromptBuilderFactory


@component
class FixtureDocumentEmbedder:
    @component.output_types(documents=List[HaystackDocument])
    def run(self, documents: List[HaystackDocument]):
        return {"documents": [replace(doc, embedding=[1.0, 0.0]) for doc in documents]}


@component
class FixtureQueryEmbedder:
    @component.output_types(embedding=List[float])
    def run(self, text: str):
        return {"embedding": [1.0, 0.0]}


@component
class FixtureGenerator:
    def __init__(self):
        self.calls = []
        self.fail = False

    @component.output_types(replies=List[ChatMessage])
    def run(self, messages: List[ChatMessage], generation_kwargs: dict = None):
        if self.fail:
            raise RuntimeError("fixture model failure")
        self.calls.append(messages)
        return {"replies": [ChatMessage.from_assistant("Controlled fixture answer")]}


@pytest.fixture
def rag_system():
    store = HaystackDocumentStore({"type": "memory", "similarity": "cosine"})
    generator = FixtureGenerator()
    with (
        patch("connectors.haystack_llm_connector.OpenAIChatGenerator", return_value=generator),
        patch("rag.haystack_pipeline.SentenceTransformersTextEmbedder", return_value=FixtureQueryEmbedder()),
        patch(
            "processors.haystack_embedding_processor.SentenceTransformersDocumentEmbedder",
            return_value=FixtureDocumentEmbedder(),
        ),
    ):
        llm = HaystackLLMFactory.create_connector({"api_key": "sk-offline-fixture", "model": "fixture"})
        prompt = PromptBuilderFactory.create_builder(config={"template_type": "academic"})
        pipeline = RAGPipelineFactory.create_pipeline(
            llm_connector=llm, document_store=store.get_document_store(), prompt_builder=prompt
        )
        processor = HaystackEmbeddingProcessor(document_store=store, config={"chunk_size": 20, "chunk_overlap": 2})
    yield store, processor, pipeline, generator, ChatSessionManager(rag_pipeline=pipeline)


def add_document(processor):
    document = Document("fixture.txt")
    document.store_content("ocr", {"text": "Controlled source text about fixture science."})
    document.add_metadata("title", "Fixture title")
    result = processor.process(document)
    assert result.is_successful(), result.get_message()
    assert result.get_data()["chunks"] == 1
    assert result.get_data()["embeddings"] == [[1.0, 0.0]]
    return document


def test_document_processing_to_haystack(rag_system):
    store, processor, _, _, _ = rag_system
    document = add_document(processor)
    docs = store.get_document_store().filter_documents({})
    assert len(docs) == 1
    assert docs[0].embedding == [1.0, 0.0]
    assert docs[0].meta["original_id"] == document.document_id
    assert docs[0].meta["metadata"]["title"] == "Fixture title"
    empty = processor.process(Document("empty.txt"))
    assert not empty.is_successful()
    assert store.get_document_count() == 1


def test_rag_query_execution(rag_system):
    _, processor, pipeline, generator, _ = rag_system
    add_document(processor)
    result = pipeline.run("What does the fixture say?")
    assert result["answer"] == "Controlled fixture answer"
    assert len(result["documents"]) == 1
    assert "Controlled source text" in generator.calls[0][-1].text
    assert "What does the fixture say?" in generator.calls[0][-1].text


def test_chat_session_with_rag(rag_system):
    _, processor, _, _, manager = rag_system
    document = add_document(processor)
    session = manager.create_session(session_id="integration-fixture")
    response, documents = session.process_query("Fixture question")
    assert response.role == "assistant"
    assert response.content == "Controlled fixture answer"
    assert documents[0]["metadata"]["original_id"] == document.document_id
    assert session.citations[response.message_id][0].document_id == documents[0]["id"]


def test_multi_turn_conversation(rag_system):
    _, processor, _, generator, manager = rag_system
    add_document(processor)
    session = manager.create_session()
    session.process_query("First fixture question")
    session.process_query("Follow-up fixture question")
    assert len(generator.calls) == 2
    assert [msg.text for msg in generator.calls[1][1:3]] == ["First fixture question", "Controlled fixture answer"]
    assert len(session.get_messages()) == 4


def test_error_handling(rag_system):
    _, processor, _, generator, manager = rag_system
    add_document(processor)
    session = manager.create_session()
    generator.fail = True
    failed, documents = session.process_query("Failure fixture")
    assert "查询处理失败" in failed.content
    assert documents == []
    generator.fail = False
    recovered, documents = session.process_query("Recovery fixture")
    assert recovered.content == "Controlled fixture answer"
    assert len(documents) == 1


def test_session_persistence_and_recovery(rag_system, tmp_path):
    _, processor, pipeline, _, manager = rag_system
    add_document(processor)
    session = manager.create_session(session_id="persisted-fixture")
    response, _ = session.process_query("Saved fixture")
    path = tmp_path / "sessions.json"
    assert manager.save_sessions(str(path))
    restored = ChatSessionManager(rag_pipeline=pipeline)
    assert restored.load_sessions(str(path))
    loaded = restored.get_session(session.session_id)
    assert loaded.get_messages() == session.get_messages()
    assert {key: [citation.to_dict() for citation in values] for key, values in loaded.citations.items()} == {
        key: [citation.to_dict() for citation in values] for key, values in session.citations.items()
    }
    assert response.message_id in loaded.citations
    next_response, _ = loaded.process_query("Next fixture")
    assert next_response.content == "Controlled fixture answer"


def test_batch_embedding_keeps_per_document_results(rag_system):
    store, processor, _, _, _ = rag_system
    current = Document("current.txt")
    current.store_content("ocr", {"text": "Current controlled fixture text."})
    legacy = Document("legacy.txt")
    legacy.store_content("OCRProcessor", "Legacy controlled fixture text.")
    empty = Document("empty.txt")
    results = processor.batch_process([current, legacy, empty])
    assert set(results) == {current.document_id, legacy.document_id, empty.document_id}
    assert results[current.document_id].is_successful()
    assert results[legacy.document_id].is_successful()
    assert results[current.document_id].get_data()["average_embedding"] == [1.0, 0.0]
    assert not results[empty.document_id].is_successful()
    assert store.get_document_count() == 2


def test_batch_embedding_failure_preserves_missing_text_error(rag_system):
    store, processor, _, _, _ = rag_system
    current = Document("current.txt")
    current.store_content("ocr", {"text": "Controlled fixture text."})
    empty = Document("empty.txt")
    with patch.object(processor.pipeline, "run", side_effect=RuntimeError("fixture embedding failure")):
        results = processor.batch_process([empty, current])
    assert not results[current.document_id].is_successful()
    assert "fixture embedding failure" in results[current.document_id].get_message()
    assert "没有可用的文本" in results[empty.document_id].get_message()
    assert store.get_document_count() == 0
