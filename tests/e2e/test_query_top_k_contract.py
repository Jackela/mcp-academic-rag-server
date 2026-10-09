"""MCP retrieval limits reach real ranked Haystack retrieval without changing defaults."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from mcp.types import CallToolResult

from rag.chat_session import ChatSessionManager
from rag.haystack_pipeline import RAGPipeline
from servers import mcp_server_sdk as sdk
from tests.integration.test_provider_rag_contract import ClientOnlyConnector, ControlledQueryEmbedder


@pytest.fixture
def query_context():
    store = InMemoryDocumentStore(embedding_similarity_function="cosine")
    store.write_documents(
        [
            Document(id="first", content="Ranked first source", embedding=[1.0, 0.0]),
            Document(id="second", content="Ranked second source", embedding=[0.8, 0.2]),
            Document(id="third", content="Ranked third source", embedding=[0.0, 1.0]),
        ]
    )
    with (
        patch("rag.haystack_pipeline.SentenceTransformersTextEmbedder", return_value=ControlledQueryEmbedder()),
        patch("socket.socket.connect", side_effect=AssertionError("No external query/model request")),
    ):
        connector = ClientOnlyConnector()
        pipeline = RAGPipeline(connector, store, retriever_top_k=3)
        manager = ChatSessionManager(rag_pipeline=pipeline)
        try:
            yield SimpleNamespace(
                is_initialized=True, rag_pipeline=pipeline, session_manager=manager, connector=connector
            )
        finally:
            manager.sessions.clear()


@pytest.mark.asyncio
async def test_tool_per_call_top_k_preserves_ranking_and_configured_default(query_context):
    result = await sdk.handle_query_documents({"query": "ranked", "top_k": 1}, context=query_context)
    assert "Ranked first source" in result[0].text
    assert "Ranked second source" not in result[0].text
    actual_prompt = "\n".join(message["content"] for message in query_context.connector.calls[-1][0])
    assert "Ranked first source" in actual_prompt and "Ranked second source" not in actual_prompt
    next_result = await sdk.handle_query_documents({"query": "ranked", "top_k": 2}, context=query_context)
    assert next_result[0].text.index("Ranked first source") < next_result[0].text.index("Ranked second source")
    assert "Ranked third source" not in next_result[0].text
    actual_prompt = "\n".join(message["content"] for message in query_context.connector.calls[-1][0])
    assert "Ranked second source" in actual_prompt and "Ranked third source" not in actual_prompt
    limited = query_context.rag_pipeline.run("ranked", top_k=1)
    assert [doc["id"] for doc in limited["documents"]] == ["first"]
    configured = query_context.rag_pipeline.run("ranked")
    assert [doc["id"] for doc in configured["documents"]] == ["first", "second", "third"]


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, -1, True, "2", None])
async def test_invalid_tool_limit_is_native_error_before_query(query_context, monkeypatch, limit):
    monkeypatch.setattr(sdk, "server_context", query_context)
    result = await sdk.handle_call_tool("query_documents", {"query": "ranked", "top_k": limit})
    assert isinstance(result, CallToolResult) and result.isError
    assert "positive integer" in result.content[0].text
    assert query_context.session_manager.sessions == {}
