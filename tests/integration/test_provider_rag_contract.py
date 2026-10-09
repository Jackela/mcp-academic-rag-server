"""Provider-neutral generation uses actual Haystack messages, retrieval and per-call overrides."""

from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest
from haystack import component
from haystack.dataclasses import ChatMessage, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

from connectors.base_llm_connector import BaseLLMConnector
from connectors.haystack_llm_connector import ConnectorChatGenerator
from connectors.openai_connector import OpenAIConnector
from rag.haystack_pipeline import RAGPipeline, RAGPipelineFactory


@component
class ControlledQueryEmbedder:
    @component.output_types(embedding=List[float])
    def run(self, text: str) -> Dict[str, List[float]]:
        return {"embedding": [1.0, 0.0]}


class ClientOnlyConnector(BaseLLMConnector):
    """Existing Google/Anthropic shape: neutral generate API, no Haystack generator attribute."""

    def __init__(self) -> None:
        super().__init__("controlled", "client-only", parameters={"temperature": 0.2})
        self.calls: List[Any] = []
        self.reply: Dict[str, Any] = {"content": "Provider answer"}

    def _get_provider_name(self) -> str:
        return "controlled-client"

    def _init_generator(self) -> None:
        pass

    def generate(
        self, messages: List[Dict[str, str]], generation_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        self.calls.append((messages, generation_kwargs))
        return self.reply


def test_real_rag_routes_client_only_provider_and_call_options():
    connector = ClientOnlyConnector()
    assert not hasattr(connector, "generator")
    store = InMemoryDocumentStore()
    store.write_documents([Document(id="source", content="Controlled source", embedding=[1.0, 0.0])])
    with (
        patch("rag.haystack_pipeline.SentenceTransformersTextEmbedder", return_value=ControlledQueryEmbedder()),
        patch("socket.socket.connect", side_effect=AssertionError("No external provider")),
    ):
        pipeline = RAGPipeline(connector, store)
        result = pipeline.run("What is the source?", generation_kwargs={"temperature": 0.7, "max_tokens": 17})
        assert "error" not in result, result
        assert result["answer"] == "Provider answer"
        assert result["documents"][0]["id"] == "source"
        messages, parameters = connector.calls[0]
        assert any("Controlled source" in message["content"] for message in messages)
        assert parameters == {"temperature": 0.7, "max_tokens": 17}
        assert connector.parameters == {"temperature": 0.2}
        connector.reply = {"content": "Failure text", "error": "Provider rejected request"}
        failed = pipeline.run("Try again")
        assert "Provider rejected request" in failed["error"]
        assert failed["documents"] == []


def test_adapter_rejects_empty_provider_response():
    connector = ClientOnlyConnector()
    connector.reply = {"content": ""}
    with pytest.raises(ValueError, match="does not contain text"):
        ConnectorChatGenerator(connector).run([ChatMessage.from_user("query")])


def test_openai_uses_native_text_and_existing_generator_per_call():
    class ControlledGenerator:
        def __init__(self, **options):
            self.calls = []

        def run(self, **options):
            self.calls.append(options)
            return {"replies": [ChatMessage.from_assistant("Actual native text")]}

    with patch("connectors.openai_connector.OpenAIChatGenerator", ControlledGenerator):
        connector = OpenAIConnector("controlled", parameters={"temperature": 0.1})
        result = connector.generate([{"role": "user", "content": "question"}], {"temperature": 0.9})
        assert result["content"] == "Actual native text"
        assert "error" not in result
        assert len(connector.generator.calls) == 1
        assert connector.generator.calls[0]["generation_kwargs"] == {"temperature": 0.9}
        assert connector.parameters == {"temperature": 0.1}


@pytest.mark.parametrize("config, expected", [({"top_k": 1}, 1), ({"top_k": 1, "retriever_top_k": 2}, 2)])
def test_factory_legacy_top_k_and_explicit_precedence(config, expected):
    store = InMemoryDocumentStore()
    store.write_documents(
        [Document(id=str(index), content=f"Source {index}", embedding=[1.0, 0.0]) for index in range(3)]
    )
    with patch("rag.haystack_pipeline.SentenceTransformersTextEmbedder", return_value=ControlledQueryEmbedder()):
        pipeline = RAGPipelineFactory.create_pipeline(ClientOnlyConnector(), store, config=config)
        result = pipeline.run("source")
        assert "error" not in result, result
        assert len(result["documents"]) == expected


@pytest.mark.parametrize(
    "config", [{"top_k": 0}, {"top_k": "2"}, {"retriever_top_k": None, "top_k": 2}, {"top_k": True}]
)
def test_factory_invalid_top_k_is_not_silently_replaced(config):
    with pytest.raises(ValueError, match="positive integer"):
        RAGPipelineFactory.create_pipeline(ClientOnlyConnector(), config=config)
