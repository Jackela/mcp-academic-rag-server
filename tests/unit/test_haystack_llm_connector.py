"""Validate the installed Haystack message contract without model requests."""

from unittest.mock import MagicMock, patch

from haystack.dataclasses import ChatMessage

from connectors.haystack_llm_connector import HaystackLLMConnector


def test_generate_uses_haystack_messages_and_per_call_parameters():
    generator = MagicMock()
    generator.run.return_value = {"replies": [ChatMessage.from_assistant("controlled answer")]}
    with patch("connectors.haystack_llm_connector.OpenAIChatGenerator", return_value=generator):
        connector = HaystackLLMConnector(api_key="offline-fixture", parameters={"temperature": 0.1})
    result = connector.generate([{"role": "user", "content": "controlled query"}], {"max_tokens": 42})
    assert result == {"content": "controlled answer", "role": "assistant", "model": connector.model}
    call = generator.run.call_args.kwargs
    assert call["messages"][0].to_dict() == ChatMessage.from_user("controlled query").to_dict()
    assert call["generation_kwargs"] == {"temperature": 0.1, "max_tokens": 42}
    assert connector.parameters == {"temperature": 0.1}
    connector.generate([{"role": "user", "content": "next query"}])
    assert generator.run.call_args.kwargs["generation_kwargs"] == {"temperature": 0.1}


def test_generate_preserves_empty_and_failed_reply_errors():
    generator = MagicMock()
    with patch("connectors.haystack_llm_connector.OpenAIChatGenerator", return_value=generator):
        connector = HaystackLLMConnector(api_key="offline-fixture")
    generator.run.return_value = {"replies": [ChatMessage.from_assistant("")]}
    empty = connector.generate([{"role": "user", "content": "query"}])
    assert "LLM reply does not contain text" in empty["error"]
    generator.run.side_effect = RuntimeError("controlled provider failure")
    failed = connector.generate([{"role": "user", "content": "query"}])
    assert failed["error"] == "controlled provider failure"
