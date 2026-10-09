"""Real Web/Haystack runtime; only model generation and embeddings are controlled fixtures."""

import importlib
import json
import os
import socket
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import List
from unittest.mock import patch
from urllib.parse import urlparse

import httpx
import requests
from haystack import component
from haystack.dataclasses import ChatMessage, Document

from core.config_manager import ConfigManager
from core.config_validator import generate_default_config
from models.process_result import ProcessResult
from processors.base_processor import BaseProcessor


class FixtureInputProcessor(BaseProcessor):
    """Controlled processing boundary; it does not extract or OCR a document."""

    def __init__(self, config=None):
        super().__init__(name="controlled-web-input", config=config or {})

    def process(self, document):
        document.store_content("FixtureInputProcessor", "Controlled fixture content")
        return ProcessResult.success_result("Controlled fixture processing")


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
            raise RuntimeError("controlled Web generator failure")
        self.calls.append(messages)
        return {"replies": [ChatMessage.from_assistant("Controlled Web fixture answer")]}


@contextmanager
def web_runtime(directory: Path):
    config = generate_default_config()
    config["processors"] = {
        name: {"enabled": name == "pre_processor", "config": {}}
        for name in ["pre_processor", "ocr_processor", "structure_processor", "embedding_processor"]
    }
    config["processor_mappings"] = {
        "pre_processor": {"module": "tests.fixtures.web_runtime", "class": "FixtureInputProcessor"}
    }
    config["llm"] = {"provider": "openai", "model": "gpt-3.5-turbo", "api_key": "${OPENAI_API_KEY}"}
    config_path = directory / "config.json"
    config_path.write_text(json.dumps(config))
    manager = ConfigManager(str(config_path))
    assert manager.is_config_valid(), manager.get_validation_report()
    generator = FixtureGenerator()
    previous_cwd = Path.cwd()
    previous_module = sys.modules.pop("webapp", None)
    real_connect = socket.socket.connect

    def local_connect(sock, address):
        if isinstance(address, tuple) and address[0] not in {"127.0.0.1", "localhost", "::1"}:
            raise AssertionError("Web contracts may connect only to local fixture services")
        return real_connect(sock, address)

    real_httpx_send = httpx.Client.send
    real_requests_send = requests.Session.send

    def local_httpx_send(client, request, *args, **kwargs):
        if request.url.host not in {"127.0.0.1", "localhost", "::1"}:
            raise AssertionError("No external HTTP request is permitted in Web fixtures")
        return real_httpx_send(client, request, *args, **kwargs)

    def local_requests_send(client, request, *args, **kwargs):
        if urlparse(request.url).hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise AssertionError("No external HTTP request is permitted in Web fixtures")
        return real_requests_send(client, request, *args, **kwargs)

    os.chdir(directory)
    try:
        with (
            patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "sk-offline-controlled-web-fixture",
                    "HAYSTACK_TELEMETRY_ENABLED": "False",
                    "TESTING": "false",
                },
            ),
            patch("socket.socket.connect", local_connect),
            patch("httpx.Client.send", local_httpx_send),
            patch("haystack.telemetry._telemetry.telemetry", None),
            patch("requests.Session.send", local_requests_send),
            patch("core.server_context.ConfigManager", return_value=manager),
            patch("connectors.openai_connector.OpenAIChatGenerator", return_value=generator),
            patch("rag.haystack_pipeline.SentenceTransformersTextEmbedder", return_value=FixtureQueryEmbedder()),
        ):
            module = importlib.import_module("webapp")
            module.app.config.update(TESTING=False, UPLOAD_FOLDER=str(directory / "uploads"))
            assert module.rag_pipeline is not None, "The actual RAG runtime must initialize"
            assert module.rag_pipeline.llm_connector.generator is generator, "Model requests must remain controlled"
            if module.rag_pipeline:
                module.rag_pipeline.document_store.write_documents(
                    [
                        Document(
                            id="web-table",
                            content="| Parameter | Value |\n| --- | --- |\n| Temperature | 25.5 |",
                            embedding=[1.0, 0.0],
                            meta={"title": "Controlled table source"},
                        ),
                        Document(
                            id="web-code",
                            content="```python\ndef fixture_average(data):\n    return sum(data) / len(data)\n```",
                            embedding=[1.0, 0.0],
                            meta={"title": "Controlled code source"},
                        ),
                    ]
                )
            try:
                yield module, generator
            finally:
                module.server_context.cleanup()
    finally:
        os.chdir(previous_cwd)
        sys.modules.pop("webapp", None)
        if previous_module is not None:
            sys.modules["webapp"] = previous_module
