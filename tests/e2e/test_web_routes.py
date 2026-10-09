"""Exercise real Flask routes and maintained Haystack/session contracts without external services."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models.document import Document
from models.process_result import ProcessResult
from tests.fixtures.web_runtime import web_runtime


class TestWebRoutes(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.runtime = web_runtime(Path(self.directory.name))
        self.web, self.generator = self.runtime.__enter__()
        self.addCleanup(self.runtime.__exit__, None, None, None)
        self.client = self.web.app.test_client()

    def test_real_context_binds_sessions_and_formats_retrieved_documents(self):
        self.assertIs(self.web.rag_pipeline, self.web.server_context.rag_pipeline)
        self.assertIs(self.web.session_manager, self.web.server_context.session_manager)
        response = self.client.post("/api/chat", json={"query": "Controlled first question"})
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        body = response.get_json()
        self.assertEqual(body["answer"], "Controlled Web fixture answer")
        self.assertEqual({item["structured_content"]["type"] for item in body["citations"]}, {"table", "code"})
        self.assertEqual({item["document_id"] for item in body["citations"]}, {"web-table", "web-code"})
        with self.client.session_transaction() as session:
            chat = self.web.session_manager.get_session(session["session_id"])
        self.assertEqual([message.role for message in chat.messages], ["user", "assistant"])
        self.assertEqual(self.client.get("/chat").status_code, 200)
        self.assertEqual(self.client.post("/api/chat/reset").status_code, 200)
        self.assertEqual(chat.messages, [])

    def test_unavailable_and_failed_generation_do_not_report_success(self):
        self.generator.fail = True
        failed = self.client.post("/api/chat", json={"query": "Controlled failure"})
        self.assertEqual(failed.status_code, 502)
        self.assertIn("error", failed.get_json())
        self.web.rag_pipeline = None
        unavailable = self.client.post("/api/chat", json={"query": "Controlled unavailable"})
        self.assertEqual(unavailable.status_code, 503)
        self.assertIn("error", unavailable.get_json())

    def test_invalid_query_has_an_explicit_client_error(self):
        for data in [{}, {"query": " "}, {"query": 42}, []]:
            with self.subTest(data=data):
                self.assertEqual(self.client.post("/api/chat", json=data).status_code, 400)

    def test_background_thread_runs_the_actual_async_pipeline_contract(self):
        called = []

        class AsyncPipeline:
            async def process_document(self, document):
                called.append(document)
                return ProcessResult.success_result("Controlled local processing")

        self.web.processing_pipeline = AsyncPipeline()
        document = Document("fixture.pdf")
        self.web.process_document_async(document, document.document_id)
        self.assertEqual(called, [document])
        self.assertEqual(self.web.document_status[document.document_id], "completed")

    def test_startup_defaults_to_loopback_and_allows_explicit_host(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(self.web.app, "run") as run:
            self.web.run_web_app()
            run.assert_called_once_with(host="127.0.0.1", port=5000, debug=False)
        with (
            patch.dict(os.environ, {"HOST": "192.0.2.1", "FLASK_ENV": "development"}),
            patch.object(self.web.app, "run") as run,
        ):
            self.web.run_web_app()
            run.assert_called_once_with(host="192.0.2.1", port=5000, debug=True)
