"""Current OCRProcessor contract: process(Document) with a connector stub."""

import unittest
from unittest.mock import MagicMock, patch

from connectors.api_connector import MistralAPIConnector
from models.document import Document
from processors.ocr_processor import OCRProcessor


class OCRTests(unittest.TestCase):
    def setUp(self):
        self.connector = MagicMock(spec=MistralAPIConnector)
        with patch("processors.ocr_processor.OCRAPIFactory.create_connector", return_value=self.connector):
            self.processor = OCRProcessor({"api_config": {"api_key": "fixture"}})
        self.document = Document("fixture.pdf")
        self.connector.upload_file.return_value = {"id": "fixture-id"}
        self.connector.get_signed_url.return_value = {"url": "https://example.invalid/fixture.pdf"}
        self.connector.process_document_ocr.return_value = {
            "pages": [{"index": 0, "markdown": "# First page"}, {"index": 1, "markdown": "Second page"}]
        }

    def test_current_api_pages_are_stored(self):
        result = self.processor.process(self.document)
        self.assertTrue(result.is_successful())
        self.assertEqual(result.data["text_by_page"], ["# First page", "Second page"])
        self.assertEqual(self.document.get_content("ocr"), result.data)
        self.connector.upload_file.assert_called_once_with(self.document.file_path, purpose="ocr")
        self.connector.get_signed_url.assert_called_once_with("fixture-id")
        self.connector.process_document_ocr.assert_called_once_with(
            "https://example.invalid/fixture.pdf", model="mistral-ocr-latest"
        )
        self.assertEqual(self.processor.config["api_config"]["api_url"], "https://api.mistral.ai/v1")

    def test_image_and_legacy_text_response(self):
        self.connector.process_image_ocr.return_value = {"pages": [{"text": "Legacy text"}]}
        result = self.processor.process(Document("fixture.PNG"))
        self.assertTrue(result.is_successful())
        self.assertEqual(result.data["text"], "Legacy text")

    def test_upload_url_and_ocr_failures_never_succeed(self):
        for phase in ["upload", "url", "ocr", "empty"]:
            with self.subTest(phase=phase):
                self.setUp()
                if phase == "upload":
                    self.connector.upload_file.return_value = {}
                elif phase == "url":
                    self.connector.get_signed_url.return_value = {}
                elif phase == "ocr":
                    self.connector.process_document_ocr.side_effect = RuntimeError("fixture failure")
                else:
                    self.connector.process_document_ocr.return_value = {"pages": [{"markdown": ""}]}
                self.assertFalse(self.processor.process(self.document).is_successful())
                self.assertIsNone(self.document.get_content("ocr"))

    def test_missing_connector_and_partial_results_fail(self):
        self.processor.api_connector = None
        self.assertFalse(self.processor.process(self.document).is_successful())
        self.processor.api_connector = self.connector
        self.document.store_content("pre_processing", {"processed_files": ["one.pdf", "two.pdf"]})
        self.connector.upload_file.side_effect = [{"id": "fixture-id"}, {}]
        self.assertFalse(self.processor.process(self.document).is_successful())
        self.assertIsNone(self.document.get_content("ocr"))
