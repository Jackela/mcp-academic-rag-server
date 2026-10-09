import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from mcp.types import CallToolResult

from connectors.api_connector import MistralAPIConnector
from models.process_result import ProcessResult
from servers import mcp_server_sdk as sdk


class SDKContracts(unittest.IsolatedAsyncioTestCase):
    async def test_tools_have_required_schemas_and_echo(self):
        tools = {tool.name: tool for tool in await sdk.handle_list_tools()}
        self.assertEqual(set(tools), {"test_connection", "validate_system", "process_document", "query_documents"})
        self.assertEqual(tools["process_document"].inputSchema["required"], ["file_path"])
        self.assertIn(
            "hello fixture", (await sdk.handle_call_tool("test_connection", {"message": "hello fixture"}))[0].text
        )

    async def test_unknown_and_missing_arguments_fail(self):
        for name, arguments, expected in [
            ("bad", {}, "Unknown tool"),
            ("process_document", {}, "required"),
            ("query_documents", {}, "required"),
            ("process_document", {"file_path": "/nonexistent/fixture.pdf"}, "not found"),
        ]:
            with self.subTest(tool=name, arguments=arguments):
                result = await sdk.handle_call_tool(name, arguments)
                self.assertIsInstance(result, CallToolResult)
                self.assertTrue(result.isError)
                self.assertIn(expected, result.content[0].text)
        # Direct processing helpers retain their historical list interface.
        helper = await sdk.handle_process_document({})
        self.assertIsInstance(helper, list)
        self.assertIn("required", helper[0].text)

    async def test_processing_success_and_failure(self):
        context = MagicMock()
        context.is_initialized = True
        context.document_pipeline.process_document = AsyncMock(return_value=ProcessResult.success_result())
        with tempfile.TemporaryDirectory() as directory, patch.object(sdk, "server_context", context):
            path = Path(directory) / "fixture.pdf"
            path.write_text("local fixture")
            result = await sdk.handle_call_tool("process_document", {"file_path": str(path)})
            self.assertIn("文档处理成功", result[0].text)
            context.document_pipeline.process_document.return_value = ProcessResult.error_result("fixture failure")
            failed = await sdk.handle_call_tool("process_document", {"file_path": str(path)})
            self.assertIsInstance(failed, CallToolResult)
            self.assertTrue(failed.isError)
            self.assertIn("fixture failure", failed.content[0].text)

    async def test_environment_missing_key_fails_without_network(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            with self.assertRaises(EnvironmentError):
                sdk.validate_environment()


class ConnectorContracts(unittest.TestCase):
    def test_ocr_url_payload_and_signed_url(self):
        connector = MistralAPIConnector("https://api.mistral.ai/v1", "fixture")
        with patch.object(connector, "make_request", return_value={"pages": []}) as request:
            connector.process_document_ocr("https://example.invalid/doc.pdf")
            self.assertEqual(request.call_args.args, ("POST", "ocr"))
            self.assertEqual(
                request.call_args.kwargs["json_data"]["document"],
                {"type": "document_url", "document_url": "https://example.invalid/doc.pdf"},
            )
            connector.get_signed_url("file-id")
            self.assertEqual(request.call_args.args, ("GET", "files/file-id/url"))

    def test_upload_multipart_has_no_json_header_and_closes_file(self):
        connector = MistralAPIConnector("https://api.mistral.ai/v1", "fixture")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.pdf"
            path.write_bytes(b"local fixture")
            response = MagicMock()
            response.headers = {"content-type": "application/json"}
            response.json.return_value = {"id": "fixture"}
            with patch("connectors.api_connector.requests.request", return_value=response) as request:
                connector.upload_file(str(path))
                self.assertEqual(request.call_args.kwargs["url"], "https://api.mistral.ai/v1/files")
                self.assertNotIn("Content-Type", request.call_args.kwargs["headers"])
                self.assertTrue(request.call_args.kwargs["files"]["file"][1].closed)
