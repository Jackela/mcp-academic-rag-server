"""CLI entry contracts against the actual local pipelines and persisted documents."""

import argparse
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from cli.chat_cli import ChatCLI
from cli.document_cli import DocumentCLI
from core.config_validator import generate_default_config
from models.document import Document
from models.process_result import ProcessResult
from processors.base_processor import BaseProcessor
from tests.fixtures.web_runtime import web_runtime


class ControlledTextProcessor(BaseProcessor):
    def __init__(self, fail=False):
        super().__init__(name="cli-controlled-text")
        self.fail = fail

    def process(self, document):
        if self.fail:
            return ProcessResult.error_result("controlled failure")
        document.store_content("ocr", {"text": "Real controlled processor content"})
        return ProcessResult.success_result("controlled success")


class CLIRuntimeTests(TestCase):
    def test_chat_current_haystack_query_and_session_disk_roundtrip(self):
        with TemporaryDirectory() as temporary, web_runtime(Path(temporary)) as (module, generator):
            with patch("cli.chat_cli.ServerContext", return_value=module.server_context):
                cli = ChatCLI(config_path=str(Path(temporary) / "config.json"))
                with patch("builtins.print") as output:
                    cli._process_user_input("Controlled actual CLI question")
                    cli._save_session()
                printed = "\n".join(str(call.args[0]) for call in output.call_args_list)
                self.assertIn("Controlled Web fixture answer", printed)
                self.assertIn("Controlled table source", printed)
                self.assertEqual(len(generator.calls), 1)
                self.assertEqual([message.role for message in cli.session.messages[-2:]], ["user", "assistant"])
                session_id = cli.session_id
                cli.session_manager.delete_session(session_id)
                cli._load_session(session_id)
                self.assertEqual(cli.session.messages[-1].content, "Controlled Web fixture answer")
                with redirect_stdout(io.StringIO()):
                    cli._replay_session()

    def test_document_actual_async_pipeline_persists_content_and_exports(self):
        with TemporaryDirectory() as temporary:
            cli = self.document_cli(Path(temporary), ControlledTextProcessor())
            source = Path(temporary) / "input.txt"
            source.write_text("fixture input")
            cli._handle_upload(argparse.Namespace(file=str(source), directory=None))
            files = list(Path(cli.storage_base_path).glob("*/document.json"))
            self.assertEqual(len(files), 1)
            data = json.loads(files[0].read_text())
            self.assertEqual(data["status"], "completed")
            self.assertTrue(any(step.get("processor") == "cli-controlled-text" for step in data["processing_history"]))
            self.assertEqual(data["content"]["ocr"]["text"], "Real controlled processor content")
            destination = Path(temporary) / "actual.md"
            cli._handle_export(argparse.Namespace(id=data["document_id"], format="markdown", output=str(destination)))
            self.assertEqual(destination.read_text(), "Real controlled processor content")
            with patch("builtins.print") as output:
                cli._handle_export(
                    argparse.Namespace(
                        id=data["document_id"], format="pdf", output=str(destination.with_suffix(".pdf"))
                    )
                )
            self.assertIn("PDF导出尚未实现", "\n".join(str(call.args[0]) for call in output.call_args_list))
            self.assertFalse(destination.with_suffix(".pdf").exists())
            cli.pipeline._executor.shutdown()

    def test_document_failure_and_missing_processor_never_fabricate_completion(self):
        with TemporaryDirectory() as temporary:
            cli = self.document_cli(Path(temporary), ControlledTextProcessor(fail=True))
            cli._process_document(Document(Path(temporary) / "failed.txt"))
            cli.pipeline.clear_processors()
            cli._process_document(Document(Path(temporary) / "missing.txt"))
            data = [json.loads(path.read_text()) for path in Path(cli.storage_base_path).glob("*/document.json")]
            self.assertEqual([record["status"] for record in data], ["error", "error"])
            cli.pipeline._executor.shutdown()

    @staticmethod
    def document_cli(directory, processor):
        config = generate_default_config()
        config["storage"]["base_path"] = str(directory / "storage")
        config["storage"]["output_path"] = str(directory / "output")
        config["logging"]["file"] = str(directory / "cli.log")
        path = directory / "config.json"
        path.write_text(json.dumps(config))
        cli = DocumentCLI(config_path=str(path))
        cli.pipeline.add_processor(processor)
        cli.processors_loaded = True
        return cli
