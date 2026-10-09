"""Document CLI routing and listing use real serialized local documents."""

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cli.document_cli import DocumentCLI
from core.config_validator import generate_default_config
from models.document import Document


class TestDocumentCLI(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        config = generate_default_config()
        config["storage"]["base_path"] = str(self.directory / "data")
        config["storage"]["output_path"] = str(self.directory / "output")
        config["logging"]["file"] = str(self.directory / "application.log")
        self.config_path = self.directory / "config.json"
        self.config_path.write_text(json.dumps(config))
        self.cli = DocumentCLI(config_path=str(self.config_path), verbose=True)
        self.document = Document(self.directory / "source.txt")
        self.document.add_tag("owned")
        target = Path(self.cli.storage_base_path) / self.document.document_id
        target.mkdir()
        (target / "document.json").write_text(
            json.dumps(self.document.to_dict(), default=self.cli._serialize_document_value)
        )

    def tearDown(self):
        self.cli.pipeline._executor.shutdown()
        self.temporary.cleanup()

    def test_init(self):
        self.assertEqual(self.cli.config_path, str(self.config_path))
        self.assertTrue(self.cli.verbose)
        self.assertEqual(self.cli.storage_base_path, str(self.directory / "data"))
        self.assertEqual(self.cli.storage_output_path, str(self.directory / "output"))

    def test_run_upload_command(self):
        args = argparse.Namespace(command="upload", config=str(self.config_path), verbose=False)
        with (
            patch.object(self.cli, "_parse_args", return_value=args),
            patch.object(self.cli, "_handle_upload") as upload,
        ):
            self.cli.run()
        upload.assert_called_once_with(args)

    def test_run_list_command(self):
        args = argparse.Namespace(command="list", config=str(self.config_path), verbose=False)
        with (
            patch.object(self.cli, "_parse_args", return_value=args),
            patch.object(self.cli, "_handle_list") as listing,
        ):
            self.cli.run()
        listing.assert_called_once_with(args)

    def test_handle_info(self):
        with patch("builtins.print") as output:
            self.cli._handle_info(argparse.Namespace(id=self.document.document_id))
        output.assert_any_call(f"文档ID: {self.document.document_id}")
        output.assert_any_call("文件名: source.txt")
        output.assert_any_call("状态: new")
        output.assert_any_call("标签: ['owned']")

    def test_handle_list(self):
        with patch("builtins.print") as output:
            self.cli._handle_list(argparse.Namespace(status="new", tag="owned", format="json"))
        listed = json.loads(output.call_args.args[0])
        self.assertEqual([record["document_id"] for record in listed], [self.document.document_id])
        with patch("builtins.print") as output:
            self.cli._handle_list(argparse.Namespace(status="completed", tag="owned", format="json"))
        self.assertEqual(json.loads(output.call_args.args[0]), [])


if __name__ == "__main__":
    unittest.main()
