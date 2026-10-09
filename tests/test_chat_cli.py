"""CLI command routing and session files use real local data without a model."""

import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cli.chat_cli import ChatCLI
from core.config_validator import generate_default_config


class TestChatCLI(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        config = generate_default_config()
        config["storage"]["base_path"] = str(self.directory / "data")
        config["storage"]["output_path"] = str(self.directory / "output")
        config["logging"]["file"] = str(self.directory / "application.log")
        self.config_path = self.directory / "config.json"
        self.config_path.write_text(json.dumps(config))
        previous = Path.cwd()
        try:
            os.chdir(self.directory)
            with patch.object(ChatCLI, "_initialize_rag_pipeline"):
                self.cli = ChatCLI(config_path=str(self.config_path), verbose=True)
        finally:
            os.chdir(previous)

    def tearDown(self):
        self.cli.context.cleanup()
        self.temporary.cleanup()

    def arguments(self, **changes):
        values = dict(config=str(self.config_path), verbose=False, list=False, export=None, session=None, replay=None)
        return argparse.Namespace(**(values | changes))

    def test_init(self):
        self.assertEqual(self.cli.config_path, str(self.config_path))
        self.assertTrue(self.cli.verbose)
        self.assertEqual(self.cli.sessions_dir, str(self.directory / "data" / "sessions"))

    def test_run_list_command(self):
        with (
            patch.object(self.cli, "_parse_args", return_value=self.arguments(list=True)),
            patch.object(self.cli, "_list_sessions") as listing,
            patch.object(self.cli, "_start_interactive_chat") as interactive,
        ):
            self.cli.run()
        listing.assert_called_once()
        interactive.assert_not_called()

    def test_run_export_command(self):
        with (
            patch.object(self.cli, "_parse_args", return_value=self.arguments(export="owned")),
            patch.object(self.cli, "_export_session") as export,
            patch.object(self.cli, "_start_interactive_chat") as interactive,
        ):
            self.cli.run()
        export.assert_called_once_with("owned")
        interactive.assert_not_called()

    def test_run_replay_command(self):
        with (
            patch.object(self.cli, "_parse_args", return_value=self.arguments(replay="owned")),
            patch.object(self.cli, "_load_session") as load,
            patch.object(self.cli, "_replay_session") as replay,
        ):
            self.cli.run()
        load.assert_called_once_with("owned")
        replay.assert_called_once()

    def test_list_sessions(self):
        self.cli.session.add_message("user", "Local written question")
        self.cli._save_session()
        with patch("builtins.print") as output:
            self.cli._list_sessions()
        self.assertTrue(any(self.cli.session_id in str(call.args) for call in output.call_args_list))

    def test_load_session(self):
        self.cli.session.add_message("user", "Local persisted question")
        self.cli._save_session()
        identity = self.cli.session_id
        self.cli.session_manager.delete_session(identity)
        self.cli._load_session(identity)
        self.assertEqual(self.cli.session.messages[-1].content, "Local persisted question")

    def test_format_timestamp(self):
        self.assertEqual(self.cli._format_timestamp("invalid"), "invalid")
        self.assertEqual(self.cli._format_timestamp("未知"), "未知")
        self.assertEqual(len(self.cli._format_timestamp(1672531200)), 16)


if __name__ == "__main__":
    unittest.main()
