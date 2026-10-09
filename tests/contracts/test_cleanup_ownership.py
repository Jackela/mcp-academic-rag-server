"""Use our own two controlled child processes; never probe other user's processes."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.utils import cleanup


class CleanupOwnership(unittest.TestCase):
    def test_registered_child_released_unregistered_same_name_survives(self):
        command = [sys.executable, "-c", "import time; time.sleep(60)", "mcp_test_fixture"]
        owned = subprocess.Popen(command)
        other = subprocess.Popen(command)
        cleaner = cleanup.ResourceCleaner()
        try:
            cleaner.register_process(owned)
            cleaner.cleanup_processes()
            self.assertIsNotNone(owned.poll())
            self.assertIsNone(other.poll())
        finally:
            for process in [owned, other]:
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=5)

    def test_emergency_preserves_unregistered_file_and_avoids_process_scans(self):
        cleaner = cleanup.ResourceCleaner()
        with tempfile.TemporaryDirectory() as directory:
            registered = Path(directory) / "registered_test.txt"
            unregistered = Path(directory) / "unregistered_test.txt"
            registered.write_text("local fixture")
            unregistered.write_text("local fixture")
            cleaner.register_temp_file(str(registered))
            with (
                patch.object(cleanup, "_global_cleaner", cleaner),
                patch.object(cleanup.subprocess, "run", side_effect=AssertionError("No broad commands")),
            ):
                cleanup.emergency_cleanup()
            self.assertFalse(registered.exists())
            self.assertTrue(unregistered.exists())
