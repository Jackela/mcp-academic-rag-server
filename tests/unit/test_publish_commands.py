"""Publishing helpers preserve literal argv and fail closed before any upload."""

import importlib.util
import subprocess
from pathlib import Path
from unittest.mock import patch


def load_script():
    spec = importlib.util.spec_from_file_location(
        "publish_script_contract", Path(__file__).parents[2] / "scripts/publish_to_pypi.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_command_preserves_metacharacters_as_literal_argument():
    module = load_script()
    args = ["python", "-m", "twine", "check", "literal;$(not-a-command).whl"]
    with patch.object(module.subprocess, "run", return_value=subprocess.CompletedProcess(args, 0, "", "")) as run:
        assert module.run_command(args, "controlled check") == ""
    assert run.call_args.args == (args,)
    assert not run.call_args.kwargs.get("shell", False)


def test_failed_artifact_check_stops_before_prompt_or_upload(tmp_path, monkeypatch):
    module = load_script()
    monkeypatch.setattr(module, "__file__", str(tmp_path / "scripts" / "publish_to_pypi.py"))
    original = Path.cwd()

    def controlled_run(args, description):
        if args[-1] == "build":
            (tmp_path / "dist").mkdir()
            (tmp_path / "dist" / "controlled.whl").write_text("fixture")
            return ""
        assert "check" in args
        return None

    try:
        with (
            patch.object(module, "run_command", side_effect=controlled_run) as run,
            patch("builtins.input", side_effect=AssertionError("No prompt or upload after validation failure")),
        ):
            assert module.main() is False
        assert len(run.call_args_list) == 2
        assert all("upload" not in call.args[0] for call in run.call_args_list)
    finally:
        monkeypatch.chdir(original)
