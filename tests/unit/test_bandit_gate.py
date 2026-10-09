"""Actual scanner findings distinguish medium/high gate failures from retained low information."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "source, expected_gate, minimum_full",
    [
        ("value = 42\n", 0, 0),
        ("import subprocess\nsubprocess.run(['python', '--version'])\n", 0, 1),
        ("import hashlib\nhashlib.md5(b'controlled')\n", 1, 1),
    ],
)
def test_actual_bandit_gate(source, expected_gate, minimum_full, tmp_path):
    root = Path(__file__).parents[2]
    candidate = tmp_path / "controlled.py"
    candidate.write_text(source)
    report = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/check_bandit.py"),
            str(candidate),
            "--output",
            str(report),
            "--config",
            str(root / "pyproject.toml"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == expected_gate, result.stdout + result.stderr
    full = json.loads((tmp_path / "report-all.json").read_text())
    assert len(full["results"]) >= minimum_full
    assert full["errors"] == []
    gate = json.loads(report.read_text())
    assert bool(gate["results"]) == bool(expected_gate)
