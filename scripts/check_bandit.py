#!/usr/bin/env python3
"""Retain all findings, then enforce the repository's existing medium/medium Bandit gate."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Sequence


def validate_report(path: Path) -> Dict[str, Any]:
    report = json.loads(path.read_text())
    if report.get("errors"):
        raise RuntimeError(f"Bandit could not scan all requested source files: {report['errors']}")
    if not isinstance(report.get("results"), list):
        raise ValueError("Bandit report is missing its findings list")
    return report


def run_checks(paths: Sequence[str], output: Path, config: Path) -> int:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    with config.open("rb") as stream:
        declared = tomllib.load(stream)["tool"]["bandit"]
    if declared.get("severity") != "medium" or declared.get("confidence") != "medium":
        raise ValueError("Bandit runner requires the declared medium severity and confidence contract")
    full_path = output.with_name(f"{output.stem}-all{output.suffix}")
    command = [sys.executable, "-m", "bandit", "-c", str(config), "-r", *paths, "-f", "json"]
    complete = subprocess.run([*command, "-o", str(full_path)], check=False)
    if complete.returncode not in (0, 1):
        raise RuntimeError(f"Bandit scanner failed with status {complete.returncode}")
    full = validate_report(full_path)
    # These are the existing severity/confidence fields in pyproject.toml; keep low findings visible.
    gate = subprocess.run([*command, "-ll", "-ii", "-o", str(output)], check=False)
    threshold_report = validate_report(output)
    print(
        f"Bandit full report: {len(full['results'])} findings; medium/medium gate: {len(threshold_report['results'])}"
    )
    return gate.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--output", type=Path, default=Path("bandit-report.json"))
    parser.add_argument("--config", type=Path, default=Path("pyproject.toml"))
    options = parser.parse_args()
    return run_checks(options.paths, options.output, options.config)


if __name__ == "__main__":
    raise SystemExit(main())
