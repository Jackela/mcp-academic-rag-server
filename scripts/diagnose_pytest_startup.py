"""Compare native pytest startup failures without replacing the full test gate.

Use only collection of the existing configuration tests. The unmodified baseline
must succeed for this diagnostic to return success; variants identify a trigger.
"""

import os
import subprocess
import sys


def main() -> None:
    common = [
        sys.executable,
        "-X",
        "faulthandler",
        "-m",
        "pytest",
        "tests/unit/test_config_manager.py",
        "--collect-only",
        "-o",
        "addopts=",
        "--cov=core",
        "-q",
    ]
    variants = {
        "baseline": [],
        "without_benchmark": ["-p", "no:benchmark"],
        "without_html": ["-p", "no:html", "-p", "no:html_fixtures", "-p", "no:metadata"],
        "without_coverage": ["--no-cov"],
        "plain_assertions": ["--assert=plain"],
        "default_warnings": ["-W", "default"],
    }
    codes = {}
    for name, flags in variants.items():
        try:
            result = subprocess.run(
                common + flags, capture_output=True, text=True, errors="replace", timeout=40, env=os.environ.copy()
            )
            codes[name] = result.returncode
            print(f"STARTUP {name}: exit={result.returncode}", flush=True)
            if result.returncode:
                print(result.stderr[-2500:], flush=True)
        except subprocess.TimeoutExpired:
            codes[name] = 124
            print(f"STARTUP {name}: timeout=40s", flush=True)
    print(f"STARTUP results: {codes}", flush=True)
    if codes["baseline"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
