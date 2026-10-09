#!/usr/bin/env python3
"""Generate release history as data, including a repository's first release."""

import argparse
import subprocess
import uuid
from pathlib import Path


def generate_changelog(tag: str) -> str:
    # Resolve the current release before deciding whether there is an earlier tag.
    current = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}"], text=True
    ).strip()
    parents = subprocess.check_output(["git", "rev-list", "--parents", "-n", "1", current], text=True).split()[1:]
    prior_tags = (
        subprocess.check_output(["git", "tag", "--merged", parents[0]], text=True).splitlines() if parents else []
    )
    revision = current
    if prior_tags:
        previous = subprocess.check_output(["git", "describe", "--tags", "--abbrev=0", parents[0]], text=True).strip()
        previous_sha = subprocess.check_output(
            ["git", "rev-parse", "--verify", f"refs/tags/{previous}^{{commit}}"], text=True
        ).strip()
        revision = f"{previous_sha}..{current}"
    return subprocess.check_output(["git", "log", "--pretty=format:- %s (%h)", revision], text=True)


def write_github_environment(path: Path, changelog: str) -> None:
    delimiter = "CHANGELOG_" + uuid.uuid4().hex
    while delimiter in changelog.splitlines():
        delimiter = "CHANGELOG_" + uuid.uuid4().hex
    with path.open("a", encoding="utf-8") as output:
        output.write(f"CHANGELOG<<{delimiter}\n{changelog}\n{delimiter}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--github-env", type=Path, required=True)
    args = parser.parse_args()
    write_github_environment(args.github_env, generate_changelog(args.tag))


if __name__ == "__main__":
    main()
