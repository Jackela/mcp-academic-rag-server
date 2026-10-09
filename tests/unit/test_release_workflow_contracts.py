"""Actual release conditions and Git history retain data without publishing anything."""

import importlib.util
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("release_changelog", ROOT / "scripts/release_changelog.py")
assert SPEC and SPEC.loader
changelog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(changelog)


def _condition_matches(expression, context):
    # This workflow deliberately uses only explicit equality clauses and AND.
    for clause in expression.split("&&"):
        match = re.fullmatch(
            r"\s*(github\.event_name|github\.event\.action|inputs\.publish_target) == '([^']+)'\s*", clause
        )
        assert match, f"Unsupported publication condition: {clause}"
        if context.get(match[1]) != match[2]:
            return False
    return True


@pytest.mark.parametrize(
    "event,target,action,expected",
    [
        ("push", "production", None, set()),
        ("pull_request", "staging", None, set()),
        ("workflow_dispatch", "none", None, set()),
        ("workflow_dispatch", "staging", None, {"deploy-staging"}),
        ("workflow_dispatch", "production", None, {"deploy-production"}),
        ("release", None, "published", {"release"}),
        ("release", None, "created", set()),
    ],
)
def test_actual_publication_conditions(event, target, action, expected):
    workflow = yaml.load((ROOT / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader)
    selection = workflow["on"]["workflow_dispatch"]["inputs"]["publish_target"]
    assert selection["default"] == "none"
    assert selection["options"] == ["none", "staging", "production"]
    context = {"github.event_name": event, "inputs.publish_target": target, "github.event.action": action}
    jobs = workflow["jobs"]
    actual = {
        name
        for name in ("deploy-staging", "deploy-production", "release")
        if _condition_matches(jobs[name]["if"], context)
    }
    assert actual == expected
    for name in ("deploy-staging", "deploy-production"):
        assert "url" not in jobs[name]["environment"]
        for step in jobs[name]["steps"]:
            assert "github.run_number" not in str(step)
    release = jobs["release"]["steps"]
    assert release[0]["with"]["fetch-depth"] == "0"
    version = next(step for step in release if step.get("id") == "version")
    assert "VERSION=$GITHUB_REF_NAME" in version["run"]
    notes = next(step for step in release if step["name"] == "Update release notes")
    assert "process.env.CHANGELOG" in notes["with"]["script"]
    assert "steps.changelog" not in notes["with"]["script"]


def _git(directory, *args):
    return subprocess.check_output(["git", "-C", str(directory), *args], text=True).strip()


@pytest.mark.parametrize("job", ["docker-build", "security-scan", "deploy-staging", "deploy-production", "release"])
def test_container_jobs_use_real_lowercase_repository(job, tmp_path):
    workflow = yaml.load((ROOT / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader)
    steps = workflow["jobs"][job]["steps"]
    normalizers = [step for step in steps if step["name"] == "Normalize container repository name"]
    assert len(normalizers) == 1
    env_file = tmp_path / "github-env"
    env = dict(os.environ, GITHUB_REPOSITORY="Jackela/MCP-Academic-RAG-Server", GITHUB_ENV=env_file.as_posix())
    bash = "bash"
    if os.name == "nt":
        # Windows PATH may resolve bash to the WSL shim instead of Git Bash.
        git_exec_path = Path(subprocess.check_output(["git", "--exec-path"], text=True).strip())
        bash = str(git_exec_path.parents[2] / "bin" / "bash.exe")
        assert Path(bash).is_file(), f"Git Bash is missing from the current Git installation: {bash}"
    subprocess.run([bash, "-e", "-c", normalizers[0]["run"]], env=env, check=True)
    assert env_file.read_text() == "IMAGE_NAME=jackela/mcp-academic-rag-server\n"
    first_image_use = next(index for index, step in enumerate(steps) if "env.IMAGE_NAME" in str(step))
    assert steps.index(normalizers[0]) < first_image_use


def test_first_and_subsequent_release_keep_untrusted_text_as_data(tmp_path, monkeypatch):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Release Fixture")
    _git(tmp_path, "config", "user.email", "fixture@example.invalid")
    subject = "Literal `${process.exit(1)}` and $(touch SHOULD_NOT_EXIST)"
    _git(tmp_path, "commit", "--allow-empty", "-m", subject)
    _git(tmp_path, "tag", "v1.0.0")
    monkeypatch.chdir(tmp_path)
    first = changelog.generate_changelog("v1.0.0")
    assert subject in first
    env_file = tmp_path / "github-env"
    changelog.write_github_environment(env_file, first + "\nEOF\nCHANGELOG<<EOF")
    lines = env_file.read_text().splitlines()
    delimiter = lines[0].split("<<", 1)[1]
    assert lines[-1] == delimiter and delimiter not in lines[1:-1]
    assert not (tmp_path / "SHOULD_NOT_EXIST").exists()
    _git(tmp_path, "commit", "--allow-empty", "-m", "Second release only")
    _git(tmp_path, "tag", "v1.0.1")
    second = changelog.generate_changelog("v1.0.1")
    assert "Second release only" in second and subject not in second
    with pytest.raises(subprocess.CalledProcessError):
        changelog.generate_changelog("missing-tag")
