"""Concrete configuration and processing failures must remain visible to callers."""

import asyncio
import json
import subprocess
import sys

import pytest

from core.config_center import ConfigCenter
from core.config_manager import ConfigManager
from core.config_validator import generate_default_config
from core.config_version_manager import ConfigVersionManager
from core.pipeline import Pipeline
from models.document import Document
from models.process_result import ProcessResult
from processors.base_processor import BaseProcessor


class ControlledProcessor(BaseProcessor):
    def __init__(self):
        super().__init__("controlled")

    def supports_file_type(self, file_type):
        return file_type == ".pdf"

    def process(self, document):
        document.store_content("controlled", "Processed source")
        return ProcessResult.success_result("Processed")


def make_config(directory):
    config = generate_default_config()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config))
    return config


def test_config_set_value_completes_without_recursive_lock_deadlock(tmp_path):
    make_config(tmp_path)
    script = (
        "from core.config_center import ConfigCenter; "
        f"c=ConfigCenter({str(tmp_path)!r}, watch_changes=False); "
        "assert c.set_value('logging.level','DEBUG',persist=False); "
        "assert c.get_value('logging.level')=='DEBUG'; c.close()"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=3)
    assert result.returncode == 0, result.stderr


def test_corrupt_config_does_not_claim_validation_success(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("not json")
    manager = ConfigManager(str(path))
    assert manager.is_config_valid() is False


def test_corrupt_hot_reload_preserves_last_valid_configuration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "owned-config"
    config = make_config(config_dir)
    config["logging"]["level"] = "DEBUG"
    (config_dir / "config.json").write_text(json.dumps(config))
    center = ConfigCenter(str(config_dir), watch_changes=False)
    before = center.get_config()
    (config_dir / "config.json").write_text("not json")
    center._reload_config()
    assert center.get_config() == before
    center.close()


def test_change_history_refers_to_the_actual_saved_version(tmp_path):
    path = tmp_path / "configuration.json"
    path.write_text('{"value":1}')
    manager = ConfigVersionManager(str(path))
    version, changes = manager.save_config_with_version({"value": 2})
    assert changes
    assert {change.version for change in changes} == {version}
    assert {item["version"] for item in manager.get_change_history()} == {version}
    assert manager.get_version(version).config_data == {"value": 2}


@pytest.mark.asyncio
async def test_pipeline_rejects_zero_concurrency_and_a_missing_start(tmp_path):
    pipeline = Pipeline()
    pipeline.add_processor(ControlledProcessor())
    document = Document(tmp_path / "input.pdf")
    try:
        with pytest.raises(ValueError):
            await asyncio.wait_for(pipeline.process_documents([document], max_concurrent=0), timeout=0.1)
        for result in [
            pipeline.process_document_sync(document, "missing"),
            await pipeline.process_document(document, "missing"),
        ]:
            assert result.success is False
    finally:
        pipeline._executor.shutdown()


@pytest.mark.asyncio
async def test_pipeline_does_not_mark_an_unprocessed_file_complete(tmp_path):
    pipeline = Pipeline()
    pipeline.add_processor(ControlledProcessor())
    try:
        for document in [Document(tmp_path / "unsupported.txt"), Document(tmp_path / "unsupported2.txt")]:
            result = await pipeline.process_document(document)
            assert result.success is False
            assert document.status != "completed"
        supported = Document(tmp_path / "supported.pdf")
        assert (await pipeline.process_document(supported)).success
        assert supported.get_content("controlled") == "Processed source"
    finally:
        pipeline._executor.shutdown()


def test_effective_environment_config_is_the_injected_server_config(tmp_path):
    from core.server_context import ServerContext

    make_config(tmp_path)
    (tmp_path / "config.production.json").write_text(json.dumps({"logging": {"level": "ERROR"}}))
    center = ConfigCenter(str(tmp_path), environment="production", watch_changes=False)
    context = ServerContext(config_manager=center.config_manager)
    assert context.config_manager is center.config_manager
    assert context.config_manager.get_value("logging.level") == "ERROR"
    assert center.set_value("logging.level", "DEBUG", persist=False)
    assert context.config_manager.get_value("logging.level") == "DEBUG"
    snapshot = center.get_config()
    snapshot["logging"]["level"] = "INVALID"
    assert center.get_value("logging.level") == "DEBUG"
    assert not center.set_value("logging.level", "INVALID", persist=False)
    assert context.config_manager.get_value("logging.level") == "DEBUG"
    center.close()


def test_failed_persistence_does_not_commit_or_notify(tmp_path, monkeypatch):
    make_config(tmp_path)
    center = ConfigCenter(str(tmp_path), watch_changes=False)
    changes = []
    center.add_change_listener(changes.append)
    path = tmp_path / "config.json"
    before = center.get_config()
    # An actual directory at the destination makes opening for writing fail.
    path.unlink()
    path.mkdir()
    assert not center.set_value("logging.level", "DEBUG")
    assert center.get_config() == before
    assert center.config_manager.get_config() == before
    assert changes == []
    center.close()


def test_corrupt_configuration_cannot_initialize_a_server(tmp_path):
    from core.server_context import ServerContext

    path = tmp_path / "config.json"
    path.write_text("not json")
    manager = ConfigManager(str(path))
    report = manager.get_validation_report()
    assert report["is_valid"] is False
    assert any("could not be loaded" in error for error in report["errors"])
    context = ServerContext(config_manager=manager)
    with pytest.raises(ValueError, match="configuration is invalid"):
        context.initialize()
    assert not context.is_initialized
    assert context.document_pipeline is None


@pytest.mark.asyncio
async def test_real_workflow_strategies_respect_milestone_options(tmp_path):
    from core.workflow_generator import WorkflowGenerator
    from core.workflow_models import WorkflowOptions, WorkflowStrategy

    path = tmp_path / "PRD.md"
    path.write_text("# Local tool\n## 需求\n- 实现文件检索接口功能")
    generator = WorkflowGenerator()
    for strategy in WorkflowStrategy:
        workflow = await generator.generate_workflow(str(path), WorkflowOptions(strategy=strategy))
        assert workflow.phases
        assert all(phase.milestones for phase in workflow.phases)
        assert workflow.get_total_effort() > 0
        disabled = await generator.generate_workflow(
            str(path), WorkflowOptions(strategy=strategy, enable_milestones=False)
        )
        assert all(not phase.milestones for phase in disabled.phases)
    with pytest.raises(NotImplementedError, match="not implemented"):
        await generator.generate_workflow(str(path), WorkflowOptions(enable_context7=True))


def test_rejected_candidate_keeps_the_previous_config_valid(tmp_path):
    make_config(tmp_path)
    manager = ConfigManager(str(tmp_path / "config.json"))
    before = manager.get_config()
    candidate = manager.get_config()
    candidate["logging"]["level"] = "INVALID"
    assert not manager.apply_config(candidate)
    assert manager.get_config() == before
    assert manager.is_config_valid()
    assert manager.set_value("logging.level", "INVALID")
    assert not manager.is_config_valid()
    assert manager.get_validation_report()["is_valid"] is False
