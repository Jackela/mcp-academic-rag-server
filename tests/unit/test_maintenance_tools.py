"""Historical tools retain actual outputs and report actual filesystem failures."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from core.config_validator import generate_default_config

ROOT = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("valid", [True, False])
def test_configuration_cli_checks_real_files_from_another_directory(tmp_path, valid):
    config = generate_default_config()
    config["logging"]["level"] = "DEBUG" if valid else "INVALID"
    path = tmp_path / "configuration.json"
    path.write_text(json.dumps(config))
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/validate_config.py"), "--config", str(path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == (0 if valid else 1), result.stdout + result.stderr
    assert ("配置验证通过" if valid else "配置验证失败") in result.stdout


def test_default_configuration_can_be_written_to_a_plain_filename(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/validate_config.py"), "--generate-default", "--output", "default.json"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((tmp_path / "default.json").read_text()) == generate_default_config()
    (tmp_path / "destination").mkdir()
    failed = subprocess.run(
        [sys.executable, str(ROOT / "tools/validate_config.py"), "--generate-default", "--output", "destination"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert failed.returncode == 1
    assert "生成默认配置失败" in failed.stdout


def test_health_checker_uses_the_actual_pillow_import_name():
    import PIL

    from tools.health_check import HealthChecker

    checker = HealthChecker()
    assert checker._package_version("pillow") == PIL.__version__


def test_example_renders_actual_graph_information(capsys):
    module = load_module("graph_example_contract", ROOT / "examples/knowledge_graph_example.py")
    module.analyze_knowledge_graph(
        {
            "statistics": {"total_entities": 1, "total_relations": 1, "entity_types": {"METHOD": 1}},
            "knowledge_graph": {
                "entities": {"Fixture method": {"type": "METHOD", "confidence": 0.8, "frequency": 2}},
                "relations": [
                    {"subject": "Fixture method", "predicate": "uses", "object": "Source", "confidence": 0.7}
                ],
                "concepts": [{"name": "Fixture concept", "confidence": 0.9}],
            },
        }
    )
    output = capsys.readouterr().out
    assert "Fixture method (METHOD)" in output
    assert "Fixture method --[uses]--> Source" in output
    assert "Fixture concept" in output
