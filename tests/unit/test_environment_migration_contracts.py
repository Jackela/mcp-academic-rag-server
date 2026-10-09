"""Real JSON environment and supported version-transition contracts."""

import json
import tempfile
import unittest
from pathlib import Path

from core.config_environment_manager import ConfigEnvironmentManager
from core.config_migration_tool import ConfigMigrationTool, MigrationStatus, ValueTransformRule


class EnvironmentContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.base = {
            "storage": {"base_path": "base", "output_path": "output"},
            "processors": {"local": {"enabled": True}},
            "llm": {"model": "local"},
        }
        self.write("config.json", self.base)
        self.write("config.development.json", {"environment": {"name": "development"}, "storage": {"base_path": "dev"}})
        self.manager = ConfigEnvironmentManager(str(self.directory))

    def write(self, name, data):
        path = self.directory / name
        path.write_text(json.dumps(data))
        return path

    def test_copy_and_create_do_not_mutate_source_or_callers(self):
        source = self.manager.get_environment_config("development")
        self.assertTrue(self.manager.copy_environment("development", "copy"))
        self.assertEqual(source["environment"]["name"], "development")
        supplied = {"storage": {"base_path": "supplied"}}
        self.assertTrue(self.manager.create_environment("custom", config=supplied))
        self.assertNotIn("environment", supplied)
        self.assertEqual(self.manager.get_environment_config("copy")["environment"]["name"], "copy")

    def test_invalid_json_switch_preserves_active_environment(self):
        self.write("config.broken.json", ["not an object"])
        manager = ConfigEnvironmentManager(str(self.directory))
        self.assertFalse(manager.set_environment("broken"))
        self.assertEqual(manager.current_environment, "development")
        self.assertTrue(manager.environments["development"].active)
        self.assertFalse(manager.environments["broken"].active)

    def test_export_cannot_claim_success_for_unreadable_config(self):
        self.write("config.broken.json", ["not an object"])
        manager = ConfigEnvironmentManager(str(self.directory))
        target = self.directory / "export.json"
        self.assertFalse(manager.export_environment("broken", str(target)))
        self.assertFalse(target.exists())

    def test_null_import_is_rejected_without_default_fabrication(self):
        imported = self.write(
            "import.json", {"environment_config": None, "export_info": {"environment_name": "imported"}}
        )
        self.assertIsNone(self.manager.import_environment(str(imported)))
        self.assertFalse((self.directory / "config.imported.json").exists())

    def test_deep_merge_update_switch_export_import_and_diff(self):
        self.assertEqual(
            self.manager.get_environment_config()["storage"], {"base_path": "dev", "output_path": "output"}
        )
        self.assertTrue(self.manager.create_environment("staging", config={"storage": {"base_path": "stage"}}))
        self.assertTrue(self.manager.update_environment_config("staging", {"storage": {"output_path": "stage-output"}}))
        self.assertTrue(self.manager.set_environment("staging"))
        self.assertEqual(
            self.manager.get_environment_config()["storage"], {"base_path": "stage", "output_path": "stage-output"}
        )
        valid, errors = self.manager.validate_environment("staging")
        self.assertTrue(valid, errors)
        diff = self.manager.get_environment_diff("development", "staging")
        self.assertIn("storage.base_path", {item["path"] for item in diff["modified"]})
        exported = self.directory / "export.json"
        self.assertTrue(self.manager.export_environment("staging", str(exported)))
        self.assertEqual(self.manager.import_environment(str(exported), "restored"), "restored")
        self.assertEqual(self.manager.get_environment_config("restored")["storage"]["base_path"], "stage")
        self.assertFalse(self.manager.delete_environment("staging"))
        self.assertFalse(self.manager.set_environment("missing"))
        self.assertEqual(self.manager.current_environment, "staging")


class MigrationContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.tool = ConfigMigrationTool()

    def legacy_config(self):
        return {
            "version": "1.0.0",
            "storage": {"data_path": "data", "result_path": "output"},
            "processors": {"local": True, "disabled": False},
            "generator": {"model_name": "local", "params": {"temperature": 5.0}},
            "document_store": {"embedding_dim": 128},
        }

    def write(self, data):
        path = self.directory / "config.json"
        path.write_text(json.dumps(data))
        return path

    def test_every_rule_on_transition_runs_including_temperature(self):
        path = self.write(self.legacy_config())
        result = self.tool.migrate_config_file(str(path))
        self.assertEqual(result.status, MigrationStatus.COMPLETED, result.errors)
        migrated = json.loads(path.read_text())
        self.assertEqual(migrated["llm"]["settings"]["temperature"], 2.0)
        self.assertEqual(migrated["storage"], {"base_path": "data", "output_path": "output"})
        self.assertEqual(migrated["processors"]["disabled"], {"enabled": False, "config": {}})
        self.assertEqual(
            migrated["vector_db"]["document_store"], {"embedding_dim": 128, "type": "memory", "similarity": "cosine"}
        )
        self.assertEqual(json.loads(Path(result.backup_path).read_text()), self.legacy_config())
        self.assertEqual(migrated["version"], "2.0.0")
        self.assertEqual(self.tool.validate_migration(self.legacy_config(), migrated), [])

    def test_unreachable_target_fails_without_rewriting_input(self):
        path = self.write(self.legacy_config())
        before = path.read_bytes()
        result = self.tool.migrate_config_file(str(path), "1.4.0", backup=False)
        self.assertEqual(result.status, MigrationStatus.FAILED)
        self.assertEqual(path.read_bytes(), before)

    def test_empty_processors_is_a_valid_detectable_legacy_input(self):
        self.assertEqual(self.tool.detect_config_version({"processors": {}}), "1.0.0")

    def test_file_transform_executes_once(self):
        calls = []

        def transform(value):
            calls.append(value)
            return value + 1

        self.tool.migration_rules = [ValueTransformRule("1.0.0", "1.1.0", "counter", transform)]
        path = self.write({"version": "1.0.0", "counter": 0})
        result = self.tool.migrate_config_file(str(path), "1.1.0", backup=False)
        self.assertEqual(result.status, MigrationStatus.COMPLETED)
        self.assertEqual(calls, [0])
        self.assertEqual(json.loads(path.read_text())["counter"], 1)

    def test_requested_backup_failure_aborts_migration(self):
        config = self.legacy_config()
        result = self.tool.migrate_config(config, backup_path=str(self.directory / "missing" / "config.json"))
        self.assertEqual(result.status, MigrationStatus.FAILED)
        self.assertTrue(result.errors)
        self.assertEqual(config, self.legacy_config())

    def test_failed_schema_transform_does_not_stamp_or_overwrite(self):
        path = self.write({"version": "1.1.0", "processors": ["invalid object shape"]})
        before = path.read_bytes()
        result = self.tool.migrate_config_file(str(path), "1.2.0", backup=False)
        self.assertEqual(result.status, MigrationStatus.FAILED)
        self.assertTrue(result.errors)
        self.assertEqual(path.read_bytes(), before)

    def test_failed_value_transform_preserves_file_and_reports_error(self):
        def fail(value):
            raise ValueError("Controlled invalid value")

        self.tool.migration_rules = [ValueTransformRule("1.0.0", "1.1.0", "counter", fail)]
        path = self.write({"version": "1.0.0", "counter": 0})
        before = path.read_bytes()
        result = self.tool.migrate_config_file(str(path), "1.1.0", backup=False)
        self.assertEqual(result.status, MigrationStatus.FAILED)
        self.assertTrue(result.errors)
        self.assertEqual(path.read_bytes(), before)

    def test_invalid_json_version_and_same_version_states(self):
        path = self.write(["not an object"])
        self.assertEqual(self.tool.migrate_config_file(str(path)).status, MigrationStatus.FAILED)
        self.assertEqual(self.tool.migrate_config({"version": 7}).status, MigrationStatus.FAILED)
        self.assertEqual(self.tool.migrate_config({"version": "2.0.0"}).status, MigrationStatus.SKIPPED)
        self.assertEqual(self.tool.migrate_config({"version": "2.0.0"}, "1.0.0").status, MigrationStatus.FAILED)

    def test_all_accepted_intermediate_targets_and_sorted_versions(self):
        self.assertEqual(self.tool.get_available_versions(), ["1.0.0", "1.1.0", "1.2.0", "1.3.0", "2.0.0"])
        for target in ["1.1.0", "1.2.0", "1.3.0", "2.0.0"]:
            with self.subTest(target=target):
                result = self.tool.migrate_config(self.legacy_config(), target)
                self.assertEqual(result.status, MigrationStatus.COMPLETED, result.errors)
                self.assertEqual(result.to_version, target)
