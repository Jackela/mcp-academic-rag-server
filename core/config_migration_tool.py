"""
配置迁移工具

提供配置格式升级、数据迁移、兼容性转换和自动迁移功能。
支持多版本配置格式间的转换和向后兼容性处理。
"""

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _config_object(value: object) -> Dict[str, Any]:
    """Require an object at the JSON/config transformation boundary."""
    if not isinstance(value, dict):
        raise ValueError("配置必须是 JSON 对象")
    return value


def _json_copy(config: Dict[str, Any]) -> Dict[str, Any]:
    return _config_object(json.loads(json.dumps(config)))


class MigrationStatus(Enum):
    """迁移状态"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class MigrationResult:
    """迁移结果"""

    status: MigrationStatus
    from_version: str
    to_version: str
    messages: List[str]
    warnings: List[str]
    errors: List[str]
    migrated_paths: List[str]
    backup_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "messages": self.messages,
            "warnings": self.warnings,
            "errors": self.errors,
            "migrated_paths": self.migrated_paths,
            "backup_path": self.backup_path,
        }


class MigrationRule:
    """迁移规则基类"""

    def __init__(self, from_version: str, to_version: str, name: str, description: str):
        self.from_version = from_version
        self.to_version = to_version
        self.name = name
        self.description = description

    def can_migrate(self, config: Dict[str, Any]) -> bool:
        """检查是否可以迁移"""
        raise NotImplementedError

    def migrate(self, config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str]]:
        """执行迁移，返回 (新配置, 消息, 警告)"""
        raise NotImplementedError


class PathRenameRule(MigrationRule):
    """路径重命名规则"""

    def __init__(self, from_version: str, to_version: str, path_mappings: Dict[str, str]):
        super().__init__(from_version, to_version, "path_rename", f"重命名配置路径: {len(path_mappings)} 个映射")
        self.path_mappings = path_mappings

    def can_migrate(self, config: Dict[str, Any]) -> bool:
        """检查是否有需要重命名的路径"""
        for old_path in self.path_mappings.keys():
            if self._path_exists(config, old_path):
                return True
        return False

    def migrate(self, config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str]]:
        """执行路径重命名"""
        new_config = _json_copy(config)  # 深拷贝
        messages = []
        warnings: List[str] = []

        for old_path, new_path in self.path_mappings.items():
            if self._path_exists(new_config, old_path):
                value = self._get_path_value(new_config, old_path)
                self._set_path_value(new_config, new_path, value)
                self._delete_path(new_config, old_path)
                messages.append(f"重命名路径: {old_path} → {new_path}")

                # 检查冲突
                if old_path != new_path and self._path_exists(config, new_path):
                    warnings.append(f"路径冲突: {new_path} 已存在，原值被覆盖")

        return new_config, messages, warnings

    def _path_exists(self, config: Dict[str, Any], path: str) -> bool:
        """检查路径是否存在"""
        try:
            self._get_path_value(config, path)
            return True
        except (KeyError, TypeError):
            return False

    def _get_path_value(self, config: Dict[str, Any], path: str) -> Any:
        """获取路径值"""
        keys = path.split(".")
        current = config

        for key in keys:
            current = current[key]

        return current

    def _set_path_value(self, config: Dict[str, Any], path: str, value: Any) -> None:
        """设置路径值"""
        keys = path.split(".")
        current = config

        # 创建中间路径
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _delete_path(self, config: Dict[str, Any], path: str) -> None:
        """删除路径"""
        keys = path.split(".")
        current = config

        # 导航到父级
        for key in keys[:-1]:
            current = current[key]

        # 删除最后一个键
        if keys[-1] in current:
            del current[keys[-1]]


class ValueTransformRule(MigrationRule):
    """值转换规则"""

    def __init__(
        self,
        from_version: str,
        to_version: str,
        path: str,
        transform_func: Callable[[Any], Any],
        description: Optional[str] = None,
    ):
        super().__init__(from_version, to_version, "value_transform", description or f"转换值: {path}")
        self.path = path
        self.transform_func = transform_func

    def can_migrate(self, config: Dict[str, Any]) -> bool:
        """检查是否需要转换"""
        return self._path_exists(config, self.path)

    def migrate(self, config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str]]:
        """执行值转换"""
        new_config = _json_copy(config)
        messages = []
        warnings: List[str] = []

        if self._path_exists(new_config, self.path):
            try:
                old_value = self._get_path_value(new_config, self.path)
                new_value = self.transform_func(old_value)
                self._set_path_value(new_config, self.path, new_value)
                messages.append(f"转换值 {self.path}: {old_value} → {new_value}")
            except Exception as e:
                raise ValueError(f"转换值失败 {self.path}: {e}") from e

        return new_config, messages, warnings

    def _path_exists(self, config: Dict[str, Any], path: str) -> bool:
        """检查路径是否存在"""
        try:
            self._get_path_value(config, path)
            return True
        except (KeyError, TypeError):
            return False

    def _get_path_value(self, config: Dict[str, Any], path: str) -> Any:
        """获取路径值"""
        keys = path.split(".")
        current = config

        for key in keys:
            current = current[key]

        return current

    def _set_path_value(self, config: Dict[str, Any], path: str, value: Any) -> None:
        """设置路径值"""
        keys = path.split(".")
        current = config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value


class StructureRule(MigrationRule):
    """结构转换规则"""

    def __init__(
        self,
        from_version: str,
        to_version: str,
        structure_func: Callable[[Dict[str, Any]], Dict[str, Any]],
        description: Optional[str] = None,
    ):
        super().__init__(from_version, to_version, "structure_transform", description or "转换配置结构")
        self.structure_func = structure_func

    def can_migrate(self, config: Dict[str, Any]) -> bool:
        """总是可以执行结构转换"""
        return True

    def migrate(self, config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str]]:
        """执行结构转换"""
        messages = []
        warnings: List[str] = []

        try:
            new_config = self.structure_func(config)
            messages.append("执行配置结构转换")
            return new_config, messages, warnings
        except Exception as e:
            raise ValueError(f"结构转换失败: {e}") from e


class ConfigMigrationTool:
    """配置迁移工具"""

    def __init__(self) -> None:
        self.migration_rules: List[MigrationRule] = []
        self.version_pattern = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")

        # 注册默认迁移规则
        self._register_default_rules()

    def _register_default_rules(self) -> None:
        """注册默认迁移规则"""
        # v1.0.0 → v1.1.0: 重命名存储配置
        self.add_rule(
            PathRenameRule(
                "1.0.0",
                "1.1.0",
                {"storage.data_path": "storage.base_path", "storage.result_path": "storage.output_path"},
            )
        )

        # v1.1.0 → v1.2.0: 处理器配置结构化
        self.add_rule(StructureRule("1.1.0", "1.2.0", self._restructure_processors, "重构处理器配置格式"))

        # v1.2.0 → v1.3.0: LLM配置标准化
        self.add_rule(
            PathRenameRule(
                "1.2.0", "1.3.0", {"generator": "llm", "llm.model_name": "llm.model", "llm.params": "llm.settings"}
            )
        )

        # 温度值范围调整
        def normalize_temperature(temp: Any) -> float:
            if isinstance(temp, (int, float)):
                return max(0.0, min(2.0, float(temp)))
            return 1.0

        self.add_rule(
            ValueTransformRule(
                "1.2.0", "1.3.0", "llm.settings.temperature", normalize_temperature, "标准化温度参数范围"
            )
        )

        # v1.3.0 → v2.0.0: 向量数据库配置重构
        self.add_rule(StructureRule("1.3.0", "2.0.0", self._restructure_vector_db, "重构向量数据库配置"))

    @staticmethod
    def _restructure_processors(config: Dict[str, Any]) -> Dict[str, Any]:
        new_config = _json_copy(config)

        if "processors" in new_config:
            for name, proc_config in new_config["processors"].items():
                if isinstance(proc_config, bool):
                    # 从布尔值转换为对象
                    new_config["processors"][name] = {"enabled": proc_config, "config": {}}
                elif isinstance(proc_config, dict) and "enabled" not in proc_config:
                    # 添加缺失的enabled字段
                    proc_config["enabled"] = True

        return new_config

    @staticmethod
    def _restructure_vector_db(config: Dict[str, Any]) -> Dict[str, Any]:
        new_config = _json_copy(config)

        # 迁移旧的document_store配置
        if "document_store" in new_config and "vector_db" not in new_config:
            new_config["vector_db"] = {"document_store": new_config["document_store"]}
            del new_config["document_store"]

        # 添加新的向量存储配置
        if "vector_db" in new_config:
            vector_config = new_config["vector_db"]
            if "document_store" in vector_config:
                ds_config = vector_config["document_store"]

                # 设置默认值
                if "type" not in ds_config:
                    ds_config["type"] = "memory"
                if "embedding_dim" not in ds_config:
                    ds_config["embedding_dim"] = 1536
                if "similarity" not in ds_config:
                    ds_config["similarity"] = "cosine"

        return new_config

    def add_rule(self, rule: MigrationRule) -> None:
        """添加迁移规则"""
        self.migration_rules.append(rule)
        logger.debug(f"添加迁移规则: {rule.name} ({rule.from_version} → {rule.to_version})")

    def detect_config_version(self, config: Dict[str, Any]) -> str:
        """检测配置版本"""
        # 检查显式版本标记
        if "version" in config:
            version = config["version"]
            if not isinstance(version, str) or not self.version_pattern.fullmatch(version):
                raise ValueError("配置版本必须是 major.minor.patch 字符串")
            return version

        # 基于配置结构推断版本
        if "vector_db" in config:
            return "2.0.0"
        elif "llm" in config:
            return "1.3.0"
        elif (
            isinstance(config.get("processors"), dict)
            and config["processors"]
            and isinstance(next(iter(config["processors"].values())), dict)
        ):
            return "1.2.0"
        elif "storage" in config and "base_path" in config["storage"]:
            return "1.1.0"
        else:
            return "1.0.0"

    def get_migration_path(self, from_version: str, to_version: str) -> List[MigrationRule]:
        """Return all rules along an exact supported ascending transition path."""
        applicable_rules: List[MigrationRule] = []
        current_version = from_version
        while current_version != to_version:
            next_rule = next(
                (
                    rule
                    for rule in self.migration_rules
                    if rule.from_version == current_version
                    and self._version_compare(rule.to_version, current_version) > 0
                    and self._version_compare(rule.to_version, to_version) <= 0
                ),
                None,
            )
            if next_rule is None:
                return []
            applicable_rules.extend(
                rule
                for rule in self.migration_rules
                if rule.from_version == current_version and rule.to_version == next_rule.to_version
            )
            current_version = next_rule.to_version
        return applicable_rules

    def _version_parts(self, version: str) -> Tuple[int, int, int]:
        match = self.version_pattern.fullmatch(version)
        if match:
            return int(match.group(1)), int(match.group(2)), int(match.group(3))
        return 0, 0, 0

    def _version_compare(self, version1: str, version2: str) -> int:
        """Compare versions, returning -1, 0 or 1."""
        v1_parts, v2_parts = self._version_parts(version1), self._version_parts(version2)
        return (v1_parts > v2_parts) - (v1_parts < v2_parts)

    def migrate_config(
        self, config: Dict[str, Any], target_version: Optional[str] = None, backup_path: Optional[str] = None
    ) -> MigrationResult:
        """Migrate once and return the existing public result schema."""
        result, _ = self._migrate_config(config, target_version, backup_path)
        return result

    def _migrate_config(
        self, config: Dict[str, Any], target_version: Optional[str], backup_path: Optional[str]
    ) -> Tuple[MigrationResult, Optional[Dict[str, Any]]]:
        current_version = "unknown"
        target_version = "2.0.0" if target_version is None else target_version
        try:
            config = _config_object(config)
            current_version = self.detect_config_version(config)
            if not self.version_pattern.fullmatch(target_version):
                raise ValueError("目标版本必须是 major.minor.patch 字符串")
            if current_version == target_version:
                return (
                    MigrationResult(
                        MigrationStatus.SKIPPED,
                        current_version,
                        target_version,
                        ["配置版本已是最新，跳过迁移"],
                        [],
                        [],
                        [],
                    ),
                    None,
                )

            migration_rules = self.get_migration_path(current_version, target_version)
            if not migration_rules:
                raise ValueError(f"无法找到从 {current_version} 到 {target_version} 的迁移路径")

            backup_file = None
            if backup_path:
                backup_file = self._create_backup(config, backup_path)
                if backup_file is None:
                    raise OSError("无法创建请求的配置备份，已停止迁移")

            current_config, messages, warnings, errors, migrated_paths = self._apply_rules(config, migration_rules)
            status = MigrationStatus.FAILED if errors else MigrationStatus.COMPLETED
            result = MigrationResult(
                status, current_version, target_version, messages, warnings, errors, migrated_paths, backup_file
            )
            if errors:
                return result, None
            current_config["version"] = target_version
            current_config["migrated_at"] = datetime.now().isoformat()
            return result, current_config
        except Exception as error:
            logger.error(f"配置迁移失败: {error}")
            return (
                MigrationResult(
                    MigrationStatus.FAILED, current_version, target_version, [], [], [f"迁移过程异常: {error}"], []
                ),
                None,
            )

    def _apply_rules(
        self, config: Dict[str, Any], rules: List[MigrationRule]
    ) -> Tuple[Dict[str, Any], List[str], List[str], List[str], List[str]]:
        current_config = _json_copy(config)
        messages: List[str] = []
        warnings: List[str] = []
        errors: List[str] = []
        migrated_paths: List[str] = []
        for rule in rules:
            if rule.can_migrate(current_config):
                try:
                    new_config, rule_messages, rule_warnings = rule.migrate(current_config)
                    current_config = _config_object(new_config)
                    messages.extend(rule_messages)
                    warnings.extend(rule_warnings)
                    migrated_paths.append(f"{rule.from_version} → {rule.to_version}")
                    logger.info(f"应用迁移规则: {rule.name}")
                except Exception as error:
                    message = f"迁移规则执行失败 {rule.name}: {error}"
                    errors.append(message)
                    logger.error(message)
        return current_config, messages, warnings, errors, migrated_paths

    def _create_backup(self, config: Dict[str, Any], backup_path: str) -> Optional[str]:
        """创建配置备份"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"{backup_path}.backup_{timestamp}.json"

            with open(backup_file, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"创建配置备份: {backup_file}")
            return backup_file

        except Exception as e:
            logger.error(f"创建备份失败: {e}")
            return None

    def migrate_config_file(
        self, config_path: str, target_version: Optional[str] = None, backup: bool = True
    ) -> MigrationResult:
        """迁移配置文件"""
        try:
            # 加载配置
            with open(config_path, "r", encoding="utf-8") as f:
                config = _config_object(json.load(f))

            # 设置备份路径
            backup_path = config_path if backup else None

            # Keep the actual transformed JSON from the single execution.
            result, current_config = self._migrate_config(config, target_version, backup_path)
            if result.status == MigrationStatus.COMPLETED and current_config is not None:
                serialized = json.dumps(current_config, indent=2, ensure_ascii=False, default=str)
                with open(config_path, "w", encoding="utf-8") as f:
                    f.write(serialized)
                logger.info(f"配置文件迁移完成: {config_path}")

            return result

        except Exception as e:
            logger.error(f"配置文件迁移失败: {e}")
            return MigrationResult(
                status=MigrationStatus.FAILED,
                from_version="unknown",
                to_version=target_version or "unknown",
                messages=[],
                warnings=[],
                errors=[f"文件迁移失败: {str(e)}"],
                migrated_paths=[],
            )

    def validate_migration(self, original_config: Dict[str, Any], migrated_config: Dict[str, Any]) -> List[str]:
        """验证迁移结果"""
        issues = []

        # 检查必需字段
        required_fields = ["storage", "processors", "llm"]
        for field in required_fields:
            if field not in migrated_config:
                issues.append(f"缺少必需字段: {field}")

        # 检查数据完整性
        if "storage" in original_config and "storage" in migrated_config:
            original_storage = original_config["storage"]
            migrated_storage = migrated_config["storage"]

            # 检查存储路径是否保留
            original_paths = set()
            migrated_paths = set()

            for key, value in original_storage.items():
                if "path" in key and isinstance(value, str):
                    original_paths.add(value)

            for key, value in migrated_storage.items():
                if "path" in key and isinstance(value, str):
                    migrated_paths.add(value)

            missing_paths = original_paths - migrated_paths
            if missing_paths:
                issues.append(f"存储路径丢失: {missing_paths}")

        return issues

    def get_available_versions(self) -> List[str]:
        """获取可用版本列表"""
        versions = set()

        for rule in self.migration_rules:
            versions.add(rule.from_version)
            versions.add(rule.to_version)

        # 排序版本
        version_list = list(versions)
        version_list.sort(key=self._version_parts)

        return version_list


# 便捷函数
def migrate_config_file(config_path: str, target_version: Optional[str] = None) -> MigrationResult:
    """便捷函数：迁移配置文件"""
    tool = ConfigMigrationTool()
    return tool.migrate_config_file(config_path, target_version)


def check_migration_needed(config_path: str) -> Tuple[bool, str, str]:
    """检查是否需要迁移"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = _config_object(json.load(f))

        tool = ConfigMigrationTool()
        current_version = tool.detect_config_version(config)
        latest_version = "2.0.0"

        needs_migration = current_version != latest_version
        return needs_migration, current_version, latest_version

    except Exception as e:
        logger.error(f"检查迁移状态失败: {e}")
        return False, "unknown", "unknown"
