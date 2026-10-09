"""
统一配置中心 - 实现中心化配置管理

提供统一的配置访问、监控、热更新和验证功能。
支持多环境配置、运行时配置更新和配置变更监听。
"""

import copy
import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypedDict

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from .config_manager import ConfigManager
from .config_validator import ConfigValidator, generate_default_config

WATCHDOG_AVAILABLE = True


class ConfigChangeEvent:
    """配置变更事件"""

    def __init__(self, key: str, old_value: Any, new_value: Any, timestamp: Optional[datetime] = None) -> None:
        self.key = key
        self.old_value = old_value
        self.new_value = new_value
        self.timestamp = timestamp or datetime.now()


# Retain the existing exported base name while using the declared watchdog runtime.
ConfigWatcherBase = FileSystemEventHandler


class _ConfigStats(TypedDict):
    total_reloads: int
    total_changes: int
    last_reload: Optional[datetime]
    validation_errors: int


class ConfigWatcher(ConfigWatcherBase):
    """配置文件监听器"""

    def __init__(self, config_center: "ConfigCenter") -> None:
        if WATCHDOG_AVAILABLE:
            super().__init__()
        self.config_center = config_center
        self.logger = logging.getLogger("config_watcher")

    def on_modified(self, event: FileSystemEvent) -> None:
        """文件修改事件处理"""
        if not event.is_directory and os.fsdecode(event.src_path).endswith(".json"):
            self.logger.info(f"检测到配置文件变更: {os.fsdecode(event.src_path)}")
            self.config_center._reload_config()


class ConfigCenter:
    """
    统一配置中心

    提供配置管理、监控、热更新和验证功能。
    支持多环境配置和运行时配置变更。
    """

    def __init__(
        self, base_config_path: str = "./config", environment: str = "default", watch_changes: bool = True
    ) -> None:
        """
        初始化配置中心

        Args:
            base_config_path: 配置文件基础路径
            environment: 环境名称 (default, development, production等)
            watch_changes: 是否监听配置文件变更
        """
        self.base_config_path = Path(base_config_path)
        self.environment = environment
        self.watch_changes = watch_changes

        # 核心组件
        self.config_manager = ConfigManager(str(self.base_config_path / "config.json"))
        self.validator = ConfigValidator()
        self.logger = logging.getLogger("config_center")

        # 配置存储
        self._config_cache: Dict[str, Any] = {}
        self._environment_configs: Dict[str, Dict[str, Any]] = {}
        self._change_listeners: List[Callable[[ConfigChangeEvent], None]] = []

        # 文件监听
        self.observer: Optional[BaseObserver] = None
        self._lock = threading.RLock()

        # 配置统计
        self.stats: _ConfigStats = {"total_reloads": 0, "total_changes": 0, "last_reload": None, "validation_errors": 0}

        # 初始化
        self._initialize()

    def _initialize(self) -> None:
        """初始化配置中心"""
        try:
            # 加载配置
            self._load_all_configs()

            # 启动文件监听
            if self.watch_changes:
                self._start_file_watcher()

            self.logger.info(f"配置中心初始化完成 - 环境: {self.environment}")

        except Exception as e:
            self.logger.error(f"配置中心初始化失败: {str(e)}")
            raise

    def _load_all_configs(self) -> None:
        """加载所有环境的配置"""
        previous_configs = self._environment_configs
        self._environment_configs = {}
        try:
            # 加载基础配置
            base_config_file = self.base_config_path / "config.json"
            if base_config_file.exists():
                with open(base_config_file, "r", encoding="utf-8") as f:
                    base_config = json.load(f)
                self._environment_configs["base"] = base_config

            # 加载环境特定配置
            env_config_file = self.base_config_path / f"config.{self.environment}.json"
            if env_config_file.exists():
                with open(env_config_file, "r", encoding="utf-8") as f:
                    env_config = json.load(f)
                self._environment_configs[self.environment] = env_config

            # 合并配置
            self._merge_configs()

        except Exception as e:
            self.logger.error(f"加载配置失败: {str(e)}")
            self._environment_configs = previous_configs
            # Keep the last valid configuration when a reload cannot be read.
            if not self._config_cache:
                self._config_cache = generate_default_config()
            raise

    def _merge_configs(self) -> None:
        """合并多环境配置"""
        merged_config = {}

        # 先应用基础配置
        if "base" in self._environment_configs:
            merged_config.update(copy.deepcopy(self._environment_configs["base"]))

        # 再应用环境特定配置
        if self.environment in self._environment_configs:
            self._deep_merge(merged_config, self._environment_configs[self.environment])

        # 验证合并后的配置
        if self.validator.validate_config(merged_config):
            if not self.config_manager.apply_config(merged_config):
                raise ValueError("Effective configuration is invalid")
            self._config_cache = self.config_manager.get_config()
            self.logger.info("配置验证通过")
        else:
            self.stats["validation_errors"] += 1
            report = self.validator.get_validation_report()
            self.logger.error(f"配置验证失败: {report['errors']}")
            raise ValueError("Merged configuration is invalid")

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """深度合并配置字典"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = copy.deepcopy(value)

    def _start_file_watcher(self) -> None:
        """启动文件监听器"""
        try:
            if not WATCHDOG_AVAILABLE:
                self.logger.warning("watchdog 未安装，文件监听功能不可用。请运行: pip install watchdog")
                return

            if not self.base_config_path.exists():
                return

            self.observer = Observer()
            self.observer.schedule(ConfigWatcher(self), str(self.base_config_path), recursive=True)
            self.observer.start()
            self.logger.info("配置文件监听已启动")

        except Exception as e:
            self.logger.error(f"启动文件监听失败: {str(e)}")

    def _reload_config(self) -> None:
        """重新加载配置"""
        with self._lock:
            try:
                old_config = self._config_cache.copy()
                self._load_all_configs()

                # 检测变更
                self._detect_changes(old_config, self._config_cache)

                self.stats["total_reloads"] += 1
                self.stats["last_reload"] = datetime.now()

                self.logger.info("配置重新加载完成")

            except Exception as e:
                self.logger.error(f"重新加载配置失败: {str(e)}")

    def _detect_changes(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> None:
        """检测配置变更并通知监听器"""
        changes: List[ConfigChangeEvent] = []

        # 检测变更的键
        self._find_changes("", old_config, new_config, changes)

        # 通知监听器
        for change in changes:
            self.stats["total_changes"] += 1
            for listener in self._change_listeners:
                try:
                    listener(change)
                except Exception as e:
                    self.logger.error(f"配置变更监听器执行失败: {str(e)}")

    def _find_changes(
        self, prefix: str, old_dict: Dict[str, Any], new_dict: Dict[str, Any], changes: List[ConfigChangeEvent]
    ) -> None:
        """递归查找配置变更"""
        all_keys = set(old_dict.keys()) | set(new_dict.keys())

        for key in all_keys:
            full_key = f"{prefix}.{key}" if prefix else key

            if key not in old_dict:
                # 新增的键
                changes.append(ConfigChangeEvent(full_key, None, new_dict[key]))
            elif key not in new_dict:
                # 删除的键
                changes.append(ConfigChangeEvent(full_key, old_dict[key], None))
            elif old_dict[key] != new_dict[key]:
                if isinstance(old_dict[key], dict) and isinstance(new_dict[key], dict):
                    # 递归检查嵌套字典
                    self._find_changes(full_key, old_dict[key], new_dict[key], changes)
                else:
                    # 值变更
                    changes.append(ConfigChangeEvent(full_key, old_dict[key], new_dict[key]))

    def get_config(self) -> Dict[str, Any]:
        """获取完整配置"""
        with self._lock:
            return copy.deepcopy(self._config_cache)

    def get_value(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值

        Args:
            key_path: 配置键路径 (如 "storage.base_path")
            default: 默认值

        Returns:
            配置值
        """
        with self._lock:
            keys = key_path.split(".")
            config = self._config_cache

            for key in keys:
                if isinstance(config, dict) and key in config:
                    config = config[key]
                else:
                    return default

            return config

    def set_value(self, key_path: str, value: Any, persist: bool = True) -> bool:
        """
        设置配置值

        Args:
            key_path: 配置键路径
            value: 新值
            persist: 是否持久化到文件

        Returns:
            是否成功设置
        """
        with self._lock:
            try:
                old_value = self.get_value(key_path)

                candidate = copy.deepcopy(self._config_cache)
                keys = key_path.split(".")
                config = candidate
                for key in keys[:-1]:
                    if key not in config:
                        config[key] = {}
                    if not isinstance(config[key], dict):
                        return False
                    config = config[key]
                config[keys[-1]] = copy.deepcopy(value)
                if not self.validator.validate_config(candidate):
                    return False
                if persist and not self._persist_config(candidate):
                    return False
                if not self.config_manager.apply_config(candidate):
                    return False
                self._config_cache = self.config_manager.get_config()

                self._notify_change(ConfigChangeEvent(key_path, old_value, value))

                self.stats["total_changes"] += 1
                return True

            except Exception as e:
                self.logger.error(f"设置配置值失败: {str(e)}")
                return False

    def _notify_change(self, change: ConfigChangeEvent) -> None:
        for listener in self._change_listeners:
            try:
                listener(change)
            except Exception as error:
                self.logger.error(f"配置变更监听器执行失败: {error}")

    def _persist_config(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """持久化配置到文件"""
        try:
            config_file = self.base_config_path / "config.json"
            config_file.parent.mkdir(parents=True, exist_ok=True)

            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(self._config_cache if config is None else config, f, indent=2, ensure_ascii=False)

            self.logger.info("配置已持久化")
            return True

        except Exception as e:
            self.logger.error(f"持久化配置失败: {str(e)}")
            return False

    def add_change_listener(self, listener: Callable[[ConfigChangeEvent], None]) -> None:
        """添加配置变更监听器"""
        self._change_listeners.append(listener)

    def remove_change_listener(self, listener: Callable[[ConfigChangeEvent], None]) -> None:
        """移除配置变更监听器"""
        if listener in self._change_listeners:
            self._change_listeners.remove(listener)

    def get_environment_config(self, environment: str) -> Optional[Dict[str, Any]]:
        """获取特定环境的配置"""
        return self._environment_configs.get(environment)

    def switch_environment(self, environment: str) -> bool:
        """切换环境配置"""
        try:
            old_env = self.environment
            self.environment = environment
            self._load_all_configs()

            self.logger.info(f"环境已切换: {old_env} -> {environment}")
            return True

        except Exception as e:
            self.logger.error(f"切换环境失败: {str(e)}")
            self.environment = old_env  # 回滚
            return False

    def get_stats(self) -> Dict[str, Any]:
        """获取配置中心统计信息"""
        with self._lock:
            return {
                **self.stats,
                "environment": self.environment,
                "total_listeners": len(self._change_listeners),
                "config_keys": len(self._config_cache),
                "environments": list(self._environment_configs.keys()),
            }

    def validate_current_config(self) -> Dict[str, Any]:
        """验证当前配置"""
        with self._lock:
            is_valid = self.validator.validate_config(self._config_cache)
            return {"is_valid": is_valid, "report": self.validator.get_validation_report()}

    def backup_config(self, backup_path: Optional[str] = None) -> str:
        """备份当前配置"""
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"./config/backup/config_backup_{timestamp}.json"

        try:
            Path(backup_path).parent.mkdir(parents=True, exist_ok=True)

            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(self._config_cache, f, indent=2, ensure_ascii=False)

            self.logger.info(f"配置已备份到: {backup_path}")
            return backup_path

        except Exception as e:
            self.logger.error(f"配置备份失败: {str(e)}")
            raise

    def restore_config(self, backup_path: str) -> bool:
        """从备份恢复配置"""
        try:
            with open(backup_path, "r", encoding="utf-8") as f:
                backup_config = json.load(f)

            if self.validator.validate_config(backup_config):
                old_config = self._config_cache.copy()
                if not self._persist_config(backup_config):
                    return False
                if not self.config_manager.apply_config(backup_config):
                    return False
                self._config_cache = self.config_manager.get_config()

                # 检测变更并通知
                self._detect_changes(old_config, self._config_cache)

                self.logger.info(f"配置已从备份恢复: {backup_path}")
                return True
            else:
                self.logger.error("备份配置验证失败，无法恢复")
                return False

        except Exception as e:
            self.logger.error(f"配置恢复失败: {str(e)}")
            return False

    def close(self) -> None:
        """关闭配置中心"""
        try:
            if self.observer:
                self.observer.stop()
                self.observer.join()

            self.logger.info("配置中心已关闭")

        except Exception as e:
            self.logger.error(f"关闭配置中心失败: {str(e)}")


# 全局配置中心实例
_config_center: Optional[ConfigCenter] = None


def get_config_center(base_config_path: str = "./config", environment: Optional[str] = None) -> ConfigCenter:
    """
    获取全局配置中心实例

    Args:
        base_config_path: 配置文件基础路径
        environment: 环境名称

    Returns:
        配置中心实例
    """
    global _config_center

    if _config_center is None:
        if environment is None:
            environment = os.getenv("CONFIG_ENV", "default")

        _config_center = ConfigCenter(base_config_path=base_config_path, environment=environment)

    return _config_center


def init_config_center(
    base_config_path: str = "./config", environment: Optional[str] = None, watch_changes: bool = True
) -> ConfigCenter:
    """
    初始化全局配置中心

    Args:
        base_config_path: 配置文件基础路径
        environment: 环境名称
        watch_changes: 是否监听文件变更

    Returns:
        配置中心实例
    """
    global _config_center

    if environment is None:
        environment = os.getenv("CONFIG_ENV", "default")

    _config_center = ConfigCenter(
        base_config_path=base_config_path, environment=environment, watch_changes=watch_changes
    )

    return _config_center
