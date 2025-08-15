"""
配置迁移工具

提供配置格式升级、数据迁移、兼容性转换和自动迁移功能。
支持多版本配置格式间的转换和向后兼容性处理。
"""

import os
import json
import shutil
from typing import Dict, Any, List, Optional, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import logging
import re

logger = logging.getLogger(__name__)

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
            'status': self.status.value,
            'from_version': self.from_version,
            'to_version': self.to_version,
            'messages': self.messages,
            'warnings': self.warnings,
            'errors': self.errors,
            'migrated_paths': self.migrated_paths,
            'backup_path': self.backup_path
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
        super().__init__(
            from_version, to_version, 
            "path_rename", 
            f"重命名配置路径: {len(path_mappings)} 个映射"
        )
        self.path_mappings = path_mappings
    
    def can_migrate(self, config: Dict[str, Any]) -> bool:
        """检查是否有需要重命名的路径"""
        for old_path in self.path_mappings.keys():
            if self._path_exists(config, old_path):
                return True
        return False
    
    def migrate(self, config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str]]:
        """执行路径重命名"""
        new_config = json.loads(json.dumps(config))  # 深拷贝
        messages = []
        warnings = []
        
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
        keys = path.split('.')
        current = config
        
        for key in keys:
            current = current[key]
        
        return current
    
    def _set_path_value(self, config: Dict[str, Any], path: str, value: Any):
        """设置路径值"""
        keys = path.split('.')
        current = config
        
        # 创建中间路径
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _delete_path(self, config: Dict[str, Any], path: str):
        """删除路径"""
        keys = path.split('.')
        current = config
        
        # 导航到父级
        for key in keys[:-1]:
            current = current[key]
        
        # 删除最后一个键
        if keys[-1] in current:
            del current[keys[-1]]

class ValueTransformRule(MigrationRule):
    """值转换规则"""
    
    def __init__(self, from_version: str, to_version: str, path: str, 
                 transform_func: Callable[[Any], Any], description: str = None):
        super().__init__(
            from_version, to_version,
            "value_transform",
            description or f"转换值: {path}"
        )
        self.path = path
        self.transform_func = transform_func
    
    def can_migrate(self, config: Dict[str, Any]) -> bool:
        """检查是否需要转换"""
        return self._path_exists(config, self.path)
    
    def migrate(self, config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str]]:
        """执行值转换"""
        new_config = json.loads(json.dumps(config))
        messages = []
        warnings = []
        
        if self._path_exists(new_config, self.path):
            try:
                old_value = self._get_path_value(new_config, self.path)
                new_value = self.transform_func(old_value)
                self._set_path_value(new_config, self.path, new_value)
                messages.append(f"转换值 {self.path}: {old_value} → {new_value}")
            except Exception as e:
                warnings.append(f"转换值失败 {self.path}: {str(e)}")
        
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
        keys = path.split('.')
        current = config
        
        for key in keys:
            current = current[key]
        
        return current
    
    def _set_path_value(self, config: Dict[str, Any], path: str, value: Any):
        """设置路径值"""
        keys = path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value

class StructureRule(MigrationRule):
    """结构转换规则"""
    
    def __init__(self, from_version: str, to_version: str, 
                 structure_func: Callable[[Dict[str, Any]], Dict[str, Any]], 
                 description: str = None):
        super().__init__(
            from_version, to_version,
            "structure_transform",
            description or "转换配置结构"
        )
        self.structure_func = structure_func
    
    def can_migrate(self, config: Dict[str, Any]) -> bool:
        """总是可以执行结构转换"""
        return True
    
    def migrate(self, config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str]]:
        """执行结构转换"""
        messages = []
        warnings = []
        
        try:
            new_config = self.structure_func(config)
            messages.append("执行配置结构转换")
            return new_config, messages, warnings
        except Exception as e:
            warnings.append(f"结构转换失败: {str(e)}")
            return config, messages, warnings

class ConfigMigrationTool:
    """配置迁移工具"""
    
    def __init__(self):
        self.migration_rules: List[MigrationRule] = []
        self.version_pattern = re.compile(r'^(\d+)\.(\d+)\.(\d+)$')
        
        # 注册默认迁移规则
        self._register_default_rules()
    
    def _register_default_rules(self):
        """注册默认迁移规则"""
        # v1.0.0 → v1.1.0: 重命名存储配置
        self.add_rule(PathRenameRule(
            "1.0.0", "1.1.0",
            {
                "storage.data_path": "storage.base_path",
                "storage.result_path": "storage.output_path"
            }
        ))
        
        # v1.1.0 → v1.2.0: 处理器配置结构化
        def restructure_processors(config: Dict[str, Any]) -> Dict[str, Any]:
            new_config = json.loads(json.dumps(config))
            
            if 'processors' in new_config:
                for name, proc_config in new_config['processors'].items():
                    if isinstance(proc_config, bool):
                        # 从布尔值转换为对象
                        new_config['processors'][name] = {
                            'enabled': proc_config,
                            'config': {}
                        }
                    elif isinstance(proc_config, dict) and 'enabled' not in proc_config:
                        # 添加缺失的enabled字段
                        proc_config['enabled'] = True
            
            return new_config
        
        self.add_rule(StructureRule(
            "1.1.0", "1.2.0",
            restructure_processors,
            "重构处理器配置格式"
        ))
        
        # v1.2.0 → v1.3.0: LLM配置标准化
        self.add_rule(PathRenameRule(
            "1.2.0", "1.3.0",
            {
                "generator": "llm",
                "llm.model_name": "llm.model",
                "llm.params": "llm.settings"
            }
        ))
        
        # 温度值范围调整
        def normalize_temperature(temp: Any) -> float:
            if isinstance(temp, (int, float)):
                return max(0.0, min(2.0, float(temp)))
            return 1.0
        
        self.add_rule(ValueTransformRule(
            "1.2.0", "1.3.0",
            "llm.settings.temperature",
            normalize_temperature,
            "标准化温度参数范围"
        ))
        
        # v1.3.0 → v2.0.0: 向量数据库配置重构
        def restructure_vector_db(config: Dict[str, Any]) -> Dict[str, Any]:
            new_config = json.loads(json.dumps(config))
            
            # 迁移旧的document_store配置
            if 'document_store' in new_config and 'vector_db' not in new_config:
                new_config['vector_db'] = {
                    'document_store': new_config['document_store']
                }
                del new_config['document_store']
            
            # 添加新的向量存储配置
            if 'vector_db' in new_config:
                vector_config = new_config['vector_db']
                if 'document_store' in vector_config:
                    ds_config = vector_config['document_store']
                    
                    # 设置默认值
                    if 'type' not in ds_config:
                        ds_config['type'] = 'memory'
                    if 'embedding_dim' not in ds_config:
                        ds_config['embedding_dim'] = 1536
                    if 'similarity' not in ds_config:
                        ds_config['similarity'] = 'cosine'
            
            return new_config
        
        self.add_rule(StructureRule(
            "1.3.0", "2.0.0",
            restructure_vector_db,
            "重构向量数据库配置"
        ))
    
    def add_rule(self, rule: MigrationRule):
        """添加迁移规则"""
        self.migration_rules.append(rule)
        logger.debug(f"添加迁移规则: {rule.name} ({rule.from_version} → {rule.to_version})")
    
    def detect_config_version(self, config: Dict[str, Any]) -> str:
        """检测配置版本"""
        # 检查显式版本标记
        if 'version' in config:
            return config['version']
        
        # 基于配置结构推断版本
        if 'vector_db' in config:
            return "2.0.0"
        elif 'llm' in config:
            return "1.3.0"
        elif 'processors' in config and isinstance(list(config['processors'].values())[0], dict):
            return "1.2.0"
        elif 'storage' in config and 'base_path' in config['storage']:
            return "1.1.0"
        else:
            return "1.0.0"
    
    def get_migration_path(self, from_version: str, to_version: str) -> List[MigrationRule]:
        """获取迁移路径"""
        # 简单实现：按版本顺序查找规则
        applicable_rules = []
        current_version = from_version
        
        while current_version != to_version:
            found_rule = None
            
            for rule in self.migration_rules:
                if rule.from_version == current_version:
                    # 检查是否朝向目标版本
                    if self._version_compare(rule.to_version, current_version) > 0:
                        found_rule = rule
                        break
            
            if found_rule:
                applicable_rules.append(found_rule)
                current_version = found_rule.to_version
            else:
                break
        
        return applicable_rules
    
    def _version_compare(self, version1: str, version2: str) -> int:
        """比较版本号，返回 -1, 0, 1"""
        def parse_version(v: str) -> Tuple[int, int, int]:
            match = self.version_pattern.match(v)
            if match:
                return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
            return (0, 0, 0)
        
        v1_parts = parse_version(version1)
        v2_parts = parse_version(version2)
        
        if v1_parts < v2_parts:
            return -1
        elif v1_parts > v2_parts:
            return 1
        else:
            return 0
    
    def migrate_config(self, config: Dict[str, Any], target_version: str = None, 
                      backup_path: str = None) -> MigrationResult:
        """迁移配置"""
        try:
            # 检测当前版本
            current_version = self.detect_config_version(config)
            
            # 设置目标版本
            if target_version is None:
                target_version = "2.0.0"  # 默认最新版本
            
            logger.info(f"开始配置迁移: {current_version} → {target_version}")
            
            # 如果版本相同，无需迁移
            if current_version == target_version:
                return MigrationResult(
                    status=MigrationStatus.SKIPPED,
                    from_version=current_version,
                    to_version=target_version,
                    messages=["配置版本已是最新，跳过迁移"],
                    warnings=[],
                    errors=[],
                    migrated_paths=[]
                )
            
            # 创建备份
            backup_file = None
            if backup_path:
                backup_file = self._create_backup(config, backup_path)
            
            # 获取迁移路径
            migration_rules = self.get_migration_path(current_version, target_version)
            
            if not migration_rules:
                return MigrationResult(
                    status=MigrationStatus.FAILED,
                    from_version=current_version,
                    to_version=target_version,
                    messages=[],
                    warnings=[],
                    errors=[f"无法找到从 {current_version} 到 {target_version} 的迁移路径"],
                    migrated_paths=[],
                    backup_path=backup_file
                )
            
            # 执行迁移
            current_config = json.loads(json.dumps(config))  # 深拷贝
            all_messages = []
            all_warnings = []
            all_errors = []
            migrated_paths = []
            
            for rule in migration_rules:
                if rule.can_migrate(current_config):
                    try:
                        new_config, messages, warnings = rule.migrate(current_config)
                        current_config = new_config
                        all_messages.extend(messages)
                        all_warnings.extend(warnings)
                        migrated_paths.append(f"{rule.from_version} → {rule.to_version}")
                        
                        logger.info(f"应用迁移规则: {rule.name}")
                    except Exception as e:
                        error_msg = f"迁移规则执行失败 {rule.name}: {str(e)}"
                        all_errors.append(error_msg)
                        logger.error(error_msg)
            
            # 设置版本标记
            current_config['version'] = target_version
            current_config['migrated_at'] = datetime.now().isoformat()
            
            # 确定状态
            status = MigrationStatus.COMPLETED
            if all_errors:
                status = MigrationStatus.FAILED
            
            return MigrationResult(
                status=status,
                from_version=current_version,
                to_version=target_version,
                messages=all_messages,
                warnings=all_warnings,
                errors=all_errors,
                migrated_paths=migrated_paths,
                backup_path=backup_file
            )
            
        except Exception as e:
            logger.error(f"配置迁移失败: {e}")
            return MigrationResult(
                status=MigrationStatus.FAILED,
                from_version=current_version if 'current_version' in locals() else "unknown",
                to_version=target_version or "unknown",
                messages=[],
                warnings=[],
                errors=[f"迁移过程异常: {str(e)}"],
                migrated_paths=[]
            )
    
    def _create_backup(self, config: Dict[str, Any], backup_path: str) -> str:
        """创建配置备份"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = f"{backup_path}.backup_{timestamp}.json"
            
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"创建配置备份: {backup_file}")
            return backup_file
            
        except Exception as e:
            logger.error(f"创建备份失败: {e}")
            return None
    
    def migrate_config_file(self, config_path: str, target_version: str = None, 
                           backup: bool = True) -> MigrationResult:
        """迁移配置文件"""
        try:
            # 加载配置
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 设置备份路径
            backup_path = config_path if backup else None
            
            # 执行迁移
            result = self.migrate_config(config, target_version, backup_path)
            
            # 保存迁移后的配置
            if result.status == MigrationStatus.COMPLETED:
                migrated_config = config.copy()
                
                # 重新应用迁移以获取最终配置
                migration_rules = self.get_migration_path(
                    self.detect_config_version(config),
                    target_version or "2.0.0"
                )
                
                current_config = json.loads(json.dumps(config))
                for rule in migration_rules:
                    if rule.can_migrate(current_config):
                        current_config, _, _ = rule.migrate(current_config)
                
                current_config['version'] = target_version or "2.0.0"
                current_config['migrated_at'] = datetime.now().isoformat()
                
                # 保存到文件
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(current_config, f, indent=2, ensure_ascii=False, default=str)
                
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
                migrated_paths=[]
            )
    
    def validate_migration(self, original_config: Dict[str, Any], 
                          migrated_config: Dict[str, Any]) -> List[str]:
        """验证迁移结果"""
        issues = []
        
        # 检查必需字段
        required_fields = ['storage', 'processors', 'llm']
        for field in required_fields:
            if field not in migrated_config:
                issues.append(f"缺少必需字段: {field}")
        
        # 检查数据完整性
        if 'storage' in original_config and 'storage' in migrated_config:
            original_storage = original_config['storage']
            migrated_storage = migrated_config['storage']
            
            # 检查存储路径是否保留
            original_paths = set()
            migrated_paths = set()
            
            for key, value in original_storage.items():
                if 'path' in key and isinstance(value, str):
                    original_paths.add(value)
            
            for key, value in migrated_storage.items():
                if 'path' in key and isinstance(value, str):
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
        version_list.sort(key=lambda v: self._version_compare("0.0.0", v))
        
        return version_list

# 便捷函数
def migrate_config_file(config_path: str, target_version: str = None) -> MigrationResult:
    """便捷函数：迁移配置文件"""
    tool = ConfigMigrationTool()
    return tool.migrate_config_file(config_path, target_version)

def check_migration_needed(config_path: str) -> Tuple[bool, str, str]:
    """检查是否需要迁移"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        tool = ConfigMigrationTool()
        current_version = tool.detect_config_version(config)
        latest_version = "2.0.0"
        
        needs_migration = current_version != latest_version
        return needs_migration, current_version, latest_version
        
    except Exception as e:
        logger.error(f"检查迁移状态失败: {e}")
        return False, "unknown", "unknown"