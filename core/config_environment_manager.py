"""
环境配置管理器

提供多环境配置管理、动态环境切换、配置继承和环境隔离功能。
支持环境配置模板、配置覆盖策略和环境特定验证。
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import copy

logger = logging.getLogger(__name__)

class EnvironmentType(Enum):
    """环境类型"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    LOCAL = "local"
    CUSTOM = "custom"

@dataclass
class EnvironmentInfo:
    """环境信息"""
    name: str
    type: EnvironmentType
    description: str
    config_file: str
    parent_env: Optional[str] = None
    active: bool = False
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

class ConfigEnvironmentManager:
    """环境配置管理器"""
    
    def __init__(self, config_dir: str = "./config"):
        self.config_dir = Path(config_dir)
        self.environments: Dict[str, EnvironmentInfo] = {}
        self.current_environment: Optional[str] = None
        self.base_config: Dict[str, Any] = {}
        self.config_cache: Dict[str, Dict[str, Any]] = {}
        
        # 配置覆盖策略
        self.override_strategy = "deep_merge"  # 可选: replace, merge, deep_merge
        
        # 初始化
        self._initialize()
    
    def _initialize(self):
        """初始化环境管理器"""
        try:
            # 确保配置目录存在
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            # 扫描现有环境
            self._scan_environments()
            
            # 加载基础配置
            self._load_base_config()
            
            # 设置默认环境
            if not self.current_environment and self.environments:
                default_env = self._get_default_environment()
                if default_env:
                    self.set_environment(default_env)
            
            logger.info(f"环境管理器初始化完成，发现 {len(self.environments)} 个环境")
            
        except Exception as e:
            logger.error(f"环境管理器初始化失败: {e}")
            raise
    
    def _scan_environments(self):
        """扫描现有环境配置"""
        self.environments.clear()
        
        # 扫描配置文件
        for config_file in self.config_dir.glob("config.*.json"):
            env_name = config_file.stem.split('.', 1)[1]  # 从 config.env.json 提取 env
            
            # 确定环境类型
            env_type = EnvironmentType.CUSTOM
            for et in EnvironmentType:
                if et.value == env_name:
                    env_type = et
                    break
            
            # 创建环境信息
            env_info = EnvironmentInfo(
                name=env_name,
                type=env_type,
                description=f"{env_name} 环境配置",
                config_file=str(config_file),
                active=False
            )
            
            self.environments[env_name] = env_info
            logger.debug(f"发现环境: {env_name} ({env_type.value})")
        
        # 添加默认环境（如果没有找到任何环境）
        if not self.environments:
            self._create_default_environments()
    
    def _create_default_environments(self):
        """创建默认环境配置"""
        default_envs = [
            (EnvironmentType.DEVELOPMENT, "开发环境 - 用于本地开发和调试"),
            (EnvironmentType.TESTING, "测试环境 - 用于自动化测试"),
            (EnvironmentType.PRODUCTION, "生产环境 - 用于正式部署")
        ]
        
        for env_type, description in default_envs:
            env_name = env_type.value
            config_file = self.config_dir / f"config.{env_name}.json"
            
            # 创建默认配置
            default_config = self._generate_default_config(env_type)
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            
            # 添加环境信息
            env_info = EnvironmentInfo(
                name=env_name,
                type=env_type,
                description=description,
                config_file=str(config_file)
            )
            
            self.environments[env_name] = env_info
            logger.info(f"创建默认环境: {env_name}")
    
    def _generate_default_config(self, env_type: EnvironmentType) -> Dict[str, Any]:
        """生成默认环境配置"""
        base_config = {
            "environment": {
                "name": env_type.value,
                "type": env_type.value,
                "debug": env_type in [EnvironmentType.DEVELOPMENT, EnvironmentType.TESTING]
            },
            "logging": {
                "level": "DEBUG" if env_type == EnvironmentType.DEVELOPMENT else "INFO"
            }
        }
        
        # 环境特定配置
        if env_type == EnvironmentType.DEVELOPMENT:
            base_config.update({
                "storage": {
                    "base_path": "./dev_data",
                    "output_path": "./dev_output"
                },
                "llm": {
                    "model": "gpt-3.5-turbo",
                    "settings": {
                        "temperature": 0.7,
                        "max_tokens": 1000
                    }
                }
            })
        elif env_type == EnvironmentType.TESTING:
            base_config.update({
                "storage": {
                    "base_path": "./test_data",
                    "output_path": "./test_output"
                },
                "llm": {
                    "model": "mock-model",
                    "settings": {
                        "temperature": 0.0,
                        "max_tokens": 100
                    }
                }
            })
        elif env_type == EnvironmentType.PRODUCTION:
            base_config.update({
                "storage": {
                    "base_path": "/app/data",
                    "output_path": "/app/output"
                },
                "llm": {
                    "model": "gpt-4",
                    "settings": {
                        "temperature": 0.3,
                        "max_tokens": 2000
                    }
                },
                "logging": {
                    "level": "WARNING"
                }
            })
        
        return base_config
    
    def _load_base_config(self):
        """加载基础配置"""
        base_config_file = self.config_dir / "config.json"
        
        if base_config_file.exists():
            try:
                with open(base_config_file, 'r', encoding='utf-8') as f:
                    self.base_config = json.load(f)
                logger.debug("加载基础配置成功")
            except Exception as e:
                logger.warning(f"加载基础配置失败: {e}")
                self.base_config = {}
        else:
            logger.info("未找到基础配置文件，将创建默认配置")
            self.base_config = self._generate_default_base_config()
            
            with open(base_config_file, 'w', encoding='utf-8') as f:
                json.dump(self.base_config, f, indent=2, ensure_ascii=False)
    
    def _generate_default_base_config(self) -> Dict[str, Any]:
        """生成默认基础配置"""
        return {
            "version": "2.0.0",
            "application": {
                "name": "MCP Academic RAG Server",
                "version": "1.0.0"
            },
            "storage": {
                "base_path": "./data",
                "output_path": "./output"
            },
            "processors": {
                "pre_processor": {
                    "enabled": True,
                    "config": {}
                },
                "ocr_processor": {
                    "enabled": True,
                    "config": {}
                },
                "embedding_processor": {
                    "enabled": True,
                    "config": {}
                }
            },
            "llm": {
                "type": "openai",
                "model": "gpt-3.5-turbo",
                "settings": {
                    "temperature": 0.7,
                    "max_tokens": 2000
                }
            },
            "vector_db": {
                "document_store": {
                    "type": "memory",
                    "embedding_dim": 1536,
                    "similarity": "cosine"
                }
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
    
    def _get_default_environment(self) -> Optional[str]:
        """获取默认环境"""
        # 优先级：development > testing > 第一个可用环境
        priority_order = [
            EnvironmentType.DEVELOPMENT.value,
            EnvironmentType.TESTING.value
        ]
        
        for env_name in priority_order:
            if env_name in self.environments:
                return env_name
        
        # 返回第一个可用环境
        return next(iter(self.environments.keys())) if self.environments else None
    
    def list_environments(self) -> List[Dict[str, Any]]:
        """列出所有环境"""
        return [
            {
                'name': env.name,
                'type': env.type.value,
                'description': env.description,
                'config_file': env.config_file,
                'active': env.active,
                'current': env.name == self.current_environment,
                'created_at': env.created_at
            }
            for env in self.environments.values()
        ]
    
    def get_environment_info(self, env_name: str) -> Optional[Dict[str, Any]]:
        """获取环境信息"""
        if env_name not in self.environments:
            return None
        
        env = self.environments[env_name]
        config = self.get_environment_config(env_name)
        
        return {
            'name': env.name,
            'type': env.type.value,
            'description': env.description,
            'config_file': env.config_file,
            'active': env.active,
            'current': env.name == self.current_environment,
            'created_at': env.created_at,
            'config': config,
            'config_size': len(json.dumps(config, default=str)) if config else 0
        }
    
    def set_environment(self, env_name: str) -> bool:
        """设置当前环境"""
        if env_name not in self.environments:
            logger.error(f"环境不存在: {env_name}")
            return False
        
        try:
            # 停用当前环境
            if self.current_environment:
                self.environments[self.current_environment].active = False
            
            # 激活新环境
            self.environments[env_name].active = True
            self.current_environment = env_name
            
            # 清除配置缓存以强制重新加载
            if env_name in self.config_cache:
                del self.config_cache[env_name]
            
            logger.info(f"切换到环境: {env_name}")
            return True
            
        except Exception as e:
            logger.error(f"切换环境失败: {e}")
            return False
    
    def get_environment_config(self, env_name: str = None) -> Optional[Dict[str, Any]]:
        """获取环境配置"""
        if env_name is None:
            env_name = self.current_environment
        
        if not env_name or env_name not in self.environments:
            return None
        
        # 检查缓存
        if env_name in self.config_cache:
            return self.config_cache[env_name]
        
        try:
            # 加载环境配置
            env_info = self.environments[env_name]
            config_file = Path(env_info.config_file)
            
            if not config_file.exists():
                logger.warning(f"环境配置文件不存在: {config_file}")
                return None
            
            with open(config_file, 'r', encoding='utf-8') as f:
                env_config = json.load(f)
            
            # 合并基础配置
            merged_config = self._merge_configs(self.base_config, env_config)
            
            # 缓存配置
            self.config_cache[env_name] = merged_config
            
            return merged_config
            
        except Exception as e:
            logger.error(f"加载环境配置失败 {env_name}: {e}")
            return None
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """合并配置"""
        if self.override_strategy == "replace":
            return override.copy()
        elif self.override_strategy == "merge":
            result = base.copy()
            result.update(override)
            return result
        else:  # deep_merge
            return self._deep_merge_configs(base, override)
    
    def _deep_merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并配置"""
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def create_environment(self, env_name: str, env_type: EnvironmentType = EnvironmentType.CUSTOM,
                          description: str = None, config: Dict[str, Any] = None,
                          parent_env: str = None) -> bool:
        """创建新环境"""
        if env_name in self.environments:
            logger.error(f"环境已存在: {env_name}")
            return False
        
        try:
            # 配置文件路径
            config_file = self.config_dir / f"config.{env_name}.json"
            
            # 生成配置
            if config is None:
                if parent_env and parent_env in self.environments:
                    # 从父环境继承
                    parent_config = self.get_environment_config(parent_env)
                    config = parent_config.copy() if parent_config else {}
                else:
                    # 使用默认配置
                    config = self._generate_default_config(env_type)
            
            # 添加环境标识
            config['environment'] = {
                'name': env_name,
                'type': env_type.value,
                'parent': parent_env
            }
            
            # 保存配置文件
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            # 创建环境信息
            env_info = EnvironmentInfo(
                name=env_name,
                type=env_type,
                description=description or f"{env_name} 环境",
                config_file=str(config_file),
                parent_env=parent_env
            )
            
            self.environments[env_name] = env_info
            
            logger.info(f"创建环境成功: {env_name}")
            return True
            
        except Exception as e:
            logger.error(f"创建环境失败 {env_name}: {e}")
            return False
    
    def delete_environment(self, env_name: str, backup: bool = True) -> bool:
        """删除环境"""
        if env_name not in self.environments:
            logger.error(f"环境不存在: {env_name}")
            return False
        
        if env_name == self.current_environment:
            logger.error(f"不能删除当前活动环境: {env_name}")
            return False
        
        try:
            env_info = self.environments[env_name]
            config_file = Path(env_info.config_file)
            
            # 创建备份
            if backup and config_file.exists():
                backup_file = config_file.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
                config_file.rename(backup_file)
                logger.info(f"环境配置已备份到: {backup_file}")
            elif config_file.exists():
                config_file.unlink()
            
            # 从缓存中删除
            if env_name in self.config_cache:
                del self.config_cache[env_name]
            
            # 删除环境信息
            del self.environments[env_name]
            
            logger.info(f"删除环境成功: {env_name}")
            return True
            
        except Exception as e:
            logger.error(f"删除环境失败 {env_name}: {e}")
            return False
    
    def copy_environment(self, source_env: str, target_env: str, 
                        description: str = None) -> bool:
        """复制环境"""
        if source_env not in self.environments:
            logger.error(f"源环境不存在: {source_env}")
            return False
        
        if target_env in self.environments:
            logger.error(f"目标环境已存在: {target_env}")
            return False
        
        try:
            # 获取源环境配置
            source_config = self.get_environment_config(source_env)
            if not source_config:
                logger.error(f"无法获取源环境配置: {source_env}")
                return False
            
            # 创建新环境
            source_env_info = self.environments[source_env]
            return self.create_environment(
                env_name=target_env,
                env_type=source_env_info.type,
                description=description or f"从 {source_env} 复制",
                config=source_config
            )
            
        except Exception as e:
            logger.error(f"复制环境失败 {source_env} → {target_env}: {e}")
            return False
    
    def update_environment_config(self, env_name: str, config_updates: Dict[str, Any],
                                 merge_strategy: str = None) -> bool:
        """更新环境配置"""
        if env_name not in self.environments:
            logger.error(f"环境不存在: {env_name}")
            return False
        
        try:
            # 获取当前配置
            current_config = self.get_environment_config(env_name)
            if not current_config:
                logger.error(f"无法获取环境配置: {env_name}")
                return False
            
            # 应用更新
            strategy = merge_strategy or self.override_strategy
            if strategy == "replace":
                new_config = config_updates.copy()
            elif strategy == "merge":
                new_config = current_config.copy()
                new_config.update(config_updates)
            else:  # deep_merge
                new_config = self._deep_merge_configs(current_config, config_updates)
            
            # 保存配置
            env_info = self.environments[env_name]
            config_file = Path(env_info.config_file)
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(new_config, f, indent=2, ensure_ascii=False)
            
            # 清除缓存
            if env_name in self.config_cache:
                del self.config_cache[env_name]
            
            logger.info(f"更新环境配置成功: {env_name}")
            return True
            
        except Exception as e:
            logger.error(f"更新环境配置失败 {env_name}: {e}")
            return False
    
    def validate_environment(self, env_name: str) -> Tuple[bool, List[str]]:
        """验证环境配置"""
        if env_name not in self.environments:
            return False, [f"环境不存在: {env_name}"]
        
        try:
            config = self.get_environment_config(env_name)
            if not config:
                return False, ["无法加载环境配置"]
            
            errors = []
            
            # 基础结构验证
            required_sections = ['storage', 'processors', 'llm']
            for section in required_sections:
                if section not in config:
                    errors.append(f"缺少必需配置段: {section}")
            
            # 路径验证
            if 'storage' in config:
                storage_config = config['storage']
                for path_key in ['base_path', 'output_path']:
                    if path_key in storage_config:
                        path_value = storage_config[path_key]
                        if not isinstance(path_value, str):
                            errors.append(f"存储路径必须是字符串: {path_key}")
            
            # 处理器配置验证
            if 'processors' in config:
                for proc_name, proc_config in config['processors'].items():
                    if not isinstance(proc_config, dict):
                        errors.append(f"处理器配置必须是对象: {proc_name}")
                    elif 'enabled' not in proc_config:
                        errors.append(f"处理器缺少enabled字段: {proc_name}")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            return False, [f"验证过程异常: {str(e)}"]
    
    def get_environment_diff(self, env1: str, env2: str) -> Optional[Dict[str, Any]]:
        """比较两个环境的配置差异"""
        if env1 not in self.environments or env2 not in self.environments:
            return None
        
        try:
            config1 = self.get_environment_config(env1)
            config2 = self.get_environment_config(env2)
            
            if not config1 or not config2:
                return None
            
            return self._calculate_config_diff(config1, config2, env1, env2)
            
        except Exception as e:
            logger.error(f"比较环境配置失败 {env1} vs {env2}: {e}")
            return None
    
    def _calculate_config_diff(self, config1: Dict[str, Any], config2: Dict[str, Any],
                              env1_name: str, env2_name: str, path: str = "") -> Dict[str, Any]:
        """计算配置差异"""
        diff = {
            'added': [],      # env2中新增的
            'removed': [],    # env1中删除的
            'modified': [],   # 修改的值
            'summary': {}
        }
        
        all_keys = set(config1.keys()) | set(config2.keys())
        
        for key in all_keys:
            current_path = f"{path}.{key}" if path else key
            
            if key not in config1:
                # env2新增
                diff['added'].append({
                    'path': current_path,
                    'value': config2[key]
                })
            elif key not in config2:
                # env1删除
                diff['removed'].append({
                    'path': current_path,
                    'value': config1[key]
                })
            elif config1[key] != config2[key]:
                # 值不同
                if isinstance(config1[key], dict) and isinstance(config2[key], dict):
                    # 递归比较
                    sub_diff = self._calculate_config_diff(
                        config1[key], config2[key], env1_name, env2_name, current_path
                    )
                    diff['added'].extend(sub_diff['added'])
                    diff['removed'].extend(sub_diff['removed'])
                    diff['modified'].extend(sub_diff['modified'])
                else:
                    # 值修改
                    diff['modified'].append({
                        'path': current_path,
                        f'{env1_name}_value': config1[key],
                        f'{env2_name}_value': config2[key]
                    })
        
        # 统计
        diff['summary'] = {
            'total_differences': len(diff['added']) + len(diff['removed']) + len(diff['modified']),
            'added_count': len(diff['added']),
            'removed_count': len(diff['removed']),
            'modified_count': len(diff['modified'])
        }
        
        return diff
    
    def export_environment(self, env_name: str, export_path: str) -> bool:
        """导出环境配置"""
        if env_name not in self.environments:
            logger.error(f"环境不存在: {env_name}")
            return False
        
        try:
            config = self.get_environment_config(env_name)
            env_info = self.environments[env_name]
            
            export_data = {
                'export_info': {
                    'exported_at': datetime.now().isoformat(),
                    'environment_name': env_name,
                    'environment_type': env_info.type.value,
                    'source_description': env_info.description
                },
                'environment_config': config
            }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"导出环境配置成功: {env_name} → {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"导出环境配置失败 {env_name}: {e}")
            return False
    
    def import_environment(self, import_path: str, env_name: str = None) -> Optional[str]:
        """导入环境配置"""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            if 'environment_config' not in import_data:
                logger.error("导入文件格式错误：缺少环境配置")
                return None
            
            config = import_data['environment_config']
            export_info = import_data.get('export_info', {})
            
            # 确定环境名称
            if env_name is None:
                env_name = export_info.get('environment_name', f"imported_{int(datetime.now().timestamp())}")
            
            # 确定环境类型
            env_type_str = export_info.get('environment_type', 'custom')
            env_type = EnvironmentType.CUSTOM
            for et in EnvironmentType:
                if et.value == env_type_str:
                    env_type = et
                    break
            
            # 创建环境
            description = f"从 {import_path} 导入"
            if export_info.get('source_description'):
                description += f" ({export_info['source_description']})"
            
            if self.create_environment(env_name, env_type, description, config):
                logger.info(f"导入环境配置成功: {env_name}")
                return env_name
            else:
                return None
                
        except Exception as e:
            logger.error(f"导入环境配置失败: {e}")
            return None

# 便捷函数
def create_environment_manager(config_dir: str = "./config") -> ConfigEnvironmentManager:
    """创建环境管理器实例"""
    return ConfigEnvironmentManager(config_dir)

def quick_switch_environment(config_dir: str, env_name: str) -> bool:
    """快速切换环境"""
    manager = ConfigEnvironmentManager(config_dir)
    return manager.set_environment(env_name)