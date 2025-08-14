"""
配置管理器类，负责加载、访问和管理系统配置。

该模块提供了ConfigManager类，用于处理系统配置的加载、访问、修改和保存。
它支持多级嵌套配置项的访问和修改，并集成了配置验证功能。
"""

import json
import os
import logging
from typing import Dict, Any, Optional
from .config_validator import ConfigValidator, generate_default_config


class ConfigManager:
    """
    配置管理器类，负责加载、访问和管理系统配置。
    
    该类提供方法用于加载配置文件、获取配置值、设置配置值，以及保存配置。
    它支持多级嵌套配置项的访问和修改，通过点号分隔的路径访问配置项。
    
    例如，可以通过"storage.base_path"访问配置中的嵌套项。
    """
    
    def __init__(self, config_path: str = "./config/config.json"):
        """
        初始化ConfigManager对象。
        
        Args:
            config_path: 配置文件路径，默认为"./config/config.json"
        """
        self.config_path = config_path
        self.config_file_path = config_path  # 添加别名以保持兼容性
        self.config = {}
        self.logger = logging.getLogger("config_manager")
        self.validator = ConfigValidator()
        self._is_validated = False
        
        # 尝试加载配置文件
        self.load_config()
    
    def load_config(self) -> bool:
        """
        从配置文件加载配置，包含验证和标准化。
        
        Returns:
            如果成功加载配置则返回True，否则返回False
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    raw_config = json.load(f)
                
                # 标准化配置（处理器名称等）
                normalized_config = self.validator.normalize_processor_config(raw_config)
                
                # 验证配置
                if self.validator.validate_config(normalized_config):
                    self.config = normalized_config
                    self._is_validated = True
                    self.logger.info(f"Successfully loaded and validated configuration from {self.config_path}", 
                                    extra={'config_path': self.config_path, 'validated': True})
                    
                    # 如果配置被标准化了，保存更新后的配置
                    if raw_config != normalized_config:
                        self.save_config()
                        self.logger.info("Configuration normalized and saved", 
                                        extra={'config_path': self.config_path, 'normalized': True})
                    
                    return True
                else:
                    # 验证失败，但仍然加载配置（允许在修复后继续使用）
                    self.config = normalized_config
                    self._is_validated = False
                    report = self.validator.get_validation_report()
                    self.logger.error(f"Configuration validation failed. Errors: {report['errors']}", 
                                     extra={'config_path': self.config_path, 'error_count': len(report['errors'])})
                    if report['warnings']:
                        self.logger.warning(f"Configuration warnings: {report['warnings']}", 
                                          extra={'config_path': self.config_path, 'warning_count': len(report['warnings'])})
                    return False
            else:
                self.logger.warning(f"配置文件 {self.config_path} 不存在，创建默认配置")
                self.config = generate_default_config()
                self._is_validated = True
                self.save_config()
                return True
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {str(e)}")
            # 使用默认配置作为后备
            self.config = generate_default_config()
            self._is_validated = True
            return False
    
    def save_config(self) -> bool:
        """
        将当前配置保存到配置文件。
        
        Returns:
            如果成功保存配置则返回True，否则返回False
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"成功保存配置到 {self.config_path}")
            return True
        except Exception as e:
            self.logger.error(f"保存配置文件失败: {str(e)}")
            return False
    
    def get_config(self) -> Dict[str, Any]:
        """
        获取完整配置字典。
        
        Returns:
            配置字典
        """
        return self.config.copy()
    
    def get_value(self, key_path: str, default: Any = None) -> Any:
        """
        获取指定键路径的配置值。
        支持使用点号分隔的多级键路径，如"storage.base_path"。
        
        Args:
            key_path: 键路径，使用点号分隔多级键
            default: 如果键不存在，则返回的默认值
            
        Returns:
            配置值，如果键不存在则返回默认值
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys:
            if isinstance(config, dict) and key in config:
                config = config[key]
            else:
                return default
        
        return config
    
    def set_value(self, key_path: str, value: Any) -> bool:
        """
        设置指定键路径的配置值。
        支持使用点号分隔的多级键路径，如"storage.base_path"。
        
        Args:
            key_path: 键路径，使用点号分隔多级键
            value: 要设置的值
            
        Returns:
            如果成功设置则返回True，否则返回False
        """
        keys = key_path.split('.')
        config = self.config
        
        # 对于多级键，遍历到倒数第二级
        for i, key in enumerate(keys[:-1]):
            if key not in config:
                config[key] = {}
            elif not isinstance(config[key], dict):
                # 如果当前级不是字典，则无法继续嵌套
                self.logger.error(f"无法设置配置项 {key_path}，因为 {'.'.join(keys[:i+1])} 不是字典")
                return False
            config = config[key]
        
        # 设置最后一级的值
        config[keys[-1]] = value
        return True
    
    def remove_key(self, key_path: str) -> bool:
        """
        移除指定键路径的配置项。
        
        Args:
            key_path: 键路径，使用点号分隔多级键
            
        Returns:
            如果成功移除则返回True，否则返回False
        """
        keys = key_path.split('.')
        config = self.config
        
        # 对于多级键，遍历到倒数第二级
        for i, key in enumerate(keys[:-1]):
            if key not in config or not isinstance(config[key], dict):
                return False
            config = config[key]
        
        # 移除最后一级的键
        if keys[-1] in config:
            del config[keys[-1]]
            return True
        return False
    
    def get_processor_config(self, processor_name: str) -> Dict[str, Any]:
        """
        获取特定处理器的配置。
        
        Args:
            processor_name: 处理器名称
            
        Returns:
            处理器配置字典，如果不存在则返回空字典
        """
        return self.get_value(f"processors.{processor_name}", {})
    
    def get_connector_config(self, connector_name: str) -> Dict[str, Any]:
        """
        获取特定连接器的配置。
        
        Args:
            connector_name: 连接器名称
            
        Returns:
            连接器配置字典，如果不存在则返回空字典
        """
        return self.get_value(f"connectors.{connector_name}", {})
        
    def reload_config(self) -> bool:
        """
        重新加载配置文件。
        
        Returns:
            如果成功重新加载配置则返回True，否则返回False
        """
        return self.load_config()
    
    def validate_current_config(self) -> bool:
        """
        验证当前配置。
        
        Returns:
            如果配置有效则返回True，否则返回False
        """
        return self.validator.validate_config(self.config)
    
    def get_validation_report(self) -> Dict[str, Any]:
        """
        获取配置验证报告。
        
        Returns:
            包含验证结果的报告字典
        """
        if not self._is_validated:
            self.validate_current_config()
        return self.validator.get_validation_report()
    
    def is_config_valid(self) -> bool:
        """
        检查配置是否有效。
        
        Returns:
            如果配置有效则返回True，否则返回False
        """
        return self._is_validated and len(self.validator.validation_errors) == 0
    
    def fix_config_issues(self) -> bool:
        """
        尝试修复配置问题。
        
        Returns:
            如果成功修复则返回True，否则返回False
        """
        try:
            # 获取默认配置作为参考
            default_config = generate_default_config()
            
            # 合并缺失的必需配置项
            if "storage" not in self.config:
                self.config["storage"] = default_config["storage"]
            
            if "processors" not in self.config:
                self.config["processors"] = default_config["processors"]
            
            # 标准化处理器名称
            self.config = self.validator.normalize_processor_config(self.config)
            
            # 重新验证
            if self.validator.validate_config(self.config):
                self._is_validated = True
                self.save_config()
                self.logger.info("配置问题已修复")
                return True
            else:
                self.logger.error("无法自动修复配置问题")
                return False
                
        except Exception as e:
            self.logger.error(f"修复配置时发生错误: {str(e)}")
            return False
