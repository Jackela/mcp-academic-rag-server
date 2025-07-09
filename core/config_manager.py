"""
配置管理器类，负责加载、访问和管理系统配置。

该模块提供了ConfigManager类，用于处理系统配置的加载、访问、修改和保存。
它支持多级嵌套配置项的访问和修改。
"""

import json
import os
import logging
from typing import Dict, Any, Optional


class ConfigManager:
    """
    配置管理器类，负责加载、访问和管理系统配置。
    
    该类提供方法用于加载配置文件、获取配置值、设置配置值，以及保存配置。
    它支持多级嵌套配置项的访问和修改，通过点号分隔的路径访问配置项。
    
    例如，可以通过"storage.base_path"访问配置中的嵌套项。
    """
    
    def __init__(self, config_path: str):
        """
        初始化ConfigManager对象。
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = {}
        self.logger = logging.getLogger("config_manager")
        
        # 尝试加载配置文件
        self.load_config()
    
    def load_config(self) -> bool:
        """
        从配置文件加载配置。
        
        Returns:
            如果成功加载配置则返回True，否则返回False
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                self.logger.info(f"成功从 {self.config_path} 加载配置")
                return True
            else:
                self.logger.warning(f"配置文件 {self.config_path} 不存在")
                return False
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {str(e)}")
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
