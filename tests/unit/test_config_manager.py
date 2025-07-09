"""
配置管理器单元测试
"""

import os
import json
import tempfile
import pytest
from core.config_manager import ConfigManager


class TestConfigManager:
    """ConfigManager单元测试类"""

    def setup_method(self):
        """每个测试方法前执行的设置"""
        # 创建一个测试配置
        self.test_config = {
            "app": {
                "name": "test-app",
                "version": "1.0.0"
            },
            "logging": {
                "level": "INFO",
                "file": "logs/app.log"
            },
            "processors": {
                "ocr": {
                    "enabled": True,
                    "api_type": "mistral"
                }
            }
        }
        
        # 创建临时配置文件
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        with open(self.temp_file.name, 'w', encoding='utf-8') as f:
            json.dump(self.test_config, f)
            
        # 创建ConfigManager实例
        self.config_manager = ConfigManager(self.temp_file.name)
        
    def teardown_method(self):
        """每个测试方法后执行的清理"""
        # 删除临时文件
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_load_config(self):
        """测试加载配置"""
        # 验证配置已正确加载
        assert self.config_manager.get_config() == self.test_config
        
    def test_get_value(self):
        """测试获取配置值"""
        # 测试简单路径
        assert self.config_manager.get_value("app.name") == "test-app"
        assert self.config_manager.get_value("logging.level") == "INFO"
        
        # 测试嵌套路径
        assert self.config_manager.get_value("processors.ocr.enabled") is True
        assert self.config_manager.get_value("processors.ocr.api_type") == "mistral"
        
        # 测试不存在的路径
        assert self.config_manager.get_value("app.missing") is None
        assert self.config_manager.get_value("app.missing", "default") == "default"
    
    def test_set_value(self):
        """测试设置配置值"""
        # 设置已存在的值
        self.config_manager.set_value("app.version", "1.1.0")
        assert self.config_manager.get_value("app.version") == "1.1.0"
        
        # 设置新值
        self.config_manager.set_value("app.description", "Test App")
        assert self.config_manager.get_value("app.description") == "Test App"
        
        # 设置嵌套路径的值
        self.config_manager.set_value("processors.ocr.model", "ocr-latest")
        assert self.config_manager.get_value("processors.ocr.model") == "ocr-latest"
        
        # 设置全新嵌套路径的值
        self.config_manager.set_value("new.nested.value", 123)
        assert self.config_manager.get_value("new.nested.value") == 123
    
    def test_remove_value(self):
        """测试删除配置值"""
        # 删除已存在的值
        self.config_manager.remove_value("app.version")
        assert self.config_manager.get_value("app.version") is None
        
        # 删除不存在的值不应抛出异常
        self.config_manager.remove_value("app.missing")
        
        # 删除嵌套路径的值
        self.config_manager.remove_value("processors.ocr.enabled")
        assert self.config_manager.get_value("processors.ocr.enabled") is None
        # 确保父节点仍然存在
        assert self.config_manager.get_value("processors.ocr") is not None
    
    def test_save_config(self):
        """测试保存配置"""
        # 修改配置
        self.config_manager.set_value("app.version", "1.1.0")
        
        # 保存到新文件
        temp_save_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name
        self.config_manager.save_config(temp_save_file)
        
        # 加载新配置并验证
        new_config_manager = ConfigManager(temp_save_file)
        assert new_config_manager.get_value("app.version") == "1.1.0"
        
        # 清理
        if os.path.exists(temp_save_file):
            os.unlink(temp_save_file)
