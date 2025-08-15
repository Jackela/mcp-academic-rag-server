"""
Core Module for MCP Academic RAG Server

核心系统组件，提供配置管理、服务器上下文、处理管道等基础功能。

主要组件:
- ConfigCenter: 统一配置中心，支持热更新和多环境
- ConfigManager: 传统配置管理器
- ConfigValidator: 配置验证器
- ServerContext: 服务器运行时上下文
- Pipeline: 文档处理管道
- ProcessorLoader: 处理器动态加载器
"""

from .config_center import ConfigCenter, get_config_center, init_config_center
from .config_manager import ConfigManager
from .config_validator import ConfigValidator
from .server_context import ServerContext
from .pipeline import Pipeline
from .processor_loader import ProcessorLoader

__all__ = [
    "ConfigCenter",
    "get_config_center", 
    "init_config_center",
    "ConfigManager",
    "ConfigValidator",
    "ServerContext",
    "Pipeline",
    "ProcessorLoader"
]