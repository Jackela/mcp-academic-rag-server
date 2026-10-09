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

# Public exports remain compatible without importing every optional backend.
from importlib import import_module

_EXPORTS = {
    "ConfigCenter": "config_center",
    "get_config_center": "config_center",
    "init_config_center": "config_center",
    "ConfigManager": "config_manager",
    "ConfigValidator": "config_validator",
    "ServerContext": "server_context",
    "Pipeline": "pipeline",
    "ProcessorLoader": "processor_loader",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value
