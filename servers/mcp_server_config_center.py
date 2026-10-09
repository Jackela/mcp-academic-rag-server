#!/usr/bin/env python3
"""
MCP Academic RAG Server with ConfigCenter Integration

演示统一配置中心集成的MCP服务器版本。
支持多环境配置、热更新、配置监听等高级功能。
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, Optional

from pydantic import AnyUrl

from core.config_center import ConfigCenter, ConfigChangeEvent, init_config_center
from core.server_context import ServerContext
from servers.mcp_server_sdk import handle_process_document as sdk_process_document
from servers.mcp_server_sdk import handle_query_documents as sdk_query_documents

try:
    import mcp.types as types
    from mcp.server import NotificationOptions, Server
    from mcp.server.models import InitializationOptions
    from mcp.types import Resource, Tool
except ImportError as e:
    print(f"MCP package not found. Please install with: pip install mcp\nError: {e}")
    sys.exit(1)


# Configure structured logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", stream=sys.stderr
)
logger = logging.getLogger("mcp-academic-rag-server-configcenter")

# Global variables
config_center: Optional[ConfigCenter] = None
server_context: Optional[ServerContext] = None
server = Server("academic-rag-server-configcenter")


def _require_center() -> ConfigCenter:
    if config_center is None:
        raise RuntimeError("Configuration center is not initialized")
    return config_center


def on_config_change(event: ConfigChangeEvent) -> None:
    """配置变更处理器"""
    logger.info("配置字段已变更: %s", event.key)

    # 根据配置变更类型执行相应操作
    if event.key.startswith("logging."):
        update_logging_config()
    elif event.key.startswith("llm."):
        update_llm_config()
    elif event.key.startswith("vector_db."):
        update_vector_db_config()


def update_logging_config() -> None:
    """更新日志配置"""
    try:
        log_level = _require_center().get_value("logging.level", "INFO")
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)

        # 更新根日志器级别
        logging.getLogger().setLevel(numeric_level)
        logger.info(f"日志级别已更新为: {log_level}")

    except Exception as e:
        logger.error(f"更新日志配置失败: {str(e)}")


def update_llm_config() -> None:
    """更新LLM配置"""
    try:
        if server_context and hasattr(server_context, "llm_connector"):
            # 这里可以重新初始化LLM连接器
            logger.info("LLM配置变更，建议重启服务器以应用更改")

    except Exception as e:
        logger.error(f"更新LLM配置失败: {str(e)}")


def update_vector_db_config() -> None:
    """更新向量数据库配置"""
    try:
        if server_context and hasattr(server_context, "vector_store"):
            # 这里可以重新初始化向量存储
            logger.info("向量数据库配置变更，建议重启服务器以应用更改")

    except Exception as e:
        logger.error(f"更新向量数据库配置失败: {str(e)}")


def validate_environment() -> bool:
    """验证环境配置"""
    try:
        # 从配置中心获取必需的配置项
        llm_provider = _require_center().get_value("llm.provider", _require_center().get_value("llm.type"))
        if not llm_provider:
            logger.error("未配置LLM提供商")
            return False

        # 检查API密钥
        if llm_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.error("未设置 OPENAI_API_KEY 环境变量")
                return False

        return True

    except Exception as e:
        logger.error(f"环境验证失败: {str(e)}")
        return False


async def initialize_server_context() -> bool:
    """初始化服务器上下文"""
    global server_context

    try:
        server_context = ServerContext(config_manager=_require_center().config_manager)
        server_context.initialize()

        logger.info("服务器上下文初始化完成")
        return True

    except Exception as e:
        logger.error(f"初始化服务器上下文失败: {str(e)}")
        return False


async def handle_list_resources() -> list[Resource]:
    """列出可用资源"""
    try:
        stats = _require_center().get_stats()

        resources = [
            Resource(
                uri=AnyUrl("config://current"),
                name="当前配置",
                description="显示当前生效的完整配置",
                mimeType="application/json",
            ),
            Resource(
                uri=AnyUrl("config://stats"),
                name="配置中心统计",
                description="显示配置中心运行统计信息",
                mimeType="application/json",
            ),
            Resource(
                uri=AnyUrl("config://validation"),
                name="配置验证报告",
                description="显示当前配置的验证结果",
                mimeType="application/json",
            ),
        ]

        # 添加环境配置资源
        for env in stats.get("environments", []):
            resources.append(
                Resource(
                    uri=AnyUrl(f"config://env/{env}"),
                    name=f"{env.title()} 环境配置",
                    description=f"显示 {env} 环境的配置内容",
                    mimeType="application/json",
                )
            )

        return resources

    except Exception as e:
        logger.error(f"列出资源失败: {str(e)}")
        return []


async def handle_read_resource(uri: AnyUrl) -> str:
    """读取资源内容"""
    resource_uri = str(uri)
    try:
        if resource_uri == "config://current":
            config = _require_center().get_config()
            return json.dumps(config, indent=2, ensure_ascii=False)

        elif resource_uri == "config://stats":
            stats = _require_center().get_stats()
            return json.dumps(stats, indent=2, ensure_ascii=False)

        elif resource_uri == "config://validation":
            validation_result = _require_center().validate_current_config()
            return json.dumps(validation_result, indent=2, ensure_ascii=False)

        elif resource_uri.startswith("config://env/"):
            env_name = resource_uri.replace("config://env/", "")
            env_config = _require_center().get_environment_config(env_name)
            if env_config:
                return json.dumps(env_config, indent=2, ensure_ascii=False)
            else:
                return json.dumps({"error": f"环境 '{env_name}' 不存在"}, indent=2)

        else:
            return json.dumps({"error": "未知资源URI"}, indent=2)

    except Exception as e:
        logger.error(f"读取资源失败: {str(e)}")
        return json.dumps({"error": str(e)}, indent=2)


async def handle_list_tools() -> list[Tool]:
    """列出可用工具"""
    return [
        Tool(
            name="process_document",
            description="处理文档并添加到向量存储",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "待处理文档的文件路径"},
                    "file_name": {"type": "string", "description": "可选显示文件名"},
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="query_documents",
            description="查询相关文档",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "查询内容"},
                    "top_k": {"type": "integer", "description": "返回结果数量", "default": 5},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_config_value",
            description="获取配置值",
            inputSchema={
                "type": "object",
                "properties": {"key_path": {"type": "string", "description": "配置键路径，如 'server.port'"}},
                "required": ["key_path"],
            },
        ),
        Tool(
            name="set_config_value",
            description="设置配置值",
            inputSchema={
                "type": "object",
                "properties": {
                    "key_path": {"type": "string", "description": "配置键路径"},
                    "value": {"description": "新的配置值"},
                    "persist": {"type": "boolean", "description": "是否持久化到文件", "default": True},
                },
                "required": ["key_path", "value"],
            },
        ),
        Tool(
            name="switch_environment",
            description="切换配置环境",
            inputSchema={
                "type": "object",
                "properties": {
                    "environment": {"type": "string", "description": "目标环境名称 (development, production等)"}
                },
                "required": ["environment"],
            },
        ),
        Tool(
            name="backup_config",
            description="备份当前配置",
            inputSchema={
                "type": "object",
                "properties": {"backup_path": {"type": "string", "description": "备份文件路径（可选）"}},
            },
        ),
        Tool(
            name="restore_config",
            description="从备份恢复配置",
            inputSchema={
                "type": "object",
                "properties": {"backup_path": {"type": "string", "description": "备份文件路径"}},
                "required": ["backup_path"],
            },
        ),
    ]


async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> list[types.TextContent]:
    """处理工具调用"""
    try:
        if name == "process_document":
            return await handle_process_document(arguments)
        elif name == "query_documents":
            return await handle_query_documents(arguments)
        elif name == "get_config_value":
            return await handle_get_config_value(arguments)
        elif name == "set_config_value":
            return await handle_set_config_value(arguments)
        elif name == "switch_environment":
            return await handle_switch_environment(arguments)
        elif name == "backup_config":
            return await handle_backup_config(arguments)
        elif name == "restore_config":
            return await handle_restore_config(arguments)
        else:
            return [types.TextContent(type="text", text=f"未知工具: {name}")]

    except Exception as e:
        logger.error(f"工具调用失败 {name}: {str(e)}")
        return [types.TextContent(type="text", text=f"工具执行失败: {str(e)}")]


async def handle_process_document(arguments: Dict[str, Any]) -> list[types.TextContent]:
    """Use the current document pipeline contract with the effective configuration."""
    if server_context is None:
        return [types.TextContent(type="text", text="服务器上下文未初始化")]
    return await sdk_process_document(arguments, context=server_context)


async def handle_query_documents(arguments: Dict[str, Any]) -> list[types.TextContent]:
    """Use the current shared-store/session RAG execution path."""
    if server_context is None:
        return [types.TextContent(type="text", text="服务器上下文未初始化")]
    return await sdk_query_documents(arguments, context=server_context)


async def handle_get_config_value(arguments: Dict[str, Any]) -> list[types.TextContent]:
    """获取配置值"""
    key_path = arguments.get("key_path", "")

    try:
        value = _require_center().get_value(key_path)
        if value is not None:
            return [
                types.TextContent(
                    type="text", text=f"配置项 '{key_path}': {json.dumps(value, indent=2, ensure_ascii=False)}"
                )
            ]
        else:
            return [types.TextContent(type="text", text=f"配置项 '{key_path}' 不存在")]

    except Exception as e:
        logger.error(f"获取配置值失败: {str(e)}")
        return [types.TextContent(type="text", text=f"获取配置失败: {str(e)}")]


async def handle_set_config_value(arguments: Dict[str, Any]) -> list[types.TextContent]:
    """设置配置值"""
    key_path = arguments.get("key_path", "")
    value = arguments.get("value")
    persist = arguments.get("persist", True)

    try:
        success = _require_center().set_value(key_path, value, persist=persist)
        if success:
            return [
                types.TextContent(
                    type="text", text=f"配置项 '{key_path}' 已设置为: {json.dumps(value, ensure_ascii=False)}"
                )
            ]
        else:
            return [types.TextContent(type="text", text=f"设置配置项 '{key_path}' 失败")]

    except Exception as e:
        logger.error(f"设置配置值失败: {str(e)}")
        return [types.TextContent(type="text", text=f"设置配置失败: {str(e)}")]


async def handle_switch_environment(arguments: Dict[str, Any]) -> list[types.TextContent]:
    """切换环境"""
    environment = arguments.get("environment", "")

    try:
        old_env = _require_center().environment
        success = _require_center().switch_environment(environment)

        if success:
            return [types.TextContent(type="text", text=f"环境已从 '{old_env}' 切换到 '{environment}'")]
        else:
            return [types.TextContent(type="text", text="环境切换失败")]

    except Exception as e:
        logger.error(f"环境切换失败: {str(e)}")
        return [types.TextContent(type="text", text=f"环境切换失败: {str(e)}")]


async def handle_backup_config(arguments: Dict[str, Any]) -> list[types.TextContent]:
    """备份配置"""
    backup_path = arguments.get("backup_path")

    try:
        path = _require_center().backup_config(backup_path)
        return [types.TextContent(type="text", text=f"配置已备份到: {path}")]

    except Exception as e:
        logger.error(f"配置备份失败: {str(e)}")
        return [types.TextContent(type="text", text=f"配置备份失败: {str(e)}")]


async def handle_restore_config(arguments: Dict[str, Any]) -> list[types.TextContent]:
    """恢复配置"""
    backup_path = arguments.get("backup_path", "")

    try:
        success = _require_center().restore_config(backup_path)
        if success:
            return [types.TextContent(type="text", text=f"配置已从 '{backup_path}' 恢复")]
        else:
            return [types.TextContent(type="text", text="配置恢复失败")]

    except Exception as e:
        logger.error(f"配置恢复失败: {str(e)}")
        return [types.TextContent(type="text", text=f"配置恢复失败: {str(e)}")]


server.list_resources()(handle_list_resources)
server.read_resource()(handle_read_resource)
server.list_tools()(handle_list_tools)
server.call_tool()(handle_call_tool)


async def main() -> None:
    """主函数"""
    global config_center

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="MCP Academic RAG Server with ConfigCenter")
    parser.add_argument("--config-dir", default="./config", help="配置文件目录")
    parser.add_argument("--environment", default="development", help="运行环境")
    parser.add_argument("--no-watch", action="store_true", help="禁用配置文件监听")
    parser.add_argument("--stdio", action="store_true", help="使用STDIO传输")

    args = parser.parse_args()

    try:
        # 初始化配置中心
        logger.info(f"初始化配置中心 - 环境: {args.environment}")
        config_center = init_config_center(
            base_config_path=args.config_dir, environment=args.environment, watch_changes=not args.no_watch
        )

        # 添加配置变更监听器
        _require_center().add_change_listener(on_config_change)

        # 验证环境
        if not validate_environment():
            logger.error("环境验证失败")
            return

        # 初始化服务器上下文
        if not await initialize_server_context():
            logger.error("服务器上下文初始化失败")
            return

        # 应用日志配置
        update_logging_config()

        # 启动服务器
        logger.info(f"启动MCP服务器 - 环境: {_require_center().environment}")

        if args.stdio:
            # STDIO传输模式
            from mcp.server.stdio import stdio_server

            async with stdio_server() as (read_stream, write_stream):
                await server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name="academic-rag-server-configcenter",
                        server_version="1.0.0",
                        capabilities=server.get_capabilities(
                            notification_options=NotificationOptions(),
                            experimental_capabilities={},
                        ),
                    ),
                )
        else:
            # 其他传输模式可以在这里实现
            logger.error("目前仅支持STDIO传输模式")

    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭服务器...")
    except Exception as e:
        logger.error(f"服务器启动失败: {str(e)}")
        raise
    finally:
        # 清理资源
        if config_center:
            _require_center().close()
        if server_context:
            server_context.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
