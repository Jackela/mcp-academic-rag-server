#!/usr/bin/env python3
"""
MCP Academic RAG Server - Python SDK Version
Official SDK implementation for better stability and development experience
"""

import asyncio
import os
import sys
import logging
from pathlib import Path
from typing import Any, Dict, List

# MCP SDK imports
from mcp.server.models import InitializationOptions
from mcp.server import Server, NotificationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Project imports
from core.server_context import ServerContext
from models.document import Document

# Configure logging to stderr (MCP requirement)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("mcp-academic-rag-sdk")

# Global server context (dependency injection)
server_context = ServerContext()

def validate_api_key(api_key: str) -> bool:
    """Validate OpenAI API key format"""
    if not api_key or not isinstance(api_key, str):
        return False
    if api_key != api_key.strip() or ' ' in api_key:
        return False
    if not api_key.startswith('sk-'):
        return False
    if len(api_key) < 21:
        return False
    return True

def validate_environment():
    """Validate required environment variables"""
    logger.info("Validating environment...")
    
    # Check API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key or not validate_api_key(api_key):
        logger.error("OPENAI_API_KEY environment variable not found or invalid")
        raise EnvironmentError("OPENAI_API_KEY environment variable is required")
    
    # Setup data directory
    data_path = os.environ.get('DATA_PATH', './data')
    Path(data_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("Environment validation passed")

# Create MCP server
app = Server("academic-rag-sdk")

@app.list_tools()
async def handle_list_tools() -> List[Tool]:
    """List available MCP tools using SDK"""
    logger.info("Listing tools via SDK")
    
    return [
        Tool(
            name="test_connection",
            description="Test MCP server connection and validate environment",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Test message to echo back",
                        "default": "Hello MCP SDK!"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="process_document",
            description="Process an academic document through OCR and analysis pipeline",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the document file to process"
                    },
                    "file_name": {
                        "type": "string",
                        "description": "Optional name for the document"
                    }
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="query_documents",
            description="Query processed documents using RAG (Retrieval-Augmented Generation)",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question to ask about the documents"
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Optional session ID for conversation context"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of relevant document chunks to retrieve",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="validate_system",
            description="Validate system configuration and dependencies",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]

@app.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls using SDK"""
    logger.info(f"Handling tool call: {name} with args: {list(arguments.keys())}")
    
    try:
        if name == "test_connection":
            return await handle_test_connection(arguments)
        elif name == "validate_system":
            return await handle_validate_system(arguments)
        elif name == "process_document":
            return await handle_process_document(arguments)
        elif name == "query_documents":
            return await handle_query_documents(arguments)
        else:
            return [TextContent(
                type="text",
                text=f"❌ Unknown tool: {name}\nAvailable tools: test_connection, validate_system, process_document, query_documents"
            )]
    
    except Exception as e:
        logger.error(f"Tool execution error: {str(e)}", exc_info=True)
        return [TextContent(
            type="text",
            text=f"❌ Error executing tool '{name}': {str(e)}"
        )]

async def handle_test_connection(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle test connection tool"""
    message = arguments.get("message", "Hello MCP SDK!")
    
    response_text = f"""✅ MCP Academic RAG Server (SDK Version) 连接成功!

📩 收到消息: {message}
🕐 服务器时间: {Path().cwd()}
🐍 Python版本: {sys.version.split()[0]}
📁 工作目录: {os.getcwd()}
🔧 MCP SDK: 已激活
🔑 API密钥: {'✅ 已配置' if os.environ.get('OPENAI_API_KEY') else '❌ 未配置'}

🚀 SDK优势:
- 官方维护，稳定可靠
- 自动协议处理
- 完善的类型支持
- 调试工具集成"""

    return [TextContent(type="text", text=response_text)]

async def handle_validate_system(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle system validation tool"""
    validation_results = []
    
    # Check API key
    api_key = os.environ.get('OPENAI_API_KEY')
    api_key_valid = validate_api_key(api_key)
    validation_results.append(f"🔑 API密钥: {'✅ 有效' if api_key_valid else '❌ 无效'}")
    
    # Check data directory
    data_path = os.environ.get('DATA_PATH', './data')
    data_dir_exists = Path(data_path).exists()
    validation_results.append(f"📁 数据目录: {'✅ 存在' if data_dir_exists else '❌ 不存在'} ({data_path})")
    
    # Check server context
    context_status = server_context.get_status()
    validation_results.append(f"🏗️ 服务器上下文: {'✅ 已初始化' if context_status['initialized'] else '❌ 未初始化'}")
    validation_results.append(f"⚙️ 配置管理器: {'✅ 已加载' if context_status['config_loaded'] else '❌ 未加载'}")
    validation_results.append(f"🔄 文档管道: {'✅ 就绪' if context_status['pipeline_ready'] else '❌ 未就绪'}")
    validation_results.append(f"🤖 RAG管道: {'✅ 启用' if context_status['rag_enabled'] else '❌ 禁用'}")
    validation_results.append(f"📊 处理器数量: {context_status['processors_count']}")
    
    # Python environment
    validation_results.append(f"🐍 Python版本: {sys.version.split()[0]}")
    validation_results.append(f"📦 MCP SDK: ✅ 已安装")
    
    response_text = "🔍 系统验证报告:\n\n" + "\n".join(validation_results)
    
    return [TextContent(type="text", text=response_text)]

async def handle_process_document(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle document processing tool"""
    file_path = arguments.get("file_path")
    file_name = arguments.get("file_name", os.path.basename(file_path) if file_path else "unknown")
    
    if not file_path:
        return [TextContent(type="text", text="❌ Error: file_path is required")]
    
    if not os.path.exists(file_path):
        return [TextContent(type="text", text=f"❌ Error: File not found: {file_path}")]
    
    try:
        # Initialize system if needed
        if not server_context.is_initialized:
            server_context.initialize()
        
        # Create document object
        document = Document(file_path)
        
        # Process document through pipeline
        if server_context.document_pipeline:
            result = await server_context.document_pipeline.process_document(document)
            
            if result.is_successful():
                response_text = f"""✅ 文档处理成功!

📄 文件名: {file_name}
🆔 文档ID: {document.document_id}
📝 处理阶段: {', '.join(document.content.keys())}
🏷️ 元数据: {len(document.metadata)} 项

处理完成，文档已准备用于查询。"""
            else:
                response_text = f"""❌ 文档处理失败:

📄 文件名: {file_name}
💬 错误信息: {result.get_message()}
🔍 详细错误: {str(result.get_error()) if result.get_error() else '未知错误'}"""
        else:
            response_text = f"""❌ 文档处理管道未初始化

📄 文件名: {file_name}
🔧 系统状态: {server_context.get_status()}

请检查服务器配置。"""
        
        return [TextContent(type="text", text=response_text)]
        
    except Exception as e:
        logger.error(f"Document processing error: {str(e)}", exc_info=True)
        return [TextContent(type="text", text=f"❌ 文档处理异常: {str(e)}")]

async def handle_query_documents(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle document query tool"""
    query = arguments.get("query")
    session_id = arguments.get("session_id")
    top_k = arguments.get("top_k", 5)
    
    if not query:
        return [TextContent(type="text", text="❌ Error: Query is required")]
    
    try:
        if not server_context.is_initialized:
            server_context.initialize()
        
        if server_context.rag_pipeline:
            # Get or create session
            if session_id:
                session = server_context.session_manager.get_session(session_id)
                if not session:
                    session = server_context.session_manager.create_session(session_id)
                    session.set_rag_pipeline(server_context.rag_pipeline)
            else:
                session = server_context.session_manager.create_session()
                session.set_rag_pipeline(server_context.rag_pipeline)
                session_id = session.session_id
            
            # Execute query
            result = session.query(query)
            
            answer = result.get("answer", "无法生成答案")
            sources = result.get("documents", [])[:top_k]
            
            response_text = f"""🤖 RAG查询结果:

❓ 问题: {query}
💬 答案: {answer}

📚 相关文档片段 (前{len(sources)}个):
"""
            
            for i, doc in enumerate(sources, 1):
                content = doc.get("content", "")
                if len(content) > 150:
                    content = content[:150] + "..."
                response_text += f"\n{i}. {content}"
                
                metadata = doc.get("metadata", {})
                if metadata:
                    response_text += f"\n   📋 元数据: {metadata}"
            
            response_text += f"\n\n🆔 会话ID: {session_id}"
            
        else:
            response_text = f"""❌ RAG管道未就绪

❓ 问题: {query}
🔧 系统状态: {server_context.get_status()}

RAG功能需要完整的文档处理管道。请先处理一些文档。"""
        
        return [TextContent(type="text", text=response_text)]
        
    except Exception as e:
        logger.error(f"Query error: {str(e)}", exc_info=True)
        return [TextContent(type="text", text=f"❌ 查询异常: {str(e)}")]

async def main():
    """Main entry point using MCP SDK"""
    logger.info("Starting MCP Academic RAG Server (SDK Version)")
    
    try:
        # Validate environment
        validate_environment()
        
        # Run server using SDK
        async with stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="academic-rag-sdk",
                    server_version="1.0.0-sdk",
                    capabilities=app.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    )
                )
            )
    except Exception as e:
        logger.error(f"Server error: {str(e)}", exc_info=True)
        sys.exit(1)

def cli_main():
    """CLI entry point for uvx installation"""
    if len(sys.argv) > 1 and sys.argv[1] == "--validate-only":
        try:
            validate_environment()
            print("✅ SDK version environment validation passed", file=sys.stderr)
            sys.exit(0)
        except Exception as e:
            print(f"❌ SDK version validation failed: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        asyncio.run(main())

if __name__ == "__main__":
    cli_main()