#!/usr/bin/env python3
"""
Minimal MCP Academic RAG Server - for debugging dependency issues
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Configure logging to stderr (MCP requirement)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("mcp-academic-rag-server-minimal")

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
    logger.info("Starting environment validation...")
    
    # Check required API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key or not validate_api_key(api_key):
        logger.error("OPENAI_API_KEY environment variable not found or invalid")
        raise EnvironmentError("OPENAI_API_KEY environment variable is required")
    
    # Setup data directory
    data_path = os.environ.get('DATA_PATH', './data')
    Path(data_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("Environment validation passed")
    return logger

async def minimal_mcp_main():
    """Minimal MCP server entry point"""
    try:
        # Validate environment
        logger = validate_environment()
        
        # Try to import MCP with better error handling
        try:
            from mcp.server.stdio import stdio_server
            from mcp.server.models import InitializationOptions
            from mcp.server import NotificationOptions, Server
            from mcp import types
        except ImportError as e:
            logger.error(f"Failed to import MCP modules: {e}")
            logger.error("Please install MCP: pip install mcp")
            raise
        
        # Create minimal server
        server = Server("academic-rag-server-minimal")
        
        @server.list_tools()
        async def handle_list_tools():
            """List available tools"""
            return [
                {
                    "name": "test_connection",
                    "description": "Test MCP server connection",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            ]
        
        @server.call_tool()
        async def handle_call_tool(name: str, arguments: dict):
            """Handle tool calls"""
            if name == "test_connection":
                return [types.TextContent(
                    type="text", 
                    text="✅ MCP Academic RAG Server connection successful!"
                )]
            else:
                return [types.TextContent(
                    type="text", 
                    text=f"❌ Unknown tool: {name}"
                )]
        
        # Run MCP server
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="academic-rag-server-minimal",
                    server_version="1.0.0-minimal",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    )
                )
            )
    except Exception as e:
        logger.error(f"Minimal MCP server error: {str(e)}", exc_info=True)
        raise

def main():
    """Entry point"""
    if len(sys.argv) > 1 and sys.argv[1] == "--validate-only":
        try:
            validate_environment()
            print("✅ Minimal server environment validation passed", file=sys.stderr)
            sys.exit(0)
        except Exception as e:
            print(f"❌ Validation failed: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            asyncio.run(minimal_mcp_main())
        except Exception as e:
            logger.error(f"Server startup failed: {str(e)}", exc_info=True)
            sys.exit(1)

if __name__ == "__main__":
    main()