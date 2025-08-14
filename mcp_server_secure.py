#!/usr/bin/env python3
"""
MCP Academic RAG Server - Secure Entry Point
Follows MCP 2024 best practices - NO user interaction in STDIO mode
"""

import asyncio
import sys
import os
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_secure_logging():
    """Setup logging that writes to stderr only (MCP best practice)"""
    import logging
    
    # Configure logging to stderr only (never stdout for STDIO transport)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stderr  # Critical: Never write to stdout in STDIO mode
    )
    return logging.getLogger("mcp-academic-rag-server")

def validate_api_key(api_key: str) -> bool:
    """Validate OpenAI API key format (sk- prefix + min 18 chars, no whitespace)"""
    if not api_key or not isinstance(api_key, str):
        return False
    # Check for whitespace
    if api_key != api_key.strip() or ' ' in api_key:
        return False
    # Check prefix and length
    if not api_key.startswith('sk-'):
        return False
    # Minimum total length of 21 (sk- + 18 chars)
    if len(api_key) < 21:
        return False
    return True

def validate_environment():
    """Validate required environment variables (MCP best practice)"""
    logger = setup_secure_logging()
    
    # Check required API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key or not validate_api_key(api_key):
        logger.error(
            "OPENAI_API_KEY environment variable not found or invalid. "
            "MCP servers require API key to be set before starting.",
            extra={
                'required_format': 'sk-xxx...',
                'current_value': 'missing or invalid',
                'setup_help': 'Set environment variable: export OPENAI_API_KEY=sk-xxx...'
            }
        )
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is required. "
            "Set it before starting the MCP server: export OPENAI_API_KEY=sk-xxx..."
        )
    
    # Setup data directory
    data_path = os.environ.get('DATA_PATH', './data')
    Path(data_path).mkdir(parents=True, exist_ok=True)
    
    logger.info(
        "Environment validation passed",
        extra={
            'api_key_status': 'valid',
            'data_path': data_path,
            'log_level': os.environ.get('LOG_LEVEL', 'INFO')
        }
    )
    return logger

async def main():
    """Main entry point with MCP-compliant configuration"""
    parser = argparse.ArgumentParser(description='MCP Academic RAG Server')
    parser.add_argument('--data-path', help='Data storage path (overrides DATA_PATH env var)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       help='Log level (overrides LOG_LEVEL env var)')
    parser.add_argument('--validate-only', action='store_true', 
                       help='Only validate environment and exit')
    
    args = parser.parse_args()
    
    # Setup environment variables from args if provided
    if args.data_path:
        os.environ['DATA_PATH'] = args.data_path
    if args.log_level:
        os.environ['LOG_LEVEL'] = args.log_level
    
    # Validate environment (will raise if invalid)
    logger = validate_environment()
    
    if args.validate_only:
        logger.info("Environment validation completed successfully")
        return
    
    try:
        # Import and initialize the main server
        from mcp_server import initialize_system, server
        from mcp.server.stdio import stdio_server
        from mcp.server.models import InitializationOptions
        from mcp.server import NotificationOptions
        
        logger.info("Starting MCP Academic RAG Server", extra={'mode': 'stdio'})
        
        # Initialize system
        initialize_system()
        
        # Run MCP server with STDIO transport
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="academic-rag-server",
                    server_version="1.0.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    )
                )
            )
    except EnvironmentError as e:
        logger.error(f"Environment error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())