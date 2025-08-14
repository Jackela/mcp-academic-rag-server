"""
MCP Academic RAG Server

A production-ready Model Context Protocol server for academic document processing and RAG queries.
"""

__version__ = "1.0.0"
__author__ = "MCP Academic RAG Team"
__email__ = "noreply@example.com"

# Package metadata
__title__ = "mcp-academic-rag-server"
__description__ = "A focused MCP server for basic academic document processing and RAG queries"
__url__ = "https://github.com/yourusername/mcp-academic-rag-server"
__license__ = "MIT"
__copyright__ = "Copyright 2024 MCP Academic RAG Team"

# Version info
__version_info__ = tuple(int(i) for i in __version__.split('.'))

# Import main components for easy access
try:
    from .mcp_server_secure import cli_main, validate_environment, main
    from .core.server_context import ServerContext
    from .core.config_manager import ConfigManager
except ImportError:
    # Handle case where dependencies are not yet installed
    pass

__all__ = [
    '__version__',
    '__version_info__',
    'cli_main',
    'validate_environment', 
    'main',
    'ServerContext',
    'ConfigManager'
]