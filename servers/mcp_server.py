#!/usr/bin/env python3
"""
MCP Academic RAG Server

This MCP server provides tools for academic document processing and querying
using a Retrieval-Augmented Generation (RAG) pipeline.
"""

import asyncio
import logging
import sys
import os
import json
from typing import Dict, Any, List, Optional
import uuid
import time

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from mcp.server.models import InitializationOptions
    from mcp.server import NotificationOptions, Server
    from mcp.types import Resource, Tool, TextContent, ImageContent, EmbeddedResource
    import mcp.types as types
except ImportError as e:
    print(f"MCP package not found. Please install with: pip install mcp\nError: {e}")
    sys.exit(1)

from core.server_context import ServerContext
from models.document import Document

# Configure structured logging (MCP best practice: never write to stdout in STDIO mode)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr  # Critical: Always use stderr for STDIO transport
)
logger = logging.getLogger("mcp-academic-rag-server")

# Server context (dependency injection container)
server_context = ServerContext()

# Server instance
server = Server("academic-rag-server")


def validate_environment() -> bool:
    """Validate required environment variables and system requirements"""
    required_vars = ['OPENAI_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}. "
                      "Some functionality may be limited.")
        return False
    
    logger.info("Environment validation passed", extra={'validated_vars': required_vars})
    return True

def initialize_system() -> None:
    """Initialize the academic RAG system using dependency injection"""
    # Validate environment first
    validate_environment()
    
    try:
        # Initialize server context with all dependencies
        server_context.initialize()
        
        logger.info(
            "Academic RAG system initialized successfully",
            extra=server_context.get_status()
        )
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {str(e)}")
        logger.debug(f"Server context status: {server_context.get_status()}")
        raise

@server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """
    List available tools for the MCP client.
    """
    return [
        Tool(
            name="process_document",
            description="Process an academic document through OCR, structure extraction, and embedding generation",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the document file to process"
                    },
                    "file_name": {
                        "type": "string", 
                        "description": "Name of the document file"
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
                        "description": "The question or query to ask about the documents"
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Optional session ID to maintain conversation context"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of relevant document chunks to retrieve (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_document_info",
            description="Get information about a processed document",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "ID of the document to get information about"
                    }
                },
                "required": ["document_id"]
            }
        ),
        Tool(
            name="list_sessions",
            description="List all chat sessions",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
    """
    Handle tool calls from the MCP client with enhanced error handling and resource management.
    """
    # Generate request context for tracing
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    logger.info(f"Handling tool call: {name}", 
               extra={'request_id': request_id, 'tool': name, 'args_keys': list(arguments.keys())})
    
    # Tool dispatch map for better performance
    tool_handlers = {
        "process_document": process_document,
        "query_documents": query_documents,
        "get_document_info": get_document_info,
        "list_sessions": list_sessions
    }
    
    try:
        # Validate tool name early
        if name not in tool_handlers:
            raise ValueError(f"Unknown tool: {name}")
        
        # Validate arguments structure
        if not isinstance(arguments, dict):
            raise ValueError(f"Invalid arguments format for tool {name}")
        
        # Execute tool with timeout protection
        handler = tool_handlers[name]
        try:
            result = await asyncio.wait_for(handler(arguments), timeout=300.0)  # 5-minute timeout
            return result
        except asyncio.TimeoutError:
            raise ValueError(f"Tool {name} timed out after 5 minutes")
    
    except ValueError as e:
        # Client error - log as warning, not error
        duration = time.time() - start_time
        logger.warning(f"Client error for tool {name}: {str(e)}", 
                      extra={'request_id': request_id, 'tool': name, 'duration': duration, 'error_type': 'client'})
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]
    
    except Exception as e:
        # Server error - log as error with full context
        duration = time.time() - start_time
        logger.error(f"Server error calling tool {name}: {str(e)}", 
                    extra={'request_id': request_id, 'tool': name, 'duration': duration, 'error_type': 'server'}, 
                    exc_info=True)
        return [types.TextContent(type="text", text=f"Internal server error: {str(e)}")]
    
    finally:
        duration = time.time() - start_time
        logger.info(f"Tool call completed: {name}", 
                   extra={'request_id': request_id, 'tool': name, 'duration': duration})

async def process_document(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """
    Process a document through the processing pipeline.
    """
    file_path = arguments.get("file_path")
    file_name = arguments.get("file_name", os.path.basename(file_path) if file_path else "unknown")
    
    logger.info(f"Starting document processing", 
               extra={'file_path': file_path, 'file_name': file_name})
    
    if not file_path:
        logger.error("No file path provided")
        return [types.TextContent(type="text", text="Error: file_path is required")]
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return [types.TextContent(type="text", text=f"Error: File not found: {file_path}")]
    
    try:
        # Create document object
        document = Document(file_path)
        
        # Process document through pipeline asynchronously
        if server_context.document_pipeline:
            result = await server_context.document_pipeline.process_document(document)
            
            if result.is_successful():
                logger.info(f"Document processed successfully", 
                           extra={'document_id': document.document_id, 'file_name': file_name, 
                                 'stages': list(document.content.keys())})
                response = {
                    "status": "success",
                    "document_id": document.document_id,
                    "file_name": document.file_name,
                    "processing_stages": list(document.content.keys()),
                    "metadata": document.metadata,
                    "message": f"Document '{file_name}' processed successfully"
                }
            else:
                response = {
                    "status": "error",
                    "message": result.get_message(),
                    "error": str(result.get_error()) if result.get_error() else None
                }
        else:
            response = {
                "status": "error",
                "message": "Document processing pipeline not initialized",
                "context_status": server_context.get_status()
            }
        
        return [types.TextContent(type="text", text=json.dumps(response, indent=2))]
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return [types.TextContent(type="text", text=f"Error processing document: {str(e)}")]

async def query_documents(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """
    Query documents using the RAG pipeline.
    """
    query = arguments.get("query")
    session_id = arguments.get("session_id")
    top_k = arguments.get("top_k", 5)
    
    if not query:
        return [types.TextContent(type="text", text="Error: Query is required")]
    
    try:
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
            
            response = {
                "status": "success",
                "session_id": session_id,
                "query": query,
                "answer": result.get("answer", "No answer generated"),
                "sources": [
                    {
                        "content": doc.get("content", "")[:200] + "..." if len(doc.get("content", "")) > 200 else doc.get("content", ""),
                        "metadata": doc.get("metadata", {})
                    }
                    for doc in result.get("documents", [])[:top_k]
                ]
            }
        else:
            response = {
                "status": "error",
                "message": "RAG pipeline not initialized",
                "context_status": server_context.get_status()
            }
        
        return [types.TextContent(type="text", text=json.dumps(response, indent=2))]
        
    except Exception as e:
        logger.error(f"Error querying documents: {str(e)}")
        return [types.TextContent(type="text", text=f"Error querying documents: {str(e)}")]

async def get_document_info(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """
    Get information about a processed document.
    """
    document_id = arguments.get("document_id")
    
    if not document_id:
        return [types.TextContent(type="text", text="Error: Document ID is required")]
    
    try:
        # This is a simplified implementation
        # In a full implementation, you would maintain a document registry
        response = {
            "status": "info",
            "message": f"Document info for ID: {document_id}",
            "note": "Full document registry not implemented in this prototype"
        }
        
        return [types.TextContent(type="text", text=json.dumps(response, indent=2))]
        
    except Exception as e:
        logger.error(f"Error getting document info: {str(e)}")
        return [types.TextContent(type="text", text=f"Error getting document info: {str(e)}")]

async def list_sessions(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """
    List all chat sessions.
    """
    try:
        sessions = server_context.session_manager.get_all_sessions()
        
        session_list = []
        for session_id, session in sessions.items():
            session_info = {
                "session_id": session_id,
                "created_at": session.created_at,
                "last_active_at": session.last_active_at,
                "message_count": len(session.messages),
                "metadata": session.metadata
            }
            session_list.append(session_info)
        
        response = {
            "status": "success",
            "sessions": session_list,
            "total_count": len(session_list)
        }
        
        return [types.TextContent(type="text", text=json.dumps(response, indent=2))]
        
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}")
        return [types.TextContent(type="text", text=f"Error listing sessions: {str(e)}")]

async def main():
    """Main function to run the MCP server."""
    try:
        # Initialize the system
        initialize_system()
        
        # Import and run server
        from mcp.server.stdio import stdio_server
        
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
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())