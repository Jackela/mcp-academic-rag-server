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

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from mcp.server.models import InitializationOptions
    from mcp.server import NotificationOptions, Server
    from mcp.types import Resource, Tool, TextContent, ImageContent, EmbeddedResource
    import mcp.types as types
except ImportError:
    print("MCP package not found. Please install with: pip install mcp")
    sys.exit(1)

from core.config_manager import ConfigManager
from core.pipeline import Pipeline
from rag.haystack_pipeline import RAGPipeline, RAGPipelineFactory
from rag.chat_session import ChatSessionManager
from connectors.haystack_llm_connector import HaystackLLMConnector
from models.document import Document
from processors.base_processor import IProcessor
import importlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-academic-rag-server")

# Global instances
config_manager = ConfigManager()
document_pipeline = None
rag_pipeline = None
session_manager = ChatSessionManager()

# Server instance
server = Server("academic-rag-server")

def load_processors() -> List[IProcessor]:
    """
    Dynamically load processors based on configuration
    
    Returns:
        List[IProcessor]: List of loaded processor instances
    """
    processors = []
    processor_configs = config_manager.get_value("processors", {})
    
    # Default processor configuration and mapping (same as app.py)
    default_processor_mapping = {
        'pre_processor': {
            'module': 'processors.pre_processor',
            'class': 'PreProcessor'
        },
        'ocr_processor': {
            'module': 'processors.ocr_processor', 
            'class': 'OCRProcessor'
        },
        'structure_processor': {
            'module': 'processors.structure_processor',
            'class': 'StructureProcessor'
        },
        'classification_processor': {
            'module': 'processors.classification_processor',
            'class': 'ClassificationProcessor'
        },
        'format_converter': {
            'module': 'processors.format_converter',
            'class': 'FormatConverterProcessor'
        },
        'embedding_processor': {
            'module': 'processors.haystack_embedding_processor',
            'class': 'HaystackEmbeddingProcessor'
        }
    }
    
    for processor_name, processor_config in processor_configs.items():
        if not processor_config.get("enabled", True):
            logger.info(f"Skipping disabled processor: {processor_name}")
            continue
            
        try:
            # Get module and class name
            if processor_name in default_processor_mapping:
                # Use default mapping
                mapping = default_processor_mapping[processor_name]
                module_path = mapping['module']
                class_name = mapping['class']
            else:
                # Use configuration mapping
                module_path = processor_config.get("module", f"processors.{processor_name}")
                class_name = processor_config.get("class", f"{processor_name.title()}Processor")
            
            # Import module
            logger.info(f"Loading processor: {processor_name} from {module_path}.{class_name}")
            module = importlib.import_module(module_path)
            processor_class = getattr(module, class_name)
            
            # Create processor instance
            processor_init_config = processor_config.get("config", {})
            processor = processor_class(config=processor_init_config)
            processors.append(processor)
            
            logger.info(f"Successfully loaded processor: {processor_name}")
            
        except Exception as e:
            logger.error(f"Failed to load processor {processor_name}: {str(e)}")
            # Continue loading other processors
    
    logger.info(f"Successfully loaded {len(processors)} processors")
    return processors

def initialize_system():
    """Initialize the academic RAG system"""
    global document_pipeline, rag_pipeline
    
    try:
        # Initialize document processing pipeline
        document_pipeline = Pipeline()
        processors = load_processors()
        
        for processor in processors:
            document_pipeline.add_processor(processor)
        
        # Initialize RAG pipeline
        llm_config = config_manager.get_value("llm", {})
        llm_connector = HaystackLLMConnector(config=llm_config)
        
        rag_config = config_manager.get_value("rag_settings", {})
        rag_pipeline = RAGPipelineFactory.create_pipeline(
            llm_connector=llm_connector,
            config=rag_config
        )
        
        logger.info("Academic RAG system initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {str(e)}")
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
    Handle tool calls from the MCP client.
    """
    try:
        if name == "process_document":
            return await process_document(arguments)
        elif name == "query_documents":
            return await query_documents(arguments)
        elif name == "get_document_info":
            return await get_document_info(arguments)
        elif name == "list_sessions":
            return await list_sessions(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    except Exception as e:
        error_msg = f"Error calling tool {name}: {str(e)}"
        logger.error(error_msg)
        return [types.TextContent(type="text", text=error_msg)]

async def process_document(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """
    Process a document through the processing pipeline.
    """
    file_path = arguments.get("file_path")
    file_name = arguments.get("file_name", os.path.basename(file_path))
    
    if not os.path.exists(file_path):
        return [types.TextContent(type="text", text=f"Error: File not found: {file_path}")]
    
    try:
        # Create document object
        document = Document(file_path)
        
        # Process document through pipeline asynchronously
        if document_pipeline:
            result = await document_pipeline.process_document(document)
            
            if result.is_successful():
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
                "message": "Document processing pipeline not initialized"
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
        if rag_pipeline:
            # Get or create session
            if session_id:
                session = session_manager.get_session(session_id)
                if not session:
                    session = session_manager.create_session(session_id)
                    session.set_rag_pipeline(rag_pipeline)
            else:
                session = session_manager.create_session()
                session.set_rag_pipeline(rag_pipeline)
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
                "message": "RAG pipeline not initialized"
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
        sessions = session_manager.get_all_sessions()
        
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