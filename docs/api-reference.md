# API Reference

🎆 **Status**: PRODUCTION READY - All APIs validated and deployed  
🚀 **Server**: Running and accepting MCP connections  
📋 **Tools**: 4 core MCP tools fully operational  

Comprehensive API reference for the MCP Academic RAG Server. This document provides detailed specifications for the four core MCP tools and their production-ready implementation.

## Overview

The MCP Academic RAG Server implements a focused, production-ready set of tools for document processing and retrieval-augmented generation through the Model Context Protocol. All tools adhere to MCP 1.0 specification standards and have been validated in production deployment.

---

## Server Specification

### Server Information

| Property | Value |
|----------|-------|
| Server Name | `academic-rag-server` |
| Protocol Version | MCP 1.0 |
| Default Port | 8000 |
| Transport | Standard I/O, TCP |

### Deployment Status 📊

**Current Status**: ✅ PRODUCTION READY  
**Validation**: All deployment methods tested  
**Health Check**: All systems operational  

#### Primary Deployment: uvx (Production Ready) ⭐
```bash
# Immediate deployment (validated)
uvx --from . mcp-academic-rag-server

# With environment configuration
OPENAI_API_KEY=your_key uvx --from . mcp-academic-rag-server
```

#### Container Deployment: Docker (Production Ready)
```bash
# Production container deployment
docker-compose -f docker-compose.simple.yml up -d

# Monitor deployment health
docker-compose logs -f mcp-academic-rag-server
```

#### Connection Verification
```bash
# Verify deployment health
python health_check.py

# Test MCP tool functionality
python test_mcp_core.py
```

---

## Core MCP Tools

### 1. process_document

Processes academic documents through OCR, text extraction, and vector indexing.

**Method**: MCP Tool Call

**Input Schema**:
```json
{
  "type": "object",
  "properties": {
    "file_path": {
      "type": "string",
      "description": "Absolute path to document file"
    },
    "file_name": {
      "type": "string",
      "description": "Optional display name for document"
    }
  },
  "required": ["file_path"],
  "additionalProperties": false
}
```

**Response Schema**:
```json
{
  "type": "object",
  "properties": {
    "status": {
      "type": "string",
      "enum": ["success", "error"]
    },
    "document_id": {
      "type": "string",
      "description": "Unique document identifier"
    },
    "file_name": {"type": "string"},
    "processing_stages": {
      "type": "array",
      "items": {"type": "string"}
    },
    "metadata": {
      "type": "object",
      "properties": {
        "page_count": {"type": "integer"},
        "processing_time": {"type": "number"},
        "file_size": {"type": "string"},
        "language": {"type": "string"}
      }
    },
    "message": {"type": "string"}
  }
}
```

**Example Usage**:
```python
# Process academic document
result = tools.process_document({
    "file_path": "/documents/research_paper.pdf",
    "file_name": "Machine Learning Research Paper"
})
```

**Error Codes**:
- `FileNotFoundError`: File path does not exist
- `UnsupportedFormat`: File format not supported
- `ProcessingTimeout`: Processing exceeded time limit
- `OCRFailureError`: OCR processing failed

---

### 2. query_documents

Executes retrieval-augmented generation queries against processed document corpus.

**Method**: MCP Tool Call

**Input Schema**:
```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "minLength": 1,
      "maxLength": 2000,
      "description": "Natural language query text"
    },
    "session_id": {
      "type": "string",
      "format": "uuid",
      "description": "Optional session context identifier"
    },
    "top_k": {
      "type": "integer",
      "minimum": 1,
      "maximum": 20,
      "default": 5,
      "description": "Number of relevant chunks to retrieve"
    }
  },
  "required": ["query"],
  "additionalProperties": false
}
```

**Response Schema**:
```json
{
  "type": "object",
  "properties": {
    "status": {
      "type": "string",
      "enum": ["success", "error"]
    },
    "session_id": {"type": "string"},
    "query": {"type": "string"},
    "answer": {"type": "string"},
    "sources": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "content": {"type": "string"},
          "metadata": {
            "type": "object",
            "properties": {
              "document_id": {"type": "string"},
              "title": {"type": "string"},
              "page": {"type": "integer"},
              "relevance_score": {"type": "number"},
              "chunk_index": {"type": "integer"}
            }
          }
        }
      }
    },
    "query_metadata": {
      "type": "object",
      "properties": {
        "processing_time": {"type": "number"},
        "llm_model": {"type": "string"},
        "total_tokens": {"type": "integer"}
      }
    }
  }
}
```

**Example Usage**:
```python
# Single query
result = tools.query_documents({
    "query": "What is deep learning?"
})

# Multi-turn conversation with session
result = tools.query_documents({
    "query": "What are its main advantages?",
    "session_id": "session_001",
    "top_k": 3
})
```

---

### 3. get_document_info

Retrieves comprehensive metadata and processing information for specified documents.

**Method**: MCP Tool Call

**Input Schema**:
```json
{
  "type": "object",
  "properties": {
    "document_id": {
      "type": "string",
      "description": "Unique document identifier from process_document response"
    }
  },
  "required": ["document_id"],
  "additionalProperties": false
}
```

**Response Schema**:
```json
{
  "type": "object",
  "properties": {
    "status": {
      "type": "string",
      "enum": ["success", "error"]
    },
    "document_info": {
      "type": "object",
      "properties": {
        "id": {"type": "string"},
        "file_name": {"type": "string"},
        "file_path": {"type": "string"},
        "file_size": {"type": "string"},
        "processing_status": {
          "type": "string",
          "enum": ["completed", "processing", "failed"]
        },
        "created_at": {"type": "string", "format": "date-time"},
        "processed_at": {"type": "string", "format": "date-time"},
        "metadata": {
          "type": "object",
          "properties": {
            "page_count": {"type": "integer"},
            "processing_time": {"type": "number"},
            "language": {"type": "string"},
            "content_type": {"type": "string"},
            "chunk_count": {"type": "integer"}
          }
        }
      }
    }
  }
}
```

**Example Usage**:
```python
# Retrieve document information
result = tools.get_document_info({
    "document_id": "doc_abc123"
})

# Check processing status
if result["document_info"]["processing_status"] == "completed":
    print("Document ready for queries")
```

---

### 4. list_sessions

Returns inventory of active conversation sessions with associated metadata.

**Method**: MCP Tool Call

**Input Schema**:
```json
{
  "type": "object",
  "properties": {},
  "additionalProperties": false
}
```

**Response Schema**:
```json
{
  "type": "object",
  "properties": {
    "status": {
      "type": "string",
      "enum": ["success", "error"]
    },
    "sessions": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "session_id": {"type": "string"},
          "created_at": {"type": "string", "format": "date-time"},
          "last_active_at": {"type": "string", "format": "date-time"},
          "message_count": {"type": "integer"},
          "document_scope": {
            "type": "array",
            "items": {"type": "string"}
          },
          "metadata": {
            "type": "object",
            "properties": {
              "user_context": {"type": "string"},
              "total_queries": {"type": "integer"},
              "avg_response_time": {"type": "number"}
            }
          }
        }
      }
    },
    "total_count": {"type": "integer"},
    "active_sessions": {"type": "integer"},
    "statistics": {
      "type": "object",
      "properties": {
        "oldest_session": {"type": "string"},
        "most_active_session": {"type": "string"},
        "total_queries": {"type": "integer"}
      }
    }
  }
}
```

**Example Usage**:
```python
# List all active sessions
result = tools.list_sessions({})

# Process session data
for session in result["sessions"]:
    print(f"Session {session['session_id']}: {session['message_count']} messages")
```

---

## Error Handling

### Standard Error Response Format

All tools return consistent error responses following this schema:

```json
{
  "type": "object",
  "properties": {
    "status": {
      "type": "string",
      "const": "error"
    },
    "error": {
      "type": "string",
      "description": "Error type identifier"
    },
    "message": {
      "type": "string",
      "description": "Human-readable error description"
    },
    "details": {
      "type": "object",
      "properties": {
        "parameter": {"type": "string"},
        "expected": {"type": "string"},
        "received": {"type": "string"}
      }
    },
    "timestamp": {"type": "string", "format": "date-time"},
    "request_id": {"type": "string"}
  },
  "required": ["status", "error", "message"]
}
```

### Common Error Types

| Error Code | HTTP Equivalent | Description | Resolution Strategy |
|------------|-----------------|-------------|--------------------|
| `ValidationError` | 400 | Invalid input parameters | Verify parameter types and required fields |
| `NotFoundError` | 404 | Resource not found | Check document ID or file path validity |
| `FileNotFoundError` | 404 | File path does not exist | Verify file path and permissions |
| `UnsupportedFormatError` | 400 | File format not supported | Use supported formats (PDF, images) |
| `ProcessingError` | 500 | Internal processing failure | Check document quality, retry operation |
| `ProcessingTimeoutError` | 504 | Operation timeout | Reduce file size or retry |
| `OCRFailureError` | 500 | OCR processing failed | Improve image quality or resolution |
| `SessionNotFoundError` | 404 | Session does not exist | Use valid session ID or create new session |
| `EmptyCorpusError` | 400 | No processed documents | Process documents before querying |
| `QueryTooLongError` | 400 | Query exceeds maximum length | Shorten query text |
| `LLMAPIError` | 502 | Language model API failure | Verify API key and connectivity |
| `RateLimitError` | 429 | Request rate exceeded | Implement exponential backoff |

---

## Configuration Reference

### Server Configuration

```json
{
  "server": {
    "name": "academic-rag-server",
    "version": "1.0.0",
    "port": 8000,
    "host": "localhost"
  }
}
```

### MCP Tools Configuration

```json
{
  "mcp": {
    "protocol_version": "1.0",
    "tools_enabled": [
      "process_document",
      "query_documents",
      "get_document_info",
      "list_sessions"
    ],
    "max_concurrent_requests": 10,
    "request_timeout": 300
  }
}
```

### Storage Configuration

```json
{
  "storage": {
    "type": "local",
    "vector_store": "faiss",
    "base_path": "./data",
    "index_name": "academic_documents",
    "backup_enabled": false,
    "cleanup_policy": {
      "enabled": true,
      "retention_days": 30
    }
  }
}
```

### Processing Configuration

```json
{
  "processing": {
    "max_concurrent_documents": 3,
    "timeout_seconds": 300,
    "ocr": {
      "enabled": true,
      "engine": "tesseract",
      "language": "eng",
      "confidence_threshold": 0.6
    },
    "text_extraction": {
      "chunk_size": 1000,
      "chunk_overlap": 200,
      "min_chunk_size": 100
    },
    "embedding": {
      "model": "sentence-transformers/all-MiniLM-L6-v2",
      "batch_size": 32,
      "device": "cpu"
    }
  }
}
```

### RAG Configuration

```json
{
  "rag": {
    "retrieval": {
      "top_k": 5,
      "similarity_threshold": 0.7,
      "reranking_enabled": false
    },
    "generation": {
      "llm_provider": "openai",
      "model": "gpt-3.5-turbo",
      "temperature": 0.1,
      "max_tokens": 1000,
      "stream_response": false
    },
    "session_management": {
      "max_sessions": 100,
      "session_timeout_minutes": 30,
      "max_messages_per_session": 50
    }
  }
}
```

---

## Performance Characteristics

### Processing Performance

| Operation | Typical Duration | Performance Factors |
|-----------|------------------|-----------------------|
| PDF processing (per page) | 2-4 seconds | Page complexity, text density |
| OCR processing (per page) | 4-8 seconds | Image resolution, text clarity |
| Document indexing | 1-2 seconds | Document size, embedding model |
| Query execution | 1-3 seconds | Corpus size, query complexity |
| Session retrieval | <500ms | Session count, storage backend |

### Resource Utilization

| Resource | Baseline | Peak Usage | Recommended Minimum |
|----------|----------|------------|---------------------|
| Memory | 500MB | 2GB | 2GB RAM |
| CPU | 10% | 80% | 2 CPU cores |
| Disk I/O | Low | High (indexing) | SSD preferred |
| Network | API calls only | Moderate | Stable internet |

### Scalability Limits

| Metric | Recommended Limit | Hard Limit | Notes |
|--------|-------------------|------------|-------|
| Concurrent document processing | 3 documents | 5 documents | Memory intensive |
| Concurrent queries | 10 requests | 20 requests | CPU bound |
| Active sessions | 50 sessions | 100 sessions | Configurable cleanup |
| Document size | 10MB | 50MB | Processing time increases |
| Document pages | 50 pages | 200 pages | OCR performance degrades |
| Corpus size | 1000 documents | 10000 documents | Query performance impact |

### Supported File Formats

| Format | Extensions | Processing Method | Size Limit | Quality Requirements |
|--------|------------|------------------|------------|---------------------|
| PDF | .pdf | Text extraction + OCR fallback | 50MB | Any quality |
| Images | .jpg, .jpeg, .png | Tesseract OCR | 10MB | 300+ DPI recommended |

---

## Health Monitoring

### Health Check Response

```json
{
  "type": "object",
  "properties": {
    "status": {
      "type": "string",
      "enum": ["healthy", "degraded", "unhealthy"]
    },
    "timestamp": {
      "type": "string",
      "format": "date-time"
    },
    "version": {"type": "string"},
    "uptime_seconds": {"type": "integer"},
    "components": {
      "type": "object",
      "properties": {
        "mcp_server": {"type": "string", "enum": ["healthy", "unhealthy"]},
        "document_storage": {"type": "string", "enum": ["healthy", "unhealthy"]},
        "vector_index": {"type": "string", "enum": ["healthy", "unhealthy"]},
        "llm_api": {"type": "string", "enum": ["healthy", "unhealthy"]}
      }
    }
  }
}
```

### System Status Response

```json
{
  "type": "object",
  "properties": {
    "documents_processed": {"type": "integer"},
    "active_sessions": {"type": "integer"},
    "total_queries": {"type": "integer"},
    "memory_usage": {
      "type": "object",
      "properties": {
        "current_mb": {"type": "number"},
        "peak_mb": {"type": "number"},
        "limit_mb": {"type": "number"}
      }
    },
    "disk_usage": {
      "type": "object",
      "properties": {
        "data_size_mb": {"type": "number"},
        "index_size_mb": {"type": "number"},
        "available_space_mb": {"type": "number"}
      }
    },
    "performance_metrics": {
      "type": "object",
      "properties": {
        "avg_processing_time_sec": {"type": "number"},
        "avg_query_time_sec": {"type": "number"},
        "success_rate_percent": {"type": "number"}
      }
    }
  }
}
```

---

## Client Integration

### Python Client Implementation

```python
class MCPAcademicRAGClient:
    """Python client for MCP Academic RAG Server."""
    
    def __init__(self, mcp_tools):
        """
        Initialize client with MCP tools interface.
        
        Args:
            mcp_tools: MCP tools interface object
        """
        self.tools = mcp_tools
    
    def process_document(self, file_path: str, file_name: str = None) -> dict:
        """
        Process document through the server pipeline.
        
        Args:
            file_path: Absolute path to document file
            file_name: Optional display name for document
        
        Returns:
            Processing result with document ID and metadata
        
        Raises:
            FileNotFoundError: If file path does not exist
            UnsupportedFormatError: If file format not supported
        """
        arguments = {"file_path": file_path}
        if file_name:
            arguments["file_name"] = file_name
        return self.tools.process_document(arguments)
    
    def query_documents(self, query: str, session_id: str = None, top_k: int = 5) -> dict:
        """
        Execute RAG query against processed documents.
        
        Args:
            query: Natural language query text
            session_id: Optional session context identifier
            top_k: Number of relevant chunks to retrieve
        
        Returns:
            Query response with generated answer and sources
        
        Raises:
            EmptyCorpusError: If no documents have been processed
            QueryTooLongError: If query exceeds maximum length
        """
        arguments = {
            "query": query,
            "top_k": top_k
        }
        if session_id:
            arguments["session_id"] = session_id
        return self.tools.query_documents(arguments)
    
    def get_document_info(self, document_id: str) -> dict:
        """
        Retrieve document processing information.
        
        Args:
            document_id: Unique document identifier
        
        Returns:
            Document metadata and processing status
        
        Raises:
            NotFoundError: If document ID does not exist
        """
        return self.tools.get_document_info({"document_id": document_id})
    
    def list_sessions(self) -> dict:
        """
        List all active conversation sessions.
        
        Returns:
            Session inventory with metadata and statistics
        """
        return self.tools.list_sessions({})
```

### MCP Client Configuration

#### Claude Desktop Configuration

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "academic-rag": {
      "command": "uvx",
      "args": ["--from", ".", "mcp-academic-rag-server"],
      "cwd": "/absolute/path/to/mcp-academic-rag-server",
      "env": {
        "OPENAI_API_KEY": "your_api_key_here",
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

#### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | OpenAI API authentication key |
| `MCP_PORT` | No | 8000 | Server listening port |
| `DATA_PATH` | No | ./data | Local data storage directory |
| `OCR_LANGUAGE` | No | eng | Tesseract OCR language code |
| `LOG_LEVEL` | No | INFO | Logging verbosity level |

---

## Version Information

### Current Version: 1.0.0

**Release Date**: 2024

**Core Features**:
- Four essential MCP tools for document processing and querying
- Local FAISS vector storage for simplified deployment
- Streamlined configuration management
- uvx package manager compatibility
- Docker deployment support with single-service architecture
- Comprehensive error handling and validation
- Session management for multi-turn conversations

**System Compatibility**:
- **MCP Protocol**: 1.0 or later
- **Python**: 3.9 or later
- **Operating Systems**: Linux, macOS, Windows
- **OpenAI API**: v1 or later
- **Deployment**: uvx, Docker, direct Python execution

**Dependencies**: 14 core packages focused on essential functionality

---

## Design Philosophy

This API reference reflects a focused MCP server implementation designed for reliability and simplicity. The server provides essential document processing and retrieval capabilities through standardized MCP tools, prioritizing ease of integration and maintenance over feature complexity.

All tools adhere to MCP 1.0 specification requirements and provide consistent interfaces for AI assistant integration.

---

*API Reference - MCP Academic RAG Server v1.0.0*