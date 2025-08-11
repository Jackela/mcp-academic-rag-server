# MCP Tools Reference Guide

This document provides comprehensive reference information for the four MCP tools implemented by the Academic RAG Server. Each tool follows the Model Context Protocol specification for standardized interaction with MCP-compatible clients.

## Tool Overview

| Tool Name | Purpose | Input Parameters | Response Format |
|-----------|---------|------------------|-----------------|
| [`process_document`](#process_document) | Document processing and indexing | File path, optional metadata | Processing status and document ID |
| [`query_documents`](#query_documents) | Retrieval-augmented generation queries | Query text, session context | Generated response with sources |
| [`get_document_info`](#get_document_info) | Document metadata retrieval | Document identifier | Comprehensive document details |
| [`list_sessions`](#list_sessions) | Session management | None | Active session inventory |

## process_document

Processes academic documents through OCR, text extraction, and vector indexing pipeline.

### Parameters

#### Required Parameters

| Parameter | Type | Description | Validation |
|-----------|------|-------------|------------|
| `file_path` | string | Absolute path to target document | Must exist and be readable |

#### Optional Parameters

| Parameter | Type | Default | Description | Validation |
|-----------|------|---------|-------------|------------|
| `file_name` | string | basename(file_path) | Display name for document | Max 255 characters |

### Request Schema

```json
{
  "type": "object",
  "properties": {
    "file_path": {
      "type": "string",
      "description": "Absolute file system path to document"
    },
    "file_name": {
      "type": "string",
      "description": "Human-readable document identifier"
    }
  },
  "required": ["file_path"],
  "additionalProperties": false
}
```

### Response Schema

#### Success Response

```json
{
  "status": "success",
  "document_id": "string",
  "file_name": "string",
  "processing_stages": ["string"],
  "metadata": {
    "page_count": "integer",
    "processing_time": "number",
    "file_size": "string",
    "language": "string"
  },
  "message": "string"
}
```

#### Error Response

```json
{
  "status": "error",
  "message": "string",
  "error": "string"
}
```

### Supported Formats

| Format | Extensions | Processing Method | Notes |
|--------|------------|------------------|-------|
| PDF | .pdf | Direct text extraction + OCR fallback | Primary supported format |
| Images | .jpg, .jpeg, .png | Tesseract OCR | Requires clear, high-resolution text |

### Error Conditions

| Error Type | Description | Resolution |
|------------|-------------|------------|
| `FileNotFoundError` | Specified file path does not exist | Verify file path and permissions |
| `UnsupportedFormatError` | File format not supported | Use supported formats (PDF, images) |
| `ProcessingTimeoutError` | Processing exceeded time limit | Reduce file size or retry |
| `OCRFailureError` | OCR processing failed | Improve image quality or resolution |

### Usage Examples

```python
# Basic document processing
result = tools.process_document({
    "file_path": "/documents/research_paper.pdf"
})

# With custom display name
result = tools.process_document({
    "file_path": "/documents/survey.pdf",
    "file_name": "Machine Learning Survey 2024"
})
```

## query_documents

Executes retrieval-augmented generation queries against processed document corpus.

### Parameters

#### Required Parameters

| Parameter | Type | Description | Validation |
|-----------|------|-------------|------------|
| `query` | string | Natural language query | 1-2000 characters |

#### Optional Parameters

| Parameter | Type | Default | Description | Validation |
|-----------|------|---------|-------------|------------|
| `session_id` | string | Auto-generated UUID | Session context identifier | Valid UUID format |
| `top_k` | integer | 5 | Number of retrieved document chunks | 1-20 range |

### Request Schema

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
      "description": "Session context identifier"
    },
    "top_k": {
      "type": "integer",
      "minimum": 1,
      "maximum": 20,
      "description": "Number of relevant chunks to retrieve"
    }
  },
  "required": ["query"],
  "additionalProperties": false
}
```

### Response Schema

#### Success Response

```json
{
  "status": "success",
  "session_id": "string",
  "query": "string",
  "answer": "string",
  "sources": [
    {
      "content": "string",
      "metadata": {
        "document_id": "string",
        "title": "string",
        "page": "integer",
        "relevance_score": "number",
        "chunk_index": "integer"
      }
    }
  ],
  "query_metadata": {
    "processing_time": "number",
    "llm_model": "string",
    "total_tokens": "integer"
  }
}
```

### Query Optimization Guidelines

#### Effective Query Patterns

**Factual Information Retrieval**
```
What is the definition of [concept]?
How does [method] work?
What are the key findings of [research area]?
```

**Comparative Analysis**
```
Compare [approach A] and [approach B]
What are the advantages of [method] over [alternative]?
How does [concept] differ from [related concept]?
```

**Procedural Queries**
```
How to implement [algorithm]?
What steps are required for [process]?
What are the prerequisites for [method]?
```

#### Session Management

Sessions maintain conversation context across multiple queries:

```python
# Initialize conversation
response1 = tools.query_documents({
    "query": "What is transformer architecture?"
})

# Continue with context
response2 = tools.query_documents({
    "query": "How does attention mechanism work in this context?",
    "session_id": response1["session_id"]
})
```

### Error Conditions

| Error Type | Description | Resolution |
|------------|-------------|------------|
| `EmptyCorpusError` | No processed documents available | Process documents first |
| `SessionNotFoundError` | Invalid session identifier | Use valid session ID or omit parameter |
| `QueryTooLongError` | Query exceeds maximum length | Shorten query text |
| `LLMAPIError` | Language model API failure | Verify API key and connectivity |

## get_document_info

Retrieves comprehensive metadata and processing information for specified documents.

### Parameters

#### Required Parameters

| Parameter | Type | Description | Validation |
|-----------|------|-------------|------------|
| `document_id` | string | Unique document identifier | Must be valid processed document ID |

### Request Schema

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

### Response Schema

```json
{
  "status": "success",
  "document_info": {
    "id": "string",
    "file_name": "string",
    "file_path": "string",
    "file_size": "string",
    "processing_status": "completed|processing|failed",
    "created_at": "string",
    "processed_at": "string",
    "metadata": {
      "page_count": "integer",
      "processing_time": "number",
      "language": "string",
      "content_type": "string",
      "chunk_count": "integer"
    },
    "processing_stages": {
      "ocr": {
        "status": "completed|failed|skipped",
        "confidence": "number",
        "duration": "number"
      },
      "embedding": {
        "status": "completed|failed",
        "model": "string",
        "dimensions": "integer",
        "duration": "number"
      }
    }
  }
}
```

### Usage Examples

```python
# Retrieve document information
result = tools.get_document_info({
    "document_id": "doc_abc123"
})

# Check processing status
if result["document_info"]["processing_status"] == "completed":
    print("Document ready for queries")
```

## list_sessions

Returns inventory of active conversation sessions with associated metadata.

### Parameters

No input parameters required.

### Request Schema

```json
{
  "type": "object",
  "properties": {},
  "additionalProperties": false
}
```

### Response Schema

```json
{
  "status": "success",
  "sessions": [
    {
      "session_id": "string",
      "created_at": "string",
      "last_active_at": "string",
      "message_count": "integer",
      "document_scope": ["string"],
      "metadata": {
        "user_context": "string",
        "total_queries": "integer",
        "avg_response_time": "number"
      }
    }
  ],
  "total_count": "integer",
  "active_sessions": "integer",
  "statistics": {
    "oldest_session": "string",
    "most_active_session": "string",
    "total_queries": "integer"
  }
}
```

### Usage Examples

```python
# List all sessions
result = tools.list_sessions({})

# Process session data
for session in result["sessions"]:
    print(f"Session {session['session_id']}: {session['message_count']} messages")
```

## Error Handling

### Standard Error Response Format

All tools return consistent error responses following this schema:

```json
{
  "status": "error",
  "error": "ErrorType",
  "message": "Human-readable error description",
  "details": {
    "parameter": "string",
    "expected": "string",
    "received": "string"
  },
  "timestamp": "string",
  "request_id": "string"
}
```

### Common Error Types

| Error Code | HTTP Equivalent | Description | Typical Causes |
|------------|-----------------|-------------|----------------|
| `ValidationError` | 400 | Invalid input parameters | Missing required fields, type mismatches |
| `NotFoundError` | 404 | Resource not found | Invalid document ID, file path |
| `ProcessingError` | 500 | Internal processing failure | OCR failure, embedding generation error |
| `TimeoutError` | 504 | Operation timeout | Large document processing, API timeouts |
| `RateLimitError` | 429 | Request rate exceeded | Too many concurrent requests |

### Error Recovery Strategies

**Transient Errors**
- Implement exponential backoff for retries
- Check service health before retry attempts
- Log errors for monitoring and analysis

**Configuration Errors**
- Validate configuration before operation
- Provide clear error messages with correction guidance
- Implement configuration validation utilities

**Resource Errors**
- Monitor disk space and memory usage
- Implement resource cleanup procedures
- Set appropriate resource limits

## Performance Characteristics

### Processing Time Estimates

| Operation | Typical Duration | Factors |
|-----------|------------------|---------|
| PDF processing (10 pages) | 15-30 seconds | Page complexity, text density |
| OCR processing (10 pages) | 30-60 seconds | Image resolution, text clarity |
| Query execution | 1-3 seconds | Corpus size, query complexity |
| Document info retrieval | <1 second | Database query performance |

### Resource Utilization

| Resource | Baseline | Peak Usage |
|----------|----------|------------|
| Memory | 500MB | 2GB (during processing) |
| CPU | 10% | 80% (during OCR/embedding) |
| Disk I/O | Minimal | High (during indexing) |
| Network | API calls only | Moderate (LLM requests) |

### Scalability Considerations

**Concurrent Operations**
- Document processing: Limit to 3 concurrent operations
- Query handling: No strict limits, scales with available resources
- Session management: Automatic cleanup after inactivity periods

**Storage Requirements**
- Original documents: User-provided files
- Processed data: ~10-20% of original document size
- Vector indices: ~5-10MB per 100 pages processed

## Integration Examples

### Python Client Implementation

```python
class MCPAcademicRAGClient:
    def __init__(self, mcp_tools):
        self.tools = mcp_tools
    
    def process_document(self, file_path, file_name=None):
        """Process a document and return processing results."""
        params = {"file_path": file_path}
        if file_name:
            params["file_name"] = file_name
        return self.tools.process_document(params)
    
    def query_with_session(self, query, session_id=None, top_k=5):
        """Execute query with optional session context."""
        params = {"query": query, "top_k": top_k}
        if session_id:
            params["session_id"] = session_id
        return self.tools.query_documents(params)
    
    def get_document_details(self, document_id):
        """Retrieve comprehensive document information."""
        return self.tools.get_document_info({"document_id": document_id})
    
    def list_active_sessions(self):
        """Get all active conversation sessions."""
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
      "cwd": "/absolute/path/to/mcp-academic-rag-server"
    }
  }
}
```

#### Environment Setup

Required environment variables:

```bash
# Essential configuration
OPENAI_API_KEY=your_openai_api_key
MCP_PORT=8000
DATA_PATH=./data

# Optional configuration
LOG_LEVEL=INFO
OCR_LANGUAGE=eng
```

## Troubleshooting

### Diagnostic Procedures

**Document Processing Issues**

1. Verify file accessibility and format
2. Check available system resources
3. Review processing logs for specific errors
4. Test with smaller documents to isolate issues

**Query Performance Problems**

1. Check document corpus size and complexity
2. Verify OpenAI API key validity and quota
3. Monitor response times and identify bottlenecks
4. Adjust retrieval parameters (top_k, similarity threshold)

**Session Management Issues**

1. Verify session ID format and validity
2. Check session timeout configuration
3. Monitor memory usage for session storage
4. Implement session cleanup procedures

### Debug Mode Operations

Enable detailed logging for troubleshooting:

```bash
LOG_LEVEL=DEBUG python mcp_server.py
```

This enables comprehensive logging including:
- MCP protocol message traces
- Document processing pipeline details
- Query execution timing and results
- Error stack traces and context

### Performance Monitoring

Monitor key metrics for optimal performance:

```python
# Example monitoring implementation
import time
import logging

logger = logging.getLogger(__name__)

def monitor_tool_performance(tool_name, operation):
    """Decorator for monitoring tool performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"{tool_name} completed in {duration:.2f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"{tool_name} failed after {duration:.2f}s: {e}")
                raise
        return wrapper
    return decorator
```