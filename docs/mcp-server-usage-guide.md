# MCP Academic RAG Server Usage Guide

## Overview

The MCP Academic RAG Server is a Model Context Protocol (MCP) server that provides advanced academic document processing and querying capabilities using Retrieval-Augmented Generation (RAG). It enables users to process academic papers, extract content, and perform intelligent queries through natural language.

## Architecture

```
MCP Client → STDIO Transport → MCP Server → RAG Pipeline → OpenAI GPT
             ↓                ↓            ↓
        Tool Calls      Server Context   Document Store
                           ↓                ↓
                    Document Pipeline    Embeddings
```

## Core Components

### 1. MCP Server (`mcp_server.py`)
- **Purpose**: Main MCP server implementation with STDIO transport
- **Transport**: Standard Input/Output for Claude Desktop integration
- **Tools**: 4 primary tools for document processing and querying
- **Error Handling**: Comprehensive error handling with structured logging

### 2. Server Context (`core/server_context.py`)
- **Purpose**: Dependency injection container managing all server components
- **Initialization**: Multi-phase setup with proper dependency ordering
- **Components**: ConfigManager, RAGPipeline, SessionManager, DocumentPipeline

### 3. RAG Pipeline (`rag/haystack_pipeline.py`)
- **Purpose**: Retrieval-Augmented Generation using Haystack 2.x
- **Capabilities**: Semantic search, context building, LLM response generation
- **Integration**: OpenAI GPT-3.5-turbo with sentence-BERT embeddings

## Available Tools

### `process_document`

Process academic documents through OCR, structure extraction, and embedding generation.

**Input Schema:**
```json
{
  "file_path": "string (required) - Path to document file",
  "file_name": "string (optional) - Display name for document"
}
```

**Example Usage:**
```json
{
  "file_path": "/Users/researcher/papers/neural_networks.pdf",
  "file_name": "Neural Networks Survey 2024"
}
```

**Output:**
```json
{
  "status": "success",
  "document_id": "abc123def",
  "file_name": "neural_networks.pdf",
  "processing_stages": ["raw", "structured", "embedded"],
  "metadata": {
    "pages": 12,
    "file_type": ".pdf",
    "processing_time": 4.2
  },
  "message": "Document processed successfully"
}
```

### `query_documents`

Query processed documents using RAG pipeline with natural language.

**Input Schema:**
```json
{
  "query": "string (required) - Research question or query",
  "session_id": "string (optional) - Session ID for conversation context",
  "top_k": "integer (optional) - Max documents to retrieve (default: 5)"
}
```

**Example Usage:**
```json
{
  "query": "What are the latest developments in transformer architectures?",
  "session_id": "research_session_1",
  "top_k": 3
}
```

**Output:**
```json
{
  "status": "success",
  "session_id": "research_session_1",
  "query": "What are the latest developments in transformer architectures?",
  "answer": "Recent transformer developments include...",
  "sources": [
    {
      "content": "Abstract content from relevant paper...",
      "metadata": {
        "document_id": "abc123",
        "file_name": "transformers_survey.pdf",
        "page": 3
      }
    }
  ]
}
```

### `get_document_info`

Retrieve information about a processed document.

**Input Schema:**
```json
{
  "document_id": "string (required) - ID of processed document"
}
```

**Example Usage:**
```json
{
  "document_id": "abc123def"
}
```

### `list_sessions`

List all active chat sessions with metadata.

**Input Schema:**
```json
{}
```

**Output:**
```json
{
  "status": "success",
  "sessions": [
    {
      "session_id": "research_session_1",
      "created_at": "2024-01-15T10:30:00Z",
      "last_active_at": "2024-01-15T11:45:00Z",
      "message_count": 5,
      "metadata": {}
    }
  ],
  "total_count": 1
}
```

## Environment Setup

### Required Environment Variables

The server supports multiple AI providers. Choose one:

```bash
# Option 1: OpenAI (default provider)
export OPENAI_API_KEY="sk-your-api-key-here"

# Option 2: Anthropic Claude 
export ANTHROPIC_API_KEY="sk-ant-your-api-key-here"
export LLM_PROVIDER="anthropic"
export LLM_MODEL="claude-3-sonnet-20240229"

# Option 3: Google Gemini
export GOOGLE_API_KEY="your-google-ai-key-here"  
export LLM_PROVIDER="google"
export LLM_MODEL="gemini-1.5-pro"
```

**📚 For detailed model comparison and setup:** [Multi-Model Setup Guide](multi-model-setup-guide.md)

### Installation Requirements

```bash
# Core MCP requirements
pip install mcp

# RAG pipeline dependencies
pip install haystack-ai sentence-transformers openai

# Document processing (optional)
pip install pypdf2 python-docx  # PDF and Word support
```

## Configuration

### Basic Configuration (`config/config.json`)

```json
{
  "llm": {
    "model": "gpt-3.5-turbo",
    "temperature": 0.7,
    "max_tokens": 1000
  },
  "rag_settings": {
    "retriever_top_k": 5,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
  },
  "processors": [
    {
      "name": "HaystackEmbeddingProcessor",
      "enabled": true
    }
  ]
}
```

### Claude Desktop Integration

Add to Claude Desktop configuration:

```json
{
  "mcpServers": {
    "academic-rag": {
      "command": "python",
      "args": ["/path/to/mcp_server.py"],
      "env": {
        "OPENAI_API_KEY": "your-api-key"
      }
    }
  }
}
```

## Usage Workflows

### 1. Document Processing Workflow

```mermaid
graph LR
    A[Upload Document] --> B[process_document]
    B --> C[OCR/Extraction]
    C --> D[Structure Analysis]
    D --> E[Embedding Generation]
    E --> F[Document Indexed]
```

**Steps:**
1. Call `process_document` with file path
2. Server processes through OCR and structure extraction
3. Content is embedded using sentence-BERT
4. Document becomes searchable via RAG pipeline

### 2. Query Workflow

```mermaid
graph LR
    A[Natural Language Query] --> B[query_documents]
    B --> C[Query Embedding]
    C --> D[Semantic Retrieval]
    D --> E[Context Building]
    E --> F[LLM Generation]
    F --> G[Response with Sources]
```

**Steps:**
1. Submit natural language query via `query_documents`
2. Query is embedded and matched against document store
3. Relevant passages are retrieved and formatted
4. LLM generates contextual response with citations

### 3. Session Management

```python
# Create session for research project
session_result = await query_documents({
    "query": "What is machine learning?",
    "session_id": "ml_research_2024"
})

# Continue conversation with context
followup_result = await query_documents({
    "query": "How does it relate to deep learning?",
    "session_id": "ml_research_2024"  # Same session maintains context
})
```

## Error Handling

### Common Error Types

**Authentication Errors:**
```json
{
  "status": "error",
  "message": "RAG pipeline not initialized",
  "context_status": {
    "rag_enabled": false,
    "error_reason": "No OpenAI API key found"
  }
}
```

**File Processing Errors:**
```json
{
  "status": "error",
  "message": "File not found: /invalid/path.pdf"
}
```

**Query Processing Errors:**
```json
{
  "status": "error",
  "message": "Error querying documents: Rate limit exceeded"
}
```

### Error Recovery Strategies

1. **Missing API Key**: Set `OPENAI_API_KEY` environment variable
2. **File Access**: Verify file path and permissions
3. **Rate Limits**: Implement exponential backoff in client
4. **Pipeline Errors**: Check server logs in stderr output

## Performance Characteristics

| Operation | Typical Time | Memory Usage | Rate Limits |
|-----------|--------------|--------------|-------------|
| **Document Processing** | 5-30 seconds | ~200MB peak | File size dependent |
| **Query Execution** | 3-5 seconds | ~50MB | OpenAI API limits |
| **Session Management** | <100ms | ~10MB per session | Memory bounded |
| **Embedding Generation** | 1-3 seconds | ~100MB | CPU dependent |

## Advanced Configuration

### Custom Embedding Models

```json
{
  "rag_settings": {
    "embedding_model": "sentence-transformers/all-mpnet-base-v2",
    "retriever_top_k": 10
  }
}
```

**Available Models:**
- `all-MiniLM-L6-v2`: Fast, 384-dim (default)
- `all-mpnet-base-v2`: Accurate, 768-dim
- `multi-qa-MiniLM-L6-cos-v1`: Optimized for Q&A

### Custom LLM Configuration

```json
{
  "llm": {
    "model": "gpt-4",
    "temperature": 0.3,
    "max_tokens": 2000,
    "presence_penalty": 0.1
  }
}
```

### Document Processing Customization

```json
{
  "processors": [
    {
      "name": "HaystackEmbeddingProcessor",
      "enabled": true,
      "chunk_size": 500,
      "chunk_overlap": 50
    }
  ]
}
```

## Security Considerations

### API Key Management
- Store API keys in environment variables, not configuration files
- Use different API keys for development and production
- Monitor API usage and implement rate limiting

### File Access
- Validate file paths to prevent directory traversal
- Implement file size limits to prevent resource exhaustion
- Consider sandboxing for untrusted documents

### Network Security
- Use HTTPS for all external API calls
- Implement timeout and retry mechanisms
- Monitor for unusual API usage patterns

## Troubleshooting

### Server Won't Start
```bash
# Check Python path and dependencies
python -c "import mcp; print('MCP available')"
python -c "from haystack import Pipeline; print('Haystack available')"

# Verify environment
echo $OPENAI_API_KEY
```

### RAG Pipeline Issues
```python
# Debug document store
print(f"Documents indexed: {rag_pipeline.document_store.count_documents()}")

# Test query embedding
embedder = SentenceTransformersTextEmbedder()
result = embedder.run("test query")
print(f"Embedding shape: {result['embedding'].shape}")
```

### Performance Optimization
- Use faster embedding models for development
- Implement document caching for frequently accessed files
- Monitor memory usage with large document collections
- Consider batch processing for multiple documents

## Integration Examples

### Python Client
```python
import asyncio
from mcp.client.stdio import stdio_client

async def process_and_query():
    async with stdio_client("python", ["mcp_server.py"]) as client:
        # Process document
        process_result = await client.call_tool(
            "process_document",
            {"file_path": "/path/to/paper.pdf"}
        )
        
        # Query document
        query_result = await client.call_tool(
            "query_documents",
            {"query": "What is the main contribution of this paper?"}
        )
        
        print(query_result)

asyncio.run(process_and_query())
```

### REST API Wrapper
```python
from fastapi import FastAPI
from mcp.client.stdio import stdio_client

app = FastAPI()

@app.post("/process")
async def process_document(file_path: str):
    async with stdio_client("python", ["mcp_server.py"]) as client:
        return await client.call_tool("process_document", {"file_path": file_path})

@app.post("/query")
async def query_documents(query: str):
    async with stdio_client("python", ["mcp_server.py"]) as client:
        return await client.call_tool("query_documents", {"query": query})
```

## Best Practices

1. **Initialization**: Always call server initialization once during startup
2. **Error Handling**: Check tool response status before processing results
3. **Session Management**: Use consistent session IDs for related queries
4. **Resource Management**: Monitor memory usage with large document collections
5. **API Limits**: Implement rate limiting and exponential backoff
6. **Logging**: Monitor stderr output for server logs and diagnostics
7. **Testing**: Test with sample documents before processing important files
8. **Validation**: Validate file paths and query parameters before API calls

## Support and Monitoring

### Logging Configuration
```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Health Check
```python
async def health_check():
    try:
        result = await client.call_tool("list_sessions", {})
        return {"status": "healthy", "server_responsive": True}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

### Performance Monitoring
- Monitor API response times
- Track document processing success rates
- Monitor memory and CPU usage
- Set up alerting for API rate limit issues