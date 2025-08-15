# MCP Academic RAG Server - User Guide

Complete user guide for the MCP Academic RAG Server, covering installation, configuration, usage, and advanced features.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Basic Usage](#basic-usage)
5. [Advanced Features](#advanced-features)
6. [API Reference](#api-reference)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

## Quick Start

Get up and running with the MCP Academic RAG Server in under 5 minutes.

### Prerequisites

- Python 3.9 or higher
- 4GB+ RAM recommended
- 10GB+ disk space for document storage

### Quick Installation

```bash
# Install via pip
pip install mcp-academic-rag-server

# Start the server
mcp-rag-server start --config-path ./config

# Access the web interface
open http://localhost:8080
```

### First Document Upload

```bash
# Upload your first document
curl -X POST "http://localhost:8080/api/v1/documents" \
     -H "Authorization: Bearer your-api-key" \
     -F "file=@research_paper.pdf" \
     -F "collection_id=my_papers"

# Query the document
curl -X POST "http://localhost:8080/api/v1/query" \
     -H "Authorization: Bearer your-api-key" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What are the main findings?",
       "collection_id": "my_papers"
     }'
```

## Installation

### Method 1: PyPI Installation (Recommended)

```bash
# Install the latest stable version
pip install mcp-academic-rag-server

# Install with optional dependencies
pip install mcp-academic-rag-server[full]  # All features
pip install mcp-academic-rag-server[ocr]   # OCR support
pip install mcp-academic-rag-server[gpu]   # GPU acceleration
```

### Method 2: Docker Installation

```bash
# Pull the official image
docker pull mcp/academic-rag-server:latest

# Run with docker-compose
curl -O https://raw.githubusercontent.com/mcp/rag-server/main/docker-compose.yml
docker-compose up -d

# Access the server
open http://localhost:8080
```

### Method 3: Source Installation

```bash
# Clone the repository
git clone https://github.com/mcp/academic-rag-server.git
cd academic-rag-server

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Run tests to verify installation
pytest tests/
```

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 4GB | 8GB+ |
| Storage | 10GB | 50GB+ SSD |
| GPU | None | NVIDIA GPU with 4GB+ VRAM |
| OS | Linux, macOS, Windows | Linux preferred |

## Configuration

### Basic Configuration

Create a configuration file `config/config.json`:

```json
{
    "server": {
        "host": "0.0.0.0",
        "port": 8080,
        "debug": false
    },
    "processing": {
        "max_file_size_mb": 100,
        "batch_size": 10,
        "enable_ocr": true
    },
    "vector_store": {
        "type": "faiss",
        "index_type": "flat",
        "dimension": 1536
    },
    "llm": {
        "provider": "openai",
        "model": "gpt-4",
        "api_key": "${OPENAI_API_KEY}",
        "max_tokens": 1000,
        "temperature": 0.7
    },
    "embeddings": {
        "provider": "openai",
        "model": "text-embedding-ada-002",
        "api_key": "${OPENAI_API_KEY}"
    }
}
```

### Environment Variables

Set these environment variables for security:

```bash
# Required
export OPENAI_API_KEY="your-openai-api-key"
export RAG_SECRET_KEY="your-secret-key-for-jwt"

# Optional
export RAG_LOG_LEVEL="INFO"
export RAG_DATA_DIR="/path/to/data"
export RAG_CACHE_DIR="/path/to/cache"
```

### Advanced Configuration

For production deployments, use the configuration center:

```python
from mcp_rag_server import ConfigCenter

# Initialize configuration center
config = ConfigCenter(
    base_config_path="./config",
    environment="production",
    watch_changes=True
)

# Update configuration programmatically
config.update_config("llm.model", "gpt-4-turbo")
config.save_config("Updated model to GPT-4 Turbo")
```

## Basic Usage

### Web Interface

1. **Access the Dashboard**: Navigate to `http://localhost:8080`
2. **Create a Collection**: Click "New Collection" and provide a name
3. **Upload Documents**: Drag and drop files or use the upload button
4. **Query Documents**: Use the search box to ask questions

### Command Line Interface

```bash
# Start the server
mcp-rag-server start

# Create a collection
mcp-rag-server collection create --name "Research Papers" --description "Academic papers collection"

# Upload documents
mcp-rag-server documents upload --collection "Research Papers" --files *.pdf

# Query documents
mcp-rag-server query --collection "Research Papers" --query "What are the latest findings in AI?"

# Check server status
mcp-rag-server status
```

### Python SDK

```python
from mcp_rag_client import RAGClient

# Initialize client
client = RAGClient(
    base_url="http://localhost:8080",
    api_key="your-api-key"
)

# Create collection
collection = client.collections.create(
    name="Research Papers",
    description="Academic research papers"
)

# Upload document
document = client.documents.upload(
    file_path="research_paper.pdf",
    collection_id=collection.id,
    metadata={
        "author": "Dr. Smith",
        "year": 2024,
        "topic": "Machine Learning"
    }
)

# Wait for processing
client.documents.wait_for_processing(document.id)

# Query documents
results = client.query(
    query="What are the main contributions?",
    collection_id=collection.id,
    max_results=5
)

# Display results
for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Text: {result.text}")
    print(f"Source: {result.metadata.get('filename')}")
    print("---")
```

## Advanced Features

### Multi-Collection Queries

Query across multiple document collections:

```python
# Query multiple collections
results = client.query(
    query="Compare machine learning approaches",
    collection_ids=["ml_papers", "ai_research", "datasets"],
    max_results=10,
    enable_cross_collection=True
)
```

### Custom Preprocessing

Configure document preprocessing for better results:

```python
from mcp_rag_server import ProcessorConfig, DocumentProcessor

# Custom preprocessing configuration
config = ProcessorConfig(
    enable_ocr=True,
    ocr_language="eng",
    extract_tables=True,
    extract_images=True,
    chunk_size=1000,
    chunk_overlap=200,
    preprocessing_pipeline=[
        "normalize_whitespace",
        "remove_headers_footers",
        "extract_citations",
        "segment_sections"
    ]
)

# Use custom processor
processor = DocumentProcessor(config)
result = processor.process_document("complex_paper.pdf")
```

### Metadata Filtering

Use metadata to filter search results:

```python
# Query with metadata filters
results = client.query(
    query="machine learning algorithms",
    collection_id="papers",
    filters={
        "year": {"gte": 2020},
        "author": {"contains": "Smith"},
        "topic": {"in": ["ML", "AI", "Deep Learning"]}
    },
    sort_by="year",
    sort_order="desc"
)
```

### Batch Operations

Process multiple documents efficiently:

```python
# Batch upload
file_paths = ["paper1.pdf", "paper2.pdf", "paper3.pdf"]
results = client.documents.batch_upload(
    file_paths=file_paths,
    collection_id="papers",
    metadata_list=[
        {"author": "Smith", "year": 2024},
        {"author": "Jones", "year": 2023},
        {"author": "Brown", "year": 2024}
    ]
)

# Batch query
queries = [
    "What is machine learning?",
    "How does deep learning work?",
    "What are neural networks?"
]
results = client.batch_query(
    queries=queries,
    collection_id="papers"
)
```

### Real-time Processing

Monitor document processing in real-time:

```python
import asyncio
from mcp_rag_client import AsyncRAGClient

async def upload_with_progress():
    async_client = AsyncRAGClient(api_key="your-api-key")
    
    # Upload with progress callback
    def progress_callback(status):
        print(f"Processing: {status.progress}% complete")
        if status.error:
            print(f"Error: {status.error}")
    
    document = await async_client.documents.upload_async(
        file_path="large_document.pdf",
        collection_id="papers",
        progress_callback=progress_callback
    )
    
    return document

# Run async upload
document = asyncio.run(upload_with_progress())
```

## API Reference

### Authentication

All API requests require authentication using API keys:

```bash
# Include API key in header
curl -H "Authorization: Bearer your-api-key" \
     "http://localhost:8080/api/v1/collections"
```

### Core Endpoints

#### Collections

```bash
# List collections
GET /api/v1/collections

# Create collection
POST /api/v1/collections
{
    "name": "Research Papers",
    "description": "Academic research collection",
    "metadata": {"domain": "AI"}
}

# Get collection details
GET /api/v1/collections/{collection_id}

# Update collection
PUT /api/v1/collections/{collection_id}

# Delete collection
DELETE /api/v1/collections/{collection_id}
```

#### Documents

```bash
# Upload document
POST /api/v1/documents
Content-Type: multipart/form-data
file=@document.pdf&collection_id=papers

# List documents
GET /api/v1/documents?collection_id=papers

# Get document details
GET /api/v1/documents/{document_id}

# Update document metadata
PUT /api/v1/documents/{document_id}

# Delete document
DELETE /api/v1/documents/{document_id}
```

#### Queries

```bash
# Execute query
POST /api/v1/query
{
    "query": "What are the main findings?",
    "collection_id": "papers",
    "max_results": 10,
    "include_sources": true
}

# Get query history
GET /api/v1/queries?user_id=user123

# Get query details
GET /api/v1/queries/{query_id}
```

### Response Formats

All API responses follow this structure:

```json
{
    "success": true,
    "data": { /* response data */ },
    "message": "Operation completed successfully",
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_12345"
}
```

Error responses:

```json
{
    "success": false,
    "error": {
        "code": "DOCUMENT_PROCESSING_FAILED",
        "message": "Unable to process document",
        "details": { /* error details */ }
    },
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_12345"
}
```

## Troubleshooting

### Common Issues

#### Document Upload Fails

**Problem**: Document upload returns "Unsupported format" error

**Solution**:
```bash
# Check supported formats
curl "http://localhost:8080/api/v1/system/supported-formats"

# Convert document to supported format
pandoc document.docx -o document.pdf
```

#### Slow Query Performance

**Problem**: Queries take too long to complete

**Solutions**:
1. Reduce chunk size in configuration
2. Use more specific queries
3. Enable GPU acceleration
4. Increase server resources

```json
{
    "processing": {
        "chunk_size": 500,
        "enable_gpu": true,
        "batch_size": 20
    }
}
```

#### Memory Issues

**Problem**: Server runs out of memory during processing

**Solutions**:
1. Reduce batch size
2. Increase system memory
3. Enable disk-based storage

```json
{
    "vector_store": {
        "type": "faiss",
        "use_disk_storage": true,
        "memory_limit_gb": 4
    }
}
```

### Error Codes Reference

| Code | Description | Solution |
|------|-------------|----------|
| `AUTH_INVALID` | Invalid API key | Check API key configuration |
| `FILE_TOO_LARGE` | File exceeds size limit | Reduce file size or increase limit |
| `COLLECTION_NOT_FOUND` | Collection doesn't exist | Create collection first |
| `PROCESSING_TIMEOUT` | Processing took too long | Increase timeout or split document |
| `QUOTA_EXCEEDED` | Usage quota exceeded | Upgrade plan or wait for reset |

### Log Analysis

Enable detailed logging for troubleshooting:

```bash
# Set log level
export RAG_LOG_LEVEL=DEBUG

# View logs
tail -f /var/log/mcp-rag-server/server.log

# Search for specific errors
grep "ERROR" /var/log/mcp-rag-server/server.log
```

### Performance Monitoring

Access the monitoring dashboard:

```bash
# Open monitoring dashboard
open http://localhost:8080/monitoring

# Check system metrics
curl "http://localhost:8080/api/v1/system/metrics"
```

## Best Practices

### Document Preparation

1. **Clean Documents**: Remove unnecessary elements before upload
2. **Consistent Formatting**: Use consistent document formats
3. **Metadata**: Add comprehensive metadata for better searchability
4. **File Naming**: Use descriptive file names

### Query Optimization

1. **Specific Queries**: Use specific, well-formed questions
2. **Context**: Provide context in longer queries
3. **Keywords**: Include relevant keywords
4. **Length**: Keep queries concise but descriptive

### Collection Management

1. **Logical Organization**: Group related documents together
2. **Consistent Metadata**: Use consistent metadata schemas
3. **Regular Cleanup**: Remove outdated documents
4. **Backup**: Regularly backup important collections

### Security

1. **API Keys**: Keep API keys secure and rotate regularly
2. **Access Control**: Implement proper user access controls
3. **HTTPS**: Use HTTPS in production
4. **Audit Logs**: Enable audit logging for compliance

### Performance

1. **Resource Monitoring**: Monitor CPU, memory, and disk usage
2. **Caching**: Enable caching for frequently accessed data
3. **Scaling**: Scale horizontally for high loads
4. **Optimization**: Regularly optimize vector indices

### Maintenance

1. **Updates**: Keep the system updated
2. **Backups**: Regular data backups
3. **Monitoring**: Set up alerts for system issues
4. **Documentation**: Keep documentation updated

## Support and Resources

### Getting Help

- **Documentation**: [https://docs.mcp-rag-server.com](https://docs.mcp-rag-server.com)
- **API Reference**: [https://api-docs.mcp-rag-server.com](https://api-docs.mcp-rag-server.com)
- **Community Forum**: [https://community.mcp-rag-server.com](https://community.mcp-rag-server.com)
- **GitHub Issues**: [https://github.com/mcp/rag-server/issues](https://github.com/mcp/rag-server/issues)

### Professional Support

- **Email Support**: support@mcp-rag-server.com
- **Enterprise Support**: Available for premium plans
- **Consulting Services**: Custom implementation assistance
- **Training**: Online and on-site training available

### Community

- **Discord**: Join our developer community
- **Reddit**: r/MCPRAGServer
- **Twitter**: @MCPRAGServer
- **Blog**: Regular updates and tutorials

This user guide provides comprehensive coverage of the MCP Academic RAG Server capabilities. For the latest updates and advanced features, please refer to the official documentation.