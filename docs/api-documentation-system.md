# API Documentation System

Comprehensive API documentation for the MCP Academic RAG Server with interactive examples and standardized formats.

## Overview

The MCP Academic RAG Server provides a comprehensive REST API for document processing, RAG queries, and system management. This documentation system provides:

- **Interactive API Documentation**: Auto-generated with Swagger/OpenAPI
- **Code Examples**: Multi-language examples for common operations
- **Authentication Guide**: Complete authentication flow documentation
- **Error Handling**: Comprehensive error codes and responses
- **Rate Limiting**: API limits and best practices

## API Documentation Structure

### Core API Endpoints

#### 1. Document Management API
```python
# Document ingestion endpoint
POST /api/v1/documents
Content-Type: multipart/form-data
Authorization: Bearer <token>

# Parameters:
# - file: Document file (PDF, DOCX, TXT)
# - metadata: Optional document metadata
# - collection_id: Target collection identifier

# Response:
{
    "document_id": "doc_12345",
    "status": "processing",
    "metadata": {
        "filename": "research_paper.pdf",
        "size_bytes": 1024000,
        "mime_type": "application/pdf"
    },
    "processing_time_estimate": "30s"
}
```

#### 2. RAG Query API
```python
# Execute RAG query
POST /api/v1/query
Content-Type: application/json
Authorization: Bearer <token>

{
    "query": "What are the latest findings in machine learning?",
    "collection_id": "research_papers",
    "max_results": 5,
    "include_sources": true,
    "model_config": {
        "temperature": 0.7,
        "max_tokens": 1000
    }
}

# Response:
{
    "query_id": "query_67890",
    "response": "Based on recent research papers...",
    "sources": [
        {
            "document_id": "doc_12345",
            "relevance_score": 0.95,
            "excerpt": "Machine learning advances in...",
            "page_number": 15
        }
    ],
    "processing_time_ms": 1250,
    "model_used": "gpt-4"
}
```

#### 3. Collection Management API
```python
# Create document collection
POST /api/v1/collections
Content-Type: application/json

{
    "name": "Research Papers",
    "description": "Academic research papers collection",
    "vector_config": {
        "embedding_model": "text-embedding-ada-002",
        "dimensions": 1536,
        "similarity_metric": "cosine"
    }
}
```

#### 4. System Status API
```python
# Get system health and metrics
GET /api/v1/system/health
Authorization: Bearer <token>

# Response:
{
    "status": "healthy",
    "version": "1.0.0",
    "uptime_seconds": 86400,
    "metrics": {
        "documents_processed": 1250,
        "queries_handled": 5400,
        "avg_response_time_ms": 450
    },
    "components": {
        "vector_store": "healthy",
        "llm_connector": "healthy",
        "document_processor": "healthy"
    }
}
```

### Authentication

#### API Key Authentication
```bash
# Include API key in header
curl -H "Authorization: Bearer your-api-key" \
     -H "Content-Type: application/json" \
     https://api.rag-server.com/v1/query
```

#### OAuth 2.0 Flow (Enterprise)
```python
# Step 1: Authorization URL
GET /oauth/authorize?client_id=your_client_id&redirect_uri=your_callback&scope=read+write

# Step 2: Token exchange
POST /oauth/token
{
    "grant_type": "authorization_code",
    "code": "auth_code_from_step1",
    "client_id": "your_client_id",
    "client_secret": "your_client_secret"
}
```

### Error Handling

#### Standard Error Response Format
```json
{
    "error": {
        "code": "DOCUMENT_PROCESSING_FAILED",
        "message": "Unable to process document: unsupported format",
        "details": {
            "supported_formats": ["pdf", "docx", "txt", "md"],
            "received_format": "xlsx"
        },
        "request_id": "req_12345",
        "timestamp": "2024-01-15T10:30:00Z"
    }
}
```

#### Common Error Codes
| Code | Status | Description | Solution |
|------|--------|-------------|----------|
| `INVALID_API_KEY` | 401 | API key missing or invalid | Check authentication headers |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests | Implement backoff strategy |
| `DOCUMENT_TOO_LARGE` | 413 | Document exceeds size limit | Split document or compress |
| `UNSUPPORTED_FORMAT` | 400 | File format not supported | Use supported formats |
| `COLLECTION_NOT_FOUND` | 404 | Collection does not exist | Create collection first |
| `PROCESSING_TIMEOUT` | 408 | Document processing timeout | Retry with smaller documents |

### Rate Limiting

#### Default Limits
- **Standard Tier**: 100 requests/minute
- **Premium Tier**: 1000 requests/minute  
- **Enterprise Tier**: Custom limits

#### Rate Limit Headers
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 85
X-RateLimit-Reset: 1642176600
```

### SDKs and Code Examples

#### Python SDK
```python
from mcp_rag_client import RAGClient

client = RAGClient(api_key="your-api-key")

# Upload document
document = client.documents.upload(
    file_path="research_paper.pdf",
    collection_id="research_papers",
    metadata={"author": "Dr. Smith", "year": 2024}
)

# Execute query
result = client.query(
    query="What are the main findings?",
    collection_id="research_papers",
    max_results=5
)
```

#### JavaScript SDK
```javascript
import { RAGClient } from '@mcp/rag-client';

const client = new RAGClient({ apiKey: 'your-api-key' });

// Upload document
const document = await client.documents.upload({
    filePath: 'research_paper.pdf',
    collectionId: 'research_papers',
    metadata: { author: 'Dr. Smith', year: 2024 }
});

// Execute query
const result = await client.query({
    query: 'What are the main findings?',
    collectionId: 'research_papers',
    maxResults: 5
});
```

#### cURL Examples
```bash
# Upload document
curl -X POST "https://api.rag-server.com/v1/documents" \
     -H "Authorization: Bearer your-api-key" \
     -F "file=@research_paper.pdf" \
     -F "collection_id=research_papers"

# Execute query
curl -X POST "https://api.rag-server.com/v1/query" \
     -H "Authorization: Bearer your-api-key" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What are the main findings?",
       "collection_id": "research_papers",
       "max_results": 5
     }'
```

## Webhook Integration

### Event Types
- `document.processed`: Document processing completed
- `document.failed`: Document processing failed
- `query.completed`: Query processing completed
- `collection.updated`: Collection metadata updated

### Webhook Payload Example
```json
{
    "event_type": "document.processed",
    "event_id": "evt_12345",
    "timestamp": "2024-01-15T10:30:00Z",
    "data": {
        "document_id": "doc_12345",
        "collection_id": "research_papers",
        "processing_time_ms": 5000,
        "status": "completed",
        "metadata": {
            "pages": 25,
            "word_count": 8500,
            "language": "en"
        }
    }
}
```

## Performance Optimization

### Best Practices
1. **Batch Operations**: Use batch endpoints for multiple documents
2. **Caching**: Implement client-side caching for frequent queries
3. **Pagination**: Use pagination for large result sets
4. **Compression**: Enable gzip compression for large requests
5. **Connection Pooling**: Reuse HTTP connections

### Monitoring and Analytics
```python
# Get detailed analytics
GET /api/v1/analytics/usage
Authorization: Bearer <token>

# Response includes:
{
    "period": "last_30_days",
    "total_requests": 15000,
    "avg_response_time_ms": 450,
    "error_rate": 0.02,
    "top_queries": ["machine learning", "data science"],
    "usage_by_endpoint": {
        "/api/v1/query": 12000,
        "/api/v1/documents": 2500,
        "/api/v1/collections": 500
    }
}
```

## Interactive Documentation

### Swagger UI Integration
The API documentation is available at `https://api.rag-server.com/docs` with:
- Interactive endpoint testing
- Request/response examples
- Schema validation
- Authentication testing

### Postman Collection
Download the complete Postman collection: `https://api.rag-server.com/postman-collection.json`

## API Versioning

### Version Strategy
- **Current Version**: v1
- **Deprecated Versions**: None
- **Beta Features**: Available in v2-beta

### Version Headers
```http
API-Version: v1
Accept: application/vnd.rag-server.v1+json
```

## Support and Resources

### Documentation Links
- **API Reference**: `/docs/api-reference.md`
- **Getting Started**: `/docs/quickstart-guide.md`
- **Authentication Guide**: `/docs/auth-guide.md`
- **Error Handling**: `/docs/error-handling.md`

### Support Channels
- **Technical Support**: support@rag-server.com
- **Community Forum**: https://community.rag-server.com
- **GitHub Issues**: https://github.com/mcp-rag-server/issues
- **Documentation Feedback**: docs@rag-server.com

## Implementation Notes

This documentation system should be generated automatically from:
1. **OpenAPI/Swagger Specifications**: Auto-generated from code annotations
2. **Code Comments**: Docstring extraction for detailed explanations
3. **Example Collections**: Maintained test cases as documentation examples
4. **Version Control**: Documentation versioned alongside API changes