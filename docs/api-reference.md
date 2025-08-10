# 📚 API Reference Documentation

Complete API reference for the MCP Academic RAG Server, including REST endpoints, data models, and integration examples.

## Table of Contents

- [Authentication](#authentication)
- [REST API Endpoints](#rest-api-endpoints)
- [Data Models](#data-models)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [SDK Integration](#sdk-integration)
- [Webhook Events](#webhook-events)

---

## Authentication

### API Key Authentication

All API requests require authentication using an API key in the header:

```http
Authorization: Bearer YOUR_API_KEY
```

### Environment Variables

```bash
# Set your API key
export MCP_RAG_API_KEY="your-api-key-here"

# Optional: Set base URL (default: http://localhost:8000)
export MCP_RAG_BASE_URL="https://your-server.com"
```

---

## REST API Endpoints

### Document Processing

#### POST /api/documents/process

Process an academic document through the RAG pipeline.

**Request:**
```http
POST /api/documents/process
Content-Type: multipart/form-data
Authorization: Bearer YOUR_API_KEY

{
  "file": <binary_file_data>,
  "filename": "research_paper.pdf",
  "metadata": {
    "source": "arxiv",
    "category": "machine_learning"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "document_id": "doc_abc123def456",
  "processing_job_id": "job_789ghi012",
  "estimated_completion": "2024-01-15T10:35:00Z",
  "message": "Document processing started"
}
```

#### GET /api/documents/{document_id}/status

Check document processing status.

**Response:**
```json
{
  "document_id": "doc_abc123def456",
  "status": "processing",
  "progress": 65,
  "current_stage": "EmbeddingProcessor",
  "stages_completed": ["PreProcessor", "OCRProcessor", "StructureProcessor"],
  "estimated_completion": "2024-01-15T10:35:00Z",
  "error": null
}
```

#### GET /api/documents/{document_id}

Retrieve document information and metadata.

**Response:**
```json
{
  "document_id": "doc_abc123def456",
  "filename": "research_paper.pdf",
  "status": "completed",
  "created_at": "2024-01-15T10:00:00Z",
  "updated_at": "2024-01-15T10:30:00Z",
  "metadata": {
    "title": "Deep Learning for Natural Language Processing",
    "authors": ["John Doe", "Jane Smith"],
    "abstract": "This paper presents...",
    "keywords": ["deep learning", "nlp", "transformers"],
    "page_count": 15,
    "language": "en",
    "file_size": 2048576,
    "processing_time": 45.2
  },
  "structure": {
    "sections": [
      {"title": "Abstract", "page_range": [1, 1], "word_count": 245},
      {"title": "Introduction", "page_range": [2, 3], "word_count": 892}
    ],
    "tables": [
      {"title": "Experimental Results", "page": 8, "row_count": 5, "col_count": 4}
    ],
    "figures": [
      {"title": "Model Architecture", "page": 5, "type": "diagram"}
    ]
  }
}
```

#### DELETE /api/documents/{document_id}

Delete a document and its associated data.

**Response:**
```json
{
  "status": "success",
  "message": "Document deleted successfully",
  "document_id": "doc_abc123def456"
}
```

### Query and Search

#### POST /api/query

Query documents using natural language.

**Request:**
```json
{
  "query": "What are the latest developments in transformer architectures?",
  "session_id": "session_xyz789",
  "filters": {
    "document_types": ["academic_paper"],
    "languages": ["en"],
    "date_range": {
      "start": "2023-01-01",
      "end": "2024-01-01"
    }
  },
  "retrieval_config": {
    "method": "hybrid",
    "top_k": 10,
    "dense_weight": 0.7,
    "sparse_weight": 0.3
  },
  "response_config": {
    "include_citations": true,
    "include_structured_content": true,
    "max_response_length": 2000
  }
}
```

**Response:**
```json
{
  "status": "success",
  "session_id": "session_xyz789",
  "query": "What are the latest developments in transformer architectures?",
  "answer": "Recent developments in transformer architectures include...",
  "citations": [
    {
      "document_id": "doc_abc123",
      "content": "The attention mechanism in transformers...",
      "relevance_score": 0.95,
      "metadata": {
        "title": "Attention Is All You Need",
        "authors": ["Vaswani et al."],
        "page": 3,
        "section": "Model Architecture"
      },
      "structured_content": {
        "type": "figure",
        "title": "Transformer Architecture",
        "data": "base64_encoded_image_data"
      }
    }
  ],
  "metadata": {
    "processing_time": 1.2,
    "retrieval_method": "hybrid",
    "documents_searched": 156,
    "model_used": "gpt-3.5-turbo"
  }
}
```

#### GET /api/search

Search documents by text or metadata.

**Parameters:**
- `q` (string): Search query
- `limit` (integer): Maximum results (default: 10, max: 100)
- `offset` (integer): Pagination offset (default: 0)
- `sort` (string): Sort by "relevance", "date", "title" (default: "relevance")
- `filters` (object): Filter criteria

**Response:**
```json
{
  "total": 45,
  "limit": 10,
  "offset": 0,
  "results": [
    {
      "document_id": "doc_abc123",
      "title": "Deep Learning Survey",
      "authors": ["Author Name"],
      "relevance_score": 0.95,
      "snippet": "Highlighted text snippet...",
      "metadata": {
        "created_at": "2024-01-15T10:00:00Z",
        "page_count": 15,
        "language": "en"
      }
    }
  ]
}
```

### Session Management

#### POST /api/sessions

Create a new chat session.

**Request:**
```json
{
  "name": "Research Discussion",
  "metadata": {
    "topic": "machine_learning",
    "user_id": "user_123"
  }
}
```

**Response:**
```json
{
  "session_id": "session_abc123",
  "name": "Research Discussion",
  "created_at": "2024-01-15T10:00:00Z",
  "status": "active"
}
```

#### GET /api/sessions

List all sessions.

**Response:**
```json
{
  "sessions": [
    {
      "session_id": "session_abc123",
      "name": "Research Discussion",
      "created_at": "2024-01-15T10:00:00Z",
      "last_activity": "2024-01-15T11:30:00Z",
      "message_count": 15,
      "status": "active"
    }
  ],
  "total": 5
}
```

#### GET /api/sessions/{session_id}/messages

Get session message history.

**Response:**
```json
{
  "session_id": "session_abc123",
  "messages": [
    {
      "message_id": "msg_001",
      "role": "user",
      "content": "What is machine learning?",
      "timestamp": "2024-01-15T10:00:00Z"
    },
    {
      "message_id": "msg_002",
      "role": "assistant",
      "content": "Machine learning is a subset of artificial intelligence...",
      "timestamp": "2024-01-15T10:00:30Z",
      "citations": [...]
    }
  ]
}
```

### System Management

#### GET /api/health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:00:00Z",
  "version": "1.2.0",
  "services": {
    "database": "healthy",
    "vector_store": "healthy",
    "document_processor": "healthy",
    "llm_service": "healthy"
  },
  "metrics": {
    "uptime": 86400,
    "processed_documents": 1250,
    "active_sessions": 15,
    "memory_usage": "2.5GB",
    "cpu_usage": "45%"
  }
}
```

#### GET /api/stats

System statistics and metrics.

**Response:**
```json
{
  "documents": {
    "total": 1250,
    "processing": 5,
    "completed": 1240,
    "failed": 5
  },
  "queries": {
    "total_today": 450,
    "average_response_time": 1.2,
    "success_rate": 0.98
  },
  "storage": {
    "total_size": "15.6GB",
    "documents": "12.1GB",
    "vectors": "3.5GB"
  }
}
```

---

## Data Models

### Document Model

```typescript
interface Document {
  document_id: string;
  filename: string;
  file_path: string;
  file_size: number;
  mime_type: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  created_at: string;
  updated_at: string;
  metadata: DocumentMetadata;
  structure: DocumentStructure;
  processing_history: ProcessingStage[];
}

interface DocumentMetadata {
  title?: string;
  authors?: string[];
  abstract?: string;
  keywords?: string[];
  language?: string;
  page_count?: number;
  word_count?: number;
  publication_date?: string;
  doi?: string;
  source?: string;
  category?: string;
  confidence_scores?: {
    ocr?: number;
    classification?: number;
    structure?: number;
  };
}

interface DocumentStructure {
  sections: Section[];
  tables: Table[];
  figures: Figure[];
  equations: Equation[];
  references: Reference[];
}

interface Section {
  title: string;
  level: number;
  page_range: [number, number];
  word_count: number;
  content_preview: string;
}
```

### Query Model

```typescript
interface QueryRequest {
  query: string;
  session_id?: string;
  filters?: QueryFilters;
  retrieval_config?: RetrievalConfig;
  response_config?: ResponseConfig;
}

interface QueryFilters {
  document_types?: string[];
  languages?: string[];
  authors?: string[];
  date_range?: {
    start: string;
    end: string;
  };
  categories?: string[];
  confidence_threshold?: number;
}

interface RetrievalConfig {
  method: 'dense' | 'sparse' | 'hybrid';
  top_k: number;
  dense_weight?: number;
  sparse_weight?: number;
  rerank?: boolean;
}

interface QueryResponse {
  status: string;
  session_id: string;
  query: string;
  answer: string;
  citations: Citation[];
  metadata: QueryMetadata;
}
```

### Citation Model

```typescript
interface Citation {
  document_id: string;
  content: string;
  relevance_score: number;
  metadata: {
    title: string;
    authors: string[];
    page?: number;
    section?: string;
    table_id?: string;
    figure_id?: string;
  };
  structured_content?: StructuredContent;
}

interface StructuredContent {
  type: 'table' | 'figure' | 'code' | 'equation';
  title: string;
  data: any;
  format?: string;
}
```

---

## Error Handling

### Error Response Format

```json
{
  "status": "error",
  "error_code": "DOCUMENT_PROCESSING_FAILED",
  "message": "Failed to process document due to unsupported format",
  "details": {
    "document_id": "doc_abc123",
    "stage": "OCRProcessor", 
    "reason": "Unsupported file format: .xyz"
  },
  "timestamp": "2024-01-15T10:00:00Z",
  "request_id": "req_def456ghi789"
}
```

### Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `INVALID_REQUEST` | Request validation failed | 400 |
| `UNAUTHORIZED` | Authentication failed | 401 |
| `FORBIDDEN` | Insufficient permissions | 403 |
| `DOCUMENT_NOT_FOUND` | Document does not exist | 404 |
| `UNSUPPORTED_FORMAT` | File format not supported | 400 |
| `PROCESSING_FAILED` | Document processing error | 500 |
| `QUERY_FAILED` | Query execution error | 500 |
| `RATE_LIMIT_EXCEEDED` | Too many requests | 429 |
| `SERVICE_UNAVAILABLE` | Service temporarily down | 503 |

---

## Rate Limiting

### Default Limits

- **Document Processing**: 10 requests/minute
- **Queries**: 100 requests/minute  
- **Search**: 200 requests/minute
- **General API**: 1000 requests/hour

### Headers

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642781400
```

### Rate Limit Response

```json
{
  "status": "error",
  "error_code": "RATE_LIMIT_EXCEEDED",
  "message": "Rate limit exceeded. Please try again later.",
  "retry_after": 60
}
```

---

## SDK Integration

### Python SDK

```python
from mcp_rag_client import MCPRAGClient

# Initialize client
client = MCPRAGClient(
    api_key="your-api-key",
    base_url="http://localhost:8000"
)

# Process document
result = client.process_document(
    file_path="/path/to/document.pdf",
    metadata={"category": "research"}
)

# Query documents
response = client.query(
    query="What is machine learning?",
    session_id="session_123",
    top_k=5
)

# Stream query results
for chunk in client.query_stream(query="Explain neural networks"):
    print(chunk.content, end="")
```

### JavaScript SDK

```javascript
import { MCPRAGClient } from '@mcp/rag-client';

const client = new MCPRAGClient({
  apiKey: 'your-api-key',
  baseUrl: 'http://localhost:8000'
});

// Process document
const result = await client.processDocument({
  file: fileBlob,
  filename: 'paper.pdf',
  metadata: { category: 'research' }
});

// Query with streaming
const stream = client.queryStream({
  query: 'What is deep learning?',
  sessionId: 'session_123'
});

for await (const chunk of stream) {
  console.log(chunk.content);
}
```

### curl Examples

```bash
# Process document
curl -X POST http://localhost:8000/api/documents/process \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "file=@paper.pdf" \
  -F "metadata={\"category\":\"research\"}"

# Query documents  
curl -X POST http://localhost:8000/api/query \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "top_k": 5,
    "include_citations": true
  }'

# Health check
curl -X GET http://localhost:8000/api/health \
  -H "Authorization: Bearer YOUR_API_KEY"
```

---

## Webhook Events

### Configuration

```json
{
  "webhook_url": "https://your-app.com/webhooks/mcp-rag",
  "events": [
    "document.processed",
    "document.failed", 
    "query.completed"
  ],
  "secret": "webhook_secret_key"
}
```

### Event Types

#### document.processed

```json
{
  "event": "document.processed",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "document_id": "doc_abc123",
    "filename": "research_paper.pdf",
    "status": "completed",
    "processing_time": 45.2,
    "metadata": {
      "title": "Document Title",
      "page_count": 15
    }
  }
}
```

#### document.failed

```json
{
  "event": "document.failed",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "document_id": "doc_abc123",
    "filename": "document.pdf",
    "error": "Unsupported file format",
    "stage": "PreProcessor"
  }
}
```

#### query.completed

```json
{
  "event": "query.completed",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "session_id": "session_xyz789",
    "query": "What is machine learning?",
    "response_time": 1.2,
    "citations_count": 5
  }
}
```

---

*Last updated: 2024-01-15*