# RAG Pipeline API Reference

## Overview

The RAG (Retrieval-Augmented Generation) Pipeline is the core component that combines semantic document retrieval with LLM-based response generation for academic research queries.

## Architecture

```
Query → Query Embedder → Retriever → Prompt Builder → LLM → Response
        (Sentence-BERT)  (Semantic)   (Context)      (OpenAI)
```

## Core Classes

### `RAGPipeline`

**Purpose**: Main orchestrator for the RAG workflow using Haystack 2.x framework.

#### Constructor

```python
RAGPipeline(
    llm_connector: HaystackLLMConnector,
    document_store: Optional[DocumentStore] = None,
    retriever_top_k: int = 5,
    prompt_builder: Optional[ChatPromptBuilder] = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
)
```

**Parameters:**
- `llm_connector`: OpenAI GPT connector for response generation
- `document_store`: Vector storage (defaults to InMemoryDocumentStore)
- `retriever_top_k`: Maximum documents to retrieve (default: 5)
- `prompt_builder`: Context formatting component
- `embedding_model`: Sentence-BERT model for embeddings

#### Key Methods

##### `run(query, chat_history=None, filters=None, generation_kwargs=None)`

**Purpose**: Execute complete RAG pipeline for a query.

**Parameters:**
- `query` (str): User's research question
- `chat_history` (list, optional): Previous conversation context
- `filters` (dict, optional): Document filtering criteria
- `generation_kwargs` (dict, optional): LLM generation parameters

**Returns:**
```python
{
    "answer": str,           # Generated response
    "documents": [           # Retrieved context
        {
            "id": str,
            "content": str,
            "metadata": dict
        }
    ],
    "query": str            # Original query
}
```

**Example:**
```python
rag_pipeline = RAGPipeline(llm_connector)
result = rag_pipeline.run("What is machine learning?")
print(result["answer"])  # LLM-generated response
print(len(result["documents"]))  # Number of retrieved docs
```

##### `update_retriever(document_store=None, top_k=None)`

**Purpose**: Dynamically update retrieval configuration.

**Parameters:**
- `document_store`: New document storage instance
- `top_k`: New maximum retrieval count

## Pipeline Components

### 1. Query Embedder
- **Technology**: SentenceTransformersTextEmbedder
- **Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Purpose**: Convert queries to semantic vectors

### 2. Retriever
- **Technology**: InMemoryEmbeddingRetriever
- **Similarity**: Dot product
- **Purpose**: Find semantically similar documents

### 3. Prompt Builder
- **Technology**: ChatPromptBuilder
- **Templates**: Academic, general, concise styles
- **Purpose**: Format context for LLM

### 4. LLM Generator
- **Technology**: OpenAIChatGenerator
- **Model**: GPT-3.5-turbo
- **Purpose**: Generate contextual responses

## Error Handling

The pipeline includes comprehensive error handling:

```python
try:
    result = rag_pipeline.run(query)
    if "error" in result:
        print(f"Pipeline error: {result['error']}")
    else:
        print(f"Success: {len(result['documents'])} docs retrieved")
except Exception as e:
    print(f"Execution error: {str(e)}")
```

## Performance Characteristics

| Metric | Typical Value |
|--------|---------------|
| **Query Processing** | 3-4 seconds |
| **Embedding Dimension** | 384 |
| **Max Documents** | 5 (configurable) |
| **Memory Usage** | ~100MB (base) |
| **Similarity Method** | Dot product |

## Usage Patterns

### Basic Query
```python
result = pipeline.run("Explain neural networks")
```

### With Chat History
```python
history = [{"role": "user", "content": "What is AI?"}]
result = pipeline.run("How does it relate to ML?", chat_history=history)
```

### With Filtering
```python
filters = {"document_type": "research_paper"}
result = pipeline.run("Latest developments", filters=filters)
```

## Configuration

### Document Store Options
- **InMemoryDocumentStore**: Fast, temporary storage
- **HaystackDocumentStore**: Persistent storage with advanced features

### Embedding Models
- **all-MiniLM-L6-v2**: Fast, 384-dim (default)
- **all-mpnet-base-v2**: Accurate, 768-dim
- **multi-qa-MiniLM-L6-cos-v1**: Optimized for Q&A

### LLM Configuration
```python
llm_connector = HaystackLLMConnector(
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=1000
)
```

## Integration

### With MCP Server
```python
class MCPServer:
    def __init__(self):
        self.rag_pipeline = RAGPipeline(llm_connector)
    
    async def handle_query_documents(self, request_id, params):
        result = self.rag_pipeline.run(params["query"])
        return self.format_mcp_response(result)
```

### Factory Pattern
```python
pipeline = RAGPipelineFactory.create_pipeline(
    llm_connector=connector,
    retriever_top_k=3,
    config={"embedding_model": "all-mpnet-base-v2"}
)
```

## Best Practices

1. **Warm-up**: Initialize pipeline once, reuse for multiple queries
2. **Batching**: Process multiple queries with same documents efficiently  
3. **Memory**: Monitor document store size for large collections
4. **Caching**: Cache embeddings for frequently accessed documents
5. **Error Handling**: Always check for errors in pipeline results

## Troubleshooting

### Common Issues

**No documents retrieved:**
```python
# Check document count
print(f"Documents in store: {pipeline.document_store.count_documents()}")

# Verify embedding generation
if pipeline.document_store.count_documents() == 0:
    print("No documents indexed - process documents first")
```

**Low retrieval relevance:**
```python
# Adjust similarity threshold or retrieval count
pipeline.update_retriever(top_k=10)  # Increase results

# Try different embedding model
pipeline = RAGPipeline(llm_connector, embedding_model="all-mpnet-base-v2")
```

**OpenAI API errors:**
```python
# Check API key and rate limits
result = pipeline.run(query)
if "error" in result and "openai" in result["error"].lower():
    print("Check OpenAI API key and quotas")
```