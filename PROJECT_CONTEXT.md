# 🚀 MCP Academic RAG Server - Comprehensive Project Context

**Generated Project Context Map | Updated: 2024-12-15**

---

## 📊 Project Overview

| Property | Value |
|----------|--------|
| **Project Name** | MCP Academic RAG Server |
| **Version** | 1.2.0 |
| **Python Version** | ≥3.9 (Current: 3.13.5) |
| **License** | MIT |
| **Status** | Production Ready |
| **Architecture** | Async, Microservices, Container-Ready |

---

## 🏗️ Technology Stack Analysis

### Core Dependencies
```yaml
Framework:
  - Flask 2.3.3          # Web framework
  - Haystack 2.16.1      # RAG framework  
  - MCP 0.9.0             # Model Context Protocol

AI/ML:
  - OpenAI 1.3.0          # LLM integration
  - sentence-transformers 2.2.2  # Embeddings
  - transformers 4.35.0   # HuggingFace
  - torch 2.1.0          # PyTorch

Document Processing:
  - PyPDF2 3.0.1         # PDF handling
  - Pillow 10.0.1        # Image processing
  - opencv-python 4.8.1.78  # Computer vision

Storage:
  - pymilvus 2.3.4       # Vector database
  - faiss-cpu 1.7.4      # Vector similarity
  - chromadb 0.4.15      # Alternative vector store

Testing:
  - pytest 7.4.3         # Test framework
  - pytest-cov 4.1.0     # Coverage
  - selenium 4.15.0      # E2E testing
```

### Development Tools
| Category | Tools | Purpose |
|----------|--------|---------|
| **Code Quality** | black, isort, flake8, mypy | Formatting, linting, type checking |
| **Security** | bandit, safety | Security scanning |
| **Performance** | pytest-benchmark, memory-profiler | Performance analysis |
| **Monitoring** | prometheus-client, loguru | Metrics and logging |

---

## ⚙️ Configuration Analysis

### Core Configuration (`config/config.json`)

**Storage Settings:**
```json
{
  "storage": {
    "base_path": "./data",
    "output_path": "./output"
  }
}
```

**Vector Database (Milvus):**
```json
{
  "vector_db": {
    "milvus": {
      "host": "localhost",
      "port": 19530,
      "collection_name": "academic_documents",
      "vector_dimension": 384,
      "index_type": "IVF_FLAT",
      "metric_type": "COSINE"
    }
  }
}
```

**Processing Pipeline:**
```json
{
  "processors": {
    "pre_processor": { "enabled": true },
    "ocr_processor": { "enabled": true, "api": "azure" },
    "structure_processor": { "enabled": true },
    "embedding_processor": { "enabled": true },
    "knowledge_graph_processor": { "enabled": true }
  }
}
```

### Environment Configuration (`.env.example`)

**Key Environment Variables:**
- `OPENAI_API_KEY` - OpenAI API access
- `AZURE_API_KEY` - Azure OCR services  
- `MILVUS_HOST/PORT` - Vector database connection
- `LOG_LEVEL` - Logging verbosity
- `MAX_CONCURRENT_DOCUMENTS` - Processing limits

---

## 🐳 Deployment Architecture

### Docker Compose Stack
```yaml
Services:
  - academic-rag-server:8000  # Main application
  - milvus:19530              # Vector database
  - etcd:2379                 # Milvus metadata
  - minio:9000                # Object storage
  - redis:6379                # Caching (optional)
  - nginx:80/443              # Reverse proxy (optional)
  - prometheus:9090           # Monitoring (optional)
  - grafana:3000              # Visualization (optional)
```

### Kubernetes Support
- **Deployment**: Application pods with health checks
- **Service**: Load balancing and service discovery  
- **ConfigMap**: Configuration management
- **PVC**: Persistent volume for data storage

---

## 📁 Codebase Structure Map

### Core Application Files
```
├── app.py              # Main server application
├── webapp.py           # Web interface server  
├── mcp_server.py       # MCP protocol server
└── health_check.py     # System health monitoring
```

### Processing Pipeline
```
processors/
├── base_processor.py           # Abstract processor base
├── pre_processor.py           # File preprocessing
├── ocr_processor.py           # Text extraction
├── structure_processor.py     # Document structure
├── classification_processor.py # Content classification
├── format_converter.py        # Format conversion
├── haystack_embedding_processor.py  # Vector embeddings
└── knowledge_graph_processor.py    # Knowledge extraction
```

### RAG System
```
rag/
├── chat_session.py      # Conversation management
├── haystack_pipeline.py # RAG pipeline
└── prompt_builder.py   # Dynamic prompting

retrievers/
└── haystack_retriever.py  # Hybrid retrieval

document_stores/
├── haystack_store.py    # Haystack integration
└── milvus_store.py     # Milvus persistence
```

### User Interfaces
```
cli/
├── document_cli.py     # Document processing CLI
└── chat_cli.py        # Chat interface CLI

templates/
├── base.html          # Base template
├── index.html         # Homepage
├── upload.html        # File upload
├── documents.html     # Document management
└── chat.html         # Chat interface

static/
├── css/style.css      # Styling
├── js/main.js         # Frontend logic
└── img/              # Images and assets
```

---

## 🔧 Core Components Analysis

### 1. Document Processing Pipeline
**Flow**: Input → PreProcessor → OCR → Structure → Classification → Embedding → Storage

| Processor | Purpose | Dependencies | Configuration |
|-----------|---------|--------------|---------------|
| **PreProcessor** | File validation, image enhancement | `Pillow`, `opencv-python` | `enhance_image: true` |
| **OCRProcessor** | Text extraction from images/PDFs | Azure/Google/Baidu APIs | `api: "azure"` |
| **StructureProcessor** | Document structure detection | `nltk`, `spacy` | `detect_sections: true` |
| **EmbeddingProcessor** | Vector generation | `sentence-transformers` | `model: "all-MiniLM-L6-v2"` |

### 2. RAG System
**Components**: Retrieval → Generation → Response

| Component | Purpose | Technology | Configuration |
|-----------|---------|------------|---------------|
| **HaystackRetriever** | Hybrid search (dense + sparse) | Haystack 2.x, FAISS | `dense_weight: 0.7` |
| **ChatSession** | Conversation state management | Python sessions | `max_history: 10` |
| **PromptBuilder** | Dynamic prompt construction | Template engine | Custom templates |

### 3. Storage Systems
| System | Purpose | Configuration | Status |
|--------|---------|---------------|---------|
| **Milvus** | Production vector storage | Port 19530, COSINE similarity | ✅ Active |
| **FAISS** | Development vector storage | CPU-only, local files | 🚧 Fallback |
| **Local Files** | Document and session storage | `./data/` directory | ✅ Active |

---

## 🔌 Integration Points

### MCP Protocol Integration
**Tools Available:**
- `process_document` → Document processing workflow
- `query_documents` → RAG-based querying  
- `get_document_info` → Metadata retrieval
- `list_sessions` → Session management

**Client Configuration:**
```json
{
  "mcpServers": {
    "academic-rag": {
      "command": "python",
      "args": ["mcp_server.py"]
    }
  }
}
```

### External API Integration
| Service | Provider | Purpose | Configuration |
|---------|----------|---------|---------------|
| **OCR** | Azure Vision | Text extraction | `AZURE_API_KEY` |
| **LLM** | OpenAI GPT-3.5/4 | Response generation | `OPENAI_API_KEY` |
| **Embeddings** | OpenAI/SentenceTransformers | Vector generation | Model selection |

---

## 🧪 Testing Framework

### Test Structure
```
tests/
├── unit/           # Component testing (90%+ coverage)
│   ├── test_config_manager.py
│   ├── test_ocr_processor.py  
│   └── test_rag_pipeline.py
├── integration/    # System integration testing
│   ├── test_rag_integration.py
│   └── test_milvus_persistence.py
├── component/      # Multi-component testing
│   └── test_processing_pipeline.py
├── performance/    # Performance benchmarking
│   └── test_async_performance.py
└── e2e/           # End-to-end workflow testing
    └── test_web_ui.py
```

### Testing Tools & Coverage
| Tool | Purpose | Configuration |
|------|---------|---------------|
| **pytest** | Test execution | `pyproject.toml` |
| **pytest-cov** | Coverage reporting | Target: 85%+ |
| **selenium** | Web UI testing | Chrome/Firefox |
| **pytest-benchmark** | Performance testing | Auto-save results |

---

## 📊 Performance & Monitoring

### System Metrics
| Metric | Target | Current Status |
|---------|--------|----------------|
| **Document Processing** | 15-30s per PDF (10 pages) | ✅ Within target |
| **Query Response** | <500ms average | ✅ Optimized |
| **Memory Usage** | <2GB standard config | ✅ Efficient |
| **Test Coverage** | 85%+ overall | ✅ 85%+ achieved |

### Monitoring Stack
| Component | Purpose | Port | Status |
|-----------|---------|------|---------|
| **Prometheus** | Metrics collection | 9090 | 🚧 Optional |
| **Grafana** | Visualization | 3000 | 🚧 Optional |
| **Health Check** | System validation | Built-in | ✅ Active |

---

## 🔒 Security Configuration

### Security Features
- **API Key Management**: Environment variable isolation
- **Input Validation**: JSON Schema validation
- **File Upload Security**: Type and size restrictions
- **Container Security**: Non-root execution
- **Configuration Validation**: Startup validation checks

### Security Tools
| Tool | Purpose | Configuration |
|------|---------|---------------|
| **bandit** | Python security analysis | Medium confidence |
| **safety** | Dependency vulnerability scan | Auto-check |
| **cryptography** | Encryption utilities | 41.0.7+ |

---

## 🚀 Development Environment

### Prerequisites
- **Python**: 3.9+ (Current: 3.13.5)
- **Memory**: 16GB+ recommended  
- **Storage**: 2GB+ free space
- **Docker**: Optional for containerized development

### Quick Start Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Run health check
python health_check.py

# Start development server
python webapp.py

# Start MCP server
python mcp_server.py

# Run tests
pytest

# Docker deployment
docker-compose up -d
```

---

## 📈 Project Health Status

### Current State (v1.2.0)
- ✅ **Core Features**: Fully implemented and tested
- ✅ **Documentation**: Comprehensive user and dev guides
- ✅ **Testing**: 85%+ coverage with CI/CD
- ✅ **Deployment**: Production-ready containers
- ✅ **Performance**: Meeting all targets

### Known Limitations
- 🔍 **Python Version**: Requires 3.9+, running 3.13.5
- 💾 **Memory**: High memory usage with large documents
- 🔗 **Dependencies**: Some pinned versions may need updates
- 🌐 **Scaling**: Single-node deployment (K8s ready)

### Upcoming Improvements
- **Q1 2025**: Multi-modal support, advanced search
- **Q2 2025**: User authentication, plugin system  
- **Q3 2025**: More LLM providers, mobile support
- **Q4 2025**: Enterprise features, HA deployment

---

## 💡 Development Recommendations

### Code Quality
1. **Maintain** current 85%+ test coverage
2. **Update** dependencies quarterly
3. **Monitor** security vulnerabilities
4. **Profile** performance regularly

### Deployment  
1. **Use** Docker Compose for development
2. **Consider** Kubernetes for production scaling
3. **Monitor** resource usage and limits
4. **Implement** proper logging and alerts

### Integration
1. **Test** MCP integration thoroughly
2. **Validate** API key rotation procedures  
3. **Document** external service dependencies
4. **Plan** for service outages and fallbacks

---

*Context generated on: 2024-12-15 | Python 3.13.5 | Status: Production Ready*

---

**🎯 This context map provides comprehensive project understanding for developers, operators, and AI assistants.**