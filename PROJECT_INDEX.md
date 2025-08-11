# 📚 MCP Academic RAG Server - Project Index

**Comprehensive project documentation and knowledge base for the MCP Academic RAG Server**

---

## 🚀 Quick Start

| Component | Description | Entry Point |
|-----------|-------------|-------------|
| **Main Application** | Core server application | `app.py` |
| **Web Interface** | User-friendly web UI | `webapp.py` |
| **MCP Server** | AI assistant integration | `mcp_server.py` |
| **CLI Tools** | Command-line interfaces | `cli/` |

---

## 📁 Project Structure Overview

### Core Components

```
mcp-academic-rag-server/
├── 🏗️ Core Architecture
│   ├── core/                    # System core components
│   │   ├── config_manager.py   # Configuration management
│   │   ├── config_validator.py # Configuration validation
│   │   └── pipeline.py         # Async processing pipeline
│   │
│   ├── models/                  # Data models
│   │   ├── document.py         # Document data model
│   │   └── process_result.py   # Processing result model
│   │
│   └── utils/                   # Utility functions
│       ├── error_handling.py   # Error handling utilities
│       ├── logging_utils.py    # Logging configuration
│       ├── monitoring_utils.py # Monitoring tools
│       ├── performance_utils.py # Performance utilities
│       ├── security_utils.py   # Security functions
│       ├── text_utils.py       # Text processing
│       └── vector_utils.py     # Vector operations
│
├── 📄 Document Processing
│   ├── processors/              # Document processors
│   │   ├── base_processor.py   # Base processor class
│   │   ├── classification_processor.py  # Content classification
│   │   ├── format_converter.py # Format conversion
│   │   ├── haystack_embedding_processor.py  # Vector embeddings
│   │   ├── knowledge_graph_processor.py     # Knowledge extraction
│   │   ├── ocr_processor.py    # OCR processing
│   │   ├── pre_processor.py    # Pre-processing
│   │   └── structure_processor.py  # Document structure
│   │
│   └── connectors/              # External API connectors
│       ├── api_connector.py    # Generic API connector
│       └── haystack_llm_connector.py  # LLM integration
│
├── 🔍 RAG System
│   ├── rag/                     # RAG components
│   │   ├── chat_session.py     # Chat session management
│   │   ├── haystack_pipeline.py # RAG pipeline
│   │   └── prompt_builder.py   # Prompt construction
│   │
│   ├── retrievers/              # Information retrieval
│   │   └── haystack_retriever.py  # Hybrid retrieval
│   │
│   └── document_stores/         # Data storage
│       ├── haystack_store.py   # Haystack storage
│       └── milvus_store.py     # Vector database
│
├── 🖥️ User Interfaces
│   ├── cli/                     # Command-line tools
│   │   ├── chat_cli.py         # Chat interface
│   │   └── document_cli.py     # Document management
│   │
│   ├── templates/               # Web templates
│   │   ├── base.html           # Base template
│   │   ├── index.html          # Homepage
│   │   ├── upload.html         # Upload interface
│   │   ├── documents.html      # Document management
│   │   ├── chat.html           # Chat interface
│   │   └── about.html          # About page
│   │
│   └── static/                  # Static assets
│       ├── css/style.css       # Styles
│       ├── js/main.js          # JavaScript
│       └── img/                # Images
│
└── ⚙️ Configuration & Deployment
    ├── config/                  # Configuration files
    │   ├── config.json         # Main configuration
    │   ├── config.json.example # Configuration template
    │   └── milvus.yaml         # Milvus settings
    │
    ├── k8s/                     # Kubernetes manifests
    │   ├── deployment.yaml     # Application deployment
    │   ├── service.yaml        # Service definition
    │   ├── configmap.yaml      # Configuration map
    │   └── pvc.yaml            # Persistent volume
    │
    ├── docker-compose.yml       # Docker orchestration
    ├── Dockerfile              # Container definition
    └── .env.example            # Environment template
```

---

## 📖 Documentation Index

### User Documentation
| Document | Purpose | Audience |
|----------|---------|----------|
| [README.md](README.md) | Project overview and quick start | All users |
| [User Guide](docs/user-guide/) | Comprehensive usage guide | End users |
| [MCP Tools Reference](docs/user-guide/mcp-tools-reference.md) | MCP integration guide | AI assistant users |
| [API Reference](docs/api-reference.md) | REST API documentation | Developers |

### Developer Documentation  
| Document | Purpose | Audience |
|----------|---------|----------|
| [Contributing Guide](CONTRIBUTING.md) | Contribution guidelines | Contributors |
| [Developer Guide](docs/developer-guide.md) | Development setup and practices | Developers |
| [Examples](examples/) | Working code examples | All developers |
| [Tests Documentation](tests/README.md) | Testing framework guide | Contributors |

### Architecture Documentation
| Document | Purpose | Audience |
|----------|---------|----------|
| [整体设计.md](整体设计.md) | Overall system design (Chinese) | Architects |
| [架构设计.md](架构设计.md) | Architecture design (Chinese) | Architects |
| [System Architecture Diagram](static/img/system_architecture.png) | Visual system overview | All users |

---

## 🔧 Component Reference

### Core Processing Pipeline

```mermaid
flowchart LR
    A[Document Input] --> B[PreProcessor]
    B --> C[OCRProcessor] 
    C --> D[StructureProcessor]
    D --> E[ClassificationProcessor]
    E --> F[EmbeddingProcessor]
    F --> G[Storage]
```

| Component | File | Purpose | Dependencies |
|-----------|------|---------|--------------|
| **PreProcessor** | `processors/pre_processor.py` | File validation and normalization | `utils/text_utils.py` |
| **OCRProcessor** | `processors/ocr_processor.py` | Text extraction from images/PDFs | `connectors/api_connector.py` |
| **StructureProcessor** | `processors/structure_processor.py` | Document structure recognition | `utils/text_utils.py` |
| **ClassificationProcessor** | `processors/classification_processor.py` | Content classification | `connectors/haystack_llm_connector.py` |
| **EmbeddingProcessor** | `processors/haystack_embedding_processor.py` | Vector embeddings generation | `document_stores/` |

### RAG System Components

| Component | File | Purpose | Key Features |
|-----------|------|---------|--------------|
| **ChatSession** | `rag/chat_session.py` | Conversation management | Session persistence, context tracking |
| **RAGPipeline** | `rag/haystack_pipeline.py` | Retrieval-Augmented Generation | Hybrid search, answer generation |
| **PromptBuilder** | `rag/prompt_builder.py` | Dynamic prompt construction | Template management, context injection |
| **HaystackRetriever** | `retrievers/haystack_retriever.py` | Hybrid information retrieval | Dense + sparse retrieval |

### Storage Systems

| System | Implementation | Use Case | Configuration |
|--------|----------------|----------|---------------|
| **Milvus** | `document_stores/milvus_store.py` | Production vector storage | `config/milvus.yaml` |
| **Haystack** | `document_stores/haystack_store.py` | Development/testing | `config/config.json` |
| **Local Files** | Built-in Python | Document and session storage | `data/` directory |

---

## 🛠️ Development Resources

### Testing Framework
```
tests/
├── unit/           # Component-level tests (90%+ coverage)
├── integration/    # System interaction tests
├── component/      # Multi-component tests  
├── performance/    # Performance benchmarks
└── e2e/           # End-to-end workflow tests
```

### Key Test Files
| Test Category | Files | Coverage |
|---------------|-------|----------|
| **Core Components** | `test_config_manager.py`, `test_pipeline.py` | Core functionality |
| **Processors** | `test_ocr_processor.py`, `test_rag_pipeline.py` | Processing logic |
| **Integration** | `test_rag_integration.py`, `test_milvus_persistence.py` | System integration |
| **Performance** | `test_async_performance.py` | Performance benchmarks |
| **E2E** | `test_web_ui.py` | Complete workflows |

### Example Scripts
| Example | File | Demonstrates |
|---------|------|-------------|
| **Document Processing** | `examples/document_processing_example.py` | Basic document workflow |
| **Chat Session** | `examples/chat_session_example.py` | Conversation management |
| **Batch Processing** | `examples/batch_processing_example.py` | Bulk document processing |
| **Interactive Chat** | `examples/interactive_chat_example.py` | Real-time interaction |
| **Knowledge Graph** | `examples/knowledge_graph_example.py` | Knowledge extraction |

---

## 🔌 Integration Points

### MCP (Model Context Protocol) Tools
| Tool | Function | Input | Output |
|------|----------|-------|--------|
| `process_document` | Process academic documents | File path, filename | Processing status |
| `query_documents` | Query processed documents | Query text, session ID | AI response + citations |
| `get_document_info` | Retrieve document metadata | Document ID | Document details |
| `list_sessions` | List chat sessions | None | Session list |

### REST API Endpoints
| Endpoint | Method | Purpose | Parameters |
|----------|--------|---------|-----------|
| `/api/upload` | POST | Upload documents | `file`, `process_immediately` |
| `/api/documents` | GET | List documents | `page`, `limit`, `filter` |
| `/api/documents/{id}` | GET/DELETE | Document operations | `id` |
| `/api/chat` | POST | Chat interactions | `query`, `session_id` |
| `/api/health` | GET | System health check | None |

### External Services Integration
| Service Type | Supported Providers | Configuration |
|--------------|-------------------|---------------|
| **OCR Services** | Azure Vision, Google Vision, Baidu OCR | API keys in `config.json` |
| **LLM Services** | OpenAI, Anthropic, Mistral, Local models | Model selection and parameters |
| **Embedding Models** | OpenAI, Sentence Transformers | Model name and dimensions |

---

## 📊 Performance & Monitoring

### System Metrics
| Metric | Target | Monitoring |
|--------|--------|------------|
| **Document Processing** | 15-30s per PDF (10 pages) | `utils/performance_utils.py` |
| **Query Response** | <500ms average | `utils/monitoring_utils.py` |
| **Memory Usage** | <2GB standard config | Built-in monitoring |
| **Test Coverage** | 85%+ overall | pytest coverage reports |

### Health Checks
| Check | Script | Purpose |
|-------|--------|---------|
| **System Health** | `health_check.py` | Overall system status |
| **Basic Health** | `tools/basic_health_check.py` | Quick connectivity test |
| **System Health** | `tools/system_health_check.py` | Comprehensive diagnostics |
| **Config Validation** | `tools/validate_config.py` | Configuration integrity |

---

## 🚀 Deployment Options

### Docker Deployment
| File | Purpose | Environment |
|------|---------|-------------|
| `Dockerfile` | Application container | Production/Development |
| `docker-compose.yml` | Service orchestration | Complete stack |
| `.env.example` | Environment template | Configuration |

### Kubernetes Deployment
| Manifest | Purpose | Resource |
|----------|---------|----------|
| `k8s/deployment.yaml` | Application deployment | Pods, containers |
| `k8s/service.yaml` | Service exposure | Load balancing |
| `k8s/configmap.yaml` | Configuration management | Config injection |
| `k8s/pvc.yaml` | Persistent storage | Data persistence |

### Local Development
| Command | Purpose | Requirements |
|---------|---------|--------------|
| `python app.py` | Start main server | Python 3.9+, dependencies |
| `python webapp.py` | Start web interface | Flask, templates |
| `python mcp_server.py` | Start MCP server | MCP client configured |

---

## 📈 Project Status & Roadmap

### Current Status (v1.2.0)
- ✅ **Core Features**: Document processing, RAG system, MCP integration
- ✅ **Testing**: 85%+ test coverage, CI/CD pipeline
- ✅ **Deployment**: Docker, Kubernetes, production-ready
- ✅ **Documentation**: Comprehensive user and developer guides

### Upcoming Features
- 🚧 **Q1 2025**: Multi-modal support, advanced search
- 📋 **Q2 2025**: User authentication, plugin system
- 💡 **Q3 2025**: More LLM providers, mobile support
- 🔮 **Q4 2025**: Enterprise features, high availability

---

## 🤝 Community & Support

### Getting Help
| Resource | Purpose | Link |
|----------|---------|------|
| **GitHub Issues** | Bug reports, feature requests | [Issues](https://github.com/yourusername/mcp-academic-rag-server/issues) |
| **Discussions** | Community questions | [Discussions](https://github.com/yourusername/mcp-academic-rag-server/discussions) |
| **Documentation** | Online docs | [Docs Site](https://yourusername.github.io/mcp-academic-rag-server/) |

### Contributing
1. **Fork** the repository
2. **Create** feature branch
3. **Follow** coding standards ([CONTRIBUTING.md](CONTRIBUTING.md))
4. **Submit** pull request
5. **Celebrate** your contribution! 🎉

---

## 📝 License & Acknowledgments

- **License**: MIT License - see [LICENSE](LICENSE)
- **Dependencies**: [requirements.txt](requirements.txt)
- **Contributors**: See [CONTRIBUTORS.md](CONTRIBUTORS.md)
- **Special Thanks**: Haystack, Milvus, MCP community

---

*Last Updated: 2024-12-15 | Version: 1.2.0 | Status: Production Ready*

---

**⭐ If this project helps you, please give it a star on GitHub!**