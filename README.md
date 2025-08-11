# MCP Academic RAG Server

[![MCP Compatible](https://img.shields.io/badge/MCP-1.0-purple.svg)](https://modelcontextprotocol.io)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Dependencies](https://img.shields.io/badge/dependencies-14_packages-green.svg)](#installation)
[![Deployment](https://img.shields.io/badge/deployment-ready-brightgreen.svg)](#deployment)
[![Status](https://img.shields.io/badge/status-production_ready-success.svg)](#quick-start)

A production-ready Model Context Protocol (MCP) server for academic document processing and retrieval-augmented generation queries. This server provides standardized tools for document analysis, OCR processing, and intelligent question-answering capabilities.

**🎯 Status**: Production Ready - Successfully deployed and validated  
**🚀 Quick Deploy**: `uvx --from . mcp-academic-rag-server`  
**📋 Architecture**: Focused MCP server (refactored from enterprise platform)

## Features

The server implements four core MCP tools:

| Tool | Description | Input | Output |
|------|-------------|-------|---------|
| `process_document` | Process PDF/image documents with OCR | File path, optional name | Document metadata and processing status |
| `query_documents` | Query processed documents using RAG | Question text, session context | Generated answer with source citations |
| `get_document_info` | Retrieve document processing details | Document ID | Comprehensive document information |
| `list_sessions` | List conversation sessions | None | Session metadata and statistics |

## Quick Start

### Prerequisites

- **Python**: 3.9 or later (tested with 3.13.5)
- **uvx**: Package manager (`pip install uvx`)
- **OpenAI API Key**: For RAG query generation
- **System**: Windows, Linux, or macOS

### Installation

#### Method 1: uvx Deployment (Production Ready) ⭐

```bash
# Clone and deploy (5-minute setup)
git clone <repository-url>
cd mcp-academic-rag-server
uvx --from . mcp-academic-rag-server
```

#### Method 2: Docker Deployment (Container Ready)

```bash
# Build and deploy with Docker
docker-compose -f docker-compose.simple.yml up -d

# View logs
docker-compose -f docker-compose.simple.yml logs -f
```

#### Method 3: Development Install

```bash
# For development and testing
pip install -e ".[dev]"
python mcp_server.py
```

### Configuration

1. Create environment configuration:
```bash
cp .env.example .env
```

2. Set required environment variables:
```bash
OPENAI_API_KEY=your_openai_api_key_here
MCP_PORT=8000
DATA_PATH=./data
```

3. Configure MCP client (Claude Desktop example):
```json
{
  "mcpServers": {
    "academic-rag": {
      "command": "uvx",
      "args": ["--from", ".", "mcp-academic-rag-server"],
      "cwd": "/path/to/mcp-academic-rag-server"
    }
  }
}
```

## Architecture

### System Design

The server implements a streamlined architecture focused on core MCP functionality:

```
MCP Academic RAG Server
├── MCP Protocol Interface
├── Document Processing Pipeline
├── RAG Query Engine  
└── Local FAISS Storage
```

### Data Flow

```
Document Input → OCR Processing → Text Chunking → Vector Embedding → FAISS Index
                                                                        ↓
User Query → Vector Retrieval → Context Assembly → LLM Generation → Response
```

### Technology Stack

- **Protocol**: Model Context Protocol (MCP) 1.0
- **Document Processing**: PyPDF2, Tesseract OCR
- **RAG Framework**: Haystack AI, Sentence Transformers
- **Vector Storage**: FAISS (local)
- **Language Model**: OpenAI API (configurable)
- **Package Management**: uvx compatible

## Usage Examples

### Document Processing

```python
# Process a research paper
result = tools.process_document({
    "file_path": "./papers/research_paper.pdf",
    "file_name": "Machine Learning Survey"
})
```

### Document Querying

```python
# Single query
result = tools.query_documents({
    "query": "What are the main contributions of this paper?",
    "top_k": 5
})

# Multi-turn conversation
result = tools.query_documents({
    "query": "How does this relate to previous work?",
    "session_id": "session_001",
    "top_k": 3
})
```

### Document Information Retrieval

```python
# Get document details
result = tools.get_document_info({
    "document_id": "doc_abc123"
})

# List active sessions
result = tools.list_sessions({})
```

## Configuration

### Server Configuration

Default configuration is stored in `config/config.json`:

```json
{
  "server": {
    "name": "academic-rag-server",
    "version": "1.0.0",
    "port": 8000
  },
  "mcp": {
    "tools_enabled": [
      "process_document",
      "query_documents", 
      "get_document_info",
      "list_sessions"
    ]
  },
  "storage": {
    "type": "local",
    "vector_store": "faiss",
    "base_path": "./data"
  }
}
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | OpenAI API authentication key |
| `MCP_PORT` | No | 8000 | Server listening port |
| `DATA_PATH` | No | ./data | Local data storage directory |
| `OCR_LANGUAGE` | No | eng | Tesseract OCR language code |
| `LOG_LEVEL` | No | INFO | Logging verbosity level |

## System Requirements

### Minimum Requirements

- **Operating System**: Linux, macOS, Windows
- **Python**: 3.9 or later
- **Memory**: 2GB RAM
- **Storage**: 500MB available space
- **Network**: Internet access for OpenAI API

### Recommended Requirements

- **Memory**: 4GB RAM or more
- **Storage**: 2GB available space
- **CPU**: Multi-core processor

### Dependencies

Core dependencies (14 packages):

```
mcp>=1.0.0                    # MCP protocol implementation
haystack-ai>=2.0.0            # RAG framework
sentence-transformers>=2.2.0  # Text embedding models
openai>=1.0.0                 # Language model API client
faiss-cpu>=1.7.0              # Vector similarity search
PyPDF2>=3.0.0                 # PDF document processing
pytesseract>=0.3.10           # OCR text recognition
Pillow>=10.0.0                # Image processing
requests>=2.31.0              # HTTP client
aiohttp>=3.9.0                # Async HTTP client
python-dotenv>=1.0.0          # Environment variable management
pydantic>=2.0.0               # Data validation
jsonschema>=4.0.0             # JSON schema validation
loguru>=0.7.0                 # Structured logging
```

## Testing

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run complete test suite
pytest tests/

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test categories
pytest tests/unit/           # Unit tests only
pytest tests/integration/    # Integration tests only
```

### Validation Scripts

```bash
# System validation
python test_final_validation.py

# Core functionality test
python test_mcp_core.py

# uvx compatibility test  
python test_uvx_install.py
```

## Deployment

### 🎆 Deployment Status: PRODUCTION READY

**✅ Validated**: All prerequisites and configuration files  
**✅ Built**: uvx package + Docker container artifacts  
**✅ Tested**: Health checks and core functionality  
**✅ Deployed**: Ready for immediate MCP integration  

---

### Production Deployment 🚀

#### Primary Method: uvx (Recommended)

```bash
# Deploy immediately (validated and ready)
cd mcp-academic-rag-server
uvx --from . mcp-academic-rag-server

# With custom configuration
OPENAI_API_KEY=your_key uvx --from . mcp-academic-rag-server
```

#### Container Deployment: Docker

```bash
# Production deployment with Docker
docker-compose -f docker-compose.simple.yml up -d

# Monitor deployment
docker-compose -f docker-compose.simple.yml logs -f mcp-academic-rag-server
```

### Development Environment

```bash
# Development install
pip install -e ".[dev]"

# Run with debug logging
LOG_LEVEL=DEBUG python mcp_server.py

# Run tests
pytest tests/
```

### Health Monitoring 📊

```bash
# Check deployment health
python health_check.py

# Verify MCP tools functionality
python test_mcp_core.py

# System validation (comprehensive)
python test_final_validation.py
```

### Deployment Verification

```bash
# Verify uvx deployment
uvx --from . mcp-academic-rag-server --help

# Verify Docker deployment
docker-compose -f docker-compose.simple.yml ps

# Test MCP client connection
# Configure claude_desktop_config.json with server details
```

## Performance

### Processing Benchmarks

- **PDF Processing**: ~3 seconds per page (standard documents)
- **OCR Processing**: ~5 seconds per page (image documents)
- **Query Response**: <2 seconds (typical queries)
- **Memory Usage**: ~500MB baseline

### Scalability Limits

- **Concurrent Document Processing**: Recommended maximum 3 documents
- **Concurrent Queries**: No hard limit
- **Document Size**: Recommended maximum 16MB
- **Session Management**: Automatic cleanup after 30 minutes of inactivity

## Troubleshooting

### Common Issues

**Document Processing Failures**
- Verify file path accessibility
- Ensure supported file format (PDF, JPG, PNG)
- Check available disk space and memory

**Query Returns Empty Results**
- Confirm documents have been successfully processed
- Verify query relevance to document content
- Adjust similarity threshold in configuration

**OCR Recognition Issues**
- Use high-resolution source documents (300+ DPI)
- Ensure clear, unblurred text
- Verify Tesseract language packs are installed

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
LOG_LEVEL=DEBUG python mcp_server.py
```

## Documentation

- [User Guide](docs/user-guide/mcp-tools-reference.md) - Comprehensive tool documentation
- [Developer Guide](docs/developer-guide.md) - Development and extension guide  
- [API Reference](docs/api-reference.md) - Complete API specification

## Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Install development dependencies: `pip install -e ".[dev]"`
4. Run tests: `pytest`
5. Submit a pull request

### Code Standards

- Follow PEP 8 style guidelines
- Maintain test coverage above 80%
- Update documentation for new features
- Ensure MCP protocol compliance

### Reporting Issues

Report bugs and feature requests through GitHub Issues. Include:

- Operating system and Python version
- Steps to reproduce the issue
- Expected vs actual behavior
- Relevant log output

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Model Context Protocol](https://modelcontextprotocol.io) for the MCP specification
- [Haystack](https://haystack.deepset.ai/) for the RAG framework
- [FAISS](https://faiss.ai/) for efficient vector similarity search
- [OpenAI](https://openai.com/) for language model capabilities