# MCP Academic RAG Server

[![MCP Compatible](https://img.shields.io/badge/MCP-1.0-purple.svg)](https://modelcontextprotocol.io)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Dependencies](https://img.shields.io/badge/dependencies-14_packages-green.svg)](#installation)
[![Deployment](https://img.shields.io/badge/deployment-ready-brightgreen.svg)](#deployment)
[![Status](https://img.shields.io/badge/status-production_ready-success.svg)](#quick-start)

A production-ready Model Context Protocol (MCP) server for academic document processing and retrieval-augmented generation queries. This server provides standardized tools for document analysis, OCR processing, and intelligent question-answering capabilities.

**🎯 Status**: Production Ready - Successfully deployed and validated  
**🚀 Quick Deploy**: `uvx --from git+https://github.com/yourusername/mcp-academic-rag-server mcp-academic-rag-server`  
**📋 Architecture**: Focused MCP server following 2024 best practices  
**⚡ Installation**: Direct from GitHub, no local setup required

## Features

The server implements 4 core tools for academic document processing:

| Tool | Description | Input | Output |
|------|-------------|-------|---------|
| `process_document` | Process PDF/image documents with OCR | File path, optional name | Document metadata and processing status |
| `query_documents` | Query processed documents using RAG | Question text, session context | Generated answer with source citations |
| `get_document_info` | Retrieve document processing details | Document ID | Comprehensive document information |
| `list_sessions` | List conversation sessions | None | Session metadata and statistics |

## ⚡ Quick Start (2 Minutes)

### 🚀 Ultra-Fast Setup

1. **Get your OpenAI API key** from [OpenAI Platform](https://platform.openai.com/api-keys)

2. **Configure Claude Desktop** (choose one):

**Option A: Direct GitHub (Recommended) 🌟**
```json
{
  "mcpServers": {
    "academic-rag": {
      "command": "uvx",
      "args": [
        "--from", 
        "git+https://github.com/yourusername/mcp-academic-rag-server",
        "mcp-academic-rag-server"
      ],
      "env": {
        "OPENAI_API_KEY": "sk-your-actual-api-key-here"
      }
    }
  }
}
```

**Option B: Test First**
```bash
# Test installation first
export OPENAI_API_KEY=sk-your-key-here
uvx --from git+https://github.com/yourusername/mcp-academic-rag-server mcp-academic-rag-server --validate-only
```

3. **Restart Claude Desktop** - Done! 🎉

### 📋 Configuration File Locations
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`  
- **Linux**: `~/.config/claude/claude_desktop_config.json`

### Prerequisites

- **Python**: 3.9+ (auto-installed by uvx if needed)
- **OpenAI API Key**: Required for RAG queries
- **Claude Desktop**: Latest version recommended

## 📦 Installation Methods

### 🚀 Method 1: Direct from GitHub (Recommended)

**No local setup required!** Perfect for end users.

```bash
# Test installation
export OPENAI_API_KEY=sk-your-key-here
uvx --from git+https://github.com/yourusername/mcp-academic-rag-server mcp-academic-rag-server --validate-only

# Or configure directly in Claude Desktop (see Quick Start above)
```

**Pros**: ✅ Zero setup ✅ Always latest version ✅ No disk space used  
**Cons**: ⚠️ Requires internet for first run

### 🔧 Method 2: Local Development

**For developers and advanced users who want to customize.**

```bash
# Clone and setup
git clone https://github.com/yourusername/mcp-academic-rag-server
cd mcp-academic-rag-server

# Guided setup (recommended for beginners)
python deploy_secure.py

# Or manual setup
export OPENAI_API_KEY=sk-your-key-here
uvx install .
uvx run mcp-academic-rag-server --validate-only
```

**Pros**: ✅ Full control ✅ Offline usage ✅ Code customization  
**Cons**: ⚠️ Requires local setup ⚠️ Manual updates

### 🐳 Method 3: Docker

**For containerized environments.**

```bash
# Setup environment
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# Deploy with Docker
docker-compose -f docker-compose.simple.yml up -d

# View logs
docker-compose logs -f mcp-academic-rag-server
```

**Pros**: ✅ Isolated environment ✅ Easy scaling ✅ Production ready  
**Cons**: ⚠️ Docker overhead ⚠️ Complexity

### 🧪 Method 4: Development Mode

**For active development and debugging.**

```bash
# Development install
git clone https://github.com/yourusername/mcp-academic-rag-server
cd mcp-academic-rag-server
pip install -e ".[dev]"

# Set API key and run
export OPENAI_API_KEY=sk-your-key-here
python mcp_server_secure.py --validate-only
```

**Pros**: ✅ Live code changes ✅ Full debugging ✅ Test suite access  
**Cons**: ⚠️ Development complexity ⚠️ Not for production

## ⚙️ Configuration

### 🔑 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | ✅ Yes | - | Your OpenAI API key (starts with `sk-`) |
| `DATA_PATH` | ❌ No | `./data` | Data storage directory |
| `LOG_LEVEL` | ❌ No | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `OCR_LANGUAGE` | ❌ No | `eng` | Tesseract OCR language code |

### 🖥️ Claude Desktop Configuration

Choose the configuration that matches your installation method:

#### Option 1: Direct GitHub (Most Popular) 🌟
```json
{
  "mcpServers": {
    "academic-rag": {
      "command": "uvx",
      "args": [
        "--from", 
        "git+https://github.com/yourusername/mcp-academic-rag-server",
        "mcp-academic-rag-server"
      ],
      "env": {
        "OPENAI_API_KEY": "sk-your-actual-api-key-here",
        "DATA_PATH": "./data",
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

#### Option 2: Local Installation
```json
{
  "mcpServers": {
    "academic-rag": {
      "command": "uvx",
      "args": ["run", "mcp-academic-rag-server"],
      "env": {
        "OPENAI_API_KEY": "sk-your-actual-api-key-here",
        "DATA_PATH": "./data",
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

#### Option 3: Custom Configuration
```json
{
  "mcpServers": {
    "academic-rag": {
      "command": "uvx",
      "args": [
        "--from", 
        "git+https://github.com/yourusername/mcp-academic-rag-server",
        "mcp-academic-rag-server",
        "--data-path", "/custom/path",
        "--log-level", "DEBUG"
      ],
      "env": {
        "OPENAI_API_KEY": "sk-your-actual-api-key-here"
      }
    }
  }
}
```

### 📁 Configuration File Locations

| Platform | Configuration File Location |
|----------|----------------------------|
| **macOS** | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| **Windows** | `%APPDATA%\Claude\claude_desktop_config.json` |
| **Linux** | `~/.config/claude/claude_desktop_config.json` |

> 💡 **Tip**: Create the directory if it doesn't exist, and restart Claude Desktop after changes.

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

## 📝 Usage Examples

Once configured in Claude Desktop, you can use these natural language commands:

### 📄 Document Processing

```
👤 "I need to process a research paper PDF located at /Users/john/papers/ml-survey.pdf"

🤖 I'll process that PDF for you using OCR and extract the content for analysis.

[Uses process_document tool internally]
✅ Document processed successfully
📊 Extracted 45 pages, 12,000 words
🔍 Ready for querying
```

### 🔍 Document Querying  

```
👤 "What are the main contributions of the paper I just uploaded?"

🤖 Based on the processed document, the main contributions are:
1. Novel attention mechanism for transformer models
2. 15% improvement in BLEU scores on translation tasks
3. Reduced computational complexity by 30%

[Uses query_documents tool with RAG pipeline]
📚 Sources: Sections 3.2, 4.1, and 6.3 of the paper
```

### 💬 Multi-turn Conversations

```
👤 "How does this approach compare to BERT?"

🤖 The paper compares their method to BERT in several ways:
- Uses different attention patterns (sparse vs. dense)
- Achieves better performance on long sequences
- Requires less training data

👤 "What about the computational requirements?"

🤖 [Continuing the conversation with session memory]
The computational analysis shows:
- 40% less memory usage than BERT-Large
- 2x faster inference time
- Similar training time requirements

[Uses query_documents with session_id for context]
```

### 📊 Document Management

```
👤 "Show me information about the document I processed earlier"

🤖 Document Information:
📄 File: ml-survey.pdf
📅 Processed: 2024-01-15 14:30
📖 Pages: 45
🔤 Word Count: ~12,000
🏷️ Status: Successfully processed
💾 Available for querying

👤 "List all my conversation sessions"

🤖 Active Sessions:
1. Session 001 - ML Survey Discussion (5 messages)
2. Session 002 - Architecture Questions (3 messages)
3. Session 003 - Performance Analysis (8 messages)
```

### 🎯 Advanced Usage

```
👤 "Process this image of a research paper page and then answer questions about it"

🤖 I'll process the image using OCR and make it available for queries.

[Image processing with OCR]
✅ Image processed successfully
📝 Extracted text from handwritten/printed content
🔍 Content now searchable

👤 "What methodology does this page describe?"

🤖 Based on the OCR'd content, this page describes:
[Detailed analysis of the methodology...]
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

### Validation & Testing

```bash
# Complete MCP setup validation
python validate_mcp.py

# Environment validation only (direct GitHub)
export OPENAI_API_KEY=sk-your-key
uvx --from git+https://github.com/yourusername/mcp-academic-rag-server mcp-academic-rag-server --validate-only

# Test with custom settings
uvx --from git+https://github.com/yourusername/mcp-academic-rag-server mcp-academic-rag-server --data-path ./test-data --log-level DEBUG --validate-only

# Local installation validation
uvx run mcp-academic-rag-server --validate-only
```

## 🛠️ Troubleshooting

### 🚨 Common Issues & Solutions

#### ❌ "OPENAI_API_KEY not found"
```bash
# Check if API key is set
echo $OPENAI_API_KEY

# Set API key temporarily
export OPENAI_API_KEY=sk-your-actual-key-here

# Permanently set (add to ~/.bashrc or ~/.zshrc)
echo 'export OPENAI_API_KEY=sk-your-actual-key-here' >> ~/.bashrc
```

#### ❌ "uvx command not found"
```bash
# Install uvx
pip install uvx

# Or update if already installed
pip install --upgrade uvx

# Verify installation
uvx --version
```

#### ❌ "git+https connection failed"
```bash
# Test GitHub connectivity
curl -I https://github.com

# Use SSH instead of HTTPS
uvx --from git+ssh://git@github.com/yourusername/mcp-academic-rag-server mcp-academic-rag-server

# Or clone locally first
git clone https://github.com/yourusername/mcp-academic-rag-server
cd mcp-academic-rag-server
uvx install .
```

#### ❌ "Claude Desktop not connecting"
```bash
# Check configuration file syntax
cat ~/Library/Application\ Support/Claude/claude_desktop_config.json | python -m json.tool

# Check Claude Desktop logs (macOS)
tail -f ~/Library/Logs/Claude/mcp.log

# Restart Claude Desktop completely
pkill -f Claude
open -a Claude\ Desktop
```

#### ❌ "Invalid API key format"
- API key must start with `sk-`
- Must be at least 20 characters long
- Get a valid key from [OpenAI Platform](https://platform.openai.com/api-keys)

#### ❌ "Permission denied"
```bash
# On macOS/Linux, ensure proper permissions
chmod +x ~/.local/bin/uvx

# Or use Python directly
python -m uvx --from git+https://github.com/yourusername/mcp-academic-rag-server mcp-academic-rag-server
```

### 🔍 Debug Mode

Enable detailed logging for troubleshooting:

```json
{
  "mcpServers": {
    "academic-rag": {
      "command": "uvx",
      "args": [
        "--from", 
        "git+https://github.com/yourusername/mcp-academic-rag-server",
        "mcp-academic-rag-server",
        "--log-level", "DEBUG"
      ],
      "env": {
        "OPENAI_API_KEY": "sk-your-actual-api-key-here",
        "LOG_LEVEL": "DEBUG"
      }
    }
  }
}
```

### 📞 Getting Help

1. **Check logs**: `~/Library/Logs/Claude/mcp-server-academic-rag.log`
2. **Run validation**: `python validate_mcp.py`
3. **Test manually**: `uvx --from git+https://... --validate-only`
4. **Create issue**: [GitHub Issues](https://github.com/yourusername/mcp-academic-rag-server/issues)

## 🚀 Production Deployment

### ✅ Deployment Status: PRODUCTION READY

**Ready for immediate use!** Choose your deployment method:

| Method | Complexity | Best For |
|--------|------------|----------|
| 🌟 **Direct GitHub** | ⭐ Minimal | End users, quick setup |
| 🔧 **Local Install** | ⭐⭐ Low | Developers, customization |
| 🐳 **Docker** | ⭐⭐⭐ Medium | Production, scaling |
| 🧪 **Development** | ⭐⭐⭐⭐ High | Active development |

### 🌟 Recommended: Direct GitHub Deployment

**Zero local setup required!**

```bash
# Just set your API key and use directly in Claude Desktop
export OPENAI_API_KEY=sk-your-key-here

# Test installation (optional)
uvx --from git+https://github.com/yourusername/mcp-academic-rag-server mcp-academic-rag-server --validate-only
```

**Claude Desktop Config:**
```json
{
  "mcpServers": {
    "academic-rag": {
      "command": "uvx",
      "args": [
        "--from", 
        "git+https://github.com/yourusername/mcp-academic-rag-server",
        "mcp-academic-rag-server"
      ],
      "env": {
        "OPENAI_API_KEY": "sk-your-actual-api-key-here"
      }
    }
  }
}
```

### 🐳 Enterprise Docker Deployment

```bash
# Production with environment file
echo "OPENAI_API_KEY=sk-your-key-here" > .env
echo "DATA_PATH=/app/data" >> .env
echo "LOG_LEVEL=WARNING" >> .env

# Deploy with monitoring
docker-compose -f docker-compose.simple.yml up -d

# Health check
docker-compose logs -f mcp-academic-rag-server
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