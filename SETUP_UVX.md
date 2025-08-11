# MCP Academic RAG Server - uvx Setup Guide

🎆 **Status**: PRODUCTION READY - Deployment validated and tested  
🚀 **Quick Deploy**: Ready for immediate deployment with uvx  
🔧 **Setup Time**: 5 minutes from clone to running  

This guide shows how to install and run the MCP Academic RAG Server using `uvx` for simple, isolated package management.

## Prerequisites

- **Python 3.9+** installed (tested with 3.13.5)
- **uvx package manager**: `pip install uvx` or `pipx install uv`
- **OpenAI API key** for RAG functionality
- **Git** for cloning the repository

## 🚀 Production Deployment (5-Minute Setup)

### Step 1: Clone and Setup
```bash
# Clone the repository
git clone <repository-url>
cd mcp-academic-rag-server

# Verify deployment readiness
echo "Checking deployment status..."
ls pyproject.toml mcp_server.py  # Should exist
```

### Step 2: Deploy with uvx (Recommended) ⭐
```bash
# Production deployment - validated and ready
uvx --from . mcp-academic-rag-server

# With environment variables
OPENAI_API_KEY=your_key uvx --from . mcp-academic-rag-server
```

### Alternative: Development Mode
```bash
# For development and debugging
pip install -e ".[dev]"
python mcp_server.py
```

## Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your OpenAI API key:
```bash
OPENAI_API_KEY=your-openai-api-key-here
```

3. (Optional) Customize settings in `config/config.json`

## Running the Server

```bash
# Run the MCP server
uvx --from . mcp-academic-rag-server

# Or with specific port
uvx --from . mcp-academic-rag-server --port 8001
```

## Verification

The server provides 4 core MCP tools:
1. `process_document` - Process PDF/text documents
2. `query_documents` - Query processed documents using RAG
3. `get_document_info` - Get metadata about processed documents
4. `list_sessions` - List active chat sessions

## Troubleshooting

### Dependencies
If you encounter dependency issues, try:
```bash
# Install with specific Python version
uvx --python 3.11 --from . mcp-academic-rag-server
```

### Tesseract OCR
On Windows, you may need to install Tesseract separately:
1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
2. Add to PATH or set `TESSERACT_CMD` environment variable

### OpenAI API
Ensure your OpenAI API key is valid and has sufficient quota.

## 📋 Deployment Verification

### Verify Successful Deployment
```bash
# Check deployment health
python health_check.py

# Test MCP tools functionality
python test_mcp_core.py

# Full system validation
python test_final_validation.py
```

### Connect MCP Client (Claude Desktop)
1. Configure `claude_desktop_config.json`:
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

2. Restart Claude Desktop
3. Test with sample document processing

## 🔧 Development Mode

For development and debugging:
```bash
# Development install with all dependencies
pip install -e ".[dev]"

# Run with debug logging
LOG_LEVEL=DEBUG python mcp_server.py

# Run test suite
pytest tests/
```

## 📺 Alternative: Docker Deployment

For containerized deployment:
```bash
# Production container deployment
docker-compose -f docker-compose.simple.yml up -d

# Monitor logs
docker-compose logs -f mcp-academic-rag-server

# Health check
docker-compose ps
```

## 🗑️ Uninstallation

```bash
# uvx manages isolation automatically - no system pollution
# Simply stop the process and delete project directory if needed
# No system-wide cleanup required
```