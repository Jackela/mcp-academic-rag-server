# 🚀 Quick Setup Guide - MCP Academic RAG Server

**Core MCP Server Setup in Under 5 Minutes**

---

## 📋 Prerequisites

- **Python 3.9+** - [Download Python](https://python.org/downloads/)
- **uvx** - Python package runner (recommended for MCP servers)
- **OpenAI API Key** - For embeddings and LLM processing

### Install uvx (if not already installed)

```bash
# Install uvx using pip
pip install uvx

# Or using pipx (if you prefer)
pipx install uvx
```

---

## ⚡ Quick Installation

### Method 1: Direct uvx Installation (Recommended)

```bash
# Install and run the MCP server directly
uvx mcp-academic-rag-server
```

That's it! The MCP server is now running and ready for use.

### Method 2: Development Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mcp-academic-rag-server.git
cd mcp-academic-rag-server

# Install in development mode
uvx --editable . mcp-academic-rag-server
```

---

## 🔧 Basic Configuration

Create a `.env` file in your project directory:

```env
# Required: OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Custom configuration
CONFIG_PATH=config/config.json
DOCUMENTS_PATH=./documents
LOG_LEVEL=INFO
```

**Default configuration works out of the box** - no complex setup required!

---

## 🤖 MCP Client Setup

### For Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "academic-rag": {
      "command": "uvx",
      "args": ["mcp-academic-rag-server"]
    }
  }
}
```

### For Other MCP Clients

The server runs on standard MCP protocol - compatible with any MCP client.

---

## 🧪 Verify Installation

Test your installation:

```bash
# Check if server starts correctly
uvx mcp-academic-rag-server --help

# Run a quick health check (if available)
uvx mcp-academic-rag-server --health-check
```

---

## 📖 Quick Usage Examples

Once connected to your MCP client:

### Process a Document
```
Please process this PDF: /path/to/document.pdf
```

### Query Documents
```
What are the main findings in the processed documents about machine learning?
```

### Get Document Info
```
Show me information about document doc_123456
```

---

## 🔍 Troubleshooting

### Common Issues

1. **"mcp package not found"**
   ```bash
   # Install MCP package
   pip install mcp
   ```

2. **"OpenAI API key missing"**
   - Set your API key in the `.env` file
   - Or set environment variable: `export OPENAI_API_KEY=your_key`

3. **"Server won't start"**
   ```bash
   # Check Python version
   python --version  # Should be 3.9+
   
   # Reinstall if needed
   uvx --force mcp-academic-rag-server
   ```

### Need Help?

- Check [Issues](https://github.com/yourusername/mcp-academic-rag-server/issues)
- See [Full Documentation](README.md) for advanced configuration

---

## 🎯 What You Get

With this basic setup, you have:

✅ **4 MCP Tools** ready to use
- `process_document` - OCR and text extraction
- `query_documents` - RAG-based question answering  
- `get_document_info` - Document metadata
- `list_sessions` - Chat session management

✅ **Basic Document Processing**
- PDF text extraction
- Simple OCR for images
- Text chunking and embeddings

✅ **Simple RAG System**
- Semantic search over documents
- Context-aware responses
- Session-based conversations

---

## ⏱️ Performance Expectations

- **Setup Time**: < 5 minutes
- **First Document**: ~30-60 seconds processing
- **Query Response**: < 5 seconds typically
- **Memory Usage**: ~200-500MB for basic operation

---

## 🔄 Next Steps

Once basic functionality is working:

1. **Process your first document** using the MCP tools
2. **Test queries** to verify RAG functionality
3. **Explore configuration** options if needed
4. **Consider optional enhancements** (see pyproject.toml extras)

---

*This setup focuses on getting the core MCP server running quickly with minimal complexity. Advanced features can be added later as needed.*