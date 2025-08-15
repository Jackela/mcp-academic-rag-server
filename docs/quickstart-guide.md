# Quick Start Guide - MCP Academic RAG Server

Get up and running with the MCP Academic RAG Server in under 10 minutes.

## Prerequisites

- Python 3.8+
- OpenAI API key
- Claude Desktop (for MCP integration)

## Installation

### 1. Clone and Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd mcp-academic-rag-server

# Install Python dependencies
pip install -r requirements.txt

# Verify installation
python -c "import mcp, haystack; print('✅ Dependencies installed')"
```

### 2. Environment Setup

```bash
# Choose ONE provider - OpenAI (default), Anthropic Claude, or Google Gemini

# Option A: OpenAI (default)
export OPENAI_API_KEY="sk-your-api-key-here"

# Option B: Anthropic Claude
export ANTHROPIC_API_KEY="sk-ant-your-api-key-here"
export LLM_PROVIDER="anthropic"
export LLM_MODEL="claude-3-sonnet-20240229"

# Option C: Google Gemini  
export GOOGLE_API_KEY="your-google-ai-key-here"
export LLM_PROVIDER="google"
export LLM_MODEL="gemini-1.5-pro"
```

**📚 Need help choosing?** See our [Multi-Model Setup Guide](docs/multi-model-setup-guide.md) for detailed comparisons.

### 3. Test Server

```bash
# Test server startup
python mcp_server.py --help

# Verify RAG pipeline
python -c "
from rag.haystack_pipeline import RAGPipeline
from connectors.haystack_llm_connector import HaystackLLMConnector
import os
connector = HaystackLLMConnector(api_key=os.getenv('OPENAI_API_KEY'))
pipeline = RAGPipeline(connector)
print('✅ RAG pipeline ready')
"
```

## Claude Desktop Integration

### 1. Configure Claude Desktop

Add to your Claude Desktop configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "academic-rag": {
      "command": "python",
      "args": ["/full/path/to/mcp_server.py"],
      "env": {
        "OPENAI_API_KEY": "sk-your-api-key-here"
      }
    }
  }
}
```

### 2. Restart Claude Desktop

Restart Claude Desktop to load the MCP server configuration.

### 3. Verify Connection

In Claude Desktop, you should see a connection indicator for the academic-rag server.

## Quick Usage Examples

### Example 1: Process a Research Paper

1. **Prepare a PDF file**: Place a PDF research paper in an accessible location
2. **Process the document**:

```
Hey Claude, I want to analyze a research paper. Can you process this document for me?

File path: /Users/researcher/papers/neural_networks_survey.pdf
```

Claude will use the `process_document` tool to:
- Extract text content from the PDF
- Generate semantic embeddings
- Index the document for querying

### Example 2: Query the Document

After processing, query the document:

```
What are the main contributions of this neural networks survey paper?
```

Claude will use the `query_documents` tool to:
- Convert your question to embeddings
- Find relevant sections in the processed document
- Generate a comprehensive answer with citations

### Example 3: Follow-up Questions

Continue the conversation with context:

```
Can you explain the transformer architecture mentioned in the paper in more detail?
```

The server maintains session context, so follow-up questions reference the same document.

## Testing the Setup

### 1. Manual Server Test

```bash
# Start server in test mode
python mcp_server.py &

# In another terminal, test with MCP client
python -c "
import asyncio
from mcp.client.stdio import stdio_client

async def test():
    async with stdio_client('python', ['mcp_server.py']) as client:
        # List available tools
        tools = await client.list_tools()
        print(f'Available tools: {[t.name for t in tools]}')
        
        # Test query (without document)
        result = await client.call_tool('query_documents', {'query': 'What is AI?'})
        print('Test query successful')

asyncio.run(test())
"
```

### 2. End-to-End Test with Sample Document

```bash
# Create a sample text document
echo "Machine learning is a subset of artificial intelligence that enables computers to learn without explicit programming." > sample_doc.txt

# Test processing
python -c "
import asyncio
from mcp.client.stdio import stdio_client

async def test_document():
    async with stdio_client('python', ['mcp_server.py']) as client:
        # Process document
        result = await client.call_tool('process_document', {
            'file_path': 'sample_doc.txt',
            'file_name': 'Sample ML Doc'
        })
        print('Document processed:', result)
        
        # Query document
        query_result = await client.call_tool('query_documents', {
            'query': 'What is machine learning?'
        })
        print('Query result:', query_result)

asyncio.run(test_document())
"
```

## Common Issues and Solutions

### Issue: "MCP package not found"

**Solution:**
```bash
pip install mcp
# or if using conda:
conda install -c conda-forge mcp
```

### Issue: "No OpenAI API key found"

**Solution:**
```bash
# Set environment variable
export OPENAI_API_KEY="sk-your-key-here"

# Or add to shell profile
echo 'export OPENAI_API_KEY="sk-your-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### Issue: "Haystack dependencies missing"

**Solution:**
```bash
pip install haystack-ai sentence-transformers openai
```

### Issue: "Claude Desktop not connecting"

**Solution:**
1. Verify configuration file path and JSON syntax
2. Check that the server path is absolute
3. Restart Claude Desktop completely
4. Check Claude Desktop logs for connection errors

### Issue: "Document processing fails"

**Solution:**
1. Verify file path is absolute and file exists
2. Check file permissions (read access required)
3. For PDFs, ensure file is not corrupted or password-protected
4. Check server logs in stderr output

## Configuration Customization

### Basic Configuration (`config/config.json`)

```json
{
  "llm": {
    "model": "gpt-3.5-turbo",
    "temperature": 0.7,
    "max_tokens": 1000
  },
  "rag_settings": {
    "retriever_top_k": 5,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
  }
}
```

### Performance Tuning

For faster responses during development:
```json
{
  "rag_settings": {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "retriever_top_k": 3
  },
  "llm": {
    "max_tokens": 500,
    "temperature": 0.3
  }
}
```

For better accuracy in production:
```json
{
  "rag_settings": {
    "embedding_model": "sentence-transformers/all-mpnet-base-v2",
    "retriever_top_k": 10
  },
  "llm": {
    "model": "gpt-4",
    "max_tokens": 2000,
    "temperature": 0.5
  }
}
```

## Next Steps

### 1. Process Your First Document
- Find a PDF research paper or document
- Use Claude Desktop to process it with the `process_document` tool
- Ask questions about the content

### 2. Explore Advanced Features
- Try different types of queries (summary, specific facts, analysis)
- Use session management for multi-turn conversations
- Experiment with different document types

### 3. Customize Configuration
- Adjust embedding models for your use case
- Tune LLM parameters for your preferred response style
- Configure document processing settings

### 4. Scale Your Usage
- Process multiple documents for comparative analysis
- Create different sessions for different research topics
- Monitor performance and optimize for your document sizes

## Useful Commands

```bash
# Check server status
python mcp_server.py --version

# Test specific components
python -m rag.haystack_pipeline  # Test RAG pipeline
python -m core.server_context     # Test server context

# View logs in real-time
python mcp_server.py 2>&1 | tail -f

# Performance testing
time python -c "
from rag.haystack_pipeline import RAGPipeline
from connectors.haystack_llm_connector import HaystackLLMConnector
import os
connector = HaystackLLMConnector(api_key=os.getenv('OPENAI_API_KEY'))
pipeline = RAGPipeline(connector)
result = pipeline.run('What is artificial intelligence?')
print('Response generated in:', result)
"
```

## Support Resources

- **Full Documentation**: See `docs/mcp-server-usage-guide.md`
- **API Reference**: See `docs/api-rag-pipeline.md`
- **Configuration**: See `config/config.json`
- **Troubleshooting**: Check stderr output for detailed error messages
- **Performance**: Monitor token usage and response times through OpenAI dashboard

## Ready to Use!

You're now ready to use the MCP Academic RAG Server with Claude Desktop. Start by processing your first document and asking questions about it!

**Pro Tip**: Start with shorter documents (5-10 pages) to get familiar with the system before processing longer academic papers.