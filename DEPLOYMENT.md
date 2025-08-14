# 🚀 MCP Academic RAG Server - Deployment Guide

Quick reference for deploying the MCP Academic RAG Server.

## ⚡ Quick Setup (2 Minutes)

### Prerequisites
- OpenAI API Key ([Get one here](https://platform.openai.com/api-keys))
- Claude Desktop (latest version)

### 1. Direct GitHub Deployment (Recommended)

**No local installation required!**

1. **Configure Claude Desktop**:
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

2. **Configuration file locations**:
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
   - **Linux**: `~/.config/claude/claude_desktop_config.json`

3. **Restart Claude Desktop** - Done! 🎉

### 2. Test Installation (Optional)

```bash
export OPENAI_API_KEY=sk-your-key-here
uvx --from git+https://github.com/yourusername/mcp-academic-rag-server mcp-academic-rag-server --validate-only
```

## 🔧 Alternative Deployment Methods

### Local Development
```bash
git clone https://github.com/yourusername/mcp-academic-rag-server
cd mcp-academic-rag-server
python deploy_secure.py
```

### Docker
```bash
echo "OPENAI_API_KEY=sk-your-key" > .env
docker-compose -f docker-compose.simple.yml up -d
```

## 🛠️ Troubleshooting

### Common Issues

**❌ "OPENAI_API_KEY not found"**
- Ensure API key is set in configuration
- API key must start with `sk-`

**❌ "uvx command not found"**
```bash
pip install uvx
```

**❌ "Claude Desktop not connecting"**
- Check JSON syntax in config file
- Restart Claude Desktop completely
- Check logs: `~/Library/Logs/Claude/mcp.log`

## 📚 Usage

Once deployed, use natural language in Claude Desktop:

- "Process this PDF document for analysis"
- "What are the main findings in the paper?"
- "Compare the methodology with previous research"

## 🔗 Links

- **Full Documentation**: [README.md](README.md)
- **Configuration Examples**: [claude_desktop_config.example.json](claude_desktop_config.example.json)
- **Issues**: [GitHub Issues](https://github.com/yourusername/mcp-academic-rag-server/issues)