# 🚀 MCP Academic RAG Server - 一键安装指南

## ⚡ 超快安装（30秒完成）

### **步骤1: 获取OpenAI API密钥** (30秒)
1. 访问 [OpenAI Platform](https://platform.openai.com/api-keys)
2. 创建API密钥 (格式: `sk-xxxxxxxx...`)

### **步骤2: 配置Claude Desktop** (30秒)

**📍 配置文件位置:**
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux**: `~/.config/claude/claude_desktop_config.json`

**📋 复制粘贴配置:**
```json
{
  "mcpServers": {
    "academic-rag": {
      "command": "uvx",
      "args": ["mcp-academic-rag-server"],
      "env": {
        "OPENAI_API_KEY": "sk-your-actual-api-key-here"
      }
    }
  }
}
```

⚠️ **重要**: 将 `sk-your-actual-api-key-here` 替换为你的真实API密钥

### **步骤3: 重启Claude Desktop** (5秒)

重启Claude Desktop，完成！🎉

---

## 🎯 立即使用

配置完成后，在Claude Desktop中直接使用：

```
📄 "请处理这个PDF文件：C:\path\to\document.pdf"
❓ "这篇论文的主要观点是什么？"  
📊 "帮我总结文档中的重要信息"
💬 "基于已处理的文档回答：机器学习的最新进展有哪些？"
```

---

## 🔧 高级配置 (可选)

### **自定义数据路径**
```json
{
  "mcpServers": {
    "academic-rag": {
      "command": "uvx",
      "args": ["mcp-academic-rag-server"],
      "env": {
        "OPENAI_API_KEY": "sk-your-api-key-here",
        "DATA_PATH": "D:\\MyDocuments\\RAG_Data",
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

### **多环境配置**
```json
{
  "mcpServers": {
    "academic-rag-prod": {
      "command": "uvx",
      "args": ["mcp-academic-rag-server"],
      "env": {
        "OPENAI_API_KEY": "sk-production-key",
        "DATA_PATH": "C:\\Production\\Data",
        "LOG_LEVEL": "WARNING"
      }
    },
    "academic-rag-test": {
      "command": "uvx", 
      "args": ["mcp-academic-rag-server"],
      "env": {
        "OPENAI_API_KEY": "sk-test-key",
        "DATA_PATH": "C:\\Test\\Data",
        "LOG_LEVEL": "DEBUG"
      }
    }
  }
}
```

---

## 🛠️ 故障排查

### **问题1: 命令找不到**
```bash
# 解决方案：确保uvx已安装
pip install uvx
```

### **问题2: API密钥无效**
- 确保API密钥以 `sk-` 开头
- 确保密钥长度足够 (51个字符)
- 检查OpenAI账户余额

### **问题3: 文档处理失败**
- 确保文档路径正确
- 支持格式：PDF, PNG, JPG, TXT
- 检查文件权限

### **查看详细日志**
**Windows**: `%APPDATA%\Claude\logs\mcp.log`
**macOS**: `~/Library/Logs/Claude/mcp.log`  
**Linux**: `~/.local/share/claude/logs/mcp.log`

---

## 📚 功能说明

| 功能 | 命令示例 | 说明 |
|------|----------|------|
| 文档处理 | `"处理文档：/path/to/file.pdf"` | OCR识别、结构分析、向量化 |
| 智能问答 | `"论文中提到了什么方法？"` | 基于文档内容的AI问答 |
| 文档信息 | `"显示文档处理状态"` | 查看已处理文档的详细信息 |
| 会话管理 | `"列出我的所有对话"` | 查看历史对话记录 |

---

## 🆘 获取帮助

- **GitHub Issues**: https://github.com/yourusername/mcp-academic-rag-server/issues
- **文档**: [README.md](README.md)
- **示例**: [examples/](examples/)

完成配置后，你就拥有了一个强大的学术文档AI助手！🚀