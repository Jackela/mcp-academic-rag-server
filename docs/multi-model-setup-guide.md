# Multi-Model Setup Guide - OpenAI, Claude & Gemini Support

The MCP Academic RAG Server now supports multiple LLM providers: **OpenAI GPT**, **Anthropic Claude**, and **Google Gemini**. This guide shows how to configure and use different models.

## Supported Providers

### 🤖 OpenAI (Default)
- **Models**: GPT-3.5-turbo, GPT-4, GPT-4-turbo, GPT-4o, GPT-4o-mini
- **API Key**: `OPENAI_API_KEY`
- **Best For**: General purpose, coding, analysis

### 🧠 Anthropic Claude  
- **Models**: Claude-3-Opus, Claude-3-Sonnet, Claude-3-Haiku, Claude-3.5-Sonnet
- **API Key**: `ANTHROPIC_API_KEY`
- **Best For**: Long-form analysis, reasoning, creative tasks

### 🔍 Google Gemini
- **Models**: Gemini-Pro, Gemini-1.5-Pro, Gemini-1.5-Flash
- **API Key**: `GOOGLE_API_KEY`  
- **Best For**: Multimodal tasks, large context windows

## Quick Setup

### 1. Get API Keys

**OpenAI**: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
**Anthropic**: [console.anthropic.com](https://console.anthropic.com/)
**Google**: [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

### 2. Set Environment Variables

```bash
# Choose ONE provider or set multiple for switching
export OPENAI_API_KEY="sk-your-openai-key-here"
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-key-here" 
export GOOGLE_API_KEY="your-google-ai-key-here"
```

### 3. Configure Provider in `config/config.json`

```json
{
  "llm": {
    "provider": "openai",
    "model": "gpt-3.5-turbo",
    "api_key": "${OPENAI_API_KEY}",
    "parameters": {
      "temperature": 0.1,
      "max_tokens": 500
    }
  }
}
```

## Configuration Examples

### OpenAI Configuration

```json
{
  "llm": {
    "provider": "openai",
    "model": "gpt-4",
    "api_key": "${OPENAI_API_KEY}",
    "parameters": {
      "temperature": 0.1,
      "max_tokens": 1000
    }
  }
}
```

**Available Models:**
- `gpt-3.5-turbo` - Fast, cost-effective
- `gpt-4` - Higher quality reasoning  
- `gpt-4-turbo` - Improved performance
- `gpt-4o` - Latest multimodal model
- `gpt-4o-mini` - Faster, lighter version

### Anthropic Claude Configuration

```json
{
  "llm": {
    "provider": "anthropic", 
    "model": "claude-3-sonnet-20240229",
    "api_key": "${ANTHROPIC_API_KEY}",
    "parameters": {
      "temperature": 0.1,
      "max_tokens": 1000
    }
  }
}
```

**Available Models:**
- `claude-3-haiku-20240307` - Fast, economical
- `claude-3-sonnet-20240229` - Balanced performance  
- `claude-3-opus-20240229` - Most capable
- `claude-3-5-sonnet-20241022` - Latest improved model
- `claude-3-5-haiku-20241022` - Fast latest model

### Google Gemini Configuration

```json
{
  "llm": {
    "provider": "google",
    "model": "gemini-1.5-pro", 
    "api_key": "${GOOGLE_API_KEY}",
    "parameters": {
      "temperature": 0.1,
      "max_output_tokens": 1000
    }
  }
}
```

**Available Models:**
- `gemini-pro` - Standard model
- `gemini-pro-vision` - Multimodal capabilities
- `gemini-1.5-pro` - Large context window (2M tokens)
- `gemini-1.5-flash` - Fast, efficient
- `gemini-1.5-flash-8b` - Lightweight version

## Claude Desktop Configuration

### Single Provider Setup

Choose one provider and set up Claude Desktop:

**OpenAI:**
```json
{
  "mcpServers": {
    "academic-rag": {
      "command": "python",
      "args": ["/path/to/mcp_server.py"],
      "env": {
        "OPENAI_API_KEY": "sk-your-openai-key-here"
      }
    }
  }
}
```

**Anthropic Claude:**
```json
{
  "mcpServers": {
    "academic-rag": {
      "command": "python", 
      "args": ["/path/to/mcp_server.py"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-your-anthropic-key-here",
        "LLM_PROVIDER": "anthropic",
        "LLM_MODEL": "claude-3-sonnet-20240229"
      }
    }
  }
}
```

**Google Gemini:**
```json
{
  "mcpServers": {
    "academic-rag": {
      "command": "python",
      "args": ["/path/to/mcp_server.py"], 
      "env": {
        "GOOGLE_API_KEY": "your-google-ai-key-here",
        "LLM_PROVIDER": "google",
        "LLM_MODEL": "gemini-1.5-pro"
      }
    }
  }
}
```

### Multiple Provider Setup

Set multiple API keys and switch by changing environment variables:

```json
{
  "mcpServers": {
    "academic-rag": {
      "command": "python",
      "args": ["/path/to/mcp_server.py"],
      "env": {
        "OPENAI_API_KEY": "sk-your-openai-key-here",
        "ANTHROPIC_API_KEY": "sk-ant-your-anthropic-key-here",  
        "GOOGLE_API_KEY": "your-google-ai-key-here",
        "LLM_PROVIDER": "anthropic",
        "LLM_MODEL": "claude-3-sonnet-20240229"
      }
    }
  }
}
```

## Environment Variable Override

You can override config settings with environment variables:

```bash
# Override provider and model
export LLM_PROVIDER="anthropic"
export LLM_MODEL="claude-3-sonnet-20240229"

# Override parameters
export LLM_TEMPERATURE="0.3"
export LLM_MAX_TOKENS="1500"
```

## Model Comparison

| Provider | Model | Speed | Quality | Context | Cost | Best Use Case |
|----------|-------|--------|---------|---------|------|---------------|
| OpenAI | GPT-3.5-turbo | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 16K | $ | General tasks |
| OpenAI | GPT-4 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 128K | $$$ | Complex reasoning |
| OpenAI | GPT-4o | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 128K | $$ | Balanced performance |
| Anthropic | Claude-3-Haiku | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 200K | $ | Fast responses |
| Anthropic | Claude-3-Sonnet | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 200K | $$ | Balanced quality |
| Anthropic | Claude-3-Opus | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 200K | $$$$ | Complex analysis |
| Google | Gemini-Pro | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 32K | $ | Multimodal tasks |
| Google | Gemini-1.5-Pro | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 2M | $$$ | Large documents |

## Installation Requirements

### Base Requirements (All Providers)
```bash
pip install mcp haystack-ai sentence-transformers
```

### Provider-Specific Dependencies

**OpenAI:**
```bash
pip install openai  # Usually included with haystack-ai
```

**Anthropic:**
```bash
pip install anthropic
```

**Google:**
```bash
pip install google-generativeai
```

**Install All:**
```bash
pip install openai anthropic google-generativeai
```

## Testing Different Models

### 1. Quick Test Script

```python
from connectors.llm_factory import LLMFactory

# Test OpenAI
openai_config = {
    "provider": "openai",
    "model": "gpt-3.5-turbo",
    "api_key": "your-openai-key",
    "parameters": {"temperature": 0.1, "max_tokens": 100}
}

connector = LLMFactory.create_from_unified_config(openai_config)
response = connector.generate([{"role": "user", "content": "What is machine learning?"}])
print(f"OpenAI: {response['content']}")

# Test Anthropic
anthropic_config = {
    "provider": "anthropic", 
    "model": "claude-3-sonnet-20240229",
    "api_key": "your-anthropic-key",
    "parameters": {"temperature": 0.1, "max_tokens": 100}
}

connector = LLMFactory.create_from_unified_config(anthropic_config)
response = connector.generate([{"role": "user", "content": "What is machine learning?"}])
print(f"Anthropic: {response['content']}")
```

### 2. Model Validation

```python
from connectors.llm_factory import LLMFactory

# Check supported providers
providers = LLMFactory.get_supported_providers()
print("Supported providers:", providers)

# Check supported models for a provider  
openai_models = LLMFactory.get_supported_models("openai")
print("OpenAI models:", openai_models)

# Validate configuration
config = {"provider": "anthropic", "model": "claude-3-sonnet-20240229"}
validation = LLMFactory.validate_config("anthropic", config)
print("Config valid:", validation["valid"])
```

## Troubleshooting

### Common Issues

**"API key not found":**
```bash
# Check environment variables
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY  
echo $GOOGLE_API_KEY

# Set missing keys
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

**"Model not supported":**
- Check supported models with `LLMFactory.get_supported_models("provider")`
- Update model name in config.json
- Verify model name spelling and case

**"Dependencies missing":**
```bash
pip install anthropic  # For Claude models
pip install google-generativeai  # For Gemini models
```

**"Rate limits exceeded":**
- Switch to different provider temporarily
- Check API usage quotas
- Implement exponential backoff

### Debug Mode

Enable detailed logging to debug model issues:

```json
{
  "logging": {
    "level": "DEBUG"
  }
}
```

Or set environment variable:
```bash
export LOG_LEVEL=DEBUG
```

## Best Practices

### 1. Provider Selection Guidelines

- **OpenAI GPT-4o**: General purpose, balanced performance
- **Claude-3-Sonnet**: Long document analysis, academic papers  
- **Gemini-1.5-Pro**: Very large documents (>100 pages)
- **GPT-3.5-turbo**: Development, testing, cost-sensitive

### 2. Parameter Tuning

**For Academic Documents:**
```json
{
  "parameters": {
    "temperature": 0.1,
    "max_tokens": 1000
  }
}
```

**For Creative Tasks:**
```json
{
  "parameters": {
    "temperature": 0.7,
    "max_tokens": 1500
  }
}
```

### 3. Cost Optimization

- Use GPT-3.5-turbo for development
- Use Claude-Haiku for fast responses
- Use Gemini-Flash for efficient processing
- Monitor token usage across providers

### 4. Switching Models

To switch models without restarting:

1. Update `config/config.json`
2. Set environment variable: `LLM_PROVIDER=anthropic`  
3. Restart Claude Desktop
4. Or use environment override in Claude Desktop config

## Advanced Configuration

### Custom Parameters by Provider

```json
{
  "llm_providers": {
    "openai": {
      "default_parameters": {
        "temperature": 0.1,
        "max_tokens": 1000,
        "presence_penalty": 0.1
      }
    },
    "anthropic": {
      "default_parameters": {
        "temperature": 0.1,
        "max_tokens": 1500,
        "stop_sequences": ["Human:", "Assistant:"]
      }
    },
    "google": {
      "default_parameters": {
        "temperature": 0.1,
        "max_output_tokens": 1200,
        "top_p": 0.8
      }
    }
  }
}
```

### Model-Specific Optimization

```json
{
  "llm": {
    "provider": "anthropic",
    "model": "claude-3-5-sonnet-20241022",
    "parameters": {
      "temperature": 0.0,
      "max_tokens": 2000
    }
  }
}
```

## Support and Resources

- **OpenAI Documentation**: [platform.openai.com/docs](https://platform.openai.com/docs)
- **Anthropic Documentation**: [docs.anthropic.com](https://docs.anthropic.com/)  
- **Google AI Documentation**: [ai.google.dev/docs](https://ai.google.dev/docs)
- **API Rate Limits**: Check each provider's documentation
- **Pricing**: Compare costs on each provider's website

Ready to use multiple AI models with your academic RAG server! 🚀