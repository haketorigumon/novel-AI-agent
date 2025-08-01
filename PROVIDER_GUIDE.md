# LLM Provider Configuration Guide

Novel AI Agent supports multiple LLM providers. This guide explains how to configure each provider.

## 🌟 Supported Providers

| Provider | API Key Required | Popular Models | Notes |
|----------|------------------|----------------|-------|
| **Ollama** | ❌ No | llama3, codellama, mistral | Local models, free |
| **OpenAI** | ✅ Yes | gpt-4, gpt-3.5-turbo | High quality, paid |
| **Anthropic** | ✅ Yes | claude-3-opus, claude-3-sonnet | Excellent reasoning |
| **Google** | ✅ Yes | gemini-pro, gemini-1.5-pro | Multimodal support |
| **Azure OpenAI** | ✅ Yes | gpt-4, gpt-35-turbo | Enterprise OpenAI |
| **Groq** | ✅ Yes | llama3-70b-8192, mixtral-8x7b | Ultra-fast inference |
| **Together AI** | ✅ Yes | meta-llama/Llama-3-70b-chat | Open source models |
| **DeepSeek** | ✅ Yes | deepseek-chat, deepseek-coder | Code-focused models |
| **Moonshot** | ✅ Yes | moonshot-v1-8k, moonshot-v1-32k | Chinese provider |
| **Zhipu AI** | ✅ Yes | glm-4, glm-3-turbo | ChatGLM models |
| **Cohere** | ✅ Yes | command, command-light | Enterprise NLP |
| **Hugging Face** | ✅ Yes | Any HF model | Open source hub |
| **Baidu** | ✅ Yes | ernie-bot, ernie-bot-turbo | Chinese Wenxin |
| **Alibaba** | ✅ Yes | qwen-turbo, qwen-plus | Chinese Dashscope |

## 🔧 Configuration Methods

### Method 1: Environment Variables (Recommended)

Set environment variables for your chosen provider:

```bash
# For OpenAI
export OPENAI_API_KEY="your_api_key_here"

# For Anthropic
export ANTHROPIC_API_KEY="your_api_key_here"

# For Google
export GOOGLE_API_KEY="your_api_key_here"

# For Groq
export GROQ_API_KEY="your_api_key_here"

# etc.
```

### Method 2: Configuration File

Edit `config.yaml`:

```yaml
llm:
  provider: "openai"  # Change to your provider
  model: "gpt-4"      # Change to your model
  api_key: "your_api_key_here"  # Optional if using env vars

api_keys:
  openai: "your_openai_key"
  anthropic: "your_anthropic_key"
  google: "your_google_key"
  # etc.
```

### Method 3: .env File

Copy `.env.example` to `.env` and uncomment/set your keys:

```bash
cp .env.example .env
# Edit .env file with your API keys
```

## 📋 Provider-Specific Setup

### 🦙 Ollama (Local, Free)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Download models
ollama pull llama3
ollama pull codellama
ollama pull mistral
```

**Configuration:**
```yaml
llm:
  provider: "ollama"
  model: "llama3"
  base_url: "http://localhost:11434"
```

### 🤖 OpenAI

1. Get API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Set environment variable: `export OPENAI_API_KEY="sk-..."`

**Configuration:**
```yaml
llm:
  provider: "openai"
  model: "gpt-4"  # or gpt-3.5-turbo
```

**Popular Models:**
- `gpt-4` - Most capable
- `gpt-3.5-turbo` - Fast and cost-effective
- `gpt-4-turbo` - Latest with larger context

### 🧠 Anthropic Claude

1. Get API key from [Anthropic Console](https://console.anthropic.com/)
2. Set environment variable: `export ANTHROPIC_API_KEY="sk-ant-..."`

**Configuration:**
```yaml
llm:
  provider: "anthropic"
  model: "claude-3-opus-20240229"
```

**Popular Models:**
- `claude-3-opus-20240229` - Most capable
- `claude-3-sonnet-20240229` - Balanced
- `claude-3-haiku-20240307` - Fast and efficient

### 🔍 Google Gemini

1. Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Set environment variable: `export GOOGLE_API_KEY="AI..."`

**Configuration:**
```yaml
llm:
  provider: "google"
  model: "gemini-pro"
```

**Popular Models:**
- `gemini-pro` - Text generation
- `gemini-1.5-pro` - Latest with large context
- `gemini-pro-vision` - Multimodal

### ☁️ Azure OpenAI

1. Create Azure OpenAI resource
2. Get API key and endpoint from Azure portal
3. Set environment variable: `export AZURE_API_KEY="your_key"`

**Configuration:**
```yaml
llm:
  provider: "azure"
  model: "gpt-4"  # Your deployment name
  base_url: "https://your-resource.openai.azure.com/openai/deployments"
```

### ⚡ Groq (Ultra-fast)

1. Get API key from [Groq Console](https://console.groq.com/keys)
2. Set environment variable: `export GROQ_API_KEY="gsk_..."`

**Configuration:**
```yaml
llm:
  provider: "groq"
  model: "llama3-70b-8192"
```

**Popular Models:**
- `llama3-70b-8192` - Most capable
- `llama3-8b-8192` - Fast
- `mixtral-8x7b-32768` - Good balance

### 🤝 Together AI

1. Get API key from [Together AI](https://api.together.xyz/settings/api-keys)
2. Set environment variable: `export TOGETHER_API_KEY="your_key"`

**Configuration:**
```yaml
llm:
  provider: "together"
  model: "meta-llama/Llama-3-70b-chat-hf"
```

### 🔬 DeepSeek

1. Get API key from [DeepSeek Platform](https://platform.deepseek.com/api_keys)
2. Set environment variable: `export DEEPSEEK_API_KEY="sk-..."`

**Configuration:**
```yaml
llm:
  provider: "deepseek"
  model: "deepseek-chat"
```

### 🌙 Moonshot AI

1. Get API key from [Moonshot Platform](https://platform.moonshot.cn/)
2. Set environment variable: `export MOONSHOT_API_KEY="sk-..."`

**Configuration:**
```yaml
llm:
  provider: "moonshot"
  model: "moonshot-v1-8k"
```

### 🇨🇳 Chinese Providers

#### Zhipu AI (ChatGLM)
```yaml
llm:
  provider: "zhipu"
  model: "glm-4"
```

#### Baidu Wenxin
```yaml
llm:
  provider: "baidu"
  model: "ernie-bot"
```

#### Alibaba Dashscope
```yaml
llm:
  provider: "alibaba"
  model: "qwen-turbo"
```

## 🚀 Quick Start Examples

### Using OpenAI GPT-4
```bash
export OPENAI_API_KEY="your_key"
python main.py generate --provider openai --model gpt-4
```

### Using Anthropic Claude
```bash
export ANTHROPIC_API_KEY="your_key"
python main.py generate --provider anthropic --model claude-3-opus-20240229
```

### Using Groq for Speed
```bash
export GROQ_API_KEY="your_key"
python main.py generate --provider groq --model llama3-70b-8192
```

## 💡 Tips and Best Practices

### Model Selection
- **For quality**: OpenAI GPT-4, Anthropic Claude-3-Opus
- **For speed**: Groq, Together AI
- **For cost**: OpenAI GPT-3.5-turbo, local Ollama
- **For coding**: DeepSeek Coder, OpenAI GPT-4
- **For Chinese**: Zhipu GLM-4, Baidu Ernie, Alibaba Qwen

### Performance Optimization
- Use faster models for character dialogue
- Use more capable models for plot planning
- Consider model switching based on task type

### Cost Management
- Monitor API usage and costs
- Use cheaper models for less critical tasks
- Consider local Ollama for development

### Error Handling
- Always set fallback providers
- Monitor API rate limits
- Handle authentication errors gracefully

## 🔧 Troubleshooting

### Common Issues

**API Key Not Found**
```bash
# Check environment variables
env | grep API_KEY

# Verify config file
cat config.yaml | grep -A 10 api_keys
```

**Connection Errors**
- Verify API key is correct
- Check internet connection
- Confirm provider service status

**Model Not Found**
- Check model name spelling
- Verify model availability for your account
- Try alternative model names

**Rate Limiting**
- Reduce request frequency
- Upgrade API plan
- Use multiple providers for load balancing

### Testing Connection

```bash
# Test your configuration
python -c "
from src.utils.llm_client import LLMClient
import asyncio

async def test():
    client = LLMClient(provider='your_provider', model='your_model', api_key='your_key')
    async with client:
        result = await client.generate('Hello, world!')
        print(f'Response: {result}')

asyncio.run(test())
"
```

## 📞 Support

For provider-specific issues:
- Check provider documentation
- Contact provider support
- Join provider community forums

For Novel AI Agent issues:
- Create GitHub issue
- Check troubleshooting guide
- Review configuration examples