"""Enhanced LLM Client supporting multiple API providers and embeddings"""

import asyncio
import json
import logging
import os
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import aiohttp
from rich.console import Console

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

console = Console()
logger = logging.getLogger(__name__)

class LLMClient:
    """Enhanced client for interacting with various LLM providers"""
    
    SUPPORTED_PROVIDERS = {
        "ollama": {
            "requires_api_key": False, 
            "default_base_url": "http://localhost:11434",
            "api_format": "ollama"
        },
        "openai": {
            "requires_api_key": True, 
            "default_base_url": "https://api.openai.com/v1",
            "api_format": "openai"
        },
        "anthropic": {
            "requires_api_key": True, 
            "default_base_url": "https://api.anthropic.com",
            "api_format": "anthropic"
        },
        "google": {
            "requires_api_key": True, 
            "default_base_url": "https://generativelanguage.googleapis.com/v1beta",
            "api_format": "google"
        },
        "azure": {
            "requires_api_key": True, 
            "default_base_url": None,  # Custom endpoint required
            "api_format": "openai"
        },
        "cohere": {
            "requires_api_key": True, 
            "default_base_url": "https://api.cohere.ai/v1",
            "api_format": "cohere"
        },
        "huggingface": {
            "requires_api_key": True, 
            "default_base_url": "https://api-inference.huggingface.co/models",
            "api_format": "huggingface"
        },
        "together": {
            "requires_api_key": True, 
            "default_base_url": "https://api.together.xyz/v1",
            "api_format": "openai"
        },
        "groq": {
            "requires_api_key": True, 
            "default_base_url": "https://api.groq.com/openai/v1",
            "api_format": "openai"
        },
        "deepseek": {
            "requires_api_key": True, 
            "default_base_url": "https://api.deepseek.com/v1",
            "api_format": "openai"
        },
        "moonshot": {
            "requires_api_key": True, 
            "default_base_url": "https://api.moonshot.cn/v1",
            "api_format": "openai"
        },
        "zhipu": {
            "requires_api_key": True, 
            "default_base_url": "https://open.bigmodel.cn/api/paas/v4",
            "api_format": "zhipu"
        },
        "baidu": {
            "requires_api_key": True, 
            "default_base_url": "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat",
            "api_format": "baidu"
        },
        "alibaba": {
            "requires_api_key": True, 
            "default_base_url": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
            "api_format": "alibaba"
        }
    }
    
    def __init__(self, provider: str = "ollama", model: str = "llama3", 
                 base_url: Optional[str] = None, api_key: Optional[str] = None,
                 temperature: float = 0.8, max_tokens: int = 4096, **kwargs):
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_params = kwargs
        self.session = None
        
        # Validate provider
        if self.provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}. Supported: {list(self.SUPPORTED_PROVIDERS.keys())}")
        
        provider_info = self.SUPPORTED_PROVIDERS[self.provider]
        
        # Set base URL
        if base_url:
            self.base_url = base_url
        else:
            self.base_url = provider_info["default_base_url"]
            if not self.base_url:
                raise ValueError(f"Provider {provider} requires a custom base_url")
        
        # Handle API key
        self.api_key = api_key or self._get_api_key_from_env()
        
        # Validate API key requirement
        if provider_info["requires_api_key"] and not self.api_key:
            logger.warning(f"Provider {provider} requires an API key. Set {provider.upper()}_API_KEY environment variable or pass api_key parameter")
        
        self.api_format = provider_info["api_format"]
    
    def _get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment variables"""
        env_keys = [
            f"{self.provider.upper()}_API_KEY",
            f"{self.provider.upper()}_KEY",
            "OPENAI_API_KEY" if self.provider in ["openai", "azure"] else None,
            "API_KEY"
        ]
        
        for key in env_keys:
            if key and os.getenv(key):
                return os.getenv(key)
        return None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using the configured LLM"""
        try:
            if self.api_format == "ollama":
                return await self._generate_ollama(prompt, system_prompt)
            elif self.api_format == "openai":
                return await self._generate_openai_compatible(prompt, system_prompt)
            elif self.api_format == "anthropic":
                return await self._generate_anthropic(prompt, system_prompt)
            elif self.api_format == "google":
                return await self._generate_google(prompt, system_prompt)
            elif self.api_format == "cohere":
                return await self._generate_cohere(prompt, system_prompt)
            elif self.api_format == "huggingface":
                return await self._generate_huggingface(prompt, system_prompt)
            elif self.api_format == "zhipu":
                return await self._generate_zhipu(prompt, system_prompt)
            elif self.api_format == "baidu":
                return await self._generate_baidu(prompt, system_prompt)
            elif self.api_format == "alibaba":
                return await self._generate_alibaba(prompt, system_prompt)
            else:
                raise ValueError(f"Unsupported API format: {self.api_format}")
        except Exception as e:
            console.print(f"[red]Error generating text with {self.provider}: {e}[/red]")
            return ""
    
    async def _generate_ollama(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using Ollama"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        async with self.session.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result.get("message", {}).get("content", "")
            else:
                error_text = await response.text()
                console.print(f"[red]Ollama API error: {response.status} - {error_text}[/red]")
                return ""
    
    async def _generate_openai_compatible(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using OpenAI-compatible API (OpenAI, Azure, Groq, Together, etc.)"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Azure OpenAI has different header format
        if self.provider == "azure":
            headers["api-key"] = self.api_key
            del headers["Authorization"]
        
        async with self.session.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result["choices"][0]["message"]["content"]
            else:
                error_text = await response.text()
                console.print(f"[red]{self.provider} API error: {response.status} - {error_text}[/red]")
                return ""
    
    async def _generate_anthropic(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using Anthropic Claude API"""
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        async with self.session.post(
            f"{self.base_url}/messages",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result["content"][0]["text"]
            else:
                error_text = await response.text()
                console.print(f"[red]Anthropic API error: {response.status} - {error_text}[/red]")
                return ""
    
    async def _generate_google(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using Google Gemini API"""
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        payload = {
            "contents": [{
                "parts": [{"text": full_prompt}]
            }],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens
            }
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
        
        async with self.session.post(
            url,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                error_text = await response.text()
                console.print(f"[red]Google API error: {response.status} - {error_text}[/red]")
                return ""
    
    async def _generate_cohere(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using Cohere API"""
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        async with self.session.post(
            f"{self.base_url}/generate",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result["generations"][0]["text"]
            else:
                error_text = await response.text()
                console.print(f"[red]Cohere API error: {response.status} - {error_text}[/red]")
                return ""
    
    async def _generate_huggingface(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using Hugging Face Inference API"""
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        payload = {
            "inputs": full_prompt,
            "parameters": {
                "temperature": self.temperature,
                "max_new_tokens": self.max_tokens
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        async with self.session.post(
            f"{self.base_url}/{self.model}",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as response:
            if response.status == 200:
                result = await response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "").replace(full_prompt, "").strip()
                return ""
            else:
                error_text = await response.text()
                console.print(f"[red]Hugging Face API error: {response.status} - {error_text}[/red]")
                return ""
    
    async def _generate_zhipu(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using Zhipu AI API"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        async with self.session.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result["choices"][0]["message"]["content"]
            else:
                error_text = await response.text()
                console.print(f"[red]Zhipu API error: {response.status} - {error_text}[/red]")
                return ""
    
    async def _generate_baidu(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using Baidu Wenxin API"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "messages": messages,
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        url = f"{self.base_url}/{self.model}?access_token={self.api_key}"
        
        async with self.session.post(
            url,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result.get("result", "")
            else:
                error_text = await response.text()
                console.print(f"[red]Baidu API error: {response.status} - {error_text}[/red]")
                return ""
    
    async def _generate_alibaba(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using Alibaba Dashscope API"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "input": {
                "messages": messages
            },
            "parameters": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        async with self.session.post(
            self.base_url,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result["output"]["text"]
            else:
                error_text = await response.text()
                console.print(f"[red]Alibaba API error: {response.status} - {error_text}[/red]")
                return ""
    
    async def generate_structured(self, prompt: str, schema: Dict[str, Any], system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate structured output based on a schema"""
        structured_prompt = f"""
{prompt}

Please respond with a valid JSON object that matches this schema:
{json.dumps(schema, indent=2)}

Response:
"""
        
        response = await self.generate(structured_prompt, system_prompt)
        
        try:
            # Try to extract JSON from the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # If JSON extraction fails, return empty dict
        console.print(f"[yellow]Warning: Could not parse structured response: {response[:100]}...[/yellow]")
        return {}
    
    async def check_connection(self) -> bool:
        """Check if the LLM service is available"""
        try:
            if self.provider == "ollama":
                async with self.session.get(f"{self.base_url}/api/tags") as response:
                    return response.status == 200
            elif self.api_format == "openai":
                headers = {"Authorization": f"Bearer {self.api_key}"}
                if self.provider == "azure":
                    headers = {"api-key": self.api_key}
                async with self.session.get(f"{self.base_url}/models", headers=headers) as response:
                    return response.status == 200
            else:
                # For other providers, try a simple generation test
                test_response = await self.generate("Hello", "You are a helpful assistant.")
                return len(test_response) > 0
        except Exception as e:
            logger.debug(f"Connection check failed for {self.provider}: {e}")
            return False
    
    @classmethod
    def get_supported_providers(cls) -> List[str]:
        """Get list of supported providers"""
        return list(cls.SUPPORTED_PROVIDERS.keys())
    
    @classmethod
    def get_provider_info(cls, provider: str) -> Dict[str, Any]:
        """Get information about a specific provider"""
        return cls.SUPPORTED_PROVIDERS.get(provider.lower(), {})
        
    async def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding for text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if embedding failed
        """
        try:
            if self.provider == "openai":
                return await self._generate_embedding_openai(text)
            elif self.provider == "cohere":
                return await self._generate_embedding_cohere(text)
            elif SENTENCE_TRANSFORMERS_AVAILABLE:
                # Fallback to local embedding model
                return await self._generate_embedding_local(text)
            else:
                console.print("[yellow]Warning: No embedding provider available. Using random embedding.[/yellow]")
                return np.random.rand(384).astype(np.float32)  # Default size
        except Exception as e:
            console.print(f"[red]Error generating embedding: {e}[/red]")
            return None
            
    async def _generate_embedding_openai(self, text: str) -> np.ndarray:
        """Generate embedding using OpenAI API"""
        payload = {
            "input": text,
            "model": "text-embedding-3-small"  # Default embedding model
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        async with self.session.post(
            f"{self.base_url}/embeddings",
            json=payload,
            headers=headers
        ) as response:
            if response.status == 200:
                result = await response.json()
                embedding = result["data"][0]["embedding"]
                return np.array(embedding, dtype=np.float32)
            else:
                error_text = await response.text()
                raise ValueError(f"OpenAI embedding error: {response.status} - {error_text}")
                
    async def _generate_embedding_cohere(self, text: str) -> np.ndarray:
        """Generate embedding using Cohere API"""
        payload = {
            "texts": [text],
            "model": "embed-english-v3.0"  # Default embedding model
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        async with self.session.post(
            f"{self.base_url}/embed",
            json=payload,
            headers=headers
        ) as response:
            if response.status == 200:
                result = await response.json()
                embedding = result["embeddings"][0]
                return np.array(embedding, dtype=np.float32)
            else:
                error_text = await response.text()
                raise ValueError(f"Cohere embedding error: {response.status} - {error_text}")
                
    async def _generate_embedding_local(self, text: str) -> np.ndarray:
        """Generate embedding using local SentenceTransformers model"""
        # This runs in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        
        # Lazy-load the model
        if not hasattr(self, "_embedding_model"):
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            
        # Generate embedding
        embedding = await loop.run_in_executor(
            None, lambda: self._embedding_model.encode(text, convert_to_numpy=True)
        )
        
        return embedding.astype(np.float32)
        
    async def batch_generate_embeddings(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            embedding = await self.generate_embedding(text)
            embeddings.append(embedding)
        return embeddings