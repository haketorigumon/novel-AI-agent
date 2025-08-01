"""LLM Client for interacting with various language models"""

import asyncio
import json
from typing import Dict, List, Optional, Any
import aiohttp
from rich.console import Console

console = Console()

class LLMClient:
    """Client for interacting with LLM providers (Ollama, OpenAI, etc.)"""
    
    def __init__(self, provider: str, model: str, base_url: str, **kwargs):
        self.provider = provider.lower()
        self.model = model
        self.base_url = base_url
        self.temperature = kwargs.get('temperature', 0.8)
        self.max_tokens = kwargs.get('max_tokens', 4096)
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using the configured LLM"""
        if self.provider == "ollama":
            return await self._generate_ollama(prompt, system_prompt)
        elif self.provider == "openai":
            return await self._generate_openai(prompt, system_prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
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
        
        try:
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
        except Exception as e:
            console.print(f"[red]Error calling Ollama: {e}[/red]")
            return ""
    
    async def _generate_openai(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using OpenAI API"""
        # This would require OpenAI API key and implementation
        # For now, return a placeholder
        return "OpenAI integration not implemented yet"
    
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
        if self.provider == "ollama":
            try:
                async with self.session.get(f"{self.base_url}/api/tags") as response:
                    return response.status == 200
            except:
                return False
        return False