"""Configuration management for Novel AI Agent"""

import yaml
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class LLMConfig(BaseModel):
    provider: str = "ollama"
    model: str = "llama3"
    base_url: Optional[str] = "http://localhost:11434"
    api_key: Optional[str] = None
    temperature: float = 0.8
    max_tokens: int = 4096

class APIKeysConfig(BaseModel):
    openai: Optional[str] = None
    anthropic: Optional[str] = None
    google: Optional[str] = None
    azure: Optional[str] = None
    cohere: Optional[str] = None
    huggingface: Optional[str] = None
    together: Optional[str] = None
    groq: Optional[str] = None
    deepseek: Optional[str] = None
    moonshot: Optional[str] = None
    zhipu: Optional[str] = None
    baidu: Optional[str] = None
    alibaba: Optional[str] = None

class StoryConfig(BaseModel):
    target_length: int = 5000000
    chapter_length: int = 5000
    save_interval: int = 1000
    output_dir: str = "output"

class AgentsConfig(BaseModel):
    max_agents: int = 10
    director_enabled: bool = True
    character_types: List[str] = ["protagonist", "antagonist", "supporting", "narrator"]

class EvolutionConfig(BaseModel):
    enabled: bool = True
    mutation_rate: float = 0.1
    evaluation_interval: int = 10000
    backup_generations: int = 5

class SimulationConfig(BaseModel):
    world_complexity: str = "medium"
    event_frequency: float = 0.3
    environment_changes: bool = True

class WebInterfaceConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 12000
    enable_cors: bool = True

class Config(BaseModel):
    llm: LLMConfig = LLMConfig()
    api_keys: APIKeysConfig = APIKeysConfig()
    story: StoryConfig = StoryConfig()
    agents: AgentsConfig = AgentsConfig()
    evolution: EvolutionConfig = EvolutionConfig()
    simulation: SimulationConfig = SimulationConfig()
    web_interface: WebInterfaceConfig = WebInterfaceConfig()
    
    def get_api_key_for_provider(self, provider: str) -> Optional[str]:
        """Get API key for a specific provider"""
        # First check if API key is specified in LLM config
        if self.llm.api_key:
            return self.llm.api_key
        
        # Then check provider-specific API keys
        provider_key = getattr(self.api_keys, provider.lower(), None)
        if provider_key:
            return provider_key
        
        return None

    @classmethod
    def load(cls, config_path: str) -> 'Config':
        """Load configuration from YAML file"""
        path = Path(config_path)
        if not path.exists():
            # Return default config if file doesn't exist
            return cls()
        
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return cls(**data)
    
    def save(self, config_path: str):
        """Save configuration to YAML file"""
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, allow_unicode=True)