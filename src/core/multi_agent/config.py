"""Configuration for multi-agent system"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class MultiAgentConfig(BaseModel):
    """Configuration for multi-agent system"""
    
    enabled: bool = True
    max_agents: int = 20
    default_agent_types: List[str] = ["assistant", "expert", "creative", "task"]
    orchestration_strategy: str = "hierarchical"  # hierarchical, peer-to-peer, hybrid
    memory_enabled: bool = True
    tool_usage_enabled: bool = True
    communication_logging: bool = True
    default_llm_temperature: float = 0.7
    default_max_tokens: int = 2048
    
    # Agent creation settings
    auto_create_agents: bool = True
    agent_creation_threshold: float = 0.7
    
    # Task management settings
    task_decomposition_enabled: bool = True
    max_concurrent_tasks: int = 10
    task_timeout_seconds: int = 300
    
    # Communication settings
    broadcast_enabled: bool = True
    direct_communication_enabled: bool = True
    message_queue_size: int = 100
    
    # Memory settings
    memory_retention_days: int = 30
    memory_importance_threshold: float = 0.3
    
    # Tool settings
    default_tools_enabled: bool = True
    tool_usage_logging: bool = True
    
    # Web interface settings
    web_visualization_enabled: bool = True
    agent_interaction_enabled: bool = True