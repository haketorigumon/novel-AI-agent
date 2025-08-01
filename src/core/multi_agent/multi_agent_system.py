"""Multi-agent system core implementation"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple, Union

from ...utils.config import Config
from ...utils.llm_client import LLMClient
from ...agents.enhanced_base_agent import EnhancedBaseAgent
from ...communication.message import Message, Conversation
from ...tools.tool_registry import ToolRegistry
from ...memory.memory_manager import MemoryManager
from ...orchestration.orchestrator import Orchestrator

class MultiAgentSystem:
    """
    Core implementation of the multi-agent system
    
    Attributes:
        config: System configuration
        llm_client: LLM client for system operations
        orchestrator: System orchestrator
        agents: Dictionary of registered agents
        tool_registry: Registry of available tools
        system_memory: Memory manager for the system
    """
    
    def __init__(self, config: Config, llm_client: Optional[LLMClient] = None):
        self.config = config
        self.llm_client = llm_client
        self.orchestrator = None
        self.agents: Dict[str, EnhancedBaseAgent] = {}
        self.tool_registry = None
        self.system_memory = None
        self.system_id = f"system_{uuid.uuid4().hex[:8]}"
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.active = False
    
    async def initialize(self):
        """Initialize the multi-agent system"""
        # Initialize tool registry
        self.tool_registry = ToolRegistry(self.config)
        await self.tool_registry.initialize()
        
        # Initialize system memory
        self.system_memory = MemoryManager(self.system_id, self.config)
        await self.system_memory.initialize()
        
        # Initialize orchestrator
        self.orchestrator = Orchestrator(self.config, self.llm_client)
        await self.orchestrator.initialize()
        
        # Register system tools
        await self._register_system_tools()
        
        self.active = True
        
        # Store initialization in memory
        await self.system_memory.store_memory(
            content="Multi-agent system initialized",
            memory_type="system",
            metadata={
                "system_id": self.system_id,
                "timestamp": datetime.now().isoformat()
            },
            importance=0.9
        )
    
    async def _register_system_tools(self):
        """Register system-specific tools"""
        self.tool_registry.register_tool(
            name="create_agent",
            description="Create a new agent with specified role and capabilities",
            function=self.create_agent
        )
        
        self.tool_registry.register_tool(
            name="process_user_request",
            description="Process a request from a user",
            function=self.process_user_request
        )
        
        self.tool_registry.register_tool(
            name="get_system_status",
            description="Get the current status of the multi-agent system",
            function=self.get_system_status
        )
    
    async def create_agent(
        self,
        agent_type: str,
        name: str,
        role: str,
        description: str = "",
        capabilities: Optional[List[str]] = None,
        personality_traits: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Create a new agent
        
        Args:
            agent_type: Type of agent to create
            name: Name of the agent
            role: Role of the agent
            description: Description of the agent
            capabilities: List of agent capabilities
            personality_traits: Dictionary of personality traits
            
        Returns:
            ID of the created agent
        """
        # Use orchestrator to create agent
        agent_id = await self.orchestrator.create_agent(
            agent_type=agent_type,
            name=name,
            role=role,
            description=description,
            capabilities=capabilities,
            personality_traits=personality_traits
        )
        
        # Store in system memory
        await self.system_memory.store_memory(
            content=f"Created agent: {name} ({agent_id}) with role: {role}",
            memory_type="system",
            metadata={
                "agent_id": agent_id,
                "agent_name": name,
                "agent_role": role,
                "agent_type": agent_type
            },
            importance=0.8
        )
        
        return agent_id
    
    async def process_user_request(self, user_id: str, request: str) -> Dict[str, Any]:
        """
        Process a request from a user
        
        Args:
            user_id: ID of the user making the request
            request: The user's request
            
        Returns:
            Response to the user
        """
        # Store request in memory
        await self.system_memory.store_memory(
            content=f"Received user request: {request}",
            memory_type="user_request",
            metadata={
                "user_id": user_id,
                "request": request
            },
            importance=0.7
        )
        
        # Use orchestrator to process request
        response = await self.orchestrator.process_user_request(user_id, request)
        
        # Store response in memory
        await self.system_memory.store_memory(
            content=f"Processed user request: {request}",
            memory_type="system",
            metadata={
                "user_id": user_id,
                "request": request,
                "response": response
            },
            importance=0.7
        )
        
        return response
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get the current status of the multi-agent system
        
        Returns:
            System status
        """
        agent_count = len(self.orchestrator.agents) if self.orchestrator else 0
        task_count = len(self.orchestrator.active_tasks) if self.orchestrator else 0
        conversation_count = len(self.orchestrator.conversations) if self.orchestrator else 0
        
        return {
            "system_id": self.system_id,
            "active": self.active,
            "agent_count": agent_count,
            "task_count": task_count,
            "conversation_count": conversation_count,
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about an agent
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Agent information
        """
        if self.orchestrator:
            return await self.orchestrator.get_agent_status(agent_id)
        
        return None
    
    async def get_all_agents(self) -> List[Dict[str, Any]]:
        """
        Get information about all agents
        
        Returns:
            List of agent information
        """
        if self.orchestrator:
            return await self.orchestrator.get_all_agents()
        
        return []
    
    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a task
        
        Args:
            task_id: ID of the task
            
        Returns:
            Task information
        """
        if self.orchestrator:
            return await self.orchestrator.get_task(task_id)
        
        return None
    
    async def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a conversation
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Conversation information
        """
        if self.orchestrator:
            return await self.orchestrator.get_conversation(conversation_id)
        
        return None
    
    async def shutdown(self):
        """Shutdown the multi-agent system"""
        self.active = False
        
        # Store shutdown in memory
        await self.system_memory.store_memory(
            content="Multi-agent system shutdown",
            memory_type="system",
            metadata={
                "system_id": self.system_id,
                "timestamp": datetime.now().isoformat()
            },
            importance=0.9
        )
        
        # Save all memories
        if self.system_memory:
            await self.system_memory._save_memories()
        
        # Shutdown orchestrator
        if self.orchestrator:
            for agent_id in list(self.orchestrator.agents.keys()):
                await self.orchestrator.unregister_agent(agent_id)