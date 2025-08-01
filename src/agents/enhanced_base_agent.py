"""Enhanced base agent class for the multi-agent system"""

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Callable

from ..utils.config import Config
from ..utils.llm_client import LLMClient
from ..memory.memory_manager import MemoryManager
from ..communication.message import Message
from ..tools.tool_registry import ToolRegistry

class EnhancedBaseAgent(ABC):
    """
    Enhanced base class for all agents in the multi-agent system.
    Provides core capabilities for communication, memory, tool usage, and reasoning.
    """
    
    def __init__(
        self, 
        agent_id: Optional[str] = None,
        name: str = "",
        role: str = "",
        description: str = "",
        config: Optional[Config] = None,
        llm_client: Optional[LLMClient] = None,
        memory_manager: Optional[MemoryManager] = None,
        tool_registry: Optional[ToolRegistry] = None
    ):
        # Core identity
        self.agent_id = agent_id or str(uuid.uuid4())
        self.name = name or f"Agent-{self.agent_id[:8]}"
        self.role = role
        self.description = description
        
        # Configuration and services
        self.config = config
        self.llm_client = llm_client
        self.memory_manager = memory_manager
        self.tool_registry = tool_registry
        
        # Agent state
        self.created_at = datetime.now()
        self.last_action = None
        self.is_active = True
        self.personality = {}
        self.goals = []
        
        # Communication
        self.message_queue = asyncio.Queue()
        self.message_handlers = {}
        self.connected_agents = set()
        
        # Capabilities
        self.capabilities = set()
        self.available_tools = set()
    
    async def initialize(self):
        """Initialize the agent with personality, goals, and capabilities"""
        if not self.memory_manager and self.config:
            from ..memory.memory_manager import MemoryManager
            self.memory_manager = MemoryManager(self.agent_id, self.config)
            await self.memory_manager.initialize()
            
        if not self.tool_registry and self.config:
            from ..tools.tool_registry import ToolRegistry
            self.tool_registry = ToolRegistry(self.config)
            await self.tool_registry.initialize()
            
        await self._generate_personality()
        await self._generate_initial_goals()
        await self._setup_capabilities()
    
    @abstractmethod
    async def _generate_personality(self):
        """Generate personality traits for this agent"""
        pass
    
    @abstractmethod
    async def _generate_initial_goals(self):
        """Generate initial goals for this agent"""
        pass
    
    async def _setup_capabilities(self):
        """Set up agent capabilities and available tools"""
        # Default implementation - override in subclasses
        self.capabilities = {"basic_reasoning", "communication"}
        
        if self.tool_registry:
            self.available_tools = await self.tool_registry.get_available_tools()
    
    async def process_message(self, message: Message) -> Optional[Message]:
        """
        Process an incoming message and generate a response
        
        Args:
            message: The incoming message to process
            
        Returns:
            Optional response message
        """
        # Check if we have a specific handler for this message type
        if message.type in self.message_handlers:
            return await self.message_handlers[message.type](message)
        
        # Default message processing
        if self.llm_client:
            response_content = await self._reason_about_message(message)
            
            if response_content:
                return Message(
                    sender_id=self.agent_id,
                    receiver_id=message.sender_id,
                    content=response_content,
                    type="response",
                    reference_message_id=message.message_id
                )
        
        return None
    
    async def _reason_about_message(self, message: Message) -> str:
        """
        Use the LLM to reason about how to respond to a message
        
        Args:
            message: The message to reason about
            
        Returns:
            Reasoning result as a string
        """
        if not self.llm_client:
            return ""
            
        async with self.llm_client as client:
            # Retrieve relevant memories
            memories = []
            if self.memory_manager:
                memories = await self.memory_manager.retrieve_relevant_memories(message.content)
            
            # Construct reasoning prompt
            reasoning_prompt = f"""
You are {self.name}, a {self.role}.
{self.description}

Your personality: {json.dumps(self.personality, indent=2)}
Your current goals: {json.dumps(self.goals, indent=2)}

You have received the following message:
From: {message.sender_id}
Type: {message.type}
Content: {message.content}

Relevant memories:
{chr(10).join(f"- {memory}" for memory in memories)}

Available tools: {', '.join(self.available_tools) if self.available_tools else 'None'}

How would you respond to this message? Consider your role, personality, goals, and available tools.
"""
            
            return await client.generate(
                reasoning_prompt,
                f"You are {self.name}, a {self.role}. Respond in a way that's consistent with your personality and goals."
            )
    
    async def send_message(self, receiver_id: str, content: str, message_type: str = "text") -> Message:
        """
        Send a message to another agent
        
        Args:
            receiver_id: ID of the receiving agent
            content: Message content
            message_type: Type of message
            
        Returns:
            The sent message
        """
        message = Message(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            content=content,
            type=message_type
        )
        
        # Store in memory if available
        if self.memory_manager:
            await self.memory_manager.store_message(message)
        
        return message
    
    async def receive_message(self, message: Message):
        """
        Receive a message and add it to the message queue
        
        Args:
            message: The message to receive
        """
        # Store in memory if available
        if self.memory_manager:
            await self.memory_manager.store_message(message)
            
        # Add to processing queue
        await self.message_queue.put(message)
    
    async def connect_to_agent(self, agent_id: str):
        """
        Connect to another agent to enable direct communication
        
        Args:
            agent_id: ID of the agent to connect to
        """
        self.connected_agents.add(agent_id)
    
    async def disconnect_from_agent(self, agent_id: str):
        """
        Disconnect from another agent
        
        Args:
            agent_id: ID of the agent to disconnect from
        """
        if agent_id in self.connected_agents:
            self.connected_agents.remove(agent_id)
    
    async def register_message_handler(self, message_type: str, handler: Callable[[Message], Optional[Message]]):
        """
        Register a handler for a specific message type
        
        Args:
            message_type: Type of message to handle
            handler: Function to handle the message
        """
        self.message_handlers[message_type] = handler
    
    async def use_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Use a tool from the tool registry
        
        Args:
            tool_name: Name of the tool to use
            **kwargs: Arguments to pass to the tool
            
        Returns:
            Result of the tool execution
        """
        if not self.tool_registry:
            return None
            
        if tool_name not in self.available_tools:
            return f"Error: Tool '{tool_name}' is not available to this agent"
            
        return await self.tool_registry.execute_tool(tool_name, **kwargs)
    
    async def reflect(self, experience: Dict[str, Any]):
        """
        Reflect on an experience to update goals and personality
        
        Args:
            experience: The experience to reflect on
        """
        if not self.llm_client:
            return
            
        async with self.llm_client as client:
            reflection_prompt = f"""
You are {self.name}, a {self.role}. You just experienced the following:
{json.dumps(experience, indent=2)}

Your current personality: {json.dumps(self.personality, indent=2)}
Your current goals: {json.dumps(self.goals, indent=2)}

Reflect on this experience. How might it change your perspective, goals, or approach to future situations?
Respond with a JSON object containing:
{{
    "reflection": "Your thoughts on the experience",
    "personality_changes": {{"trait": change_amount}},  // -0.1 to +0.1
    "new_goals": ["any new goals to add"],
    "completed_goals": ["any goals that are now complete"]
}}
"""
            
            reflection = await client.generate_structured(
                reflection_prompt,
                {
                    "reflection": "string",
                    "personality_changes": "object",
                    "new_goals": "array",
                    "completed_goals": "array"
                },
                f"You are {self.name}, reflecting on your experiences and growth."
            )
            
            if reflection:
                # Apply personality changes
                for trait, change in reflection.get("personality_changes", {}).items():
                    if trait in self.personality:
                        self.personality[trait] = max(0, min(1, 
                            self.personality[trait] + change))
                
                # Add new goals
                for goal in reflection.get("new_goals", []):
                    if goal not in self.goals:
                        self.goals.append(goal)
                
                # Remove completed goals
                for goal in reflection.get("completed_goals", []):
                    if goal in self.goals:
                        self.goals.remove(goal)
                
                # Store reflection in memory
                if self.memory_manager:
                    await self.memory_manager.store_memory({
                        "type": "reflection",
                        "experience": experience,
                        "reflection": reflection.get("reflection", ""),
                        "changes_made": {
                            "personality": reflection.get("personality_changes", {}),
                            "new_goals": reflection.get("new_goals", []),
                            "completed_goals": reflection.get("completed_goals", [])
                        }
                    })
    
    async def run(self):
        """
        Main agent loop - process messages from the queue
        """
        while self.is_active:
            try:
                # Get the next message from the queue
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                
                # Process the message
                response = await self.process_message(message)
                
                # Mark the message as processed
                self.message_queue.task_done()
                
                # Return the response if any
                if response:
                    return response
                    
            except asyncio.TimeoutError:
                # No message received within timeout, continue
                pass
            except Exception as e:
                print(f"Error in agent {self.agent_id} run loop: {e}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state of the agent"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "role": self.role,
            "personality": self.personality,
            "goals": self.goals,
            "capabilities": list(self.capabilities),
            "available_tools": list(self.available_tools),
            "connected_agents": list(self.connected_agents),
            "created_at": self.created_at.isoformat(),
            "last_action": self.last_action,
            "is_active": self.is_active
        }