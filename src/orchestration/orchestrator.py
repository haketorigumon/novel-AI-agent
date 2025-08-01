"""Multi-agent orchestrator for coordinating agent interactions"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple, Union

from ..utils.config import Config
from ..utils.llm_client import LLMClient
from ..agents.enhanced_base_agent import EnhancedBaseAgent
from ..communication.message import Message, Conversation
from ..tools.tool_registry import ToolRegistry
from ..memory.memory_manager import MemoryManager

class Orchestrator:
    """
    Orchestrates interactions between multiple agents
    
    Attributes:
        config: System configuration
        llm_client: LLM client for orchestrator reasoning
        agents: Dictionary of registered agents
        conversations: Dictionary of active conversations
        tool_registry: Registry of available tools
        memory_manager: Memory manager for the orchestrator
    """
    
    def __init__(self, config: Config, llm_client: Optional[LLMClient] = None):
        self.config = config
        self.llm_client = llm_client
        self.agents: Dict[str, EnhancedBaseAgent] = {}
        self.conversations: Dict[str, Conversation] = {}
        self.tool_registry = None
        self.memory_manager = None
        self.orchestrator_id = f"orchestrator_{uuid.uuid4().hex[:8]}"
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self):
        """Initialize the orchestrator"""
        # Initialize tool registry
        self.tool_registry = ToolRegistry(self.config)
        await self.tool_registry.initialize()
        
        # Initialize memory manager
        self.memory_manager = MemoryManager(self.orchestrator_id, self.config)
        await self.memory_manager.initialize()
        
        # Register orchestrator-specific tools
        await self._register_orchestrator_tools()
    
    async def _register_orchestrator_tools(self):
        """Register orchestrator-specific tools"""
        self.tool_registry.register_tool(
            name="create_agent",
            description="Create a new agent with specified role and capabilities",
            function=self.create_agent
        )
        
        self.tool_registry.register_tool(
            name="create_conversation",
            description="Create a new conversation between agents",
            function=self.create_conversation
        )
        
        self.tool_registry.register_tool(
            name="send_message",
            description="Send a message from one agent to another",
            function=self.send_message
        )
    
    async def register_agent(self, agent: EnhancedBaseAgent):
        """
        Register an agent with the orchestrator
        
        Args:
            agent: The agent to register
        """
        self.agents[agent.agent_id] = agent
        
        # Store in memory
        await self.memory_manager.store_memory(
            content=f"Registered agent: {agent.name} ({agent.agent_id}) with role: {agent.role}",
            memory_type="system",
            metadata={
                "agent_id": agent.agent_id,
                "agent_name": agent.name,
                "agent_role": agent.role
            }
        )
    
    async def unregister_agent(self, agent_id: str):
        """
        Unregister an agent
        
        Args:
            agent_id: ID of the agent to unregister
        """
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            
            # Store in memory
            await self.memory_manager.store_memory(
                content=f"Unregistered agent: {agent.name} ({agent.agent_id})",
                memory_type="system",
                metadata={
                    "agent_id": agent.agent_id,
                    "agent_name": agent.name
                }
            )
            
            # Remove from agents
            del self.agents[agent_id]
    
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
        # Import dynamically to avoid circular imports
        from ..agents.types.task_agent import TaskAgent
        from ..agents.types.assistant_agent import AssistantAgent
        from ..agents.types.expert_agent import ExpertAgent
        from ..agents.types.creative_agent import CreativeAgent
        
        # Create agent based on type
        agent = None
        if agent_type == "task":
            agent = TaskAgent(
                name=name,
                role=role,
                description=description,
                config=self.config,
                llm_client=self.llm_client,
                tool_registry=self.tool_registry
            )
        elif agent_type == "assistant":
            agent = AssistantAgent(
                name=name,
                role=role,
                description=description,
                config=self.config,
                llm_client=self.llm_client,
                tool_registry=self.tool_registry
            )
        elif agent_type == "expert":
            agent = ExpertAgent(
                name=name,
                role=role,
                description=description,
                config=self.config,
                llm_client=self.llm_client,
                tool_registry=self.tool_registry
            )
        elif agent_type == "creative":
            agent = CreativeAgent(
                name=name,
                role=role,
                description=description,
                config=self.config,
                llm_client=self.llm_client,
                tool_registry=self.tool_registry
            )
        else:
            return f"Error: Unknown agent type '{agent_type}'"
        
        if agent:
            # Initialize agent
            await agent.initialize()
            
            # Set capabilities
            if capabilities:
                agent.capabilities = set(capabilities)
            
            # Set personality traits
            if personality_traits:
                agent.personality.update(personality_traits)
            
            # Register agent
            await self.register_agent(agent)
            
            return agent.agent_id
        
        return "Error: Failed to create agent"
    
    async def create_conversation(self, participants: List[str], metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new conversation between agents
        
        Args:
            participants: List of agent IDs to include in the conversation
            metadata: Additional conversation metadata
            
        Returns:
            ID of the created conversation
        """
        # Verify all participants exist
        for agent_id in participants:
            if agent_id not in self.agents:
                return f"Error: Agent '{agent_id}' not found"
        
        # Create conversation
        conversation = Conversation(
            participants=participants,
            metadata=metadata or {}
        )
        
        # Store conversation
        self.conversations[conversation.conversation_id] = conversation
        
        # Store in memory
        await self.memory_manager.store_memory(
            content=f"Created conversation {conversation.conversation_id} with participants: {', '.join(participants)}",
            memory_type="system",
            metadata={
                "conversation_id": conversation.conversation_id,
                "participants": participants
            }
        )
        
        return conversation.conversation_id
    
    async def send_message(
        self,
        sender_id: str,
        receiver_id: str,
        content: Any,
        message_type: str = "text",
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Send a message from one agent to another
        
        Args:
            sender_id: ID of the sending agent
            receiver_id: ID of the receiving agent
            content: Message content
            message_type: Type of message
            conversation_id: ID of the conversation to add the message to
            metadata: Additional message metadata
            
        Returns:
            ID of the sent message
        """
        # Verify sender and receiver exist
        if sender_id not in self.agents and sender_id != self.orchestrator_id:
            return f"Error: Sender '{sender_id}' not found"
        
        if receiver_id not in self.agents and receiver_id != self.orchestrator_id:
            return f"Error: Receiver '{receiver_id}' not found"
        
        # Create message
        message = Message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            content=content,
            type=message_type,
            metadata=metadata or {}
        )
        
        # Add to conversation if specified
        if conversation_id:
            if conversation_id not in self.conversations:
                return f"Error: Conversation '{conversation_id}' not found"
            
            conversation = self.conversations[conversation_id]
            
            # Verify sender and receiver are participants
            if sender_id not in conversation.participants and sender_id != self.orchestrator_id:
                return f"Error: Sender '{sender_id}' is not a participant in the conversation"
            
            if receiver_id not in conversation.participants and receiver_id != self.orchestrator_id:
                return f"Error: Receiver '{receiver_id}' is not a participant in the conversation"
            
            # Add message to conversation
            conversation.add_message(message)
        
        # Deliver message to receiver
        if receiver_id in self.agents:
            await self.agents[receiver_id].receive_message(message)
        
        # Store in memory
        await self.memory_manager.store_message(message)
        
        return message.message_id
    
    async def broadcast_message(
        self,
        sender_id: str,
        content: Any,
        message_type: str = "broadcast",
        exclude_agents: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Broadcast a message to all agents
        
        Args:
            sender_id: ID of the sending agent
            content: Message content
            message_type: Type of message
            exclude_agents: List of agent IDs to exclude
            metadata: Additional message metadata
            
        Returns:
            List of sent message IDs
        """
        exclude_agents = exclude_agents or []
        message_ids = []
        
        for agent_id in self.agents:
            if agent_id not in exclude_agents and agent_id != sender_id:
                message_id = await self.send_message(
                    sender_id=sender_id,
                    receiver_id=agent_id,
                    content=content,
                    message_type=message_type,
                    metadata=metadata
                )
                message_ids.append(message_id)
        
        return message_ids
    
    async def create_task(
        self,
        task_description: str,
        task_type: str = "general",
        assigned_agents: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new task
        
        Args:
            task_description: Description of the task
            task_type: Type of task
            assigned_agents: List of agent IDs assigned to the task
            metadata: Additional task metadata
            
        Returns:
            ID of the created task
        """
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        # Create task
        task = {
            "task_id": task_id,
            "description": task_description,
            "type": task_type,
            "assigned_agents": assigned_agents or [],
            "status": "created",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Store task
        self.active_tasks[task_id] = task
        
        # Store in memory
        await self.memory_manager.store_memory(
            content=f"Created task {task_id}: {task_description}",
            memory_type="task",
            metadata={
                "task_id": task_id,
                "task_type": task_type,
                "assigned_agents": assigned_agents or []
            }
        )
        
        # Notify assigned agents
        if assigned_agents:
            for agent_id in assigned_agents:
                if agent_id in self.agents:
                    await self.send_message(
                        sender_id=self.orchestrator_id,
                        receiver_id=agent_id,
                        content={
                            "task_id": task_id,
                            "description": task_description,
                            "type": task_type
                        },
                        message_type="task_assignment"
                    )
        
        return task_id
    
    async def update_task_status(self, task_id: str, status: str, result: Optional[Any] = None) -> bool:
        """
        Update the status of a task
        
        Args:
            task_id: ID of the task to update
            status: New status of the task
            result: Optional task result
            
        Returns:
            True if successful, False otherwise
        """
        if task_id not in self.active_tasks:
            return False
        
        # Update task
        self.active_tasks[task_id]["status"] = status
        self.active_tasks[task_id]["updated_at"] = datetime.now().isoformat()
        
        if result is not None:
            self.active_tasks[task_id]["result"] = result
        
        # Store in memory
        await self.memory_manager.store_memory(
            content=f"Updated task {task_id} status to {status}",
            memory_type="task",
            metadata={
                "task_id": task_id,
                "status": status,
                "has_result": result is not None
            }
        )
        
        # Notify assigned agents
        assigned_agents = self.active_tasks[task_id].get("assigned_agents", [])
        for agent_id in assigned_agents:
            if agent_id in self.agents:
                await self.send_message(
                    sender_id=self.orchestrator_id,
                    receiver_id=agent_id,
                    content={
                        "task_id": task_id,
                        "status": status,
                        "result": result
                    },
                    message_type="task_update"
                )
        
        return True
    
    async def process_user_request(self, user_id: str, request: str) -> Dict[str, Any]:
        """
        Process a request from a user
        
        Args:
            user_id: ID of the user making the request
            request: The user's request
            
        Returns:
            Response to the user
        """
        # Use LLM to determine the best way to handle the request
        if not self.llm_client:
            return {
                "success": False,
                "error": "LLM client not available"
            }
        
        async with self.llm_client as client:
            # Analyze request
            analysis_prompt = f"""
You are an orchestrator for a multi-agent system. A user has made the following request:

"{request}"

Analyze this request and determine:
1. What type of task is this? (e.g., question answering, creative writing, problem solving, etc.)
2. Which agent types would be best suited to handle this request?
3. Should this be handled by a single agent or multiple agents collaborating?
4. What specific capabilities are needed to fulfill this request?

Respond with a JSON object containing:
{{
    "task_type": "string",
    "agent_types": ["string"],
    "collaboration_needed": boolean,
    "required_capabilities": ["string"],
    "task_description": "string"
}}
"""
            
            analysis = await client.generate_structured(
                analysis_prompt,
                {
                    "task_type": "string",
                    "agent_types": "array",
                    "collaboration_needed": "boolean",
                    "required_capabilities": "array",
                    "task_description": "string"
                },
                "You are an orchestrator analyzing user requests to determine the best way to handle them."
            )
            
            if not analysis:
                return {
                    "success": False,
                    "error": "Failed to analyze request"
                }
            
            # Create task
            task_id = await self.create_task(
                task_description=analysis.get("task_description", request),
                task_type=analysis.get("task_type", "general"),
                metadata={
                    "user_id": user_id,
                    "original_request": request,
                    "analysis": analysis
                }
            )
            
            # Determine which agents to use
            if analysis.get("collaboration_needed", False):
                # Create or select multiple agents
                agent_ids = []
                for agent_type in analysis.get("agent_types", ["assistant"]):
                    # Check if we have an existing agent of this type
                    existing_agents = [
                        agent_id for agent_id, agent in self.agents.items()
                        if agent.role.lower() == agent_type.lower()
                    ]
                    
                    if existing_agents:
                        agent_ids.append(existing_agents[0])
                    else:
                        # Create new agent
                        new_agent_id = await self.create_agent(
                            agent_type="task" if agent_type == "general" else agent_type,
                            name=f"{agent_type.capitalize()} Agent",
                            role=agent_type,
                            description=f"Agent specialized in {agent_type} tasks",
                            capabilities=analysis.get("required_capabilities", [])
                        )
                        agent_ids.append(new_agent_id)
                
                # Create conversation for collaboration
                conversation_id = await self.create_conversation(
                    participants=agent_ids,
                    metadata={
                        "task_id": task_id,
                        "user_id": user_id
                    }
                )
                
                # Update task with assigned agents and conversation
                self.active_tasks[task_id]["assigned_agents"] = agent_ids
                self.active_tasks[task_id]["conversation_id"] = conversation_id
                
                # Send initial message to start collaboration
                await self.send_message(
                    sender_id=self.orchestrator_id,
                    receiver_id=agent_ids[0],  # Send to first agent
                    content={
                        "task_id": task_id,
                        "user_request": request,
                        "collaboration_needed": True,
                        "collaborators": agent_ids
                    },
                    message_type="task_assignment",
                    conversation_id=conversation_id
                )
                
                return {
                    "success": True,
                    "message": "Request is being processed by multiple agents",
                    "task_id": task_id,
                    "conversation_id": conversation_id,
                    "assigned_agents": agent_ids
                }
            else:
                # Use single agent
                agent_type = analysis.get("agent_types", ["assistant"])[0]
                
                # Check if we have an existing agent of this type
                existing_agents = [
                    agent_id for agent_id, agent in self.agents.items()
                    if agent.role.lower() == agent_type.lower()
                ]
                
                agent_id = None
                if existing_agents:
                    agent_id = existing_agents[0]
                else:
                    # Create new agent
                    agent_id = await self.create_agent(
                        agent_type="assistant" if agent_type == "general" else agent_type,
                        name=f"{agent_type.capitalize()} Agent",
                        role=agent_type,
                        description=f"Agent specialized in {agent_type} tasks",
                        capabilities=analysis.get("required_capabilities", [])
                    )
                
                # Update task with assigned agent
                self.active_tasks[task_id]["assigned_agents"] = [agent_id]
                
                # Send task to agent
                await self.send_message(
                    sender_id=self.orchestrator_id,
                    receiver_id=agent_id,
                    content={
                        "task_id": task_id,
                        "user_request": request
                    },
                    message_type="task_assignment"
                )
                
                return {
                    "success": True,
                    "message": "Request is being processed",
                    "task_id": task_id,
                    "assigned_agent": agent_id
                }
    
    async def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of an agent
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Agent status
        """
        if agent_id not in self.agents:
            return None
        
        return self.agents[agent_id].get_state()
    
    async def get_all_agents(self) -> List[Dict[str, Any]]:
        """
        Get information about all registered agents
        
        Returns:
            List of agent information
        """
        return [agent.get_state() for agent in self.agents.values()]
    
    async def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a conversation
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Conversation data
        """
        if conversation_id not in self.conversations:
            return None
        
        return self.conversations[conversation_id].to_dict()
    
    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a task
        
        Args:
            task_id: ID of the task
            
        Returns:
            Task data
        """
        return self.active_tasks.get(task_id)