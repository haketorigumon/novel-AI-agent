"""Assistant agent for general-purpose assistance"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Set

from ..enhanced_base_agent import EnhancedBaseAgent
from ...utils.config import Config
from ...utils.llm_client import LLMClient
from ...memory.memory_manager import MemoryManager
from ...tools.tool_registry import ToolRegistry
from ...communication.message import Message

class AssistantAgent(EnhancedBaseAgent):
    """
    Agent specialized in providing general assistance and answering questions
    
    Attributes:
        conversation_history: Dictionary of conversation histories by user ID
    """
    
    def __init__(
        self,
        name: str,
        role: str,
        description: str = "",
        agent_id: Optional[str] = None,
        config: Optional[Config] = None,
        llm_client: Optional[LLMClient] = None,
        memory_manager: Optional[MemoryManager] = None,
        tool_registry: Optional[ToolRegistry] = None
    ):
        super().__init__(
            agent_id=agent_id,
            name=name,
            role=role,
            description=description,
            config=config,
            llm_client=llm_client,
            memory_manager=memory_manager,
            tool_registry=tool_registry
        )
        
        self.conversation_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Register message handlers
        self.message_handlers = {
            "text": self._handle_text_message,
            "query": self._handle_query,
            "command": self._handle_command,
            "task_assignment": self._handle_task_assignment
        }
    
    async def _generate_personality(self):
        """Generate personality traits for this agent"""
        self.personality = {
            "helpfulness": 0.9,
            "friendliness": 0.8,
            "patience": 0.7,
            "knowledgeability": 0.8,
            "empathy": 0.7
        }
    
    async def _generate_initial_goals(self):
        """Generate initial goals for this agent"""
        self.goals = [
            "Provide helpful and accurate information",
            "Assist users in accomplishing their goals",
            "Maintain a friendly and supportive demeanor",
            "Continuously improve knowledge and capabilities"
        ]
    
    async def _setup_capabilities(self):
        """Set up agent capabilities"""
        self.capabilities = {
            "basic_reasoning",
            "communication",
            "information_retrieval",
            "language_understanding"
        }
        
        if self.tool_registry:
            self.available_tools = await self.tool_registry.get_available_tools(self.capabilities)
    
    async def _handle_text_message(self, message: Message) -> Optional[Message]:
        """
        Handle text message
        
        Args:
            message: Text message
            
        Returns:
            Response message
        """
        # Add to conversation history
        sender_id = message.sender_id
        if sender_id not in self.conversation_history:
            self.conversation_history[sender_id] = []
        
        self.conversation_history[sender_id].append({
            "role": "user",
            "content": message.content,
            "timestamp": message.timestamp.isoformat()
        })
        
        # Generate response
        response = await self._generate_response(sender_id, message.content)
        
        # Add response to conversation history
        self.conversation_history[sender_id].append({
            "role": "assistant",
            "content": response,
            "timestamp": message.timestamp.isoformat()
        })
        
        return Message(
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            content=response,
            type="text",
            reference_message_id=message.message_id
        )
    
    async def _handle_query(self, message: Message) -> Optional[Message]:
        """
        Handle query message
        
        Args:
            message: Query message
            
        Returns:
            Response message
        """
        query = message.content
        
        # Add to conversation history
        sender_id = message.sender_id
        if sender_id not in self.conversation_history:
            self.conversation_history[sender_id] = []
        
        self.conversation_history[sender_id].append({
            "role": "user",
            "content": query,
            "timestamp": message.timestamp.isoformat()
        })
        
        # Generate response
        response = await self._generate_response(sender_id, query)
        
        # Add response to conversation history
        self.conversation_history[sender_id].append({
            "role": "assistant",
            "content": response,
            "timestamp": message.timestamp.isoformat()
        })
        
        return Message(
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            content=response,
            type="response",
            reference_message_id=message.message_id
        )
    
    async def _handle_command(self, message: Message) -> Optional[Message]:
        """
        Handle command message
        
        Args:
            message: Command message
            
        Returns:
            Response message
        """
        command = message.content
        
        if isinstance(command, str):
            # Simple text command
            return Message(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                content=f"Received command: {command}",
                type="acknowledgement",
                reference_message_id=message.message_id
            )
        elif isinstance(command, dict):
            # Structured command
            command_type = command.get("type")
            
            if command_type == "use_tool":
                # Use a tool
                tool_name = command.get("tool_name")
                tool_args = command.get("args", {})
                
                if not tool_name:
                    return Message(
                        sender_id=self.agent_id,
                        receiver_id=message.sender_id,
                        content="Error: Tool name not provided",
                        type="error",
                        reference_message_id=message.message_id
                    )
                
                # Execute tool
                tool_result = await self.use_tool(tool_name, **tool_args)
                
                return Message(
                    sender_id=self.agent_id,
                    receiver_id=message.sender_id,
                    content={
                        "tool_name": tool_name,
                        "result": tool_result
                    },
                    type="tool_result",
                    reference_message_id=message.message_id
                )
            elif command_type == "clear_history":
                # Clear conversation history
                sender_id = message.sender_id
                if sender_id in self.conversation_history:
                    self.conversation_history[sender_id] = []
                
                return Message(
                    sender_id=self.agent_id,
                    receiver_id=message.sender_id,
                    content="Conversation history cleared",
                    type="acknowledgement",
                    reference_message_id=message.message_id
                )
            else:
                return Message(
                    sender_id=self.agent_id,
                    receiver_id=message.sender_id,
                    content=f"Unknown command type: {command_type}",
                    type="error",
                    reference_message_id=message.message_id
                )
        
        return Message(
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            content="Invalid command format",
            type="error",
            reference_message_id=message.message_id
        )
    
    async def _handle_task_assignment(self, message: Message) -> Optional[Message]:
        """
        Handle task assignment message
        
        Args:
            message: Task assignment message
            
        Returns:
            Response message
        """
        task_data = message.content
        task_id = task_data.get("task_id")
        user_request = task_data.get("user_request", "")
        
        if not task_id:
            return Message(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                content="Error: Task ID not provided",
                type="error",
                reference_message_id=message.message_id
            )
        
        # Process the user request
        response = await self._generate_response("task_" + task_id, user_request)
        
        return Message(
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            content={
                "task_id": task_id,
                "status": "completed",
                "result": response
            },
            type="task_result",
            reference_message_id=message.message_id
        )
    
    async def _generate_response(self, user_id: str, message: str) -> str:
        """
        Generate a response to a message
        
        Args:
            user_id: ID of the user
            message: User message
            
        Returns:
            Generated response
        """
        if not self.llm_client:
            return "Error: LLM client not available"
        
        async with self.llm_client as client:
            # Retrieve relevant memories
            memories = []
            if self.memory_manager:
                memories = await self.memory_manager.retrieve_relevant_memories(message)
            
            # Get conversation history
            conversation = self.conversation_history.get(user_id, [])
            recent_conversation = conversation[-10:] if len(conversation) > 10 else conversation
            
            # Construct response prompt
            response_prompt = f"""
You are {self.name}, a {self.role}.
{self.description}

Your personality: {json.dumps(self.personality, indent=2)}
Your current goals: {json.dumps(self.goals, indent=2)}

Recent conversation:
{json.dumps(recent_conversation, indent=2)}

User message: {message}

Relevant memories:
{chr(10).join(f"- {memory}" for memory in memories)}

Available tools: {', '.join(self.available_tools) if self.available_tools else 'None'}

Provide a helpful, accurate, and friendly response to the user's message.
"""
            
            return await client.generate(
                response_prompt,
                f"You are {self.name}, a {self.role}. Respond to user messages in a helpful and friendly manner."
            )
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state of the agent"""
        state = super().get_state()
        state.update({
            "active_conversations": len(self.conversation_history)
        })
        return state