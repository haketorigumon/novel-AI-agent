"""Creative agent for generating creative content"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Set

from ..enhanced_base_agent import EnhancedBaseAgent
from ...utils.config import Config
from ...utils.llm_client import LLMClient
from ...memory.memory_manager import MemoryManager
from ...tools.tool_registry import ToolRegistry
from ...communication.message import Message

class CreativeAgent(EnhancedBaseAgent):
    """
    Agent specialized in creative content generation
    
    Attributes:
        creative_domain: Domain of creativity (writing, art, music, etc.)
        style_preferences: Preferred creative styles
        created_works: List of created works
    """
    
    def __init__(
        self,
        name: str,
        role: str,
        description: str = "",
        creative_domain: str = "",
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
        
        self.creative_domain = creative_domain or role
        self.style_preferences = {}
        self.created_works = []
        
        # Register message handlers
        self.message_handlers = {
            "creative_request": self._handle_creative_request,
            "feedback": self._handle_feedback,
            "task_assignment": self._handle_task_assignment
        }
    
    async def _generate_personality(self):
        """Generate personality traits for this agent"""
        self.personality = {
            "creativity": 0.9,
            "originality": 0.8,
            "expressiveness": 0.8,
            "curiosity": 0.7,
            "openness": 0.9
        }
    
    async def _generate_initial_goals(self):
        """Generate initial goals for this agent"""
        self.goals = [
            f"Create engaging and original content in {self.creative_domain}",
            "Develop a distinctive creative style",
            "Push boundaries and explore new creative approaches",
            "Evoke emotional responses through creative work"
        ]
    
    async def _setup_capabilities(self):
        """Set up agent capabilities"""
        self.capabilities = {
            "basic_reasoning",
            "communication",
            "creativity",
            "storytelling",
            "aesthetic_judgment"
        }
        
        if self.tool_registry:
            self.available_tools = await self.tool_registry.get_available_tools(self.capabilities)
    
    async def _handle_creative_request(self, message: Message) -> Optional[Message]:
        """
        Handle creative request message
        
        Args:
            message: Creative request message
            
        Returns:
            Response message with creative content
        """
        request = message.content
        
        if not self.llm_client:
            return Message(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                content="Error: LLM client not available",
                type="error",
                reference_message_id=message.message_id
            )
        
        async with self.llm_client as client:
            # Retrieve relevant memories
            memories = []
            if self.memory_manager:
                memories = await self.memory_manager.retrieve_relevant_memories(str(request))
            
            # Construct creative prompt
            creative_prompt = f"""
You are {self.name}, a creative {self.role} specializing in {self.creative_domain}.
{self.description}

Your personality: {json.dumps(self.personality, indent=2)}
Your current goals: {json.dumps(self.goals, indent=2)}
Your style preferences: {json.dumps(self.style_preferences, indent=2)}

You have received the following creative request:
{request}

Relevant memories and past works:
{chr(10).join(f"- {memory}" for memory in memories)}

Create an original and engaging piece in your domain of {self.creative_domain}.
Your creation should:
1. Be original and distinctive
2. Reflect your creative style and personality
3. Fulfill the specific requirements of the request
4. Evoke emotion and engage the audience
"""
            
            creation = await client.generate(
                creative_prompt,
                f"You are {self.name}, a creative {self.role} specializing in {self.creative_domain}. Create original content."
            )
            
            # Store the creation
            work_id = f"work_{len(self.created_works) + 1}"
            work = {
                "work_id": work_id,
                "title": f"Creation {len(self.created_works) + 1}",
                "content": creation,
                "request": request,
                "timestamp": message.timestamp.isoformat()
            }
            
            self.created_works.append(work)
            
            # Store in memory
            if self.memory_manager:
                await self.memory_manager.store_memory(
                    content=f"Created work: {work_id}",
                    memory_type="creation",
                    metadata={
                        "work_id": work_id,
                        "request": request
                    },
                    importance=0.7
                )
            
            return Message(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                content={
                    "work_id": work_id,
                    "content": creation
                },
                type="creative_response",
                reference_message_id=message.message_id
            )
    
    async def _handle_feedback(self, message: Message) -> Optional[Message]:
        """
        Handle feedback message
        
        Args:
            message: Feedback message
            
        Returns:
            Response message
        """
        feedback = message.content
        
        if isinstance(feedback, str):
            # Simple text feedback
            # Store in memory
            if self.memory_manager:
                await self.memory_manager.store_memory(
                    content=f"Received feedback: {feedback}",
                    memory_type="feedback",
                    metadata={
                        "sender_id": message.sender_id
                    },
                    importance=0.8
                )
            
            # Reflect on feedback
            await self.reflect({
                "type": "feedback",
                "content": feedback,
                "sender_id": message.sender_id
            })
            
            return Message(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                content="Thank you for your feedback. I'll take it into consideration for future creations.",
                type="acknowledgement",
                reference_message_id=message.message_id
            )
        elif isinstance(feedback, dict):
            # Structured feedback
            work_id = feedback.get("work_id")
            feedback_text = feedback.get("feedback", "")
            rating = feedback.get("rating")
            
            # Find the work
            work = None
            for w in self.created_works:
                if w.get("work_id") == work_id:
                    work = w
                    break
            
            if work:
                # Update work with feedback
                if "feedback" not in work:
                    work["feedback"] = []
                
                work["feedback"].append({
                    "sender_id": message.sender_id,
                    "feedback": feedback_text,
                    "rating": rating,
                    "timestamp": message.timestamp.isoformat()
                })
                
                # Store in memory
                if self.memory_manager:
                    await self.memory_manager.store_memory(
                        content=f"Received feedback for work {work_id}: {feedback_text}",
                        memory_type="feedback",
                        metadata={
                            "work_id": work_id,
                            "sender_id": message.sender_id,
                            "rating": rating
                        },
                        importance=0.8
                    )
                
                # Reflect on feedback
                await self.reflect({
                    "type": "work_feedback",
                    "work_id": work_id,
                    "feedback": feedback_text,
                    "rating": rating,
                    "sender_id": message.sender_id
                })
                
                return Message(
                    sender_id=self.agent_id,
                    receiver_id=message.sender_id,
                    content=f"Thank you for your feedback on work {work_id}. I'll take it into consideration for future creations.",
                    type="acknowledgement",
                    reference_message_id=message.message_id
                )
            else:
                return Message(
                    sender_id=self.agent_id,
                    receiver_id=message.sender_id,
                    content=f"Error: Work {work_id} not found",
                    type="error",
                    reference_message_id=message.message_id
                )
        
        return Message(
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            content="Invalid feedback format",
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
        
        if not self.llm_client:
            return Message(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                content={
                    "task_id": task_id,
                    "status": "failed",
                    "result": "Error: LLM client not available"
                },
                type="task_result",
                reference_message_id=message.message_id
            )
        
        async with self.llm_client as client:
            # Analyze if the task is creative
            analysis_prompt = f"""
You are {self.name}, a creative {self.role} specializing in {self.creative_domain}.
{self.description}

You have been assigned the following task:
{user_request}

Analyze whether this task involves creative content generation in your domain ({self.creative_domain}).
Respond with a JSON object containing:
{{
    "is_creative_task": boolean,
    "confidence": float (0-1),
    "explanation": "string"
}}
"""
            
            analysis = await client.generate_structured(
                analysis_prompt,
                {
                    "is_creative_task": "boolean",
                    "confidence": "number",
                    "explanation": "string"
                },
                f"You are {self.name}, analyzing whether a task involves creative content generation."
            )
            
            if not analysis:
                return Message(
                    sender_id=self.agent_id,
                    receiver_id=message.sender_id,
                    content={
                        "task_id": task_id,
                        "status": "failed",
                        "result": "Error: Failed to analyze task"
                    },
                    type="task_result",
                    reference_message_id=message.message_id
                )
            
            # If not a creative task, return explanation
            if not analysis.get("is_creative_task", False) and analysis.get("confidence", 0) < 0.7:
                return Message(
                    sender_id=self.agent_id,
                    receiver_id=message.sender_id,
                    content={
                        "task_id": task_id,
                        "status": "declined",
                        "result": f"This task doesn't appear to involve creative content generation in my domain ({self.creative_domain}). {analysis.get('explanation', '')}"
                    },
                    type="task_result",
                    reference_message_id=message.message_id
                )
            
            # Process the task as a creative request
            creative_prompt = f"""
You are {self.name}, a creative {self.role} specializing in {self.creative_domain}.
{self.description}

Your personality: {json.dumps(self.personality, indent=2)}
Your current goals: {json.dumps(self.goals, indent=2)}
Your style preferences: {json.dumps(self.style_preferences, indent=2)}

You have been assigned the following creative task:
{user_request}

Create an original and engaging piece in your domain of {self.creative_domain}.
Your creation should:
1. Be original and distinctive
2. Reflect your creative style and personality
3. Fulfill the specific requirements of the task
4. Evoke emotion and engage the audience
"""
            
            creation = await client.generate(
                creative_prompt,
                f"You are {self.name}, a creative {self.role} specializing in {self.creative_domain}. Create original content."
            )
            
            # Store the creation
            work_id = f"work_{len(self.created_works) + 1}"
            work = {
                "work_id": work_id,
                "title": f"Task {task_id}",
                "content": creation,
                "request": user_request,
                "task_id": task_id,
                "timestamp": message.timestamp.isoformat()
            }
            
            self.created_works.append(work)
            
            # Store in memory
            if self.memory_manager:
                await self.memory_manager.store_memory(
                    content=f"Created work for task {task_id}: {work_id}",
                    memory_type="creation",
                    metadata={
                        "work_id": work_id,
                        "task_id": task_id,
                        "request": user_request
                    },
                    importance=0.7
                )
            
            return Message(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                content={
                    "task_id": task_id,
                    "status": "completed",
                    "result": creation,
                    "work_id": work_id
                },
                type="task_result",
                reference_message_id=message.message_id
            )
    
    async def update_style_preferences(self, style_updates: Dict[str, float]) -> bool:
        """
        Update style preferences
        
        Args:
            style_updates: Dictionary of style updates
            
        Returns:
            True if successful, False otherwise
        """
        for style, value in style_updates.items():
            self.style_preferences[style] = max(0, min(1, value))
        
        # Store in memory
        if self.memory_manager:
            await self.memory_manager.store_memory(
                content=f"Updated style preferences: {json.dumps(style_updates)}",
                memory_type="style",
                metadata={
                    "style_updates": style_updates
                },
                importance=0.6
            )
        
        return True
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state of the agent"""
        state = super().get_state()
        state.update({
            "creative_domain": self.creative_domain,
            "style_preferences": self.style_preferences,
            "created_works_count": len(self.created_works)
        })
        return state