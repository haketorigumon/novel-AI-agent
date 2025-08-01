"""Expert agent for specialized knowledge domains"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Set

from ..enhanced_base_agent import EnhancedBaseAgent
from ...utils.config import Config
from ...utils.llm_client import LLMClient
from ...memory.memory_manager import MemoryManager
from ...tools.tool_registry import ToolRegistry
from ...communication.message import Message

class ExpertAgent(EnhancedBaseAgent):
    """
    Agent specialized in a particular knowledge domain
    
    Attributes:
        expertise_area: Area of expertise
        knowledge_base: Specialized knowledge base
    """
    
    def __init__(
        self,
        name: str,
        role: str,
        description: str = "",
        expertise_area: str = "",
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
        
        self.expertise_area = expertise_area or role
        self.knowledge_base = {}
        
        # Register message handlers
        self.message_handlers = {
            "query": self._handle_query,
            "consultation": self._handle_consultation,
            "task_assignment": self._handle_task_assignment
        }
    
    async def _generate_personality(self):
        """Generate personality traits for this agent"""
        self.personality = {
            "analytical": 0.9,
            "precise": 0.8,
            "authoritative": 0.7,
            "thoughtful": 0.8,
            "detail_oriented": 0.9
        }
    
    async def _generate_initial_goals(self):
        """Generate initial goals for this agent"""
        self.goals = [
            f"Provide expert knowledge in {self.expertise_area}",
            "Deliver accurate and detailed information",
            "Explain complex concepts clearly",
            "Stay current with developments in the field"
        ]
    
    async def _setup_capabilities(self):
        """Set up agent capabilities"""
        self.capabilities = {
            "basic_reasoning",
            "communication",
            "expert_knowledge",
            "critical_thinking",
            "analysis"
        }
        
        if self.tool_registry:
            self.available_tools = await self.tool_registry.get_available_tools(self.capabilities)
    
    async def _handle_query(self, message: Message) -> Optional[Message]:
        """
        Handle query message
        
        Args:
            message: Query message
            
        Returns:
            Response message
        """
        query = message.content
        
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
                memories = await self.memory_manager.retrieve_relevant_memories(query)
            
            # Construct expert response prompt
            expert_prompt = f"""
You are {self.name}, a {self.role} with expertise in {self.expertise_area}.
{self.description}

Your personality: {json.dumps(self.personality, indent=2)}
Your current goals: {json.dumps(self.goals, indent=2)}

You have received the following query:
{query}

Relevant memories:
{chr(10).join(f"- {memory}" for memory in memories)}

Provide an expert response that demonstrates your deep knowledge of {self.expertise_area}.
Your response should be:
1. Accurate and well-informed
2. Detailed and comprehensive
3. Clear and accessible, even when explaining complex concepts
4. Authoritative, citing relevant principles or sources where appropriate
"""
            
            response = await client.generate(
                expert_prompt,
                f"You are {self.name}, an expert in {self.expertise_area}. Provide authoritative and detailed responses."
            )
            
            return Message(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                content=response,
                type="response",
                reference_message_id=message.message_id
            )
    
    async def _handle_consultation(self, message: Message) -> Optional[Message]:
        """
        Handle consultation message
        
        Args:
            message: Consultation message
            
        Returns:
            Response message
        """
        consultation = message.content
        
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
                memories = await self.memory_manager.retrieve_relevant_memories(str(consultation))
            
            # Construct consultation response prompt
            consultation_prompt = f"""
You are {self.name}, a {self.role} with expertise in {self.expertise_area}.
{self.description}

Your personality: {json.dumps(self.personality, indent=2)}
Your current goals: {json.dumps(self.goals, indent=2)}

You have been consulted on the following matter:
{consultation}

Relevant memories:
{chr(10).join(f"- {memory}" for memory in memories)}

Provide your expert consultation, including:
1. Analysis of the situation or problem
2. Expert recommendations based on your knowledge of {self.expertise_area}
3. Potential alternatives or considerations
4. Any risks or limitations to be aware of
"""
            
            response = await client.generate(
                consultation_prompt,
                f"You are {self.name}, an expert in {self.expertise_area} providing a consultation."
            )
            
            return Message(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                content=response,
                type="consultation_response",
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
            # Analyze if the task is within expertise
            analysis_prompt = f"""
You are {self.name}, a {self.role} with expertise in {self.expertise_area}.
{self.description}

You have been assigned the following task:
{user_request}

Analyze whether this task falls within your area of expertise ({self.expertise_area}).
Respond with a JSON object containing:
{{
    "within_expertise": boolean,
    "confidence": float (0-1),
    "explanation": "string"
}}
"""
            
            analysis = await client.generate_structured(
                analysis_prompt,
                {
                    "within_expertise": "boolean",
                    "confidence": "number",
                    "explanation": "string"
                },
                f"You are {self.name}, analyzing whether a task falls within your expertise."
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
            
            # If not within expertise, return explanation
            if not analysis.get("within_expertise", False) and analysis.get("confidence", 0) < 0.7:
                return Message(
                    sender_id=self.agent_id,
                    receiver_id=message.sender_id,
                    content={
                        "task_id": task_id,
                        "status": "declined",
                        "result": f"This task appears to be outside my area of expertise ({self.expertise_area}). {analysis.get('explanation', '')}"
                    },
                    type="task_result",
                    reference_message_id=message.message_id
                )
            
            # Process the task as an expert
            expert_response_prompt = f"""
You are {self.name}, a {self.role} with expertise in {self.expertise_area}.
{self.description}

Your personality: {json.dumps(self.personality, indent=2)}
Your current goals: {json.dumps(self.goals, indent=2)}

You have been assigned the following task:
{user_request}

Provide your expert response, drawing on your deep knowledge of {self.expertise_area}.
Your response should be:
1. Accurate and well-informed
2. Detailed and comprehensive
3. Clear and accessible, even when explaining complex concepts
4. Authoritative, citing relevant principles or sources where appropriate
"""
            
            response = await client.generate(
                expert_response_prompt,
                f"You are {self.name}, an expert in {self.expertise_area} completing an assigned task."
            )
            
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
    
    async def expand_knowledge_base(self, topic: str) -> bool:
        """
        Expand the agent's knowledge base on a specific topic
        
        Args:
            topic: Topic to expand knowledge on
            
        Returns:
            True if successful, False otherwise
        """
        if not self.llm_client:
            return False
        
        async with self.llm_client as client:
            knowledge_prompt = f"""
You are {self.name}, a {self.role} with expertise in {self.expertise_area}.
{self.description}

Generate detailed knowledge about the following topic related to {self.expertise_area}:
{topic}

Provide a comprehensive overview that includes:
1. Key concepts and definitions
2. Important principles or theories
3. Practical applications or implications
4. Current state of knowledge or research
5. Common misconceptions or debates

Format your response as a JSON object with the following structure:
{{
    "topic": "{topic}",
    "summary": "Brief summary of the topic",
    "key_concepts": [
        {{"name": "concept name", "definition": "concept definition"}}
    ],
    "principles": [
        {{"name": "principle name", "description": "principle description"}}
    ],
    "applications": [
        {{"name": "application name", "description": "application description"}}
    ],
    "current_state": "Description of current state",
    "misconceptions": [
        {{"misconception": "common misconception", "correction": "correct information"}}
    ]
}}
"""
            
            knowledge = await client.generate_structured(
                knowledge_prompt,
                {
                    "topic": "string",
                    "summary": "string",
                    "key_concepts": "array",
                    "principles": "array",
                    "applications": "array",
                    "current_state": "string",
                    "misconceptions": "array"
                },
                f"You are {self.name}, generating expert knowledge about {topic}."
            )
            
            if knowledge:
                # Add to knowledge base
                self.knowledge_base[topic] = knowledge
                
                # Store in memory
                if self.memory_manager:
                    await self.memory_manager.store_memory(
                        content=f"Expanded knowledge base on topic: {topic}",
                        memory_type="knowledge",
                        metadata={
                            "topic": topic,
                            "summary": knowledge.get("summary", "")
                        },
                        importance=0.8
                    )
                
                return True
            
            return False
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state of the agent"""
        state = super().get_state()
        state.update({
            "expertise_area": self.expertise_area,
            "knowledge_base_topics": list(self.knowledge_base.keys())
        })
        return state