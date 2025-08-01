"""Task-oriented agent for handling specific tasks"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Set

from ..enhanced_base_agent import EnhancedBaseAgent
from ...utils.config import Config
from ...utils.llm_client import LLMClient
from ...memory.memory_manager import MemoryManager
from ...tools.tool_registry import ToolRegistry
from ...communication.message import Message

class TaskAgent(EnhancedBaseAgent):
    """
    Agent specialized in completing specific tasks
    
    Attributes:
        active_tasks: Dictionary of active tasks
        completed_tasks: List of completed tasks
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
        
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.completed_tasks: List[Dict[str, Any]] = []
        
        # Register message handlers
        self.message_handlers = {
            "task_assignment": self._handle_task_assignment,
            "task_update": self._handle_task_update,
            "query": self._handle_query,
            "command": self._handle_command
        }
    
    async def _generate_personality(self):
        """Generate personality traits for this agent"""
        self.personality = {
            "efficiency": 0.8,
            "thoroughness": 0.7,
            "reliability": 0.9,
            "adaptability": 0.6,
            "focus": 0.8
        }
    
    async def _generate_initial_goals(self):
        """Generate initial goals for this agent"""
        self.goals = [
            "Complete assigned tasks efficiently and accurately",
            "Continuously improve task performance",
            "Collaborate effectively with other agents when needed",
            "Maintain a high standard of work quality"
        ]
    
    async def _setup_capabilities(self):
        """Set up agent capabilities"""
        self.capabilities = {
            "basic_reasoning",
            "task_planning",
            "task_execution",
            "communication"
        }
        
        if self.tool_registry:
            self.available_tools = await self.tool_registry.get_available_tools(self.capabilities)
    
    async def _handle_task_assignment(self, message: Message) -> Optional[Message]:
        """
        Handle task assignment message
        
        Args:
            message: Task assignment message
            
        Returns:
            Optional response message
        """
        task_data = message.content
        task_id = task_data.get("task_id")
        
        if not task_id:
            return Message(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                content="Error: Task ID not provided",
                type="error",
                reference_message_id=message.message_id
            )
        
        # Store task
        self.active_tasks[task_id] = {
            "task_id": task_id,
            "description": task_data.get("user_request", ""),
            "status": "assigned",
            "collaboration_needed": task_data.get("collaboration_needed", False),
            "collaborators": task_data.get("collaborators", []),
            "message_id": message.message_id
        }
        
        # Process task
        task_result = await self._process_task(task_id)
        
        # Create response
        return Message(
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            content={
                "task_id": task_id,
                "status": "completed",
                "result": task_result
            },
            type="task_result",
            reference_message_id=message.message_id
        )
    
    async def _handle_task_update(self, message: Message) -> Optional[Message]:
        """
        Handle task update message
        
        Args:
            message: Task update message
            
        Returns:
            Optional response message
        """
        task_data = message.content
        task_id = task_data.get("task_id")
        
        if not task_id or task_id not in self.active_tasks:
            return Message(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                content=f"Error: Task {task_id} not found",
                type="error",
                reference_message_id=message.message_id
            )
        
        # Update task
        self.active_tasks[task_id]["status"] = task_data.get("status", self.active_tasks[task_id]["status"])
        
        if "result" in task_data:
            self.active_tasks[task_id]["result"] = task_data["result"]
        
        # Acknowledge update
        return Message(
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            content=f"Task {task_id} updated",
            type="acknowledgement",
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
            
            # Construct query prompt
            query_prompt = f"""
You are {self.name}, a {self.role}.
{self.description}

Your personality: {json.dumps(self.personality, indent=2)}
Your current goals: {json.dumps(self.goals, indent=2)}

You have received the following query:
{query}

Relevant memories:
{chr(10).join(f"- {memory}" for memory in memories)}

Provide a helpful and accurate response to this query.
"""
            
            response = await client.generate(
                query_prompt,
                f"You are {self.name}, a {self.role}. Respond to queries in a helpful and accurate manner."
            )
            
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
    
    async def _process_task(self, task_id: str) -> Any:
        """
        Process a task
        
        Args:
            task_id: ID of the task to process
            
        Returns:
            Task result
        """
        if task_id not in self.active_tasks:
            return "Error: Task not found"
        
        task = self.active_tasks[task_id]
        
        # Update task status
        task["status"] = "processing"
        
        if not self.llm_client:
            task["status"] = "failed"
            return "Error: LLM client not available"
        
        async with self.llm_client as client:
            # Determine if task requires tools
            task_analysis_prompt = f"""
You are {self.name}, a {self.role}.
{self.description}

You have been assigned the following task:
{task["description"]}

Available tools: {', '.join(self.available_tools) if self.available_tools else 'None'}

Analyze this task and determine:
1. What steps are needed to complete this task?
2. Which tools (if any) would be helpful for completing this task?
3. Is collaboration with other agents needed? If so, what kind of collaboration?

Respond with a JSON object containing:
{{
    "steps": ["list of steps to complete the task"],
    "tools_needed": ["list of tool names that would be helpful"],
    "collaboration_needed": boolean,
    "collaboration_type": "string (only if collaboration_needed is true)"
}}
"""
            
            task_analysis = await client.generate_structured(
                task_analysis_prompt,
                {
                    "steps": "array",
                    "tools_needed": "array",
                    "collaboration_needed": "boolean",
                    "collaboration_type": "string"
                },
                f"You are {self.name}, analyzing a task to determine how to complete it."
            )
            
            if not task_analysis:
                task["status"] = "failed"
                return "Error: Failed to analyze task"
            
            # Store analysis in task
            task["analysis"] = task_analysis
            
            # Execute task steps
            result = ""
            for i, step in enumerate(task_analysis.get("steps", [])):
                step_prompt = f"""
You are {self.name}, a {self.role}.
{self.description}

You are working on the following task:
{task["description"]}

You are currently on step {i+1} of {len(task_analysis.get("steps", []))}: {step}

Previous result: {result}

Available tools: {', '.join(self.available_tools) if self.available_tools else 'None'}

Execute this step and provide the result. If you need to use a tool, specify which tool and what arguments to use.
"""
                
                step_result = await client.generate(
                    step_prompt,
                    f"You are {self.name}, executing a step in a task."
                )
                
                # Check if step result indicates tool usage
                if any(tool in step_result.lower() for tool in self.available_tools):
                    # Extract tool name and arguments using LLM
                    tool_extraction_prompt = f"""
Based on the following step result, extract the tool that needs to be used and its arguments:

{step_result}

Available tools: {', '.join(self.available_tools)}

Respond with a JSON object containing:
{{
    "tool_name": "name of the tool to use",
    "args": {{
        "argument1": "value1",
        "argument2": "value2",
        ...
    }}
}}
"""
                    
                    tool_info = await client.generate_structured(
                        tool_extraction_prompt,
                        {
                            "tool_name": "string",
                            "args": "object"
                        },
                        "You are extracting tool usage information from text."
                    )
                    
                    if tool_info and "tool_name" in tool_info:
                        tool_name = tool_info["tool_name"]
                        tool_args = tool_info.get("args", {})
                        
                        # Use tool
                        tool_result = await self.use_tool(tool_name, **tool_args)
                        
                        # Update step result with tool result
                        step_result += f"\n\nTool result ({tool_name}): {tool_result}"
                
                # Append step result to overall result
                result += f"\n\nStep {i+1}: {step}\n{step_result}"
            
            # Mark task as completed
            task["status"] = "completed"
            task["result"] = result.strip()
            
            # Move to completed tasks
            self.completed_tasks.append(task)
            del self.active_tasks[task_id]
            
            return result.strip()
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state of the agent"""
        state = super().get_state()
        state.update({
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks)
        })
        return state