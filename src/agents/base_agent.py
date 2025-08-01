"""Base agent class for all story agents"""

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..utils.config import Config
from ..utils.llm_client import LLMClient

class BaseAgent(ABC):
    """Base class for all agents in the story generation system"""
    
    def __init__(self, agent_id: str, config: Config, llm_client: LLMClient):
        self.agent_id = agent_id
        self.config = config
        self.llm_client = llm_client
        self.memory = []
        self.personality = {}
        self.goals = []
        self.relationships = {}
        self.created_at = datetime.now()
        self.last_action = None
    
    async def initialize(self):
        """Initialize the agent with personality and goals"""
        await self._generate_personality()
        await self._generate_initial_goals()
    
    @abstractmethod
    async def _generate_personality(self):
        """Generate personality traits for this agent"""
        pass
    
    @abstractmethod
    async def _generate_initial_goals(self):
        """Generate initial goals for this agent"""
        pass
    
    async def add_memory(self, memory_item: Dict[str, Any]):
        """Add a memory item to the agent's memory"""
        memory_item["timestamp"] = datetime.now().isoformat()
        self.memory.append(memory_item)
        
        # Keep memory within reasonable bounds
        if len(self.memory) > 100:
            self.memory = self.memory[-50:]  # Keep last 50 memories
    
    async def get_recent_memories(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent memories"""
        return self.memory[-count:] if self.memory else []
    
    async def update_relationship(self, other_agent_id: str, relationship_change: Dict[str, Any]):
        """Update relationship with another agent"""
        if other_agent_id not in self.relationships:
            self.relationships[other_agent_id] = {
                "trust": 0.5,
                "affection": 0.5,
                "respect": 0.5,
                "history": []
            }
        
        # Update relationship values
        for key, value in relationship_change.items():
            if key in self.relationships[other_agent_id] and key != "history":
                self.relationships[other_agent_id][key] = max(0, min(1, 
                    self.relationships[other_agent_id][key] + value))
        
        # Add to history
        self.relationships[other_agent_id]["history"].append({
            "timestamp": datetime.now().isoformat(),
            "change": relationship_change
        })
    
    async def make_decision(self, context: Dict[str, Any], options: List[str]) -> str:
        """Make a decision based on context and available options"""
        async with self.llm_client as client:
            decision_prompt = f"""
You are {self.agent_id} with the following characteristics:
Personality: {json.dumps(self.personality, indent=2)}
Current Goals: {json.dumps(self.goals, indent=2)}
Recent Memories: {json.dumps(await self.get_recent_memories(5), indent=2)}

Current Context: {json.dumps(context, indent=2)}

Available Options:
{chr(10).join(f"- {option}" for option in options)}

Based on your personality, goals, and the current situation, which option would you choose?
Respond with just the chosen option exactly as listed above.
"""
            
            response = await client.generate(
                decision_prompt,
                f"You are {self.agent_id}, a character in a story. Make decisions that are consistent with your personality and goals."
            )
            
            # Find the best matching option
            response_lower = response.lower().strip()
            for option in options:
                if option.lower() in response_lower:
                    return option
            
            # If no exact match, return the first option as fallback
            return options[0] if options else ""
    
    async def reflect_on_experience(self, experience: Dict[str, Any]):
        """Reflect on a recent experience and potentially update goals/personality"""
        async with self.llm_client as client:
            reflection_prompt = f"""
You are {self.agent_id}. You just experienced the following:
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
                f"You are {self.agent_id}, reflecting on your experiences and growth."
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
                
                # Add reflection to memory
                await self.add_memory({
                    "type": "reflection",
                    "experience": experience,
                    "reflection": reflection.get("reflection", ""),
                    "changes_made": {
                        "personality": reflection.get("personality_changes", {}),
                        "new_goals": reflection.get("new_goals", []),
                        "completed_goals": reflection.get("completed_goals", [])
                    }
                })
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state of the agent"""
        return {
            "agent_id": self.agent_id,
            "personality": self.personality,
            "goals": self.goals,
            "memory_count": len(self.memory),
            "relationships": {k: {
                "trust": v["trust"],
                "affection": v["affection"], 
                "respect": v["respect"]
            } for k, v in self.relationships.items()},
            "created_at": self.created_at.isoformat(),
            "last_action": self.last_action
        }