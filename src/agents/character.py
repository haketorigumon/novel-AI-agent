"""Character Agent - Represents individual characters in the story"""

import json
import random
from typing import Dict, List, Optional, Any

from .base_agent import BaseAgent
from ..utils.config import Config
from ..utils.llm_client import LLMClient

class CharacterAgent(BaseAgent):
    """Agent representing a character in the story"""
    
    def __init__(self, agent_id: str, character_type: str, config: Config, llm_client: LLMClient, world_simulation):
        super().__init__(agent_id, config, llm_client)
        self.character_type = character_type
        self.world_simulation = world_simulation
        self.character_profile = {}
        self.current_location = None
        self.current_state = "neutral"
        self.dialogue_style = {}
        self.backstory = ""
        self.character_arc = {}
    
    async def _generate_personality(self):
        """Generate personality traits based on character type"""
        async with self.llm_client as client:
            personality_prompt = f"""
Create a detailed personality for a {self.character_type} character in a novel.

Generate personality traits (0.0 to 1.0 scale):
- extroversion: how outgoing and social
- agreeableness: how cooperative and trusting
- conscientiousness: how organized and responsible
- neuroticism: how emotionally unstable
- openness: how open to new experiences
- courage: how brave in face of danger
- intelligence: intellectual capability
- empathy: ability to understand others
- ambition: drive to achieve goals
- loyalty: faithfulness to others

Also create:
- Core values (3-5 values)
- Fears and phobias
- Strengths and weaknesses
- Quirks and mannerisms

Respond with JSON:
{{
    "traits": {{
        "extroversion": 0.7,
        "agreeableness": 0.6,
        "conscientiousness": 0.8,
        "neuroticism": 0.3,
        "openness": 0.9,
        "courage": 0.8,
        "intelligence": 0.7,
        "empathy": 0.6,
        "ambition": 0.9,
        "loyalty": 0.8
    }},
    "core_values": ["justice", "family", "knowledge"],
    "fears": ["heights", "betrayal"],
    "strengths": ["quick thinking", "leadership"],
    "weaknesses": ["impatience", "pride"],
    "quirks": ["taps fingers when thinking", "always carries a book"]
}}
"""
            
            personality_data = await client.generate_structured(
                personality_prompt,
                {
                    "traits": "object",
                    "core_values": "array",
                    "fears": "array",
                    "strengths": "array",
                    "weaknesses": "array",
                    "quirks": "array"
                },
                f"You are creating a {self.character_type} character for an epic novel."
            )
            
            if personality_data:
                self.personality = personality_data.get("traits", {})
                self.character_profile.update({
                    "core_values": personality_data.get("core_values", []),
                    "fears": personality_data.get("fears", []),
                    "strengths": personality_data.get("strengths", []),
                    "weaknesses": personality_data.get("weaknesses", []),
                    "quirks": personality_data.get("quirks", [])
                })
    
    async def _generate_initial_goals(self):
        """Generate character-specific goals"""
        async with self.llm_client as client:
            goals_prompt = f"""
Create initial goals for a {self.character_type} character with this personality:
{json.dumps(self.personality, indent=2)}
Character profile: {json.dumps(self.character_profile, indent=2)}

Create 3-5 goals that are:
1. Specific to this character type and personality
2. Suitable for a long novel (can evolve over time)
3. Create potential for conflict and growth
4. Mix of short-term and long-term objectives

Examples for different character types:
- Protagonist: "Find the truth about my past", "Protect my loved ones"
- Antagonist: "Gain ultimate power", "Destroy my enemies"
- Supporting: "Help the hero succeed", "Find my own purpose"

Respond with a JSON array of goal strings:
["goal1", "goal2", "goal3"]
"""
            
            response = await client.generate(goals_prompt, 
                f"You are creating goals for a {self.character_type} character.")
            
            try:
                # Try to extract JSON array from response
                start_idx = response.find('[')
                end_idx = response.rfind(']') + 1
                if start_idx != -1 and end_idx != 0:
                    goals_json = response[start_idx:end_idx]
                    self.goals = json.loads(goals_json)
            except:
                # Fallback goals based on character type
                self.goals = self._get_default_goals()
    
    def _get_default_goals(self) -> List[str]:
        """Get default goals based on character type"""
        defaults = {
            "protagonist": ["Overcome the central challenge", "Grow as a person", "Protect others"],
            "antagonist": ["Achieve ultimate goal", "Defeat the protagonist", "Gain power"],
            "supporting": ["Help the protagonist", "Find personal fulfillment", "Survive the story"],
            "narrator": ["Tell the story effectively", "Reveal truth gradually", "Guide the reader"]
        }
        return defaults.get(self.character_type, ["Survive", "Find purpose", "Make a difference"])
    
    async def generate_backstory(self):
        """Generate detailed backstory for the character"""
        async with self.llm_client as client:
            backstory_prompt = f"""
Create a compelling backstory for this {self.character_type} character:

Personality: {json.dumps(self.personality, indent=2)}
Profile: {json.dumps(self.character_profile, indent=2)}
Goals: {json.dumps(self.goals, indent=2)}

Create a backstory that:
1. Explains how they developed their personality traits
2. Provides motivation for their current goals
3. Includes formative experiences and relationships
4. Sets up potential character growth
5. Creates hooks for story development

Write 2-3 paragraphs of rich backstory.
"""
            
            self.backstory = await client.generate(backstory_prompt,
                f"You are creating a rich backstory for a {self.character_type} character.")
            
            await self.add_memory({
                "type": "backstory_created",
                "backstory": self.backstory
            })
    
    async def contribute_to_story(self, story_state: Dict, world_context: Dict, scene_plan: Optional[Dict]) -> Optional[str]:
        """Contribute to the current story segment"""
        # Check if this character should be active in this scene
        if scene_plan and "character_focus" in scene_plan:
            if self.character_type not in scene_plan["character_focus"]:
                # Not focused in this scene, small chance of minor contribution
                if random.random() > 0.2:
                    return None
        
        # Determine character's current emotional state and motivation
        current_motivation = await self._assess_current_motivation(story_state, world_context, scene_plan)
        
        if not current_motivation:
            return None
        
        # Generate character contribution
        contribution = await self._generate_contribution(story_state, world_context, scene_plan, current_motivation)
        
        if contribution:
            # Update character state based on contribution
            await self._update_character_state(contribution, world_context)
            
            self.last_action = f"Contributed to story: {contribution[:50]}..."
        
        return contribution
    
    async def _assess_current_motivation(self, story_state: Dict, world_context: Dict, scene_plan: Optional[Dict]) -> Optional[Dict]:
        """Assess what motivates the character in the current situation"""
        async with self.llm_client as client:
            motivation_prompt = f"""
You are {self.agent_id}, a {self.character_type} character with:
Personality: {json.dumps(self.personality, indent=2)}
Goals: {json.dumps(self.goals, indent=2)}
Recent memories: {json.dumps(await self.get_recent_memories(3), indent=2)}

Current situation:
Story state: {json.dumps(story_state, indent=2)}
World context: {json.dumps(world_context, indent=2)}
Scene plan: {json.dumps(scene_plan, indent=2) if scene_plan else 'None'}

What is your primary motivation in this situation? What do you want to achieve or avoid?
How do you feel about the current circumstances?

Respond with JSON:
{{
    "primary_motivation": "what you want most right now",
    "emotional_state": "current emotional state",
    "urgency": 0.7,
    "confidence": 0.6,
    "approach": "how you plan to act",
    "concerns": ["concern1", "concern2"]
}}
"""
            
            motivation = await client.generate_structured(
                motivation_prompt,
                {
                    "primary_motivation": "string",
                    "emotional_state": "string",
                    "urgency": "number",
                    "confidence": "number", 
                    "approach": "string",
                    "concerns": "array"
                },
                f"You are {self.agent_id}, responding authentically to the current situation."
            )
            
            return motivation
    
    async def _generate_contribution(self, story_state: Dict, world_context: Dict, scene_plan: Optional[Dict], motivation: Dict) -> Optional[str]:
        """Generate this character's contribution to the story"""
        async with self.llm_client as client:
            contribution_prompt = f"""
You are {self.agent_id}, a {self.character_type} character contributing to the story.

Your character:
- Personality: {json.dumps(self.personality, indent=2)}
- Backstory: {self.backstory}
- Current goals: {json.dumps(self.goals, indent=2)}
- Current motivation: {json.dumps(motivation, indent=2)}

Current story context:
- Word count: {story_state.get('current_word_count', 0):,}
- Chapter: {story_state.get('current_chapter', 1)}
- World context: {json.dumps(world_context, indent=2)}
- Scene plan: {json.dumps(scene_plan, indent=2) if scene_plan else 'None'}

Write your contribution to this scene (200-500 words). This could include:
- Your actions and reactions
- Dialogue (stay true to your character voice)
- Internal thoughts and feelings
- Interactions with the environment or other characters
- Advancing your personal goals

Write in third person, focusing on your character's perspective and actions.
Make it engaging and true to your personality.
"""
            
            contribution = await client.generate(contribution_prompt,
                f"You are {self.agent_id}, a {self.character_type} character in an epic story.")
            
            if contribution and len(contribution.strip()) > 50:
                await self.add_memory({
                    "type": "story_contribution",
                    "contribution": contribution,
                    "motivation": motivation,
                    "scene_context": scene_plan
                })
                return contribution
        
        return None
    
    async def _update_character_state(self, contribution: str, world_context: Dict):
        """Update character's internal state based on their contribution"""
        # Simple state updates based on contribution content
        contribution_lower = contribution.lower()
        
        if any(word in contribution_lower for word in ["angry", "furious", "rage"]):
            self.current_state = "angry"
        elif any(word in contribution_lower for word in ["sad", "grief", "sorrow"]):
            self.current_state = "sad"
        elif any(word in contribution_lower for word in ["happy", "joy", "excited"]):
            self.current_state = "happy"
        elif any(word in contribution_lower for word in ["afraid", "scared", "terrified"]):
            self.current_state = "fearful"
        else:
            self.current_state = "neutral"
        
        # Update location if mentioned
        for location in world_context.get("locations", []):
            if location.lower() in contribution_lower:
                self.current_location = location
                break
    
    async def interact_with_character(self, other_character: 'CharacterAgent', context: Dict) -> Optional[str]:
        """Generate interaction with another character"""
        async with self.llm_client as client:
            interaction_prompt = f"""
You are {self.agent_id} ({self.character_type}) interacting with {other_character.agent_id} ({other_character.character_type}).

Your personality: {json.dumps(self.personality, indent=2)}
Their personality: {json.dumps(other_character.personality, indent=2)}
Your relationship: {json.dumps(self.relationships.get(other_character.agent_id, {}), indent=2)}
Context: {json.dumps(context, indent=2)}

Generate a brief interaction (dialogue and actions) between your characters.
Focus on:
1. Staying true to both personalities
2. Advancing character relationships
3. Creating engaging dialogue
4. Moving the story forward

Write 100-300 words of interaction.
"""
            
            interaction = await client.generate(interaction_prompt,
                f"You are {self.agent_id} interacting with {other_character.agent_id}.")
            
            if interaction:
                # Update relationship based on interaction
                relationship_change = await self._analyze_interaction_impact(interaction, other_character)
                if relationship_change:
                    await self.update_relationship(other_character.agent_id, relationship_change)
                    await other_character.update_relationship(self.agent_id, relationship_change)
                
                await self.add_memory({
                    "type": "character_interaction",
                    "other_character": other_character.agent_id,
                    "interaction": interaction,
                    "relationship_change": relationship_change
                })
            
            return interaction
    
    async def _analyze_interaction_impact(self, interaction: str, other_character: 'CharacterAgent') -> Dict[str, float]:
        """Analyze how an interaction affects the relationship"""
        interaction_lower = interaction.lower()
        changes = {}
        
        # Simple sentiment analysis
        positive_words = ["help", "support", "friend", "trust", "love", "care", "protect"]
        negative_words = ["betray", "hurt", "enemy", "hate", "anger", "fight", "attack"]
        
        positive_count = sum(1 for word in positive_words if word in interaction_lower)
        negative_count = sum(1 for word in negative_words if word in interaction_lower)
        
        if positive_count > negative_count:
            changes = {"trust": 0.1, "affection": 0.05, "respect": 0.05}
        elif negative_count > positive_count:
            changes = {"trust": -0.1, "affection": -0.05, "respect": -0.05}
        
        return changes
    
    def get_character_summary(self) -> Dict[str, Any]:
        """Get a summary of this character"""
        return {
            "agent_id": self.agent_id,
            "character_type": self.character_type,
            "personality": self.personality,
            "goals": self.goals,
            "current_state": self.current_state,
            "current_location": self.current_location,
            "backstory_length": len(self.backstory),
            "memory_count": len(self.memory),
            "relationships_count": len(self.relationships)
        }