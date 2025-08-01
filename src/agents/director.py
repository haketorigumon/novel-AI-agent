"""Director Agent - Orchestrates story flow and manages narrative structure"""

import json
from typing import Dict, List, Optional, Any

from .base_agent import BaseAgent
from ..utils.config import Config
from ..utils.llm_client import LLMClient

class DirectorAgent(BaseAgent):
    """Director agent that guides story development and maintains narrative coherence"""
    
    def __init__(self, config: Config, llm_client: LLMClient, world_simulation):
        super().__init__("director", config, llm_client)
        self.world_simulation = world_simulation
        self.story_outline = {}
        self.current_arc = "setup"
        self.plot_threads = []
        self.pacing_targets = {}
    
    async def _generate_personality(self):
        """Generate director personality focused on storytelling"""
        self.personality = {
            "creativity": 0.8,
            "structure_focus": 0.9,
            "character_development": 0.85,
            "plot_complexity": 0.7,
            "pacing_awareness": 0.9,
            "theme_consistency": 0.8,
            "dramatic_tension": 0.75
        }
    
    async def _generate_initial_goals(self):
        """Generate initial storytelling goals"""
        self.goals = [
            "Create compelling character arcs",
            "Maintain narrative pacing",
            "Develop central themes",
            "Build dramatic tension",
            "Ensure plot coherence",
            "Guide story to satisfying conclusion"
        ]
        
        # Generate initial story outline
        await self._create_story_outline()
    
    async def _create_story_outline(self):
        """Create high-level story outline"""
        async with self.llm_client as client:
            outline_prompt = f"""
Create a high-level outline for a {self.config.story.target_length:,} word novel.
The story should be engaging, complex, and suitable for long-form development.

Structure the outline with:
1. Main theme and genre
2. Central conflict
3. Major story arcs (setup, rising action, climax, resolution)
4. Key plot points and turning points
5. Character development milestones
6. Pacing guidelines

Respond with a detailed JSON structure:
{{
    "theme": "main theme",
    "genre": "primary genre",
    "central_conflict": "main conflict description",
    "story_arcs": {{
        "setup": {{"description": "", "target_word_count": 0, "key_events": []}},
        "rising_action": {{"description": "", "target_word_count": 0, "key_events": []}},
        "climax": {{"description": "", "target_word_count": 0, "key_events": []}},
        "resolution": {{"description": "", "target_word_count": 0, "key_events": []}}
    }},
    "plot_threads": [
        {{"name": "", "description": "", "priority": 0}}
    ],
    "character_arcs": [
        {{"character_type": "", "arc_description": "", "key_moments": []}}
    ]
}}
"""
            
            self.story_outline = await client.generate_structured(
                outline_prompt,
                {
                    "theme": "string",
                    "genre": "string", 
                    "central_conflict": "string",
                    "story_arcs": "object",
                    "plot_threads": "array",
                    "character_arcs": "array"
                },
                "You are a master storyteller creating an epic novel outline."
            )
            
            if self.story_outline:
                self.plot_threads = self.story_outline.get("plot_threads", [])
                await self.add_memory({
                    "type": "story_outline_created",
                    "outline": self.story_outline
                })
    
    async def plan_next_scene(self, story_state: Dict, world_context: Dict) -> Dict[str, Any]:
        """Plan the next scene based on current story state and world context"""
        current_word_count = story_state.get("current_word_count", 0)
        current_chapter = story_state.get("current_chapter", 1)
        
        # Determine current story arc
        await self._update_current_arc(current_word_count)
        
        async with self.llm_client as client:
            scene_prompt = f"""
As the director of this story, plan the next scene.

Story Context:
- Current word count: {current_word_count:,}
- Current chapter: {current_chapter}
- Current story arc: {self.current_arc}
- Story outline: {json.dumps(self.story_outline, indent=2)}
- Active plot threads: {json.dumps(self.plot_threads, indent=2)}

World Context: {json.dumps(world_context, indent=2)}

Recent story memories: {json.dumps(await self.get_recent_memories(3), indent=2)}

Plan the next scene with:
1. Scene purpose and goals
2. Key events that should happen
3. Character focus and development opportunities
4. Pacing and tension level
5. World elements to incorporate
6. Plot threads to advance

Respond with JSON:
{{
    "scene_purpose": "what this scene accomplishes",
    "key_events": ["event1", "event2"],
    "character_focus": ["character_type1", "character_type2"],
    "tension_level": 0.7,
    "pacing": "fast/medium/slow",
    "world_elements": ["element1", "element2"],
    "plot_threads_to_advance": ["thread1"],
    "emotional_tone": "tone description",
    "scene_length_target": 800
}}
"""
            
            scene_plan = await client.generate_structured(
                scene_prompt,
                {
                    "scene_purpose": "string",
                    "key_events": "array",
                    "character_focus": "array", 
                    "tension_level": "number",
                    "pacing": "string",
                    "world_elements": "array",
                    "plot_threads_to_advance": "array",
                    "emotional_tone": "string",
                    "scene_length_target": "number"
                },
                "You are a master director orchestrating an epic story."
            )
            
            if scene_plan:
                await self.add_memory({
                    "type": "scene_planned",
                    "scene_plan": scene_plan,
                    "story_context": {
                        "word_count": current_word_count,
                        "chapter": current_chapter,
                        "arc": self.current_arc
                    }
                })
                
                self.last_action = f"Planned scene: {scene_plan.get('scene_purpose', 'Unknown')}"
            
            return scene_plan or {}
    
    async def _update_current_arc(self, word_count: int):
        """Update current story arc based on word count"""
        if not self.story_outline or "story_arcs" not in self.story_outline:
            return
        
        arcs = self.story_outline["story_arcs"]
        total_target = self.config.story.target_length
        
        # Calculate arc boundaries (rough estimates)
        setup_end = total_target * 0.25
        rising_end = total_target * 0.75
        climax_end = total_target * 0.9
        
        if word_count < setup_end:
            new_arc = "setup"
        elif word_count < rising_end:
            new_arc = "rising_action"
        elif word_count < climax_end:
            new_arc = "climax"
        else:
            new_arc = "resolution"
        
        if new_arc != self.current_arc:
            old_arc = self.current_arc
            self.current_arc = new_arc
            await self.add_memory({
                "type": "arc_transition",
                "from_arc": old_arc,
                "to_arc": new_arc,
                "word_count": word_count
            })
    
    async def evaluate_story_quality(self, recent_content: str, story_state: Dict) -> Dict[str, Any]:
        """Evaluate the quality of recent story content"""
        async with self.llm_client as client:
            evaluation_prompt = f"""
As a story director, evaluate this recent story content:

Content: {recent_content}

Story Context:
- Current arc: {self.current_arc}
- Word count: {story_state.get('current_word_count', 0):,}
- Chapter: {story_state.get('current_chapter', 1)}

Evaluate on these criteria (0.0 to 1.0):
1. Character development quality
2. Plot advancement effectiveness
3. Pacing appropriateness
4. Dialogue quality
5. Descriptive writing quality
6. Emotional engagement
7. Consistency with story outline

Also provide:
- Strengths of this content
- Areas for improvement
- Suggestions for future scenes

Respond with JSON:
{{
    "scores": {{
        "character_development": 0.8,
        "plot_advancement": 0.7,
        "pacing": 0.6,
        "dialogue": 0.9,
        "description": 0.8,
        "emotional_engagement": 0.7,
        "consistency": 0.9
    }},
    "overall_score": 0.77,
    "strengths": ["strength1", "strength2"],
    "improvements": ["improvement1", "improvement2"],
    "suggestions": ["suggestion1", "suggestion2"]
}}
"""
            
            evaluation = await client.generate_structured(
                evaluation_prompt,
                {
                    "scores": "object",
                    "overall_score": "number",
                    "strengths": "array",
                    "improvements": "array", 
                    "suggestions": "array"
                },
                "You are an expert story editor providing constructive feedback."
            )
            
            if evaluation:
                await self.add_memory({
                    "type": "story_evaluation",
                    "evaluation": evaluation,
                    "content_evaluated": recent_content[:200] + "..."
                })
            
            return evaluation or {}
    
    async def adjust_story_direction(self, evaluation: Dict[str, Any], story_state: Dict):
        """Adjust story direction based on quality evaluation"""
        if not evaluation or "overall_score" not in evaluation:
            return
        
        overall_score = evaluation["overall_score"]
        
        # If quality is low, adjust approach
        if overall_score < 0.6:
            improvements = evaluation.get("improvements", [])
            suggestions = evaluation.get("suggestions", [])
            
            # Update goals based on feedback
            for improvement in improvements:
                if "pacing" in improvement.lower():
                    if "slow pacing" not in self.goals:
                        self.goals.append("improve story pacing")
                elif "character" in improvement.lower():
                    if "enhance character development" not in self.goals:
                        self.goals.append("enhance character development")
                elif "dialogue" in improvement.lower():
                    if "improve dialogue quality" not in self.goals:
                        self.goals.append("improve dialogue quality")
            
            await self.add_memory({
                "type": "direction_adjustment",
                "reason": f"Low quality score: {overall_score}",
                "improvements_needed": improvements,
                "new_goals_added": [g for g in self.goals if "improve" in g or "enhance" in g]
            })
    
    def get_story_outline(self) -> Dict[str, Any]:
        """Get the current story outline"""
        return self.story_outline
    
    def get_current_arc(self) -> str:
        """Get the current story arc"""
        return self.current_arc
    
    def get_plot_threads(self) -> List[Dict[str, Any]]:
        """Get active plot threads"""
        return self.plot_threads