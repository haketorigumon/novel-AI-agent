"""World Simulation - Dynamic environment for story generation"""

import json
import random
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from ..utils.config import Config
from ..utils.llm_client import LLMClient

class WorldSimulation:
    """Manages the dynamic world state and environmental changes"""
    
    def __init__(self, config: Config, llm_client: LLMClient):
        self.config = config
        self.llm_client = llm_client
        self.world_state = {}
        self.locations = []
        self.events_history = []
        self.active_events = []
        self.world_rules = {}
        self.time_progression = {
            "current_time": "morning",
            "current_day": 1,
            "season": "spring",
            "weather": "clear"
        }
    
    async def initialize(self):
        """Initialize the world simulation"""
        await self._create_world_foundation()
        await self._generate_initial_locations()
        await self._establish_world_rules()
    
    async def _create_world_foundation(self):
        """Create the basic world foundation"""
        async with self.llm_client as client:
            world_prompt = f"""
Create a rich, dynamic world suitable for a {self.config.story.target_length:,} word novel.
The world should be complex enough to support long-form storytelling with multiple locations, cultures, and systems.

Create a world with:
1. Basic world type (fantasy, sci-fi, modern, historical, etc.)
2. Core world rules and physics
3. Major geographical features
4. Political/social systems
5. Economic systems
6. Magic/technology level
7. Major conflicts or tensions
8. Cultural elements

Respond with JSON:
{{
    "world_type": "fantasy/sci-fi/modern/historical",
    "name": "world name",
    "description": "brief world description",
    "core_rules": ["rule1", "rule2"],
    "geography": {{
        "continents": ["continent1", "continent2"],
        "major_features": ["mountain range", "ocean", "desert"]
    }},
    "political_system": "description of governance",
    "technology_level": "description of tech/magic",
    "major_conflicts": ["conflict1", "conflict2"],
    "cultures": [
        {{"name": "culture1", "description": "brief description"}},
        {{"name": "culture2", "description": "brief description"}}
    ]
}}
"""
            
            world_foundation = await client.generate_structured(
                world_prompt,
                {
                    "world_type": "string",
                    "name": "string",
                    "description": "string",
                    "core_rules": "array",
                    "geography": "object",
                    "political_system": "string",
                    "technology_level": "string",
                    "major_conflicts": "array",
                    "cultures": "array"
                },
                "You are a master world-builder creating an epic fantasy world."
            )
            
            if world_foundation:
                self.world_state.update(world_foundation)
    
    async def _generate_initial_locations(self):
        """Generate initial locations for the world"""
        async with self.llm_client as client:
            locations_prompt = f"""
Based on this world foundation:
{json.dumps(self.world_state, indent=2)}

Create 8-12 diverse locations that could serve as settings for story scenes.
Each location should be:
1. Visually distinct and memorable
2. Suitable for different types of scenes (action, dialogue, mystery, etc.)
3. Connected to the world's geography and culture
4. Have potential for interesting events

For each location, provide:
- Name
- Type (city, wilderness, building, etc.)
- Description
- Notable features
- Potential story uses
- Connected locations

Respond with JSON array:
[
    {{
        "name": "location name",
        "type": "city/wilderness/building/etc",
        "description": "detailed description",
        "notable_features": ["feature1", "feature2"],
        "story_potential": ["potential use1", "potential use2"],
        "connections": ["connected_location1"],
        "atmosphere": "mood/feeling of the place"
    }}
]
"""
            
            response = await client.generate(locations_prompt,
                "You are creating diverse, interesting locations for an epic story.")
            
            try:
                # Extract JSON array from response
                start_idx = response.find('[')
                end_idx = response.rfind(']') + 1
                if start_idx != -1 and end_idx != 0:
                    locations_json = response[start_idx:end_idx]
                    self.locations = json.loads(locations_json)
            except:
                # Fallback locations
                self.locations = self._get_default_locations()
    
    def _get_default_locations(self) -> List[Dict[str, Any]]:
        """Get default locations if generation fails"""
        return [
            {
                "name": "The Capital City",
                "type": "city",
                "description": "A bustling metropolis with towering spires",
                "notable_features": ["royal palace", "market district", "city walls"],
                "story_potential": ["political intrigue", "urban adventures"],
                "connections": ["The Royal Road"],
                "atmosphere": "busy and diverse"
            },
            {
                "name": "The Ancient Forest",
                "type": "wilderness",
                "description": "A mysterious woodland with ancient trees",
                "notable_features": ["hidden paths", "ancient ruins", "mystical creatures"],
                "story_potential": ["mystery", "supernatural encounters"],
                "connections": ["Forest Edge Village"],
                "atmosphere": "mysterious and primordial"
            }
        ]
    
    async def _establish_world_rules(self):
        """Establish rules for how the world operates"""
        self.world_rules = {
            "time_progression": True,
            "weather_changes": True,
            "seasonal_effects": True,
            "event_frequency": self.config.simulation.event_frequency,
            "environment_changes": self.config.simulation.environment_changes,
            "cause_and_effect": True
        }
    
    async def get_current_context(self) -> Dict[str, Any]:
        """Get current world context for story generation"""
        return {
            "world_state": self.world_state,
            "current_location": self._get_current_primary_location(),
            "available_locations": [loc["name"] for loc in self.locations],
            "time_progression": self.time_progression,
            "active_events": self.active_events,
            "recent_events": self.events_history[-3:] if self.events_history else [],
            "world_mood": await self._assess_world_mood()
        }
    
    def _get_current_primary_location(self) -> Optional[Dict[str, Any]]:
        """Get the primary location for the current scene"""
        if self.locations:
            # Simple logic: rotate through locations or pick based on recent events
            if self.events_history:
                last_event = self.events_history[-1]
                event_location = last_event.get("location")
                for loc in self.locations:
                    if loc["name"] == event_location:
                        return loc
            
            # Default to first location or random selection
            return random.choice(self.locations)
        return None
    
    async def _assess_world_mood(self) -> str:
        """Assess the current mood/atmosphere of the world"""
        if not self.events_history:
            return "neutral"
        
        recent_events = self.events_history[-5:]
        
        # Simple sentiment analysis of recent events
        positive_keywords = ["celebration", "victory", "peace", "prosperity", "joy"]
        negative_keywords = ["war", "disaster", "death", "conflict", "crisis"]
        
        positive_count = 0
        negative_count = 0
        
        for event in recent_events:
            event_text = json.dumps(event).lower()
            positive_count += sum(1 for word in positive_keywords if word in event_text)
            negative_count += sum(1 for word in negative_keywords if word in event_text)
        
        if positive_count > negative_count:
            return "hopeful"
        elif negative_count > positive_count:
            return "tense"
        else:
            return "neutral"
    
    async def update_from_story_events(self, story_content: str):
        """Update world state based on story events"""
        if not story_content:
            return
        
        # Advance time
        await self._advance_time()
        
        # Generate environmental events
        if random.random() < self.config.simulation.event_frequency:
            await self._generate_environmental_event(story_content)
        
        # Update weather
        if random.random() < 0.3:  # 30% chance of weather change
            await self._update_weather()
        
        # Process story impact on world
        await self._process_story_impact(story_content)
    
    async def _advance_time(self):
        """Advance the world's time progression"""
        time_periods = ["morning", "afternoon", "evening", "night"]
        current_index = time_periods.index(self.time_progression["current_time"])
        
        # Advance to next time period
        next_index = (current_index + 1) % len(time_periods)
        self.time_progression["current_time"] = time_periods[next_index]
        
        # If we've completed a full day cycle
        if next_index == 0:
            self.time_progression["current_day"] += 1
            
            # Advance season occasionally
            if self.time_progression["current_day"] % 30 == 0:
                seasons = ["spring", "summer", "autumn", "winter"]
                current_season_index = seasons.index(self.time_progression["season"])
                self.time_progression["season"] = seasons[(current_season_index + 1) % len(seasons)]
    
    async def _update_weather(self):
        """Update weather conditions"""
        weather_options = ["clear", "cloudy", "rainy", "stormy", "foggy", "windy"]
        
        # Weather influenced by season
        season = self.time_progression["season"]
        if season == "winter":
            weather_options.extend(["snowy", "cold", "icy"])
        elif season == "summer":
            weather_options.extend(["hot", "humid", "sunny"])
        
        self.time_progression["weather"] = random.choice(weather_options)
    
    async def _generate_environmental_event(self, story_context: str):
        """Generate a random environmental event"""
        async with self.llm_client as client:
            event_prompt = f"""
Generate a minor environmental event for this world:
World: {json.dumps(self.world_state, indent=2)}
Current time: {self.time_progression}
Recent story context: {story_context[:300]}...

Create a small environmental event that:
1. Fits the world and current conditions
2. Could influence the story atmosphere
3. Is not too disruptive to ongoing narrative
4. Adds flavor and immersion

Examples: weather changes, animal sightings, distant sounds, natural phenomena, etc.

Respond with JSON:
{{
    "event_type": "environmental",
    "name": "brief event name",
    "description": "what happens",
    "location": "where it occurs",
    "impact": "how it might affect the story",
    "duration": "how long it lasts"
}}
"""
            
            event = await client.generate_structured(
                event_prompt,
                {
                    "event_type": "string",
                    "name": "string",
                    "description": "string",
                    "location": "string",
                    "impact": "string",
                    "duration": "string"
                },
                "You are creating atmospheric environmental events for a story world."
            )
            
            if event:
                event["timestamp"] = datetime.now().isoformat()
                self.events_history.append(event)
                self.active_events.append(event)
                
                # Remove old events
                if len(self.events_history) > 50:
                    self.events_history = self.events_history[-25:]
    
    async def _process_story_impact(self, story_content: str):
        """Process how story events impact the world"""
        story_lower = story_content.lower()
        
        # Simple impact detection
        impacts = []
        
        if any(word in story_lower for word in ["battle", "fight", "war", "attack"]):
            impacts.append("conflict_occurred")
        
        if any(word in story_lower for word in ["magic", "spell", "enchant", "mystical"]):
            impacts.append("magical_activity")
        
        if any(word in story_lower for word in ["travel", "journey", "move", "arrive"]):
            impacts.append("movement_occurred")
        
        if any(word in story_lower for word in ["discovery", "found", "revealed", "uncovered"]):
            impacts.append("discovery_made")
        
        # Record impacts
        if impacts:
            impact_event = {
                "event_type": "story_impact",
                "impacts": impacts,
                "timestamp": datetime.now().isoformat(),
                "story_excerpt": story_content[:200] + "..."
            }
            self.events_history.append(impact_event)
    
    async def get_location_details(self, location_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific location"""
        for location in self.locations:
            if location["name"].lower() == location_name.lower():
                return location
        return None
    
    async def add_new_location(self, location_data: Dict[str, Any]):
        """Add a new location to the world"""
        self.locations.append(location_data)
    
    def get_world_summary(self) -> Dict[str, Any]:
        """Get a summary of the current world state"""
        return {
            "world_name": self.world_state.get("name", "Unknown World"),
            "world_type": self.world_state.get("world_type", "Unknown"),
            "locations_count": len(self.locations),
            "events_count": len(self.events_history),
            "active_events_count": len(self.active_events),
            "current_time": self.time_progression,
            "world_mood": "neutral"  # Would call _assess_world_mood() but it's async
        }