"""Main Novel AI Agent - Orchestrates story generation and self-evolution"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional
from rich.console import Console
from rich.progress import Progress, TaskID
from datetime import datetime

from ..utils.config import Config
from ..utils.llm_client import LLMClient
from ..agents.director import DirectorAgent
from ..agents.character import CharacterAgent
from ..simulation.world import WorldSimulation
from ..evolution.code_evolver import CodeEvolver

console = Console()

class NovelAIAgent:
    """Main orchestrator for the novel generation system"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm_client = None
        self.director = None
        self.characters = []
        self.world_simulation = None
        self.code_evolver = None
        self.story_state = {
            "current_word_count": 0,
            "current_chapter": 1,
            "story_content": [],
            "character_states": {},
            "world_state": {},
            "generation_metadata": {}
        }
        
    async def initialize(self):
        """Initialize all components of the novel agent"""
        console.print("[blue]Initializing Novel AI Agent...[/blue]")
        
        # Initialize LLM client
        api_key = self.config.get_api_key_for_provider(self.config.llm.provider)
        self.llm_client = LLMClient(
            provider=self.config.llm.provider,
            model=self.config.llm.model,
            base_url=self.config.llm.base_url,
            api_key=api_key,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens
        )
        
        # Check LLM connection
        async with self.llm_client as client:
            if not await client.check_connection():
                console.print("[red]Warning: Cannot connect to LLM service. Please ensure Ollama is running.[/red]")
        
        # Initialize world simulation
        self.world_simulation = WorldSimulation(self.config, self.llm_client)
        await self.world_simulation.initialize()
        
        # Initialize director agent
        if self.config.agents.director_enabled:
            self.director = DirectorAgent(self.config, self.llm_client, self.world_simulation)
            await self.director.initialize()
        
        # Initialize character agents
        await self._initialize_characters()
        
        # Initialize code evolver
        if self.config.evolution.enabled:
            self.code_evolver = CodeEvolver(self.config, self.llm_client)
            await self.code_evolver.initialize()
        
        console.print("[green]Novel AI Agent initialized successfully![/green]")
    
    async def _initialize_characters(self):
        """Initialize character agents based on configuration"""
        console.print("[blue]Creating character agents...[/blue]")
        
        # Create initial characters based on character types
        for i, char_type in enumerate(self.config.agents.character_types):
            if len(self.characters) >= self.config.agents.max_agents:
                break
                
            character = CharacterAgent(
                agent_id=f"{char_type}_{i}",
                character_type=char_type,
                config=self.config,
                llm_client=self.llm_client,
                world_simulation=self.world_simulation
            )
            await character.initialize()
            self.characters.append(character)
        
        console.print(f"[green]Created {len(self.characters)} character agents[/green]")
    
    async def generate_novel(self):
        """Main novel generation loop"""
        await self.initialize()
        
        console.print(f"[bold blue]Starting novel generation (target: {self.config.story.target_length:,} words)[/bold blue]")
        
        with Progress() as progress:
            task = progress.add_task(
                "[green]Generating novel...", 
                total=self.config.story.target_length
            )
            
            while self.story_state["current_word_count"] < self.config.story.target_length:
                # Generate next story segment
                segment = await self._generate_story_segment()
                
                if segment:
                    # Add to story
                    self.story_state["story_content"].append(segment)
                    word_count = len(segment.split())
                    self.story_state["current_word_count"] += word_count
                    
                    # Update progress
                    progress.update(task, completed=self.story_state["current_word_count"])
                    
                    # Save periodically
                    if self.story_state["current_word_count"] % self.config.story.save_interval == 0:
                        await self._save_progress()
                    
                    # Check for chapter completion
                    if word_count >= self.config.story.chapter_length:
                        self.story_state["current_chapter"] += 1
                        console.print(f"[cyan]Completed Chapter {self.story_state['current_chapter'] - 1}[/cyan]")
                    
                    # Trigger evolution if enabled
                    if (self.config.evolution.enabled and 
                        self.story_state["current_word_count"] % self.config.evolution.evaluation_interval == 0):
                        await self._trigger_evolution()
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)
        
        # Final save
        await self._save_final_novel()
        console.print("[bold green]Novel generation completed![/bold green]")
    
    async def _generate_story_segment(self) -> Optional[str]:
        """Generate the next segment of the story"""
        try:
            # Get current context from world simulation
            world_context = await self.world_simulation.get_current_context()
            
            # Let director plan the next scene if enabled
            scene_plan = None
            if self.director:
                scene_plan = await self.director.plan_next_scene(
                    self.story_state, world_context
                )
            
            # Have characters contribute to the story
            character_contributions = []
            for character in self.characters:
                contribution = await character.contribute_to_story(
                    self.story_state, world_context, scene_plan
                )
                if contribution:
                    character_contributions.append(contribution)
            
            # Synthesize contributions into coherent narrative
            if character_contributions:
                segment = await self._synthesize_narrative(
                    character_contributions, scene_plan, world_context
                )
                
                # Update world state based on story events
                await self.world_simulation.update_from_story_events(segment)
                
                return segment
            
        except Exception as e:
            console.print(f"[red]Error generating story segment: {e}[/red]")
        
        return None
    
    async def _synthesize_narrative(self, contributions: List[str], scene_plan: Optional[Dict], world_context: Dict) -> str:
        """Synthesize character contributions into coherent narrative"""
        async with self.llm_client as client:
            synthesis_prompt = f"""
You are a master storyteller synthesizing multiple character perspectives into a coherent narrative.

Current story context:
- Chapter: {self.story_state['current_chapter']}
- Word count: {self.story_state['current_word_count']:,}
- World context: {json.dumps(world_context, indent=2)}

Scene plan: {json.dumps(scene_plan, indent=2) if scene_plan else 'None'}

Character contributions:
{chr(10).join(f"- {contrib}" for contrib in contributions)}

Synthesize these contributions into a compelling narrative segment of approximately 500-1000 words.
Focus on:
1. Maintaining narrative flow and consistency
2. Developing character arcs
3. Advancing the plot
4. Rich, immersive descriptions
5. Engaging dialogue and action

Narrative segment:
"""
            
            return await client.generate(
                synthesis_prompt,
                "You are an expert novelist creating engaging, high-quality fiction."
            )
    
    async def _save_progress(self):
        """Save current progress to disk"""
        output_dir = Path(self.config.story.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save story state
        state_file = output_dir / "story_state.json"
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(self.story_state, f, indent=2, ensure_ascii=False)
        
        # Save current story content
        story_file = output_dir / f"novel_progress_{self.story_state['current_word_count']}.txt"
        with open(story_file, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(self.story_state['story_content']))
        
        console.print(f"[dim]Progress saved ({self.story_state['current_word_count']:,} words)[/dim]")
    
    async def _save_final_novel(self):
        """Save the completed novel"""
        output_dir = Path(self.config.story.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_file = output_dir / f"novel_complete_{timestamp}.txt"
        
        with open(final_file, 'w', encoding='utf-8') as f:
            f.write("# Generated Novel\n\n")
            f.write(f"Generated by Novel AI Agent\n")
            f.write(f"Completion date: {datetime.now().isoformat()}\n")
            f.write(f"Total words: {self.story_state['current_word_count']:,}\n")
            f.write(f"Total chapters: {self.story_state['current_chapter']}\n\n")
            f.write("---\n\n")
            f.write('\n\n'.join(self.story_state['story_content']))
        
        console.print(f"[bold green]Final novel saved to: {final_file}[/bold green]")
    
    async def _trigger_evolution(self):
        """Trigger code evolution process"""
        if self.code_evolver:
            console.print("[yellow]Triggering code evolution...[/yellow]")
            try:
                await self.code_evolver.evolve_system(self.story_state)
                console.print("[green]Code evolution completed[/green]")
            except Exception as e:
                console.print(f"[red]Evolution error: {e}[/red]")
    
    async def evolve_system(self, generations: int = 1):
        """Manually trigger system evolution"""
        await self.initialize()
        
        if not self.code_evolver:
            console.print("[red]Code evolution is disabled in configuration[/red]")
            return
        
        for gen in range(generations):
            console.print(f"[blue]Evolution generation {gen + 1}/{generations}[/blue]")
            await self.code_evolver.evolve_system(self.story_state)
            await asyncio.sleep(1)  # Brief pause between generations
        
        console.print("[green]Manual evolution completed[/green]")
    
    def get_status(self) -> Dict:
        """Get current status of the novel generation"""
        return {
            "word_count": self.story_state["current_word_count"],
            "target_length": self.config.story.target_length,
            "progress_percentage": (self.story_state["current_word_count"] / self.config.story.target_length) * 100,
            "current_chapter": self.story_state["current_chapter"],
            "characters_count": len(self.characters),
            "evolution_enabled": self.config.evolution.enabled
        }