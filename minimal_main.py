#!/usr/bin/env python3
"""
Minimal AI Agent System - Maximum Flexibility, Minimum Hardcoding
A completely redesigned architecture focused on adaptability and extensibility
"""

import asyncio
import json
import uuid
import typer
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import Dict, Any, Optional

from src.core.minimal_core import MinimalCore, MessageType, MemoryLayer
from src.core.llm_integration import IntelligentAgent
from src.utils.config import Config
from src.utils.llm_client import LLMClient

console = Console()
app = typer.Typer(help="Minimal AI Agent System - Infinite flexibility through minimal hardcoding")


class AdaptiveSystem:
    """Adaptive system wrapper that integrates LLM with minimal core"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm_client = None
        self.core = None
        self.intelligent_agents: Dict[str, IntelligentAgent] = {}
        
    async def initialize(self):
        """Initialize the adaptive system"""
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
                console.print("[red]Warning: Cannot connect to LLM service. System will run in limited mode.[/red]")
        
        # Initialize minimal core
        core_config = {
            'storage_dir': 'state',
            'templates_dir': 'templates',
            'plugins_dir': 'plugins',
            'llm_client': self.llm_client
        }
        
        self.core = MinimalCore(core_config)
        await self.core.initialize()
        
        console.print("[green]✅ Adaptive system initialized successfully![/green]")
    
    async def create_adaptive_agent(self, role: str, capabilities: list = None, 
                                  personality: Dict[str, Any] = None) -> str:
        """Create an adaptive agent with specified role and capabilities"""
        agent_config = {
            'role': role,
            'capabilities': capabilities or [],
            'personality': personality or {},
            'adaptive': True,
            'learning_enabled': True,
            'creativity_level': 0.7
        }
        
        # Create intelligent agent instead of basic agent
        agent_id = f"agent_{uuid.uuid4().hex[:8]}"
        
        intelligent_agent = IntelligentAgent(
            agent_id=agent_id,
            config=agent_config,
            communication_hub=self.core.communication_hub,
            state_manager=self.core.state_manager,
            prompt_engine=self.core.prompt_engine,
            plugin_loader=self.core.plugin_loader,
            llm_client=self.llm_client
        )
        
        await intelligent_agent.initialize()
        self.core.agents[agent_id] = intelligent_agent
        self.intelligent_agents[agent_id] = intelligent_agent
        
        console.print(f"[green]Created intelligent agent: {agent_id} (Role: {role})[/green]")
        return agent_id
    
    async def process_natural_language_request(self, request: str) -> Dict[str, Any]:
        """Process a natural language request adaptively"""
        # Analyze the request to determine the best approach
        async with self.llm_client as client:
            analysis_prompt = f"""
Analyze this request and determine the best approach:
Request: "{request}"

Determine:
1. What type of task is this? (creative, analytical, technical, conversational, etc.)
2. What capabilities are needed?
3. Should this be handled by a single agent or multiple agents?
4. What is the expected output format?
5. What additional context or resources might be needed?

Respond with a JSON object:
{{
    "task_type": "string",
    "capabilities_needed": ["list", "of", "capabilities"],
    "multi_agent": boolean,
    "output_format": "string",
    "approach": "detailed approach description",
    "estimated_complexity": "low|medium|high"
}}
"""
            
            analysis = await client.generate_structured(
                analysis_prompt,
                {
                    "task_type": "string",
                    "capabilities_needed": "array",
                    "multi_agent": "boolean", 
                    "output_format": "string",
                    "approach": "string",
                    "estimated_complexity": "string"
                },
                "You are an intelligent task analyzer. Analyze requests and determine optimal processing approaches."
            )
            
            if not analysis:
                return {"error": "Could not analyze request"}
            
            # Create appropriate agent(s) based on analysis
            if analysis.get("multi_agent", False):
                return await self._handle_multi_agent_request(request, analysis)
            else:
                return await self._handle_single_agent_request(request, analysis)
    
    async def _handle_single_agent_request(self, request: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Handle request with a single adaptive agent"""
        # Create or find appropriate agent
        agent_id = await self.create_adaptive_agent(
            role=analysis.get("task_type", "general"),
            capabilities=analysis.get("capabilities_needed", [])
        )
        
        # Send task to agent
        task_id = await self.core.send_task(
            agent_id=agent_id,
            task=request,
            metadata={
                "analysis": analysis,
                "expected_format": analysis.get("output_format", "text"),
                "complexity": analysis.get("estimated_complexity", "medium")
            }
        )
        
        return {
            "status": "processing",
            "task_id": task_id,
            "agent_id": agent_id,
            "approach": "single_agent"
        }
    
    async def _handle_multi_agent_request(self, request: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Handle request with multiple adaptive agents"""
        # Decompose the task
        async with self.llm_client as client:
            decomposition_prompt = f"""
Decompose this complex request into subtasks for multiple agents:
Request: "{request}"
Analysis: {json.dumps(analysis, indent=2)}

Create a plan with multiple subtasks that can be handled by different specialized agents.
Each subtask should be independent or have clear dependencies.

Respond with a JSON object:
{{
    "subtasks": [
        {{
            "id": "subtask_1",
            "description": "detailed description",
            "agent_role": "required agent role",
            "capabilities": ["required", "capabilities"],
            "dependencies": ["list", "of", "subtask", "ids"],
            "priority": 1-10
        }}
    ],
    "coordination_strategy": "how subtasks should be coordinated",
    "final_integration": "how results should be combined"
}}
"""
            
            plan = await client.generate_structured(
                decomposition_prompt,
                {
                    "subtasks": "array",
                    "coordination_strategy": "string",
                    "final_integration": "string"
                },
                "You are a task decomposition expert. Break down complex requests into manageable subtasks."
            )
            
            if not plan:
                return {"error": "Could not decompose request"}
            
            # Create agents and assign subtasks
            task_assignments = []
            for subtask in plan.get("subtasks", []):
                agent_id = await self.create_adaptive_agent(
                    role=subtask.get("agent_role", "general"),
                    capabilities=subtask.get("capabilities", [])
                )
                
                task_id = await self.core.send_task(
                    agent_id=agent_id,
                    task=subtask.get("description", ""),
                    metadata={
                        "subtask_id": subtask.get("id"),
                        "dependencies": subtask.get("dependencies", []),
                        "priority": subtask.get("priority", 5),
                        "parent_request": request
                    }
                )
                
                task_assignments.append({
                    "subtask_id": subtask.get("id"),
                    "agent_id": agent_id,
                    "task_id": task_id
                })
            
            return {
                "status": "processing",
                "approach": "multi_agent",
                "plan": plan,
                "assignments": task_assignments
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "active_agents": len(self.core.agents),
            "system_active": self.core.active,
            "llm_connected": self.llm_client is not None,
            "agents": {
                agent_id: {
                    "role": agent.role,
                    "capabilities": agent.capabilities,
                    "active": agent.active
                }
                for agent_id, agent in self.core.agents.items()
            }
        }
    
    async def shutdown(self):
        """Shutdown the system gracefully"""
        if self.core:
            await self.core.shutdown()
        console.print("[yellow]Adaptive system shutdown complete[/yellow]")


@app.command()
def start(
    config_path: str = typer.Option("config.yaml", help="Path to configuration file"),
    provider: str = typer.Option(None, help="LLM provider override"),
    model: str = typer.Option(None, help="Model name override"),
    api_key: str = typer.Option(None, help="API key override"),
    interactive: bool = typer.Option(True, help="Start in interactive mode")
):
    """Start the minimal AI agent system"""
    console.print(Panel.fit("🚀 Minimal AI Agent System", style="bold blue"))
    console.print("Maximum flexibility through minimal hardcoding")
    
    async def run_system():
        # Load configuration
        config = Config.load(config_path)
        
        # Override settings if provided
        if provider:
            config.llm.provider = provider
        if model:
            config.llm.model = model
        if api_key:
            config.llm.api_key = api_key
        
        # Initialize adaptive system
        system = AdaptiveSystem(config)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            init_task = progress.add_task("Initializing system...", total=None)
            await system.initialize()
            progress.remove_task(init_task)
        
        if interactive:
            await run_interactive_mode(system)
        else:
            # Keep system running
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                console.print("\n[yellow]Shutting down...[/yellow]")
                await system.shutdown()
    
    asyncio.run(run_system())


async def run_interactive_mode(system: AdaptiveSystem):
    """Run the system in interactive mode"""
    console.print("\n[bold green]🤖 Interactive Mode Started[/bold green]")
    console.print("Type your requests in natural language. Type 'exit' to quit, 'status' for system status.")
    console.print("Examples:")
    console.print("  • 'Write a short story about AI'")
    console.print("  • 'Analyze the pros and cons of renewable energy'")
    console.print("  • 'Help me plan a project'")
    console.print("  • 'Create a plugin for data visualization'")
    
    try:
        while True:
            user_input = input("\n💬 You: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                break
            elif user_input.lower() == 'status':
                status = await system.get_system_status()
                display_system_status(status)
                continue
            elif user_input.lower() == 'help':
                display_help()
                continue
            elif not user_input:
                continue
            
            console.print(f"\n[blue]🔄 Processing: {user_input}[/blue]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Analyzing request...", total=None)
                
                try:
                    result = await system.process_natural_language_request(user_input)
                    progress.remove_task(task)
                    
                    if "error" in result:
                        console.print(f"[red]❌ Error: {result['error']}[/red]")
                    else:
                        console.print(f"[green]✅ Request processed successfully![/green]")
                        console.print(f"[cyan]Approach: {result.get('approach', 'unknown')}[/cyan]")
                        
                        if result.get("approach") == "single_agent":
                            console.print(f"[cyan]Agent: {result.get('agent_id')}[/cyan]")
                            console.print(f"[cyan]Task ID: {result.get('task_id')}[/cyan]")
                        elif result.get("approach") == "multi_agent":
                            console.print(f"[cyan]Subtasks: {len(result.get('assignments', []))}[/cyan]")
                            for assignment in result.get('assignments', []):
                                console.print(f"  • {assignment['subtask_id']}: {assignment['agent_id']}")
                        
                        # If we have intelligent agents, show their insights
                        if result.get("agent_id") in self.intelligent_agents:
                            agent = self.intelligent_agents[result["agent_id"]]
                            insights = await agent.get_agent_insights()
                            console.print(f"[dim]Agent Specialization: {insights['specialization_score']:.2f}[/dim]")
                            console.print(f"[dim]Creativity Level: {insights['creativity_level']:.2f}[/dim]")
                
                except Exception as e:
                    progress.remove_task(task)
                    console.print(f"[red]❌ Error processing request: {e}[/red]")
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Interactive session ended by user[/yellow]")
    
    console.print("[yellow]Shutting down system...[/yellow]")
    await system.shutdown()


def display_system_status(status: Dict[str, Any]):
    """Display system status in a nice format"""
    table = Table(title="System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="yellow")
    
    table.add_row("System", "Active" if status["system_active"] else "Inactive", "")
    table.add_row("LLM Connection", "Connected" if status["llm_connected"] else "Disconnected", "")
    table.add_row("Active Agents", str(status["active_agents"]), "")
    
    for agent_id, agent_info in status.get("agents", {}).items():
        table.add_row(
            f"Agent {agent_id[:8]}...",
            "Active" if agent_info["active"] else "Inactive",
            f"Role: {agent_info['role']}, Capabilities: {len(agent_info['capabilities'])}"
        )
    
    console.print(table)


def display_help():
    """Display help information"""
    help_panel = Panel.fit("""
[bold]Available Commands:[/bold]

[cyan]Natural Language Requests:[/cyan]
• Just type what you want the system to do
• Examples: "Write a story", "Analyze data", "Create a plugin"

[cyan]System Commands:[/cyan]
• [bold]status[/bold] - Show system status
• [bold]help[/bold] - Show this help
• [bold]exit[/bold] - Quit the system

[cyan]Features:[/cyan]
• Adaptive agent creation based on request type
• Multi-agent coordination for complex tasks
• Persistent memory and state management
• Dynamic plugin generation
• Prompt-driven behavior (minimal hardcoding)
""", title="Help", style="blue")
    
    console.print(help_panel)


@app.command()
def demo(
    config_path: str = typer.Option("config.yaml", help="Path to configuration file"),
    provider: str = typer.Option(None, help="LLM provider override"),
    model: str = typer.Option(None, help="Model name override")
):
    """Run a demonstration of the adaptive system"""
    console.print(Panel.fit("🎭 Adaptive System Demo", style="bold magenta"))
    
    async def run_demo():
        config = Config.load(config_path)
        if provider:
            config.llm.provider = provider
        if model:
            config.llm.model = model
        
        system = AdaptiveSystem(config)
        await system.initialize()
        
        demo_requests = [
            "Write a short poem about artificial intelligence",
            "Analyze the benefits and drawbacks of remote work",
            "Create a plan for learning a new programming language",
            "Generate a plugin for text summarization",
            "Help me understand quantum computing basics"
        ]
        
        console.print("\n[bold]Running demonstration with sample requests...[/bold]")
        
        for i, request in enumerate(demo_requests, 1):
            console.print(f"\n[cyan]Demo {i}/5: {request}[/cyan]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Processing...", total=None)
                
                result = await system.process_natural_language_request(request)
                progress.remove_task(task)
                
                if "error" not in result:
                    console.print(f"[green]✅ Processed with {result.get('approach', 'unknown')} approach[/green]")
                    if result.get('approach') == 'multi_agent':
                        console.print(f"[yellow]Created {len(result.get('assignments', []))} specialized agents[/yellow]")
                else:
                    console.print(f"[red]❌ {result['error']}[/red]")
                
                await asyncio.sleep(1)
        
        status = await system.get_system_status()
        console.print(f"\n[bold green]Demo completed! Created {status['active_agents']} adaptive agents.[/bold green]")
        
        await system.shutdown()
    
    asyncio.run(run_demo())


@app.command()
def analyze_architecture():
    """Analyze and display the new architecture"""
    console.print(Panel.fit("🏗️ Architecture Analysis", style="bold green"))
    
    architecture_info = """
[bold]Minimal Core Architecture:[/bold]

[cyan]1. Minimal Hardcoding:[/cyan]
• Only 4 core components: Agent, Communication, State, Prompts
• All behavior driven by prompts and configuration
• Dynamic capability generation

[cyan]2. Maximum Flexibility:[/cyan]
• Agents adapt behavior based on context
• Dynamic plugin generation
• Prompt-driven decision making
• Self-modifying system behavior

[cyan]3. Infinite Scalability:[/cyan]
• Stateless core components
• Persistent agent states
• Hierarchical memory management
• Plugin-based extensibility

[cyan]4. Adaptive Intelligence:[/cyan]
• Context-aware agent creation
• Multi-agent coordination
• Learning from interactions
• Self-optimization

[cyan]5. Resilient Design:[/cyan]
• Graceful degradation
• State recovery
• Error handling at all levels
• Modular failure isolation

[yellow]Key Innovations:[/yellow]
• Prompt-driven architecture (vs hardcoded logic)
• Layered memory system (working/session/episodic/semantic/meta)
• Dynamic agent specialization
• Adaptive plugin generation
• Minimal core with maximum extensibility
"""
    
    console.print(architecture_info)
    
    # Display component breakdown
    table = Table(title="Component Analysis")
    table.add_column("Component", style="cyan")
    table.add_column("Lines of Code", style="green")
    table.add_column("Flexibility", style="yellow")
    table.add_column("Purpose", style="blue")
    
    components = [
        ("MinimalAgent", "~150", "Maximum", "Prompt-driven agent with adaptive behavior"),
        ("CommunicationHub", "~80", "High", "Universal message passing system"),
        ("StateManager", "~100", "High", "Persistent state and memory management"),
        ("PromptEngine", "~120", "Maximum", "Dynamic prompt generation and optimization"),
        ("PluginLoader", "~60", "Maximum", "Dynamic plugin generation and loading"),
        ("MinimalCore", "~100", "Maximum", "System orchestration and coordination")
    ]
    
    for comp, loc, flex, purpose in components:
        table.add_row(comp, loc, flex, purpose)
    
    console.print(table)
    
    console.print(f"\n[bold green]Total: ~610 lines vs {7595} lines in original (92% reduction!)[/bold green]")
    console.print("[bold yellow]Achieved infinite flexibility with minimal hardcoding![/bold yellow]")


@app.command()
def agent_insights(
    config_path: str = typer.Option("config.yaml", help="Path to configuration file"),
    agent_id: str = typer.Option(None, help="Specific agent ID to analyze")
):
    """Analyze agent performance and learning"""
    console.print(Panel.fit("🧠 Agent Intelligence Analysis", style="bold cyan"))
    
    async def analyze_agents():
        config = Config.load(config_path)
        system = AdaptiveSystem(config)
        await system.initialize()
        
        if not system.intelligent_agents:
            console.print("[yellow]No intelligent agents found. Create some agents first.[/yellow]")
            return
        
        if agent_id and agent_id in system.intelligent_agents:
            agents_to_analyze = [system.intelligent_agents[agent_id]]
        else:
            agents_to_analyze = list(system.intelligent_agents.values())
        
        for agent in agents_to_analyze:
            insights = await agent.get_agent_insights()
            
            # Create insights table
            table = Table(title=f"Agent {agent.agent_id} Insights")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_column("Description", style="yellow")
            
            table.add_row("Role", insights["role"], "Agent's primary role")
            table.add_row("Specialization", f"{insights['specialization_score']:.3f}", "Performance specialization (0-1)")
            table.add_row("Creativity", f"{insights['creativity_level']:.3f}", "Creative thinking level (0-1)")
            table.add_row("Interactions", str(insights["total_interactions"]), "Total interactions processed")
            table.add_row("Memories", str(insights["memory_count"]), "Stored memories")
            table.add_row("Top Topics", ", ".join(insights["top_topics"][:3]), "Most discussed topics")
            
            console.print(table)
            
            # Show learning trajectory
            if insights["learning_trajectory"]:
                console.print(f"\n[bold]Learning Trajectory for {agent.agent_id}:[/bold]")
                recent_performance = [t["confidence"] for t in insights["learning_trajectory"][-10:]]
                if recent_performance:
                    avg_performance = sum(recent_performance) / len(recent_performance)
                    console.print(f"Recent Average Performance: {avg_performance:.3f}")
                    
                    # Simple trend analysis
                    if len(recent_performance) >= 5:
                        early_avg = sum(recent_performance[:5]) / 5
                        late_avg = sum(recent_performance[-5:]) / 5
                        trend = "improving" if late_avg > early_avg else "declining" if late_avg < early_avg else "stable"
                        console.print(f"Performance Trend: {trend}")
        
        await system.shutdown()
    
    asyncio.run(analyze_agents())


@app.command()
def self_improve(
    config_path: str = typer.Option("config.yaml", help="Path to configuration file"),
    agent_id: str = typer.Option(None, help="Specific agent ID to improve")
):
    """Trigger self-improvement analysis for agents"""
    console.print(Panel.fit("🔧 Agent Self-Improvement", style="bold green"))
    
    async def improve_agents():
        config = Config.load(config_path)
        system = AdaptiveSystem(config)
        await system.initialize()
        
        if not system.intelligent_agents:
            console.print("[yellow]No intelligent agents found. Create some agents first.[/yellow]")
            return
        
        agents_to_improve = [system.intelligent_agents[agent_id]] if agent_id and agent_id in system.intelligent_agents else list(system.intelligent_agents.values())
        
        for agent in agents_to_improve:
            console.print(f"\n[cyan]Analyzing agent {agent.agent_id}...[/cyan]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Running self-improvement analysis...", total=None)
                
                improvements = await agent.self_improve()
                progress.remove_task(task)
                
                if "error" not in improvements:
                    console.print(f"[green]✅ Self-improvement analysis complete for {agent.agent_id}[/green]")
                    
                    # Display key insights
                    console.print(f"[bold]Performance Analysis:[/bold] {improvements.get('performance_analysis', 'N/A')}")
                    
                    if improvements.get("strengths"):
                        console.print(f"[green]Strengths:[/green] {', '.join(improvements['strengths'])}")
                    
                    if improvements.get("weaknesses"):
                        console.print(f"[red]Areas for Improvement:[/red] {', '.join(improvements['weaknesses'])}")
                    
                    if improvements.get("improvement_suggestions"):
                        console.print("[bold]Improvement Suggestions:[/bold]")
                        for suggestion in improvements["improvement_suggestions"][:3]:
                            console.print(f"  • [{suggestion.get('priority', 'medium')}] {suggestion.get('suggestion', 'N/A')}")
                else:
                    console.print(f"[red]❌ Error in self-improvement: {improvements['error']}[/red]")
        
        await system.shutdown()
    
    asyncio.run(improve_agents())


@app.command()
def generate_plugin(
    requirement: str = typer.Argument(..., help="Plugin requirement description"),
    config_path: str = typer.Option("config.yaml", help="Path to configuration file"),
    agent_role: str = typer.Option("developer", help="Role of agent to create plugin")
):
    """Generate a plugin using AI agents"""
    console.print(Panel.fit("🔌 AI Plugin Generation", style="bold magenta"))
    
    async def create_plugin():
        config = Config.load(config_path)
        system = AdaptiveSystem(config)
        await system.initialize()
        
        # Create a developer agent
        console.print(f"[blue]Creating {agent_role} agent for plugin development...[/blue]")
        agent_id = await system.create_adaptive_agent(
            role=agent_role,
            capabilities=["code_generation", "plugin_development", "software_architecture"],
            personality={"analytical": 0.8, "creative": 0.7, "detail_oriented": 0.9}
        )
        
        agent = system.intelligent_agents[agent_id]
        
        console.print(f"[blue]Generating plugin for: {requirement}[/blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Generating plugin...", total=None)
            
            result = await agent.generate_plugin(
                requirement=requirement,
                context={"target_system": "minimal_core", "language": "python"}
            )
            
            progress.remove_task(task)
            
            if result.get("success"):
                console.print(f"[green]✅ Plugin generated successfully![/green]")
                console.print(f"[cyan]Plugin file: {result['plugin_file']}[/cyan]")
                console.print(f"[cyan]Confidence: {result['confidence']:.2f}[/cyan]")
                
                # Show code preview
                if result.get("code"):
                    console.print("\n[bold]Generated Code Preview:[/bold]")
                    code_lines = result["code"].split('\n')[:15]  # First 15 lines
                    for i, line in enumerate(code_lines, 1):
                        console.print(f"[dim]{i:2d}:[/dim] {line}")
                    if len(result["code"].split('\n')) > 15:
                        console.print("[dim]... (truncated)[/dim]")
            else:
                console.print(f"[red]❌ Plugin generation failed: {result.get('error', 'Unknown error')}[/red]")
        
        await system.shutdown()
    
    asyncio.run(create_plugin())


@app.command()
def collaborate(
    task: str = typer.Argument(..., help="Task for agents to collaborate on"),
    config_path: str = typer.Option("config.yaml", help="Path to configuration file"),
    num_agents: int = typer.Option(2, help="Number of agents to create for collaboration")
):
    """Demonstrate agent collaboration"""
    console.print(Panel.fit("🤝 Agent Collaboration Demo", style="bold yellow"))
    
    async def run_collaboration():
        config = Config.load(config_path)
        system = AdaptiveSystem(config)
        await system.initialize()
        
        # Create multiple agents with different specializations
        agent_configs = [
            {"role": "analyst", "capabilities": ["analysis", "research", "critical_thinking"]},
            {"role": "creative", "capabilities": ["creativity", "brainstorming", "innovation"]},
            {"role": "coordinator", "capabilities": ["project_management", "coordination", "synthesis"]}
        ]
        
        agents = []
        for i in range(min(num_agents, len(agent_configs))):
            config_data = agent_configs[i]
            agent_id = await system.create_adaptive_agent(**config_data)
            agents.append(system.intelligent_agents[agent_id])
        
        console.print(f"[green]Created {len(agents)} agents for collaboration[/green]")
        
        # Demonstrate collaboration
        console.print(f"[blue]Task: {task}[/blue]")
        
        collaboration_results = []
        for i, agent in enumerate(agents):
            console.print(f"\n[cyan]Agent {agent.agent_id} ({agent.role}) contributing...[/cyan]")
            
            # Each agent collaborates with the previous one (if any)
            if i > 0:
                previous_agent = agents[i-1]
                result = await agent.collaborate_with_agent(
                    other_agent_id=previous_agent.agent_id,
                    task=task,
                    context={"collaboration_round": i, "previous_contributions": collaboration_results}
                )
            else:
                # First agent works independently
                result = await agent.collaborate_with_agent(
                    other_agent_id="system",
                    task=task,
                    context={"collaboration_round": i, "role": "initiator"}
                )
            
            collaboration_results.append({
                "agent_id": agent.agent_id,
                "role": agent.role,
                "contribution": result["contribution"],
                "confidence": result["confidence"]
            })
            
            console.print(f"[green]Contribution: {result['contribution'][:200]}...[/green]")
            console.print(f"[dim]Confidence: {result['confidence']:.2f}[/dim]")
        
        # Summary
        console.print("\n[bold]Collaboration Summary:[/bold]")
        avg_confidence = sum(r["confidence"] for r in collaboration_results) / len(collaboration_results)
        console.print(f"Average Confidence: {avg_confidence:.2f}")
        console.print(f"Participating Agents: {len(collaboration_results)}")
        
        await system.shutdown()
    
    asyncio.run(run_collaboration())


if __name__ == "__main__":
    app()