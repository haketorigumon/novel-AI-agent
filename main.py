#!/usr/bin/env python3
"""
Novel AI Agent - Main Entry Point
Combines Dynamic World Story Simulation with Darwin-Godel Machine for self-improving novel generation
"""

import asyncio
import typer
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from src.core.novel_agent import NovelAIAgent
from src.web.server import WebServer
from src.utils.config import Config

console = Console()
app = typer.Typer(help="Novel AI Agent - Self-improving long-form story generator")

@app.command()
def generate(
    config_path: str = typer.Option("config.yaml", help="Path to configuration file"),
    output_dir: str = typer.Option("output", help="Output directory for generated content"),
    web_interface: bool = typer.Option(False, help="Enable web interface"),
    auto_evolve: bool = typer.Option(True, help="Enable automatic code evolution")
):
    """Generate a novel using the AI agent system"""
    console.print(Panel.fit("🤖 Novel AI Agent Starting...", style="bold blue"))
    
    # Load configuration
    config = Config.load(config_path)
    config.story.output_dir = output_dir
    config.evolution.enabled = auto_evolve
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize the novel agent
    agent = NovelAIAgent(config)
    
    if web_interface:
        # Start web interface
        server = WebServer(agent, config)
        asyncio.run(server.start())
    else:
        # Run in CLI mode
        asyncio.run(agent.generate_novel())

@app.command()
def evolve(
    config_path: str = typer.Option("config.yaml", help="Path to configuration file"),
    generations: int = typer.Option(5, help="Number of evolution generations"),
):
    """Manually trigger code evolution"""
    console.print(Panel.fit("🧬 Starting Code Evolution...", style="bold green"))
    
    config = Config.load(config_path)
    agent = NovelAIAgent(config)
    
    asyncio.run(agent.evolve_system(generations))

@app.command()
def web(
    config_path: str = typer.Option("config.yaml", help="Path to configuration file"),
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(12000, help="Port to bind to")
):
    """Start the web interface"""
    console.print(Panel.fit("🌐 Starting Web Interface...", style="bold cyan"))
    
    config = Config.load(config_path)
    config.web_interface.host = host
    config.web_interface.port = port
    
    agent = NovelAIAgent(config)
    server = WebServer(agent, config)
    
    asyncio.run(server.start())

if __name__ == "__main__":
    app()