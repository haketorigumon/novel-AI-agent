#!/usr/bin/env python3
"""
Novel AI Agent - Main Entry Point
Combines Dynamic World Story Simulation with Darwin-Godel Machine for self-improving novel generation
Now with Multi-Agent System capabilities
"""

import asyncio
import typer
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from src.core.novel_agent import NovelAIAgent
from src.core.multi_agent.multi_agent_system import MultiAgentSystem
from src.web.server import WebServer
from src.utils.config import Config
from src.utils.llm_client import LLMClient

console = Console()
app = typer.Typer(help="Novel AI Agent - Self-improving long-form story generator and multi-agent system")

@app.command()
def generate(
    config_path: str = typer.Option("config.yaml", help="Path to configuration file"),
    output_dir: str = typer.Option("output", help="Output directory for generated content"),
    web_interface: bool = typer.Option(False, help="Enable web interface"),
    auto_evolve: bool = typer.Option(True, help="Enable automatic code evolution"),
    provider: str = typer.Option(None, help="LLM provider (ollama, openai, anthropic, google, etc.)"),
    model: str = typer.Option(None, help="Model name"),
    api_key: str = typer.Option(None, help="API key for the provider")
):
    """Generate a novel using the AI agent system"""
    console.print(Panel.fit("🤖 Novel AI Agent Starting...", style="bold blue"))
    
    # Load configuration
    config = Config.load(config_path)
    config.story.output_dir = output_dir
    config.evolution.enabled = auto_evolve
    
    # Override LLM settings if provided via command line
    if provider:
        config.llm.provider = provider
    if model:
        config.llm.model = model
    if api_key:
        config.llm.api_key = api_key
    
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

@app.command()
def providers():
    """List all supported LLM providers"""
    from src.utils.llm_client import LLMClient
    
    console.print(Panel.fit("🤖 Supported LLM Providers", style="bold green"))
    
    providers = LLMClient.get_supported_providers()
    
    for provider in providers:
        info = LLMClient.get_provider_info(provider)
        api_key_required = "✅ Required" if info.get("requires_api_key") else "❌ Not Required"
        console.print(f"• {provider.upper()}: API Key {api_key_required}")
    
    console.print("\n📖 For detailed setup instructions, see: PROVIDER_GUIDE.md")
    console.print("💡 Example: python main.py generate --provider openai --model gpt-4")

@app.command()
def test_connection(
    provider: str = typer.Option("ollama", help="Provider to test"),
    model: str = typer.Option("llama3", help="Model to test"),
    api_key: str = typer.Option(None, help="API key for testing")
):
    """Test connection to an LLM provider"""
    console.print(f"🔍 Testing connection to {provider} with model {model}...")
    
    async def test():
        from src.utils.llm_client import LLMClient
        
        client = LLMClient(provider=provider, model=model, api_key=api_key)
        async with client:
            if await client.check_connection():
                console.print(f"✅ Successfully connected to {provider}")
                
                # Test generation
                response = await client.generate("Hello, world!", "You are a helpful assistant.")
                if response:
                    console.print(f"📝 Test response: {response[:100]}...")
                    console.print("🎉 Provider is working correctly!")
                else:
                    console.print("⚠️ Connection successful but no response generated")
            else:
                console.print(f"❌ Failed to connect to {provider}")
                console.print("💡 Check your API key and network connection")
    
    asyncio.run(test())

@app.command()
def multi_agent(
    config_path: str = typer.Option("config.yaml", help="Path to configuration file"),
    output_dir: str = typer.Option("output", help="Output directory for generated content"),
    web_interface: bool = typer.Option(True, help="Enable web interface"),
    provider: str = typer.Option(None, help="LLM provider (ollama, openai, anthropic, google, etc.)"),
    model: str = typer.Option(None, help="Model name"),
    api_key: str = typer.Option(None, help="API key for the provider"),
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(12000, help="Port to bind to")
):
    """Start the multi-agent system"""
    console.print(Panel.fit("🤖 Multi-Agent System Starting...", style="bold magenta"))
    
    # Load configuration
    config = Config.load(config_path)
    config.story.output_dir = output_dir
    
    # Override LLM settings if provided via command line
    if provider:
        config.llm.provider = provider
    if model:
        config.llm.model = model
    if api_key:
        config.llm.api_key = api_key
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    async def start_multi_agent_system():
        # Initialize LLM client
        api_key = config.get_api_key_for_provider(config.llm.provider)
        llm_client = LLMClient(
            provider=config.llm.provider,
            model=config.llm.model,
            base_url=config.llm.base_url,
            api_key=api_key,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens
        )
        
        # Check LLM connection
        async with llm_client as client:
            if not await client.check_connection():
                console.print("[red]Warning: Cannot connect to LLM service. Please ensure the LLM service is running.[/red]")
                return
        
        # Initialize multi-agent system
        system = MultiAgentSystem(config, llm_client)
        await system.initialize()
        
        console.print("[green]Multi-agent system initialized successfully![/green]")
        
        if web_interface:
            # Start web interface
            config.web_interface.host = host
            config.web_interface.port = port
            
            # TODO: Create a dedicated web interface for the multi-agent system
            # For now, we'll use a simple message to indicate it's running
            console.print(f"[cyan]Web interface would start on {host}:{port}[/cyan]")
            console.print("[yellow]Web interface for multi-agent system is not yet implemented.[/yellow]")
            
            # Keep the system running
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                console.print("[yellow]Shutting down multi-agent system...[/yellow]")
                await system.shutdown()
        else:
            # Run a simple demo
            console.print("[blue]Running multi-agent system demo...[/blue]")
            
            # Create a few agents
            console.print("[blue]Creating agents...[/blue]")
            
            assistant_id = await system.create_agent(
                agent_type="assistant",
                name="General Assistant",
                role="assistant",
                description="A helpful assistant that can answer general questions"
            )
            
            expert_id = await system.create_agent(
                agent_type="expert",
                name="AI Expert",
                role="expert",
                description="An expert in artificial intelligence and machine learning",
                expertise_area="artificial intelligence"
            )
            
            creative_id = await system.create_agent(
                agent_type="creative",
                name="Creative Writer",
                role="writer",
                description="A creative writer specializing in fiction",
                creative_domain="fiction writing"
            )
            
            console.print(f"[green]Created agents: Assistant ({assistant_id}), Expert ({expert_id}), Creative ({creative_id})[/green]")
            
            # Process a few sample requests
            console.print("[blue]Processing sample requests...[/blue]")
            
            requests = [
                "What is artificial intelligence?",
                "Write a short story about a robot learning to feel emotions",
                "What are the differences between supervised and unsupervised learning?"
            ]
            
            for i, request in enumerate(requests):
                console.print(f"[cyan]Request {i+1}: {request}[/cyan]")
                response = await system.process_user_request(f"user_{i}", request)
                console.print(f"[green]Response: {response}[/green]")
                await asyncio.sleep(1)
            
            console.print("[green]Demo completed![/green]")
            await system.shutdown()
    
    asyncio.run(start_multi_agent_system())

@app.command()
def agent_chat(
    config_path: str = typer.Option("config.yaml", help="Path to configuration file"),
    output_dir: str = typer.Option("output", help="Output directory for generated content"),
    provider: str = typer.Option(None, help="LLM provider (ollama, openai, anthropic, google, etc.)"),
    model: str = typer.Option(None, help="Model name"),
    api_key: str = typer.Option(None, help="API key for the provider")
):
    """Start an interactive chat with the multi-agent system"""
    console.print(Panel.fit("💬 Multi-Agent Chat Starting...", style="bold yellow"))
    
    # Load configuration
    config = Config.load(config_path)
    config.story.output_dir = output_dir
    
    # Override LLM settings if provided via command line
    if provider:
        config.llm.provider = provider
    if model:
        config.llm.model = model
    if api_key:
        config.llm.api_key = api_key
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    async def start_agent_chat():
        # Initialize LLM client
        api_key = config.get_api_key_for_provider(config.llm.provider)
        llm_client = LLMClient(
            provider=config.llm.provider,
            model=config.llm.model,
            base_url=config.llm.base_url,
            api_key=api_key,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens
        )
        
        # Check LLM connection
        async with llm_client as client:
            if not await client.check_connection():
                console.print("[red]Warning: Cannot connect to LLM service. Please ensure the LLM service is running.[/red]")
                return
        
        # Initialize multi-agent system
        system = MultiAgentSystem(config, llm_client)
        await system.initialize()
        
        console.print("[green]Multi-agent system initialized successfully![/green]")
        
        # Create a default assistant agent
        assistant_id = await system.create_agent(
            agent_type="assistant",
            name="Chat Assistant",
            role="assistant",
            description="A helpful assistant that can answer questions and engage in conversation"
        )
        
        console.print("[green]Created chat assistant agent[/green]")
        console.print("[blue]You can now chat with the assistant. Type 'exit' to quit.[/blue]")
        
        user_id = "interactive_user"
        
        try:
            while True:
                # Get user input
                user_input = input("\nYou: ")
                
                if user_input.lower() in ["exit", "quit", "bye"]:
                    break
                
                # Process user request
                response = await system.process_user_request(user_id, user_input)
                
                # Display response
                if response.get("success", False):
                    task_id = response.get("task_id")
                    
                    # Wait for task to complete
                    max_wait = 30  # seconds
                    for _ in range(max_wait):
                        task = await system.get_task(task_id)
                        if task and task.get("status") in ["completed", "failed", "declined"]:
                            break
                        await asyncio.sleep(1)
                    
                    # Get task result
                    task = await system.get_task(task_id)
                    if task and task.get("status") == "completed":
                        result = task.get("result", "No result available")
                        console.print(f"\n[bold green]Assistant:[/bold green] {result}")
                    else:
                        console.print("\n[bold red]Assistant:[/bold red] Sorry, I couldn't process your request.")
                else:
                    console.print(f"\n[bold red]Error:[/bold red] {response.get('error', 'Unknown error')}")
        
        except KeyboardInterrupt:
            console.print("\n[yellow]Chat session ended by user.[/yellow]")
        
        console.print("[yellow]Shutting down multi-agent system...[/yellow]")
        await system.shutdown()
    
    asyncio.run(start_agent_chat())

if __name__ == "__main__":
    app()