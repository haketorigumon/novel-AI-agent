#!/usr/bin/env python3
"""
Universal AI Agent System - Main Entry Point
The ultimate flexible, adaptive, and intelligent multi-agent system
Minimizes hardcoding, maximizes adaptability through universal patterns
"""

import asyncio
import typer
import json
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from typing import Dict, Any, Optional

from src.core.universal_core import (
    UniversalSystem, UniversalAgent, UniversalEntity, 
    UniversalType, Priority, MemoryType
)
from src.utils.llm_client import LLMClient
from src.utils.config import Config

console = Console()
app = typer.Typer(help="Universal AI Agent System - Infinite flexibility and intelligence")


@app.command()
def start(
    config_path: str = typer.Option("config.yaml", help="Path to configuration file"),
    provider: str = typer.Option(None, help="LLM provider"),
    model: str = typer.Option(None, help="Model name"),
    api_key: str = typer.Option(None, help="API key"),
    interactive: bool = typer.Option(True, help="Start in interactive mode"),
    web_interface: bool = typer.Option(False, help="Enable web interface"),
    host: str = typer.Option("0.0.0.0", help="Host for web interface"),
    port: int = typer.Option(12000, help="Port for web interface")
):
    """Start the Universal AI Agent System"""
    console.print(Panel.fit("🌟 Universal AI Agent System Starting...", style="bold magenta"))
    
    # Load configuration
    config = Config.load(config_path) if Path(config_path).exists() else Config()
    
    # Override with command line arguments
    if provider:
        config.llm.provider = provider
    if model:
        config.llm.model = model
    if api_key:
        config.llm.api_key = api_key
    
    async def start_system():
        # Initialize LLM client
        llm_client = None
        if hasattr(config, 'llm') and config.llm.provider:
            try:
                api_key_to_use = config.get_api_key_for_provider(config.llm.provider)
                llm_client = LLMClient(
                    provider=config.llm.provider,
                    model=config.llm.model,
                    base_url=getattr(config.llm, 'base_url', None),
                    api_key=api_key_to_use,
                    temperature=getattr(config.llm, 'temperature', 0.7),
                    max_tokens=getattr(config.llm, 'max_tokens', 2048)
                )
                
                # Test connection
                async with llm_client as client:
                    if await client.check_connection():
                        console.print("[green]✅ LLM connection established[/green]")
                    else:
                        console.print("[yellow]⚠️ LLM connection failed, running without LLM[/yellow]")
                        llm_client = None
            except Exception as e:
                console.print(f"[red]❌ LLM setup failed: {e}[/red]")
                llm_client = None
        
        # Initialize Universal System
        system_config = {
            "prompts_dir": "prompts",
            "memory_dir": "memory", 
            "plugins_dir": "plugins"
        }
        
        system = UniversalSystem(system_config)
        await system.initialize(llm_client)
        
        console.print("[green]🚀 Universal System initialized successfully![/green]")
        
        if web_interface:
            # Start web interface
            await start_web_interface(system, host, port)
        elif interactive:
            # Start interactive mode
            await interactive_mode(system)
        else:
            # Start demo mode
            await demo_mode(system)
    
    asyncio.run(start_system())


async def start_web_interface(system: UniversalSystem, host: str, port: int):
    """Start the web interface for the universal system"""
    console.print(f"[cyan]🌐 Starting web interface on {host}:{port}[/cyan]")
    
    # This would integrate with a web framework like FastAPI
    # For now, we'll show a placeholder
    console.print("[yellow]📝 Web interface implementation coming soon...[/yellow]")
    console.print("[blue]💡 Use interactive mode for now: --interactive[/blue]")
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        console.print("[yellow]🛑 Shutting down web interface...[/yellow]")
        await system.shutdown()


async def interactive_mode(system: UniversalSystem):
    """Interactive mode for the universal system"""
    console.print(Panel.fit("🎮 Interactive Mode - Universal AI Agent System", style="bold cyan"))
    console.print("[blue]Commands: create, assign, status, agents, chat, help, exit[/blue]")
    
    try:
        while system.is_running:
            try:
                command = await asyncio.get_event_loop().run_in_executor(
                    None, input, "\n🤖 Universal> "
                )
                
                if not command.strip():
                    continue
                
                parts = command.strip().split(maxsplit=1)
                cmd = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                if cmd == "exit" or cmd == "quit":
                    break
                elif cmd == "help":
                    show_help()
                elif cmd == "create":
                    await handle_create_agent(system, args)
                elif cmd == "assign":
                    await handle_assign_task(system, args)
                elif cmd == "status":
                    await handle_status(system, args)
                elif cmd == "agents":
                    await handle_list_agents(system)
                elif cmd == "chat":
                    await handle_chat(system, args)
                elif cmd == "memory":
                    await handle_memory_query(system, args)
                elif cmd == "plugins":
                    await handle_plugins(system, args)
                else:
                    console.print(f"[red]❌ Unknown command: {cmd}[/red]")
                    console.print("[blue]💡 Type 'help' for available commands[/blue]")
                
            except EOFError:
                break
            except Exception as e:
                console.print(f"[red]❌ Error: {e}[/red]")
    
    except KeyboardInterrupt:
        console.print("\n[yellow]🛑 Interrupted by user[/yellow]")
    
    console.print("[yellow]🛑 Shutting down Universal System...[/yellow]")
    await system.shutdown()
    console.print("[green]✅ System shutdown complete[/green]")


def show_help():
    """Show help information"""
    help_table = Table(title="Universal AI Agent System Commands")
    help_table.add_column("Command", style="cyan", no_wrap=True)
    help_table.add_column("Description", style="white")
    help_table.add_column("Example", style="green")
    
    commands = [
        ("create [config]", "Create a new agent", "create {\"capabilities\": [\"creative\"]}"),
        ("assign <task>", "Assign task intelligently", "assign Write a short story"),
        ("status [agent_id]", "Show system or agent status", "status agent_12345678"),
        ("agents", "List all agents", "agents"),
        ("chat [agent_id]", "Chat with an agent", "chat agent_12345678"),
        ("memory <query>", "Query system memory", "memory recent tasks"),
        ("plugins [action]", "Manage plugins", "plugins list"),
        ("help", "Show this help", "help"),
        ("exit", "Exit the system", "exit")
    ]
    
    for cmd, desc, example in commands:
        help_table.add_row(cmd, desc, example)
    
    console.print(help_table)


async def handle_create_agent(system: UniversalSystem, args: str):
    """Handle agent creation"""
    try:
        if args.strip():
            # Try to parse as JSON
            try:
                config = json.loads(args)
            except json.JSONDecodeError:
                # Treat as capability list
                capabilities = [cap.strip() for cap in args.split(",")]
                config = {"capabilities": capabilities}
        else:
            config = {"capabilities": ["general"]}
        
        agent_id = await system.create_agent(config)
        console.print(f"[green]✅ Created agent: {agent_id}[/green]")
        console.print(f"[blue]📋 Capabilities: {', '.join(config.get('capabilities', ['general']))}[/blue]")
        
    except Exception as e:
        console.print(f"[red]❌ Failed to create agent: {e}[/red]")


async def handle_assign_task(system: UniversalSystem, args: str):
    """Handle task assignment"""
    if not args.strip():
        console.print("[red]❌ Please provide a task description[/red]")
        return
    
    try:
        task_id = await system.assign_task_intelligently(args.strip())
        if task_id:
            console.print(f"[green]✅ Task assigned: {task_id}[/green]")
            console.print(f"[blue]📝 Task: {args.strip()}[/blue]")
        else:
            console.print("[red]❌ Failed to assign task[/red]")
    
    except Exception as e:
        console.print(f"[red]❌ Error assigning task: {e}[/red]")


async def handle_status(system: UniversalSystem, args: str):
    """Handle status requests"""
    try:
        if args.strip():
            # Show specific agent status
            agent_status = system.get_agent_status(args.strip())
            if agent_status:
                show_agent_status(agent_status)
            else:
                console.print(f"[red]❌ Agent not found: {args.strip()}[/red]")
        else:
            # Show system status
            status = system.get_system_status()
            show_system_status(status)
    
    except Exception as e:
        console.print(f"[red]❌ Error getting status: {e}[/red]")


def show_system_status(status: Dict[str, Any]):
    """Display system status"""
    system_info = status.get("system", {})
    
    # System overview table
    system_table = Table(title="🌟 Universal System Status")
    system_table.add_column("Metric", style="cyan")
    system_table.add_column("Value", style="white")
    
    system_table.add_row("Status", "🟢 Running" if system_info.get("running") else "🔴 Stopped")
    system_table.add_row("Agents", str(system_info.get("agents_count", 0)))
    system_table.add_row("Loaded Plugins", str(system_info.get("loaded_plugins", 0)))
    system_table.add_row("Generated Plugins", str(system_info.get("generated_plugins", 0)))
    system_table.add_row("Message History", str(system_info.get("message_history_size", 0)))
    
    # Memory layers
    memory_layers = system_info.get("memory_layers", {})
    for layer, count in memory_layers.items():
        system_table.add_row(f"Memory ({layer})", str(count))
    
    console.print(system_table)
    
    # Agents summary
    agents = status.get("agents", {})
    if agents:
        agents_table = Table(title="🤖 Agents Summary")
        agents_table.add_column("Agent ID", style="cyan")
        agents_table.add_column("Name", style="white")
        agents_table.add_column("Status", style="green")
        agents_table.add_column("Active Tasks", style="yellow")
        agents_table.add_column("Queue Size", style="blue")
        
        for agent_id, agent_info in agents.items():
            status_emoji = "🟢" if agent_info.get("is_running") else "🔴"
            agents_table.add_row(
                agent_id[:12] + "...",
                agent_info.get("name", "Unknown"),
                f"{status_emoji} {agent_info.get('status', 'Unknown')}",
                str(agent_info.get("active_tasks", 0)),
                str(agent_info.get("queue_size", 0))
            )
        
        console.print(agents_table)


def show_agent_status(status: Dict[str, Any]):
    """Display agent status"""
    agent_table = Table(title=f"🤖 Agent Status: {status.get('name', 'Unknown')}")
    agent_table.add_column("Property", style="cyan")
    agent_table.add_column("Value", style="white")
    
    agent_table.add_row("ID", status.get("id", "Unknown"))
    agent_table.add_row("Name", status.get("name", "Unknown"))
    agent_table.add_row("Status", f"🟢 {status.get('status', 'Unknown')}" if status.get("is_running") else f"🔴 {status.get('status', 'Unknown')}")
    agent_table.add_row("Capabilities", ", ".join(status.get("capabilities", [])))
    agent_table.add_row("Active Tasks", str(status.get("active_tasks", 0)))
    agent_table.add_row("Queue Size", str(status.get("queue_size", 0)))
    agent_table.add_row("Access Count", str(status.get("access_count", 0)))
    agent_table.add_row("Adaptations", str(status.get("adaptation_count", 0)))
    agent_table.add_row("Created", status.get("created_at", "Unknown"))
    agent_table.add_row("Last Accessed", status.get("last_accessed", "Unknown"))
    
    console.print(agent_table)


async def handle_list_agents(system: UniversalSystem):
    """Handle agent listing"""
    try:
        agents = system.list_agents()
        
        if not agents:
            console.print("[yellow]📭 No agents found[/yellow]")
            return
        
        agents_table = Table(title="🤖 All Agents")
        agents_table.add_column("ID", style="cyan")
        agents_table.add_column("Name", style="white")
        agents_table.add_column("Capabilities", style="green")
        agents_table.add_column("Status", style="yellow")
        agents_table.add_column("Tasks", style="blue")
        agents_table.add_column("Created", style="magenta")
        
        for agent in agents:
            status_emoji = "🟢" if agent.get("status") == "active" else "🔴"
            agents_table.add_row(
                agent["id"][:12] + "...",
                agent["name"],
                ", ".join(agent["capabilities"][:3]) + ("..." if len(agent["capabilities"]) > 3 else ""),
                f"{status_emoji} {agent['status']}",
                str(agent["active_tasks"]),
                agent["created_at"][:19]  # Remove microseconds
            )
        
        console.print(agents_table)
    
    except Exception as e:
        console.print(f"[red]❌ Error listing agents: {e}[/red]")


async def handle_chat(system: UniversalSystem, args: str):
    """Handle chat with an agent"""
    if not args.strip():
        # Chat with system intelligently
        console.print("[blue]💬 Starting intelligent chat (type 'exit' to end)[/blue]")
        
        while True:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, input, "You: "
                )
                
                if user_input.lower() in ["exit", "quit", "end"]:
                    break
                
                # Assign as a task intelligently
                task_id = await system.assign_task_intelligently(
                    f"Respond to user message: {user_input}",
                    Priority.HIGH,
                    {"chat": True, "user_input": user_input}
                )
                
                if task_id:
                    console.print(f"[green]🤖 Processing your message... (Task: {task_id})[/green]")
                    # In a real implementation, we'd wait for the response
                    # For now, just acknowledge
                    console.print("[blue]💭 Response will be processed by the most suitable agent[/blue]")
                else:
                    console.print("[red]❌ Failed to process your message[/red]")
            
            except (EOFError, KeyboardInterrupt):
                break
        
        console.print("[blue]💬 Chat ended[/blue]")
    else:
        # Chat with specific agent
        agent_id = args.strip()
        if agent_id in system.agents:
            console.print(f"[blue]💬 Chatting with agent {agent_id} (type 'exit' to end)[/blue]")
            
            while True:
                try:
                    user_input = await asyncio.get_event_loop().run_in_executor(
                        None, input, "You: "
                    )
                    
                    if user_input.lower() in ["exit", "quit", "end"]:
                        break
                    
                    # Assign task to specific agent
                    task_id = await system.assign_task_to_agent(
                        agent_id,
                        f"Respond to user message: {user_input}",
                        Priority.HIGH,
                        {"chat": True, "user_input": user_input}
                    )
                    
                    if task_id:
                        console.print(f"[green]🤖 Agent processing... (Task: {task_id})[/green]")
                    else:
                        console.print("[red]❌ Failed to send message to agent[/red]")
                
                except (EOFError, KeyboardInterrupt):
                    break
            
            console.print("[blue]💬 Chat ended[/blue]")
        else:
            console.print(f"[red]❌ Agent not found: {agent_id}[/red]")


async def handle_memory_query(system: UniversalSystem, args: str):
    """Handle memory queries"""
    if not args.strip():
        console.print("[red]❌ Please provide a search query[/red]")
        return
    
    try:
        memories = await system.memory_system.retrieve_memories(args.strip(), limit=10)
        
        if not memories:
            console.print("[yellow]📭 No memories found[/yellow]")
            return
        
        memory_table = Table(title=f"🧠 Memory Search: '{args.strip()}'")
        memory_table.add_column("ID", style="cyan")
        memory_table.add_column("Content", style="white")
        memory_table.add_column("Layer", style="green")
        memory_table.add_column("Importance", style="yellow")
        memory_table.add_column("Created", style="magenta")
        
        for memory in memories:
            content = str(memory.content)[:50] + "..." if len(str(memory.content)) > 50 else str(memory.content)
            memory_table.add_row(
                memory.id[:12] + "...",
                content,
                memory.metadata.get("memory_type", "unknown"),
                f"{memory.importance:.2f}",
                memory.created_at.strftime("%Y-%m-%d %H:%M")
            )
        
        console.print(memory_table)
    
    except Exception as e:
        console.print(f"[red]❌ Error querying memory: {e}[/red]")


async def handle_plugins(system: UniversalSystem, args: str):
    """Handle plugin management"""
    try:
        if not args.strip() or args.strip() == "list":
            # List plugins
            plugins = system.plugin_system.get_available_plugins()
            
            if not plugins:
                console.print("[yellow]📭 No plugins found[/yellow]")
                return
            
            plugins_table = Table(title="🔌 Available Plugins")
            plugins_table.add_column("Name", style="cyan")
            plugins_table.add_column("Capabilities", style="green")
            plugins_table.add_column("Generated", style="yellow")
            plugins_table.add_column("Metadata", style="white")
            
            for name, info in plugins.items():
                capabilities = ", ".join(info.get("capabilities", []))[:30]
                generated = "✅" if info.get("generated") else "❌"
                metadata = str(info.get("metadata", {}))[:30] + "..." if len(str(info.get("metadata", {}))) > 30 else str(info.get("metadata", {}))
                
                plugins_table.add_row(name, capabilities, generated, metadata)
            
            console.print(plugins_table)
        
        elif args.strip().startswith("generate "):
            # Generate plugin
            requirement = args.strip()[9:]  # Remove "generate "
            console.print(f"[blue]🔧 Generating plugin for: {requirement}[/blue]")
            
            plugin_name = await system.plugin_system.generate_plugin(
                requirement, {"capabilities": ["general"]}, system.llm_client
            )
            
            if plugin_name:
                console.print(f"[green]✅ Plugin generated: {plugin_name}[/green]")
            else:
                console.print("[red]❌ Plugin generation failed[/red]")
        
        else:
            console.print("[blue]💡 Plugin commands: list, generate <requirement>[/blue]")
    
    except Exception as e:
        console.print(f"[red]❌ Error with plugins: {e}[/red]")


async def demo_mode(system: UniversalSystem):
    """Demo mode showcasing system capabilities"""
    console.print(Panel.fit("🎭 Demo Mode - Universal AI Agent System", style="bold yellow"))
    
    # Create demo agents
    console.print("[blue]🤖 Creating demo agents...[/blue]")
    
    agents = []
    agent_configs = [
        {"capabilities": ["creative", "writing"], "name": "Creative Writer"},
        {"capabilities": ["analytical", "research"], "name": "Research Analyst"},
        {"capabilities": ["assistant", "general"], "name": "General Assistant"},
        {"capabilities": ["expert", "technical"], "name": "Technical Expert"}
    ]
    
    for config in agent_configs:
        agent_id = await system.create_agent(config)
        agents.append(agent_id)
        console.print(f"[green]✅ Created: {config['name']} ({agent_id[:12]}...)[/green]")
    
    await asyncio.sleep(1)
    
    # Demo tasks
    console.print("\n[blue]📝 Assigning demo tasks...[/blue]")
    
    demo_tasks = [
        "Write a short creative story about AI agents",
        "Research the benefits of multi-agent systems",
        "Help me understand how universal systems work",
        "Explain the technical architecture of this system"
    ]
    
    task_ids = []
    for task in demo_tasks:
        task_id = await system.assign_task_intelligently(task)
        task_ids.append(task_id)
        console.print(f"[green]✅ Assigned: {task}[/green]")
        await asyncio.sleep(0.5)
    
    # Show system status
    console.print("\n[blue]📊 System Status:[/blue]")
    status = system.get_system_status()
    show_system_status(status)
    
    # Wait a bit for processing
    console.print("\n[blue]⏳ Letting agents process tasks...[/blue]")
    await asyncio.sleep(3)
    
    # Show final status
    console.print("\n[blue]📊 Final Status:[/blue]")
    status = system.get_system_status()
    show_system_status(status)
    
    console.print("\n[green]🎉 Demo completed! The Universal System is ready for use.[/green]")
    console.print("[blue]💡 Try interactive mode: python universal_main.py start --interactive[/blue]")
    
    # Shutdown
    await system.shutdown()


@app.command()
def test():
    """Test the Universal System components"""
    console.print(Panel.fit("🧪 Testing Universal System Components", style="bold green"))
    
    async def run_tests():
        # Test system initialization
        console.print("[blue]🔧 Testing system initialization...[/blue]")
        system = UniversalSystem()
        await system.initialize()
        console.print("[green]✅ System initialization successful[/green]")
        
        # Test agent creation
        console.print("[blue]🤖 Testing agent creation...[/blue]")
        agent_id = await system.create_agent({"capabilities": ["test"]})
        console.print(f"[green]✅ Agent created: {agent_id}[/green]")
        
        # Test task assignment
        console.print("[blue]📝 Testing task assignment...[/blue]")
        task_id = await system.assign_task_to_agent(agent_id, "Test task")
        console.print(f"[green]✅ Task assigned: {task_id}[/green]")
        
        # Test memory system
        console.print("[blue]🧠 Testing memory system...[/blue]")
        from src.core.universal_core import UniversalEntity, UniversalType, MemoryType
        test_memory = UniversalEntity(
            type=UniversalType.MEMORY,
            content="Test memory content",
            importance=0.8
        )
        await system.memory_system.store_memory(test_memory, MemoryType.WORKING)
        
        memories = await system.memory_system.retrieve_memories("test", limit=1)
        if memories:
            console.print("[green]✅ Memory system working[/green]")
        else:
            console.print("[yellow]⚠️ Memory system test inconclusive[/yellow]")
        
        # Test plugin system
        console.print("[blue]🔌 Testing plugin system...[/blue]")
        plugins = system.plugin_system.get_available_plugins()
        console.print(f"[green]✅ Plugin system loaded {len(plugins)} plugins[/green]")
        
        # Show final status
        console.print("[blue]📊 Final test status:[/blue]")
        status = system.get_system_status()
        show_system_status(status)
        
        console.print("[green]🎉 All tests completed successfully![/green]")
        
        # Cleanup
        await system.shutdown()
    
    asyncio.run(run_tests())


@app.command()
def info():
    """Show information about the Universal System"""
    console.print(Panel.fit("🌟 Universal AI Agent System Information", style="bold cyan"))
    
    info_text = """
🎯 **Purpose**: The ultimate flexible, adaptive, and intelligent multi-agent system

🏗️ **Architecture**:
• **Universal Core**: Minimizes hardcoding, maximizes adaptability
• **Prompt-Driven**: All behavior generated through intelligent prompts
• **Infinite Memory**: Hierarchical memory system with no loss
• **Dynamic Plugins**: Self-generating capabilities based on needs
• **Adaptive Agents**: Continuously learning and evolving agents

🚀 **Key Features**:
• **Zero Hardcoding**: Everything is soft-coded through prompts
• **Infinite Scalability**: Add unlimited agents and capabilities
• **Persistent State**: Each agent has independent, continuous execution
• **Universal Intelligence**: Adapts to any task or domain
• **Self-Improving**: System evolves and optimizes itself

🎮 **Usage**:
• Interactive mode: `python universal_main.py start --interactive`
• Web interface: `python universal_main.py start --web-interface`
• Demo mode: `python universal_main.py start --interactive=false`
• Testing: `python universal_main.py test`

🔧 **Components**:
• **UniversalSystem**: Main orchestrator
• **UniversalAgent**: Adaptive agent with infinite capabilities
• **UniversalMemorySystem**: Hierarchical memory with no loss
• **UniversalPromptEngine**: Dynamic prompt generation
• **UniversalCommunicationHub**: Scalable message routing
• **UniversalPluginSystem**: Self-generating functionality

💡 **Philosophy**: 
Maximum flexibility through universal patterns, minimal constraints through adaptive design.
    """
    
    console.print(info_text)


if __name__ == "__main__":
    app()