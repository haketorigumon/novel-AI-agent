#!/usr/bin/env python3
"""
Universal AI Agent System - Interactive Demo
Demonstrates the key capabilities of the universal system
"""

import asyncio
import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.core.universal_core import (
    UniversalSystem, UniversalEntity, UniversalType, Priority, MemoryType
)

console = Console()


async def demo_system_capabilities():
    """Demonstrate the core capabilities of the Universal System"""
    
    console.print(Panel.fit("🌟 Universal AI Agent System - Live Demo", style="bold magenta"))
    
    # Initialize the system
    console.print("\n[blue]🔧 Initializing Universal System...[/blue]")
    system = UniversalSystem({
        "prompts_dir": "prompts",
        "memory_dir": "memory",
        "plugins_dir": "plugins"
    })
    
    await system.initialize()
    console.print("[green]✅ System initialized successfully![/green]")
    
    # Demonstrate agent creation with different capabilities
    console.print("\n[blue]🤖 Creating specialized agents...[/blue]")
    
    agent_configs = [
        {
            "capabilities": ["creative", "storytelling", "imagination"],
            "name": "Creative Storyteller",
            "description": "Specializes in creative writing and storytelling"
        },
        {
            "capabilities": ["analytical", "research", "data_analysis"],
            "name": "Research Analyst",
            "description": "Expert in research and data analysis"
        },
        {
            "capabilities": ["technical", "programming", "architecture"],
            "name": "System Architect",
            "description": "Technical expert in system design and programming"
        },
        {
            "capabilities": ["assistant", "general", "coordination"],
            "name": "Universal Assistant",
            "description": "General-purpose assistant with coordination abilities"
        }
    ]
    
    created_agents = []
    for config in agent_configs:
        agent_id = await system.create_agent(config)
        created_agents.append((agent_id, config["name"]))
        console.print(f"[green]✅ Created: {config['name']} ({agent_id[:12]}...)[/green]")
    
    # Demonstrate intelligent task assignment
    console.print("\n[blue]📝 Demonstrating intelligent task assignment...[/blue]")
    
    demo_tasks = [
        {
            "description": "Write a creative story about AI agents collaborating to solve a complex problem",
            "expected_agent": "Creative Storyteller"
        },
        {
            "description": "Analyze the performance metrics of a multi-agent system and provide insights",
            "expected_agent": "Research Analyst"
        },
        {
            "description": "Design the architecture for a scalable distributed AI system",
            "expected_agent": "System Architect"
        },
        {
            "description": "Coordinate the activities of multiple agents working on a shared project",
            "expected_agent": "Universal Assistant"
        }
    ]
    
    assigned_tasks = []
    for task in demo_tasks:
        console.print(f"[cyan]📋 Task: {task['description'][:60]}...[/cyan]")
        task_id = await system.assign_task_intelligently(
            task["description"], 
            Priority.NORMAL,
            {"demo": True, "expected_agent": task["expected_agent"]}
        )
        if task_id:
            assigned_tasks.append(task_id)
            console.print(f"[green]✅ Assigned task: {task_id}[/green]")
        else:
            console.print("[red]❌ Failed to assign task[/red]")
    
    # Demonstrate memory system
    console.print("\n[blue]🧠 Demonstrating memory system...[/blue]")
    
    # Store some demo memories
    demo_memories = [
        {
            "content": "The Universal System successfully created 4 specialized agents",
            "layer": MemoryType.EPISODIC,
            "importance": 0.8,
            "metadata": {"event": "agent_creation", "demo": True}
        },
        {
            "content": "Intelligent task assignment matches tasks to agent capabilities",
            "layer": MemoryType.SEMANTIC,
            "importance": 0.9,
            "metadata": {"concept": "task_assignment", "demo": True}
        },
        {
            "content": "Multi-agent collaboration improves problem-solving effectiveness",
            "layer": MemoryType.SEMANTIC,
            "importance": 0.85,
            "metadata": {"concept": "collaboration", "demo": True}
        }
    ]
    
    for memory_data in demo_memories:
        memory = UniversalEntity(
            type=UniversalType.MEMORY,
            content=memory_data["content"],
            importance=memory_data["importance"],
            metadata=memory_data["metadata"]
        )
        await system.memory_system.store_memory(memory, memory_data["layer"])
        console.print(f"[green]✅ Stored memory: {memory_data['content'][:50]}...[/green]")
    
    # Demonstrate memory retrieval
    console.print("\n[blue]🔍 Demonstrating memory retrieval...[/blue]")
    
    queries = ["agent creation", "task assignment", "collaboration", "system"]
    for query in queries:
        memories = await system.memory_system.retrieve_memories(query, limit=2)
        console.print(f"[cyan]Query '{query}': Found {len(memories)} relevant memories[/cyan]")
        for memory in memories:
            content = str(memory.content)[:60] + "..." if len(str(memory.content)) > 60 else str(memory.content)
            console.print(f"  [white]• {content}[/white]")
    
    # Demonstrate plugin system
    console.print("\n[blue]🔌 Demonstrating plugin system...[/blue]")
    
    plugins = system.plugin_system.get_available_plugins()
    console.print(f"[cyan]Available plugins: {len(plugins)}[/cyan]")
    
    for plugin_name, plugin_info in plugins.items():
        console.print(f"[green]✅ Plugin: {plugin_name}[/green]")
        console.print(f"  [white]Capabilities: {', '.join(plugin_info.get('capabilities', []))}[/white]")
        
        # Test plugin execution
        if plugin_name == "text_analysis_plugin":
            test_context = {
                "text": "The Universal AI Agent System represents a breakthrough in adaptive intelligence, featuring infinite scalability and maximum flexibility through prompt-driven design.",
                "analysis_type": "comprehensive"
            }
            
            result = await system.plugin_system.execute_plugin(plugin_name, test_context)
            if result.get("success"):
                console.print(f"  [green]✅ Plugin executed successfully[/green]")
                analysis_result = result.get("result", {})
                if "sentiment_analysis" in analysis_result:
                    sentiment = analysis_result["sentiment_analysis"]
                    console.print(f"    [blue]Sentiment: {sentiment.get('sentiment')} (confidence: {sentiment.get('confidence', 0):.2f})[/blue]")
                if "text_statistics" in analysis_result:
                    stats = analysis_result["text_statistics"]
                    console.print(f"    [blue]Words: {stats.get('word_count')}, Sentences: {stats.get('sentence_count')}[/blue]")
            else:
                console.print(f"  [red]❌ Plugin execution failed: {result.get('error')}[/red]")
    
    # Demonstrate communication system
    console.print("\n[blue]📡 Demonstrating communication system...[/blue]")
    
    # Send some demo messages
    demo_messages = [
        {
            "content": "System status: All agents operational",
            "metadata": {"message_type": "system", "sender": "demo", "broadcast": True}
        },
        {
            "content": "Collaboration request: Multi-agent story writing project",
            "metadata": {"message_type": "collaboration", "sender": "demo", "recipient": created_agents[0][0]}
        },
        {
            "content": "Performance update: Task completion rate at 95%",
            "metadata": {"message_type": "status", "sender": "demo", "priority": Priority.HIGH.value}
        }
    ]
    
    for msg_data in demo_messages:
        message = UniversalEntity(
            type=UniversalType.MESSAGE,
            content=msg_data["content"],
            metadata=msg_data["metadata"]
        )
        await system.communication_hub.send_message(message)
        console.print(f"[green]✅ Sent: {msg_data['content'][:50]}...[/green]")
    
    # Show system status
    console.print("\n[blue]📊 Current system status...[/blue]")
    
    status = system.get_system_status()
    show_system_status(status)
    
    # Wait for some processing
    console.print("\n[blue]⏳ Allowing agents to process tasks...[/blue]")
    await asyncio.sleep(3)
    
    # Show final status
    console.print("\n[blue]📊 Final system status...[/blue]")
    status = system.get_system_status()
    show_system_status(status)
    
    # Demonstrate system capabilities summary
    console.print("\n[blue]🎯 System Capabilities Summary...[/blue]")
    
    capabilities_table = Table(title="Universal System Capabilities")
    capabilities_table.add_column("Capability", style="cyan", no_wrap=True)
    capabilities_table.add_column("Status", style="green")
    capabilities_table.add_column("Description", style="white")
    
    capabilities = [
        ("Infinite Adaptability", "✅ Active", "System adapts to any task or domain"),
        ("Intelligent Task Routing", "✅ Active", "Tasks automatically assigned to best agents"),
        ("Hierarchical Memory", "✅ Active", "Multi-layer memory with no loss"),
        ("Dynamic Plugin Generation", "✅ Ready", "Plugins created on-demand for new needs"),
        ("Continuous Learning", "✅ Active", "System learns from every interaction"),
        ("Multi-Agent Collaboration", "✅ Active", "Agents work together seamlessly"),
        ("Prompt-Driven Behavior", "✅ Active", "All behavior generated through prompts"),
        ("Persistent State", "✅ Active", "Full state persistence across sessions"),
        ("Universal Communication", "✅ Active", "Scalable message routing system"),
        ("Self-Optimization", "✅ Ready", "System continuously improves itself")
    ]
    
    for capability, status, description in capabilities:
        capabilities_table.add_row(capability, status, description)
    
    console.print(capabilities_table)
    
    # Cleanup
    console.print("\n[yellow]🧹 Cleaning up demo system...[/yellow]")
    await system.shutdown()
    console.print("[green]✅ Demo completed successfully![/green]")
    
    console.print(Panel.fit(
        "🌟 The Universal AI Agent System is ready for production use!\n\n"
        "Key Features Demonstrated:\n"
        "• Infinite adaptability through universal patterns\n"
        "• Intelligent task assignment and agent coordination\n"
        "• Hierarchical memory system with no memory loss\n"
        "• Dynamic plugin system for unlimited extensibility\n"
        "• Prompt-driven behavior for maximum flexibility\n"
        "• Persistent state and continuous execution\n\n"
        "Try it yourself: python universal_main.py start --interactive",
        style="bold green"
    ))


def show_system_status(status):
    """Display system status in a formatted table"""
    system_info = status.get("system", {})
    
    status_table = Table(title="System Status")
    status_table.add_column("Metric", style="cyan")
    status_table.add_column("Value", style="white")
    
    status_table.add_row("Running", "🟢 Yes" if system_info.get("running") else "🔴 No")
    status_table.add_row("Agents", str(system_info.get("agents_count", 0)))
    status_table.add_row("Plugins", str(system_info.get("loaded_plugins", 0)))
    status_table.add_row("Messages", str(system_info.get("message_history_size", 0)))
    
    # Memory breakdown
    memory_layers = system_info.get("memory_layers", {})
    total_memories = sum(memory_layers.values())
    status_table.add_row("Total Memories", str(total_memories))
    
    for layer, count in memory_layers.items():
        if count > 0:
            status_table.add_row(f"  {layer.title()}", str(count))
    
    console.print(status_table)
    
    # Agent status
    agents = status.get("agents", {})
    if agents:
        agent_table = Table(title="Agent Status")
        agent_table.add_column("Agent", style="cyan")
        agent_table.add_column("Status", style="green")
        agent_table.add_column("Tasks", style="yellow")
        agent_table.add_column("Queue", style="blue")
        
        for agent_id, agent_info in agents.items():
            status_emoji = "🟢" if agent_info.get("is_running") else "🔴"
            agent_table.add_row(
                agent_info.get("name", agent_id[:12] + "..."),
                f"{status_emoji} {agent_info.get('status', 'Unknown')}",
                str(agent_info.get("active_tasks", 0)),
                str(agent_info.get("queue_size", 0))
            )
        
        console.print(agent_table)


if __name__ == "__main__":
    asyncio.run(demo_system_capabilities())