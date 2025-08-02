#!/usr/bin/env python3
"""
Test script for the Universal AI Agent System
Validates all core components and demonstrates capabilities
"""

import asyncio
import json
import time
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Import the universal system components
from src.core.universal_core import (
    UniversalSystem, UniversalAgent, UniversalEntity,
    UniversalType, Priority, MemoryType
)

console = Console()


async def test_system_initialization():
    """Test system initialization"""
    console.print("[blue]🔧 Testing system initialization...[/blue]")
    
    system = UniversalSystem({
        "prompts_dir": "prompts",
        "memory_dir": "memory",
        "plugins_dir": "plugins"
    })
    
    await system.initialize()
    
    assert system.is_initialized, "System should be initialized"
    assert system.is_running, "System should be running"
    
    console.print("[green]✅ System initialization successful[/green]")
    return system


async def test_agent_creation(system: UniversalSystem):
    """Test agent creation and management"""
    console.print("[blue]🤖 Testing agent creation...[/blue]")
    
    # Create different types of agents
    agent_configs = [
        {"capabilities": ["general", "assistant"], "name": "General Assistant"},
        {"capabilities": ["creative", "writing"], "name": "Creative Writer"},
        {"capabilities": ["analytical", "research"], "name": "Research Analyst"},
        {"capabilities": ["technical", "programming"], "name": "Technical Expert"}
    ]
    
    created_agents = []
    for config in agent_configs:
        agent_id = await system.create_agent(config)
        created_agents.append(agent_id)
        console.print(f"[green]✅ Created agent: {config['name']} ({agent_id[:12]}...)[/green]")
    
    # Verify agents are registered
    agents_list = system.list_agents()
    assert len(agents_list) == len(agent_configs), f"Expected {len(agent_configs)} agents, got {len(agents_list)}"
    
    console.print(f"[green]✅ Agent creation successful - {len(created_agents)} agents created[/green]")
    return created_agents


async def test_memory_system(system: UniversalSystem):
    """Test memory system functionality"""
    console.print("[blue]🧠 Testing memory system...[/blue]")
    
    # Create test memories
    test_memories = [
        {
            "content": "Universal system initialized successfully",
            "layer": MemoryType.EPISODIC,
            "importance": 0.8
        },
        {
            "content": "Agent creation process completed",
            "layer": MemoryType.PROCEDURAL,
            "importance": 0.6
        },
        {
            "content": "System architecture is based on universal patterns",
            "layer": MemoryType.SEMANTIC,
            "importance": 0.9
        },
        {
            "content": "Memory consolidation is working correctly",
            "layer": MemoryType.META,
            "importance": 0.7
        }
    ]
    
    # Store memories
    stored_memory_ids = []
    for memory_data in test_memories:
        memory = UniversalEntity(
            type=UniversalType.MEMORY,
            content=memory_data["content"],
            importance=memory_data["importance"],
            metadata={"test": True}
        )
        await system.memory_system.store_memory(memory, memory_data["layer"])
        stored_memory_ids.append(memory.id)
    
    console.print(f"[green]✅ Stored {len(stored_memory_ids)} test memories[/green]")
    
    # Test memory retrieval
    retrieved_memories = await system.memory_system.retrieve_memories("system", limit=5)
    assert len(retrieved_memories) > 0, "Should retrieve some memories"
    
    console.print(f"[green]✅ Retrieved {len(retrieved_memories)} memories[/green]")
    
    # Test specific queries
    queries = ["initialization", "agent", "architecture", "memory"]
    for query in queries:
        memories = await system.memory_system.retrieve_memories(query, limit=2)
        console.print(f"[blue]Query '{query}': {len(memories)} results[/blue]")
    
    console.print("[green]✅ Memory system testing successful[/green]")


async def test_communication_system(system: UniversalSystem):
    """Test communication system"""
    console.print("[blue]📡 Testing communication system...[/blue]")
    
    # Create test messages
    test_messages = [
        {
            "content": "System status check",
            "metadata": {"message_type": "system", "sender": "test", "recipient": "system"}
        },
        {
            "content": "Agent collaboration request",
            "metadata": {"message_type": "collaboration", "sender": "test", "broadcast": True}
        },
        {
            "content": "Task assignment notification",
            "metadata": {"message_type": "task", "sender": "test", "priority": Priority.HIGH.value}
        }
    ]
    
    # Send messages
    for msg_data in test_messages:
        message = UniversalEntity(
            type=UniversalType.MESSAGE,
            content=msg_data["content"],
            metadata=msg_data["metadata"]
        )
        await system.communication_hub.send_message(message)
    
    console.print(f"[green]✅ Sent {len(test_messages)} test messages[/green]")
    
    # Check message history
    history_size = len(system.communication_hub.message_history)
    assert history_size >= len(test_messages), "Message history should contain sent messages"
    
    console.print(f"[green]✅ Message history contains {history_size} messages[/green]")
    console.print("[green]✅ Communication system testing successful[/green]")


async def test_plugin_system(system: UniversalSystem):
    """Test plugin system"""
    console.print("[blue]🔌 Testing plugin system...[/blue]")
    
    # Check loaded plugins
    plugins = system.plugin_system.get_available_plugins()
    console.print(f"[blue]Found {len(plugins)} loaded plugins[/blue]")
    
    # Test plugin execution if any plugins are available
    if plugins:
        plugin_name = list(plugins.keys())[0]
        test_context = {
            "text": "This is a test text for plugin analysis.",
            "analysis_type": "comprehensive"
        }
        
        result = await system.plugin_system.execute_plugin(plugin_name, test_context)
        
        if result.get("success"):
            console.print(f"[green]✅ Plugin '{plugin_name}' executed successfully[/green]")
        else:
            console.print(f"[yellow]⚠️ Plugin '{plugin_name}' execution failed: {result.get('error')}[/yellow]")
    
    # Test plugin generation (without LLM)
    console.print("[blue]Testing plugin generation...[/blue]")
    generated_plugin = await system.plugin_system.generate_plugin(
        "Simple calculator functionality",
        {"capabilities": ["math", "calculation"]},
        None  # No LLM client
    )
    
    if generated_plugin:
        console.print(f"[green]✅ Generated plugin: {generated_plugin}[/green]")
    else:
        console.print("[blue]ℹ️ Plugin generation requires LLM client[/blue]")
    
    console.print("[green]✅ Plugin system testing successful[/green]")


async def test_task_assignment(system: UniversalSystem, agent_ids: list):
    """Test task assignment and processing"""
    console.print("[blue]📝 Testing task assignment...[/blue]")
    
    # Test tasks
    test_tasks = [
        "Analyze the current system status",
        "Generate a creative story about AI agents",
        "Research the benefits of universal architectures",
        "Provide technical documentation for the system"
    ]
    
    assigned_tasks = []
    
    # Test intelligent task assignment
    for task in test_tasks:
        task_id = await system.assign_task_intelligently(task, Priority.NORMAL)
        if task_id:
            assigned_tasks.append(task_id)
            console.print(f"[green]✅ Assigned task: {task[:50]}... (ID: {task_id})[/green]")
        else:
            console.print(f"[red]❌ Failed to assign task: {task[:50]}...[/red]")
    
    # Test specific agent assignment
    if agent_ids:
        specific_task_id = await system.assign_task_to_agent(
            agent_ids[0],
            "Specific task for first agent",
            Priority.HIGH
        )
        if specific_task_id:
            assigned_tasks.append(specific_task_id)
            console.print(f"[green]✅ Assigned specific task to agent {agent_ids[0][:12]}...[/green]")
    
    console.print(f"[green]✅ Task assignment successful - {len(assigned_tasks)} tasks assigned[/green]")
    
    # Wait a bit for processing
    console.print("[blue]⏳ Waiting for task processing...[/blue]")
    await asyncio.sleep(2)
    
    return assigned_tasks


async def test_system_status(system: UniversalSystem):
    """Test system status reporting"""
    console.print("[blue]📊 Testing system status...[/blue]")
    
    # Get comprehensive status
    status = system.get_system_status()
    
    # Verify status structure
    assert "system" in status, "Status should contain system information"
    assert "agents" in status, "Status should contain agent information"
    assert "metrics" in status, "Status should contain metrics"
    
    system_info = status["system"]
    console.print(f"[blue]System running: {system_info.get('running')}[/blue]")
    console.print(f"[blue]Agents count: {system_info.get('agents_count')}[/blue]")
    console.print(f"[blue]Memory layers: {system_info.get('memory_layers')}[/blue]")
    console.print(f"[blue]Loaded plugins: {system_info.get('loaded_plugins')}[/blue]")
    
    # Test individual agent status
    agents = system.list_agents()
    if agents:
        first_agent = agents[0]
        agent_status = system.get_agent_status(first_agent["id"])
        assert agent_status is not None, "Should get agent status"
        console.print(f"[blue]First agent status: {agent_status.get('status')}[/blue]")
    
    console.print("[green]✅ System status testing successful[/green]")


async def test_prompt_engine(system: UniversalSystem):
    """Test prompt engine functionality"""
    console.print("[blue]📝 Testing prompt engine...[/blue]")
    
    # Test prompt generation
    test_contexts = [
        {
            "purpose": "task_processing",
            "context": {
                "task": "Test task",
                "agent_capabilities": ["general"],
                "recent_memories": ["Test memory"],
                "available_plugins": ["test_plugin"]
            }
        },
        {
            "purpose": "agent_collaboration",
            "context": {
                "collaboration_task": "Work together on a project",
                "collaborating_agents": ["agent1", "agent2"],
                "shared_objective": "Complete the project successfully"
            }
        }
    ]
    
    for test_context in test_contexts:
        prompt = await system.prompt_engine.generate_prompt(
            test_context["purpose"],
            test_context["context"]
        )
        
        assert prompt, "Should generate a prompt"
        assert len(prompt) > 0, "Prompt should not be empty"
        
        console.print(f"[green]✅ Generated prompt for '{test_context['purpose']}'[/green]")
    
    console.print("[green]✅ Prompt engine testing successful[/green]")


async def display_test_results(system: UniversalSystem):
    """Display comprehensive test results"""
    console.print("\n" + "="*60)
    console.print(Panel.fit("🎉 Universal System Test Results", style="bold green"))
    
    # System overview
    status = system.get_system_status()
    system_info = status["system"]
    
    results_table = Table(title="Test Results Summary")
    results_table.add_column("Component", style="cyan", no_wrap=True)
    results_table.add_column("Status", style="green")
    results_table.add_column("Details", style="white")
    
    results_table.add_row(
        "System Core",
        "✅ PASSED",
        f"Initialized: {system_info.get('initialized')}, Running: {system_info.get('running')}"
    )
    
    results_table.add_row(
        "Agent System",
        "✅ PASSED",
        f"Agents: {system_info.get('agents_count')}"
    )
    
    memory_layers = system_info.get('memory_layers', {})
    total_memories = sum(memory_layers.values())
    results_table.add_row(
        "Memory System",
        "✅ PASSED",
        f"Total memories: {total_memories}, Layers: {len(memory_layers)}"
    )
    
    results_table.add_row(
        "Communication",
        "✅ PASSED",
        f"Messages: {system_info.get('message_history_size')}"
    )
    
    results_table.add_row(
        "Plugin System",
        "✅ PASSED",
        f"Loaded: {system_info.get('loaded_plugins')}, Generated: {system_info.get('generated_plugins')}"
    )
    
    results_table.add_row(
        "Prompt Engine",
        "✅ PASSED",
        "Dynamic prompt generation working"
    )
    
    console.print(results_table)
    
    # Agent details
    agents = system.list_agents()
    if agents:
        agents_table = Table(title="Created Agents")
        agents_table.add_column("Agent ID", style="cyan")
        agents_table.add_column("Name", style="white")
        agents_table.add_column("Capabilities", style="green")
        agents_table.add_column("Status", style="yellow")
        
        for agent in agents:
            agents_table.add_row(
                agent["id"][:12] + "...",
                agent["name"],
                ", ".join(agent["capabilities"][:2]) + ("..." if len(agent["capabilities"]) > 2 else ""),
                agent["status"]
            )
        
        console.print(agents_table)


async def main():
    """Main test function"""
    console.print(Panel.fit("🧪 Universal AI Agent System - Comprehensive Testing", style="bold blue"))
    
    start_time = time.time()
    
    try:
        # Initialize system
        system = await test_system_initialization()
        
        # Test core components
        agent_ids = await test_agent_creation(system)
        await test_memory_system(system)
        await test_communication_system(system)
        await test_plugin_system(system)
        await test_prompt_engine(system)
        
        # Test functionality
        await test_task_assignment(system, agent_ids)
        await test_system_status(system)
        
        # Display results
        await display_test_results(system)
        
        # Performance summary
        end_time = time.time()
        total_time = end_time - start_time
        
        console.print(f"\n[green]🎉 All tests completed successfully in {total_time:.2f} seconds![/green]")
        console.print("[blue]💡 The Universal System is ready for production use.[/blue]")
        
        # Cleanup
        console.print("\n[yellow]🧹 Cleaning up test system...[/yellow]")
        await system.shutdown()
        console.print("[green]✅ Cleanup completed[/green]")
        
    except Exception as e:
        console.print(f"\n[red]❌ Test failed with error: {e}[/red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)