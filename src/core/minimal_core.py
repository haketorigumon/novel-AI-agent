"""
Minimal Core System - The foundation of the flexible AI agent system
Only contains the absolute essentials: Agent framework, Communication, State persistence, Prompt engine
"""

import asyncio
import json
import uuid
import pickle
import aiofiles
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Set
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum


class MessageType(Enum):
    """Types of messages in the system"""
    TASK = "task"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    SYSTEM = "system"
    MEMORY = "memory"
    PLUGIN = "plugin"


@dataclass
class Message:
    """Universal message format"""
    id: str
    type: MessageType
    sender: str
    recipient: Optional[str]
    content: Any
    metadata: Dict[str, Any]
    timestamp: datetime
    priority: int = 5  # 1-10, 10 is highest
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'type': self.type.value,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        data['type'] = MessageType(data['type'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class MemoryLayer(Enum):
    """Memory layers for hierarchical memory management"""
    WORKING = "working"      # Current context, volatile
    SESSION = "session"      # Current session, semi-persistent
    EPISODIC = "episodic"    # Specific experiences, persistent
    SEMANTIC = "semantic"    # General knowledge, persistent
    META = "meta"           # Memory about memory, persistent


@dataclass
class Memory:
    """Universal memory structure"""
    id: str
    layer: MemoryLayer
    content: Any
    metadata: Dict[str, Any]
    importance: float  # 0.0 to 1.0
    access_count: int
    last_accessed: datetime
    created_at: datetime
    expires_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'layer': self.layer.value,
            'last_accessed': self.last_accessed.isoformat(),
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        data['layer'] = MemoryLayer(data['layer'])
        data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data['expires_at']:
            data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        return cls(**data)


class PromptEngine:
    """Dynamic prompt generation and optimization engine"""
    
    def __init__(self, templates_dir: str = "templates"):
        self.templates_dir = Path(templates_dir)
        self.templates: Dict[str, str] = {}
        self.template_metadata: Dict[str, Dict[str, Any]] = {}
        self.usage_stats: Dict[str, Dict[str, Any]] = {}
        
    async def initialize(self):
        """Initialize the prompt engine"""
        await self._load_templates()
        
    async def _load_templates(self):
        """Load prompt templates from files"""
        if not self.templates_dir.exists():
            self.templates_dir.mkdir(parents=True, exist_ok=True)
            await self._create_default_templates()
        
        for template_file in self.templates_dir.glob("*.txt"):
            template_name = template_file.stem
            async with aiofiles.open(template_file, 'r', encoding='utf-8') as f:
                self.templates[template_name] = await f.read()
            
            # Load metadata if exists
            metadata_file = template_file.with_suffix('.json')
            if metadata_file.exists():
                async with aiofiles.open(metadata_file, 'r', encoding='utf-8') as f:
                    self.template_metadata[template_name] = json.loads(await f.read())
    
    async def _create_default_templates(self):
        """Create default prompt templates"""
        default_templates = {
            "agent_initialization": """You are an AI agent with the following configuration:
ID: {agent_id}
Role: {role}
Capabilities: {capabilities}
Current Context: {context}

Your primary directive is to be adaptive, intelligent, and helpful while maintaining consistency with your role and capabilities.

Available actions: {available_actions}
Current memory: {recent_memories}

Respond with your next action or thought process.""",
            
            "task_processing": """You have received a new task:
Task ID: {task_id}
Task Type: {task_type}
Description: {description}
Context: {context}
Priority: {priority}

Your current state:
- Role: {role}
- Capabilities: {capabilities}
- Recent memories: {recent_memories}
- Available tools: {available_tools}

Analyze this task and determine:
1. Can you handle this task directly?
2. Do you need to collaborate with other agents?
3. What tools or resources do you need?
4. What is your approach?

Respond with a structured plan.""",
            
            "memory_consolidation": """Review and consolidate the following memories:
{memories}

Current memory layers:
- Working: {working_count} items
- Session: {session_count} items  
- Episodic: {episodic_count} items
- Semantic: {semantic_count} items

Determine:
1. Which memories should be promoted to higher layers?
2. Which memories can be compressed or summarized?
3. Which memories should be forgotten?
4. What patterns or insights emerge?

Respond with consolidation actions.""",
            
            "plugin_generation": """Generate a plugin for the following requirement:
Requirement: {requirement}
Context: {context}
Available interfaces: {interfaces}

The plugin should:
1. Be self-contained and modular
2. Follow the plugin interface specification
3. Include proper error handling
4. Be optimized for the specific use case

Generate the plugin code and metadata.""",
            
            "adaptive_response": """You are in an adaptive response mode. The system has encountered:
Situation: {situation}
Current state: {current_state}
Available options: {options}
Historical context: {history}

Your role is to:
1. Analyze the situation
2. Consider all available options
3. Choose the most appropriate response
4. Adapt your behavior if needed
5. Learn from this experience

Provide your adaptive response and reasoning.""",
            
            "agent_collaboration": """You are collaborating with another agent on a task:
Task: {task}
Collaborating with: {other_agent}
Your capabilities: {my_capabilities}
Context: {context}
Previous collaborations: {collaboration_history}

Your role in this collaboration:
1. Contribute your unique perspective and capabilities
2. Build upon or complement the other agent's work
3. Ensure effective communication and coordination
4. Focus on achieving the best possible outcome

Provide your contribution to this collaborative effort."""
        }
        
        for name, template in default_templates.items():
            template_file = self.templates_dir / f"{name}.txt"
            async with aiofiles.open(template_file, 'w', encoding='utf-8') as f:
                await f.write(template)
            
            # Create metadata
            metadata = {
                "description": f"Default template for {name}",
                "version": "1.0",
                "parameters": self._extract_parameters(template),
                "usage_count": 0,
                "success_rate": 0.0
            }
            
            metadata_file = template_file.with_suffix('.json')
            async with aiofiles.open(metadata_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(metadata, indent=2))
    
    def _extract_parameters(self, template: str) -> List[str]:
        """Extract parameter names from template"""
        import re
        return list(set(re.findall(r'\{(\w+)\}', template)))
    
    async def generate_prompt(self, template_name: str, **kwargs) -> str:
        """Generate a prompt from template with parameters"""
        if template_name not in self.templates:
            # Try to generate template dynamically
            await self._generate_dynamic_template(template_name, kwargs)
        
        template = self.templates.get(template_name, "")
        if not template:
            return f"Error: Template '{template_name}' not found"
        
        try:
            # Track usage
            if template_name not in self.usage_stats:
                self.usage_stats[template_name] = {"count": 0, "last_used": None}
            
            self.usage_stats[template_name]["count"] += 1
            self.usage_stats[template_name]["last_used"] = datetime.now().isoformat()
            
            return template.format(**kwargs)
        except KeyError as e:
            return f"Error: Missing parameter {e} for template '{template_name}'"
    
    async def _generate_dynamic_template(self, template_name: str, context: Dict[str, Any]):
        """Generate a template dynamically based on context"""
        # This is where the system becomes truly adaptive
        # For now, create a basic template
        self.templates[template_name] = f"""Dynamic template for {template_name}:
Context: {{context}}
Parameters: {{parameters}}

Please process this request appropriately."""


class StateManager:
    """Manages persistent state for agents and system"""
    
    def __init__(self, storage_dir: str = "state"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache: Dict[str, Dict[str, Memory]] = {}
        self.state_cache: Dict[str, Dict[str, Any]] = {}
        
    async def save_agent_state(self, agent_id: str, state: Dict[str, Any]):
        """Save agent state to persistent storage"""
        state_file = self.storage_dir / f"{agent_id}_state.json"
        async with aiofiles.open(state_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(state, indent=2, default=str))
        
        self.state_cache[agent_id] = state
    
    async def load_agent_state(self, agent_id: str) -> Dict[str, Any]:
        """Load agent state from persistent storage"""
        if agent_id in self.state_cache:
            return self.state_cache[agent_id]
        
        state_file = self.storage_dir / f"{agent_id}_state.json"
        if state_file.exists():
            async with aiofiles.open(state_file, 'r', encoding='utf-8') as f:
                state = json.loads(await f.read())
                self.state_cache[agent_id] = state
                return state
        
        return {}
    
    async def save_memory(self, agent_id: str, memory: Memory):
        """Save memory to persistent storage"""
        if agent_id not in self.memory_cache:
            self.memory_cache[agent_id] = {}
        
        self.memory_cache[agent_id][memory.id] = memory
        
        # Save to file for persistence
        memory_file = self.storage_dir / f"{agent_id}_memories.pkl"
        async with aiofiles.open(memory_file, 'wb') as f:
            await f.write(pickle.dumps(self.memory_cache[agent_id]))
    
    async def load_memories(self, agent_id: str, layer: Optional[MemoryLayer] = None) -> List[Memory]:
        """Load memories from persistent storage"""
        if agent_id not in self.memory_cache:
            memory_file = self.storage_dir / f"{agent_id}_memories.pkl"
            if memory_file.exists():
                async with aiofiles.open(memory_file, 'rb') as f:
                    self.memory_cache[agent_id] = pickle.loads(await f.read())
            else:
                self.memory_cache[agent_id] = {}
        
        memories = list(self.memory_cache[agent_id].values())
        
        if layer:
            memories = [m for m in memories if m.layer == layer]
        
        return sorted(memories, key=lambda m: m.last_accessed, reverse=True)
    
    async def cleanup_expired_memories(self, agent_id: str):
        """Clean up expired memories"""
        if agent_id not in self.memory_cache:
            return
        
        now = datetime.now()
        expired_ids = []
        
        for memory_id, memory in self.memory_cache[agent_id].items():
            if memory.expires_at and memory.expires_at < now:
                expired_ids.append(memory_id)
        
        for memory_id in expired_ids:
            del self.memory_cache[agent_id][memory_id]
        
        if expired_ids:
            # Save updated memories
            memory_file = self.storage_dir / f"{agent_id}_memories.pkl"
            async with aiofiles.open(memory_file, 'wb') as f:
                await f.write(pickle.dumps(self.memory_cache[agent_id]))


class CommunicationHub:
    """Central communication system for all agents"""
    
    def __init__(self):
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.subscribers: Dict[str, Set[Callable]] = {}
        self.message_history: List[Message] = []
        self.active_conversations: Dict[str, List[Message]] = {}
        
    async def send_message(self, message: Message):
        """Send a message through the system"""
        await self.message_queue.put(message)
        self.message_history.append(message)
        
        # Notify subscribers
        if message.type.value in self.subscribers:
            for callback in self.subscribers[message.type.value]:
                try:
                    await callback(message)
                except Exception as e:
                    print(f"Error in message callback: {e}")
    
    def subscribe(self, message_type: str, callback: Callable):
        """Subscribe to messages of a specific type"""
        if message_type not in self.subscribers:
            self.subscribers[message_type] = set()
        self.subscribers[message_type].add(callback)
    
    def unsubscribe(self, message_type: str, callback: Callable):
        """Unsubscribe from messages"""
        if message_type in self.subscribers:
            self.subscribers[message_type].discard(callback)
    
    async def get_messages(self, recipient: str, limit: int = 10) -> List[Message]:
        """Get messages for a specific recipient"""
        messages = [
            msg for msg in self.message_history[-100:]  # Last 100 messages
            if msg.recipient == recipient or msg.recipient is None
        ]
        return messages[-limit:]
    
    async def start_conversation(self, participants: List[str]) -> str:
        """Start a new conversation"""
        conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
        self.active_conversations[conversation_id] = []
        return conversation_id
    
    async def add_to_conversation(self, conversation_id: str, message: Message):
        """Add message to a conversation"""
        if conversation_id in self.active_conversations:
            self.active_conversations[conversation_id].append(message)


class PluginInterface(ABC):
    """Interface for all plugins"""
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]):
        """Initialize the plugin"""
        pass
    
    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the plugin functionality"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities this plugin provides"""
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get plugin metadata"""
        pass


class PluginLoader:
    """Dynamic plugin loading system"""
    
    def __init__(self, plugins_dir: str = "plugins"):
        self.plugins_dir = Path(plugins_dir)
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_plugins: Dict[str, PluginInterface] = {}
        self.plugin_metadata: Dict[str, Dict[str, Any]] = {}
    
    async def load_plugin(self, plugin_name: str) -> Optional[PluginInterface]:
        """Load a plugin dynamically"""
        # This would implement dynamic plugin loading
        # For now, return None
        return None
    
    async def generate_plugin(self, requirement: str, context: Dict[str, Any]) -> str:
        """Generate a plugin based on requirements"""
        # This is where the system becomes truly adaptive
        # Generate plugin code based on requirements
        plugin_code = f"""
# Auto-generated plugin for: {requirement}
# Generated at: {datetime.now().isoformat()}

from src.core.minimal_core import PluginInterface
from typing import Dict, Any, List

class GeneratedPlugin(PluginInterface):
    async def initialize(self, config: Dict[str, Any]):
        self.config = config
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Implementation based on requirement: {requirement}
        return {{"status": "success", "result": "Plugin executed"}}
    
    def get_capabilities(self) -> List[str]:
        return ["{requirement}"]
    
    def get_metadata(self) -> Dict[str, Any]:
        return {{
            "name": "generated_plugin",
            "version": "1.0",
            "description": "Auto-generated plugin for {requirement}",
            "generated_at": "{datetime.now().isoformat()}"
        }}
"""
        
        plugin_file = self.plugins_dir / f"generated_{uuid.uuid4().hex[:8]}.py"
        async with aiofiles.open(plugin_file, 'w', encoding='utf-8') as f:
            await f.write(plugin_code)
        
        return str(plugin_file)
    
    async def save_generated_plugin(self, plugin_code: str, plugin_name: str) -> str:
        """Save generated plugin code to file"""
        plugin_file = self.plugins_dir / f"{plugin_name}.py"
        async with aiofiles.open(plugin_file, 'w', encoding='utf-8') as f:
            await f.write(plugin_code)
        
        return str(plugin_file)


class MinimalAgent:
    """Minimal agent implementation - maximum flexibility through prompts"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any], 
                 communication_hub: CommunicationHub,
                 state_manager: StateManager,
                 prompt_engine: PromptEngine,
                 plugin_loader: PluginLoader):
        self.agent_id = agent_id
        self.config = config
        self.communication_hub = communication_hub
        self.state_manager = state_manager
        self.prompt_engine = prompt_engine
        self.plugin_loader = plugin_loader
        
        self.role = config.get('role', 'general')
        self.capabilities = config.get('capabilities', [])
        self.active = False
        self.execution_context: Dict[str, Any] = {}
        
        # Subscribe to messages
        self.communication_hub.subscribe("task", self._handle_task)
        self.communication_hub.subscribe("system", self._handle_system_message)
    
    async def initialize(self):
        """Initialize the agent"""
        # Load persistent state
        state = await self.state_manager.load_agent_state(self.agent_id)
        self.execution_context.update(state)
        
        # Initialize with prompt
        prompt = await self.prompt_engine.generate_prompt(
            "agent_initialization",
            agent_id=self.agent_id,
            role=self.role,
            capabilities=self.capabilities,
            context=self.execution_context,
            available_actions=self._get_available_actions(),
            recent_memories=await self._get_recent_memories()
        )
        
        # Process initialization prompt (would use LLM here)
        self.active = True
        
        # Save state
        await self._save_state()
    
    async def _handle_task(self, message: Message):
        """Handle incoming task"""
        if message.recipient != self.agent_id:
            return
        
        # Generate task processing prompt
        prompt = await self.prompt_engine.generate_prompt(
            "task_processing",
            task_id=message.id,
            task_type=message.metadata.get('task_type', 'unknown'),
            description=message.content,
            context=message.metadata,
            priority=message.priority,
            role=self.role,
            capabilities=self.capabilities,
            recent_memories=await self._get_recent_memories(),
            available_tools=self._get_available_tools()
        )
        
        # Process with LLM (placeholder)
        result = await self._process_with_llm(prompt)
        
        # Send response
        response = Message(
            id=f"resp_{uuid.uuid4().hex[:8]}",
            type=MessageType.RESPONSE,
            sender=self.agent_id,
            recipient=message.sender,
            content=result,
            metadata={"original_task": message.id},
            timestamp=datetime.now()
        )
        
        await self.communication_hub.send_message(response)
        
        # Store memory
        await self._store_memory(
            content=f"Processed task: {message.content}",
            layer=MemoryLayer.EPISODIC,
            importance=0.7,
            metadata={"task_id": message.id, "result": result}
        )
    
    async def _handle_system_message(self, message: Message):
        """Handle system messages"""
        # Adaptive response to system messages
        prompt = await self.prompt_engine.generate_prompt(
            "adaptive_response",
            situation=message.content,
            current_state=self.execution_context,
            options=self._get_available_actions(),
            history=await self._get_recent_memories()
        )
        
        # Process and adapt
        await self._process_with_llm(prompt)
    
    async def _process_with_llm(self, prompt: str) -> str:
        """Process prompt with LLM - placeholder for actual LLM integration"""
        # This would integrate with the actual LLM client
        return f"Processed: {prompt[:100]}..."
    
    async def _get_recent_memories(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get recent memories"""
        memories = await self.state_manager.load_memories(self.agent_id)
        return [memory.to_dict() for memory in memories[:count]]
    
    def _get_available_actions(self) -> List[str]:
        """Get available actions for this agent"""
        return [
            "send_message",
            "store_memory",
            "load_plugin",
            "generate_plugin",
            "collaborate",
            "adapt_behavior"
        ]
    
    def _get_available_tools(self) -> List[str]:
        """Get available tools"""
        return list(self.plugin_loader.loaded_plugins.keys())
    
    async def _store_memory(self, content: Any, layer: MemoryLayer, 
                          importance: float, metadata: Dict[str, Any] = None):
        """Store a memory"""
        memory = Memory(
            id=f"mem_{uuid.uuid4().hex[:8]}",
            layer=layer,
            content=content,
            metadata=metadata or {},
            importance=importance,
            access_count=0,
            last_accessed=datetime.now(),
            created_at=datetime.now(),
            expires_at=self._calculate_expiry(layer, importance)
        )
        
        await self.state_manager.save_memory(self.agent_id, memory)
    
    def _calculate_expiry(self, layer: MemoryLayer, importance: float) -> Optional[datetime]:
        """Calculate when memory should expire"""
        if layer == MemoryLayer.WORKING:
            return datetime.now() + timedelta(hours=1)
        elif layer == MemoryLayer.SESSION:
            return datetime.now() + timedelta(days=1)
        elif importance < 0.3:
            return datetime.now() + timedelta(days=7)
        else:
            return None  # Never expires
    
    async def _save_state(self):
        """Save current state"""
        await self.state_manager.save_agent_state(self.agent_id, self.execution_context)
    
    async def shutdown(self):
        """Shutdown the agent gracefully"""
        await self._save_state()
        self.active = False


class MinimalCore:
    """The minimal core system that orchestrates everything"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.communication_hub = CommunicationHub()
        self.state_manager = StateManager(config.get('storage_dir', 'state'))
        self.prompt_engine = PromptEngine(config.get('templates_dir', 'templates'))
        self.plugin_loader = PluginLoader(config.get('plugins_dir', 'plugins'))
        
        self.agents: Dict[str, MinimalAgent] = {}
        self.system_id = f"core_{uuid.uuid4().hex[:8]}"
        self.active = False
    
    async def initialize(self):
        """Initialize the core system"""
        await self.prompt_engine.initialize()
        self.active = True
        
        # Start message processing
        asyncio.create_task(self._process_messages())
        
        # Start memory consolidation
        asyncio.create_task(self._consolidate_memories())
    
    async def create_agent(self, agent_config: Dict[str, Any]) -> str:
        """Create a new agent"""
        agent_id = f"agent_{uuid.uuid4().hex[:8]}"
        
        agent = MinimalAgent(
            agent_id=agent_id,
            config=agent_config,
            communication_hub=self.communication_hub,
            state_manager=self.state_manager,
            prompt_engine=self.prompt_engine,
            plugin_loader=self.plugin_loader
        )
        
        await agent.initialize()
        self.agents[agent_id] = agent
        
        return agent_id
    
    async def send_task(self, agent_id: str, task: str, metadata: Dict[str, Any] = None) -> str:
        """Send a task to an agent"""
        message = Message(
            id=f"task_{uuid.uuid4().hex[:8]}",
            type=MessageType.TASK,
            sender=self.system_id,
            recipient=agent_id,
            content=task,
            metadata=metadata or {},
            timestamp=datetime.now(),
            priority=5
        )
        
        await self.communication_hub.send_message(message)
        return message.id
    
    async def _process_messages(self):
        """Process messages continuously"""
        while self.active:
            try:
                # This would process the message queue
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Error processing messages: {e}")
    
    async def _consolidate_memories(self):
        """Consolidate memories periodically"""
        while self.active:
            try:
                for agent_id in self.agents:
                    await self.state_manager.cleanup_expired_memories(agent_id)
                    
                    # Generate consolidation prompt
                    memories = await self.state_manager.load_memories(agent_id)
                    if len(memories) > 50:  # Consolidate when too many memories
                        prompt = await self.prompt_engine.generate_prompt(
                            "memory_consolidation",
                            memories=[m.to_dict() for m in memories[:20]],
                            working_count=len([m for m in memories if m.layer == MemoryLayer.WORKING]),
                            session_count=len([m for m in memories if m.layer == MemoryLayer.SESSION]),
                            episodic_count=len([m for m in memories if m.layer == MemoryLayer.EPISODIC]),
                            semantic_count=len([m for m in memories if m.layer == MemoryLayer.SEMANTIC])
                        )
                        
                        # Process consolidation (would use LLM)
                
                await asyncio.sleep(300)  # Every 5 minutes
            except Exception as e:
                print(f"Error consolidating memories: {e}")
    
    async def shutdown(self):
        """Shutdown the system gracefully"""
        self.active = False
        
        for agent in self.agents.values():
            await agent.shutdown()
        
        print("Minimal core system shutdown complete")