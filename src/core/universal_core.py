"""
Universal Core System - The ultimate flexible AI agent architecture
Minimizes hardcoding, maximizes adaptability through prompt-driven design
Achieves infinite flexibility, scalability, and intelligence through universal patterns
"""

import asyncio
import json
import uuid
import pickle
import hashlib
import importlib
import inspect
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Set, Type, AsyncGenerator
from pathlib import Path
from dataclasses import dataclass, asdict, field
from enum import Enum, auto
from collections import defaultdict, deque
import aiofiles
import weakref


class UniversalType(Enum):
    """Universal types for maximum flexibility"""
    AGENT = auto()
    TASK = auto()
    MESSAGE = auto()
    MEMORY = auto()
    PLUGIN = auto()
    PROMPT = auto()
    STATE = auto()
    CONTEXT = auto()
    CAPABILITY = auto()
    PATTERN = auto()


class Priority(Enum):
    """Universal priority system"""
    CRITICAL = 10
    HIGH = 8
    NORMAL = 5
    LOW = 3
    BACKGROUND = 1


class MemoryType(Enum):
    """Hierarchical memory types"""
    WORKING = "working"      # Immediate context
    EPISODIC = "episodic"    # Specific experiences
    SEMANTIC = "semantic"    # General knowledge
    PROCEDURAL = "procedural" # How-to knowledge
    META = "meta"           # Self-awareness
    COLLECTIVE = "collective" # Shared knowledge


@dataclass
class UniversalEntity:
    """Universal entity that can represent anything in the system"""
    id: str = field(default_factory=lambda: f"entity_{uuid.uuid4().hex[:8]}")
    type: UniversalType = UniversalType.AGENT
    name: str = ""
    description: str = ""
    content: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    capabilities: Set[str] = field(default_factory=set)
    relationships: Dict[str, Set[str]] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    priority: Priority = Priority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    importance: float = 0.5
    expires_at: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.name:
            self.name = f"{self.type.name.lower()}_{self.id}"
    
    def update(self, **kwargs):
        """Update entity with new data"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.now()
        self.access_count += 1
        self.accessed_at = datetime.now()
    
    def add_capability(self, capability: str):
        """Add a capability"""
        self.capabilities.add(capability)
        self.updated_at = datetime.now()
    
    def add_relationship(self, relation_type: str, entity_id: str):
        """Add a relationship to another entity"""
        if relation_type not in self.relationships:
            self.relationships[relation_type] = set()
        self.relationships[relation_type].add(entity_id)
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            **asdict(self),
            'type': self.type.name,
            'priority': self.priority.name,
            'capabilities': list(self.capabilities),
            'relationships': {k: list(v) for k, v in self.relationships.items()},
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'accessed_at': self.accessed_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UniversalEntity':
        """Create from dictionary"""
        data['type'] = UniversalType[data['type']]
        data['priority'] = Priority[data['priority']]
        data['capabilities'] = set(data.get('capabilities', []))
        data['relationships'] = {k: set(v) for k, v in data.get('relationships', {}).items()}
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        data['accessed_at'] = datetime.fromisoformat(data['accessed_at'])
        if data.get('expires_at'):
            data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        return cls(**data)


class UniversalPromptEngine:
    """Universal prompt engine that generates any prompt dynamically"""
    
    def __init__(self, templates_dir: str = "prompts"):
        self.templates_dir = Path(templates_dir)
        self.templates: Dict[str, str] = {}
        self.template_patterns: Dict[str, Dict[str, Any]] = {}
        self.generation_history: List[Dict[str, Any]] = []
        self.optimization_data: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
    async def initialize(self):
        """Initialize the prompt engine"""
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        await self._load_templates()
        await self._initialize_meta_prompts()
    
    async def _load_templates(self):
        """Load existing templates"""
        for template_file in self.templates_dir.glob("*.txt"):
            template_name = template_file.stem
            async with aiofiles.open(template_file, 'r', encoding='utf-8') as f:
                self.templates[template_name] = await f.read()
    
    async def _initialize_meta_prompts(self):
        """Initialize meta-prompts for self-generation"""
        meta_prompts = {
            "generate_prompt": """You are a universal prompt generator. Create a prompt for the following purpose:

Purpose: {purpose}
Context: {context}
Target Entity Type: {entity_type}
Required Capabilities: {capabilities}
Expected Output Format: {output_format}
Constraints: {constraints}

The prompt should be:
1. Clear and specific
2. Adaptable to different contexts
3. Optimized for the target entity type
4. Include all necessary parameters as {{parameter_name}}
5. Follow best practices for AI interaction

Generate the prompt:""",

            "optimize_prompt": """You are a prompt optimizer. Improve the following prompt based on usage data:

Original Prompt: {original_prompt}
Usage Statistics: {usage_stats}
Performance Metrics: {performance_metrics}
Common Issues: {issues}
Success Patterns: {success_patterns}

Optimize the prompt to:
1. Improve clarity and effectiveness
2. Reduce ambiguity
3. Enhance parameter handling
4. Better align with successful patterns
5. Address common issues

Optimized Prompt:""",

            "adapt_prompt": """You are a prompt adapter. Modify the following prompt for a new context:

Base Prompt: {base_prompt}
New Context: {new_context}
Target Changes: {target_changes}
Constraints: {constraints}
Adaptation Requirements: {requirements}

Adapt the prompt to:
1. Fit the new context perfectly
2. Maintain core functionality
3. Incorporate required changes
4. Respect all constraints
5. Preserve successful elements

Adapted Prompt:""",

            "meta_analyze": """You are a meta-analyzer for prompt systems. Analyze the following prompt ecosystem:

System State: {system_state}
Prompt Usage Patterns: {usage_patterns}
Performance Data: {performance_data}
Entity Interactions: {interactions}
Emerging Patterns: {patterns}

Provide insights on:
1. System effectiveness
2. Optimization opportunities
3. Emerging needs
4. Pattern recognition
5. Recommended improvements

Analysis:"""
        }
        
        for name, prompt in meta_prompts.items():
            if name not in self.templates:
                self.templates[name] = prompt
                await self._save_template(name, prompt)
    
    async def _save_template(self, name: str, template: str):
        """Save template to file"""
        template_file = self.templates_dir / f"{name}.txt"
        async with aiofiles.open(template_file, 'w', encoding='utf-8') as f:
            await f.write(template)
    
    async def generate_prompt(self, purpose: str, context: Dict[str, Any], 
                            llm_client=None) -> str:
        """Generate a prompt dynamically"""
        # Check if we have a suitable existing template
        template_name = self._find_suitable_template(purpose, context)
        
        if template_name and template_name in self.templates:
            return await self._apply_template(template_name, context)
        
        # Generate new prompt using meta-prompt
        if llm_client:
            new_prompt = await self._generate_new_prompt(purpose, context, llm_client)
            if new_prompt:
                # Cache the new prompt
                prompt_name = self._generate_prompt_name(purpose)
                self.templates[prompt_name] = new_prompt
                await self._save_template(prompt_name, new_prompt)
                return await self._apply_template(prompt_name, context)
        
        # Fallback to basic template
        return await self._create_fallback_prompt(purpose, context)
    
    def _find_suitable_template(self, purpose: str, context: Dict[str, Any]) -> Optional[str]:
        """Find the most suitable existing template"""
        purpose_words = set(purpose.lower().split())
        best_match = None
        best_score = 0
        
        for template_name in self.templates.keys():
            template_words = set(template_name.lower().replace('_', ' ').split())
            score = len(purpose_words.intersection(template_words))
            if score > best_score:
                best_score = score
                best_match = template_name
        
        return best_match if best_score > 0 else None
    
    async def _generate_new_prompt(self, purpose: str, context: Dict[str, Any], 
                                 llm_client) -> Optional[str]:
        """Generate a new prompt using LLM"""
        try:
            meta_prompt = self.templates.get("generate_prompt", "")
            if not meta_prompt:
                return None
            
            prompt_context = {
                "purpose": purpose,
                "context": json.dumps(context, indent=2),
                "entity_type": context.get("entity_type", "unknown"),
                "capabilities": ", ".join(context.get("capabilities", [])),
                "output_format": context.get("output_format", "structured response"),
                "constraints": ", ".join(context.get("constraints", []))
            }
            
            filled_prompt = meta_prompt.format(**prompt_context)
            response = await llm_client.generate(filled_prompt, "You are a helpful assistant.")
            
            # Track generation
            self.generation_history.append({
                "purpose": purpose,
                "context": context,
                "generated_at": datetime.now().isoformat(),
                "success": bool(response)
            })
            
            return response
        except Exception as e:
            print(f"Error generating prompt: {e}")
            return None
    
    async def _apply_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """Apply template with context"""
        template = self.templates.get(template_name, "")
        if not template:
            return f"Template '{template_name}' not found"
        
        try:
            # Extract parameters from template
            import re
            parameters = set(re.findall(r'\{(\w+)\}', template))
            
            # Fill in available parameters
            filled_context = {}
            for param in parameters:
                if param in context:
                    filled_context[param] = context[param]
                else:
                    filled_context[param] = f"[{param}]"  # Placeholder
            
            return template.format(**filled_context)
        except Exception as e:
            return f"Error applying template: {e}"
    
    async def _create_fallback_prompt(self, purpose: str, context: Dict[str, Any]) -> str:
        """Create a basic fallback prompt"""
        return f"""Task: {purpose}

Context: {json.dumps(context, indent=2)}

Please process this request appropriately and provide a structured response."""
    
    def _generate_prompt_name(self, purpose: str) -> str:
        """Generate a name for a new prompt"""
        # Create a hash-based name
        purpose_hash = hashlib.md5(purpose.encode()).hexdigest()[:8]
        clean_purpose = "".join(c for c in purpose if c.isalnum() or c in " _").strip()
        clean_purpose = "_".join(clean_purpose.lower().split())[:30]
        return f"{clean_purpose}_{purpose_hash}"
    
    async def optimize_prompt(self, template_name: str, usage_data: Dict[str, Any], 
                            llm_client) -> bool:
        """Optimize an existing prompt based on usage data"""
        if template_name not in self.templates or not llm_client:
            return False
        
        try:
            optimize_prompt = self.templates.get("optimize_prompt", "")
            if not optimize_prompt:
                return False
            
            context = {
                "original_prompt": self.templates[template_name],
                "usage_stats": json.dumps(usage_data.get("stats", {})),
                "performance_metrics": json.dumps(usage_data.get("performance", {})),
                "issues": ", ".join(usage_data.get("issues", [])),
                "success_patterns": ", ".join(usage_data.get("success_patterns", []))
            }
            
            filled_prompt = optimize_prompt.format(**context)
            optimized = await llm_client.generate(filled_prompt, "You are a helpful assistant.")
            
            if optimized:
                # Backup original
                backup_name = f"{template_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.templates[backup_name] = self.templates[template_name]
                
                # Update with optimized version
                self.templates[template_name] = optimized
                await self._save_template(template_name, optimized)
                return True
        
        except Exception as e:
            print(f"Error optimizing prompt: {e}")
        
        return False


class UniversalMemorySystem:
    """Universal memory system with infinite context and no memory loss"""
    
    def __init__(self, storage_dir: str = "memory"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Multi-layer memory storage
        self.memory_layers: Dict[MemoryType, Dict[str, UniversalEntity]] = {
            layer: {} for layer in MemoryType
        }
        
        # Memory indices for fast retrieval
        self.content_index: Dict[str, Set[str]] = defaultdict(set)
        self.temporal_index: Dict[str, List[str]] = defaultdict(list)
        self.importance_index: Dict[float, Set[str]] = defaultdict(set)
        self.relationship_index: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        
        # Memory consolidation queue
        self.consolidation_queue: asyncio.Queue = asyncio.Queue()
        self.consolidation_task: Optional[asyncio.Task] = None
        
        # Memory statistics
        self.access_patterns: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.consolidation_history: List[Dict[str, Any]] = []
    
    async def initialize(self):
        """Initialize the memory system"""
        await self._load_persistent_memories()
        self.consolidation_task = asyncio.create_task(self._consolidation_worker())
    
    async def store_memory(self, entity: UniversalEntity, memory_type: MemoryType = MemoryType.WORKING):
        """Store a memory in the appropriate layer"""
        entity.type = UniversalType.MEMORY
        entity.metadata["memory_type"] = memory_type.value
        
        # Store in appropriate layer
        self.memory_layers[memory_type][entity.id] = entity
        
        # Update indices
        await self._update_indices(entity)
        
        # Queue for consolidation if needed
        if memory_type == MemoryType.WORKING:
            await self.consolidation_queue.put(entity.id)
        
        # Persist if important
        if entity.importance > 0.7 or memory_type in [MemoryType.SEMANTIC, MemoryType.META]:
            await self._persist_memory(entity)
    
    async def retrieve_memories(self, query: str, context: Dict[str, Any] = None, 
                              limit: int = 10) -> List[UniversalEntity]:
        """Retrieve relevant memories using universal search"""
        # Multi-dimensional search
        candidates = set()
        
        # Content-based search
        query_words = set(query.lower().split())
        for word in query_words:
            if word in self.content_index:
                candidates.update(self.content_index[word])
        
        # Context-based search
        if context:
            for key, value in context.items():
                if isinstance(value, str):
                    value_words = set(value.lower().split())
                    for word in value_words:
                        if word in self.content_index:
                            candidates.update(self.content_index[word])
        
        # Collect candidate memories
        memories = []
        for memory_id in candidates:
            for layer in self.memory_layers.values():
                if memory_id in layer:
                    memories.append(layer[memory_id])
                    break
        
        # Score and rank memories
        scored_memories = []
        for memory in memories:
            score = await self._calculate_relevance_score(memory, query, context)
            scored_memories.append((score, memory))
        
        # Sort by score and return top results
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in scored_memories[:limit]]
    
    async def _calculate_relevance_score(self, memory: UniversalEntity, query: str, 
                                       context: Dict[str, Any] = None) -> float:
        """Calculate relevance score for a memory"""
        score = 0.0
        
        # Content similarity
        if memory.content:
            content_str = str(memory.content).lower()
            query_words = set(query.lower().split())
            content_words = set(content_str.split())
            if content_words:
                score += len(query_words.intersection(content_words)) / len(content_words)
        
        # Importance weight
        score += memory.importance * 0.3
        
        # Recency weight
        time_diff = (datetime.now() - memory.accessed_at).total_seconds()
        recency_score = max(0, 1 - time_diff / (24 * 3600))  # Decay over 24 hours
        score += recency_score * 0.2
        
        # Access frequency weight
        score += min(memory.access_count / 100, 0.2)
        
        # Context relevance
        if context:
            context_score = 0
            for key, value in context.items():
                if key in memory.metadata and memory.metadata[key] == value:
                    context_score += 0.1
            score += context_score
        
        return score
    
    async def _update_indices(self, entity: UniversalEntity):
        """Update all memory indices"""
        # Content index
        if entity.content:
            content_words = str(entity.content).lower().split()
            for word in content_words:
                self.content_index[word].add(entity.id)
        
        # Temporal index
        date_key = entity.created_at.strftime("%Y-%m-%d")
        self.temporal_index[date_key].append(entity.id)
        
        # Importance index
        self.importance_index[entity.importance].add(entity.id)
        
        # Relationship index
        for relation_type, related_ids in entity.relationships.items():
            for related_id in related_ids:
                self.relationship_index[entity.id][relation_type].add(related_id)
                self.relationship_index[related_id][f"inverse_{relation_type}"].add(entity.id)
    
    async def _consolidation_worker(self):
        """Background worker for memory consolidation"""
        while True:
            try:
                memory_id = await self.consolidation_queue.get()
                await self._consolidate_memory(memory_id)
                self.consolidation_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in memory consolidation: {e}")
    
    async def _consolidate_memory(self, memory_id: str):
        """Consolidate a memory from working to appropriate long-term layer"""
        working_memory = self.memory_layers[MemoryType.WORKING].get(memory_id)
        if not working_memory:
            return
        
        # Determine target layer based on importance and content
        target_layer = await self._determine_target_layer(working_memory)
        
        if target_layer != MemoryType.WORKING:
            # Move to target layer
            del self.memory_layers[MemoryType.WORKING][memory_id]
            working_memory.metadata["memory_type"] = target_layer.value
            self.memory_layers[target_layer][memory_id] = working_memory
            
            # Update importance based on consolidation
            working_memory.importance = min(1.0, working_memory.importance + 0.1)
            
            # Persist if important
            if working_memory.importance > 0.5:
                await self._persist_memory(working_memory)
            
            # Record consolidation
            self.consolidation_history.append({
                "memory_id": memory_id,
                "from_layer": MemoryType.WORKING.value,
                "to_layer": target_layer.value,
                "timestamp": datetime.now().isoformat(),
                "importance": working_memory.importance
            })
    
    async def _determine_target_layer(self, memory: UniversalEntity) -> MemoryType:
        """Determine the appropriate memory layer for consolidation"""
        # High importance -> Semantic
        if memory.importance > 0.8:
            return MemoryType.SEMANTIC
        
        # Meta information -> Meta
        if "meta" in str(memory.content).lower() or "self" in memory.metadata:
            return MemoryType.META
        
        # Procedural knowledge -> Procedural
        if any(word in str(memory.content).lower() for word in ["how", "step", "process", "method"]):
            return MemoryType.PROCEDURAL
        
        # Shared knowledge -> Collective
        if memory.metadata.get("shared", False):
            return MemoryType.COLLECTIVE
        
        # Default -> Episodic
        return MemoryType.EPISODIC
    
    async def _persist_memory(self, memory: UniversalEntity):
        """Persist memory to disk"""
        memory_file = self.storage_dir / f"{memory.id}.json"
        async with aiofiles.open(memory_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(memory.to_dict(), indent=2))
    
    async def _load_persistent_memories(self):
        """Load persistent memories from disk"""
        for memory_file in self.storage_dir.glob("*.json"):
            try:
                async with aiofiles.open(memory_file, 'r', encoding='utf-8') as f:
                    data = json.loads(await f.read())
                    memory = UniversalEntity.from_dict(data)
                    memory_type = MemoryType(memory.metadata.get("memory_type", "episodic"))
                    self.memory_layers[memory_type][memory.id] = memory
                    await self._update_indices(memory)
            except Exception as e:
                print(f"Error loading memory {memory_file}: {e}")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.consolidation_task:
            self.consolidation_task.cancel()
            try:
                await self.consolidation_task
            except asyncio.CancelledError:
                pass


class UniversalCommunicationHub:
    """Universal communication system for infinite scalability"""
    
    def __init__(self):
        self.message_queues: Dict[str, asyncio.Queue] = defaultdict(lambda: asyncio.Queue())
        self.subscribers: Dict[str, Set[Callable]] = defaultdict(set)
        self.message_history: deque = deque(maxlen=10000)
        self.active_conversations: Dict[str, List[UniversalEntity]] = {}
        self.routing_rules: Dict[str, Callable] = {}
        self.middleware: List[Callable] = []
        
        # Communication patterns
        self.patterns: Dict[str, Dict[str, Any]] = {}
        self.statistics: Dict[str, Dict[str, Any]] = defaultdict(dict)
    
    async def send_message(self, message: UniversalEntity):
        """Send a message through the universal communication system"""
        message.type = UniversalType.MESSAGE
        message.accessed_at = datetime.now()
        
        # Apply middleware
        for middleware_func in self.middleware:
            message = await middleware_func(message)
            if not message:  # Middleware can filter messages
                return
        
        # Route message
        recipients = await self._route_message(message)
        
        # Deliver to recipients
        for recipient_id in recipients:
            await self.message_queues[recipient_id].put(message)
        
        # Store in history
        self.message_history.append(message)
        
        # Notify subscribers
        await self._notify_subscribers(message)
        
        # Update statistics
        self._update_statistics(message, recipients)
    
    async def _route_message(self, message: UniversalEntity) -> Set[str]:
        """Route message to appropriate recipients"""
        recipients = set()
        
        # Direct recipient
        if "recipient" in message.metadata:
            recipients.add(message.metadata["recipient"])
        
        # Broadcast
        if message.metadata.get("broadcast", False):
            recipients.update(self.message_queues.keys())
        
        # Custom routing rules
        for rule_name, rule_func in self.routing_rules.items():
            try:
                rule_recipients = await rule_func(message)
                if rule_recipients:
                    recipients.update(rule_recipients)
            except Exception as e:
                print(f"Error in routing rule {rule_name}: {e}")
        
        # Pattern-based routing
        pattern_recipients = await self._pattern_based_routing(message)
        recipients.update(pattern_recipients)
        
        return recipients
    
    async def _pattern_based_routing(self, message: UniversalEntity) -> Set[str]:
        """Route based on learned communication patterns"""
        recipients = set()
        
        # Analyze message content for patterns
        content_str = str(message.content).lower()
        
        for pattern_name, pattern_data in self.patterns.items():
            if self._matches_pattern(content_str, pattern_data):
                recipients.update(pattern_data.get("typical_recipients", []))
        
        return recipients
    
    def _matches_pattern(self, content: str, pattern_data: Dict[str, Any]) -> bool:
        """Check if content matches a communication pattern"""
        keywords = pattern_data.get("keywords", [])
        return any(keyword in content for keyword in keywords)
    
    async def _notify_subscribers(self, message: UniversalEntity):
        """Notify all subscribers of the message"""
        message_type = message.metadata.get("message_type", "general")
        
        for callback in self.subscribers[message_type]:
            try:
                await callback(message)
            except Exception as e:
                print(f"Error in subscriber callback: {e}")
    
    def _update_statistics(self, message: UniversalEntity, recipients: Set[str]):
        """Update communication statistics"""
        sender = message.metadata.get("sender", "unknown")
        
        if sender not in self.statistics:
            self.statistics[sender] = {
                "messages_sent": 0,
                "recipients_reached": set(),
                "message_types": defaultdict(int),
                "last_activity": None
            }
        
        self.statistics[sender]["messages_sent"] += 1
        self.statistics[sender]["recipients_reached"].update(recipients)
        self.statistics[sender]["last_activity"] = datetime.now().isoformat()
        
        message_type = message.metadata.get("message_type", "general")
        self.statistics[sender]["message_types"][message_type] += 1
    
    def subscribe(self, message_type: str, callback: Callable):
        """Subscribe to messages of a specific type"""
        self.subscribers[message_type].add(callback)
    
    def unsubscribe(self, message_type: str, callback: Callable):
        """Unsubscribe from messages"""
        self.subscribers[message_type].discard(callback)
    
    def add_routing_rule(self, name: str, rule_func: Callable):
        """Add a custom routing rule"""
        self.routing_rules[name] = rule_func
    
    def add_middleware(self, middleware_func: Callable):
        """Add middleware for message processing"""
        self.middleware.append(middleware_func)
    
    async def get_messages(self, recipient_id: str, limit: int = 10) -> List[UniversalEntity]:
        """Get messages for a specific recipient"""
        messages = []
        queue = self.message_queues[recipient_id]
        
        for _ in range(min(limit, queue.qsize())):
            try:
                message = queue.get_nowait()
                messages.append(message)
                queue.task_done()
            except asyncio.QueueEmpty:
                break
        
        return messages
    
    async def learn_communication_patterns(self):
        """Learn communication patterns from message history"""
        # Analyze recent messages for patterns
        recent_messages = list(self.message_history)[-1000:]  # Last 1000 messages
        
        # Group by sender-recipient pairs
        pairs = defaultdict(list)
        for message in recent_messages:
            sender = message.metadata.get("sender", "unknown")
            recipient = message.metadata.get("recipient", "broadcast")
            pairs[(sender, recipient)].append(message)
        
        # Extract patterns
        for (sender, recipient), messages in pairs.items():
            if len(messages) < 3:  # Need minimum messages to establish pattern
                continue
            
            pattern_name = f"{sender}_to_{recipient}"
            
            # Extract common keywords
            all_content = " ".join(str(msg.content) for msg in messages)
            words = all_content.lower().split()
            word_freq = defaultdict(int)
            for word in words:
                word_freq[word] += 1
            
            # Get most common words as keywords
            keywords = [word for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]]
            
            self.patterns[pattern_name] = {
                "sender": sender,
                "typical_recipients": [recipient],
                "keywords": keywords,
                "frequency": len(messages),
                "last_seen": messages[-1].accessed_at.isoformat()
            }


class UniversalPluginSystem:
    """Universal plugin system for infinite extensibility"""
    
    def __init__(self, plugins_dir: str = "plugins"):
        self.plugins_dir = Path(plugins_dir)
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        
        self.loaded_plugins: Dict[str, Any] = {}
        self.plugin_metadata: Dict[str, Dict[str, Any]] = {}
        self.plugin_capabilities: Dict[str, Set[str]] = defaultdict(set)
        self.plugin_dependencies: Dict[str, Set[str]] = defaultdict(set)
        
        # Dynamic plugin generation
        self.generation_templates: Dict[str, str] = {}
        self.generated_plugins: Dict[str, str] = {}
    
    async def initialize(self):
        """Initialize the plugin system"""
        await self._load_existing_plugins()
        await self._setup_generation_templates()
    
    async def _load_existing_plugins(self):
        """Load existing plugins from the plugins directory"""
        for plugin_file in self.plugins_dir.glob("*.py"):
            if plugin_file.name.startswith("__"):
                continue
            
            try:
                await self._load_plugin_file(plugin_file)
            except Exception as e:
                print(f"Error loading plugin {plugin_file}: {e}")
    
    async def _load_plugin_file(self, plugin_file: Path):
        """Load a single plugin file"""
        plugin_name = plugin_file.stem
        
        # Dynamic import
        spec = importlib.util.spec_from_file_location(plugin_name, plugin_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find plugin classes
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and hasattr(obj, 'get_capabilities'):
                plugin_instance = obj()
                await plugin_instance.initialize({})
                
                self.loaded_plugins[plugin_name] = plugin_instance
                self.plugin_capabilities[plugin_name] = set(plugin_instance.get_capabilities())
                
                if hasattr(plugin_instance, 'get_metadata'):
                    self.plugin_metadata[plugin_name] = plugin_instance.get_metadata()
    
    async def _setup_generation_templates(self):
        """Setup templates for dynamic plugin generation"""
        self.generation_templates = {
            "basic_plugin": '''"""
Dynamically generated plugin: {plugin_name}
Purpose: {purpose}
Generated at: {timestamp}
"""

import asyncio
from typing import Dict, List, Any

class {class_name}:
    """Dynamically generated plugin for {purpose}"""
    
    def __init__(self):
        self.name = "{plugin_name}"
        self.purpose = "{purpose}"
        self.capabilities = {capabilities}
        self.metadata = {metadata}
    
    async def initialize(self, config: Dict[str, Any]):
        """Initialize the plugin"""
        self.config = config
        return True
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the plugin functionality"""
        # Generated implementation
        {implementation}
        
        return {{"success": True, "result": result}}
    
    def get_capabilities(self) -> List[str]:
        """Get plugin capabilities"""
        return list(self.capabilities)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get plugin metadata"""
        return self.metadata
''',
            
            "advanced_plugin": '''"""
Advanced dynamically generated plugin: {plugin_name}
Purpose: {purpose}
Generated at: {timestamp}
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

class {class_name}:
    """Advanced dynamically generated plugin for {purpose}"""
    
    def __init__(self):
        self.name = "{plugin_name}"
        self.purpose = "{purpose}"
        self.capabilities = {capabilities}
        self.metadata = {metadata}
        self.state = {{}}
        self.history = []
        self.performance_metrics = {{}}
    
    async def initialize(self, config: Dict[str, Any]):
        """Initialize the plugin"""
        self.config = config
        self.state = {{"initialized_at": datetime.now().isoformat()}}
        return True
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the plugin functionality"""
        start_time = datetime.now()
        
        try:
            # Generated implementation
            {implementation}
            
            # Record execution
            execution_time = (datetime.now() - start_time).total_seconds()
            self.history.append({{
                "timestamp": start_time.isoformat(),
                "context": context,
                "result": result,
                "execution_time": execution_time,
                "success": True
            }})
            
            # Update performance metrics
            self._update_performance_metrics(execution_time, True)
            
            return {{"success": True, "result": result, "execution_time": execution_time}}
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.history.append({{
                "timestamp": start_time.isoformat(),
                "context": context,
                "error": str(e),
                "execution_time": execution_time,
                "success": False
            }})
            
            self._update_performance_metrics(execution_time, False)
            
            return {{"success": False, "error": str(e), "execution_time": execution_time}}
    
    def _update_performance_metrics(self, execution_time: float, success: bool):
        """Update performance metrics"""
        if "total_executions" not in self.performance_metrics:
            self.performance_metrics = {{
                "total_executions": 0,
                "successful_executions": 0,
                "total_execution_time": 0.0,
                "average_execution_time": 0.0,
                "success_rate": 0.0
            }}
        
        self.performance_metrics["total_executions"] += 1
        if success:
            self.performance_metrics["successful_executions"] += 1
        
        self.performance_metrics["total_execution_time"] += execution_time
        self.performance_metrics["average_execution_time"] = (
            self.performance_metrics["total_execution_time"] / 
            self.performance_metrics["total_executions"]
        )
        self.performance_metrics["success_rate"] = (
            self.performance_metrics["successful_executions"] / 
            self.performance_metrics["total_executions"]
        )
    
    def get_capabilities(self) -> List[str]:
        """Get plugin capabilities"""
        return list(self.capabilities)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get plugin metadata"""
        return {{
            **self.metadata,
            "state": self.state,
            "performance_metrics": self.performance_metrics,
            "history_length": len(self.history)
        }}
    
    def get_state(self) -> Dict[str, Any]:
        """Get plugin state"""
        return self.state
    
    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get execution history"""
        return self.history[-limit:]
'''
        }
    
    async def generate_plugin(self, requirement: str, context: Dict[str, Any], 
                            llm_client=None) -> Optional[str]:
        """Generate a plugin dynamically based on requirements"""
        plugin_name = self._generate_plugin_name(requirement)
        
        if llm_client:
            # Use LLM to generate implementation
            implementation = await self._generate_implementation(requirement, context, llm_client)
        else:
            # Use template-based generation
            implementation = await self._generate_template_implementation(requirement, context)
        
        if not implementation:
            return None
        
        # Create plugin code
        plugin_code = await self._create_plugin_code(plugin_name, requirement, implementation, context)
        
        # Save and load plugin
        plugin_file = self.plugins_dir / f"{plugin_name}.py"
        async with aiofiles.open(plugin_file, 'w', encoding='utf-8') as f:
            await f.write(plugin_code)
        
        # Load the new plugin
        try:
            await self._load_plugin_file(plugin_file)
            self.generated_plugins[plugin_name] = plugin_code
            return plugin_name
        except Exception as e:
            print(f"Error loading generated plugin: {e}")
            return None
    
    async def _generate_implementation(self, requirement: str, context: Dict[str, Any], 
                                     llm_client) -> Optional[str]:
        """Generate implementation using LLM"""
        prompt = f"""Generate Python code implementation for the following requirement:

Requirement: {requirement}
Context: {json.dumps(context, indent=2)}

The implementation should:
1. Be a complete function body that assigns result to a variable
2. Handle the specific requirement effectively
3. Use the provided context appropriately
4. Include proper error handling
5. Be efficient and well-structured

Generate only the implementation code (function body):"""
        
        try:
            response = await llm_client.generate(prompt, "You are a helpful coding assistant.")
            return response
        except Exception as e:
            print(f"Error generating implementation: {e}")
            return None
    
    async def _generate_template_implementation(self, requirement: str, 
                                             context: Dict[str, Any]) -> str:
        """Generate basic template implementation"""
        return f'''        # Template implementation for: {requirement}
        result = {{
            "message": "Plugin executed successfully",
            "requirement": "{requirement}",
            "context": context,
            "timestamp": datetime.now().isoformat()
        }}'''
    
    async def _create_plugin_code(self, plugin_name: str, requirement: str, 
                                implementation: str, context: Dict[str, Any]) -> str:
        """Create complete plugin code"""
        class_name = self._to_class_name(plugin_name)
        
        capabilities = context.get("capabilities", ["general"])
        metadata = {
            "name": plugin_name,
            "purpose": requirement,
            "generated": True,
            "generated_at": datetime.now().isoformat(),
            "version": "1.0"
        }
        
        template = self.generation_templates.get("advanced_plugin", self.generation_templates["basic_plugin"])
        
        return template.format(
            plugin_name=plugin_name,
            class_name=class_name,
            purpose=requirement,
            timestamp=datetime.now().isoformat(),
            capabilities=capabilities,
            metadata=metadata,
            implementation=implementation
        )
    
    def _generate_plugin_name(self, requirement: str) -> str:
        """Generate a plugin name from requirement"""
        # Clean and format requirement
        clean_req = "".join(c for c in requirement if c.isalnum() or c in " _").strip()
        words = clean_req.lower().split()[:3]  # Take first 3 words
        name = "_".join(words)
        
        # Add hash for uniqueness
        req_hash = hashlib.md5(requirement.encode()).hexdigest()[:6]
        return f"{name}_{req_hash}"
    
    def _to_class_name(self, plugin_name: str) -> str:
        """Convert plugin name to class name"""
        words = plugin_name.split("_")
        return "".join(word.capitalize() for word in words)
    
    async def execute_plugin(self, plugin_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a plugin"""
        if plugin_name not in self.loaded_plugins:
            return {"success": False, "error": f"Plugin '{plugin_name}' not found"}
        
        try:
            plugin = self.loaded_plugins[plugin_name]
            result = await plugin.execute(context)
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_available_plugins(self) -> Dict[str, Dict[str, Any]]:
        """Get all available plugins with their metadata"""
        return {
            name: {
                "capabilities": list(self.plugin_capabilities[name]),
                "metadata": self.plugin_metadata.get(name, {}),
                "generated": name in self.generated_plugins
            }
            for name in self.loaded_plugins.keys()
        }
    
    def find_plugins_by_capability(self, capability: str) -> List[str]:
        """Find plugins that provide a specific capability"""
        return [
            name for name, capabilities in self.plugin_capabilities.items()
            if capability in capabilities
        ]


class UniversalAgent:
    """Universal agent with infinite adaptability and intelligence"""
    
    def __init__(self, agent_id: str = None, config: Dict[str, Any] = None):
        self.id = agent_id or f"agent_{uuid.uuid4().hex[:8]}"
        self.config = config or {}
        
        # Core components
        self.entity = UniversalEntity(
            id=self.id,
            type=UniversalType.AGENT,
            name=self.config.get("name", f"Agent-{self.id}"),
            description=self.config.get("description", "Universal adaptive agent"),
            capabilities=set(self.config.get("capabilities", ["general"])),
            state={"status": "initializing", "created_at": datetime.now().isoformat()}
        )
        
        # System components (will be injected)
        self.prompt_engine: Optional[UniversalPromptEngine] = None
        self.memory_system: Optional[UniversalMemorySystem] = None
        self.communication_hub: Optional[UniversalCommunicationHub] = None
        self.plugin_system: Optional[UniversalPluginSystem] = None
        self.llm_client = None
        
        # Agent-specific state
        self.execution_context: Dict[str, Any] = {}
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.learning_data: Dict[str, Any] = defaultdict(list)
        self.adaptation_history: List[Dict[str, Any]] = []
        
        # Execution environment
        self.execution_queue: asyncio.Queue = asyncio.Queue()
        self.execution_task: Optional[asyncio.Task] = None
        self.is_running = False
    
    async def initialize(self, prompt_engine: UniversalPromptEngine,
                        memory_system: UniversalMemorySystem,
                        communication_hub: UniversalCommunicationHub,
                        plugin_system: UniversalPluginSystem,
                        llm_client=None):
        """Initialize the agent with system components"""
        self.prompt_engine = prompt_engine
        self.memory_system = memory_system
        self.communication_hub = communication_hub
        self.plugin_system = plugin_system
        self.llm_client = llm_client
        
        # Subscribe to relevant messages
        self.communication_hub.subscribe("task", self._handle_task_message)
        self.communication_hub.subscribe("system", self._handle_system_message)
        
        # Start execution environment
        self.execution_task = asyncio.create_task(self._execution_worker())
        self.is_running = True
        
        # Update state
        self.entity.state["status"] = "active"
        self.entity.state["initialized_at"] = datetime.now().isoformat()
        
        # Store initial memory
        init_memory = UniversalEntity(
            type=UniversalType.MEMORY,
            content=f"Agent {self.id} initialized with capabilities: {', '.join(self.entity.capabilities)}",
            metadata={"event": "initialization", "agent_id": self.id},
            importance=0.8
        )
        await self.memory_system.store_memory(init_memory, MemoryType.EPISODIC)
    
    async def _execution_worker(self):
        """Continuous execution worker for the agent"""
        while self.is_running:
            try:
                # Process execution queue
                try:
                    task_data = await asyncio.wait_for(self.execution_queue.get(), timeout=1.0)
                    await self._process_task(task_data)
                    self.execution_queue.task_done()
                except asyncio.TimeoutError:
                    # Periodic maintenance
                    await self._periodic_maintenance()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in agent {self.id} execution worker: {e}")
                await asyncio.sleep(1)
    
    async def _process_task(self, task_data: Dict[str, Any]):
        """Process a task using universal intelligence"""
        task_id = task_data.get("id", f"task_{uuid.uuid4().hex[:8]}")
        
        try:
            # Store task in active tasks
            self.active_tasks[task_id] = {
                **task_data,
                "status": "processing",
                "started_at": datetime.now().isoformat()
            }
            
            # Generate processing prompt
            context = {
                "task": task_data,
                "agent_capabilities": list(self.entity.capabilities),
                "recent_memories": await self._get_recent_memories(),
                "available_plugins": list(self.plugin_system.loaded_plugins.keys()) if self.plugin_system else [],
                "execution_context": self.execution_context
            }
            
            prompt = await self.prompt_engine.generate_prompt(
                "task_processing", context, self.llm_client
            )
            
            # Process with LLM if available
            if self.llm_client:
                response = await self.llm_client.generate(prompt, "You are a helpful assistant.")
                result = await self._interpret_response(response, task_data)
            else:
                result = await self._fallback_processing(task_data)
            
            # Update task status
            self.active_tasks[task_id].update({
                "status": "completed",
                "result": result,
                "completed_at": datetime.now().isoformat()
            })
            
            # Store memory of task completion
            task_memory = UniversalEntity(
                type=UniversalType.MEMORY,
                content=f"Completed task: {task_data.get('description', 'Unknown task')}",
                metadata={
                    "event": "task_completion",
                    "task_id": task_id,
                    "agent_id": self.id,
                    "result": result
                },
                importance=0.6
            )
            await self.memory_system.store_memory(task_memory, MemoryType.EPISODIC)
            
            # Send result message
            result_message = UniversalEntity(
                type=UniversalType.MESSAGE,
                content=result,
                metadata={
                    "message_type": "task_result",
                    "task_id": task_id,
                    "sender": self.id,
                    "recipient": task_data.get("requester")
                }
            )
            await self.communication_hub.send_message(result_message)
            
        except Exception as e:
            # Handle task failure
            self.active_tasks[task_id].update({
                "status": "failed",
                "error": str(e),
                "failed_at": datetime.now().isoformat()
            })
            
            # Send error message
            error_message = UniversalEntity(
                type=UniversalType.MESSAGE,
                content=f"Task failed: {str(e)}",
                metadata={
                    "message_type": "task_error",
                    "task_id": task_id,
                    "sender": self.id,
                    "recipient": task_data.get("requester")
                }
            )
            await self.communication_hub.send_message(error_message)
    
    async def _interpret_response(self, response: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Interpret LLM response and execute actions"""
        # Try to parse as JSON first
        try:
            parsed_response = json.loads(response)
            if isinstance(parsed_response, dict):
                return await self._execute_structured_response(parsed_response, task_data)
        except json.JSONDecodeError:
            pass
        
        # Fallback to text interpretation
        return await self._execute_text_response(response, task_data)
    
    async def _execute_structured_response(self, response: Dict[str, Any], 
                                         task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a structured response"""
        result = {"type": "structured", "content": response}
        
        # Check for plugin execution requests
        if "plugin" in response:
            plugin_name = response["plugin"]
            plugin_context = response.get("context", {})
            plugin_result = await self.plugin_system.execute_plugin(plugin_name, plugin_context)
            result["plugin_result"] = plugin_result
        
        # Check for memory operations
        if "remember" in response:
            memory_content = response["remember"]
            memory = UniversalEntity(
                type=UniversalType.MEMORY,
                content=memory_content,
                metadata={"source": "agent_decision", "agent_id": self.id},
                importance=response.get("importance", 0.5)
            )
            await self.memory_system.store_memory(memory, MemoryType.WORKING)
        
        # Check for communication requests
        if "communicate" in response:
            comm_data = response["communicate"]
            message = UniversalEntity(
                type=UniversalType.MESSAGE,
                content=comm_data.get("content", ""),
                metadata={
                    "message_type": comm_data.get("type", "general"),
                    "sender": self.id,
                    "recipient": comm_data.get("recipient")
                }
            )
            await self.communication_hub.send_message(message)
        
        return result
    
    async def _execute_text_response(self, response: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a text response"""
        # Simple text processing and action extraction
        result = {"type": "text", "content": response}
        
        # Look for action keywords
        response_lower = response.lower()
        
        if "remember" in response_lower:
            # Extract and store important information
            memory = UniversalEntity(
                type=UniversalType.MEMORY,
                content=response,
                metadata={"source": "agent_response", "agent_id": self.id},
                importance=0.4
            )
            await self.memory_system.store_memory(memory, MemoryType.WORKING)
        
        return result
    
    async def _fallback_processing(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback processing when no LLM is available"""
        return {
            "type": "fallback",
            "content": f"Processed task: {task_data.get('description', 'Unknown task')}",
            "agent_id": self.id,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _get_recent_memories(self, limit: int = 5) -> List[str]:
        """Get recent memories as context"""
        memories = await self.memory_system.retrieve_memories(
            query=f"agent:{self.id}",
            context={"agent_id": self.id},
            limit=limit
        )
        return [str(memory.content) for memory in memories]
    
    async def _handle_task_message(self, message: UniversalEntity):
        """Handle incoming task messages"""
        if message.metadata.get("recipient") == self.id:
            task_data = {
                "id": message.metadata.get("task_id", message.id),
                "description": str(message.content),
                "requester": message.metadata.get("sender"),
                "priority": message.priority.value,
                "metadata": message.metadata
            }
            await self.execution_queue.put(task_data)
    
    async def _handle_system_message(self, message: UniversalEntity):
        """Handle system messages"""
        content = str(message.content).lower()
        
        if "shutdown" in content:
            await self.shutdown()
        elif "adapt" in content:
            await self._trigger_adaptation(message.metadata)
        elif "learn" in content:
            await self._trigger_learning(message.metadata)
    
    async def _periodic_maintenance(self):
        """Periodic maintenance tasks"""
        # Clean up completed tasks
        current_time = datetime.now()
        completed_tasks = [
            task_id for task_id, task in self.active_tasks.items()
            if task.get("status") in ["completed", "failed"] and
            "completed_at" in task and
            (current_time - datetime.fromisoformat(task["completed_at"])).total_seconds() > 3600
        ]
        
        for task_id in completed_tasks:
            del self.active_tasks[task_id]
        
        # Update entity access time
        self.entity.accessed_at = current_time
        self.entity.access_count += 1
    
    async def _trigger_adaptation(self, context: Dict[str, Any]):
        """Trigger agent adaptation based on context"""
        adaptation_data = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "previous_capabilities": list(self.entity.capabilities),
            "previous_state": dict(self.entity.state)
        }
        
        # Analyze context and adapt
        if "new_capability" in context:
            self.entity.add_capability(context["new_capability"])
        
        if "state_update" in context:
            self.entity.state.update(context["state_update"])
        
        # Record adaptation
        adaptation_data["new_capabilities"] = list(self.entity.capabilities)
        adaptation_data["new_state"] = dict(self.entity.state)
        self.adaptation_history.append(adaptation_data)
        
        # Store adaptation memory
        adaptation_memory = UniversalEntity(
            type=UniversalType.MEMORY,
            content=f"Agent adapted: {context}",
            metadata={
                "event": "adaptation",
                "agent_id": self.id,
                "adaptation_data": adaptation_data
            },
            importance=0.9
        )
        await self.memory_system.store_memory(adaptation_memory, MemoryType.META)
    
    async def _trigger_learning(self, context: Dict[str, Any]):
        """Trigger learning process"""
        learning_type = context.get("type", "general")
        learning_data = context.get("data", {})
        
        self.learning_data[learning_type].append({
            "timestamp": datetime.now().isoformat(),
            "data": learning_data,
            "context": context
        })
        
        # Store learning memory
        learning_memory = UniversalEntity(
            type=UniversalType.MEMORY,
            content=f"Learning event: {learning_type}",
            metadata={
                "event": "learning",
                "agent_id": self.id,
                "learning_type": learning_type,
                "learning_data": learning_data
            },
            importance=0.7
        )
        await self.memory_system.store_memory(learning_memory, MemoryType.PROCEDURAL)
    
    async def assign_task(self, task_description: str, requester: str = "system", 
                         priority: Priority = Priority.NORMAL, metadata: Dict[str, Any] = None):
        """Assign a task to this agent"""
        task_data = {
            "id": f"task_{uuid.uuid4().hex[:8]}",
            "description": task_description,
            "requester": requester,
            "priority": priority.value,
            "metadata": metadata or {},
            "assigned_at": datetime.now().isoformat()
        }
        
        await self.execution_queue.put(task_data)
        return task_data["id"]
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "id": self.id,
            "name": self.entity.name,
            "status": self.entity.state.get("status", "unknown"),
            "capabilities": list(self.entity.capabilities),
            "active_tasks": len(self.active_tasks),
            "queue_size": self.execution_queue.qsize(),
            "is_running": self.is_running,
            "created_at": self.entity.created_at.isoformat(),
            "last_accessed": self.entity.accessed_at.isoformat(),
            "access_count": self.entity.access_count,
            "adaptation_count": len(self.adaptation_history)
        }
    
    async def shutdown(self):
        """Shutdown the agent gracefully"""
        self.is_running = False
        
        if self.execution_task:
            self.execution_task.cancel()
            try:
                await self.execution_task
            except asyncio.CancelledError:
                pass
        
        # Update state
        self.entity.state["status"] = "shutdown"
        self.entity.state["shutdown_at"] = datetime.now().isoformat()
        
        # Store shutdown memory
        shutdown_memory = UniversalEntity(
            type=UniversalType.MEMORY,
            content=f"Agent {self.id} shutdown",
            metadata={
                "event": "shutdown",
                "agent_id": self.id,
                "final_state": dict(self.entity.state)
            },
            importance=0.8
        )
        if self.memory_system:
            await self.memory_system.store_memory(shutdown_memory, MemoryType.EPISODIC)


class UniversalSystem:
    """The ultimate universal AI agent system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Core systems
        self.prompt_engine = UniversalPromptEngine(
            self.config.get("prompts_dir", "prompts")
        )
        self.memory_system = UniversalMemorySystem(
            self.config.get("memory_dir", "memory")
        )
        self.communication_hub = UniversalCommunicationHub()
        self.plugin_system = UniversalPluginSystem(
            self.config.get("plugins_dir", "plugins")
        )
        
        # Agent management
        self.agents: Dict[str, UniversalAgent] = {}
        self.agent_registry: Dict[str, Dict[str, Any]] = {}
        
        # System state
        self.is_initialized = False
        self.is_running = False
        self.llm_client = None
        
        # System intelligence
        self.system_memory: Dict[str, Any] = {}
        self.global_patterns: Dict[str, Any] = {}
        self.system_metrics: Dict[str, Any] = defaultdict(dict)
    
    async def initialize(self, llm_client=None):
        """Initialize the universal system"""
        if self.is_initialized:
            return
        
        self.llm_client = llm_client
        
        # Initialize core systems
        await self.prompt_engine.initialize()
        await self.memory_system.initialize()
        await self.plugin_system.initialize()
        
        # Setup system-level communication patterns
        self.communication_hub.subscribe("system", self._handle_system_message)
        self.communication_hub.subscribe("agent_request", self._handle_agent_request)
        
        # Create system memory
        system_init_memory = UniversalEntity(
            type=UniversalType.MEMORY,
            content="Universal system initialized",
            metadata={
                "event": "system_initialization",
                "timestamp": datetime.now().isoformat(),
                "config": self.config
            },
            importance=1.0
        )
        await self.memory_system.store_memory(system_init_memory, MemoryType.META)
        
        self.is_initialized = True
        self.is_running = True
    
    async def create_agent(self, agent_config: Dict[str, Any] = None) -> str:
        """Create a new universal agent"""
        if not self.is_initialized:
            await self.initialize()
        
        agent_config = agent_config or {}
        agent = UniversalAgent(config=agent_config)
        
        # Initialize agent with system components
        await agent.initialize(
            self.prompt_engine,
            self.memory_system,
            self.communication_hub,
            self.plugin_system,
            self.llm_client
        )
        
        # Register agent
        self.agents[agent.id] = agent
        self.agent_registry[agent.id] = {
            "config": agent_config,
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        # Store agent creation memory
        agent_memory = UniversalEntity(
            type=UniversalType.MEMORY,
            content=f"Created agent {agent.id} with capabilities: {', '.join(agent.entity.capabilities)}",
            metadata={
                "event": "agent_creation",
                "agent_id": agent.id,
                "config": agent_config
            },
            importance=0.8
        )
        await self.memory_system.store_memory(agent_memory, MemoryType.EPISODIC)
        
        return agent.id
    
    async def assign_task_to_agent(self, agent_id: str, task_description: str, 
                                 priority: Priority = Priority.NORMAL,
                                 metadata: Dict[str, Any] = None) -> Optional[str]:
        """Assign a task to a specific agent"""
        if agent_id not in self.agents:
            return None
        
        agent = self.agents[agent_id]
        task_id = await agent.assign_task(task_description, "system", priority, metadata)
        
        # Update system metrics
        self.system_metrics["tasks"]["total"] = self.system_metrics["tasks"].get("total", 0) + 1
        self.system_metrics["tasks"]["assigned"] = self.system_metrics["tasks"].get("assigned", 0) + 1
        
        return task_id
    
    async def assign_task_intelligently(self, task_description: str, 
                                      priority: Priority = Priority.NORMAL,
                                      metadata: Dict[str, Any] = None) -> Optional[str]:
        """Intelligently assign a task to the most suitable agent"""
        if not self.agents:
            # Create a general agent if none exist
            agent_id = await self.create_agent({"capabilities": ["general"]})
            return await self.assign_task_to_agent(agent_id, task_description, priority, metadata)
        
        # Find the most suitable agent
        best_agent_id = await self._find_best_agent_for_task(task_description, metadata)
        
        if best_agent_id:
            return await self.assign_task_to_agent(best_agent_id, task_description, priority, metadata)
        
        # If no suitable agent found, create one
        required_capabilities = await self._extract_required_capabilities(task_description, metadata)
        agent_config = {"capabilities": required_capabilities}
        agent_id = await self.create_agent(agent_config)
        
        return await self.assign_task_to_agent(agent_id, task_description, priority, metadata)
    
    async def _find_best_agent_for_task(self, task_description: str, 
                                      metadata: Dict[str, Any] = None) -> Optional[str]:
        """Find the best agent for a given task"""
        if not self.agents:
            return None
        
        # Score agents based on capabilities and current load
        agent_scores = {}
        
        for agent_id, agent in self.agents.items():
            score = 0.0
            
            # Capability matching
            task_words = set(task_description.lower().split())
            capability_matches = sum(1 for cap in agent.entity.capabilities 
                                   if any(word in cap.lower() for word in task_words))
            score += capability_matches * 0.4
            
            # Load balancing (prefer less busy agents)
            active_tasks = len(agent.active_tasks)
            queue_size = agent.execution_queue.qsize()
            load_penalty = (active_tasks + queue_size) * 0.1
            score -= load_penalty
            
            # Recent success rate (if available)
            # This would be enhanced with actual performance tracking
            score += 0.1  # Base score for active agents
            
            agent_scores[agent_id] = score
        
        # Return agent with highest score
        if agent_scores:
            return max(agent_scores.items(), key=lambda x: x[1])[0]
        
        return None
    
    async def _extract_required_capabilities(self, task_description: str, 
                                           metadata: Dict[str, Any] = None) -> List[str]:
        """Extract required capabilities from task description"""
        # Basic keyword-based capability extraction
        capabilities = ["general"]
        
        task_lower = task_description.lower()
        
        if any(word in task_lower for word in ["write", "create", "generate", "compose"]):
            capabilities.append("creative")
        
        if any(word in task_lower for word in ["analyze", "research", "study", "investigate"]):
            capabilities.append("analytical")
        
        if any(word in task_lower for word in ["code", "program", "develop", "implement"]):
            capabilities.append("programming")
        
        if any(word in task_lower for word in ["help", "assist", "support", "answer"]):
            capabilities.append("assistant")
        
        if any(word in task_lower for word in ["expert", "specialist", "professional"]):
            capabilities.append("expert")
        
        return capabilities
    
    async def _handle_system_message(self, message: UniversalEntity):
        """Handle system-level messages"""
        content = str(message.content).lower()
        
        if "create_agent" in content:
            # Extract agent configuration from message
            agent_config = message.metadata.get("agent_config", {})
            agent_id = await self.create_agent(agent_config)
            
            # Send response
            response = UniversalEntity(
                type=UniversalType.MESSAGE,
                content=f"Created agent: {agent_id}",
                metadata={
                    "message_type": "system_response",
                    "sender": "system",
                    "recipient": message.metadata.get("sender"),
                    "agent_id": agent_id
                }
            )
            await self.communication_hub.send_message(response)
        
        elif "shutdown" in content:
            await self.shutdown()
    
    async def _handle_agent_request(self, message: UniversalEntity):
        """Handle requests from agents"""
        request_type = message.metadata.get("request_type", "unknown")
        
        if request_type == "create_plugin":
            # Agent requesting plugin creation
            requirement = message.metadata.get("requirement", "")
            context = message.metadata.get("context", {})
            
            plugin_name = await self.plugin_system.generate_plugin(
                requirement, context, self.llm_client
            )
            
            # Send response
            response = UniversalEntity(
                type=UniversalType.MESSAGE,
                content=f"Plugin created: {plugin_name}" if plugin_name else "Plugin creation failed",
                metadata={
                    "message_type": "plugin_response",
                    "sender": "system",
                    "recipient": message.metadata.get("sender"),
                    "plugin_name": plugin_name,
                    "success": bool(plugin_name)
                }
            )
            await self.communication_hub.send_message(response)
        
        elif request_type == "memory_query":
            # Agent requesting memory search
            query = message.metadata.get("query", "")
            context = message.metadata.get("context", {})
            
            memories = await self.memory_system.retrieve_memories(query, context)
            
            # Send response
            response = UniversalEntity(
                type=UniversalType.MESSAGE,
                content=memories,
                metadata={
                    "message_type": "memory_response",
                    "sender": "system",
                    "recipient": message.metadata.get("sender"),
                    "query": query
                }
            )
            await self.communication_hub.send_message(response)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "system": {
                "initialized": self.is_initialized,
                "running": self.is_running,
                "agents_count": len(self.agents),
                "memory_layers": {
                    layer.value: len(memories) 
                    for layer, memories in self.memory_system.memory_layers.items()
                },
                "loaded_plugins": len(self.plugin_system.loaded_plugins),
                "generated_plugins": len(self.plugin_system.generated_plugins),
                "message_history_size": len(self.communication_hub.message_history),
                "active_conversations": len(self.communication_hub.active_conversations)
            },
            "agents": {
                agent_id: agent.get_status() 
                for agent_id, agent in self.agents.items()
            },
            "metrics": dict(self.system_metrics)
        }
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific agent"""
        if agent_id in self.agents:
            return self.agents[agent_id].get_status()
        return None
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all agents with their basic info"""
        return [
            {
                "id": agent_id,
                "name": agent.entity.name,
                "capabilities": list(agent.entity.capabilities),
                "status": agent.entity.state.get("status", "unknown"),
                "active_tasks": len(agent.active_tasks),
                "created_at": agent.entity.created_at.isoformat()
            }
            for agent_id, agent in self.agents.items()
        ]
    
    async def shutdown(self):
        """Shutdown the entire system gracefully"""
        self.is_running = False
        
        # Shutdown all agents
        for agent in self.agents.values():
            await agent.shutdown()
        
        # Cleanup memory system
        await self.memory_system.cleanup()
        
        # Store system shutdown memory
        shutdown_memory = UniversalEntity(
            type=UniversalType.MEMORY,
            content="Universal system shutdown",
            metadata={
                "event": "system_shutdown",
                "timestamp": datetime.now().isoformat(),
                "final_metrics": dict(self.system_metrics)
            },
            importance=1.0
        )
        await self.memory_system.store_memory(shutdown_memory, MemoryType.META)