"""Memory management for agents"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from ..utils.config import Config
from ..communication.message import Message

class Memory:
    """
    Represents a single memory item
    
    Attributes:
        memory_id: Unique identifier for the memory
        content: Memory content
        memory_type: Type of memory (e.g., episodic, semantic, procedural)
        timestamp: Time the memory was created
        metadata: Additional memory metadata
        importance: Importance score (0-1)
    """
    
    def __init__(
        self,
        content: Any,
        memory_type: str = "episodic",
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
        memory_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ):
        self.memory_id = memory_id or f"{memory_type}_{datetime.now().isoformat()}"
        self.content = content
        self.memory_type = memory_type
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
        self.importance = importance
        self.last_accessed = self.timestamp
        self.access_count = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary"""
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "memory_type": self.memory_type,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "importance": self.importance,
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Create memory from dictionary"""
        memory = cls(
            memory_id=data["memory_id"],
            content=data["content"],
            memory_type=data["memory_type"],
            metadata=data["metadata"],
            importance=data["importance"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )
        
        memory.last_accessed = datetime.fromisoformat(data["last_accessed"])
        memory.access_count = data["access_count"]
        
        return memory
    
    def access(self):
        """Record memory access"""
        self.last_accessed = datetime.now()
        self.access_count += 1


class MemoryManager:
    """
    Manages agent memories with different types and retrieval mechanisms
    
    Attributes:
        agent_id: ID of the agent this memory belongs to
        config: System configuration
        memories: Dictionary of memories by ID
        memory_by_type: Dictionary of memories organized by type
    """
    
    def __init__(self, agent_id: str, config: Config):
        self.agent_id = agent_id
        self.config = config
        self.memories: Dict[str, Memory] = {}
        self.memory_by_type: Dict[str, List[Memory]] = {
            "episodic": [],
            "semantic": [],
            "procedural": [],
            "message": []
        }
        self.memory_dir = Path(config.story.output_dir) / "memories" / agent_id
        self.llm_client = None
    
    async def initialize(self):
        """Initialize the memory manager"""
        # Create memory directory
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing memories if any
        await self._load_memories()
    
    async def _load_memories(self):
        """Load memories from disk"""
        memory_file = self.memory_dir / "memories.json"
        if memory_file.exists():
            try:
                with open(memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for memory_data in data:
                    memory = Memory.from_dict(memory_data)
                    self.memories[memory.memory_id] = memory
                    
                    if memory.memory_type not in self.memory_by_type:
                        self.memory_by_type[memory.memory_type] = []
                    
                    self.memory_by_type[memory.memory_type].append(memory)
            except Exception as e:
                print(f"Error loading memories: {e}")
    
    async def _save_memories(self):
        """Save memories to disk"""
        memory_file = self.memory_dir / "memories.json"
        
        try:
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump([memory.to_dict() for memory in self.memories.values()], f, indent=2)
        except Exception as e:
            print(f"Error saving memories: {e}")
    
    async def store_memory(self, content: Any, memory_type: str = "episodic", metadata: Optional[Dict[str, Any]] = None, importance: float = 0.5) -> Memory:
        """
        Store a new memory
        
        Args:
            content: Memory content
            memory_type: Type of memory
            metadata: Additional memory metadata
            importance: Importance score (0-1)
            
        Returns:
            The created memory
        """
        memory = Memory(
            content=content,
            memory_type=memory_type,
            metadata=metadata,
            importance=importance
        )
        
        self.memories[memory.memory_id] = memory
        
        if memory_type not in self.memory_by_type:
            self.memory_by_type[memory_type] = []
        
        self.memory_by_type[memory_type].append(memory)
        
        # Save periodically (could be optimized to batch saves)
        await self._save_memories()
        
        return memory
    
    async def store_message(self, message: Message) -> Memory:
        """
        Store a message as a memory
        
        Args:
            message: The message to store
            
        Returns:
            The created memory
        """
        # Determine importance based on message type
        importance = 0.7 if message.type in ["command", "query", "response"] else 0.5
        
        return await self.store_memory(
            content=message.to_dict(),
            memory_type="message",
            metadata={
                "sender_id": message.sender_id,
                "receiver_id": message.receiver_id,
                "message_type": message.type
            },
            importance=importance
        )
    
    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        """
        Get a specific memory by ID
        
        Args:
            memory_id: ID of the memory to retrieve
            
        Returns:
            The memory if found, None otherwise
        """
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            memory.access()
            return memory
        
        return None
    
    async def get_memories_by_type(self, memory_type: str, limit: Optional[int] = None) -> List[Memory]:
        """
        Get memories of a specific type
        
        Args:
            memory_type: Type of memories to retrieve
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of memories
        """
        if memory_type not in self.memory_by_type:
            return []
        
        memories = sorted(self.memory_by_type[memory_type], key=lambda m: m.timestamp, reverse=True)
        
        for memory in memories[:limit]:
            memory.access()
        
        return memories[:limit] if limit else memories
    
    async def get_recent_memories(self, limit: int = 10) -> List[Memory]:
        """
        Get the most recent memories
        
        Args:
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of recent memories
        """
        all_memories = list(self.memories.values())
        recent_memories = sorted(all_memories, key=lambda m: m.timestamp, reverse=True)
        
        for memory in recent_memories[:limit]:
            memory.access()
        
        return recent_memories[:limit]
    
    async def get_important_memories(self, limit: int = 10) -> List[Memory]:
        """
        Get the most important memories
        
        Args:
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of important memories
        """
        all_memories = list(self.memories.values())
        important_memories = sorted(all_memories, key=lambda m: m.importance, reverse=True)
        
        for memory in important_memories[:limit]:
            memory.access()
        
        return important_memories[:limit]
    
    async def retrieve_relevant_memories(self, query: str, limit: int = 5) -> List[str]:
        """
        Retrieve memories relevant to a query
        
        Args:
            query: Query to find relevant memories
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of relevant memory contents as strings
        """
        # Simple keyword matching for now
        # In a real implementation, this would use embeddings and semantic search
        query_terms = query.lower().split()
        
        scored_memories = []
        for memory in self.memories.values():
            content_str = str(memory.content).lower()
            score = sum(1 for term in query_terms if term in content_str)
            if score > 0:
                scored_memories.append((memory, score))
        
        # Sort by score and get top results
        relevant_memories = [memory for memory, score in sorted(scored_memories, key=lambda x: x[1], reverse=True)[:limit]]
        
        # Mark as accessed
        for memory in relevant_memories:
            memory.access()
        
        # Return content as strings
        return [str(memory.content) for memory in relevant_memories]
    
    async def forget_memory(self, memory_id: str):
        """
        Remove a memory
        
        Args:
            memory_id: ID of the memory to remove
        """
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            
            # Remove from type-based collections
            if memory.memory_type in self.memory_by_type:
                self.memory_by_type[memory.memory_type] = [m for m in self.memory_by_type[memory.memory_type] if m.memory_id != memory_id]
            
            # Remove from main collection
            del self.memories[memory_id]
            
            # Save changes
            await self._save_memories()
    
    async def consolidate_memories(self):
        """
        Consolidate memories to create higher-level insights
        This would typically use the LLM to summarize and extract patterns
        """
        # This is a placeholder for a more sophisticated implementation
        # In a real system, this would use the LLM to create semantic memories from episodic ones
        pass