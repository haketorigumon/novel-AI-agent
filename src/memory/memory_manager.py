"""Memory management for agents with vector-based retrieval"""

import asyncio
import json
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

from ..utils.config import Config
from ..communication.message import Message

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

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
        embedding: Vector embedding of the memory content for semantic search
    """
    
    def __init__(
        self,
        content: Any,
        memory_type: str = "episodic",
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
        memory_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        embedding: Optional[np.ndarray] = None
    ):
        self.memory_id = memory_id or f"{memory_type}_{datetime.now().isoformat()}"
        self.content = content
        self.memory_type = memory_type
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
        self.importance = importance
        self.last_accessed = self.timestamp
        self.access_count = 0
        self.embedding = embedding
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary"""
        result = {
            "memory_id": self.memory_id,
            "content": self.content,
            "memory_type": self.memory_type,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "importance": self.importance,
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count
        }
        
        # Store embedding if available
        if self.embedding is not None:
            result["embedding"] = self.embedding.tolist()
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Create memory from dictionary"""
        embedding = None
        if "embedding" in data:
            embedding = np.array(data["embedding"], dtype=np.float32)
            
        memory = cls(
            memory_id=data["memory_id"],
            content=data["content"],
            memory_type=data["memory_type"],
            metadata=data["metadata"],
            importance=data["importance"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            embedding=embedding
        )
        
        memory.last_accessed = datetime.fromisoformat(data["last_accessed"])
        memory.access_count = data["access_count"]
        
        return memory
    
    def access(self):
        """Record memory access"""
        self.last_accessed = datetime.now()
        self.access_count += 1
        
    def get_text_content(self) -> str:
        """Get text representation of the memory content for embedding"""
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, dict):
            # For message memories or other structured content
            if "content" in self.content:
                return str(self.content["content"])
            else:
                return json.dumps(self.content)
        else:
            return str(self.content)


class MemoryManager:
    """
    Manages agent memories with different types and retrieval mechanisms
    
    Attributes:
        agent_id: ID of the agent this memory belongs to
        config: System configuration
        memories: Dictionary of memories by ID
        memory_by_type: Dictionary of memories organized by type
        index: FAISS index for vector-based retrieval
        embedding_size: Size of memory embeddings
    """
    
    def __init__(self, agent_id: str, config: Config):
        self.agent_id = agent_id
        self.config = config
        self.memories: Dict[str, Memory] = {}
        self.memory_by_type: Dict[str, List[Memory]] = {
            "episodic": [],
            "semantic": [],
            "procedural": [],
            "message": [],
            "system": []
        }
        self.memory_dir = Path(config.story.output_dir) / "memories" / agent_id
        self.llm_client = None
        
        # Vector search setup
        self.embedding_size = 384  # Default embedding size
        self.index = None
        self.memory_ids = []  # To map index positions to memory IDs
        self.use_vector_search = FAISS_AVAILABLE
    
    async def initialize(self):
        """Initialize the memory manager"""
        # Create memory directory
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing memories if any
        await self._load_memories()
        
        # Initialize vector index if FAISS is available
        if self.use_vector_search:
            await self._initialize_vector_index()
    
    async def _initialize_vector_index(self):
        """Initialize the vector index for semantic search"""
        if not self.use_vector_search:
            return
            
        # Create a new index
        self.index = faiss.IndexFlatL2(self.embedding_size)
        
        # Add existing memories with embeddings to the index
        embeddings = []
        self.memory_ids = []
        
        for memory_id, memory in self.memories.items():
            if memory.embedding is not None:
                embeddings.append(memory.embedding)
                self.memory_ids.append(memory_id)
        
        if embeddings:
            embeddings_array = np.array(embeddings).astype(np.float32)
            self.index.add(embeddings_array)
    
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
    
    async def _generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding for text using the LLM client
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if embedding failed
        """
        if not self.llm_client:
            return None
            
        try:
            # Use the LLM client to generate the embedding
            embedding = await self.llm_client.generate_embedding(text)
            if embedding is not None:
                # Update embedding size if needed
                if self.embedding_size != embedding.shape[0]:
                    self.embedding_size = embedding.shape[0]
                    # Reinitialize index if size changed
                    if self.use_vector_search:
                        self.index = faiss.IndexFlatL2(self.embedding_size)
                        await self._initialize_vector_index()
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
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
        # Create memory object
        memory = Memory(
            content=content,
            memory_type=memory_type,
            metadata=metadata,
            importance=importance
        )
        
        # Generate embedding if vector search is enabled
        if self.use_vector_search and self.llm_client:
            text_content = memory.get_text_content()
            embedding = await self._generate_embedding(text_content)
            if embedding is not None:
                memory.embedding = embedding
                
                # Add to index
                if self.index is not None:
                    self.index.add(np.array([embedding]).astype(np.float32))
                    self.memory_ids.append(memory.memory_id)
        
        # Store memory
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
        if self.use_vector_search and self.index is not None and len(self.memory_ids) > 0:
            # Vector-based semantic search
            return await self._vector_search(query, limit)
        else:
            # Fallback to keyword matching
            return await self._keyword_search(query, limit)
    
    async def _vector_search(self, query: str, limit: int) -> List[str]:
        """
        Perform vector-based semantic search
        
        Args:
            query: Query to find relevant memories
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of relevant memory contents as strings
        """
        # Generate embedding for query
        query_embedding = await self._generate_embedding(query)
        if query_embedding is None:
            return await self._keyword_search(query, limit)
        
        # Search the index
        k = min(limit, len(self.memory_ids))
        if k == 0:
            return []
            
        distances, indices = self.index.search(np.array([query_embedding]).astype(np.float32), k)
        
        # Get corresponding memories
        relevant_memories = []
        for idx in indices[0]:
            if 0 <= idx < len(self.memory_ids):
                memory_id = self.memory_ids[idx]
                memory = self.memories.get(memory_id)
                if memory:
                    memory.access()
                    relevant_memories.append(memory)
        
        # Return content as strings
        return [memory.get_text_content() for memory in relevant_memories]
    
    async def _keyword_search(self, query: str, limit: int) -> List[str]:
        """
        Perform keyword-based search
        
        Args:
            query: Query to find relevant memories
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of relevant memory contents as strings
        """
        query_terms = query.lower().split()
        
        scored_memories = []
        for memory in self.memories.values():
            content_str = memory.get_text_content().lower()
            score = sum(1 for term in query_terms if term in content_str)
            if score > 0:
                scored_memories.append((memory, score))
        
        # Sort by score and get top results
        relevant_memories = [memory for memory, score in sorted(scored_memories, key=lambda x: x[1], reverse=True)[:limit]]
        
        # Mark as accessed
        for memory in relevant_memories:
            memory.access()
        
        # Return content as strings
        return [memory.get_text_content() for memory in relevant_memories]
    
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
            
            # Remove from vector index if present
            if self.use_vector_search and memory_id in self.memory_ids:
                # This is a simplified approach - in a real implementation, we would need to rebuild the index
                # or use a more sophisticated approach to remove items from the index
                idx = self.memory_ids.index(memory_id)
                self.memory_ids[idx] = None  # Mark as removed
            
            # Save changes
            await self._save_memories()
    
    async def consolidate_memories(self, memory_type: str = "episodic", target_type: str = "semantic", max_memories: int = 10):
        """
        Consolidate memories to create higher-level insights
        
        Args:
            memory_type: Type of memories to consolidate
            target_type: Type of the consolidated memory
            max_memories: Maximum number of memories to consolidate at once
        """
        if not self.llm_client:
            return
            
        # Get recent memories of the specified type
        memories = await self.get_memories_by_type(memory_type, max_memories)
        
        if not memories:
            return
            
        # Extract content
        memory_contents = [memory.get_text_content() for memory in memories]
        
        # Use LLM to consolidate
        try:
            # This is a placeholder - in a real implementation, this would call the LLM client
            # to generate a summary or extract insights from the memories
            consolidated_content = f"Consolidated {len(memory_contents)} {memory_type} memories"
            
            # Store as a new memory
            await self.store_memory(
                content=consolidated_content,
                memory_type=target_type,
                metadata={
                    "source_memories": [memory.memory_id for memory in memories],
                    "consolidation_time": datetime.now().isoformat()
                },
                importance=0.8
            )
        except Exception as e:
            print(f"Error consolidating memories: {e}")
    
    async def summarize_memories(self, limit: int = 100) -> str:
        """
        Generate a summary of the agent's memories
        
        Args:
            limit: Maximum number of memories to include in the summary
            
        Returns:
            Summary text
        """
        if not self.memories:
            return "No memories available."
            
        # Get recent memories
        recent_memories = await self.get_recent_memories(limit)
        
        # Group by type
        memories_by_type = {}
        for memory in recent_memories:
            if memory.memory_type not in memories_by_type:
                memories_by_type[memory.memory_type] = []
            memories_by_type[memory.memory_type].append(memory)
        
        # Generate summary
        summary = []
        summary.append(f"Memory Summary for Agent {self.agent_id}")
        summary.append(f"Total Memories: {len(self.memories)}")
        
        for memory_type, memories in memories_by_type.items():
            summary.append(f"\n{memory_type.capitalize()} Memories ({len(memories)}):")
            for i, memory in enumerate(memories[:5], 1):
                summary.append(f"  {i}. {memory.get_text_content()[:100]}...")
            
            if len(memories) > 5:
                summary.append(f"  ... and {len(memories) - 5} more {memory_type} memories")
        
        return "\n".join(summary)