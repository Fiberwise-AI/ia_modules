"""
Advanced Memory Manager for sophisticated memory strategies.

This implements multiple memory types working together:
- Semantic memory: Long-term knowledge
- Episodic memory: Event sequences
- Working memory: Short-term buffer
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memories."""
    SEMANTIC = "semantic"  # Long-term knowledge
    EPISODIC = "episodic"  # Event sequences
    WORKING = "working"    # Short-term buffer
    COMPRESSED = "compressed"  # Compressed old memories


@dataclass
class Memory:
    """A memory entry."""
    content: str
    memory_type: MemoryType
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5  # 0-1 scale
    access_count: int = 0
    last_accessed: Optional[float] = None
    embedding: Optional[List[float]] = None
    id: Optional[str] = None

    def __post_init__(self):
        """Generate ID if not provided."""
        if self.id is None:
            import uuid
            self.id = str(uuid.uuid4())


@dataclass
class MemoryConfig:
    """Configuration for memory manager."""
    semantic_enabled: bool = True
    episodic_enabled: bool = True
    working_memory_size: int = 10
    compression_threshold: int = 50  # Number of memories before compression
    compression_enabled: bool = True
    embedding_model: str = "text-embedding-ada-002"
    storage_backend: str = "in_memory"  # "in_memory", "sqlite", "vector"
    max_tokens: int = 4000  # Max tokens for context window
    importance_threshold: float = 0.3  # Min importance to keep
    enable_embeddings: bool = True


class MemoryManager:
    """
    Advanced memory manager coordinating multiple memory types.

    Manages semantic, episodic, and working memory with automatic
    compression and intelligent retrieval.
    """

    def __init__(self, config: MemoryConfig):
        """
        Initialize memory manager.

        Args:
            config: Memory configuration
        """
        self.config = config
        self.memories: Dict[str, Memory] = {}

        # Initialize memory subsystems
        if config.semantic_enabled:
            from .semantic_memory import SemanticMemory
            self.semantic = SemanticMemory(
                embedding_model=config.embedding_model,
                enable_embeddings=config.enable_embeddings
            )
        else:
            self.semantic = None

        if config.episodic_enabled:
            from .episodic_memory import EpisodicMemory
            self.episodic = EpisodicMemory()
        else:
            self.episodic = None

        from .working_memory import WorkingMemory
        self.working = WorkingMemory(size=config.working_memory_size)

        if config.compression_enabled:
            from .compression import MemoryCompressor
            self.compressor = MemoryCompressor()
        else:
            self.compressor = None

        # Initialize storage backend
        self._init_storage_backend(config.storage_backend)

    def _init_storage_backend(self, backend_type: str) -> None:
        """Initialize storage backend."""
        if backend_type == "in_memory":
            from .storage_backends.in_memory_backend import InMemoryBackend
            self.storage = InMemoryBackend()
        elif backend_type == "sqlite":
            from .storage_backends.sqlite_backend import SQLiteBackend
            self.storage = SQLiteBackend()
        elif backend_type == "vector":
            from .storage_backends.vector_backend import VectorBackend
            self.storage = VectorBackend()
        else:
            raise ValueError(f"Unknown storage backend: {backend_type}")

    async def add(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a memory to the system.

        The memory will be automatically categorized and stored in
        appropriate memory subsystems.

        Args:
            content: Memory content
            metadata: Optional metadata

        Returns:
            Memory ID
        """
        import time

        if metadata is None:
            metadata = {}

        # Create memory
        memory = Memory(
            content=content,
            memory_type=self._determine_memory_type(content, metadata),
            timestamp=time.time(),
            metadata=metadata,
            importance=metadata.get("importance", 0.5)
        )

        logger.info(f"Adding memory: {memory.id} (type: {memory.memory_type.value})")

        # Store in global registry
        self.memories[memory.id] = memory

        # Route to appropriate subsystem
        if memory.memory_type == MemoryType.SEMANTIC and self.semantic:
            await self.semantic.add(memory)
        elif memory.memory_type == MemoryType.EPISODIC and self.episodic:
            await self.episodic.add(memory)
        elif memory.memory_type == MemoryType.WORKING:
            await self.working.add(memory)

        # Store in backend
        await self.storage.store(memory)

        # Check if compression needed
        if self.config.compression_enabled and len(self.memories) >= self.config.compression_threshold:
            await self.compress()

        return memory.id

    async def retrieve(
        self,
        query: str,
        k: int = 5,
        memory_types: Optional[List[MemoryType]] = None,
        min_importance: Optional[float] = None
    ) -> List[Memory]:
        """
        Retrieve relevant memories.

        Args:
            query: Query string
            k: Number of memories to retrieve
            memory_types: Optional filter by memory type
            min_importance: Optional minimum importance threshold

        Returns:
            List of relevant memories
        """
        logger.info(f"Retrieving memories for query: {query[:50]}...")

        results = []

        # Query each memory subsystem
        if memory_types is None or MemoryType.SEMANTIC in memory_types:
            if self.semantic:
                semantic_results = await self.semantic.retrieve(query, k)
                results.extend(semantic_results)

        if memory_types is None or MemoryType.EPISODIC in memory_types:
            if self.episodic:
                episodic_results = await self.episodic.retrieve(query, k)
                results.extend(episodic_results)

        if memory_types is None or MemoryType.WORKING in memory_types:
            working_results = await self.working.get_all()
            results.extend(working_results)

        # Filter by importance if specified
        if min_importance is not None:
            results = [m for m in results if m.importance >= min_importance]

        # Update access statistics
        import time
        for memory in results:
            memory.access_count += 1
            memory.last_accessed = time.time()

        # Sort by relevance/importance and limit
        results = self._rank_memories(results, query)[:k]

        logger.info(f"Retrieved {len(results)} memories")
        return results

    async def get_context_window(
        self,
        query: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Get formatted context for LLM within token budget.

        Args:
            query: Optional query to retrieve relevant context
            max_tokens: Maximum tokens (defaults to config)

        Returns:
            Formatted context string
        """
        if max_tokens is None:
            max_tokens = self.config.max_tokens

        # Always include working memory
        working_memories = await self.working.get_all()
        context_parts = []

        # Format working memory
        if working_memories:
            context_parts.append("## Recent Context")
            for mem in working_memories:
                context_parts.append(f"- {mem.content}")

        # Retrieve relevant long-term memories if query provided
        if query:
            relevant = await self.retrieve(
                query,
                k=10,
                memory_types=[MemoryType.SEMANTIC, MemoryType.EPISODIC]
            )
            if relevant:
                context_parts.append("\n## Relevant Background")
                for mem in relevant:
                    context_parts.append(f"- {mem.content}")

        context = "\n".join(context_parts)

        # Truncate to token limit (rough estimate: 4 chars per token)
        max_chars = max_tokens * 4
        if len(context) > max_chars:
            context = context[:max_chars] + "..."
            logger.warning(f"Context truncated to {max_tokens} tokens")

        return context

    async def compress(self) -> int:
        """
        Compress old memories to save space.

        Returns:
            Number of memories compressed
        """
        if not self.compressor:
            logger.warning("Compression not enabled")
            return 0

        logger.info("Starting memory compression...")

        # Get memories eligible for compression
        import time
        now = time.time()
        age_threshold = 86400  # 24 hours

        old_memories = [
            mem for mem in self.memories.values()
            if (now - mem.timestamp) > age_threshold
            and mem.memory_type != MemoryType.COMPRESSED
            and mem.importance < 0.8  # Don't compress important memories
        ]

        if not old_memories:
            logger.info("No memories eligible for compression")
            return 0

        # Compress memories
        compressed = await self.compressor.compress(old_memories)

        # Store compressed memory
        for comp_mem in compressed:
            await self.add(
                comp_mem.content,
                metadata={
                    **comp_mem.metadata,
                    "compressed": True,
                    "original_count": len(old_memories)
                }
            )

        # Remove originals
        for mem in old_memories:
            del self.memories[mem.id]
            await self.storage.delete(mem.id)

        logger.info(f"Compressed {len(old_memories)} memories into {len(compressed)}")
        return len(old_memories)

    async def clear(self, memory_types: Optional[List[MemoryType]] = None) -> None:
        """
        Clear memories.

        Args:
            memory_types: Optional list of memory types to clear
        """
        if memory_types is None:
            # Clear all
            self.memories.clear()
            if self.semantic:
                await self.semantic.clear()
            if self.episodic:
                await self.episodic.clear()
            await self.working.clear()
            await self.storage.clear()
        else:
            # Clear specific types
            to_remove = [
                mem_id for mem_id, mem in self.memories.items()
                if mem.memory_type in memory_types
            ]
            for mem_id in to_remove:
                del self.memories[mem_id]
                await self.storage.delete(mem_id)

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        stats = {
            "total_memories": len(self.memories),
            "by_type": {},
            "avg_importance": 0.0,
            "total_accesses": 0
        }

        for mem_type in MemoryType:
            count = sum(
                1 for m in self.memories.values()
                if m.memory_type == mem_type
            )
            stats["by_type"][mem_type.value] = count

        if self.memories:
            stats["avg_importance"] = sum(
                m.importance for m in self.memories.values()
            ) / len(self.memories)
            stats["total_accesses"] = sum(
                m.access_count for m in self.memories.values()
            )

        return stats

    def _determine_memory_type(
        self, content: str, metadata: Dict[str, Any]
    ) -> MemoryType:
        """Determine appropriate memory type for content."""
        # Check metadata hint
        if "memory_type" in metadata:
            type_str = metadata["memory_type"]
            return MemoryType(type_str)

        # Check for explicit type indicators
        if metadata.get("is_fact") or metadata.get("is_knowledge"):
            return MemoryType.SEMANTIC

        if metadata.get("is_event") or metadata.get("timestamp"):
            return MemoryType.EPISODIC

        # Default to working memory for recent interactions
        return MemoryType.WORKING

    def _rank_memories(self, memories: List[Memory], query: str) -> List[Memory]:
        """Rank memories by relevance and importance."""
        import time
        now = time.time()

        def score_memory(mem: Memory) -> float:
            score = mem.importance

            # Boost recently accessed
            if mem.last_accessed:
                recency = 1.0 / (1.0 + (now - mem.last_accessed) / 3600)
                score += recency * 0.3

            # Boost frequently accessed
            access_boost = min(mem.access_count / 10.0, 0.2)
            score += access_boost

            # Boost working memory
            if mem.memory_type == MemoryType.WORKING:
                score += 0.5

            return score

        return sorted(memories, key=score_memory, reverse=True)
