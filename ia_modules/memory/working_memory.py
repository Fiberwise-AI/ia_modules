"""
Working Memory - Short-term buffer with priority management.
"""

from typing import List, Optional, Any, Deque
from collections import deque
import logging

logger = logging.getLogger(__name__)


class WorkingMemory:
    """
    Working memory for short-term information storage.

    Uses LRU (Least Recently Used) eviction with importance-based
    priority management.
    """

    def __init__(self, size: int = 10):
        """
        Initialize working memory.

        Args:
            size: Maximum number of items in working memory
        """
        self.size = size
        self.memories: Deque[Any] = deque(maxlen=size)
        self.memory_dict: dict[str, Any] = {}  # For O(1) lookup

    async def add(self, memory: Any) -> None:
        """
        Add memory to working memory.

        If capacity is exceeded, evicts least important/recent memory.

        Args:
            memory: Memory object to add
        """
        # Check if already exists
        if memory.id in self.memory_dict:
            # Remove old version
            self.memories = deque(
                [m for m in self.memories if m.id != memory.id],
                maxlen=self.size
            )

        # Add new memory
        if len(self.memories) >= self.size:
            # Evict based on importance
            evicted = self._evict()
            if evicted:
                logger.debug(f"Evicted memory: {evicted.id}")

        self.memories.append(memory)
        self.memory_dict[memory.id] = memory

    async def get_all(self) -> List[Any]:
        """
        Get all memories in working memory.

        Returns most recent first.

        Returns:
            List of memories
        """
        return list(reversed(self.memories))

    async def get(self, memory_id: str) -> Optional[Any]:
        """
        Get specific memory by ID.

        Args:
            memory_id: Memory ID

        Returns:
            Memory object or None
        """
        return self.memory_dict.get(memory_id)

    async def remove(self, memory_id: str) -> bool:
        """
        Remove specific memory.

        Args:
            memory_id: Memory ID to remove

        Returns:
            True if removed, False if not found
        """
        if memory_id in self.memory_dict:
            self.memories = deque(
                [m for m in self.memories if m.id != memory_id],
                maxlen=self.size
            )
            del self.memory_dict[memory_id]
            return True
        return False

    async def clear(self) -> None:
        """Clear all working memory."""
        self.memories.clear()
        self.memory_dict.clear()

    def _evict(self) -> Optional[Any]:
        """
        Evict least important/recent memory.

        Returns:
            Evicted memory or None
        """
        if not self.memories:
            return None

        # Find memory with lowest score
        min_score = float('inf')
        min_idx = 0

        for idx, memory in enumerate(self.memories):
            score = self._calculate_eviction_score(memory, idx)
            if score < min_score:
                min_score = score
                min_idx = idx

        # Remove from position
        evicted = self.memories[min_idx]
        temp_list = list(self.memories)
        temp_list.pop(min_idx)
        self.memories = deque(temp_list, maxlen=self.size)
        del self.memory_dict[evicted.id]

        return evicted

    def _calculate_eviction_score(self, memory: Any, position: int) -> float:
        """
        Calculate eviction score (higher = keep longer).

        Args:
            memory: Memory to score
            position: Position in deque (0 = oldest)

        Returns:
            Eviction score
        """
        import time

        score = memory.importance * 10  # Base score from importance

        # Add recency bonus (newer = higher score)
        recency = position / max(len(self.memories), 1)
        score += recency * 5

        # Add access count bonus
        score += min(memory.access_count, 5) * 0.5

        # Add time-based recency
        now = time.time()
        age = now - memory.timestamp
        time_recency = 1.0 / (1.0 + age / 3600)  # Hours
        score += time_recency * 3

        return score

    def get_size(self) -> int:
        """Get current size of working memory."""
        return len(self.memories)

    def get_capacity(self) -> int:
        """Get maximum capacity."""
        return self.size

    def is_full(self) -> bool:
        """Check if working memory is at capacity."""
        return len(self.memories) >= self.size
