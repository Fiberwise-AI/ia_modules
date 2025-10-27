"""
Episodic Memory - Event sequences with temporal indexing.
"""

from typing import List, Optional, Any, Dict, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class EpisodicMemory:
    """
    Episodic memory for storing event sequences.

    Maintains temporal relationships and supports time-based queries.
    """

    def __init__(self):
        """Initialize episodic memory."""
        self.memories: Dict[str, Any] = {}
        self.temporal_index: List[Tuple[float, str]] = []  # (timestamp, memory_id)
        self.tag_index: Dict[str, List[str]] = defaultdict(list)

    async def add(self, memory: Any) -> None:
        """
        Add memory to episodic storage.

        Args:
            memory: Memory object to store
        """
        self.memories[memory.id] = memory

        # Add to temporal index
        self.temporal_index.append((memory.timestamp, memory.id))
        self.temporal_index.sort()  # Keep sorted by time

        # Add to tag index
        tags = memory.metadata.get("tags", [])
        for tag in tags:
            self.tag_index[tag].append(memory.id)

    async def retrieve(
        self,
        query: str,
        k: int = 5,
        time_range: Optional[Tuple[float, float]] = None,
        tags: Optional[List[str]] = None
    ) -> List[Any]:
        """
        Retrieve episodic memories.

        Args:
            query: Query string
            k: Number of results
            time_range: Optional (start, end) timestamp range
            tags: Optional list of tags to filter by

        Returns:
            List of memories
        """
        candidates = set(self.memories.keys())

        # Filter by time range
        if time_range:
            start, end = time_range
            time_filtered = set()
            for timestamp, mem_id in self.temporal_index:
                if start <= timestamp <= end:
                    time_filtered.add(mem_id)
            candidates &= time_filtered

        # Filter by tags
        if tags:
            tag_filtered = set()
            for tag in tags:
                tag_filtered.update(self.tag_index.get(tag, []))
            candidates &= tag_filtered

        # Score candidates by query relevance
        scores = []
        query_words = set(query.lower().split())

        for mem_id in candidates:
            memory = self.memories[mem_id]
            content_words = set(memory.content.lower().split())
            overlap = len(query_words & content_words)

            # Boost score by recency
            import time
            now = time.time()
            recency = 1.0 / (1.0 + (now - memory.timestamp) / 3600)  # Hours
            score = overlap + recency * 0.5

            if score > 0:
                scores.append((memory, score))

        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        return [mem for mem, _ in scores[:k]]

    async def get_recent(self, k: int = 10) -> List[Any]:
        """
        Get most recent memories.

        Args:
            k: Number of memories to return

        Returns:
            List of recent memories
        """
        recent_ids = [mem_id for _, mem_id in self.temporal_index[-k:]]
        recent_ids.reverse()  # Most recent first
        return [self.memories[mem_id] for mem_id in recent_ids if mem_id in self.memories]

    async def get_time_range(
        self,
        start: float,
        end: float
    ) -> List[Any]:
        """
        Get memories within time range.

        Args:
            start: Start timestamp
            end: End timestamp

        Returns:
            List of memories
        """
        results = []
        for timestamp, mem_id in self.temporal_index:
            if start <= timestamp <= end:
                results.append(self.memories[mem_id])
            elif timestamp > end:
                break
        return results

    async def get_by_tags(self, tags: List[str]) -> List[Any]:
        """
        Get memories with specific tags.

        Args:
            tags: List of tags

        Returns:
            List of memories
        """
        mem_ids = set()
        for tag in tags:
            mem_ids.update(self.tag_index.get(tag, []))

        return [self.memories[mem_id] for mem_id in mem_ids]

    async def get_sequence(
        self,
        start_id: str,
        length: int = 5
    ) -> List[Any]:
        """
        Get sequence of memories starting from a specific memory.

        Args:
            start_id: Starting memory ID
            length: Length of sequence

        Returns:
            List of memories in sequence
        """
        if start_id not in self.memories:
            return []

        start_memory = self.memories[start_id]
        start_time = start_memory.timestamp

        # Find position in temporal index
        start_idx = None
        for idx, (timestamp, mem_id) in enumerate(self.temporal_index):
            if mem_id == start_id:
                start_idx = idx
                break

        if start_idx is None:
            return [start_memory]

        # Get sequence
        end_idx = min(start_idx + length, len(self.temporal_index))
        sequence_ids = [mem_id for _, mem_id in self.temporal_index[start_idx:end_idx]]

        return [self.memories[mem_id] for mem_id in sequence_ids]

    async def clear(self) -> None:
        """Clear all episodic memories."""
        self.memories.clear()
        self.temporal_index.clear()
        self.tag_index.clear()
