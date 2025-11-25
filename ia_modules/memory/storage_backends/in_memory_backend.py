"""In-memory storage backend for memories."""

from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class InMemoryBackend:
    """
    Simple in-memory storage backend.

    Stores memories in a dictionary. Not persistent across restarts.
    """

    def __init__(self):
        """Initialize in-memory backend."""
        self.storage: Dict[str, Any] = {}

    async def store(self, memory: Any) -> None:
        """
        Store a memory.

        Args:
            memory: Memory object to store
        """
        self.storage[memory.id] = memory

    async def retrieve(self, memory_id: str) -> Optional[Any]:
        """
        Retrieve a memory by ID.

        Args:
            memory_id: Memory ID

        Returns:
            Memory object or None
        """
        return self.storage.get(memory_id)

    async def delete(self, memory_id: str) -> bool:
        """
        Delete a memory.

        Args:
            memory_id: Memory ID to delete

        Returns:
            True if deleted, False if not found
        """
        if memory_id in self.storage:
            del self.storage[memory_id]
            return True
        return False

    async def list_all(self) -> List[Any]:
        """
        List all memories.

        Returns:
            List of all memories
        """
        return list(self.storage.values())

    async def clear(self) -> None:
        """Clear all stored memories."""
        self.storage.clear()

    async def count(self) -> int:
        """
        Count stored memories.

        Returns:
            Number of memories
        """
        return len(self.storage)
