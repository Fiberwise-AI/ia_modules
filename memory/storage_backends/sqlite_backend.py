"""SQLite storage backend for persistent memory storage."""

from typing import Optional, List, Any
import sqlite3
import json
import logging
import pickle
import os

logger = logging.getLogger(__name__)


class SQLiteBackend:
    """
    SQLite-based persistent storage backend.

    Stores memories in a SQLite database for persistence across restarts.
    """

    def __init__(self, db_path: str = "memory.db"):
        """
        Initialize SQLite backend.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                importance REAL NOT NULL,
                access_count INTEGER NOT NULL,
                last_accessed REAL,
                metadata TEXT,
                embedding BLOB
            )
        """)

        conn.commit()
        conn.close()

    async def store(self, memory: Any) -> None:
        """
        Store a memory.

        Args:
            memory: Memory object to store
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO memories
                (id, content, memory_type, timestamp, importance, access_count,
                 last_accessed, metadata, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory.id,
                memory.content,
                memory.memory_type.value,
                memory.timestamp,
                memory.importance,
                memory.access_count,
                memory.last_accessed,
                json.dumps(memory.metadata),
                pickle.dumps(memory.embedding) if memory.embedding else None
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            conn.rollback()
        finally:
            conn.close()

    async def retrieve(self, memory_id: str) -> Optional[Any]:
        """
        Retrieve a memory by ID.

        Args:
            memory_id: Memory ID

        Returns:
            Memory object or None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, content, memory_type, timestamp, importance,
                   access_count, last_accessed, metadata, embedding
            FROM memories
            WHERE id = ?
        """, (memory_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return self._row_to_memory(row)
        return None

    async def delete(self, memory_id: str) -> bool:
        """
        Delete a memory.

        Args:
            memory_id: Memory ID to delete

        Returns:
            True if deleted, False if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        deleted = cursor.rowcount > 0

        conn.commit()
        conn.close()

        return deleted

    async def list_all(self) -> List[Any]:
        """
        List all memories.

        Returns:
            List of all memories
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, content, memory_type, timestamp, importance,
                   access_count, last_accessed, metadata, embedding
            FROM memories
            ORDER BY timestamp DESC
        """)

        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_memory(row) for row in rows]

    async def clear(self) -> None:
        """Clear all stored memories."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM memories")

        conn.commit()
        conn.close()

    async def count(self) -> int:
        """
        Count stored memories.

        Returns:
            Number of memories
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM memories")
        count = cursor.fetchone()[0]

        conn.close()
        return count

    def _row_to_memory(self, row: tuple) -> Any:
        """Convert database row to Memory object."""
        from ..memory_manager import Memory, MemoryType

        memory_id, content, memory_type, timestamp, importance, \
            access_count, last_accessed, metadata_json, embedding_blob = row

        return Memory(
            id=memory_id,
            content=content,
            memory_type=MemoryType(memory_type),
            timestamp=timestamp,
            importance=importance,
            access_count=access_count,
            last_accessed=last_accessed,
            metadata=json.loads(metadata_json) if metadata_json else {},
            embedding=pickle.loads(embedding_blob) if embedding_blob else None
        )
