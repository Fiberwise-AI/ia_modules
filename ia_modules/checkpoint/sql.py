"""SQL-based checkpoint storage implementation"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import uuid

from .core import BaseCheckpointer, Checkpoint, CheckpointSaveError, CheckpointLoadError


class SQLCheckpointer(BaseCheckpointer):
    """
    SQL-based checkpoint storage using DatabaseManager.

    Works with PostgreSQL and SQLite using named parameters (:param).

    Example:
        >>> from nexusql import DatabaseManager
        >>> db = DatabaseManager("postgresql://localhost/mydb")
                >>> checkpointer = SQLCheckpointer(db)
    """

    def __init__(self, db_manager):
        """
        Initialize with DatabaseManager.

        Args:
            db_manager: DatabaseManager instance (already connected)
        """
        self.db = db_manager

        # Schema created by migrations (V003__checkpoint_system.sql)
        if not self.db.table_exists("pipeline_checkpoints"):
            raise CheckpointSaveError(
                "pipeline_checkpoints table not found. Run database migrations first."
            )

    async def save_checkpoint(
        self,
        thread_id: str,
        pipeline_id: str,
        step_id: str,
        step_index: int,
        state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        step_name: Optional[str] = None,
        parent_checkpoint_id: Optional[str] = None
    ) -> str:
        """Save checkpoint to SQL database"""
        checkpoint_id = str(uuid.uuid4())

        query = """
        INSERT INTO pipeline_checkpoints (
            checkpoint_id, thread_id, pipeline_id, step_id, step_index,
            step_name, state, metadata, parent_checkpoint_id, timestamp
        ) VALUES (
            :checkpoint_id, :thread_id, :pipeline_id, :step_id, :step_index,
            :step_name, :state, :metadata, :parent_checkpoint_id, :timestamp
        )
        """

        params = {
            "checkpoint_id": checkpoint_id,
            "thread_id": thread_id,
            "pipeline_id": pipeline_id,
            "step_id": step_id,
            "step_index": step_index,
            "step_name": step_name,
            "state": json.dumps(state),
            "metadata": json.dumps(metadata or {}),
            "parent_checkpoint_id": parent_checkpoint_id,
            "timestamp": datetime.now().isoformat()
        }

        try:
            await self.db.execute(query, params)
            return checkpoint_id
        except Exception as e:
            raise CheckpointSaveError(f"Failed to save checkpoint: {e}")

    async def load_checkpoint(
        self,
        thread_id: str,
        checkpoint_id: Optional[str] = None
    ) -> Optional[Checkpoint]:
        """Load checkpoint from SQL database"""
        if checkpoint_id:
            query = """
            SELECT * FROM pipeline_checkpoints
            WHERE thread_id = :thread_id AND checkpoint_id = :checkpoint_id
            """
            params = {"thread_id": thread_id, "checkpoint_id": checkpoint_id}
        else:
            query = """
            SELECT * FROM pipeline_checkpoints
            WHERE thread_id = :thread_id
            ORDER BY timestamp DESC
            LIMIT 1
            """
            params = {"thread_id": thread_id}

        try:
            row = await self.db.fetch_one(query, params)
            if not row:
                return None

            return Checkpoint(
                checkpoint_id=row["checkpoint_id"],
                thread_id=row["thread_id"],
                pipeline_id=row["pipeline_id"],
                step_id=row["step_id"],
                step_index=row["step_index"],
                step_name=row["step_name"],
                state=json.loads(row["state"]),
                timestamp=datetime.fromisoformat(row["timestamp"]),
                metadata=json.loads(row["metadata"]),
                parent_checkpoint_id=row["parent_checkpoint_id"]
            )
        except Exception as e:
            raise CheckpointLoadError(f"Failed to load checkpoint: {e}")

    async def list_checkpoints(
        self,
        thread_id: str,
        pipeline_id: Optional[str] = None,
        limit: int = 10,
        offset: int = 0
    ) -> List[Checkpoint]:
        """List checkpoints from SQL database"""
        if pipeline_id:
            query = """
            SELECT * FROM pipeline_checkpoints
            WHERE thread_id = :thread_id AND pipeline_id = :pipeline_id
            ORDER BY timestamp DESC
            LIMIT :limit OFFSET :offset
            """
            params = {
                "thread_id": thread_id,
                "pipeline_id": pipeline_id,
                "limit": limit,
                "offset": offset
            }
        else:
            query = """
            SELECT * FROM pipeline_checkpoints
            WHERE thread_id = :thread_id
            ORDER BY timestamp DESC
            LIMIT :limit OFFSET :offset
            """
            params = {"thread_id": thread_id, "limit": limit, "offset": offset}

        try:
            # fetch_all is sync, returns list directly
            import asyncio
            loop = asyncio.get_event_loop()
            rows = await loop.run_in_executor(None, self.db.fetch_all, query, params)
            return [
                Checkpoint(
                    checkpoint_id=row["checkpoint_id"],
                    thread_id=row["thread_id"],
                    pipeline_id=row["pipeline_id"],
                    step_id=row["step_id"],
                    step_index=row["step_index"],
                    step_name=row["step_name"],
                    state=json.loads(row["state"]),
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    metadata=json.loads(row["metadata"]),
                    parent_checkpoint_id=row["parent_checkpoint_id"]
                )
                for row in rows
            ]
        except Exception as e:
            raise CheckpointLoadError(f"Failed to list checkpoints: {e}")

    async def delete_checkpoint(self, thread_id: str, checkpoint_id: str) -> bool:
        """Delete specific checkpoint"""
        query = """
        DELETE FROM pipeline_checkpoints
        WHERE thread_id = :thread_id AND checkpoint_id = :checkpoint_id
        """
        params = {"thread_id": thread_id, "checkpoint_id": checkpoint_id}

        try:
            await self.db.execute(query, params)
            return True
        except Exception:
            return False

    async def delete_checkpoints(
        self,
        thread_id: str,
        pipeline_id: Optional[str] = None
    ) -> int:
        """Delete all checkpoints for thread (optionally filtered by pipeline)"""
        if pipeline_id:
            query = """
            DELETE FROM pipeline_checkpoints
            WHERE thread_id = :thread_id AND pipeline_id = :pipeline_id
            """
            params = {"thread_id": thread_id, "pipeline_id": pipeline_id}
        else:
            query = """
            DELETE FROM pipeline_checkpoints
            WHERE thread_id = :thread_id
            """
            params = {"thread_id": thread_id}

        try:
            result = await self.db.execute(query, params)
            return result.rowcount if hasattr(result, 'rowcount') else 0
        except Exception:
            return 0

    async def get_checkpoint_stats(
        self,
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get checkpoint statistics"""
        if thread_id:
            query = """
            SELECT COUNT(*) as total, MIN(timestamp) as oldest, MAX(timestamp) as newest
            FROM pipeline_checkpoints
            WHERE thread_id = :thread_id
            """
            params = {"thread_id": thread_id}
            row = await self.db.fetch_one(query, params)

            return {
                'total_checkpoints': row['total'] if row else 0,
                'oldest_checkpoint': datetime.fromisoformat(row['oldest']) if row and row['oldest'] else None,
                'newest_checkpoint': datetime.fromisoformat(row['newest']) if row and row['newest'] else None,
                'thread_id': thread_id
            }
        else:
            query = """
            SELECT COUNT(*) as total, MIN(timestamp) as oldest, MAX(timestamp) as newest,
                   COUNT(DISTINCT thread_id) as thread_count
            FROM pipeline_checkpoints
            """
            row = await self.db.fetch_one(query)

            threads_query = "SELECT DISTINCT thread_id FROM pipeline_checkpoints"
            threads = await self.db.fetch_all(threads_query)

            return {
                'total_checkpoints': row['total'] if row else 0,
                'oldest_checkpoint': datetime.fromisoformat(row['oldest']) if row and row['oldest'] else None,
                'newest_checkpoint': datetime.fromisoformat(row['newest']) if row and row['newest'] else None,
                'threads': [t['thread_id'] for t in threads] if threads else []
            }

    async def close(self) -> None:
        """Close (no-op for SQL - DatabaseManager handles connections)"""
        pass
