"""
SQL-based checkpoint storage using DatabaseInterface

Supports PostgreSQL, SQLite, MySQL, and DuckDB through unified interface.
"""

import uuid
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

from ia_modules.database.interfaces import DatabaseInterface, DatabaseType

from .core import (
    BaseCheckpointer,
    Checkpoint,
    CheckpointSaveError,
    CheckpointLoadError,
    CheckpointDeleteError,
    CheckpointStatus
)


class SQLCheckpointer(BaseCheckpointer):
    """
    SQL-based checkpoint storage using DatabaseInterface.

    Works with PostgreSQL, SQLite, MySQL, and DuckDB through the unified interface.

    Example:
        >>> from ia_modules.database import get_database_interface
        >>> db = await get_database_interface("postgresql://localhost/mydb")
        >>> checkpointer = SQLCheckpointer(db)
        >>> await checkpointer.initialize()
    """

    def __init__(self, db_interface: DatabaseInterface):
        """
        Initialize with any SQL database.

        Args:
            db_interface: DatabaseInterface instance (already connected)
        """
        self.db = db_interface
        self._initialized = False

    async def initialize(self) -> None:
        """Create checkpoint schema if not exists"""
        await self._create_schema()
        self._initialized = True

    async def _create_schema(self) -> None:
        """Create checkpoint table with database-specific adaptations"""
        # Adapt schema based on database type
        if self.db.db_type == DatabaseType.POSTGRESQL:
            schema_sql = """
                CREATE TABLE IF NOT EXISTS pipeline_checkpoints (
                    checkpoint_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    thread_id VARCHAR(255) NOT NULL,
                    pipeline_id VARCHAR(255) NOT NULL,
                    step_id VARCHAR(255) NOT NULL,
                    step_index INTEGER NOT NULL,
                    step_name VARCHAR(255),
                    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
                    state JSONB NOT NULL,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    status VARCHAR(50) DEFAULT 'completed',
                    parent_checkpoint_id UUID,

                    CONSTRAINT fk_parent_checkpoint
                        FOREIGN KEY (parent_checkpoint_id)
                        REFERENCES pipeline_checkpoints(checkpoint_id)
                        ON DELETE SET NULL
                );

                CREATE INDEX IF NOT EXISTS idx_checkpoint_thread_id
                    ON pipeline_checkpoints(thread_id);
                CREATE INDEX IF NOT EXISTS idx_checkpoint_pipeline_id
                    ON pipeline_checkpoints(pipeline_id);
                CREATE INDEX IF NOT EXISTS idx_checkpoint_timestamp
                    ON pipeline_checkpoints(timestamp);
                CREATE INDEX IF NOT EXISTS idx_checkpoint_thread_timestamp
                    ON pipeline_checkpoints(thread_id, timestamp DESC);
            """
        else:
            # SQLite, MySQL, DuckDB - use TEXT for UUID, JSON for state
            schema_sql = """
                CREATE TABLE IF NOT EXISTS pipeline_checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    pipeline_id TEXT NOT NULL,
                    step_id TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    step_name TEXT,
                    timestamp TEXT NOT NULL,
                    state TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    status TEXT DEFAULT 'completed',
                    parent_checkpoint_id TEXT,

                    FOREIGN KEY (parent_checkpoint_id)
                        REFERENCES pipeline_checkpoints(checkpoint_id)
                        ON DELETE SET NULL
                );

                CREATE INDEX IF NOT EXISTS idx_checkpoint_thread_id
                    ON pipeline_checkpoints(thread_id);
                CREATE INDEX IF NOT EXISTS idx_checkpoint_pipeline_id
                    ON pipeline_checkpoints(pipeline_id);
                CREATE INDEX IF NOT EXISTS idx_checkpoint_timestamp
                    ON pipeline_checkpoints(timestamp);
                CREATE INDEX IF NOT EXISTS idx_checkpoint_thread_timestamp
                    ON pipeline_checkpoints(thread_id, timestamp);
            """

        # Execute schema creation (handle multi-statement SQL)
        for statement in schema_sql.split(';'):
            statement = statement.strip()
            if statement:
                result = await self.db.execute_query(statement)
                if not result.success:
                    raise CheckpointSaveError(f"Failed to create schema: {result.error_message}")

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
        if not self._initialized:
            raise CheckpointSaveError("Checkpointer not initialized")

        checkpoint_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        # Serialize state and metadata
        state_json = json.dumps(state)
        metadata_json = json.dumps(metadata or {})

        # Insert checkpoint
        if self.db.db_type == DatabaseType.POSTGRESQL:
            query = """
                INSERT INTO pipeline_checkpoints
                (checkpoint_id, thread_id, pipeline_id, step_id, step_index,
                 step_name, timestamp, state, metadata, status, parent_checkpoint_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9::jsonb, $10, $11)
            """
            params = (
                checkpoint_id, thread_id, pipeline_id, step_id, step_index,
                step_name or step_id, timestamp, state_json, metadata_json,
                CheckpointStatus.COMPLETED.value, parent_checkpoint_id
            )
        else:
            # SQLite, MySQL, DuckDB use ? placeholders
            query = """
                INSERT INTO pipeline_checkpoints
                (checkpoint_id, thread_id, pipeline_id, step_id, step_index,
                 step_name, timestamp, state, metadata, status, parent_checkpoint_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (
                checkpoint_id, thread_id, pipeline_id, step_id, step_index,
                step_name or step_id, timestamp, state_json, metadata_json,
                CheckpointStatus.COMPLETED.value, parent_checkpoint_id
            )

        result = await self.db.execute_query(query, params)

        if not result.success:
            raise CheckpointSaveError(f"Failed to save checkpoint: {result.error_message}")

        return checkpoint_id

    async def load_checkpoint(
        self,
        thread_id: str,
        checkpoint_id: Optional[str] = None
    ) -> Optional[Checkpoint]:
        """Load checkpoint from SQL database"""
        if not self._initialized:
            raise CheckpointLoadError("Checkpointer not initialized")

        if checkpoint_id:
            # Load specific checkpoint
            if self.db.db_type == DatabaseType.POSTGRESQL:
                query = """
                    SELECT * FROM pipeline_checkpoints
                    WHERE thread_id = $1 AND checkpoint_id = $2
                """
                params = (thread_id, checkpoint_id)
            else:
                query = """
                    SELECT * FROM pipeline_checkpoints
                    WHERE thread_id = ? AND checkpoint_id = ?
                """
                params = (thread_id, checkpoint_id)
        else:
            # Load latest checkpoint
            if self.db.db_type == DatabaseType.POSTGRESQL:
                query = """
                    SELECT * FROM pipeline_checkpoints
                    WHERE thread_id = $1
                    ORDER BY timestamp DESC
                    LIMIT 1
                """
                params = (thread_id,)
            else:
                query = """
                    SELECT * FROM pipeline_checkpoints
                    WHERE thread_id = ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                """
                params = (thread_id,)

        result = await self.db.fetch_one(query, params)

        if not result.success or not result.data:
            return None

        row = result.get_first_row()
        return self._row_to_checkpoint(row)

    async def list_checkpoints(
        self,
        thread_id: str,
        limit: int = 10,
        offset: int = 0
    ) -> List[Checkpoint]:
        """List checkpoints for thread"""
        if not self._initialized:
            raise CheckpointLoadError("Checkpointer not initialized")

        if self.db.db_type == DatabaseType.POSTGRESQL:
            query = """
                SELECT * FROM pipeline_checkpoints
                WHERE thread_id = $1
                ORDER BY timestamp DESC
                LIMIT $2 OFFSET $3
            """
            params = (thread_id, limit, offset)
        else:
            query = """
                SELECT * FROM pipeline_checkpoints
                WHERE thread_id = ?
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            """
            params = (thread_id, limit, offset)

        result = await self.db.fetch_all(query, params)

        if not result.success:
            raise CheckpointLoadError(f"Failed to list checkpoints: {result.error_message}")

        return [self._row_to_checkpoint(row) for row in result.data]

    async def delete_checkpoints(
        self,
        thread_id: str,
        before: Optional[datetime] = None,
        keep_latest: int = 0
    ) -> int:
        """Delete checkpoints"""
        if not self._initialized:
            raise CheckpointDeleteError("Checkpointer not initialized")

        if keep_latest > 0:
            # Delete all but keep N latest
            if self.db.db_type == DatabaseType.POSTGRESQL:
                query = """
                    DELETE FROM pipeline_checkpoints
                    WHERE checkpoint_id IN (
                        SELECT checkpoint_id FROM pipeline_checkpoints
                        WHERE thread_id = $1
                        ORDER BY timestamp DESC
                        OFFSET $2
                    )
                """
                params = (thread_id, keep_latest)
            else:
                query = """
                    DELETE FROM pipeline_checkpoints
                    WHERE checkpoint_id IN (
                        SELECT checkpoint_id FROM pipeline_checkpoints
                        WHERE thread_id = ?
                        ORDER BY timestamp DESC
                        LIMIT -1 OFFSET ?
                    )
                """
                params = (thread_id, keep_latest)
        elif before:
            # Delete checkpoints before timestamp
            before_iso = before.isoformat()
            if self.db.db_type == DatabaseType.POSTGRESQL:
                query = """
                    DELETE FROM pipeline_checkpoints
                    WHERE thread_id = $1 AND timestamp < $2
                """
                params = (thread_id, before_iso)
            else:
                query = """
                    DELETE FROM pipeline_checkpoints
                    WHERE thread_id = ? AND timestamp < ?
                """
                params = (thread_id, before_iso)
        else:
            # Delete all for thread
            if self.db.db_type == DatabaseType.POSTGRESQL:
                query = "DELETE FROM pipeline_checkpoints WHERE thread_id = $1"
                params = (thread_id,)
            else:
                query = "DELETE FROM pipeline_checkpoints WHERE thread_id = ?"
                params = (thread_id,)

        result = await self.db.execute_query(query, params)

        if not result.success:
            raise CheckpointDeleteError(f"Failed to delete checkpoints: {result.error_message}")

        return result.row_count

    async def get_checkpoint_stats(
        self,
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get checkpoint statistics"""
        if not self._initialized:
            return {}

        if thread_id:
            # Stats for specific thread
            if self.db.db_type == DatabaseType.POSTGRESQL:
                query = """
                    SELECT
                        COUNT(*) as total,
                        MIN(timestamp) as oldest,
                        MAX(timestamp) as newest
                    FROM pipeline_checkpoints
                    WHERE thread_id = $1
                """
                params = (thread_id,)
            else:
                query = """
                    SELECT
                        COUNT(*) as total,
                        MIN(timestamp) as oldest,
                        MAX(timestamp) as newest
                    FROM pipeline_checkpoints
                    WHERE thread_id = ?
                """
                params = (thread_id,)

            result = await self.db.fetch_one(query, params)

            if not result.success or not result.data:
                return {'total_checkpoints': 0, 'thread_id': thread_id}

            row = result.get_first_row()
            return {
                'total_checkpoints': row['total'],
                'oldest_checkpoint': datetime.fromisoformat(row['oldest']) if row['oldest'] else None,
                'newest_checkpoint': datetime.fromisoformat(row['newest']) if row['newest'] else None,
                'thread_id': thread_id
            }
        else:
            # Stats for all threads
            query = """
                SELECT
                    COUNT(*) as total,
                    MIN(timestamp) as oldest,
                    MAX(timestamp) as newest,
                    COUNT(DISTINCT thread_id) as thread_count
                FROM pipeline_checkpoints
            """

            result = await self.db.fetch_one(query)

            if not result.success or not result.data:
                return {'total_checkpoints': 0, 'threads': []}

            row = result.get_first_row()

            # Get list of threads
            if self.db.db_type == DatabaseType.POSTGRESQL:
                threads_query = "SELECT DISTINCT thread_id FROM pipeline_checkpoints"
            else:
                threads_query = "SELECT DISTINCT thread_id FROM pipeline_checkpoints"

            threads_result = await self.db.fetch_all(threads_query)
            threads = [r['thread_id'] for r in threads_result.data] if threads_result.success else []

            return {
                'total_checkpoints': row['total'],
                'oldest_checkpoint': datetime.fromisoformat(row['oldest']) if row['oldest'] else None,
                'newest_checkpoint': datetime.fromisoformat(row['newest']) if row['newest'] else None,
                'threads': threads
            }

    async def close(self) -> None:
        """Close database connection"""
        if self.db:
            await self.db.disconnect()

    def _row_to_checkpoint(self, row: Dict[str, Any]) -> Checkpoint:
        """Convert database row to Checkpoint object"""
        # Parse JSON fields
        state = json.loads(row['state']) if isinstance(row['state'], str) else row['state']
        metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']

        return Checkpoint(
            checkpoint_id=row['checkpoint_id'],
            thread_id=row['thread_id'],
            pipeline_id=row['pipeline_id'],
            pipeline_version=row.get('pipeline_version'),
            step_id=row['step_id'],
            step_index=row['step_index'],
            step_name=row.get('step_name', row['step_id']),
            timestamp=datetime.fromisoformat(row['timestamp']) if isinstance(row['timestamp'], str) else row['timestamp'],
            state=state,
            metadata=metadata,
            status=CheckpointStatus(row.get('status', 'completed')),
            parent_checkpoint_id=row.get('parent_checkpoint_id')
        )
