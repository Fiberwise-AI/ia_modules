"""
SQL-based conversation memory using DatabaseInterface
"""

import uuid
import json
from typing import Dict, List, Optional, Any
from datetime import datetime



from .core import ConversationMemory, Message, MessageRole


class SQLConversationMemory(ConversationMemory):
    """
    SQL-based conversation memory.

    Uses DatabaseInterface for PostgreSQL, SQLite, MySQL, DuckDB support.
    """

    def __init__(self, db_manager):
        self.db = db_manager
    async def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        function_call: Optional[Dict[str, Any]] = None,
        function_name: Optional[str] = None
    ) -> str:
        """Add message to database"""
        message_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        metadata_json = json.dumps(metadata or {})
        function_call_json = json.dumps(function_call) if function_call else None

        if self.db.db_type == DatabaseType.POSTGRESQL:
            query = """
                INSERT INTO conversation_messages
                (message_id, thread_id, user_id, role, content, metadata, timestamp, function_call, function_name)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8::jsonb, $9)
            """
            params = (message_id, thread_id, user_id, role, content, metadata_json, timestamp, function_call_json, function_name)
        else:
            query = """
                INSERT INTO conversation_messages
                (message_id, thread_id, user_id, role, content, metadata, timestamp, function_call, function_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (message_id, thread_id, user_id, role, content, metadata_json, timestamp, function_call_json, function_name)

        await self.db.execute_async(query, params)
        return message_id

    async def get_messages(
        self,
        thread_id: str,
        limit: int = 10,
        offset: int = 0
    ) -> List[Message]:
        """Get messages for thread"""
        if self.db.db_type == DatabaseType.POSTGRESQL:
            query = """
                SELECT * FROM conversation_messages
                WHERE thread_id = $1
                ORDER BY timestamp DESC
                LIMIT $2 OFFSET $3
            """
            params = (thread_id, limit, offset)
        else:
            query = """
                SELECT * FROM conversation_messages
                WHERE thread_id = ?
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            """
            params = (thread_id, limit, offset)

        result = await self.db.fetch_all(query, params)
        return [self._row_to_message(row) for row in result.data]

    async def get_user_messages(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Message]:
        """Get all messages for user"""
        if self.db.db_type == DatabaseType.POSTGRESQL:
            query = """
                SELECT * FROM conversation_messages
                WHERE user_id = $1
                ORDER BY timestamp DESC
                LIMIT $2 OFFSET $3
            """
            params = (user_id, limit, offset)
        else:
            query = """
                SELECT * FROM conversation_messages
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            """
            params = (user_id, limit, offset)

        result = await self.db.fetch_all(query, params)
        return [self._row_to_message(row) for row in result.data]

    async def search_messages(
        self,
        query: str,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Message]:
        """Search messages by content"""
        conditions = ["content LIKE ?"]
        params = [f"%{query}%"]

        if thread_id:
            conditions.append("thread_id = ?")
            params.append(thread_id)

        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)

        where_clause = " AND ".join(conditions)

        if self.db.db_type == DatabaseType.POSTGRESQL:
            # Convert to PostgreSQL placeholders
            pg_where = where_clause
            for i in range(len(params)):
                pg_where = pg_where.replace('?', f'${i+1}', 1)

            sql_query = f"""
                SELECT * FROM conversation_messages
                WHERE {pg_where}
                ORDER BY timestamp DESC
                LIMIT ${len(params) + 1}
            """
            params.append(limit)
        else:
            sql_query = f"""
                SELECT * FROM conversation_messages
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ?
            """
            params.append(limit)

        result = await self.db.fetch_all(sql_query, tuple(params))
        return [self._row_to_message(row) for row in result.data]

    async def delete_thread(self, thread_id: str) -> int:
        """Delete all messages in thread"""
        if self.db.db_type == DatabaseType.POSTGRESQL:
            query = "DELETE FROM conversation_messages WHERE thread_id = $1"
            params = (thread_id,)
        else:
            query = "DELETE FROM conversation_messages WHERE thread_id = ?"
            params = (thread_id,)

        result = await self.db.execute_async(query, params)
        return result.row_count

    async def get_thread_stats(self, thread_id: str) -> Dict[str, Any]:
        """Get thread statistics"""
        if self.db.db_type == DatabaseType.POSTGRESQL:
            query = """
                SELECT
                    COUNT(*) as count,
                    MIN(timestamp) as first_msg,
                    MAX(timestamp) as last_msg,
                    COUNT(DISTINCT user_id) as participant_count
                FROM conversation_messages
                WHERE thread_id = $1
            """
            params = (thread_id,)
        else:
            query = """
                SELECT
                    COUNT(*) as count,
                    MIN(timestamp) as first_msg,
                    MAX(timestamp) as last_msg,
                    COUNT(DISTINCT user_id) as participant_count
                FROM conversation_messages
                WHERE thread_id = ?
            """
            params = (thread_id,)

        result = await self.db.fetch_one(query, params)
        row = result.get_first_row()

        return {
            'message_count': row['count'],
            'first_message': datetime.fromisoformat(row['first_msg']) if row['first_msg'] else None,
            'last_message': datetime.fromisoformat(row['last_msg']) if row['last_msg'] else None,
            'thread_id': thread_id
        }

    async def close(self) -> None:
        """Close database connection"""
        await self.db.disconnect()

    def _row_to_message(self, row: Dict[str, Any]) -> Message:
        """Convert database row to Message"""
        metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
        function_call = json.loads(row['function_call']) if row['function_call'] and isinstance(row['function_call'], str) else row['function_call']

        return Message(
            message_id=row['message_id'],
            thread_id=row['thread_id'],
            user_id=row['user_id'],
            role=MessageRole(row['role']),
            content=row['content'],
            metadata=metadata,
            timestamp=datetime.fromisoformat(row['timestamp']) if isinstance(row['timestamp'], str) else row['timestamp'],
            function_call=function_call,
            function_name=row.get('function_name')
        )
