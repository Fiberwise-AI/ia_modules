"""
Simple SQLite implementation for testing and lightweight deployments.
"""

import aiosqlite
from typing import Dict, Any, List, Optional, Tuple
from ia_modules.database.interfaces import DatabaseInterface, DatabaseType, QueryResult, create_query_result, create_error_result


class SimpleSQLite(DatabaseInterface):
    """Simple async SQLite implementation."""

    def __init__(self, database_path: str = ":memory:"):
        super().__init__(f"sqlite:///{database_path}", DatabaseType.SQLITE)
        self.database_path = database_path
        self.connection = None

    async def connect(self) -> bool:
        """Connect to SQLite database."""
        try:
            self.connection = await aiosqlite.connect(self.database_path)
            self.connection.row_factory = aiosqlite.Row
            self.is_connected = True
            return True
        except Exception as e:
            return False

    async def disconnect(self) -> None:
        """Disconnect from database."""
        if self.connection:
            await self.connection.close()
            self.is_connected = False

    async def execute_query(self, query: str, parameters: Optional[Tuple] = None) -> QueryResult:
        """Execute a SQL query."""
        try:
            cursor = await self.connection.execute(query, parameters or ())
            await self.connection.commit()

            # For SELECT queries, fetch results
            if query.strip().upper().startswith("SELECT"):
                rows = await cursor.fetchall()
                data = [dict(row) for row in rows]
                return create_query_result(success=True, data=data)

            return create_query_result(success=True, data=[])
        except Exception as e:
            return create_error_result(str(e))

    async def fetch_all(self, query: str, parameters: Optional[Tuple] = None) -> QueryResult:
        """Fetch all results."""
        return await self.execute_query(query, parameters)

    async def fetch_one(self, query: str, parameters: Optional[Tuple] = None) -> QueryResult:
        """Fetch one result."""
        try:
            cursor = await self.connection.execute(query, parameters or ())
            row = await cursor.fetchone()
            if row:
                data = [dict(row)]
                return create_query_result(success=True, data=data)
            return create_query_result(success=True, data=[])
        except Exception as e:
            return create_error_result(str(e))

    async def table_exists(self, table_name: str) -> bool:
        """Check if table exists."""
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
        result = await self.fetch_one(query, (table_name,))
        return len(result.data) > 0
