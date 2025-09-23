"""
Database providers for IA Modules
"""

import sqlite3
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

from .interfaces import DatabaseInterface, QueryResult, ConnectionConfig, DatabaseType

logger = logging.getLogger(__name__)


def create_database_provider(config: ConnectionConfig) -> DatabaseInterface:
    """Factory function to create database provider based on config."""
    if config.database_type == DatabaseType.SQLITE:
        return SQLiteProvider(config.database_url, config.database_type)
    else:
        raise ValueError(f"Unsupported database type: {config.database_type}")


class SQLiteProvider(DatabaseInterface):
    """SQLite database provider implementation."""
    
    def __init__(self, connection_string: str, db_type: DatabaseType):
        super().__init__(connection_string, db_type)
        self._connection = None
        
    async def initialize(self) -> bool:
        """Initialize the database provider"""
        return await self.connect()
    
    async def shutdown(self) -> None:
        """Shutdown the database provider"""
        await self.disconnect()
    
    async def connect(self) -> bool:
        """Connect to SQLite database."""
        try:
            # Extract path from sqlite:///path or just use as path
            if self.connection_string.startswith("sqlite:///"):
                db_path = self.connection_string[10:]  # Remove "sqlite:///"
            elif self.connection_string.startswith("sqlite://"):
                db_path = self.connection_string[9:]   # Remove "sqlite://"
            else:
                db_path = self.connection_string
            
            # Create directory if it doesn't exist
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            
            self._connection = sqlite3.connect(db_path)
            self._connection.row_factory = sqlite3.Row
            self.is_connected = True
            logger.info(f"Connected to SQLite database: {db_path}")
            return True
            
        except Exception as e:
            logger.error(f"SQLite connection failed: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from SQLite database."""
        if self._connection:
            self._connection.close()
            self._connection = None
            self.is_connected = False
            logger.info("Disconnected from SQLite database")
    
    async def execute_query(self, query: str, parameters: Optional[Union[Tuple, Dict]] = None) -> QueryResult:
        """Execute a SQL query."""
        if not self._connection:
            return QueryResult(success=False, data=[], row_count=0, error_message="Database not connected")
        
        try:
            cursor = self._connection.cursor()
            if parameters:
                cursor.execute(query, parameters)
            else:
                cursor.execute(query)
            
            self._connection.commit()
            
            # For SELECT queries, return data
            if query.strip().upper().startswith('SELECT'):
                rows = cursor.fetchall()
                data = [dict(row) for row in rows]
                return QueryResult(success=True, data=data, row_count=len(data))
            else:
                # For non-SELECT queries, return row count
                return QueryResult(success=True, data=[], row_count=cursor.rowcount)
                
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return QueryResult(success=False, data=[], row_count=0, error_message=str(e))
    
    async def execute_many(self, query: str, parameters_list: List[Tuple]) -> QueryResult:
        """Execute a query multiple times with different parameters."""
        if not self._connection:
            return QueryResult(success=False, data=[], row_count=0, error_message="Database not connected")
        
        try:
            cursor = self._connection.cursor()
            cursor.executemany(query, parameters_list)
            self._connection.commit()
            return QueryResult(success=True, data=[], row_count=cursor.rowcount)
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return QueryResult(success=False, data=[], row_count=0, error_message=str(e))
    
    async def fetch_all(self, query: str, parameters: Optional[Union[Tuple, Dict]] = None) -> QueryResult:
        """Fetch all results from a query."""
        return await self.execute_query(query, parameters)
    
    async def fetch_one(self, query: str, parameters: Optional[Union[Tuple, Dict]] = None) -> QueryResult:
        """Fetch one result from a query."""
        if not self._connection:
            return QueryResult(success=False, data=[], row_count=0, error_message="Database not connected")
        
        try:
            cursor = self._connection.cursor()
            if parameters:
                cursor.execute(query, parameters)
            else:
                cursor.execute(query)
            
            row = cursor.fetchone()
            if row:
                data = [dict(row)]
                return QueryResult(success=True, data=data, row_count=1)
            else:
                return QueryResult(success=True, data=[], row_count=0)
                
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return QueryResult(success=False, data=[], row_count=0, error_message=str(e))
    
    async def create_table(self, table_name: str, schema: Dict[str, str]) -> bool:
        """Create a table with given schema."""
        try:
            columns = []
            for col_name, col_type in schema.items():
                columns.append(f"{col_name} {col_type}")
            
            query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)})"
            result = await self.execute_query(query)
            return result.success
            
        except Exception as e:
            logger.error(f"Table creation failed: {e}")
            return False
    
    async def table_exists(self, table_name: str) -> bool:
        """Check if table exists."""
        try:
            query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
            result = await self.fetch_one(query, (table_name,))
            return result.success and len(result.data) > 0
            
        except Exception as e:
            logger.error(f"Table existence check failed: {e}")
            return False
    
    async def get_table_schema(self, table_name: str) -> Dict[str, str]:
        """Get table schema."""
        try:
            query = f"PRAGMA table_info({table_name})"
            result = await self.fetch_all(query)
            
            if not result.success:
                return {}
            
            schema = {}
            for row in result.data:
                schema[row['name']] = row['type']
            
            return schema
            
        except Exception as e:
            logger.error(f"Schema retrieval failed: {e}")
            return {}
    
    async def execute_script(self, script: str) -> QueryResult:
        """Execute a multi-statement SQL script."""
        if not self._connection:
            return QueryResult(success=False, data=[], row_count=0, error_message="Database not connected")
        
        try:
            cursor = self._connection.cursor()
            # Use executescript to correctly run full SQL scripts including
            # CREATE TRIGGER ... BEGIN ... END; blocks which contain semicolons.
            cursor.executescript(script)
            self._connection.commit()
            return QueryResult(success=True, data=[], row_count=0)
            
        except Exception as e:
            logger.error(f"Script execution failed: {e}")
            return QueryResult(success=False, data=[], row_count=0, error_message=str(e))