"""
NexusQL adapter - wraps nexusql.DatabaseManager to implement DatabaseInterface
"""

from typing import Optional, Dict, List, Any
from pathlib import Path
import logging

try:
    from nexusql import DatabaseManager as NexuSQLManager
    NEXUSQL_AVAILABLE = True
except ImportError:
    NEXUSQL_AVAILABLE = False
    NexuSQLManager = None

from ..interfaces import DatabaseInterface, QueryResult, create_query_result, create_error_result

logger = logging.getLogger(__name__)


class NexuSQLAdapter(DatabaseInterface):
    """
    Adapter that wraps nexusql.DatabaseManager to implement DatabaseInterface.

    Features:
    - Multi-database support (SQLite, PostgreSQL, MySQL, MSSQL)
    - Automatic SQL translation from PostgreSQL canonical syntax
    - Named parameters with :param syntax
    - Built-in migration system
    """

    def __init__(self, database_url: str):
        """
        Initialize NexusQL adapter.

        Args:
            database_url: Database connection URL
        """
        if not NEXUSQL_AVAILABLE:
            raise ImportError(
                "nexusql is not installed. Install with: pip install nexusql"
            )

        self.database_url = database_url
        self._db = NexuSQLManager(database_url)

    def connect(self) -> bool:
        """Connect to the database"""
        return self._db.connect()

    def disconnect(self):
        """Disconnect from the database"""
        self._db.disconnect()

    async def close(self):
        """Async-compatible close method"""
        await self._db.close()

    def execute(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """
        Execute a query with named parameters.

        Args:
            query: SQL query with :param_name placeholders
            params: Dict like {"param_name": "value"}

        Returns:
            List[Dict]: Query results (empty list for non-SELECT queries)
        """
        return self._db.execute(query, params)

    async def execute_async(self, query: str, params: Optional[Dict] = None) -> Any:
        """Async-compatible execute method"""
        return await self._db.execute_async(query, params)

    def fetch_one(self, query: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Fetch one row with named parameters.

        Args:
            query: SQL with :param_name placeholders
            params: Dict like {"param_name": "value"}

        Returns:
            Optional[Dict]: Single row or None
        """
        return self._db.fetch_one(query, params)

    def fetch_all(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """
        Fetch all rows with named parameters.

        Args:
            query: SQL with :param_name placeholders
            params: Dict like {"param_name": "value"}

        Returns:
            List[Dict]: All matching rows
        """
        return self._db.fetch_all(query, params)

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists"""
        return self._db.table_exists(table_name)

    async def execute_script(self, script: str) -> QueryResult:
        """Execute a SQL script (multiple statements)"""
        return await self._db.execute_script(script)

    async def initialize(
        self,
        apply_schema: bool = True,
        app_migration_paths: Optional[List[str]] = None
    ) -> bool:
        """
        Initialize the database connection and optionally apply schema migrations

        Args:
            apply_schema: Whether to apply migrations
            app_migration_paths: List of paths to app-specific migration directories

        Returns:
            bool: True if successful
        """
        return await self._db.initialize(apply_schema, app_migration_paths)

    # Expose underlying nexusql instance for advanced features
    @property
    def nexusql(self) -> NexuSQLManager:
        """Access underlying NexusQL instance for advanced features"""
        return self._db

    @property
    def config(self):
        """Access database configuration"""
        return self._db.config

    @property
    def database_type(self):
        """Get database type"""
        return self._db.config.database_type
