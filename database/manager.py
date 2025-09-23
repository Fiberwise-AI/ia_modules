"""
Database manager implementation
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional, Any, Dict, List
from .interfaces import ConnectionConfig, DatabaseType

logger = logging.getLogger(__name__)


class DatabaseInterfaceAdapter:
    """Adapter to make DatabaseManager compatible with migration system interface"""
    
    def __init__(self, db_manager: 'DatabaseManager'):
        self.db_manager = db_manager
    
    async def table_exists(self, table_name: str) -> bool:
        """Check if a table exists"""
        try:
            result = self.db_manager.fetch_one(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,)
            )
            return result is not None
        except Exception:
            return False
    
    async def create_table(self, table_name: str, schema: Dict[str, str]) -> bool:
        """Create a table with the given schema"""
        try:
            # Convert schema dict to SQL
            columns = []
            for column_name, column_def in schema.items():
                columns.append(f"{column_name} {column_def}")
            
            schema_sql = ", ".join(columns)
            self.db_manager.create_table(table_name, schema_sql)
            return True
        except Exception as e:
            logger.error(f"Failed to create table {table_name}: {e}")
            return False
    
    async def fetch_all(self, query: str, params: Optional[Dict] = None) -> 'QueryResult':
        """Fetch all rows from query"""
        try:
            # Convert named parameters to tuple if needed
            if params:
                # Simple parameter substitution for named parameters
                for key, value in params.items():
                    query = query.replace(f":{key}", "?")
                param_tuple = tuple(params.values())
            else:
                param_tuple = None
            
            data = self.db_manager.fetch_all(query, param_tuple)
            return QueryResult(success=True, data=data)
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return QueryResult(success=False, error=str(e))
    
    async def execute_query(self, query: str, params: Optional[Dict] = None) -> 'QueryResult':
        """Execute a query"""
        try:
            # Convert named parameters to tuple if needed
            if params:
                for key, value in params.items():
                    query = query.replace(f":{key}", "?")
                param_tuple = tuple(params.values())
            else:
                param_tuple = None
            
            self.db_manager.execute(query, param_tuple)
            return QueryResult(success=True)
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return QueryResult(success=False, error=str(e))
    
    async def execute_script(self, script: str) -> 'QueryResult':
        """Execute a SQL script"""
        try:
            # Split script into individual statements and execute each
            statements = [stmt.strip() for stmt in script.split(';') if stmt.strip()]
            for statement in statements:
                self.db_manager.execute(statement)
            return QueryResult(success=True)
        except Exception as e:
            logger.error(f"Script execution failed: {e}")
            return QueryResult(success=False, error=str(e))


class QueryResult:
    """Simple query result class for migration compatibility"""
    
    def __init__(self, success: bool, data: Optional[Any] = None, error: Optional[str] = None):
        self.success = success
        self.data = data
        self.error = error


class DatabaseManager:
    """Database manager for handling database operations"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.config = ConnectionConfig.from_url(database_url)
        self._connection = None
        
    async def initialize(self, apply_schema: bool = True, app_migration_paths: Optional[List[str]] = None) -> bool:
        """
        Initialize the database connection and optionally apply schema migrations
        
        Args:
            apply_schema: Whether to apply migrations
            app_migration_paths: List of paths to app-specific migration directories
        """
        if not self.connect():
            return False
        
        if not apply_schema:
            return True
        
        try:
            # Import migrations module here to avoid circular imports
            from .migrations import MigrationRunner
            
            # Create a simple database interface adapter for the migration runner
            db_interface = DatabaseInterfaceAdapter(self)
            
            # Run system migrations first
            system_migration_path = Path(__file__).parent / "migrations"
            system_runner = MigrationRunner(db_interface, system_migration_path)
            
            logger.info("Running system migrations...")
            if not await system_runner.run_pending_migrations():
                logger.error("System migrations failed")
                return False
            
            # Run app-specific migrations if provided
            if app_migration_paths:
                for app_path_str in app_migration_paths:
                    app_path = Path(app_path_str)
                    if not app_path.exists():
                        logger.warning(f"App migration path does not exist: {app_path}")
                        continue
                    
                    logger.info(f"Running app migrations from: {app_path}")
                    app_runner = MigrationRunner(db_interface, app_path)
                    
                    if not await app_runner.run_pending_migrations():
                        logger.error(f"App migrations failed for: {app_path}")
                        return False
            
            logger.info("All migrations completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Migration execution failed: {e}")
            return False
        
    def connect(self) -> bool:
        """Connect to the database"""
        try:
            if self.config.database_type == DatabaseType.SQLITE:
                # Extract path from sqlite:///path or just use as path
                if self.database_url.startswith("sqlite:///"):
                    db_path = self.database_url[10:]  # Remove "sqlite:///"
                elif self.database_url.startswith("sqlite://"):
                    db_path = self.database_url[9:]   # Remove "sqlite://"
                else:
                    db_path = self.database_url
                
                # Create directory if it doesn't exist
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
                
                self._connection = sqlite3.connect(db_path)
                self._connection.row_factory = sqlite3.Row
                logger.info(f"Connected to SQLite database: {db_path}")
                return True
            else:
                logger.error(f"Database type {self.config.database_type} not implemented yet")
                return False
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the database"""
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("Disconnected from database")
    
    async def close(self):
        """Async-compatible close method"""
        self.disconnect()
    
    async def execute_query(self, query: str, params: Optional[List] = None) -> Any:
        """Async-compatible execute method"""
        return self.execute(query, tuple(params) if params else None)
    
    def execute(self, query: str, params: Optional[tuple] = None) -> Any:
        """Execute a query"""
        if not self._connection:
            raise RuntimeError("Database not connected")
        
        cursor = self._connection.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        self._connection.commit()
        return cursor
    
    def fetch_one(self, query: str, params: Optional[tuple] = None) -> Optional[Dict]:
        """Fetch one row from query result"""
        cursor = self.execute(query, params)
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def fetch_all(self, query: str, params: Optional[tuple] = None) -> List[Dict]:
        """Fetch all rows from query result"""
        cursor = self.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def create_table(self, table_name: str, schema: str):
        """Create a table with the given schema"""
        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({schema})"
        self.execute(query)
        logger.info(f"Created table: {table_name}")
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()