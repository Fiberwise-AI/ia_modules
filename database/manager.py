"""
Database manager implementation
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional, Any, Dict, List
from .interfaces import ConnectionConfig, DatabaseType

try:
    import psycopg2
    import psycopg2.extras
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

logger = logging.getLogger(__name__)


# DatabaseInterfaceAdapter DELETED - DatabaseManager now handles everything directly


class QueryResult:
    """Simple query result class for migration compatibility"""
    
    def __init__(self, success: bool, data: Optional[Any] = None, error: Optional[str] = None):
        self.success = success
        self.data = data
        self.error = error
        self.error_message = error  # Alias for compatibility


class DatabaseManager:
    """Database manager for handling database operations"""

    def __init__(self, database_url_or_config):
        """
        Initialize DatabaseManager.

        Args:
            database_url_or_config: Either a database URL string or a ConnectionConfig object
        """
        if isinstance(database_url_or_config, ConnectionConfig):
            self.config = database_url_or_config
            self.database_url = database_url_or_config.database_url
        else:
            self.database_url = database_url_or_config
            self.config = ConnectionConfig.from_url(database_url_or_config)
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

            # Run system migrations using DatabaseManager directly
            system_migration_path = Path(__file__).parent / "migrations"
            system_runner = MigrationRunner(self, system_migration_path)
            
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
                    app_runner = MigrationRunner(self, app_path)
                    
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

            elif self.config.database_type == DatabaseType.POSTGRESQL:
                if not PSYCOPG2_AVAILABLE:
                    logger.error("psycopg2 not installed. Install with: pip install psycopg2-binary")
                    return False

                logger.info(f"Connecting to PostgreSQL: {self.database_url}")
                self._connection = psycopg2.connect(
                    self.database_url,
                    cursor_factory=psycopg2.extras.RealDictCursor
                )
                self._connection.autocommit = False
                logger.info(f"✓ Connected to PostgreSQL database")
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
    
    def _translate_sql(self, sql: str) -> str:
        """
        Translate SQL from PostgreSQL syntax to target database syntax.

        PostgreSQL is the canonical source syntax for migrations.
        This method provides automatic translation to other database dialects.

        Supported translations:
        - Data types: SERIAL, BOOLEAN, VARCHAR, UUID, JSONB, TIMESTAMP
        - Functions: NOW(), gen_random_uuid()
        - Type casting: ::type syntax
        - Constraints: PostgreSQL-specific constraint syntax

        Returns:
            Translated SQL string for the target database
        """
        if self.config.database_type == DatabaseType.POSTGRESQL:
            # No translation needed - PostgreSQL is canonical
            return sql

        if self.config.database_type == DatabaseType.SQLITE:
            import re
            result = sql

            # Remove transaction statements (executescript handles transactions)
            result = re.sub(r'^\s*BEGIN\s+TRANSACTION\s*;', '', result, flags=re.IGNORECASE | re.MULTILINE)
            result = re.sub(r'^\s*COMMIT\s*;', '', result, flags=re.IGNORECASE | re.MULTILINE)
            result = re.sub(r'^\s*ROLLBACK\s*;', '', result, flags=re.IGNORECASE | re.MULTILINE)

            # Data types - order matters!
            # SERIAL PRIMARY KEY must be replaced before SERIAL alone
            result = result.replace('SERIAL PRIMARY KEY', 'INTEGER PRIMARY KEY AUTOINCREMENT')
            result = re.sub(r'\bSERIAL\b', 'INTEGER', result)

            # BOOLEAN → INTEGER (SQLite uses 0/1 for boolean)
            result = re.sub(r'\bBOOLEAN\b', 'INTEGER', result)

            # Boolean literals TRUE/FALSE → 1/0
            result = re.sub(r'\bTRUE\b', '1', result)
            result = re.sub(r'\bFALSE\b', '0', result)

            # VARCHAR/CHAR → TEXT (SQLite has flexible TEXT type)
            result = re.sub(r'\bVARCHAR\s*\(\s*\d+\s*\)', 'TEXT', result)
            result = re.sub(r'\bVARCHAR\b', 'TEXT', result)
            result = re.sub(r'\bCHAR\s*\(\s*\d+\s*\)', 'TEXT', result)

            # JSONB/JSON → TEXT (SQLite stores JSON as TEXT)
            result = re.sub(r'\bJSONB\b', 'TEXT', result)
            result = re.sub(r'\bJSON\b', 'TEXT', result)

            # UUID → TEXT (SQLite stores UUIDs as TEXT)
            result = re.sub(r'\bUUID\b', 'TEXT', result)

            # TIMESTAMP → TEXT (SQLite uses TEXT for timestamps)
            result = re.sub(r'\bTIMESTAMP\b', 'TEXT', result)

            # Functions
            result = re.sub(r'\bNOW\(\)', 'CURRENT_TIMESTAMP', result)
            result = re.sub(r'\bCURRENT_DATE\b', "date('now')", result)
            result = re.sub(r'\bCURRENT_TIME\b', "time('now')", result)

            # gen_random_uuid() → remove (SQLite doesn't support function defaults in DDL)
            # Applications should generate UUIDs before insert
            result = re.sub(r'DEFAULT\s+gen_random_uuid\(\)', '', result)
            result = re.sub(r'gen_random_uuid\(\)', "lower(hex(randomblob(16)))", result)

            # PostgreSQL type casting (::type) → remove for SQLite
            # Examples: '{}'::jsonb, 'text'::varchar
            result = re.sub(r"'([^']*)'::jsonb", r"'\1'", result)
            result = re.sub(r'::jsonb\b', '', result)
            result = re.sub(r'::json\b', '', result)
            result = re.sub(r'::varchar\b', '', result)
            result = re.sub(r'::text\b', '', result)
            result = re.sub(r'::uuid\b', '', result)

            # Remove SQL comments (-- style) before collapsing whitespace
            result = re.sub(r'--[^\n]*\n', '\n', result)

            # Clean up multiple spaces/tabs but KEEP newlines (needed for executescript)
            result = re.sub(r'[ \t]+', ' ', result)
            result = re.sub(r'\n\s+', '\n', result)

            return result

        # MySQL translation (future)
        if self.config.database_type == DatabaseType.MYSQL:
            import re
            result = sql
            # AUTO_INCREMENT instead of SERIAL
            result = result.replace('SERIAL PRIMARY KEY', 'INTEGER PRIMARY KEY AUTO_INCREMENT')
            result = re.sub(r'\bSERIAL\b', 'INTEGER AUTO_INCREMENT', result)
            # TINYINT(1) instead of BOOLEAN
            result = re.sub(r'\bBOOLEAN\b', 'TINYINT(1)', result)
            # JSON instead of JSONB
            result = result.replace('JSONB', 'JSON')
            # Functions
            result = result.replace('gen_random_uuid()', 'UUID()')
            return result

        # For unknown databases, return as-is and hope for the best
        logger.warning(f"No SQL translation rules for {self.config.database_type}, using SQL as-is")
        return sql

    def _convert_params(self, query: str, params: Optional[Dict] = None):
        """
        Convert named parameters (:param) to database-specific format.

        PostgreSQL: :param → %(param)s with dict params
        SQLite: :param → ? with tuple params (positional)

        Returns: (converted_query, converted_params)
        """
        if not params:
            return query, None

        if self.config.database_type == DatabaseType.POSTGRESQL:
            # PostgreSQL uses %(name)s format with dict
            new_query = query
            for key in params.keys():
                new_query = new_query.replace(f":{key}", f"%({key})s")
            return new_query, params

        elif self.config.database_type == DatabaseType.SQLITE:
            # SQLite uses ? with positional tuple
            new_query = query
            param_list = []
            for key, value in params.items():
                new_query = new_query.replace(f":{key}", "?", 1)
                param_list.append(value)
            return new_query, tuple(param_list)

        else:
            return query, tuple(params.values()) if params else None

    async def execute_async(self, query: str, params: Optional[Dict] = None) -> Any:
        """
        Async-compatible execute method.

        Args:
            query: SQL query with named parameters like :param_name
            params: Dict of parameters {"param_name": value}
        """
        return self.execute(query, params)

    # Deprecated alias for backwards compatibility
    async def execute_query(self, query: str, params: Optional[Dict] = None) -> Any:
        """Deprecated: Use execute_async() instead."""
        return await self.execute_async(query, params)

    def _execute_raw(self, query: str, params: Optional[Dict] = None):
        """
        Internal method to execute query and return cursor.

        Args:
            query: SQL query with :param_name placeholders
            params: Dict like {"param_name": "value"}

        Returns:
            cursor object
        """
        if not self._connection:
            raise RuntimeError("Database not connected")

        # Translate SQL to target database dialect
        translated_query = self._translate_sql(query)

        # Convert named params to database-specific format
        converted_query, converted_params = self._convert_params(translated_query, params)

        cursor = self._connection.cursor()
        if converted_params:
            cursor.execute(converted_query, converted_params)
        else:
            cursor.execute(converted_query)

        return cursor

    def execute(self, query: str, params: Optional[Dict] = None) -> QueryResult:
        """
        Execute a query with named parameters.

        Args:
            query: SQL query with :param_name placeholders
            params: Dict like {"param_name": "value"}

        Returns:
            QueryResult with success status
        """
        try:
            cursor = self._execute_raw(query, params)
            self._connection.commit()

            return QueryResult(
                success=True,
                data=[],
                error=None
            )
        except Exception as e:
            # Rollback on error to clean up transaction state
            if self._connection:
                try:
                    self._connection.rollback()
                except Exception:
                    pass  # Ignore rollback errors
            return QueryResult(
                success=False,
                data=[],
                error=str(e)
            )

    def fetch_one(self, query: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Fetch one row with named parameters.

        Args:
            query: SQL with :param_name placeholders
            params: Dict like {"param_name": "value"}
        """
        try:
            cursor = self._execute_raw(query, params)
            row = cursor.fetchone()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"fetch_one failed: {e}")
            return None

    def fetch_all(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """
        Fetch all rows with named parameters.

        Args:
            query: SQL with :param_name placeholders
            params: Dict like {"param_name": "value"}
        """
        try:
            cursor = self._execute_raw(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"fetch_all failed: {e}")
            return []
    
    def create_table(self, table_name: str, schema: str):
        """Create a table with the given schema"""
        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({schema})"
        self.execute(query)
        logger.info(f"Created table: {table_name}")
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists - works on PostgreSQL and SQLite"""
        try:
            if self.config.database_type == DatabaseType.POSTGRESQL:
                result = self.fetch_one("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = :table_name
                    )
                """, {"table_name": table_name})
                return result and result.get('exists', False)
            else:  # SQLite
                result = self.fetch_one(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=:table_name",
                    {"table_name": table_name}
                )
                return result is not None
        except Exception as e:
            logger.error(f"table_exists failed: {e}")
            if self.config.database_type == DatabaseType.POSTGRESQL and self._connection:
                try:
                    self._connection.rollback()
                except:
                    pass
            return False

    async def execute_script(self, script: str) -> 'QueryResult':
        """Execute a SQL script (multiple statements)"""
        try:
            # Translate SQL to target database dialect
            translated_script = self._translate_sql(script)

            if self.config.database_type == DatabaseType.SQLITE:
                # SQLite has executescript() which handles multiple statements
                self._connection.executescript(translated_script)
                self._connection.commit()
            else:
                # For PostgreSQL/MySQL, execute statements one by one
                statements = [stmt.strip() for stmt in translated_script.split(';') if stmt.strip()]
                for statement in statements:
                    self.execute(statement)

            return QueryResult(success=True)
        except Exception as e:
            logger.error(f"Script execution failed: {e}")
            if self.config.database_type == DatabaseType.POSTGRESQL and self._connection:
                try:
                    self._connection.rollback()
                except:
                    pass
            return QueryResult(success=False, error=str(e))

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()