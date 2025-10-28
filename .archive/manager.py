"""
Database manager implementation
"""

import sqlite3
import logging
import re

from pathlib import Path
from typing import Optional, Any, Dict, List
from .interfaces import ConnectionConfig, DatabaseType, QueryResult

try:
    import psycopg2
    import psycopg2.extras
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

logger = logging.getLogger(__name__)


# DatabaseInterfaceAdapter DELETED - DatabaseManager now handles everything directly


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

            elif self.config.database_type == DatabaseType.MYSQL:
                import pymysql
                import pymysql.cursors
                from urllib.parse import urlparse

                # Parse the database URL
                parsed = urlparse(self.database_url)

                self._connection = pymysql.connect(
                    host=self.config.host or parsed.hostname or 'localhost',
                    port=self.config.port or parsed.port or 3306,
                    user=self.config.username or parsed.username,
                    password=self.config.password or parsed.password,
                    database=self.config.database_name or parsed.path.lstrip('/'),
                    cursorclass=pymysql.cursors.DictCursor,
                    autocommit=False
                )
                logger.info(f"✓ Connected to MySQL database")
                return True

            elif self.config.database_type == DatabaseType.MSSQL:
                import pyodbc
                from urllib.parse import urlparse, parse_qs

                # Check if this is a raw ODBC connection string (starts with DRIVER={...})
                if self.database_url.startswith('DRIVER=') or self.database_url.startswith('Driver='):
                    # Raw ODBC connection string provided
                    self._connection = pyodbc.connect(self.database_url)
                else:
                    # Parse the database URL (SQLAlchemy format: mssql+pyodbc://user:pass@host:port/db?driver=...)
                    parsed = urlparse(self.database_url)
                    query_params = parse_qs(parsed.query) if parsed.query else {}

                    # Get driver from query params or auto-detect
                    driver = None
                    if 'driver' in query_params:
                        driver = query_params['driver'][0]
                    else:
                        # Detect available driver (prefer 17 over 18, as 18 has TLS issues)
                        available_drivers = pyodbc.drivers()
                        for preferred in ['ODBC Driver 17 for SQL Server', 'ODBC Driver 18 for SQL Server', 'SQL Server']:
                            if preferred in available_drivers:
                                driver = preferred
                                break

                    if not driver:
                        raise RuntimeError("No MSSQL ODBC driver found. Install ODBC Driver 17 or 18 for SQL Server")

                    # Build ODBC connection string
                    conn_str_parts = [
                        f"DRIVER={{{driver}}}",
                        f"SERVER={self.config.host or parsed.hostname or 'localhost'},{self.config.port or parsed.port or 1433}",
                        f"DATABASE={self.config.database_name or parsed.path.lstrip('/') or 'master'}",
                        f"UID={self.config.username or parsed.username}",
                        f"PWD={self.config.password or parsed.password}"
                    ]

                    # Add additional query params (like TrustServerCertificate)
                    for key, values in query_params.items():
                        if key.lower() != 'driver':  # Skip driver, already added
                            conn_str_parts.append(f"{key}={values[0]}")

                    # If TrustServerCertificate not provided, add it by default
                    if 'TrustServerCertificate' not in query_params and 'trustservercertificate' not in query_params:
                        conn_str_parts.append("TrustServerCertificate=yes")

                    conn_str = ";".join(conn_str_parts)
                    self._connection = pyodbc.connect(conn_str)

                self._connection.autocommit = False
                logger.info(f"✓ Connected to MSSQL database")
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
            # Translate SQLite-style to PostgreSQL if needed

            result = sql

            # INTEGER PRIMARY KEY (SQLite auto-increment) → SERIAL PRIMARY KEY (PostgreSQL)
            result = re.sub(r'\bINTEGER\s+PRIMARY\s+KEY\s+AUTOINCREMENT\b', 'SERIAL PRIMARY KEY', result, flags=re.IGNORECASE)
            result = re.sub(r'\bINTEGER\s+PRIMARY\s+KEY\b', 'SERIAL PRIMARY KEY', result, flags=re.IGNORECASE)

            return result

        if self.config.database_type == DatabaseType.SQLITE:
            
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

        # MySQL translation
        if self.config.database_type == DatabaseType.MYSQL:
            result = sql

            # AUTO_INCREMENT instead of SERIAL
            result = result.replace('SERIAL PRIMARY KEY', 'INT PRIMARY KEY AUTO_INCREMENT')
            result = re.sub(r'\bSERIAL\b', 'INT AUTO_INCREMENT', result)

            # Note: Do NOT add AUTO_INCREMENT to INTEGER PRIMARY KEY
            # In PostgreSQL canonical syntax, INTEGER PRIMARY KEY does NOT auto-increment
            # Only SERIAL PRIMARY KEY auto-increments

            # TINYINT(1) instead of BOOLEAN
            result = re.sub(r'\bBOOLEAN\b', 'TINYINT(1)', result)

            # Boolean literals TRUE/FALSE → 1/0
            result = re.sub(r'\bTRUE\b', '1', result)
            result = re.sub(r'\bFALSE\b', '0', result)

            # JSON instead of JSONB
            result = re.sub(r'\bJSONB\b', 'JSON', result)

            # UUID → CHAR(36) (MySQL stores UUIDs as strings)
            result = re.sub(r'\bUUID\b', 'CHAR(36)', result)

            # MySQL doesn't support DEFAULT with functions except for TIMESTAMP columns
            # Remove DEFAULT gen_random_uuid() - apps must provide UUIDs
            result = re.sub(r'\s+DEFAULT\s+gen_random_uuid\(\)', '', result, flags=re.IGNORECASE)

            # MySQL doesn't support DEFAULT on JSON/TEXT columns
            # Remove DEFAULT for JSON columns (common pattern: DEFAULT '{}'::jsonb or DEFAULT '{}')
            result = re.sub(r'\s+DEFAULT\s+\'[^\']*\'::jsonb', '', result, flags=re.IGNORECASE)
            result = re.sub(r'(\bJSON\b[^,\)]*)\s+DEFAULT\s+\'[^\']*\'', r'\1', result, flags=re.IGNORECASE)

            # TIMESTAMP handling (MySQL has different default behavior)
            # Keep TIMESTAMP as-is, it works in MySQL

            # Functions (for queries, not for defaults)
            result = re.sub(r'gen_random_uuid\(\)', 'UUID()', result)
            # NOW() works in MySQL, keep it

            # PostgreSQL type casting (::type) → remove for MySQL
            result = re.sub(r"'([^']*)'::jsonb", r"'\1'", result)
            result = re.sub(r'::jsonb\b', '', result)
            result = re.sub(r'::json\b', '', result)
            result = re.sub(r'::varchar\b', '', result)
            result = re.sub(r'::text\b', '', result)
            result = re.sub(r'::uuid\b', '', result)

            # MySQL doesn't support CREATE INDEX IF NOT EXISTS
            # Remove IF NOT EXISTS from index creation
            result = re.sub(r'CREATE\s+INDEX\s+IF\s+NOT\s+EXISTS\s+', 'CREATE INDEX ', result, flags=re.IGNORECASE)

            return result

        # MSSQL translation
        if self.config.database_type == DatabaseType.MSSQL:
            result = sql

            # IDENTITY instead of SERIAL
            result = result.replace('SERIAL PRIMARY KEY', 'INT PRIMARY KEY IDENTITY(1,1)')
            result = re.sub(r'\bSERIAL\b', 'INT IDENTITY(1,1)', result)

            # Note: Do NOT add IDENTITY to INTEGER PRIMARY KEY
            # In PostgreSQL canonical syntax, INTEGER PRIMARY KEY does NOT auto-increment
            # Only SERIAL PRIMARY KEY auto-increments

            # BIT instead of BOOLEAN
            result = re.sub(r'\bBOOLEAN\b', 'BIT', result)

            # Boolean literals TRUE/FALSE → 1/0
            result = re.sub(r'\bTRUE\b', '1', result)
            result = re.sub(r'\bFALSE\b', '0', result)

            # NVARCHAR(MAX) instead of JSONB/JSON
            result = re.sub(r'\bJSONB\b', 'NVARCHAR(MAX)', result)
            result = re.sub(r'\bJSON\b', 'NVARCHAR(MAX)', result)

            # UNIQUEIDENTIFIER instead of UUID
            result = re.sub(r'\bUUID\b', 'UNIQUEIDENTIFIER', result)

            # VARCHAR → NVARCHAR for better Unicode support
            result = re.sub(r'\bVARCHAR\s*\((\d+)\)', r'NVARCHAR(\1)', result)
            result = re.sub(r'\bVARCHAR\b', 'NVARCHAR(MAX)', result)
            result = re.sub(r'\bTEXT\b', 'NVARCHAR(MAX)', result)

            # TIMESTAMP → DATETIME2
            result = re.sub(r'\bTIMESTAMP\b', 'DATETIME2', result)

            # Functions
            result = re.sub(r'\bNOW\(\)', 'GETDATE()', result)
            result = re.sub(r'gen_random_uuid\(\)', 'NEWID()', result)
            result = re.sub(r'\bCURRENT_TIMESTAMP\b', 'GETDATE()', result)

            # PostgreSQL type casting (::type) → CAST syntax
            result = re.sub(r"'([^']*)'::jsonb", r"'\1'", result)
            result = re.sub(r'::jsonb\b', '', result)
            result = re.sub(r'::json\b', '', result)
            result = re.sub(r'::varchar\b', '', result)
            result = re.sub(r'::text\b', '', result)
            result = re.sub(r'::uuid\b', '', result)

            # MSSQL doesn't support ON DELETE SET NULL for self-referencing FKs with CASCADE
            # Change to NO ACTION to avoid conflicts
            result = re.sub(r'ON\s+DELETE\s+SET\s+NULL', 'ON DELETE NO ACTION', result, flags=re.IGNORECASE)
            result = re.sub(r'ON\s+DELETE\s+CASCADE', 'ON DELETE NO ACTION', result, flags=re.IGNORECASE)

            # CREATE TABLE IF NOT EXISTS → MSSQL conditional syntax
            # MSSQL doesn't support IF NOT EXISTS in CREATE TABLE
            # Use a line-by-line approach to wrap each CREATE TABLE statement
            lines = result.split('\n')
            processed_lines = []
            i = 0
            while i < len(lines):
                line = lines[i]
                # Check if this line starts a CREATE TABLE IF NOT EXISTS
                if re.match(r'^\s*CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+(\w+)', line, re.IGNORECASE):
                    table_name_match = re.search(r'CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+(\w+)', line, re.IGNORECASE)
                    table_name = table_name_match.group(1)

                    # Remove IF NOT EXISTS from this line
                    line = re.sub(r'IF\s+NOT\s+EXISTS\s+', '', line, flags=re.IGNORECASE)

                    # Add the IF NOT EXISTS wrapper
                    processed_lines.append(f"IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[{table_name}]') AND type = 'U')")
                    processed_lines.append("BEGIN")
                    processed_lines.append("    " + line.strip())

                    # Collect all lines until we find the closing );
                    i += 1
                    paren_count = line.count('(') - line.count(')')
                    while i < len(lines):
                        current_line = lines[i]
                        paren_count += current_line.count('(') - current_line.count(')')
                        processed_lines.append("    " + current_line.strip())

                        # Check if we've reached the end of the CREATE TABLE statement
                        if paren_count <= 0 and (');' in current_line or current_line.strip().endswith(')')):
                            # Remove semicolon if present
                            if processed_lines[-1].rstrip().endswith(');'):
                                processed_lines[-1] = processed_lines[-1].rstrip()[:-1]  # Remove the semicolon
                            elif processed_lines[-1].rstrip().endswith(';'):
                                processed_lines[-1] = processed_lines[-1].rstrip()[:-1]  # Remove the semicolon
                            processed_lines.append("END")
                            break
                        i += 1
                else:
                    processed_lines.append(line)
                i += 1

            result = '\n'.join(processed_lines)

            # Also handle CREATE INDEX IF NOT EXISTS (MSSQL doesn't support it either)
            result = re.sub(r'CREATE\s+INDEX\s+IF\s+NOT\s+EXISTS\s+', 'CREATE INDEX ', result, flags=re.IGNORECASE)

            # LIMIT/OFFSET → OFFSET/FETCH NEXT
            # MSSQL uses: OFFSET x ROWS FETCH NEXT y ROWS ONLY
            # PostgreSQL uses: LIMIT y OFFSET x

            # Handle LIMIT with OFFSET
            result = re.sub(
                r'\bLIMIT\s+(\d+)\s+OFFSET\s+(\d+)',
                r'OFFSET \2 ROWS FETCH NEXT \1 ROWS ONLY',
                result,
                flags=re.IGNORECASE
            )

            # Handle just LIMIT (no OFFSET)
            result = re.sub(
                r'\bLIMIT\s+(\d+)(?!\s+OFFSET)',
                r'OFFSET 0 ROWS FETCH NEXT \1 ROWS ONLY',
                result,
                flags=re.IGNORECASE
            )

            return result

        # For unknown databases, return as-is and hope for the best
        logger.warning(f"No SQL translation rules for {self.config.database_type}, using SQL as-is")
        return sql

    def _convert_params(self, query: str, params: Optional[Dict] = None):
        """
        Convert named parameters (:param) to database-specific format.

        PostgreSQL: :param → %(param)s with dict params
        MySQL: :param → %s with tuple params (positional)
        MSSQL: :param → ? with tuple params (positional)
        SQLite: :param → ? with tuple params (positional)

        Returns: (converted_query, converted_params)
        """
        if not params:
            return query, None

        # Handle case where params is passed as tuple (legacy code)
        if isinstance(params, (tuple, list)):
            import traceback
            logger.error(f"Parameters passed as {type(params).__name__}, expected dict")
            logger.error(f"Query: {query[:200]}")
            logger.error(f"Params: {params}")
            logger.error(f"Stack trace:\n{''.join(traceback.format_stack())}")
            # For backwards compatibility with positional params
            if self.config.database_type in [DatabaseType.SQLITE, DatabaseType.MYSQL, DatabaseType.MSSQL]:
                return query, tuple(params) if isinstance(params, list) else params
            else:
                raise TypeError(f"Parameters must be a dict for {self.config.database_type}, got {type(params).__name__}")

        if self.config.database_type == DatabaseType.POSTGRESQL:
            # PostgreSQL uses %(name)s format with dict
            new_query = query
            for key in params.keys():
                new_query = new_query.replace(f":{key}", f"%({key})s")

            # Convert booleans to integers if column type is INTEGER
            # PostgreSQL is strict about types - bool can't go into INTEGER columns
            converted_params = {}
            for key, value in params.items():
                if isinstance(value, bool):
                    converted_params[key] = 1 if value else 0
                else:
                    converted_params[key] = value

            return new_query, converted_params

        elif self.config.database_type == DatabaseType.MYSQL:
            # MySQL uses %s with positional tuple
            # IMPORTANT: Build param_list in order of appearance in query, not dict iteration order
            import re

            # Find all :param_name patterns in query in order of appearance
            param_pattern = r':(\w+)'
            param_names_in_order = re.findall(param_pattern, query)

            # Build param list in query order
            param_list = []
            for param_name in param_names_in_order:
                if param_name not in params:
                    raise ValueError(f"Parameter :{param_name} used in query but not provided in params")
                value = params[param_name]
                # Convert booleans to integers for MySQL INTEGER columns
                if isinstance(value, bool):
                    param_list.append(1 if value else 0)
                else:
                    param_list.append(value)

            # Replace all :param_name with %s in one pass
            new_query = re.sub(param_pattern, '%s', query)

            return new_query, tuple(param_list)

        elif self.config.database_type in [DatabaseType.SQLITE, DatabaseType.MSSQL]:
            # SQLite and MSSQL use ? with positional tuple
            # IMPORTANT: Build param_list in order of appearance in query, not dict iteration order
            import re

            # Find all :param_name patterns in query in order of appearance
            param_pattern = r':(\w+)'
            param_names_in_order = re.findall(param_pattern, query)

            # Build param list in query order
            param_list = []
            for param_name in param_names_in_order:
                if param_name not in params:
                    raise ValueError(f"Parameter :{param_name} used in query but not provided in params")
                value = params[param_name]
                # Convert booleans to integers for SQLite
                if self.config.database_type == DatabaseType.SQLITE and isinstance(value, bool):
                    param_list.append(1 if value else 0)
                else:
                    param_list.append(value)

            # Replace all :param_name with ? in one pass
            new_query = re.sub(param_pattern, '?', query)

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

    def execute(self, query: str, params: Optional[Dict] = None):
        """
        Execute a query with named parameters.

        For SELECT queries, returns List[Dict] with results.
        For INSERT/UPDATE/DELETE, returns List[Dict] (empty) for success.

        Args:
            query: SQL query with :param_name placeholders
            params: Dict like {"param_name": "value"}

        Returns:
            List[Dict]: Query results (empty list for non-SELECT queries)
        """
        try:
            cursor = self._execute_raw(query, params)

            # Check if this is a SELECT query
            query_upper = query.strip().upper()
            if query_upper.startswith('SELECT') or query_upper.startswith('SHOW') or query_upper.startswith('DESCRIBE'):
                # Fetch results for SELECT queries
                rows = cursor.fetchall()
                if not rows:
                    return []

                # Convert rows to dicts
                # For pyodbc (MSSQL), need to use cursor.description to get column names
                if self.config.database_type == DatabaseType.MSSQL:
                    columns = [column[0] for column in cursor.description]
                    return [dict(zip(columns, row)) for row in rows]
                else:
                    # SQLite and MySQL return dict-like rows
                    return [dict(row) for row in rows]
            else:
                # For INSERT/UPDATE/DELETE, just commit
                self._connection.commit()
                return []

        except Exception as e:
            # Rollback on error to clean up transaction state
            if self._connection:
                try:
                    self._connection.rollback()
                    logger.debug("Rolled back transaction after error")
                except Exception:
                    pass  # Ignore rollback errors
            logger.error(f"Query execution failed: {e}")
            raise

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
            if not row:
                return None

            # Handle different cursor types
            if self.config.database_type == DatabaseType.MSSQL:
                # pyodbc Row object - convert to dict using column names
                return {cursor.description[i][0]: row[i] for i in range(len(row))}
            else:
                # PostgreSQL/MySQL dict cursor or SQLite Row
                return dict(row)
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

            # Handle different cursor types
            if self.config.database_type == DatabaseType.MSSQL:
                # pyodbc Row objects - convert to dicts
                return [{cursor.description[i][0]: row[i] for i in range(len(row))} for row in rows]
            else:
                # PostgreSQL/MySQL dict cursor or SQLite Row
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
        """Check if a table exists - works on all supported databases"""
        try:
            if self.config.database_type == DatabaseType.POSTGRESQL:
                result = self.fetch_one("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = :table_name
                    )
                """, {"table_name": table_name})
                return result and result.get('exists', False)

            elif self.config.database_type == DatabaseType.MYSQL:
                result = self.fetch_one("""
                    SELECT COUNT(*) as count FROM information_schema.tables
                    WHERE table_name = :table_name
                    AND table_schema = DATABASE()
                """, {"table_name": table_name})
                return result and result.get('count', 0) > 0

            elif self.config.database_type == DatabaseType.MSSQL:
                result = self.fetch_one("""
                    SELECT COUNT(*) as count FROM information_schema.tables
                    WHERE table_name = :table_name
                """, {"table_name": table_name})
                return result and result.get('count', 0) > 0

            else:  # SQLite
                result = self.fetch_one(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=:table_name",
                    {"table_name": table_name}
                )
                return result is not None

        except Exception as e:
            logger.error(f"table_exists failed: {e}")
            if self.config.database_type in [DatabaseType.POSTGRESQL, DatabaseType.MYSQL, DatabaseType.MSSQL] and self._connection:
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
                # For PostgreSQL/MySQL/MSSQL, execute statements one by one
                statements = [stmt.strip() for stmt in translated_script.split(';') if stmt.strip()]
                for statement in statements:
                    # Skip empty statements
                    if not statement:
                        continue
                    self.execute(statement)

            return QueryResult(success=True, data=[], row_count=0)
        except Exception as e:
            logger.error(f"Script execution failed: {e}")
            if self.config.database_type in [DatabaseType.POSTGRESQL, DatabaseType.MYSQL, DatabaseType.MSSQL] and self._connection:
                try:
                    self._connection.rollback()
                except:
                    pass
            return QueryResult(success=False, data=[], row_count=0, error_message=str(e))

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()