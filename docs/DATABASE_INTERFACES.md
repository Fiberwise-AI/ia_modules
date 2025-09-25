# IA Modules - Database Interface System

## Overview

The IA Modules database system provides a comprehensive, multi-backend database abstraction layer designed for enterprise applications. It features async/await support, connection pooling, migration management, and a clean interface abstraction that supports SQLite, PostgreSQL, MySQL, DuckDB, and Cloudflare D1.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Database Interfaces](#database-interfaces)
- [Connection Management](#connection-management)
- [Migration System](#migration-system)
- [Database Manager](#database-manager)
- [Query Result System](#query-result-system)
- [Supported Database Types](#supported-database-types)
- [Configuration Patterns](#configuration-patterns)
- [Best Practices](#best-practices)
- [Performance Considerations](#performance-considerations)

## Architecture Overview

The database system is built around three core concepts:

1. **DatabaseInterface**: Abstract interface for database operations
2. **DatabaseManager**: Concrete implementation with connection management
3. **MigrationRunner**: Schema version management and migrations

```
┌─────────────────────────────────────────────────────────┐
│                Database System Architecture              │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────────────────┐   │
│  │  Application    │  │     Service Registry        │   │
│  │     Layer       │→ │   (Database Injection)      │   │
│  └─────────────────┘  └─────────────────────────────┘   │
│            │                         │                  │
│            ▼                         ▼                  │
│  ┌─────────────────────────────────────────────────┐   │
│  │         DatabaseManager                         │   │
│  │  ┌─────────────────────────────────────────┐    │   │
│  │  │        Connection Pool              │    │   │
│  │  │  ┌───────────┐  ┌───────────────┐   │    │   │
│  │  │  │  SQLite   │  │  PostgreSQL   │   │    │   │
│  │  │  │Connection │  │  Connection   │   │    │   │
│  │  │  └───────────┘  └───────────────┘   │    │   │
│  │  └─────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────┘   │
│            │                         │                  │
│            ▼                         ▼                  │
│  ┌─────────────────┐  ┌─────────────────────────────┐   │
│  │ DatabaseInterface │  │    Migration System        │   │
│  │    (Abstract)    │  │  ┌─────────────────────┐    │   │
│  │                 │  │  │   Migration Runner  │    │   │
│  │  ┌──────────────┐│  │  │  ┌───────────────┐ │    │   │
│  │  │Query Methods ││  │  │  │   V001.sql    │ │    │   │
│  │  │- execute()   ││  │  │  │   V002.sql    │ │    │   │
│  │  │- fetch_all() ││  │  │  │   V003.sql    │ │    │   │
│  │  │- fetch_one() ││  │  │  └───────────────┘ │    │   │
│  │  └──────────────┘│  │  └─────────────────────┘    │   │
│  └─────────────────┘  └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Database Interfaces

### DatabaseInterface (Abstract Base Class)

The core abstraction that defines the contract for all database operations.

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple

class DatabaseInterface(ABC):
    def __init__(self, connection_string: str, db_type: DatabaseType):
        self.connection_string = connection_string
        self.db_type = db_type
        self.is_connected = False

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the database"""
        pass

    @abstractmethod
    async def execute_query(self, query: str, parameters: Optional[Tuple] = None) -> QueryResult:
        """Execute a SQL query"""
        pass

    @abstractmethod
    async def fetch_all(self, query: str, parameters: Optional[Tuple] = None) -> QueryResult:
        """Fetch all results from a query"""
        pass

    @abstractmethod
    async def table_exists(self, table_name: str) -> bool:
        """Check if table exists"""
        pass

    async def health_check(self) -> bool:
        """Check database health"""
        try:
            result = await self.execute_query("SELECT 1 as health_check")
            return result.success
        except Exception:
            return False
```

### Key Interface Methods

#### Connection Management
- `connect()`: Establish database connection
- `disconnect()`: Close database connection
- `health_check()`: Verify database connectivity

#### Query Operations
- `execute_query()`: Execute SQL with optional parameters
- `fetch_all()`: Retrieve all rows from query
- `fetch_one()`: Retrieve single row from query

#### Schema Operations
- `table_exists()`: Check table existence
- `create_table()`: Create new table with schema

## Connection Management

### ConnectionConfig

Centralized configuration for database connections:

```python
@dataclass
class ConnectionConfig:
    database_type: DatabaseType
    database_url: str
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    database_name: Optional[str] = None

    @classmethod
    def from_url(cls, database_url: str) -> 'ConnectionConfig':
        """Create configuration from database URL"""
        if database_url.startswith('sqlite'):
            return cls(database_type=DatabaseType.SQLITE, database_url=database_url)
        elif database_url.startswith('postgresql'):
            return cls(database_type=DatabaseType.POSTGRESQL, database_url=database_url)
        # Additional database types...
```

### Database Types

```python
class DatabaseType(Enum):
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    DUCKDB = "duckdb"
    CLOUDFLARE_D1 = "cloudflare_d1"
```

### Connection Examples

```python
# SQLite connection
sqlite_config = ConnectionConfig.from_url("sqlite:///./app_data.db")

# PostgreSQL connection
postgres_config = ConnectionConfig.from_url(
    "postgresql://user:password@localhost:5432/database"
)

# DuckDB connection
duckdb_config = ConnectionConfig.from_url("duckdb:///./analytics.duckdb")
```

## Migration System

### MigrationRunner

Comprehensive migration management with versioning and rollback capabilities.

**Key Features:**
- Version-based migration tracking
- Automatic migration discovery
- Checksum validation
- Support for both system and app migrations
- Rollback capabilities

```python
class MigrationRunner:
    def __init__(
        self,
        database: DatabaseInterface,
        migration_path: Optional[Path] = None,
        migration_type: str = "app"
    ):
        self.database = database
        self.migration_path = migration_path
        self.migration_type = migration_type
        self._migrations_table = "ia_migrations"

    async def run_pending_migrations(self) -> bool:
        """Run all pending migrations"""
        # Initialize migration table
        await self.initialize_migration_table()

        # Get pending migrations
        pending_migrations = await self.get_pending_migrations()

        # Run each migration
        for migration_file in pending_migrations:
            success = await self.run_migration_file(migration_file)
            if not success:
                return False

        return True
```

### Migration File Format

Migration files follow the naming convention: `V{version}__{description}.sql`

```sql
-- V001__create_base_schema.sql
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    username TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pipelines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slug TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    pipeline_config TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_pipelines_slug ON pipelines(slug);
```

### Migration Tracking Table

The system automatically creates a tracking table:

```sql
CREATE TABLE IF NOT EXISTS ia_migrations (
    version TEXT NOT NULL,
    filename TEXT NOT NULL,
    migration_type TEXT NOT NULL CHECK (migration_type IN ('system', 'app')),
    description TEXT NOT NULL,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    checksum TEXT,
    PRIMARY KEY (version, migration_type)
);
```

### Migration Types

1. **System Migrations**: Core IA Modules schema
2. **App Migrations**: Application-specific schema extensions

```python
# Run system migrations
system_runner = MigrationRunner(db_interface, system_migration_path, "system")
await system_runner.run_pending_migrations()

# Run app migrations
app_runner = MigrationRunner(db_interface, app_migration_path, "app")
await app_runner.run_pending_migrations()
```

## Database Manager

### DatabaseManager Implementation

Concrete implementation providing connection management and query execution:

```python
class DatabaseManager:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.config = ConnectionConfig.from_url(database_url)
        self._connection = None

    async def initialize(self, apply_schema: bool = True, app_migration_paths: Optional[List[str]] = None) -> bool:
        """Initialize database with optional schema application"""
        if not self.connect():
            return False

        if apply_schema:
            return await self._run_migrations(app_migration_paths)

        return True

    def execute(self, query: str, params: Optional[tuple] = None) -> Any:
        """Execute query with parameter binding"""
        if not self._connection:
            raise RuntimeError("Database not connected")

        cursor = self._connection.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)

        self._connection.commit()
        return cursor
```

### Context Manager Support

```python
# Automatic connection management
with DatabaseManager("sqlite:///./app.db") as db:
    result = db.fetch_all("SELECT * FROM users WHERE active = ?", (True,))
    print(f"Found {len(result)} active users")
```

### Async-Compatible Methods

```python
# Async query execution
async def get_user_count():
    async with AsyncDatabaseManager("sqlite:///./app.db") as db:
        result = await db.execute_query("SELECT COUNT(*) as count FROM users")
        return result.data[0]['count'] if result.success else 0
```

## Query Result System

### QueryResult Class

Standardized result container for all database operations:

```python
@dataclass
class QueryResult:
    success: bool
    data: List[Dict[str, Any]]
    row_count: int
    error_message: Optional[str] = None
    execution_time_ms: Optional[float] = None

    def get_first_row(self) -> Optional[Dict[str, Any]]:
        """Get first row if available"""
        return self.data[0] if self.data else None

    def get_column_values(self, column_name: str) -> List[Any]:
        """Get all values for a specific column"""
        return [row.get(column_name) for row in self.data if column_name in row]
```

### Query Result Usage

```python
# Execute query and handle results
result = await db.fetch_all("SELECT id, name, email FROM users WHERE status = ?", ("active",))

if result.success:
    print(f"Found {result.row_count} active users")

    # Access individual rows
    for user in result.data:
        print(f"User: {user['name']} <{user['email']}>")

    # Get first row
    first_user = result.get_first_row()

    # Get column values
    user_emails = result.get_column_values('email')
else:
    print(f"Query failed: {result.error_message}")
```

### Utility Functions

```python
# Create successful result
success_result = create_query_result(
    success=True,
    data=[{'id': 1, 'name': 'John'}],
    execution_time_ms=15.5
)

# Create error result
error_result = create_error_result("Table 'users' does not exist")
```

## Supported Database Types

### SQLite Configuration

```python
# File-based SQLite
sqlite_config = ConnectionConfig.from_url("sqlite:///./data/app.db")

# In-memory SQLite (testing)
memory_config = ConnectionConfig.from_url("sqlite:///:memory:")

# Create directory for SQLite file
Path("./data").mkdir(parents=True, exist_ok=True)
```

### PostgreSQL Configuration

```python
# Standard PostgreSQL
postgres_config = ConnectionConfig.from_url(
    "postgresql://username:password@localhost:5432/database_name"
)

# PostgreSQL with SSL
postgres_ssl_config = ConnectionConfig.from_url(
    "postgresql://user:pass@host:5432/db?sslmode=require"
)
```

### DuckDB Configuration

```python
# File-based DuckDB (analytics)
duckdb_config = ConnectionConfig.from_url("duckdb:///./analytics.duckdb")

# In-memory DuckDB
memory_duckdb = ConnectionConfig.from_url("duckdb:///:memory:")
```

### Cloudflare D1 Configuration

```python
# Cloudflare D1 (edge computing)
d1_config = ConnectionConfig.from_url(
    "cloudflare_d1://database_id?account_id=account&api_token=token"
)
```

## Configuration Patterns

### Development Environment

```python
# Simple SQLite for local development
DATABASE_URL = "sqlite:///./dev_data.db"

db_manager = DatabaseManager(DATABASE_URL)
await db_manager.initialize(apply_schema=True)
```

### Production Environment

```python
# PostgreSQL with connection pooling
DATABASE_URL = "postgresql://user:pass@prod-db:5432/app_db"

db_manager = DatabaseManager(DATABASE_URL)
await db_manager.initialize(
    apply_schema=True,
    app_migration_paths=["./migrations", "./custom_migrations"]
)
```

### Multi-Environment Setup

```python
import os

# Environment-based configuration
database_configs = {
    'development': "sqlite:///./dev.db",
    'testing': "sqlite:///:memory:",
    'staging': "postgresql://user:pass@staging-db:5432/staging_db",
    'production': "postgresql://user:pass@prod-db:5432/prod_db"
}

environment = os.getenv('ENVIRONMENT', 'development')
DATABASE_URL = database_configs[environment]
```

### Service Injection Integration

```python
from ia_modules.pipeline.services import ServiceRegistry

# Register database service
services = ServiceRegistry(
    database=db_manager,
    http=http_client
)

# Access in pipeline steps
class DatabaseStep(Step):
    async def work(self, data: Dict[str, Any]) -> Any:
        db = self.get_db()
        result = await db.fetch_all("SELECT * FROM processed_data WHERE id = ?", (data['id'],))

        return {
            "database_result": result.data,
            "row_count": result.row_count
        }
```

## Best Practices

### Connection Management

1. **Use Connection Pooling**: For production environments
2. **Implement Health Checks**: Monitor database connectivity
3. **Handle Connection Failures**: Graceful degradation patterns
4. **Resource Cleanup**: Always close connections

```python
# Robust connection handling
class RobustDatabaseManager(DatabaseManager):
    async def execute_with_retry(self, query: str, params: Optional[tuple] = None, retries: int = 3) -> QueryResult:
        for attempt in range(retries):
            try:
                return await self.execute_query(query, params)
            except ConnectionError:
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

### Query Optimization

1. **Use Parameter Binding**: Prevent SQL injection
2. **Batch Operations**: Reduce round trips
3. **Index Strategy**: Optimize query performance
4. **Connection Reuse**: Minimize connection overhead

```python
# Parameterized queries
users = await db.fetch_all(
    "SELECT * FROM users WHERE department = ? AND active = ?",
    (department, True)
)

# Batch insert
await db.execute_many(
    "INSERT INTO logs (timestamp, message, level) VALUES (?, ?, ?)",
    [(now, msg, level) for msg, level in log_entries]
)
```

### Migration Best Practices

1. **Version Consistency**: Use semantic versioning
2. **Backwards Compatibility**: Design reversible migrations
3. **Test Migrations**: Validate on staging environment
4. **Backup Strategy**: Always backup before migrations

```python
# Migration with rollback plan
# V005__add_user_preferences.sql
-- Add new column with default
ALTER TABLE users ADD COLUMN preferences TEXT DEFAULT '{}';

-- Create index for performance
CREATE INDEX IF NOT EXISTS idx_users_preferences ON users(preferences);

-- Rollback: DROP COLUMN preferences; DROP INDEX idx_users_preferences;
```

### Error Handling

```python
async def safe_database_operation():
    try:
        result = await db.fetch_all("SELECT * FROM critical_table")
        if not result.success:
            logger.error(f"Database query failed: {result.error_message}")
            return None

        return result.data

    except Exception as e:
        logger.error(f"Unexpected database error: {e}")
        # Implement fallback strategy
        return await get_cached_data()
```

## Performance Considerations

### Query Performance

- **Index Usage**: Create indexes for frequently queried columns
- **Query Planning**: Use EXPLAIN to analyze query execution
- **Parameter Binding**: Avoid string concatenation in queries
- **Result Set Limiting**: Use LIMIT clauses for large datasets

### Connection Performance

- **Connection Pooling**: Reuse database connections
- **Async Operations**: Non-blocking database calls
- **Batch Processing**: Group related operations
- **Health Monitoring**: Track connection pool metrics

### Memory Management

- **Result Set Streaming**: For large query results
- **Connection Cleanup**: Proper resource disposal
- **Query Result Caching**: Cache frequently accessed data
- **Lazy Loading**: Load data on-demand

This database interface system provides a robust, scalable foundation for data persistence in IA Modules applications, supporting multiple database backends with consistent APIs and comprehensive migration management.