# Database Abstraction Layer

Pluggable database backend system for ia_modules with support for NexusQL and SQLAlchemy.

## Quick Start

### Using NexusQL (Default)

```python
from ia_modules.database import get_database

# Simple usage
db = get_database("sqlite:///app.db")
db.connect()

# Execute queries
users = db.execute("SELECT * FROM users WHERE id = :id", {"id": 1})

# Fetch operations
user = db.fetch_one("SELECT * FROM users WHERE email = :email", {"email": "test@example.com"})
all_users = db.fetch_all("SELECT * FROM users")

db.disconnect()
```

### Using SQLAlchemy

```python
from ia_modules.database import get_database

# Specify SQLAlchemy backend
db = get_database("sqlite:///app.db", backend="sqlalchemy")
db.connect()

# Same interface works!
users = db.execute("SELECT * FROM users WHERE id = :id", {"id": 1})
user = db.fetch_one("SELECT * FROM users WHERE email = :email", {"email": "test@example.com"})

db.disconnect()
```

## Features

### NexusQL Backend

**Pros:**
- ✅ **Multi-database support**: SQLite, PostgreSQL, MySQL, MSSQL
- ✅ **SQL Translation**: Write PostgreSQL syntax, works everywhere
- ✅ **Lightweight**: No ORM overhead
- ✅ **Built-in migrations**: Automatic schema versioning
- ✅ **Simple**: Easy to use for straightforward queries

**Best for:**
- Multi-database applications
- Simple CRUD operations
- Projects that need SQL translation
- Lightweight database access

**Installation:**
```bash
pip install nexusql
```

### SQLAlchemy Backend

**Pros:**
- ✅ **Full ORM**: Complete object-relational mapping
- ✅ **Advanced features**: Complex queries, relationships, lazy loading
- ✅ **Mature ecosystem**: Alembic, Flask-SQLAlchemy, etc.
- ✅ **Connection pooling**: Production-ready connection management
- ✅ **Extensive database support**: 30+ database backends

**Best for:**
- Complex data models with relationships
- Large-scale applications
- Projects already using SQLAlchemy
- Need for advanced ORM features

**Installation:**
```bash
pip install sqlalchemy
```

## Configuration

### Method 1: Explicit Backend Selection

```python
from ia_modules.database import get_database

# NexusQL
db = get_database("sqlite:///app.db", backend="nexusql")

# SQLAlchemy
db = get_database("sqlite:///app.db", backend="sqlalchemy")
```

### Method 2: Environment Variable

```bash
# Set environment variable
export IA_DATABASE_BACKEND=sqlalchemy

# Or in Python
import os
os.environ["IA_DATABASE_BACKEND"] = "sqlalchemy"
```

```python
from ia_modules.database import get_database

# Will use SQLAlchemy because of environment variable
db = get_database("sqlite:///app.db")
```

### Method 3: Global Default

```python
from ia_modules.database import set_default_backend, DatabaseBackend, get_database

# Set global default
set_default_backend(DatabaseBackend.SQLALCHEMY)

# All future get_database() calls will use SQLAlchemy
db = get_database("sqlite:///app.db")
```

## Usage Examples

### Basic CRUD Operations

Both backends support the same interface:

```python
from ia_modules.database import get_database

db = get_database("sqlite:///app.db")  # Uses default backend
db.connect()

# CREATE
db.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        email TEXT UNIQUE NOT NULL,
        name TEXT
    )
""")

# INSERT
db.execute(
    "INSERT INTO users (email, name) VALUES (:email, :name)",
    {"email": "alice@example.com", "name": "Alice"}
)

# SELECT (fetch_one)
user = db.fetch_one(
    "SELECT * FROM users WHERE email = :email",
    {"email": "alice@example.com"}
)
print(user)  # {'id': 1, 'email': 'alice@example.com', 'name': 'Alice'}

# SELECT (fetch_all)
all_users = db.fetch_all("SELECT * FROM users")

# UPDATE
db.execute(
    "UPDATE users SET name = :name WHERE id = :id",
    {"name": "Alice Smith", "id": 1}
)

# DELETE
db.execute("DELETE FROM users WHERE id = :id", {"id": 1})

db.disconnect()
```

### Context Manager

```python
from ia_modules.database import get_database

with get_database("sqlite:///app.db") as db:
    users = db.execute("SELECT * FROM users")
    # Auto-disconnect on exit
```

### Async Support

```python
from ia_modules.database import get_database

db = get_database("sqlite:///app.db")
await db.initialize(apply_schema=True)

result = await db.execute_async("SELECT * FROM users")

await db.close()
```

### Advanced: SQLAlchemy-Specific Features

When using SQLAlchemy backend, you can access advanced features:

```python
from ia_modules.database import get_database

db = get_database("postgresql://user:pass@localhost/db", backend="sqlalchemy")
db.connect()

# Access underlying SQLAlchemy engine
engine = db.engine

# Access session for ORM operations
session = db.session

# Create new session
new_session = db.get_new_session()

# Transaction management
with db.begin_transaction():
    db.execute("INSERT INTO users (name) VALUES (:name)", {"name": "Bob"})
    # Auto-commit on success, rollback on exception

db.disconnect()
```

### Advanced: NexusQL-Specific Features

When using NexusQL backend, you can access SQL translation and multi-database features:

```python
from ia_modules.database import get_database

db = get_database("sqlite:///app.db", backend="nexusql")
db.connect()

# Access underlying NexusQL instance
nexusql = db.nexusql

# Get database type
db_type = db.database_type
print(db_type)  # DatabaseType.SQLITE

# SQL translation happens automatically
db.execute("""
    CREATE TABLE products (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255),
        active BOOLEAN DEFAULT TRUE,
        metadata JSONB
    )
""")
# Works on SQLite! SERIAL → INTEGER AUTOINCREMENT, BOOLEAN → INTEGER, etc.

db.disconnect()
```

## Switching Backends

You can switch between backends without changing your code:

```python
# config.py
DATABASE_URL = "postgresql://user:pass@localhost/db"
DATABASE_BACKEND = "nexusql"  # or "sqlalchemy"

# app.py
from ia_modules.database import get_database
from config import DATABASE_URL, DATABASE_BACKEND

db = get_database(DATABASE_URL, backend=DATABASE_BACKEND)
```

## Comparison Table

| Feature | NexusQL | SQLAlchemy |
|---------|---------|------------|
| **Multi-database** | ✅ Built-in | ✅ Via dialects |
| **SQL Translation** | ✅ Automatic | ❌ Manual |
| **ORM** | ❌ Raw SQL only | ✅ Full ORM |
| **Migrations** | ✅ Built-in | ⚠️ Use Alembic |
| **Connection Pooling** | ⚠️ Basic | ✅ Advanced |
| **Performance** | ⚡ Fast (no ORM) | ⚡ Fast (with optimization) |
| **Learning Curve** | 🟢 Easy | 🟡 Moderate |
| **Ecosystem** | 🟡 New | 🟢 Mature |
| **Best for** | Simple queries | Complex models |

## Migration Guide

### From Direct NexusQL to Abstraction Layer

**Before:**
```python
from nexusql import DatabaseManager

db = DatabaseManager("sqlite:///app.db")
db.connect()
```

**After:**
```python
from ia_modules.database import get_database

db = get_database("sqlite:///app.db", backend="nexusql")
db.connect()
```

### Adding SQLAlchemy Support to Existing Code

**Step 1:** Install SQLAlchemy
```bash
pip install sqlalchemy
```

**Step 2:** Change one line
```python
# Old (NexusQL only)
db = get_database("sqlite:///app.db")

# New (SQLAlchemy)
db = get_database("sqlite:///app.db", backend="sqlalchemy")
```

**Step 3:** That's it! Your code continues to work.

## Testing Different Backends

```python
import pytest
from ia_modules.database import get_database

@pytest.fixture(params=["nexusql", "sqlalchemy"])
def db(request):
    """Test with both backends"""
    backend = request.param
    database = get_database("sqlite:///:memory:", backend=backend)
    database.connect()
    yield database
    database.disconnect()

def test_user_creation(db):
    """This test runs with both NexusQL and SQLAlchemy"""
    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    db.execute("INSERT INTO users (name) VALUES (:name)", {"name": "Test"})
    user = db.fetch_one("SELECT * FROM users WHERE name = :name", {"name": "Test"})
    assert user["name"] == "Test"
```

## API Reference

### DatabaseInterface

All backends implement this interface:

**Connection Management:**
- `connect() -> bool`: Connect to database
- `disconnect()`: Disconnect from database
- `async close()`: Async close method

**Query Execution:**
- `execute(query: str, params: Dict) -> List[Dict]`: Execute query
- `async execute_async(query: str, params: Dict)`: Async execute
- `fetch_one(query: str, params: Dict) -> Optional[Dict]`: Fetch single row
- `fetch_all(query: str, params: Dict) -> List[Dict]`: Fetch all rows

**Utility:**
- `table_exists(table_name: str) -> bool`: Check if table exists
- `async execute_script(script: str) -> QueryResult`: Execute SQL script
- `async initialize(apply_schema: bool, app_migration_paths: List[str]) -> bool`: Initialize with migrations

### Factory Functions

- `get_database(url: str, backend: str, **kwargs) -> DatabaseInterface`: Get database instance
- `set_default_backend(backend: DatabaseBackend)`: Set global default
- `get_nexusql_database(url: str) -> NexuSQLAdapter`: Get NexusQL instance
- `get_sqlalchemy_database(url: str, **kwargs) -> SQLAlchemyAdapter`: Get SQLAlchemy instance

## Troubleshooting

### ImportError: nexusql is not installed

```bash
pip install nexusql
```

### ImportError: sqlalchemy is not installed

```bash
pip install sqlalchemy
```

### How do I use SQLAlchemy ORM models?

Access the underlying SQLAlchemy session:

```python
db = get_database("sqlite:///app.db", backend="sqlalchemy")
db.connect()

# Access session for ORM
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String)

# Use with session
db.session.add(User(name="Alice"))
db.session.commit()
```

### Can I mix backends in the same application?

Yes! Create different instances:

```python
# NexusQL for simple queries
cache_db = get_database("sqlite:///cache.db", backend="nexusql")

# SQLAlchemy for complex ORM
main_db = get_database("postgresql://localhost/main", backend="sqlalchemy")
```

## Best Practices

1. **Use NexusQL for**: Simple CRUD, multi-database apps, SQL translation needs
2. **Use SQLAlchemy for**: Complex models, relationships, existing SQLAlchemy projects
3. **Use environment variables** for easy backend switching in different environments
4. **Write backend-agnostic code** using the DatabaseInterface
5. **Access advanced features** via `.nexusql` or `.session` properties when needed

## License

MIT
