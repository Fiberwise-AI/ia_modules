# Database System Research & Planning

## Current State Analysis

### What We Have Now

**Architecture:**
```
DatabaseManager (sync wrapper)
    ↓
DatabaseInterfaceAdapter (hacky conversion layer)
    ↓
DatabaseInterface (abstract base)
    ├── SQLiteProvider (incomplete)
    ├── PostgreSQLInterface (new, minimal)
    └── SimpleSQLite (async, minimal)
```

**Problems:**
1. **Multiple competing implementations** - DatabaseManager vs DatabaseInterface implementations
2. **Inconsistent APIs** - Some sync, some async, different parameter styles
3. **SQLite-specific queries hardcoded** - Adapter uses SQLite syntax for PostgreSQL
4. **Parameter placeholder hell** - `?` vs `%s` vs `$1` conversion everywhere
5. **Manual schema creation** - Every service had `_create_schema()` methods
6. **Poor transaction handling** - Rollback logic scattered, autocommit issues
7. **No connection pooling** - Single connection per database
8. **Migration system uses different interface** - DatabaseInterfaceAdapter bridges the gap

### What's Actually Being Used

**In showcase app:**
- `DatabaseManager` - Main connection, runs migrations
- `DatabaseInterfaceAdapter` - Bridges to migrations and checkpointer
- `PostgreSQLInterface` - Used by SQLCheckpointer
- All three exist because nothing is unified

**In library modules:**
- `SQLCheckpointer` expects `DatabaseInterface`
- `SQLMetricStorage` expects `ConnectionConfig`
- `SQLConversationMemory` expects `DatabaseInterface`

**Result:** Chaos. Too many abstractions, none work together cleanly.

---

## Research: How Other Systems Handle This

### 1. SQLAlchemy Core (SQL-focused, no ORM)

**Approach:** Connection pooling + text SQL with parameter binding

```python
from sqlalchemy import create_engine, text

# One engine, handles all database types
engine = create_engine("postgresql://...")

# Write raw SQL, parameters are database-agnostic
with engine.connect() as conn:
    result = conn.execute(
        text("SELECT * FROM users WHERE id = :user_id"),
        {"user_id": 123}
    )
```

**Pros:**
- ✅ Write raw SQL (your preference)
- ✅ Automatic dialect detection (PostgreSQL, SQLite, MySQL all work)
- ✅ Connection pooling built-in
- ✅ Transaction management handled
- ✅ Parameter binding abstraction (no ? vs %s vs $1 hell)
- ✅ Battle-tested, mature

**Cons:**
- ❌ External dependency
- ❌ Learning curve for engine/connection lifecycle
- ❌ Slightly heavier than raw psycopg2

**Verdict:** **Strong candidate** - Solves 90% of our problems

---

### 2. SQLAlchemy ORM (Class-based models)

**Approach:** Define models, let ORM generate SQL

```python
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import Session, declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String)

# Query using objects
session.query(User).filter(User.id == 123).first()
```

**Pros:**
- ✅ Type safety
- ✅ No SQL syntax errors
- ✅ Automatic migrations with Alembic

**Cons:**
- ❌ You said "not dependent on classes" - this IS classes
- ❌ More abstraction = less control
- ❌ Heavier dependency

**Verdict:** **Not recommended** based on your preference for raw SQL

---

### 3. asyncpg (PostgreSQL-specific, async)

**Approach:** Pure async PostgreSQL driver

```python
import asyncpg

pool = await asyncpg.create_pool("postgresql://...")
async with pool.acquire() as conn:
    result = await conn.fetch("SELECT * FROM users WHERE id = $1", 123)
```

**Pros:**
- ✅ Fastest PostgreSQL driver
- ✅ Async-native
- ✅ Simple API
- ✅ Connection pooling

**Cons:**
- ❌ PostgreSQL only (no SQLite for dev/testing)
- ❌ Different parameter syntax ($1, $2)
- ❌ Would need separate SQLite solution

**Verdict:** **Could work** for production-only code, but need SQLite abstraction

---

### 4. Databases (async abstraction over multiple DBs)

**Approach:** Unified async interface for PostgreSQL, SQLite, MySQL

```python
from databases import Database

database = Database("postgresql://...")
await database.connect()

result = await database.fetch_all(
    "SELECT * FROM users WHERE id = :user_id",
    {"user_id": 123}
)
```

**Pros:**
- ✅ Async-first
- ✅ Supports PostgreSQL, SQLite, MySQL
- ✅ Raw SQL with named parameters
- ✅ Simple API
- ✅ Built on SQLAlchemy Core (gets dialect handling)

**Cons:**
- ❌ Less mature than SQLAlchemy
- ❌ Still maintains separate connection per call

**Verdict:** **Good middle ground** - async + raw SQL + multi-DB

---

### 5. Psycopg3 (PostgreSQL, modern)

**Approach:** Modern rewrite of psycopg2 with async support

```python
import psycopg

async with await psycopg.AsyncConnection.connect("postgresql://...") as conn:
    async with conn.cursor() as cur:
        await cur.execute("SELECT * FROM users WHERE id = %s", (123,))
        result = await cur.fetchall()
```

**Pros:**
- ✅ Modern, async-native
- ✅ Better than psycopg2
- ✅ Connection pooling

**Cons:**
- ❌ PostgreSQL only
- ❌ Still need SQLite solution

**Verdict:** **Use for PostgreSQL**, but need abstraction layer

---

## Recommendation: Hybrid Approach

### Proposal: "Raw SQL + Smart Abstraction"

Keep the philosophy: **write raw SQL, no ORM**, but fix the infrastructure.

### Architecture

```
┌─────────────────────────────────────┐
│  Application Code                   │
│  (writes raw SQL)                   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Database (unified interface)       │
│  - async def execute(sql, params)   │
│  - async def fetch_all(sql, params) │
│  - async def fetch_one(sql, params) │
│  - Connection pooling               │
│  - Transaction management           │
└──────────────┬──────────────────────┘
               │
         ┌─────┴──────┐
         ▼            ▼
┌─────────────┐  ┌──────────────┐
│ PostgreSQL  │  │   SQLite     │
│ (asyncpg)   │  │ (aiosqlite)  │
└─────────────┘  └──────────────┘
```

### Implementation Options

#### Option A: Use SQLAlchemy Core (RECOMMENDED)

**Why:**
- Solves parameter binding (no ? vs %s vs $1)
- Connection pooling built-in
- Dialect detection automatic
- Transaction management
- Battle-tested
- Can still write raw SQL

**Code:**
```python
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

class Database:
    def __init__(self, url: str):
        self.engine = create_engine(url, poolclass=NullPool)  # Or use pooling

    async def execute(self, sql: str, params: dict = None):
        """Execute SQL with named parameters"""
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            conn.commit()
            return result

    async def fetch_all(self, sql: str, params: dict = None):
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            return [dict(row) for row in result]

    async def fetch_one(self, sql: str, params: dict = None):
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            row = result.fetchone()
            return dict(row) if row else None
```

**Usage:**
```python
db = Database("postgresql://localhost/mydb")

# Works on PostgreSQL AND SQLite with same code
users = await db.fetch_all(
    "SELECT * FROM users WHERE age > :min_age",
    {"min_age": 18}
)
```

**Migration:**
- Keep existing SQL migration files
- Use SQLAlchemy's engine to run them
- No Alembic needed (unless you want it)

---

#### Option B: Custom Abstraction (Current Direction)

Keep building our own, but do it right:

**Improvements Needed:**
1. **Single async interface** - Kill DatabaseManager, only use async DatabaseInterface
2. **Parameter binding** - Convert `:name` → `%s`/`$1` based on dialect
3. **Connection pooling** - Use asyncpg pool for PostgreSQL, aiosqlite for SQLite
4. **Transaction context manager** - `async with db.transaction()`
5. **Proper error handling** - Rollback on failure

**Code:**
```python
class Database:
    def __init__(self, url: str):
        self.url = url
        self.dialect = self._detect_dialect(url)
        self.pool = None  # asyncpg.Pool or aiosqlite connection

    async def connect(self):
        if self.dialect == "postgresql":
            self.pool = await asyncpg.create_pool(self.url)
        elif self.dialect == "sqlite":
            self.pool = await aiosqlite.connect(self.url)

    async def execute(self, sql: str, params: dict = None):
        """Convert :name params to $1/$2 or ? based on dialect"""
        sql, param_list = self._convert_params(sql, params)

        if self.dialect == "postgresql":
            async with self.pool.acquire() as conn:
                return await conn.execute(sql, *param_list)
        else:
            return await self.pool.execute(sql, param_list)

    def _convert_params(self, sql: str, params: dict):
        """Convert :name to $1, $2, etc for PostgreSQL or ? for SQLite"""
        if not params:
            return sql, []

        if self.dialect == "postgresql":
            # :name → $1, $2, ...
            new_sql = sql
            param_list = []
            for i, (key, value) in enumerate(params.items(), 1):
                new_sql = new_sql.replace(f":{key}", f"${i}")
                param_list.append(value)
            return new_sql, param_list
        else:
            # :name → ?
            new_sql = sql
            param_list = []
            for key, value in params.items():
                new_sql = new_sql.replace(f":{key}", "?")
                param_list.append(value)
            return new_sql, param_list
```

**Pros:**
- No external dependencies (besides drivers)
- Full control
- Learn by building

**Cons:**
- More work
- Will hit edge cases SQLAlchemy already solved
- Maintenance burden

---

## SQL Dialect Differences to Handle

### Common Pain Points

| Feature | PostgreSQL | SQLite | MySQL |
|---------|-----------|---------|--------|
| **Placeholders** | `$1, $2` | `?` | `%s` |
| **AUTOINCREMENT** | `SERIAL` | `AUTOINCREMENT` | `AUTO_INCREMENT` |
| **BOOLEAN** | `BOOLEAN` | `INTEGER` | `TINYINT(1)` |
| **JSON** | `JSONB` | `TEXT` | `JSON` |
| **UUID** | `UUID` | `TEXT` | `CHAR(36)` |
| **DATETIME** | `TIMESTAMP` | `TEXT` | `DATETIME` |
| **RETURNING** | Yes | `RETURNING` (3.35+) | No |
| **Schema check** | `information_schema.tables` | `sqlite_master` | `information_schema.tables` |

### How SQLAlchemy Handles It

SQLAlchemy Core has **dialects** that abstract these differences:

```python
# This works on all databases
from sqlalchemy import Column, Integer, String, Boolean, JSON

table = Table('users',
    Column('id', Integer, primary_key=True),  # → SERIAL/AUTOINCREMENT/AUTO_INCREMENT
    Column('active', Boolean),                 # → BOOLEAN/INTEGER/TINYINT
    Column('metadata', JSON)                   # → JSONB/TEXT/JSON
)
```

### If We Build Custom

Need a **Dialect** class:

```python
class Dialect:
    @abstractmethod
    def placeholder(self, index: int) -> str:
        """Return parameter placeholder for this index"""
        pass

    @abstractmethod
    def table_exists_sql(self) -> str:
        """Return SQL to check if table exists"""
        pass

class PostgreSQLDialect(Dialect):
    def placeholder(self, index: int) -> str:
        return f"${index}"

    def table_exists_sql(self) -> str:
        return """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = $1
            )
        """

class SQLiteDialect(Dialect):
    def placeholder(self, index: int) -> str:
        return "?"

    def table_exists_sql(self) -> str:
        return "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
```

---

## Migration System

### Current System

**File-based migrations** - This is GOOD, keep it:
```
migrations/
  V001__create_users.sql
  V002__add_email.sql
  V003__create_posts.sql
```

**Tracking table** - `ia_migrations` table tracks applied versions

**Problems:**
- Migrations use database-specific SQL
- Need to write PostgreSQL AND SQLite versions

### Option 1: Database-Specific Migration Files

```
migrations/
  postgresql/
    V001__create_users.sql
    V002__add_email.sql
  sqlite/
    V001__create_users.sql
    V002__add_email.sql
```

**Pros:** Full control per database
**Cons:** Maintenance hell, duplication

### Option 2: Abstract Migration Syntax (like Alembic)

Write migrations in Python, generate SQL:

```python
# migrations/V001_create_users.py
def upgrade(db):
    db.create_table('users',
        Column('id', Integer, primary_key=True),
        Column('name', String(255))
    )

def downgrade(db):
    db.drop_table('users')
```

**Pros:** One migration, works on all DBs
**Cons:** Python, not SQL (you prefer SQL)

### Option 3: Minimal SQL Dialect (RECOMMENDED)

Write migrations with **safe subset** of SQL that works everywhere:

**Allowed:**
```sql
-- Use generic types
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name VARCHAR(255),
    active INTEGER,  -- Use INTEGER for boolean
    data TEXT        -- Use TEXT for JSON
);
```

**Forbidden:**
```sql
-- Don't use database-specific features
SERIAL, AUTOINCREMENT, BOOLEAN, JSONB, UUID
```

**Migration runner** translates on-the-fly:

```python
class MigrationRunner:
    def run_migration(self, sql: str):
        # Translate based on dialect
        if self.dialect == "postgresql":
            sql = sql.replace("INTEGER PRIMARY KEY", "SERIAL PRIMARY KEY")
            sql = sql.replace("INTEGER -- boolean", "BOOLEAN")

        self.db.execute(sql)
```

---

## Recommendations

### Phase 1: Simplify (Immediate)

**Goal:** Make current system work cleanly

1. **Delete DatabaseManager** - Only use async DatabaseInterface
2. **Fix DatabaseInterfaceAdapter** - Proper PostgreSQL support
3. **Add transaction rollback** - Handle failures correctly
4. **Parameter binding** - Fix `?` vs `%s` conversion

**Files to change:**
- `database/manager.py` - Delete sync code
- `database/interfaces.py` - Make this the only interface
- `database/postgres.py` - Make this complete
- `database/sqlite_simple.py` - Make this complete

### Phase 2: Unify (1-2 weeks)

**Goal:** One database abstraction for everything

**Option A: Adopt SQLAlchemy Core** (RECOMMENDED)
- Install: `pip install sqlalchemy`
- Replace all database code with SQLAlchemy engine
- Keep raw SQL everywhere
- Get connection pooling, parameter binding, transactions for free

**Option B: Build Custom Abstraction**
- Create proper async interface
- Add connection pooling (asyncpg.Pool)
- Add dialect system for SQL differences
- Add transaction context managers

### Phase 3: Clean Migrations (1 week)

1. **Make migrations PostgreSQL-first**
2. **Test on SQLite** - Add compatibility where needed
3. **Remove schema creation** from services (DONE)
4. **Add migration tests**

---

## Proposed Final Architecture

### With SQLAlchemy Core (RECOMMENDED)

```python
# database/__init__.py
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

class Database:
    """Unified async database interface using SQLAlchemy Core"""

    def __init__(self, url: str):
        self.engine = create_engine(
            url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10
        )

    async def execute(self, sql: str, params: dict = None):
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            conn.commit()
            return result.rowcount

    async def fetch_all(self, sql: str, params: dict = None):
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            return [dict(row._mapping) for row in result]

    async def fetch_one(self, sql: str, params: dict = None):
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            row = result.fetchone()
            return dict(row._mapping) if row else None

    async def transaction(self):
        """Transaction context manager"""
        return self.engine.begin()

    async def table_exists(self, table_name: str) -> bool:
        """Check if table exists - works on all databases"""
        from sqlalchemy import inspect
        inspector = inspect(self.engine)
        return table_name in inspector.get_table_names()

# Usage everywhere
db = Database("postgresql://localhost/mydb")

# Works on PostgreSQL, SQLite, MySQL
users = await db.fetch_all(
    "SELECT * FROM users WHERE age > :min_age",
    {"min_age": 18}
)

# Transactions
async with db.transaction() as trans:
    await db.execute("INSERT INTO users (name) VALUES (:name)", {"name": "Alice"})
    await db.execute("INSERT INTO logs (action) VALUES (:action)", {"action": "user_created"})
    # Auto-commit on success, rollback on exception
```

### Usage in Services

```python
# checkpoint/sql.py
class SQLCheckpointer:
    def __init__(self, db: Database):  # Takes unified Database
        self.db = db

    async def save_checkpoint(self, ...):
        # Write raw SQL with named parameters
        await self.db.execute("""
            INSERT INTO pipeline_checkpoints (thread_id, pipeline_id, state)
            VALUES (:thread_id, :pipeline_id, :state)
        """, {
            "thread_id": thread_id,
            "pipeline_id": pipeline_id,
            "state": json.dumps(state)
        })
```

---

## Action Items

### Decision Needed

**Which approach?**
- [ ] **Option A: Adopt SQLAlchemy Core** - Fastest, most robust
- [ ] **Option B: Build custom abstraction** - More work, more control

### If SQLAlchemy Core (Option A)

1. Add dependency: `sqlalchemy>=2.0`
2. Create new `database/core.py` with unified Database class
3. Update all services to use new interface
4. Run tests
5. Delete old DatabaseManager, DatabaseInterfaceAdapter

**Estimated time:** 1-2 days

### If Custom (Option B)

1. Design Dialect system
2. Implement PostgreSQL dialect (asyncpg)
3. Implement SQLite dialect (aiosqlite)
4. Add connection pooling
5. Add parameter conversion
6. Add transaction management
7. Update all services
8. Extensive testing

**Estimated time:** 1-2 weeks

---

## My Recommendation

**Use SQLAlchemy Core**

**Why:**
- ✅ You get to write raw SQL (your preference)
- ✅ Solves 90% of current problems immediately
- ✅ Battle-tested, mature, well-documented
- ✅ Handles all the dialect differences automatically
- ✅ Connection pooling built-in
- ✅ Transaction management built-in
- ✅ Can switch databases without code changes
- ✅ Small footprint (Core only, no ORM)

**This is the pragmatic choice.** Building our own would be a fun learning exercise, but SQLAlchemy Core already solved this exact problem perfectly.

---

## Next Steps

1. **Review this document**
2. **Decide: SQLAlchemy Core or Custom?**
3. **Create implementation plan**
4. **Execute**

Let me know what you think!
