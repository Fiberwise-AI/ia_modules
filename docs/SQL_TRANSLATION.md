# SQL Translation System

## Overview

The ia_modules database layer provides **automatic SQL translation** from **PostgreSQL syntax** (our canonical format) to all supported database backends: PostgreSQL, MySQL, MSSQL, and SQLite.

**Why PostgreSQL as the canonical syntax?**
- ✅ Most advanced SQL features (JSONB, UUID, advanced data types)
- ✅ Industry-standard syntax
- ✅ Rich type system
- ✅ Superior to SQLite's limited type system
- ✅ Widely adopted in enterprise applications

## Supported Databases

| Database | Support Level | Notes |
|----------|---------------|-------|
| **PostgreSQL** | ✅ Native | Canonical syntax - no translation needed |
| **MySQL** | ✅ Full | All features translated |
| **MSSQL** | ✅ Full | All features translated |
| **SQLite** | ✅ Full | All features translated |

## Write Once, Run Anywhere

Write your SQL in **PostgreSQL syntax** and it will automatically work on all supported databases:

```python
from ia_modules.database import DatabaseManager, ConnectionConfig, DatabaseType

# Works with ANY database!
db = DatabaseManager("postgresql://localhost/mydb")  # Or mysql://, mssql://, sqlite:///
db.connect()

# Write in PostgreSQL syntax - it translates automatically
db.execute("""
    CREATE TABLE users (
        id SERIAL PRIMARY KEY,
        email VARCHAR(255) NOT NULL,
        is_active BOOLEAN DEFAULT TRUE,
        metadata JSONB,
        user_id UUID,
        created_at TIMESTAMP DEFAULT NOW()
    )
""")
```

This SQL will be automatically translated to:
- **PostgreSQL**: No changes (native syntax)
- **MySQL**: `SERIAL` → `INT AUTO_INCREMENT`, `BOOLEAN` → `TINYINT(1)`, `JSONB` → `JSON`, `UUID` → `CHAR(36)`
- **MSSQL**: `SERIAL` → `INT IDENTITY(1,1)`, `BOOLEAN` → `BIT`, `JSONB` → `NVARCHAR(MAX)`, `UUID` → `UNIQUEIDENTIFIER`, `NOW()` → `GETDATE()`
- **SQLite**: `SERIAL` → `INTEGER AUTOINCREMENT`, `BOOLEAN` → `INTEGER`, `JSONB` → `TEXT`, `UUID` → `TEXT`

---

## Data Type Translations

### Auto-Incrementing Primary Keys

**PostgreSQL (canonical):**
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    -- or
    id INTEGER PRIMARY KEY  -- Also auto-increments
);
```

**Translations:**
| PostgreSQL | MySQL | MSSQL | SQLite |
|------------|-------|-------|--------|
| `SERIAL PRIMARY KEY` | `INT AUTO_INCREMENT` | `INT IDENTITY(1,1)` | `INTEGER AUTOINCREMENT` |
| `INTEGER PRIMARY KEY` | `INTEGER AUTO_INCREMENT` | `INTEGER IDENTITY(1,1)` | `INTEGER PRIMARY KEY` |
| `BIGSERIAL` | `BIGINT AUTO_INCREMENT` | `BIGINT IDENTITY(1,1)` | `INTEGER` |

### Boolean Types

**PostgreSQL (canonical):**
```sql
CREATE TABLE settings (
    is_enabled BOOLEAN DEFAULT TRUE,
    is_deleted BOOLEAN DEFAULT FALSE
);
```

**Translations:**
| PostgreSQL | MySQL | MSSQL | SQLite |
|------------|-------|-------|--------|
| `BOOLEAN` | `TINYINT(1)` | `BIT` | `INTEGER` |
| `TRUE` | `1` | `1` | `1` |
| `FALSE` | `0` | `0` | `0` |

**Important:** When querying boolean fields:
- PostgreSQL: Use `WHERE is_active = TRUE`
- Others: Use `WHERE is_active = 1` (or the translation handles it automatically)

### UUID Types

**PostgreSQL (canonical):**
```sql
CREATE TABLE sessions (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    token UUID DEFAULT gen_random_uuid()
);
```

**Translations:**
| PostgreSQL | MySQL | MSSQL | SQLite |
|------------|-------|-------|--------|
| `UUID` | `CHAR(36)` | `UNIQUEIDENTIFIER` | `TEXT` |
| `gen_random_uuid()` | `UUID()` | `NEWID()` | `lower(hex(randomblob(16)))` |

**Storage formats:**
- PostgreSQL: Native 128-bit UUID
- MySQL: String `'550e8400-e29b-41d4-a716-446655440000'`
- MSSQL: Native UNIQUEIDENTIFIER
- SQLite: String `'550e8400-e29b-41d4-a716-446655440000'`

### JSON/JSONB Types

**PostgreSQL (canonical):**
```sql
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    metadata JSONB NOT NULL,
    settings JSON
);
```

**Translations:**
| PostgreSQL | MySQL | MSSQL | SQLite |
|------------|-------|-------|--------|
| `JSONB` | `JSON` | `NVARCHAR(MAX)` | `TEXT` |
| `JSON` | `JSON` | `NVARCHAR(MAX)` | `TEXT` |

**Features:**
- PostgreSQL: Binary JSON with indexing support
- MySQL: Native JSON type with functions
- MSSQL: JSON stored as text, use `ISJSON()` for validation
- SQLite: JSON stored as text, use `json_valid()` for validation

### String Types

**PostgreSQL (canonical):**
```sql
CREATE TABLE posts (
    title VARCHAR(255) NOT NULL,
    slug VARCHAR(100),
    content TEXT
);
```

**Translations:**
| PostgreSQL | MySQL | MSSQL | SQLite |
|------------|-------|-------|--------|
| `VARCHAR(n)` | `VARCHAR(n)` | `NVARCHAR(n)` | `TEXT` |
| `VARCHAR` | `VARCHAR(255)` | `NVARCHAR(MAX)` | `TEXT` |
| `TEXT` | `TEXT` | `NVARCHAR(MAX)` | `TEXT` |
| `CHAR(n)` | `CHAR(n)` | `NCHAR(n)` | `TEXT` |

**Note:** MSSQL uses `NVARCHAR` (Unicode) instead of `VARCHAR` for better international character support.

### Timestamp Types

**PostgreSQL (canonical):**
```sql
CREATE TABLE events (
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Translations:**
| PostgreSQL | MySQL | MSSQL | SQLite |
|------------|-------|-------|--------|
| `TIMESTAMP` | `TIMESTAMP` | `DATETIME2` | `TEXT` |
| `NOW()` | `NOW()` | `GETDATE()` | `CURRENT_TIMESTAMP` |
| `CURRENT_TIMESTAMP` | `CURRENT_TIMESTAMP` | `GETDATE()` | `CURRENT_TIMESTAMP` |

**Storage formats:**
- PostgreSQL: 8 bytes, microsecond precision
- MySQL: 4 bytes, second precision (TIMESTAMP) or 8 bytes (DATETIME)
- MSSQL: 6-8 bytes, 100 nanosecond precision
- SQLite: TEXT in ISO-8601 format `'YYYY-MM-DD HH:MM:SS.SSS'`

---

## Function Translations

### Date/Time Functions

| PostgreSQL (canonical) | MySQL | MSSQL | SQLite |
|------------------------|-------|-------|--------|
| `NOW()` | `NOW()` | `GETDATE()` | `CURRENT_TIMESTAMP` |
| `CURRENT_DATE` | `CURDATE()` | `CAST(GETDATE() AS DATE)` | `date('now')` |
| `CURRENT_TIME` | `CURTIME()` | `CAST(GETDATE() AS TIME)` | `time('now')` |

### UUID Generation

| PostgreSQL (canonical) | MySQL | MSSQL | SQLite |
|------------------------|-------|-------|--------|
| `gen_random_uuid()` | `UUID()` | `NEWID()` | `lower(hex(randomblob(16)))` |

---

## Type Casting

**PostgreSQL (canonical) uses `::` syntax:**
```sql
SELECT '{"key": "value"}'::jsonb;
SELECT 'example'::varchar;
```

**Automatic translation:**
- PostgreSQL: `::jsonb` → kept as-is
- MySQL: `::jsonb` → removed (not needed)
- MSSQL: `::jsonb` → removed (not needed)
- SQLite: `::jsonb` → removed (not needed)

---

## Best Practices

### 1. Always Use PostgreSQL Syntax

✅ **Correct:**
```sql
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    status VARCHAR(50),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

❌ **Avoid database-specific syntax:**
```sql
-- Don't use MySQL-specific syntax
CREATE TABLE orders (
    id INT AUTO_INCREMENT PRIMARY KEY,  -- Use SERIAL instead
    status VARCHAR(50),
    metadata JSON,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 2. Use Standard SQL When Possible

✅ **Good:**
```sql
SELECT * FROM users WHERE status = 'active' AND created_at > NOW() - INTERVAL '7 days';
```

❌ **Database-specific:**
```sql
-- MySQL-specific
SELECT * FROM users WHERE status = 'active' AND created_at > DATE_SUB(NOW(), INTERVAL 7 DAY);
```

### 3. Test Migrations on All Targets

When creating database migrations:

1. Write in PostgreSQL syntax
2. Test on PostgreSQL first
3. Test on other databases to verify translation
4. Check the translated SQL in logs if needed

### 4. Handle Edge Cases

Some features don't translate perfectly:

**Window Functions:** Supported on PostgreSQL, MySQL 8+, MSSQL. Not on SQLite.
```sql
-- PostgreSQL
SELECT name, ROW_NUMBER() OVER (ORDER BY created_at) as row_num
FROM users;
```

**Full-Text Search:** Each database has different syntax.
```sql
-- PostgreSQL
SELECT * FROM articles WHERE to_tsvector('english', content) @@ to_tsquery('search');

-- MySQL
SELECT * FROM articles WHERE MATCH(content) AGAINST('search');

-- MSSQL
SELECT * FROM articles WHERE CONTAINS(content, 'search');
```

**Solution:** Use the database abstraction for complex features, or write database-specific code when needed.

---

## Migration Examples

### Example 1: User Table

**PostgreSQL source:**
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    profile JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

**Automatic translations:**

<details>
<summary><strong>MySQL</strong></summary>

```sql
CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    is_active TINYINT(1) DEFAULT 1,
    profile JSON,
    created_at TIMESTAMP DEFAULT NOW()
);
```
</details>

<details>
<summary><strong>MSSQL</strong></summary>

```sql
CREATE TABLE users (
    id INT PRIMARY KEY IDENTITY(1,1),
    username NVARCHAR(50) UNIQUE NOT NULL,
    email NVARCHAR(255) UNIQUE NOT NULL,
    is_active BIT DEFAULT 1,
    profile NVARCHAR(MAX),
    created_at DATETIME2 DEFAULT GETDATE()
);
```
</details>

<details>
<summary><strong>SQLite</strong></summary>

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    is_active INTEGER DEFAULT 1,
    profile TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```
</details>

### Example 2: Session Table with UUIDs

**PostgreSQL source:**
```sql
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    token TEXT NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
```

**Automatic translations:**

<details>
<summary><strong>MySQL</strong></summary>

```sql
CREATE TABLE sessions (
    id CHAR(36) PRIMARY KEY DEFAULT UUID(),
    user_id CHAR(36) NOT NULL,
    token TEXT NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
```
</details>

<details>
<summary><strong>MSSQL</strong></summary>

```sql
CREATE TABLE sessions (
    id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    user_id UNIQUEIDENTIFIER NOT NULL,
    token NVARCHAR(MAX) NOT NULL,
    expires_at DATETIME2 NOT NULL,
    created_at DATETIME2 DEFAULT GETDATE()
);
```
</details>

<details>
<summary><strong>SQLite</strong></summary>

```sql
CREATE TABLE sessions (
    id TEXT PRIMARY KEY DEFAULT lower(hex(randomblob(16))),
    user_id TEXT NOT NULL,
    token TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```
</details>

---

## Testing Your SQL

### Enable Translation Logging

To see the translated SQL:

```python
import logging

# Enable DEBUG logging for database manager
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('ia_modules.database.manager')
logger.setLevel(logging.DEBUG)

db = DatabaseManager("mysql://localhost/test")
db.connect()

# You'll see logs like:
# DEBUG: Translating SQL for DatabaseType.MYSQL
# DEBUG: Original: CREATE TABLE users (id SERIAL PRIMARY KEY);
# DEBUG: Translated: CREATE TABLE users (id INT PRIMARY KEY AUTO_INCREMENT);

db.execute("CREATE TABLE users (id SERIAL PRIMARY KEY);")
```

### Test Script

```python
from ia_modules.database import DatabaseManager, ConnectionConfig, DatabaseType

# Test SQL on all databases
databases = [
    ("postgresql://testuser:testpass@localhost:5432/test", "PostgreSQL"),
    ("mysql://testuser:testpass@localhost:3306/test", "MySQL"),
    ("mssql://sa:TestPass123!@localhost:1433/test", "MSSQL"),
    ("sqlite:///test.db", "SQLite"),
]

test_sql = """
    CREATE TABLE test_translation (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255),
        is_active BOOLEAN DEFAULT TRUE,
        metadata JSONB,
        created_at TIMESTAMP DEFAULT NOW()
    )
"""

for url, db_name in databases:
    print(f"\nTesting {db_name}...")
    try:
        db = DatabaseManager(url)
        db.connect()

        # Drop table if exists
        try:
            db.execute("DROP TABLE IF EXISTS test_translation")
        except:
            pass

        # Create table with translation
        db.execute(test_sql)

        # Verify table exists
        if db.table_exists("test_translation"):
            print(f"✅ {db_name}: Table created successfully")
        else:
            print(f"❌ {db_name}: Table creation failed")

        db.disconnect()
    except Exception as e:
        print(f"❌ {db_name}: Error - {e}")
```

---

## Advanced Features

### Custom Translation Rules

If you need to add custom translation rules, extend the `DatabaseManager._translate_sql()` method:

```python
from ia_modules.database import DatabaseManager

class CustomDatabaseManager(DatabaseManager):
    def _translate_sql(self, sql: str) -> str:
        # Call parent translation first
        result = super()._translate_sql(sql)

        # Add custom rules
        if self.config.database_type == DatabaseType.MYSQL:
            # Custom MySQL transformations
            result = result.replace('CUSTOM_TYPE', 'VARCHAR(255)')

        return result
```

### Bypassing Translation

To bypass translation (use raw SQL):

```python
# Translation is automatic, but you can use database-specific syntax
# when absolutely necessary

if db.config.database_type == DatabaseType.MYSQL:
    db.execute("CREATE TABLE foo (id INT AUTO_INCREMENT PRIMARY KEY)")
elif db.config.database_type == DatabaseType.POSTGRESQL:
    db.execute("CREATE TABLE foo (id SERIAL PRIMARY KEY)")
```

**Note:** This is NOT recommended. Use PostgreSQL syntax and let the translator handle it.

---

## Limitations

### Features Not Translated

Some PostgreSQL-specific features cannot be automatically translated:

1. **Array Types** (`INTEGER[]`)
   - PostgreSQL: Native support
   - Others: Store as JSON or separate table

2. **ENUM Types**
   - PostgreSQL: Native ENUM
   - MySQL: ENUM (similar)
   - MSSQL: Use CHECK constraints or lookup tables
   - SQLite: Use CHECK constraints

3. **Advanced Indexes**
   - PostgreSQL: GIN, GiST indexes for JSONB/arrays
   - Others: Standard B-tree indexes only

4. **Full-Text Search**
   - Each database has different syntax
   - Use database-specific implementations

5. **Window Functions**
   - PostgreSQL: Full support
   - MySQL: 8.0+ only
   - MSSQL: Full support
   - SQLite: 3.25+ only

### Workarounds

For advanced features, use conditional logic:

```python
if db.config.database_type == DatabaseType.POSTGRESQL:
    # Use advanced PostgreSQL features
    db.execute("CREATE INDEX idx_meta ON docs USING GIN (metadata)")
else:
    # Fallback for other databases
    db.execute("CREATE INDEX idx_meta ON docs (metadata)")
```

---

## Troubleshooting

### SQL Not Translating

**Problem:** SQL seems to not be translated
**Solution:** Check if you're using `execute()` method - translation happens automatically in `_execute_raw()`

### Type Mismatch Errors

**Problem:** "Data type mismatch" errors
**Solution:** Check the translation table above. Some types may need explicit handling.

### Boolean Values Not Working

**Problem:** `WHERE is_active = TRUE` fails on MySQL/MSSQL
**Solution:** The translation handles this automatically. If you're using raw SQL, use `1` instead of `TRUE`.

### UUID Errors

**Problem:** UUID format errors
**Solution:** Ensure you're using string format `'550e8400-e29b-41d4-a716-446655440000'` when inserting into MySQL/SQLite

---

## Reference

### Complete Translation Table

| PostgreSQL | MySQL | MSSQL | SQLite | Notes |
|------------|-------|-------|--------|-------|
| `SERIAL` | `INT AUTO_INCREMENT` | `INT IDENTITY(1,1)` | `INTEGER AUTOINCREMENT` | Auto-increment integer |
| `BIGSERIAL` | `BIGINT AUTO_INCREMENT` | `BIGINT IDENTITY(1,1)` | `INTEGER` | Auto-increment big integer |
| `INTEGER PRIMARY KEY` | `INTEGER AUTO_INCREMENT` | `INTEGER IDENTITY(1,1)` | `INTEGER PRIMARY KEY` | Auto-incrementing PK |
| `BOOLEAN` | `TINYINT(1)` | `BIT` | `INTEGER` | Boolean type |
| `TRUE` | `1` | `1` | `1` | Boolean true |
| `FALSE` | `0` | `0` | `0` | Boolean false |
| `UUID` | `CHAR(36)` | `UNIQUEIDENTIFIER` | `TEXT` | UUID type |
| `JSONB` | `JSON` | `NVARCHAR(MAX)` | `TEXT` | Binary JSON |
| `JSON` | `JSON` | `NVARCHAR(MAX)` | `TEXT` | JSON text |
| `VARCHAR(n)` | `VARCHAR(n)` | `NVARCHAR(n)` | `TEXT` | Variable char |
| `VARCHAR` | `VARCHAR(255)` | `NVARCHAR(MAX)` | `TEXT` | Var char no limit |
| `TEXT` | `TEXT` | `NVARCHAR(MAX)` | `TEXT` | Long text |
| `TIMESTAMP` | `TIMESTAMP` | `DATETIME2` | `TEXT` | Timestamp |
| `NOW()` | `NOW()` | `GETDATE()` | `CURRENT_TIMESTAMP` | Current time |
| `gen_random_uuid()` | `UUID()` | `NEWID()` | Function | Generate UUID |
| `CURRENT_DATE` | `CURDATE()` | `CAST(GETDATE() AS DATE)` | `date('now')` | Current date |
| `CURRENT_TIME` | `CURTIME()` | `CAST(GETDATE() AS TIME)` | `time('now')` | Current time |

---

## Summary

- ✅ **Write once in PostgreSQL syntax**
- ✅ **Run anywhere** (PostgreSQL, MySQL, MSSQL, SQLite)
- ✅ **Automatic translation** - no manual conversion needed
- ✅ **Production-ready** - all translations tested
- ✅ **Type-safe** - proper type mapping for each database
- ✅ **No lock-in** - switch databases without rewriting SQL

The SQL translation system allows you to focus on building features, not managing database compatibility! 🚀
