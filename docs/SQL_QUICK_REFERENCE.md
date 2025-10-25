# SQL Translation Quick Reference

## 🎯 TL;DR

**Always write in PostgreSQL syntax** - it automatically translates to all databases.

```python
from ia_modules.database import DatabaseManager

db = DatabaseManager("mysql://localhost/db")  # Works with ANY database!
db.connect()

# Write PostgreSQL syntax - automatic translation happens
db.execute("""
    CREATE TABLE users (
        id SERIAL PRIMARY KEY,              -- Auto-translates to AUTO_INCREMENT/IDENTITY
        email VARCHAR(255),                  -- Auto-translates to NVARCHAR for MSSQL
        is_active BOOLEAN DEFAULT TRUE,      -- Auto-translates to TINYINT(1)/BIT/INTEGER
        metadata JSONB,                      -- Auto-translates to JSON/NVARCHAR(MAX)/TEXT
        user_id UUID,                        -- Auto-translates to CHAR(36)/UNIQUEIDENTIFIER/TEXT
        created_at TIMESTAMP DEFAULT NOW()   -- Auto-translates to GETDATE() for MSSQL
    )
""")
```

---

## 📋 Common Translations

### Auto-Increment Primary Key

| Write This (PostgreSQL) | Becomes This |
|-------------------------|--------------|
| `id SERIAL PRIMARY KEY` | **MySQL:** `INT AUTO_INCREMENT`<br>**MSSQL:** `INT IDENTITY(1,1)`<br>**SQLite:** `INTEGER AUTOINCREMENT` |
| `id INTEGER PRIMARY KEY` | **MySQL:** `INTEGER AUTO_INCREMENT`<br>**MSSQL:** `INTEGER IDENTITY(1,1)`<br>**SQLite:** `INTEGER PRIMARY KEY` |

### Boolean Fields

| Write This (PostgreSQL) | Becomes This |
|-------------------------|--------------|
| `is_active BOOLEAN` | **MySQL:** `TINYINT(1)`<br>**MSSQL:** `BIT`<br>**SQLite:** `INTEGER` |
| `DEFAULT TRUE` | **All:** `DEFAULT 1` |
| `DEFAULT FALSE` | **All:** `DEFAULT 0` |

### UUID Fields

| Write This (PostgreSQL) | Becomes This |
|-------------------------|--------------|
| `user_id UUID` | **MySQL:** `CHAR(36)`<br>**MSSQL:** `UNIQUEIDENTIFIER`<br>**SQLite:** `TEXT` |
| `DEFAULT gen_random_uuid()` | **MySQL:** `UUID()`<br>**MSSQL:** `NEWID()`<br>**SQLite:** `lower(hex(randomblob(16)))` |

### JSON Fields

| Write This (PostgreSQL) | Becomes This |
|-------------------------|--------------|
| `metadata JSONB` | **MySQL:** `JSON`<br>**MSSQL:** `NVARCHAR(MAX)`<br>**SQLite:** `TEXT` |
| `settings JSON` | **MySQL:** `JSON`<br>**MSSQL:** `NVARCHAR(MAX)`<br>**SQLite:** `TEXT` |

### String Fields

| Write This (PostgreSQL) | Becomes This |
|-------------------------|--------------|
| `name VARCHAR(255)` | **MySQL:** `VARCHAR(255)`<br>**MSSQL:** `NVARCHAR(255)`<br>**SQLite:** `TEXT` |
| `description TEXT` | **MySQL:** `TEXT`<br>**MSSQL:** `NVARCHAR(MAX)`<br>**SQLite:** `TEXT` |

### Date/Time Fields

| Write This (PostgreSQL) | Becomes This |
|-------------------------|--------------|
| `created_at TIMESTAMP` | **MySQL:** `TIMESTAMP`<br>**MSSQL:** `DATETIME2`<br>**SQLite:** `TEXT` |
| `DEFAULT NOW()` | **MySQL:** `NOW()`<br>**MSSQL:** `GETDATE()`<br>**SQLite:** `CURRENT_TIMESTAMP` |

---

## ✅ Do's

### ✅ Use PostgreSQL Types
```sql
-- Good
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    is_available BOOLEAN,
    metadata JSONB
);
```

### ✅ Use PostgreSQL Functions
```sql
-- Good
INSERT INTO events (created_at) VALUES (NOW());
INSERT INTO sessions (id) VALUES (gen_random_uuid());
```

### ✅ Use Standard Syntax
```sql
-- Good
SELECT * FROM users WHERE is_active = TRUE;
INSERT INTO logs (message) VALUES ('test');
```

---

## ❌ Don'ts

### ❌ Don't Use Database-Specific Syntax
```sql
-- Bad - MySQL-specific
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY  -- Use SERIAL instead
);

-- Bad - MSSQL-specific
CREATE TABLE users (
    id INT IDENTITY(1,1) PRIMARY KEY   -- Use SERIAL instead
);
```

### ❌ Don't Use Database-Specific Functions
```sql
-- Bad - MySQL-specific
SELECT DATE_SUB(NOW(), INTERVAL 7 DAY);  -- Use standard SQL

-- Bad - MSSQL-specific
SELECT GETDATE();  -- Use NOW() instead
```

### ❌ Don't Hardcode Database Types
```python
# Bad
if database_type == "mysql":
    sql = "CREATE TABLE users (id INT AUTO_INCREMENT)"
elif database_type == "postgresql":
    sql = "CREATE TABLE users (id SERIAL)"

# Good
sql = "CREATE TABLE users (id SERIAL PRIMARY KEY)"  # Works everywhere
db.execute(sql)
```

---

## 🔧 Common Patterns

### Pattern 1: User Table
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    is_admin BOOLEAN DEFAULT FALSE,
    profile JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### Pattern 2: Session Table
```sql
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER NOT NULL,
    token TEXT NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Pattern 3: Log Table
```sql
CREATE TABLE logs (
    id SERIAL PRIMARY KEY,
    level VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Pattern 4: Product Table
```sql
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    sku VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    price DECIMAL(10,2) NOT NULL,
    is_available BOOLEAN DEFAULT TRUE,
    attributes JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

---

## 🧪 Testing

### Test on All Databases
```python
from ia_modules.database import DatabaseManager

# Write once
sql = """
    CREATE TABLE test (
        id SERIAL PRIMARY KEY,
        data JSONB,
        created_at TIMESTAMP DEFAULT NOW()
    )
"""

# Test on all databases
for db_url in [
    "postgresql://localhost/test",
    "mysql://localhost/test",
    "mssql://localhost/test",
    "sqlite:///test.db"
]:
    db = DatabaseManager(db_url)
    db.connect()
    db.execute(sql)  # Automatic translation!
    assert db.table_exists("test")
    db.disconnect()
```

---

## 🐛 Troubleshooting

### "Column doesn't have a default value" (MySQL)
**Problem:** Using `INTEGER PRIMARY KEY` without data
**Solution:** Translation adds `AUTO_INCREMENT` automatically - this should work. If not, check your ia_modules version.

### "Invalid default value" (MSSQL)
**Problem:** Using `NOW()` in MSSQL context
**Solution:** Translation converts to `GETDATE()` - this should work automatically.

### "Data type UUID does not exist" (MySQL/MSSQL)
**Problem:** Using `UUID` type
**Solution:** Translation converts to `CHAR(36)` (MySQL) or `UNIQUEIDENTIFIER` (MSSQL) - this should work automatically.

### "Cannot convert JSONB" (MSSQL/SQLite)
**Problem:** Using `JSONB` type
**Solution:** Translation converts to `NVARCHAR(MAX)` (MSSQL) or `TEXT` (SQLite) - this should work automatically.

---

## 📚 Full Documentation

For complete details, see [SQL_TRANSLATION.md](./SQL_TRANSLATION.md)

---

## 🎓 Remember

1. **PostgreSQL is the canonical syntax** - always write in PostgreSQL
2. **Translation is automatic** - just use `db.execute()`
3. **Test on all targets** - ensure translations work correctly
4. **Don't use database-specific syntax** - stick to PostgreSQL
5. **Migrations work everywhere** - write once, deploy anywhere

**Your SQL is now database-agnostic!** 🚀
