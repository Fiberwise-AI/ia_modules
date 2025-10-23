# PostgreSQL Testing Setup Guide

This guide explains how to set up and run PostgreSQL integration tests.

## Why PostgreSQL Tests Matter

The multi-backend tests ([test_database_multi_backend.py](unit/test_database_multi_backend.py)) only use SQLite by default. This misses critical PostgreSQL-specific issues:

1. **Transaction Strictness** - PostgreSQL aborts transactions on async/sync mismatches (SQLite doesn't)
2. **Type System** - PostgreSQL has UUID, JSONB, BOOLEAN types (SQLite converts to TEXT/INTEGER)
3. **Parameter Binding** - PostgreSQL uses `%(name)s` format (SQLite uses `?`)
4. **SQL Translation** - PostgreSQL should NOT have SQL translated (stays as-is)

## Quick Start

### 1. Install PostgreSQL

**macOS:**
```bash
brew install postgresql@15
brew services start postgresql@15
```

**Ubuntu/Debian:**
```bash
sudo apt-get install postgresql-15
sudo systemctl start postgresql
```

**Windows:**
Download installer from https://www.postgresql.org/download/windows/

### 2. Create Test Database

```bash
# Connect as postgres user
psql -U postgres

# In psql console:
CREATE DATABASE ia_modules_test;
CREATE USER testuser WITH PASSWORD 'testpass';
GRANT ALL PRIVILEGES ON DATABASE ia_modules_test TO testuser;
\q
```

Or one-liner:
```bash
createdb ia_modules_test
```

### 3. Set Environment Variable

**Bash/Zsh:**
```bash
export TEST_POSTGRESQL_URL="postgresql://testuser:testpass@localhost:5432/ia_modules_test"
```

**PowerShell:**
```powershell
$env:TEST_POSTGRESQL_URL="postgresql://testuser:testpass@localhost:5432/ia_modules_test"
```

**Or create a `.env` file in tests/ directory:**
```bash
cd tests/
cp .env.example .env
# Edit .env and set TEST_POSTGRESQL_URL
```

### 4. Run Tests

```bash
# Run PostgreSQL integration tests
pytest tests/integration/test_postgresql_integration.py -v

# Run all tests including PostgreSQL
pytest tests/ -v
```

## Docker Setup (Recommended for CI/CD)

### Using Docker Compose

Create `docker-compose.test.yml`:

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: ia_modules_test
      POSTGRES_USER: testuser
      POSTGRES_PASSWORD: testpass
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U testuser"]
      interval: 5s
      timeout: 5s
      retries: 5
```

Run:
```bash
# Start PostgreSQL
docker-compose -f docker-compose.test.yml up -d

# Wait for health check
docker-compose -f docker-compose.test.yml ps

# Set connection string
export TEST_POSTGRESQL_URL="postgresql://testuser:testpass@localhost:5432/ia_modules_test"

# Run tests
pytest tests/integration/test_postgresql_integration.py -v

# Cleanup
docker-compose -f docker-compose.test.yml down -v
```

### Using Docker Run

```bash
# Start PostgreSQL container
docker run -d \
  --name postgres-test \
  -e POSTGRES_DB=ia_modules_test \
  -e POSTGRES_USER=testuser \
  -e POSTGRES_PASSWORD=testpass \
  -p 5432:5432 \
  postgres:15

# Wait for startup (or check with docker logs)
sleep 5

# Set connection string
export TEST_POSTGRESQL_URL="postgresql://testuser:testpass@localhost:5432/ia_modules_test"

# Run tests
pytest tests/integration/test_postgresql_integration.py -v

# Cleanup
docker stop postgres-test
docker rm postgres-test
```

## Test Coverage

The PostgreSQL integration tests verify:

### Parameter Binding Tests
- ✅ Named parameters with INSERT/UPDATE/DELETE
- ✅ Special characters (quotes, newlines) in parameters
- ✅ NULL value handling
- ✅ Multiple inserts with different parameters

### PostgreSQL Data Types
- ✅ UUID type (NOT translated to TEXT)
- ✅ JSONB type (NOT translated to TEXT)
- ✅ BOOLEAN type (NOT translated to INTEGER)
- ✅ TIMESTAMP with NOW() function (NOT translated)

### Transaction Tests
- ✅ Async execute maintains transaction consistency
- ✅ Rollback on errors (no partial commits)

### Migration Tests
- ✅ Migration table creation in PostgreSQL
- ✅ Migration records are persisted
- ✅ Migration idempotency (no duplicates)
- ✅ PostgreSQL-specific syntax in migrations works

### Data Verification Tests
- ✅ Data persists across multiple queries
- ✅ UPDATE operations persist changes
- ✅ DELETE operations remove data
- ✅ Complex JOINs work correctly

## What Makes These Tests Special

### 1. Actual Database Verification

Unlike the multi-backend tests that just check `result.success`, these tests:
```python
# Insert data
db.execute("INSERT INTO users (name) VALUES (:name)", {"name": "Alice"})

# Verify it ACTUALLY exists in the database
row = db.fetch_one("SELECT * FROM users WHERE name = :name", {"name": "Alice"})
assert row is not None  # Proves data is really there
assert row["name"] == "Alice"
```

### 2. PostgreSQL-Specific Type Testing

```python
# Create table with UUID (PostgreSQL native type)
db.execute("CREATE TABLE test (id UUID PRIMARY KEY)")

# Verify the column is ACTUALLY a UUID in PostgreSQL
row = db.fetch_one("""
    SELECT data_type FROM information_schema.columns
    WHERE table_name = 'test' AND column_name = 'id'
""")
assert row["data_type"] == "uuid"  # Not TEXT!
```

### 3. Transaction Consistency

```python
# This was the bug that only PostgreSQL caught
result = await db.execute_async("INSERT INTO test VALUES (:val)", {"val": 1})
assert result.success

# If transaction failed, this would return None
row = db.fetch_one("SELECT * FROM test")
assert row is not None  # Proves transaction committed
```

### 4. Automatic Cleanup

Each test gets a unique table name with timestamp:
```python
test_id = "20251022_143052_123456"
db.execute(f"CREATE TABLE test_users_{test_id} ...")
```

After tests, cleanup drops all `test_*` tables:
```python
# Cleanup in fixture
for row in db.fetch_all("SELECT tablename FROM pg_tables WHERE tablename LIKE 'test_%'"):
    db.execute(f"DROP TABLE IF EXISTS {row['tablename']} CASCADE")
```

## CI/CD Integration

### GitHub Actions

```yaml
name: PostgreSQL Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_DB: ia_modules_test
          POSTGRES_USER: testuser
          POSTGRES_PASSWORD: testpass
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest pytest-asyncio

      - name: Run PostgreSQL integration tests
        env:
          TEST_POSTGRESQL_URL: postgresql://testuser:testpass@localhost:5432/ia_modules_test
        run: |
          pytest tests/integration/test_postgresql_integration.py -v
```

### GitLab CI

```yaml
test_postgresql:
  image: python:3.11
  services:
    - name: postgres:15
      alias: postgres
  variables:
    POSTGRES_DB: ia_modules_test
    POSTGRES_USER: testuser
    POSTGRES_PASSWORD: testpass
    TEST_POSTGRESQL_URL: postgresql://testuser:testpass@postgres:5432/ia_modules_test
  script:
    - pip install -e .
    - pip install pytest pytest-asyncio
    - pytest tests/integration/test_postgresql_integration.py -v
```

## Troubleshooting

### Tests are skipped

```
SKIPPED [1] PostgreSQL not configured (set TEST_POSTGRESQL_URL)
```

**Solution:** Set the environment variable:
```bash
export TEST_POSTGRESQL_URL="postgresql://user:pass@localhost:5432/ia_modules_test"
```

### Connection refused

```
psycopg2.OperationalError: could not connect to server: Connection refused
```

**Solutions:**
1. Check PostgreSQL is running: `pg_isready` or `docker ps`
2. Check port is correct (default 5432)
3. Check firewall allows connections

### Authentication failed

```
psycopg2.OperationalError: FATAL:  password authentication failed for user "testuser"
```

**Solutions:**
1. Verify password in connection string
2. Check user exists: `psql -U postgres -c "\du"`
3. Grant permissions: `GRANT ALL PRIVILEGES ON DATABASE ia_modules_test TO testuser;`

### Database does not exist

```
psycopg2.OperationalError: FATAL:  database "ia_modules_test" does not exist
```

**Solution:** Create the database:
```bash
createdb ia_modules_test
# or
psql -U postgres -c "CREATE DATABASE ia_modules_test;"
```

### Tests fail with "relation already exists"

**Solution:** The cleanup might have failed. Manually drop test tables:
```bash
psql -U testuser -d ia_modules_test -c "
SELECT 'DROP TABLE IF EXISTS ' || tablename || ' CASCADE;'
FROM pg_tables
WHERE schemaname = 'public' AND tablename LIKE 'test_%';
" | psql -U testuser -d ia_modules_test
```

## Comparing to SQLite Tests

| Aspect | SQLite Tests | PostgreSQL Tests |
|--------|-------------|------------------|
| **Transaction Strictness** | Lenient (allows async/sync mix) | Strict (aborts on errors) |
| **Data Types** | All become TEXT/INTEGER | Native UUID, JSONB, BOOLEAN |
| **SQL Translation** | Applied (TIMESTAMP→TEXT) | NOT applied (stays PostgreSQL) |
| **Parameter Format** | `?` positional | `%(name)s` named |
| **Database Location** | In-memory (`:memory:`) | Real database server |
| **Cleanup** | Automatic (destroyed on close) | Manual (DROP TABLE) |
| **CI/CD Setup** | None needed | Requires service container |
| **What It Tests** | Logic works in general | PostgreSQL-specific issues |

## Best Practices

1. **Always run PostgreSQL tests before merging** - SQLite tests alone are not sufficient
2. **Use Docker for consistency** - Same PostgreSQL version everywhere
3. **Clean up test data** - Use unique table names with timestamps
4. **Test actual data persistence** - Don't just check `result.success`
5. **Verify types are correct** - Query `information_schema.columns` to check types
6. **Test transactions** - Use async methods and verify commits work

## Summary

- **SQLite tests** (multi-backend): Fast, convenient, test basic logic
- **PostgreSQL tests** (integration): Slow, real database, catch production bugs

**Both are essential.** SQLite tests run in development, PostgreSQL tests run in CI/CD.
