# Multi-Backend Database Testing

The IA Modules database system supports multiple database backends through automatic SQL translation. To ensure compatibility, tests are parameterized to run against all configured databases.

## Supported Databases

- **SQLite** - Always available (uses in-memory database for tests)
- **PostgreSQL** - Requires connection URL via environment variable
- **MySQL** - Requires connection URL via environment variable
- **MSSQL** - Requires connection URL via environment variable

## Running Tests

### SQLite Only (Default)

```bash
pytest tests/unit/test_database_multi_backend.py
```

SQLite tests run automatically without any configuration.

### With PostgreSQL

```bash
export TEST_POSTGRESQL_URL="postgresql://user:password@localhost:5432/testdb"
pytest tests/unit/test_database_multi_backend.py
```

### With MySQL

```bash
export TEST_MYSQL_URL="mysql://user:password@localhost:3306/testdb"
pytest tests/unit/test_database_multi_backend.py
```

### With MSSQL

```bash
export TEST_MSSQL_URL="mssql://user:password@localhost:1433/testdb"
pytest tests/unit/test_database_multi_backend.py
```

### With All Databases

```bash
export TEST_POSTGRESQL_URL="postgresql://user:password@localhost:5432/testdb"
export TEST_MYSQL_URL="mysql://user:password@localhost:3306/testdb"
export TEST_MSSQL_URL="mssql://user:password@localhost:1433/testdb"
pytest tests/unit/test_database_multi_backend.py
```

## Test Database Setup

### PostgreSQL

```bash
# Create test database
createdb ia_modules_test

# Or with custom user
createdb -U myuser ia_modules_test
```

### MySQL

```bash
# Create test database
mysql -u root -p -e "CREATE DATABASE ia_modules_test;"

# Create test user (optional)
mysql -u root -p -e "CREATE USER 'testuser'@'localhost' IDENTIFIED BY 'testpass';"
mysql -u root -p -e "GRANT ALL PRIVILEGES ON ia_modules_test.* TO 'testuser'@'localhost';"
```

### MSSQL

```sql
-- Create test database
CREATE DATABASE ia_modules_test;

-- Create test user (optional)
CREATE LOGIN testuser WITH PASSWORD = 'TestPass123!';
USE ia_modules_test;
CREATE USER testuser FOR LOGIN testuser;
ALTER ROLE db_owner ADD MEMBER testuser;
```

## Docker Compose Setup

For convenience, you can use Docker to run test databases:

```yaml
# docker-compose.test.yml
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

  mysql:
    image: mysql:8
    environment:
      MYSQL_DATABASE: ia_modules_test
      MYSQL_USER: testuser
      MYSQL_PASSWORD: testpass
      MYSQL_ROOT_PASSWORD: rootpass
    ports:
      - "3306:3306"

  mssql:
    image: mcr.microsoft.com/mssql/server:2022-latest
    environment:
      ACCEPT_EULA: Y
      SA_PASSWORD: TestPass123!
      MSSQL_DB: ia_modules_test
    ports:
      - "1433:1433"
```

Run with:

```bash
docker-compose -f docker-compose.test.yml up -d

# Set environment variables
export TEST_POSTGRESQL_URL="postgresql://testuser:testpass@localhost:5432/ia_modules_test"
export TEST_MYSQL_URL="mysql://testuser:testpass@localhost:3306/ia_modules_test"
export TEST_MSSQL_URL="mssql://sa:TestPass123!@localhost:1433/ia_modules_test"

# Run tests
pytest tests/unit/test_database_multi_backend.py

# Cleanup
docker-compose -f docker-compose.test.yml down
```

## What Gets Tested

### Basic Operations
- Connection establishment
- Execute with named parameters
- Fetch one/all with named parameters
- Table existence checks

### Migration System
- Migration table creation
- Migration tracking
- Migration idempotency (running twice doesn't duplicate)

### SQL Translation
Tests that PostgreSQL syntax is correctly translated:
- `BOOLEAN` → `INTEGER` (SQLite), stays `BOOLEAN` (PostgreSQL/MySQL)
- `JSONB` → `TEXT` (SQLite), stays `JSONB` (PostgreSQL)
- `VARCHAR(n)` → `TEXT` (SQLite), stays `VARCHAR(n)` (others)
- `UUID` → `TEXT` (SQLite), stays `UUID` (PostgreSQL)
- `NOW()` → `CURRENT_TIMESTAMP` (SQLite)

### Async Operations
- `execute_async()` method
- `execute_script()` with multiple statements

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Multi-Backend Tests

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

      mysql:
        image: mysql:8
        env:
          MYSQL_DATABASE: ia_modules_test
          MYSQL_USER: testuser
          MYSQL_PASSWORD: testpass
          MYSQL_ROOT_PASSWORD: rootpass
        options: >-
          --health-cmd="mysqladmin ping"
          --health-interval=10s
          --health-timeout=5s
          --health-retries=5
        ports:
          - 3306:3306

    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest pytest-asyncio

      - name: Run multi-backend tests
        env:
          TEST_POSTGRESQL_URL: postgresql://testuser:testpass@localhost:5432/ia_modules_test
          TEST_MYSQL_URL: mysql://testuser:testpass@localhost:3306/ia_modules_test
        run: |
          pytest tests/unit/test_database_multi_backend.py -v
```

## Fixtures

The `conftest.py` provides two parameterized fixtures:

### `db_config`
Provides `ConnectionConfig` instances for each configured database.

```python
def test_something(db_config):
    db = DatabaseManager(db_config)
    # ... test code
```

### `db_manager`
Provides connected `DatabaseManager` instances with automatic cleanup.

```python
def test_something(db_manager):
    # db_manager is already connected
    result = db_manager.execute("SELECT 1")
    # ... test code
    # cleanup happens automatically
```

## Writing Multi-Backend Tests

### DO: Use Generic SQL in Tests

```python
def test_basic_query(db_manager):
    # Generic SQL works everywhere
    db_manager.execute("""
        CREATE TABLE test_table (
            id INTEGER PRIMARY KEY,
            name TEXT
        )
    """)
```

### DO: Use PostgreSQL Syntax (Will Be Translated)

```python
def test_with_postgres_syntax(db_manager):
    # PostgreSQL syntax - translated automatically
    db_manager.execute("""
        CREATE TABLE test_table (
            id UUID PRIMARY KEY,
            data JSONB,
            is_active BOOLEAN DEFAULT TRUE
        )
    """)
```

### DON'T: Use Database-Specific Features

```python
def test_bad_example(db_manager):
    # DON'T - PostgreSQL-specific array type won't translate
    db_manager.execute("""
        CREATE TABLE test_table (
            tags TEXT[]
        )
    """)
```

### DO: Skip Tests for Specific Backends

```python
import pytest
from ia_modules.database import DatabaseType

def test_postgres_only(db_manager):
    if db_manager.config.database_type != DatabaseType.POSTGRESQL:
        pytest.skip("PostgreSQL-only feature")

    # PostgreSQL-specific test
    db_manager.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
```

## Troubleshooting

### Tests Fail on PostgreSQL but Pass on SQLite

This usually means:
1. Async/sync method mismatch (PostgreSQL is stricter)
2. Transaction handling issues
3. SQL syntax that needs translation

### Tests Fail on MySQL

Common issues:
1. Reserved keywords (use backticks: `` `order` ``)
2. Different default collations
3. Strict mode rejecting NULL values

### Tests Fail on MSSQL

Common issues:
1. Different quote characters (`[table]` vs `"table"`)
2. No boolean type (use `BIT`)
3. Different function names

## Performance Notes

- SQLite tests are fastest (in-memory)
- PostgreSQL/MySQL/MSSQL tests require network round trips
- Consider running SQLite tests in development, all backends in CI
