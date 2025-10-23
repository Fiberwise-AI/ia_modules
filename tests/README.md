# IA Modules Test Suite

Comprehensive testing for the IA Modules pipeline system with multi-database backend support.

## Test Structure

```
tests/
├── unit/                           # Unit tests (fast, isolated)
│   ├── test_core.py               # Core pipeline logic
│   ├── test_database_manager.py   # Database manager (SQLite only)
│   ├── test_database_multi_backend.py  # Multi-database tests (parameterized)
│   └── ...
├── integration/                    # Integration tests (real dependencies)
│   ├── test_postgresql_integration.py  # PostgreSQL-specific tests
│   └── ...
├── e2e/                           # End-to-end tests (full pipeline)
│   └── test_comprehensive_e2e.py
├── conftest.py                    # Shared fixtures
├── .env.example                   # Example environment config
├── MULTI_BACKEND_TESTING.md       # Multi-database testing guide
├── POSTGRESQL_TESTING_SETUP.md    # PostgreSQL setup guide
└── README.md                      # This file
```

## Quick Start

### Run All Tests (SQLite only)
```bash
pytest tests/ -v
```

### Run Specific Test Categories
```bash
# Unit tests only (fast)
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# E2E tests only
pytest tests/e2e/ -v
```

### Run with PostgreSQL Tests
```bash
# 1. Set up PostgreSQL (see POSTGRESQL_TESTING_SETUP.md)
export TEST_POSTGRESQL_URL="postgresql://user:pass@localhost:5432/ia_modules_test"

# 2. Run all tests
pytest tests/ -v

# 3. Run only PostgreSQL tests
pytest tests/integration/test_postgresql_integration.py -v
```

## Test Categories

### 1. Unit Tests - `tests/unit/`

**Purpose:** Test individual components in isolation

**Characteristics:**
- Fast (< 1 second each)
- No external dependencies
- Use in-memory SQLite
- Mock external services

**Examples:**
- `test_core.py` - Pipeline step execution
- `test_database_manager.py` - Database operations
- `test_benchmark_framework.py` - Benchmarking logic

**Run:** `pytest tests/unit/ -v`

### 2. Multi-Backend Tests - `tests/unit/test_database_multi_backend.py`

**Purpose:** Verify database abstraction works across all supported databases

**Characteristics:**
- Parameterized fixtures run same tests on multiple databases
- SQLite always runs (in-memory)
- PostgreSQL/MySQL/MSSQL run if configured

**Coverage:**
- ✅ Basic operations (connect, execute, fetch)
- ✅ Migration system (table creation, tracking, idempotency)
- ✅ SQL translation (PostgreSQL → SQLite/MySQL/MSSQL)
- ✅ Async operations

**Run:**
```bash
# SQLite only
pytest tests/unit/test_database_multi_backend.py -v

# With PostgreSQL
export TEST_POSTGRESQL_URL="postgresql://user:pass@localhost:5432/testdb"
pytest tests/unit/test_database_multi_backend.py -v
```

### 3. PostgreSQL Integration Tests - `tests/integration/test_postgresql_integration.py`

**Purpose:** Verify PostgreSQL-specific behavior and catch production bugs

**Why Needed:** SQLite is too lenient and misses:
- Transaction strictness (async/sync mismatches)
- Native type support (UUID, JSONB, BOOLEAN)
- Parameter binding differences
- Migration SQL translation issues

**Coverage:**
- ✅ Parameter binding (named params, special chars, NULLs)
- ✅ PostgreSQL data types (UUID, JSONB, BOOLEAN, TIMESTAMP)
- ✅ Transaction consistency (async execute, rollback)
- ✅ Migration system (table creation, record persistence)
- ✅ Data verification (INSERT, UPDATE, DELETE, JOINs)

**Setup:** See [POSTGRESQL_TESTING_SETUP.md](POSTGRESQL_TESTING_SETUP.md)

**Run:**
```bash
export TEST_POSTGRESQL_URL="postgresql://user:pass@localhost:5432/ia_modules_test"
pytest tests/integration/test_postgresql_integration.py -v
```

### 4. Integration Tests - `tests/integration/`

**Purpose:** Test component interactions with real dependencies

**Characteristics:**
- Medium speed (1-5 seconds each)
- May use real databases, file systems, etc.
- Test end-to-end flows within subsystems

**Examples:**
- `test_error_handling_integration.py` - Retry/fallback mechanisms
- `test_importer_integration.py` - Pipeline import system

**Run:** `pytest tests/integration/ -v`

### 5. E2E Tests - `tests/e2e/`

**Purpose:** Test complete pipeline execution from start to finish

**Characteristics:**
- Slower (5-30 seconds each)
- Full system integration
- Test real-world scenarios

**Examples:**
- `test_comprehensive_e2e.py` - Various pipeline types
- `test_parallel_e2e.py` - Parallel execution

**Run:** `pytest tests/e2e/ -v`

## Environment Configuration

### Option 1: Environment Variables

```bash
export TEST_POSTGRESQL_URL="postgresql://user:pass@localhost:5432/ia_modules_test"
export TEST_MYSQL_URL="mysql://user:pass@localhost:3306/ia_modules_test"
export TEST_MSSQL_URL="mssql://user:pass@localhost:1433/ia_modules_test"
export TEST_REDIS_URL="redis://localhost:6379/0"
```

### Option 2: .env File

```bash
cd tests/
cp .env.example .env
# Edit .env with your database URLs
```

The test suite automatically loads `.env` files.

## Database Support Matrix

| Database | Unit Tests | Multi-Backend | Integration | Setup Required |
|----------|-----------|---------------|-------------|----------------|
| **SQLite** | ✅ Always | ✅ Always | ❌ | None (in-memory) |
| **PostgreSQL** | ❌ | ✅ Optional | ✅ Dedicated tests | Set TEST_POSTGRESQL_URL |
| **MySQL** | ❌ | ✅ Optional | 🚧 Coming soon | Set TEST_MYSQL_URL |
| **MSSQL** | ❌ | ✅ Optional | 🚧 Coming soon | Set TEST_MSSQL_URL |

## Test Fixtures

### Shared Fixtures (`conftest.py`)

**`db_config`** - Parameterized database configurations
```python
def test_something(db_config):
    db = DatabaseManager(db_config)
    # Test runs for each configured database
```

**`db_manager`** - Connected database managers with auto-cleanup
```python
def test_something(db_manager):
    # db_manager is already connected
    result = db_manager.execute("SELECT 1")
    # Cleanup happens automatically
```

### PostgreSQL Fixtures (`test_postgresql_integration.py`)

**`pg_config`** - PostgreSQL connection config
**`pg_db`** - Connected PostgreSQL with table cleanup
**`pg_db_clean`** - Fresh database with migrations table dropped

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

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
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest pytest-asyncio

      - name: Run tests
        env:
          TEST_POSTGRESQL_URL: postgresql://testuser:testpass@localhost:5432/ia_modules_test
        run: |
          pytest tests/ -v --tb=short
```

## Running Tests in Docker

```bash
# Start test databases
docker-compose -f docker-compose.test.yml up -d

# Run tests
export TEST_POSTGRESQL_URL="postgresql://testuser:testpass@localhost:5432/ia_modules_test"
pytest tests/ -v

# Cleanup
docker-compose -f docker-compose.test.yml down -v
```

## Test Coverage

```bash
# Run with coverage
pytest tests/ --cov=ia_modules --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Running Pipelines via CLI

You can also run pipelines directly using the command line interface:

```bash
# Basic usage
python tests/pipeline_runner.py <pipeline_file> --input '<json_input>'

# Examples:
# Simple pipeline with topic
python tests/pipeline_runner.py tests/pipelines/simple_pipeline/pipeline.json --input '{"topic": "artificial intelligence"}'

# Conditional pipeline with data
python tests/pipeline_runner.py tests/pipelines/conditional_pipeline/pipeline.json --input '{"raw_data": [{"quality_score": 0.95, "content": "high quality data"}]}'

# Parallel pipeline with dataset
python tests/pipeline_runner.py tests/pipelines/parallel_pipeline/pipeline.json --input '{"loaded_data": [{"id": 1, "value": 100}, {"id": 2, "value": 200}]}'
```

### CLI Options

```
Usage:
  python tests/pipeline_runner.py <pipeline_file> [options]
  python tests/pipeline_runner.py --slug <pipeline_slug> --db-url <url> [options]

Options:
  --input <json>        Input data as JSON string
  --output <path>       Output folder for results (creates timestamped subfolders)
  --db-url <url>        Database URL (required for --slug)

Examples with output:
  # Save to specific folder (creates timestamped subfolder inside)
  python tests/pipeline_runner.py tests/pipelines/simple_pipeline/pipeline.json --input '{"topic": "AI"}' --output ./results/
```

The CLI runner will:
- Execute the pipeline step by step
- Create a timestamped folder for each run
- Save both the pipeline result and log file in the timestamped folder
- Display execution logs and step-by-step results
- Show the complete pipeline execution flow

**Output behavior:**
- Creates a timestamped subfolder: `pipeline_run_YYYYMMDD_HHMMSS`
- Saves `pipeline_result.json` and `pipeline.log` in that folder
- If `--output` specified, creates timestamped folder inside that directory
- If no `--output` specified, creates timestamped folder in current directory

**Example output structure:**
```
./results/
├── pipeline_run_20250925_030716/
│   ├── pipeline_result.json
│   └── pipeline.log
└── pipeline_run_20250925_031203/
    ├── pipeline_result.json
    └── pipeline.log
```

## Writing Tests

### Testing with Multiple Databases

Use the parameterized `db_manager` fixture:

```python
def test_my_feature(db_manager):
    """This test runs on all configured databases"""
    db_manager.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
    # Test logic here
    # Works on SQLite, PostgreSQL, MySQL, MSSQL
```

### Testing PostgreSQL Specifically

Use the PostgreSQL fixtures:

```python
def test_uuid_type(pg_db):
    """PostgreSQL-only test"""
    db, test_id = pg_db

    db.execute(f"""
        CREATE TABLE test_{test_id} (
            id UUID PRIMARY KEY
        )
    """)

    # Test PostgreSQL-specific behavior
```

### Verifying Data Exists

Don't just check `result.success`:

```python
# ❌ BAD - Only checks success
result = db.execute("INSERT INTO users VALUES (:name)", {"name": "Alice"})
assert result.success

# ✅ GOOD - Verifies data actually exists
result = db.execute("INSERT INTO users VALUES (:name)", {"name": "Alice"})
assert result.success

row = db.fetch_one("SELECT * FROM users WHERE name = :name", {"name": "Alice"})
assert row is not None
assert row["name"] == "Alice"
```

## Current Status

**Total Tests:** 866
**Passing:** 853 (98.5%)
**Failing:** 0
**Errors:** 13 (Redis - expected, Redis not running)
**Skipped:** 18 (PostgreSQL integration - no database configured)

## Common Issues

### All PostgreSQL tests skipped

**Symptom:**
```
SKIPPED [18] PostgreSQL not configured (set TEST_POSTGRESQL_URL)
```

**Solution:** Set the environment variable:
```bash
export TEST_POSTGRESQL_URL="postgresql://user:pass@localhost:5432/ia_modules_test"
```

See [POSTGRESQL_TESTING_SETUP.md](POSTGRESQL_TESTING_SETUP.md) for full setup guide.

### Redis errors

**Symptom:**
```
ERROR tests/unit/test_redis_metric_storage.py::test_redis_storage_initialization
redis.exceptions.ConnectionError: Error 10061 connecting to localhost:6379
```

**This is expected** if Redis is not running. Redis tests are optional.

**Solution (optional):**
```bash
# Install and start Redis
docker run -d -p 6379:6379 redis:7
```

## Documentation

- **[MULTI_BACKEND_TESTING.md](MULTI_BACKEND_TESTING.md)** - Guide for running tests across multiple databases
- **[POSTGRESQL_TESTING_SETUP.md](POSTGRESQL_TESTING_SETUP.md)** - Complete PostgreSQL setup guide
- **[.env.example](.env.example)** - Example environment configuration

## Contributing

When adding new database features:

1. ✅ Add unit tests (SQLite, fast)
2. ✅ Add multi-backend tests (all databases)
3. ✅ Add PostgreSQL integration tests (if PostgreSQL-specific)
4. ✅ Verify tests pass with PostgreSQL connection
5. ✅ Update documentation

## Summary

- **Unit tests** - Fast, isolated, always run
- **Multi-backend tests** - Same test, multiple databases, parameterized
- **PostgreSQL integration** - Real database, catches production bugs
- **Integration tests** - Component interactions
- **E2E tests** - Full pipeline execution

**For development:** Run unit tests
**Before commit:** Run all tests with SQLite
**Before merge:** Run all tests with PostgreSQL
**In CI/CD:** Run all tests with all databases
