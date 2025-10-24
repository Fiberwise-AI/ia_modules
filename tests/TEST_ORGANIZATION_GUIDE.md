# Test Organization Guide

This guide explains how tests are organized in the ia_modules project and when to use unit tests vs integration tests.

## Directory Structure

```
tests/
├── unit/                          # Unit tests (no external dependencies)
│   ├── test_auth_*.py            # Authentication module tests
│   ├── test_pipeline_*.py        # Pipeline logic tests
│   ├── test_telemetry_*.py       # Telemetry tests
│   └── ...
├── integration/                   # Integration tests (require external services)
│   ├── test_database_mysql.py    # MySQL integration tests
│   ├── test_database_mssql.py    # MSSQL integration tests
│   ├── test_redis_metric_storage.py  # Redis integration tests
│   ├── test_observability_integration.py  # Observability stack tests
│   └── ...
├── e2e/                          # End-to-end tests (future)
├── performance/                   # Performance/benchmark tests (future)
└── conftest.py                   # Shared fixtures
```

## When to Use Unit Tests

**Location:** `tests/unit/`

**Characteristics:**
- **Fast**: Execute in milliseconds
- **Isolated**: No external dependencies (databases, Redis, APIs, file system)
- **Mocked**: External services are mocked using `unittest.mock` or `pytest-mock`
- **Pure Logic**: Test business logic, algorithms, data transformations
- **Repeatable**: Same input always produces same output

**Examples:**
```python
# tests/unit/test_pipeline_core.py
def test_step_execution():
    """Test step execution logic without external dependencies"""
    step = PipelineStep(name="test", function=lambda x: x * 2)
    result = step.execute(5)
    assert result == 10

# tests/unit/test_auth_security.py
def test_password_hashing():
    """Test password hashing is deterministic"""
    password = "test123"
    hash1 = hash_password(password)
    hash2 = hash_password(password)
    assert verify_password(password, hash1)
    assert verify_password(password, hash2)
```

**When to Use:**
- Testing pure functions
- Testing class methods with no I/O
- Testing data validation logic
- Testing configuration parsing
- Testing utility functions
- Testing error handling logic

## When to Use Integration Tests

**Location:** `tests/integration/`

**Characteristics:**
- **Slower**: May take seconds to execute
- **External Dependencies**: Require databases, Redis, message queues, APIs
- **Real Services**: Use actual Docker containers or test instances
- **Component Interaction**: Test how modules work together
- **Setup/Teardown**: Require database migrations, data seeding, cleanup

**Examples:**
```python
# tests/integration/test_database_mysql.py
@pytest.mark.mysql
@pytest.mark.integration
def test_mysql_transaction_rollback(mysql_db):
    """Test MySQL transaction rollback behavior"""
    mysql_db.execute("BEGIN TRANSACTION")
    mysql_db.execute("INSERT INTO test_table (name) VALUES (?)", ("test",))
    mysql_db.execute("ROLLBACK")

    result = mysql_db.execute("SELECT COUNT(*) as count FROM test_table")
    assert result[0]['count'] == 0

# tests/integration/test_redis_metric_storage.py
@pytest.mark.redis
@pytest.mark.integration
async def test_redis_metric_persistence(redis_storage):
    """Test metrics are actually stored in Redis"""
    await redis_storage.store_metric("test_metric", 42)
    value = await redis_storage.get_metric("test_metric")
    assert value == 42
```

**When to Use:**
- Testing database queries and transactions
- Testing cache behavior (Redis, Memcached)
- Testing API integrations
- Testing message queue interactions
- Testing file I/O operations
- Testing authentication flows with real tokens
- Testing metrics export to monitoring systems

## Test Markers

Use pytest markers to categorize and selectively run tests:

### Available Markers

```python
# Integration test markers
@pytest.mark.integration     # Requires external services
@pytest.mark.redis           # Requires Redis
@pytest.mark.postgres        # Requires PostgreSQL
@pytest.mark.mysql           # Requires MySQL
@pytest.mark.mssql           # Requires MSSQL
@pytest.mark.observability   # Requires observability stack

# Other markers
@pytest.mark.slow            # Takes >1 second
@pytest.mark.e2e             # End-to-end test
```

### Using Markers in Tests

```python
import pytest

@pytest.mark.mysql
@pytest.mark.integration
class TestMySQLOperations:
    """MySQL integration tests"""

    def test_insert(self, mysql_db):
        # Test implementation
        pass

@pytest.mark.redis
@pytest.mark.integration
async def test_redis_cache(redis_client):
    # Test implementation
    pass
```

### Running Tests by Marker

```bash
# Run only unit tests (no external dependencies)
pytest tests/unit/ -v

# Run all integration tests
pytest tests/integration/ -v -m integration

# Run only MySQL integration tests
pytest tests/integration/ -v -m mysql

# Run only observability tests
pytest tests/integration/ -v -m observability

# Run integration tests EXCEPT observability
pytest tests/integration/ -v -m "integration and not observability"

# Run fast tests only (exclude slow)
pytest tests/ -v -m "not slow"

# Run specific database tests
pytest tests/integration/ -v -m "mysql or postgres"
```

## Decision Tree: Unit vs Integration

```
                   Start
                     |
                     v
        Does the code interact with
        external services? (DB, Redis,
        API, file system, network)
                     |
          +----------+----------+
          |                     |
         YES                   NO
          |                     |
          v                     v
    Integration Test       Unit Test
          |                     |
          |                     v
          |              Mock external
          |              dependencies
          v
    Requires running
    Docker containers
    or test servers
```

## Redis Tests: Why They're Integration Tests

**Before (incorrect location):** `tests/unit/test_redis_metric_storage.py`

**After (correct location):** `tests/integration/test_redis_metric_storage.py`

**Reasoning:**
1. ✅ Requires external Redis server
2. ✅ Tests actual network communication
3. ✅ Tests data persistence across connections
4. ✅ Tests Redis-specific features (TTL, transactions)
5. ✅ Requires Docker container in CI/CD
6. ❌ Cannot run without Redis server
7. ❌ Not purely in-memory

Even though the tests use `skipif` to skip when Redis is unavailable, they still require a real Redis instance to run properly, making them integration tests.

## CI/CD Integration

### GitHub Actions Workflow

The `.github/workflows/test.yml` runs tests in separate jobs:

1. **Test Job**: Runs unit tests with minimal services (just Redis for backward compatibility)
2. **Integration Job**: Runs all integration tests with full service stack:
   - PostgreSQL
   - MySQL
   - MSSQL
   - Redis
   - Prometheus
   - OpenTelemetry Collector

### Local Development

```bash
# Quick feedback: Run only unit tests
pytest tests/unit/ -v

# Before committing: Run unit tests with coverage
pytest tests/unit/ -v --cov

# Before pushing: Run all tests (requires Docker)
docker-compose -f tests/docker-compose.test.yml up -d
pytest tests/ -v --cov
docker-compose -f tests/docker-compose.test.yml down
```

## Test Fixtures and Shared Code

### Shared Fixtures (conftest.py)

```python
# tests/conftest.py
import pytest

@pytest.fixture
def db_config():
    """Shared database configuration"""
    return get_database_configs()

@pytest.fixture
async def mysql_db(mysql_config):
    """MySQL database connection for integration tests"""
    db = DatabaseManager(mysql_config)
    db.connect()
    yield db
    db.disconnect()
```

### Database Configs

The `conftest.py` provides parameterized fixtures that automatically test against all available databases:

```python
@pytest.fixture(params=get_database_configs(), ids=lambda x: x[0])
async def db_manager(request):
    """Test against ALL available databases (SQLite, PostgreSQL, MySQL, MSSQL)"""
    config = request.param[1]
    db = DatabaseManager(config)
    db.connect()
    yield db
    db.disconnect()
```

## Best Practices

### DO ✅

1. **Keep unit tests fast** (< 100ms per test)
2. **Mock external dependencies in unit tests**
3. **Use real services in integration tests**
4. **Mark tests appropriately** with `@pytest.mark.*`
5. **Clean up after integration tests** (drop tables, flush cache)
6. **Use fixtures for setup/teardown**
7. **Test one thing per test function**
8. **Use descriptive test names** that explain what is being tested

### DON'T ❌

1. **Don't put integration tests in `tests/unit/`**
2. **Don't make unit tests depend on external services**
3. **Don't skip cleanup in integration tests** (causes flaky tests)
4. **Don't test implementation details** (test behavior)
5. **Don't have tests depend on each other** (order independence)
6. **Don't hardcode credentials or secrets in tests**
7. **Don't commit test databases** to version control

## Example Test File Structure

### Unit Test Example

```python
# tests/unit/test_pipeline_routing.py
"""
Unit tests for pipeline routing logic.
No external dependencies required.
"""
import pytest
from ia_modules.pipeline.routing import RouteCondition, Router

class TestRouteCondition:
    """Test routing condition evaluation"""

    def test_simple_condition(self):
        condition = RouteCondition("output.value > 10")
        assert condition.evaluate({"output": {"value": 15}}) == True
        assert condition.evaluate({"output": {"value": 5}}) == False

    def test_complex_condition(self):
        condition = RouteCondition("output.status == 'success' and output.count > 0")
        assert condition.evaluate({
            "output": {"status": "success", "count": 5}
        }) == True
```

### Integration Test Example

```python
# tests/integration/test_database_mysql.py
"""
Integration tests for MySQL database backend.
Requires MySQL server running (via Docker).
"""
import pytest
from ia_modules.database import DatabaseManager, ConnectionConfig, DatabaseType

@pytest.mark.mysql
@pytest.mark.integration
class TestMySQLTransactions:
    """Test MySQL transaction support"""

    def test_commit_transaction(self, mysql_db):
        """Test committing a transaction to real MySQL"""
        mysql_db.execute("CREATE TABLE test_table (id INT, name VARCHAR(100))")

        mysql_db.execute("START TRANSACTION")
        mysql_db.execute("INSERT INTO test_table VALUES (1, 'test')")
        mysql_db.execute("COMMIT")

        result = mysql_db.execute("SELECT COUNT(*) as count FROM test_table")
        assert result[0]['count'] == 1

        # Cleanup
        mysql_db.execute("DROP TABLE test_table")
```

## Troubleshooting

### "Tests are being skipped"

```bash
# Check which tests would run
pytest tests/integration/ --collect-only

# Check why tests are being skipped
pytest tests/integration/ -v -rs

# Run even if markers don't match
pytest tests/integration/test_database_mysql.py -v
```

### "Cannot connect to database/Redis"

```bash
# Make sure Docker services are running
docker-compose -f tests/docker-compose.test.yml ps

# Check service logs
docker-compose -f tests/docker-compose.test.yml logs mysql

# Restart services
docker-compose -f tests/docker-compose.test.yml restart
```

### "Tests are flaky"

Common causes:
1. Tests depend on each other (fix: make tests independent)
2. Leftover data from previous runs (fix: proper cleanup in fixtures)
3. Race conditions in async code (fix: use proper async fixtures)
4. Hardcoded timeouts too short (fix: use health checks)

## Summary

| Aspect | Unit Tests | Integration Tests |
|--------|-----------|------------------|
| **Location** | `tests/unit/` | `tests/integration/` |
| **Speed** | < 100ms | 1-10 seconds |
| **Dependencies** | None (mocked) | Real services |
| **Markers** | None needed | `@pytest.mark.integration`, `@pytest.mark.mysql`, etc. |
| **CI/CD** | Always run | Run in separate job |
| **Purpose** | Test logic | Test integration |
| **Isolation** | Complete | Partial |

**Golden Rule:** If it talks to a database, cache, API, file system, or network, it's an integration test.
