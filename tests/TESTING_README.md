# Testing Guide

## Quick Start

### Run All Tests (No External Services)
```bash
pytest tests/
```
- Unit tests run without external dependencies
- Tests requiring Redis/PostgreSQL/MySQL/MSSQL are skipped automatically

### Run Tests with Docker Compose (Recommended)

**Linux/Mac:**
```bash
cd tests
./docker-test-runner.sh
```

**Windows PowerShell:**
```powershell
cd tests
.\docker-test-runner.ps1
```

This will:
1. Start PostgreSQL, MySQL, MSSQL, and Redis in Docker containers
2. Wait for all services to be healthy
3. Run the full test suite with all integrations
4. Prompt to clean up containers after tests

### Manual Docker Compose

**Start services:**
```bash
cd tests
docker-compose -f docker-compose.test.yml up -d
```

**Check service health:**
```bash
docker-compose -f docker-compose.test.yml ps
```

**Run tests:**
```bash
export TEST_POSTGRESQL_URL="postgresql://testuser:testpass@localhost:5432/ia_modules_test"
export TEST_MYSQL_URL="mysql://testuser:testpass@localhost:3306/ia_modules_test"
export TEST_MSSQL_URL="mssql://testuser:TestPass123!@localhost:1433/ia_modules_test"
export TEST_REDIS_URL="redis://localhost:6379/0"

pytest tests/
```

**Stop services:**
```bash
docker-compose -f docker-compose.test.yml down -v
```

## Test Organization

### Unit Tests (`tests/unit/`)
- Fast, no external dependencies
- Mock external services
- Run in milliseconds
- Always pass in CI/CD

### Integration Tests (`tests/integration/`)
- Test interactions with databases
- Require Docker Compose services
- Skip gracefully if services unavailable
- Run in seconds

### E2E Tests (`tests/e2e/`)
- Full pipeline execution tests
- Test complete workflows
- Run in seconds to minutes

## Test Markers

Run specific test types:

```bash
# Skip slow tests
pytest -m "not slow"

# Only integration tests
pytest -m "integration"

# Only e2e tests
pytest -m "e2e"

# Skip tests requiring external services
pytest -m "not redis and not postgres"
```

## Coverage

**Generate coverage report:**
```bash
pytest tests/ --cov=ia_modules --cov-report=html
```

**View report:**
```bash
# Linux/Mac
open htmlcov/index.html

# Windows
start htmlcov/index.html
```

**Current coverage:** ~49%

## Test Services

| Service | Port | Container Name | Credentials |
|---------|------|----------------|-------------|
| PostgreSQL | 5432 | `ia_modules_test_postgres` | testuser/testpass |
| MySQL | 3306 | `ia_modules_test_mysql` | testuser/testpass |
| MSSQL | 1433 | `ia_modules_test_mssql` | sa/TestPass123! |
| Redis | 6379 | `ia_modules_test_redis` | (no password) |

## Troubleshooting

### Tests Skip with "Redis server not running"
**Solution:** Start Docker Compose services:
```bash
cd tests && docker-compose -f docker-compose.test.yml up -d
```

### Port Already in Use
**Solution:** Stop existing services:
```bash
# Check what's using the port
lsof -i :6379  # Linux/Mac
netstat -ano | findstr :6379  # Windows

# Stop existing Docker containers
docker ps
docker stop <container-id>
```

### Database Connection Errors
**Solution:** Check service health:
```bash
docker-compose -f tests/docker-compose.test.yml ps
docker-compose -f tests/docker-compose.test.yml logs postgres
```

### Deprecation Warnings
All deprecation warnings have been fixed:
- ✅ Redis `close()` → `aclose()`
- ✅ Database async operations properly awaited

## CI/CD

### GitHub Actions

Tests run automatically on:
- Push to `main` or `develop`
- Pull requests

**Test job:**
- Runs unit + integration + e2e tests
- Redis service included
- Coverage uploaded to Codecov

**Integration job:**
- Runs with PostgreSQL + Redis services
- Full database integration tests

View workflow: `.github/workflows/test.yml`

## Adding New Tests

### Unit Test Template
```python
"""Tests for new_module."""
import pytest
from new_module import function_to_test


class TestNewFeature:
    """Test new feature functionality."""

    def test_basic_case(self):
        """Test basic functionality."""
        result = function_to_test()
        assert result == expected

    def test_error_case(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            function_to_test(invalid_input)
```

### Integration Test Template
```python
"""Integration tests for new_module."""
import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_with_database(test_db):
    """Test with real database."""
    result = await function_with_db(test_db)
    assert result is not None
```

## Best Practices

1. **Fast Tests:** Unit tests should run in <1s
2. **Isolated Tests:** No shared state between tests
3. **Clear Names:** Test names describe what they test
4. **One Assert:** Focus each test on one behavior
5. **Mock External:** Mock external APIs and services
6. **Clean Up:** Use fixtures for setup/teardown
7. **Skip Gracefully:** Skip if dependencies unavailable

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py](https://coverage.readthedocs.io/)
- [Docker Compose](https://docs.docker.com/compose/)
