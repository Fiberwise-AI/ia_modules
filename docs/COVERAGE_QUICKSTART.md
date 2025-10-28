# Coverage Improvement Quick Start Guide

**Goal**: Start improving coverage TODAY! 🚀

This guide gives you concrete commands to run right now to begin improving test coverage.

---

## Step 1: Check Current Coverage (2 minutes)

```bash
cd ia_modules

# Run tests with coverage
python -m pytest tests/unit/ --cov=ia_modules --cov-report=term-missing --cov-report=html -q

# Open detailed HTML report
open htmlcov/index.html  # macOS
# OR
start htmlcov/index.html  # Windows
# OR
xdg-open htmlcov/index.html  # Linux
```

**What to look for**: Red/yellow highlighted lines in HTML report = missing coverage

---

## Step 2: Pick Your Priority Module (5 minutes)

Based on the coverage report, choose ONE module to improve:

### 🔴 **Quick Wins** (Easiest, 30 min - 2 hours)

| Module | Current | Target | Statements | Difficulty |
|--------|---------|--------|------------|------------|
| `auth/__init__.py` | 75% | 100% | 2 | ⭐ Easy |
| `checkpoint/__init__.py` | 69% | 100% | 4 | ⭐ Easy |
| `telemetry/metrics.py` | 97% | 100% | 3 | ⭐ Easy |
| `benchmarking/telemetry_bridge.py` | 29% | 70% | 18 | ⭐⭐ Medium |

### 🟡 **High Impact** (Medium effort, 4-8 hours)

| Module | Current | Target | Statements | Difficulty |
|--------|---------|--------|------------|------------|
| `checkpoint/redis.py` | 7% | 70% | 96 | ⭐⭐⭐ Hard |
| `checkpoint/sql.py` | 16% | 70% | 61 | ⭐⭐ Medium |
| `telemetry/exporters.py` | 62% | 80% | 63 | ⭐⭐⭐ Hard |
| `telemetry/integration.py` | 53% | 75% | 61 | ⭐⭐⭐ Hard |

### 🔴 **Critical** (High effort, 1-3 days)

| Module | Current | Target | Statements | Difficulty |
|--------|---------|--------|------------|------------|
| `reliability/mode_enforcer.py` | 0% | 75% | 97 | ⭐⭐⭐ Hard |
| `reliability/replay.py` | 0% | 75% | 108 | ⭐⭐⭐ Hard |
| `reliability/slo_tracker.py` | 0% | 75% | 100 | ⭐⭐⭐ Hard |
| `telemetry/opentelemetry_exporter.py` | 0% | 70% | 130 | ⭐⭐⭐⭐ Very Hard |

---

## Step 3: Example - Let's Fix `auth/__init__.py` (30 minutes)

### 3.1 Check What's Missing

```bash
# Run coverage for just auth module
python -m pytest tests/unit/test_auth*.py --cov=ia_modules.auth --cov-report=term-missing -v

# Output shows:
# auth/__init__.py     8      2      0      0  75.00%   25-27
```

Lines 25-27 are missing!

### 3.2 Look at the Source

```bash
# View the missing lines
cat -n ia_modules/auth/__init__.py | sed -n '20,30p'
```

### 3.3 Find Existing Test File

```bash
# Check if test file exists
ls tests/unit/test_auth*.py

# If it exists, edit it
# If not, create it
```

### 3.4 Add Missing Tests

```python
# In tests/unit/test_auth.py or tests/unit/test_auth_module.py

def test_auth_module_imports():
    """Test that all auth components can be imported"""
    from ia_modules.auth import (
        AuthMiddleware,
        SessionManager,
        # ... add whatever is on lines 25-27
    )
    assert AuthMiddleware is not None
    assert SessionManager is not None

def test_auth_module_has_version():
    """Test auth module has version or attributes"""
    import ia_modules.auth as auth
    # Add assertions for whatever lines 25-27 define
```

### 3.5 Verify Improvement

```bash
# Run tests again
python -m pytest tests/unit/test_auth*.py --cov=ia_modules.auth --cov-report=term-missing -v

# Should now show:
# auth/__init__.py     8      0      0      0  100.00%
```

✅ **Success!** You just improved coverage by +0.02%!

---

## Step 4: Example - Fix `telemetry/metrics.py` (1 hour)

This module is already at 97.65%, just need 3 statements!

```bash
# Check what's missing
python -m pytest tests/unit/test_telemetry_metrics.py --cov=ia_modules.telemetry.metrics --cov-report=term-missing -v

# Output:
# telemetry/metrics.py   183      3     30      2  97.65%   36-37, 266->264, 369
```

### Missing Lines:
- **Lines 36-37**: Probably an edge case (null check, exception handling)
- **Branch 266→264**: Conditional branch not covered
- **Line 369**: Error handler or cleanup code

### Add Tests:

```python
# In tests/unit/test_telemetry_metrics.py

def test_metric_with_none_value():
    """Test metric handles None value (covers lines 36-37)"""
    metric = Metric(name="test", value=None)
    # Assert behavior

def test_metric_label_filtering_edge_case():
    """Test branch at line 266"""
    metric = Metric(name="test", value=100, labels={"key": ""})
    filtered = metric.filter_labels()
    # This might trigger the 266->264 branch

def test_metric_cleanup_on_error():
    """Test line 369 - likely cleanup or error path"""
    with pytest.raises(ValueError):
        metric = Metric(name="", value=100)  # Invalid name
    # Verify cleanup happened
```

Run again:
```bash
python -m pytest tests/unit/test_telemetry_metrics.py --cov=ia_modules.telemetry.metrics --cov-report=term-missing -v
```

---

## Step 5: Tackle a Harder Module - `checkpoint/redis.py` (4 hours)

Currently at **7.79%** (12/108 covered), let's get to **70%** (75/108)

### 5.1 Read the Module

```bash
# Understand what it does
cat ia_modules/checkpoint/redis.py | head -50
```

### 5.2 Check Existing Tests

```bash
# Find existing checkpoint tests
ls tests/unit/test_checkpoint*.py
ls tests/integration/test_*checkpoint*.py

# Run them to see current coverage
python -m pytest tests/unit/test_checkpoint_redis.py --cov=ia_modules.checkpoint.redis --cov-report=annotate -v
```

### 5.3 Create Test Structure

```python
# In tests/unit/test_checkpoint_redis.py

import pytest
from unittest.mock import Mock, patch, MagicMock
from ia_modules.checkpoint.redis import RedisCheckpointer

class TestRedisCheckpointer:
    """Test RedisCheckpointer with mocked Redis"""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client"""
        with patch('redis.Redis') as mock:
            mock_client = MagicMock()
            mock.return_value = mock_client
            yield mock_client

    def test_init(self, mock_redis):
        """Test RedisCheckpointer initialization"""
        checkpointer = RedisCheckpointer(host='localhost', port=6379)
        assert checkpointer.redis_client is not None

    def test_save_checkpoint(self, mock_redis):
        """Test saving a checkpoint"""
        checkpointer = RedisCheckpointer()
        data = {"key": "value", "number": 42}

        checkpoint_id = checkpointer.save(data, name="test_checkpoint")

        assert checkpoint_id is not None
        mock_redis.set.assert_called()

    def test_load_checkpoint(self, mock_redis):
        """Test loading a checkpoint"""
        checkpointer = RedisCheckpointer()

        # Mock Redis to return saved data
        mock_redis.get.return_value = b'{"key": "value", "number": 42}'

        data = checkpointer.load("checkpoint_123")

        assert data["key"] == "value"
        assert data["number"] == 42

    def test_list_checkpoints(self, mock_redis):
        """Test listing all checkpoints"""
        checkpointer = RedisCheckpointer()

        # Mock Redis to return keys
        mock_redis.keys.return_value = [b'ckpt:id1', b'ckpt:id2']

        checkpoints = checkpointer.list()

        assert len(checkpoints) == 2

    def test_delete_checkpoint(self, mock_redis):
        """Test deleting a checkpoint"""
        checkpointer = RedisCheckpointer()

        checkpointer.delete("checkpoint_123")

        mock_redis.delete.assert_called_with("checkpoint_123")

    def test_connection_error_handling(self, mock_redis):
        """Test handling Redis connection errors"""
        mock_redis.set.side_effect = ConnectionError("Redis unavailable")

        checkpointer = RedisCheckpointer()

        with pytest.raises(ConnectionError):
            checkpointer.save({"data": "test"})

    # Add 15-20 more tests to reach 70% coverage
    # Focus on: serialization, deserialization, error cases, edge cases
```

### 5.4 Run and Iterate

```bash
# Run tests
python -m pytest tests/unit/test_checkpoint_redis.py -v

# Check coverage
python -m pytest tests/unit/test_checkpoint_redis.py --cov=ia_modules.checkpoint.redis --cov-report=term-missing

# Keep adding tests until you hit 70%!
```

---

## Step 6: Track Your Progress

### Daily Coverage Check
```bash
# Run this every morning
python -m pytest tests/unit/ --cov=ia_modules --cov-report=term | grep "TOTAL"

# Track in a file
echo "$(date): $(pytest tests/ --cov=ia_modules -q | grep TOTAL)" >> coverage_log.txt
```

### Generate Trend Graph
```bash
# Install coverage badge generator
pip install coverage-badge

# Generate badge
coverage-badge -o coverage.svg -f

# View coverage over time
cat coverage_log.txt
```

---

## Pro Tips 💡

### 1. **Start Small**
Don't try to fix all modules at once. Pick ONE module, get it to your target, then move to the next.

### 2. **Use Coverage HTML Report**
The HTML report shows exactly which lines are missing. It's your roadmap!

```bash
pytest tests/ --cov=ia_modules --cov-report=html
open htmlcov/index.html
```

### 3. **Mock External Dependencies**
For Redis, databases, HTTP clients - always mock in unit tests:

```python
from unittest.mock import Mock, patch

@patch('redis.Redis')
def test_with_mock_redis(mock_redis):
    # Your test here
    pass
```

### 4. **Test Edge Cases**
- Empty inputs
- None values
- Invalid data
- Exception paths
- Boundary conditions

### 5. **Run Specific Tests Quickly**
```bash
# Test just one file
pytest tests/unit/test_auth.py -v

# Test just one class
pytest tests/unit/test_auth.py::TestAuthMiddleware -v

# Test just one function
pytest tests/unit/test_auth.py::TestAuthMiddleware::test_init -v
```

### 6. **Watch Mode for TDD**
```bash
# Install pytest-watch
pip install pytest-watch

# Auto-run tests on file changes
ptw tests/unit/test_checkpoint_redis.py -- --cov=ia_modules.checkpoint.redis
```

---

## Measuring Success

### After 1 Hour:
- ✅ Fixed 1-2 small modules (auth, checkpoint init)
- ✅ Coverage: 65% → 65.5% (+0.5%)

### After 1 Day:
- ✅ Fixed 1 medium module (checkpoint/sql or benchmarking/telemetry_bridge)
- ✅ Coverage: 65% → 68% (+3%)

### After 1 Week:
- ✅ Fixed 2-3 hard modules (reliability modules)
- ✅ Coverage: 65% → 73% (+8%)

### After 1 Month:
- ✅ Fixed all high-priority modules
- ✅ Coverage: 65% → 85% (+20%)

---

## Quick Reference Commands

```bash
# Check overall coverage
pytest tests/ --cov=ia_modules --cov-report=term-missing

# Check one module
pytest tests/unit/test_X.py --cov=ia_modules.X --cov-report=term-missing

# Generate HTML report
pytest tests/ --cov=ia_modules --cov-report=html && open htmlcov/index.html

# Check coverage delta (compare with main branch)
pytest tests/ --cov=ia_modules --cov-report=json
# Compare coverage.json with previous run

# Run only fast tests
pytest tests/unit/ -v

# Run with verbose output
pytest tests/unit/ -vv

# Run and stop on first failure
pytest tests/unit/ -x

# Show print statements
pytest tests/unit/ -s
```

---

## Get Started NOW! 🚀

```bash
# 1. Check current coverage
cd ia_modules
pytest tests/unit/ --cov=ia_modules --cov-report=html -q

# 2. Open the report
open htmlcov/index.html

# 3. Pick a module with low coverage
# 4. Add tests!
# 5. Run coverage again
# 6. Celebrate! 🎉
```

---

**Remember**: Every test you write makes the codebase more reliable. You've got this! 💪

**Next Steps**: See [COVERAGE_IMPROVEMENT_PLAN.md](COVERAGE_IMPROVEMENT_PLAN.md) for the full 6-week roadmap.
