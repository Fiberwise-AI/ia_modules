# IA Modules - Testing Guide

## Overview

This comprehensive testing guide covers all aspects of testing the IA Modules framework, including unit tests, integration tests, pipeline testing, and end-to-end testing scenarios. The framework provides robust testing utilities and example pipelines for thorough validation.

## Table of Contents

- [Test Architecture](#test-architecture)
- [Test Categories](#test-categories)
- [Running Tests](#running-tests)
- [Unit Testing](#unit-testing)
- [Integration Testing](#integration-testing)
- [Pipeline Testing](#pipeline-testing)
- [Test Utilities](#test-utilities)
- [Example Test Pipelines](#example-test-pipelines)
- [Mocking and Fixtures](#mocking-and-fixtures)
- [Performance Testing](#performance-testing)
- [Best Practices](#best-practices)

## Test Architecture

The testing framework is organized into several layers:

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_core.py        # Core pipeline component tests
│   ├── test_services.py    # Service registry tests
│   ├── test_runner.py      # Pipeline runner tests
│   ├── test_importer.py    # Pipeline importer tests
│   └── test_*.py           # Other unit tests
├── integration/             # Integration tests
│   ├── test_pipeline_integration.py
│   ├── test_database_integration.py
│   └── test_*.py
├── e2e/                    # End-to-end tests
│   └── test_complete_workflows.py
├── pipelines/              # Test pipeline configurations
│   ├── simple_pipeline/    # Basic test pipeline
│   ├── conditional_pipeline/  # Conditional flow tests
│   ├── parallel_pipeline/  # Parallel execution tests
│   └── agent_pipeline/     # AI agent tests
├── conftest.py             # Global test configuration
└── pipeline_runner.py      # Standalone pipeline test runner
```

## Test Categories

### 1. Unit Tests (`tests/unit/`)

Test individual components in isolation with mocked dependencies.

**Coverage Areas:**
- Pipeline core components (Step, Pipeline, TemplateParameterResolver)
- Service registry functionality
- Database interfaces and managers
- Authentication models
- Utility functions

### 2. Integration Tests (`tests/integration/`)

Test component interactions and system integration.

**Coverage Areas:**
- Pipeline execution with real database
- Service injection and dependency resolution
- Template parameter resolution in context
- Flow control and conditional routing
- Migration system with database

### 3. End-to-End Tests (`tests/e2e/`)

Test complete workflows from JSON configuration to final output.

**Coverage Areas:**
- Full pipeline execution workflows
- Real database operations
- File system interactions
- WebSocket communications (if applicable)

### 4. Pipeline Tests (`tests/pipelines/`)

Validate pipeline configurations and step implementations.

**Coverage Areas:**
- JSON configuration validation
- Step implementation correctness
- Flow definition validation
- Parameter template resolution

## Running Tests

### Prerequisites

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-mock pytest-cov

# Ensure ia_modules is in Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/ia_modules"
```

### Running All Tests

```bash
# Run all tests with verbose output
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=ia_modules --cov-report=html

# Run specific test categories
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v
python -m pytest tests/e2e/ -v
```

### Running Specific Tests

```bash
# Run specific test file
python -m pytest tests/unit/test_core.py -v

# Run specific test function
python -m pytest tests/unit/test_core.py::test_step_execution -v

# Run tests with pattern matching
python -m pytest -k "test_pipeline" -v
```

### Platform-Specific Considerations

#### Windows Testing

```bash
# Use Windows-specific event loop policy
set PYTHONASYNCIODEBUG=1
python -m pytest tests/ -v
```

#### Configuration in `conftest.py`

```python
import sys
import asyncio

# Set up event loop policy for Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
```

## Unit Testing

### Core Component Tests

#### Step Testing

```python
# tests/unit/test_core.py
import pytest
from ia_modules.pipeline.core import Step
from ia_modules.pipeline.services import ServiceRegistry

class MockStep(Step):
    async def run(self, data: dict) -> dict:
        return {"test_result": "success", "input_data": data}

def test_step_execution():
    """Test basic step execution"""
    step = MockStep("test_step", {"param": "value"})

    # Mock services
    services = ServiceRegistry()
    step.set_services(services)

    # Test properties
    assert step.name == "test_step"
    assert step.config == {"param": "value"}

    # Test service access
    assert step.get_db() is None  # No database service registered

@pytest.mark.asyncio
async def test_step_run():
    """Test step run method with logging"""
    step = MockStep("test_step", {})

    result = await step.run({"input": "test_data"})

    # Verify result structure
    assert "test_result" in result
    assert result["test_result"] == "success"
    assert result["input_data"]["input"] == "test_data"
```

#### Template Parameter Resolution Testing

```python
def test_template_parameter_resolver():
    """Test template parameter resolution"""
    from ia_modules.pipeline.core import TemplateParameterResolver

    context = {
        "pipeline_input": {"business_type": "retail"},
        "steps": {"geocoder": {"result": {"city": "New York"}}},
        "parameters": {"custom_value": "test_value"}
    }

    config = {
        "url": "{pipeline_input.business_type}",
        "city": "{steps.geocoder.result.city}",
        "value": "{parameters.custom_value}",
        "nested": {
            "param": "{pipeline_input.business_type}"
        }
    }

    resolved = TemplateParameterResolver.resolve_parameters(config, context)

    assert resolved["url"] == "retail"
    assert resolved["city"] == "New York"
    assert resolved["value"] == "test_value"
    assert resolved["nested"]["param"] == "retail"

def test_template_parameter_extraction():
    """Test template parameter extraction"""
    config = {
        "field1": "{pipeline_input.data}",
        "field2": "{steps.step1.result}",
        "field3": "no template here"
    }

    parameters = TemplateParameterResolver.extract_template_parameters(config)

    assert "pipeline_input.data" in parameters
    assert "steps.step1.result" in parameters
    assert len(parameters) == 2
```

### Service Registry Testing

```python
# tests/unit/test_services.py
import pytest
from ia_modules.pipeline.services import ServiceRegistry, CentralLoggingService

def test_service_registry():
    """Test service registry functionality"""
    registry = ServiceRegistry()

    # Test service registration
    mock_service = {"type": "mock"}
    registry.register("mock_service", mock_service)

    # Test service retrieval
    assert registry.get("mock_service") == mock_service
    assert registry.has("mock_service") is True
    assert registry.get("nonexistent") is None
    assert registry.has("nonexistent") is False

def test_central_logging_service():
    """Test central logging service"""
    logger = CentralLoggingService()

    # Test execution ID setting
    logger.set_execution_id("test_exec_123")
    assert logger.current_execution_id == "test_exec_123"

    # Test logging methods
    logger.info("Test info message", "test_step")
    logger.error("Test error message", "test_step")
    logger.warning("Test warning message")

    assert len(logger.execution_logs) == 3

    # Verify log entries
    info_log = logger.execution_logs[0]
    assert info_log.level == "INFO"
    assert info_log.message == "Test info message"
    assert info_log.step_name == "test_step"

    # Test log clearing
    logger.clear_logs()
    assert len(logger.execution_logs) == 0
    assert logger.current_execution_id is None
```

### Database Testing

```python
# tests/unit/test_database.py
import pytest
from unittest.mock import Mock, AsyncMock
from ia_modules.database.interfaces import QueryResult, create_query_result, create_error_result

def test_query_result():
    """Test QueryResult functionality"""
    data = [{"id": 1, "name": "John"}, {"id": 2, "name": "Jane"}]
    result = QueryResult(success=True, data=data, row_count=2)

    # Test basic properties
    assert result.success is True
    assert result.row_count == 2

    # Test first row access
    first_row = result.get_first_row()
    assert first_row["name"] == "John"

    # Test column value extraction
    names = result.get_column_values("name")
    assert names == ["John", "Jane"]

def test_query_result_utilities():
    """Test query result utility functions"""
    # Test success result creation
    success_result = create_query_result(data=[{"id": 1}])
    assert success_result.success is True
    assert success_result.row_count == 1

    # Test error result creation
    error_result = create_error_result("Test error message")
    assert error_result.success is False
    assert error_result.error_message == "Test error message"
    assert error_result.row_count == 0
```

## Integration Testing

### Pipeline Integration Tests

```python
# tests/integration/test_pipeline_integration.py
import pytest
import tempfile
import json
from pathlib import Path

from ia_modules.pipeline.runner import run_pipeline_from_json
from ia_modules.pipeline.services import ServiceRegistry
from ia_modules.database.manager import DatabaseManager

@pytest.mark.asyncio
async def test_complete_pipeline_execution():
    """Test complete pipeline execution with database"""

    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
        db_url = f"sqlite:///{tmp_db.name}"

    # Setup database
    db_manager = DatabaseManager(db_url)
    await db_manager.initialize()

    # Setup services
    services = ServiceRegistry()
    services.register('database', db_manager)

    # Create test pipeline configuration
    pipeline_config = {
        "name": "Test Integration Pipeline",
        "version": "1.0",
        "steps": [
            {
                "id": "test_step",
                "step_class": "TestIntegrationStep",
                "module": "tests.fixtures.test_steps"
            }
        ],
        "flow": {
            "start_at": "test_step",
            "paths": []
        }
    }

    # Write pipeline to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_pipeline:
        json.dump(pipeline_config, tmp_pipeline)
        pipeline_file = tmp_pipeline.name

    # Execute pipeline
    result = await run_pipeline_from_json(
        pipeline_file=pipeline_file,
        input_data={"test_input": "integration_test"},
        services=services
    )

    # Verify results
    assert "test_step" in result
    assert result["test_step"]["status"] == "success"

    # Cleanup
    Path(pipeline_file).unlink()
    Path(tmp_db.name).unlink()

@pytest.mark.asyncio
async def test_conditional_flow_integration():
    """Test conditional flow execution"""
    pipeline_file = "tests/pipelines/conditional_pipeline/pipeline.json"

    # Test high quality path
    result_high = await run_pipeline_from_json(
        pipeline_file=pipeline_file,
        input_data={"quality_score": 0.9, "data": [{"test": "data"}]}
    )

    # Verify high quality processing occurred
    assert "high_quality_processor" in result_high

    # Test low quality path
    result_low = await run_pipeline_from_json(
        pipeline_file=pipeline_file,
        input_data={"quality_score": 0.3, "data": [{"test": "data"}]}
    )

    # Verify low quality processing occurred
    assert "low_quality_processor" in result_low
```

### Database Integration Tests

```python
# tests/integration/test_database_integration.py
import pytest
import tempfile
from pathlib import Path

from ia_modules.database.manager import DatabaseManager
from ia_modules.database.migrations import MigrationRunner

@pytest.mark.asyncio
async def test_database_initialization():
    """Test database initialization with migrations"""

    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
        db_url = f"sqlite:///{tmp_db.name}"

    # Initialize database
    db_manager = DatabaseManager(db_url)
    success = await db_manager.initialize(apply_schema=True)

    assert success is True
    assert db_manager._connection is not None

    # Verify tables were created
    tables_result = db_manager.fetch_all(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )

    table_names = [row['name'] for row in tables_result]
    assert 'ia_migrations' in table_names

    # Cleanup
    db_manager.disconnect()
    Path(tmp_db.name).unlink()

@pytest.mark.asyncio
async def test_migration_system():
    """Test migration system functionality"""

    # Create temporary database and migration directory
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
        db_url = f"sqlite:///{tmp_db.name}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        migration_dir = Path(tmp_dir)

        # Create test migration
        migration_content = """
        CREATE TABLE test_table (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        );
        """

        migration_file = migration_dir / "V001__create_test_table.sql"
        migration_file.write_text(migration_content)

        # Setup database
        db_manager = DatabaseManager(db_url)
        await db_manager.initialize(apply_schema=False)

        # Run migrations
        migration_runner = MigrationRunner(
            database=db_manager,
            migration_path=migration_dir,
            migration_type="test"
        )

        success = await migration_runner.run_pending_migrations()
        assert success is True

        # Verify migration was applied
        applied_migrations = await migration_runner.get_applied_migrations()
        assert len(applied_migrations) == 1
        assert applied_migrations[0].version == "V001__create_test_table"

        # Verify table was created
        table_exists = await db_manager.table_exists("test_table")
        assert table_exists is True

        # Cleanup
        db_manager.disconnect()
        Path(tmp_db.name).unlink()
```

## Pipeline Testing

### Test Pipeline Runner

The framework includes a standalone pipeline test runner for development and debugging:

```python
# tests/pipeline_runner.py
def run_pipeline_test(
    pipeline_file: str,
    input_data: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Run a pipeline test with minimal setup"""

    # Load JSON configuration
    json_path = Path(pipeline_file)
    if not json_path.exists():
        raise FileNotFoundError(f"Pipeline configuration file not found: {pipeline_file}")

    with open(json_path, 'r', encoding='utf-8') as f:
        pipeline_config = json.load(f)

    # Provide default input data if none given
    if input_data is None:
        input_data = {
            "input_data": [
                {"name": "example1", "value": 10},
                {"name": "example2", "value": 20}
            ]
        }

    # Create minimal services registry
    services = ServiceRegistry()

    # Create and run pipeline
    pipeline = create_pipeline_from_json(pipeline_config, services)
    result = asyncio.run(pipeline.run(input_data))

    return result
```

**Usage:**

```bash
# Run test pipeline
python tests/pipeline_runner.py tests/pipelines/simple_pipeline/pipeline.json

# Run with custom input
python tests/pipeline_runner.py tests/pipelines/simple_pipeline/pipeline.json --input '{"topic": "test"}'
```

## Test Utilities

### Mock Database Manager

```python
# tests/fixtures/mock_database.py
class MockDatabaseManager:
    """Mock database manager for testing"""

    def __init__(self):
        self.data = {}
        self.queries_executed = []

    def fetch_all(self, query: str, params: tuple = None) -> List[Dict]:
        self.queries_executed.append((query, params))
        return self.data.get('fetch_all_result', [])

    def fetch_one(self, query: str, params: tuple = None) -> Optional[Dict]:
        self.queries_executed.append((query, params))
        return self.data.get('fetch_one_result')

    def execute(self, query: str, params: tuple = None):
        self.queries_executed.append((query, params))
        return self

    def set_mock_data(self, key: str, value: Any):
        """Set mock data for responses"""
        self.data[key] = value
```

### Test Step Implementations

```python
# tests/fixtures/test_steps.py
from ia_modules.pipeline.core import Step
from typing import Dict, Any

class TestDataStep(Step):
    """Test step that generates data"""

    async def run(self, data: Dict[str, Any]) -> Any:
        return {
            "generated_data": [1, 2, 3, 4, 5],
            "timestamp": "2024-01-01T00:00:00Z",
            "status": "success"
        }

class TestDatabaseStep(Step):
    """Test step that uses database service"""

    async def run(self, data: Dict[str, Any]) -> Any:
        db = self.get_db()
        if not db:
            return {"error": "No database service"}

        # Perform mock database operation
        result = db.fetch_all("SELECT * FROM test_table")

        return {
            "database_result": result,
            "query_count": len(db.queries_executed) if hasattr(db, 'queries_executed') else 0
        }

class TestConditionalStep(Step):
    """Test step that returns different results for conditional flow testing"""

    async def run(self, data: Dict[str, Any]) -> Any:
        quality_score = data.get("quality_score", 0.5)

        return {
            "quality_score": quality_score,
            "quality_category": "high" if quality_score > 0.8 else "low",
            "processing_recommended": quality_score > 0.5
        }
```

## Example Test Pipelines

### Simple Pipeline

```json
{
    "name": "Simple Data Processing Pipeline",
    "description": "A simple pipeline for basic data processing",
    "version": "1.0.0",
    "steps": [
        {
            "id": "step1",
            "name": "Data Cleaner",
            "step_class": "DataCleanerStep",
            "module": "tests.pipelines.simple_pipeline.steps.data_cleaner"
        },
        {
            "id": "step2",
            "name": "Data Analyzer",
            "step_class": "DataAnalyzerStep",
            "module": "tests.pipelines.simple_pipeline.steps.data_analyzer"
        }
    ],
    "flow": {
        "start_at": "step1",
        "paths": [
            {
                "from_step": "step1",
                "to_step": "step2",
                "condition": {"type": "always"}
            }
        ]
    }
}
```

### Conditional Pipeline

```json
{
    "name": "Conditional Processing Pipeline",
    "description": "Pipeline with conditional flow based on data quality",
    "version": "1.0.0",
    "steps": [
        {
            "id": "quality_checker",
            "step_class": "QualityCheckerStep",
            "module": "tests.pipelines.conditional_pipeline.steps.quality_checker"
        },
        {
            "id": "high_quality_processor",
            "step_class": "HighQualityProcessorStep",
            "module": "tests.pipelines.conditional_pipeline.steps.high_quality_processor"
        },
        {
            "id": "low_quality_processor",
            "step_class": "LowQualityProcessorStep",
            "module": "tests.pipelines.conditional_pipeline.steps.low_quality_processor"
        }
    ],
    "flow": {
        "start_at": "quality_checker",
        "paths": [
            {
                "from_step": "quality_checker",
                "to_step": "high_quality_processor",
                "condition": {
                    "type": "expression",
                    "config": {
                        "source": "result.quality_score",
                        "operator": "greater_than",
                        "value": 0.8
                    }
                }
            },
            {
                "from_step": "quality_checker",
                "to_step": "low_quality_processor",
                "condition": {
                    "type": "expression",
                    "config": {
                        "source": "result.quality_score",
                        "operator": "less_than_or_equal",
                        "value": 0.8
                    }
                }
            }
        ]
    }
}
```

## Mocking and Fixtures

### Pytest Fixtures

```python
# conftest.py
import pytest
from ia_modules.pipeline.services import ServiceRegistry
from tests.fixtures.mock_database import MockDatabaseManager

@pytest.fixture
def service_registry():
    """Provide a clean service registry for each test"""
    return ServiceRegistry()

@pytest.fixture
def mock_database():
    """Provide a mock database manager"""
    return MockDatabaseManager()

@pytest.fixture
def service_registry_with_db(service_registry, mock_database):
    """Provide service registry with mock database"""
    service_registry.register('database', mock_database)
    return service_registry

@pytest.fixture
def sample_pipeline_config():
    """Provide sample pipeline configuration"""
    return {
        "name": "Test Pipeline",
        "version": "1.0",
        "steps": [
            {
                "id": "test_step",
                "step_class": "TestStep",
                "module": "tests.fixtures.test_steps"
            }
        ],
        "flow": {
            "start_at": "test_step",
            "paths": []
        }
    }
```

### Using Fixtures in Tests

```python
def test_pipeline_with_fixtures(service_registry_with_db, sample_pipeline_config):
    """Test using pytest fixtures"""
    from ia_modules.pipeline.runner import create_pipeline_from_json

    # Create pipeline
    pipeline = create_pipeline_from_json(sample_pipeline_config, service_registry_with_db)

    assert len(pipeline.steps) == 1
    assert pipeline.steps[0].name == "test_step"

    # Test database service injection
    step = pipeline.steps[0]
    db = step.get_db()
    assert db is not None
    assert hasattr(db, 'queries_executed')
```

## Performance Testing

### Load Testing

```python
# tests/performance/test_performance.py
import asyncio
import time
import pytest

@pytest.mark.asyncio
async def test_pipeline_performance():
    """Test pipeline execution performance"""

    # Create performance test pipeline
    pipeline_config = {
        "name": "Performance Test Pipeline",
        "steps": [
            {
                "id": "perf_step",
                "step_class": "PerformanceTestStep",
                "module": "tests.fixtures.performance_steps"
            }
        ],
        "flow": {
            "start_at": "perf_step",
            "paths": []
        }
    }

    # Measure execution time
    start_time = time.time()

    pipeline = create_pipeline_from_json(pipeline_config)
    result = await pipeline.run({"test_data": list(range(1000))})

    end_time = time.time()
    execution_time = end_time - start_time

    # Assert performance criteria
    assert execution_time < 1.0  # Should complete within 1 second
    assert "perf_step" in result

@pytest.mark.asyncio
async def test_concurrent_pipelines():
    """Test concurrent pipeline execution"""

    async def run_single_pipeline(pipeline_id: int):
        pipeline_config = {
            "name": f"Concurrent Pipeline {pipeline_id}",
            "steps": [{
                "id": f"step_{pipeline_id}",
                "step_class": "ConcurrentTestStep",
                "module": "tests.fixtures.performance_steps"
            }],
            "flow": {
                "start_at": f"step_{pipeline_id}",
                "paths": []
            }
        }

        pipeline = create_pipeline_from_json(pipeline_config)
        return await pipeline.run({"pipeline_id": pipeline_id})

    # Run 10 pipelines concurrently
    start_time = time.time()

    tasks = [run_single_pipeline(i) for i in range(10)]
    results = await asyncio.gather(*tasks)

    end_time = time.time()

    # Verify all pipelines completed
    assert len(results) == 10

    # Concurrent execution should be faster than sequential
    assert end_time - start_time < 5.0  # Should complete within 5 seconds
```

## Best Practices

### Test Organization

1. **Separate Concerns**: Keep unit, integration, and e2e tests in separate directories
2. **Use Fixtures**: Create reusable fixtures for common test setup
3. **Mock External Dependencies**: Mock databases, APIs, and file systems in unit tests
4. **Test Real Integrations**: Use real services in integration tests
5. **Cover Edge Cases**: Test error conditions and boundary cases

### Test Data Management

```python
# tests/fixtures/test_data.py
def get_sample_input_data():
    """Get standardized test input data"""
    return {
        "topic": "artificial intelligence",
        "data": [
            {"id": 1, "name": "item1", "value": 10, "category": "A"},
            {"id": 2, "name": "item2", "value": 20, "category": "B"},
            {"id": 3, "name": "item3", "value": 30, "category": "A"}
        ],
        "metadata": {
            "source": "test_data",
            "timestamp": "2024-01-01T00:00:00Z"
        }
    }

def get_expected_output_data():
    """Get expected output data for validation"""
    return {
        "processed_count": 3,
        "categories": ["A", "B"],
        "total_value": 60,
        "status": "success"
    }
```

### Error Testing

```python
@pytest.mark.asyncio
async def test_step_error_handling():
    """Test step error handling"""

    class ErrorStep(Step):
        async def run(self, data):
            raise ValueError("Intentional test error")

    step = ErrorStep("error_step", {})

    # Test that error is properly propagated
    with pytest.raises(ValueError, match="Intentional test error"):
        await step.run({"test": "data"})

@pytest.mark.asyncio
async def test_pipeline_error_recovery():
    """Test pipeline behavior with step errors"""

    pipeline_config = {
        "name": "Error Test Pipeline",
        "steps": [
            {
                "id": "error_step",
                "step_class": "ErrorStep",
                "module": "tests.fixtures.error_steps"
            }
        ],
        "flow": {
            "start_at": "error_step",
            "paths": []
        }
    }

    pipeline = create_pipeline_from_json(pipeline_config)

    with pytest.raises(Exception):
        await pipeline.run({"test": "data"})
```

### Cleanup and Teardown

```python
@pytest.fixture
def temp_database():
    """Provide temporary database that cleans up after test"""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
        db_url = f"sqlite:///{tmp_db.name}"

    db_manager = DatabaseManager(db_url)
    yield db_manager

    # Cleanup
    db_manager.disconnect()
    Path(tmp_db.name).unlink()
```

### Continuous Integration

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', '3.11']

    steps:
    - uses: actions/checkout@v2
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .
        pip install pytest pytest-asyncio pytest-cov

    - name: Run tests
      run: |
        python -m pytest tests/ -v --cov=ia_modules --cov-report=xml

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v1
      with:
        file: ./coverage.xml
```

This comprehensive testing guide provides the foundation for maintaining high code quality and reliability in the IA Modules framework through systematic testing at all levels.