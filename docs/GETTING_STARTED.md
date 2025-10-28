# Getting Started with IA Modules

Build AI workflows as directed graphs with execution tracking and reliability monitoring.

## Installation

```bash
# Install from source
git clone <repository-url>
cd ia_modules
pip install -e .

# Install with CLI visualization support
pip install -e ".[cli]"
```

**Dependencies:**
- Python 3.9+
- nexusql (database adapter for SQLAlchemy)

## Quick Start

### 1. Your First Pipeline

Create a simple pipeline JSON configuration:

**pipeline.json:**
```json
{
  "name": "hello_pipeline",
  "steps": [
    {
      "id": "greet",
      "module": "my_steps",
      "step_class": "GreetingStep",
      "config": {
        "greeting": "Hello"
      }
    }
  ],
  "flow": {
    "start_at": "greet",
    "paths": []
  }
}
```

**my_steps.py:**
```python
from ia_modules.pipeline.core import Step

class GreetingStep(Step):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.greeting = config.get('greeting', 'Hello')

    async def execute(self, data):
        message = f"{self.greeting}, World!"
        print(message)
        return {"message": message}
```

**Run it:**
```bash
ia-modules run pipeline.json
```

### 2. Using the CLI

```bash
# Run a pipeline
ia-modules run pipeline.json

# Validate a pipeline definition
ia-modules validate pipeline.json

# Format pipeline JSON
ia-modules format pipeline.json --in-place

# Visualize pipeline structure
ia-modules visualize pipeline.json --output graph.png
```

See [CLI_TOOL_DOCUMENTATION.md](CLI_TOOL_DOCUMENTATION.md) for details.

## Core Concepts

### Pipeline Structure

Pipelines use a graph-based flow with JSON configuration:

```json
{
  "name": "example_pipeline",
  "steps": [
    {
      "id": "fetch",
      "module": "my_steps",
      "step_class": "FetchStep"
    },
    {
      "id": "process",
      "module": "my_steps",
      "step_class": "ProcessStep"
    }
  ],
  "flow": {
    "start_at": "fetch",
    "paths": [
      {
        "from_step": "fetch",
        "to_step": "process"
      }
    ]
  }
}
```

### Step Implementation

Steps receive data from previous steps and return results:

```python
from ia_modules.pipeline.core import Step

class MyStep(Step):
    def __init__(self, name, config):
        super().__init__(name, config)

    async def execute(self, data):
        # data contains results from previous steps
        input_value = data.get('previous_step_result')

        # Process data
        result = self.process(input_value)

        # Return result for next steps
        return {"result": result}
```

## Essential Features

### 1. Execution Tracking

Track pipeline executions with ExecutionContext:

```python
from ia_modules.pipeline.runner import run_pipeline_from_json
from ia_modules.pipeline.services import ServiceRegistry
from ia_modules.pipeline.core import ExecutionContext

# Create services and execution context
services = ServiceRegistry()
execution_context = ExecutionContext(
    execution_id='job-123',
    pipeline_id='my-pipeline',
    user_id='user-456'
)

# Run pipeline with tracking
result = await run_pipeline_from_json(
    'pipeline.json',
    input_data={'key': 'value'},
    services=services,
    execution_context=execution_context
)
```

### 2. Database Integration

Use nexusql for database operations:

```python
from nexusql import DatabaseManager
from ia_modules.pipeline.services import ServiceRegistry

# Setup database
db_manager = DatabaseManager({'database_url': 'sqlite:///pipeline.db'})

# Register with services
services = ServiceRegistry()
services.register('database', db_manager)

# Use in steps
class MyStep(Step):
    async def execute(self, data):
        db = self.services.get('database')
        result = db.fetch_one("SELECT * FROM results WHERE id = ?", (1,))
        return {"data": result}
```

### 3. Conditional Routing

Route execution based on conditions:

```json
{
  "flow": {
    "start_at": "validate",
    "paths": [
      {
        "from_step": "validate",
        "to_step": "process_valid",
        "condition": {
          "type": "field_equals",
          "field": "valid",
          "value": true
        }
      },
      {
        "from_step": "validate",
        "to_step": "handle_invalid",
        "condition": {
          "type": "field_equals",
          "field": "valid",
          "value": false
        }
      }
    ]
  }
}
```

### 4. Parallel Execution

Execute multiple branches concurrently:

```json
{
  "flow": {
    "start_at": "split",
    "paths": [
      {"from_step": "split", "to_step": "process_a"},
      {"from_step": "split", "to_step": "process_b"},
      {"from_step": "split", "to_step": "process_c"},
      {"from_step": "process_a", "to_step": "merge"},
      {"from_step": "process_b", "to_step": "merge"},
      {"from_step": "process_c", "to_step": "merge"}
    ]
  }
}
```

All three process steps run concurrently, then merge combines results.

## Production Setup

### Database Configuration

Use nexusql with PostgreSQL for production:

```python
from nexusql import DatabaseManager

# PostgreSQL configuration
db_config = {
    'database_url': 'postgresql://user:pass@localhost/pipelines'
}

db_manager = DatabaseManager(db_config)
services.register('database', db_manager)
```

### WebSocket Integration

Real-time updates for pipeline execution:

```python
from ia_modules.pipeline.runner import run_pipeline_from_json

# Run with WebSocket notifications
result = await run_pipeline_from_json(
    'pipeline.json',
    input_data=data,
    services=services,
    execution_context=execution_context,
    websocket_manager=ws_manager,
    user_id=user_id
)
```

### Error Handling in Steps

Handle errors within step execution:

```python
class MyStep(Step):
    async def execute(self, data):
        try:
            result = await self.process_data(data)
            return {"success": True, "result": result}
        except Exception as e:
            # Log error and return failure
            logger.error(f"Step failed: {e}")
            return {"success": False, "error": str(e)}
```

## Complete Example

Here's a complete example with database integration:

**pipeline.json:**
```json
{
  "name": "data_processor",
  "steps": [
    {
      "id": "fetch",
      "module": "my_steps",
      "step_class": "FetchStep"
    },
    {
      "id": "validate",
      "module": "my_steps",
      "step_class": "ValidateStep"
    },
    {
      "id": "save",
      "module": "my_steps",
      "step_class": "SaveStep"
    }
  ],
  "flow": {
    "start_at": "fetch",
    "paths": [
      {"from_step": "fetch", "to_step": "validate"},
      {"from_step": "validate", "to_step": "save"}
    ]
  }
}
```

**my_steps.py:**
```python
from ia_modules.pipeline.core import Step

class FetchStep(Step):
    async def execute(self, data):
        # Fetch data
        return {"raw_data": "example data"}

class ValidateStep(Step):
    async def execute(self, data):
        raw_data = data.get('raw_data')
        is_valid = len(raw_data) > 0
        return {"valid": is_valid, "data": raw_data}

class SaveStep(Step):
    async def execute(self, data):
        # Save to database
        db = self.services.get('database')
        execution_id = data.get('execution_context', {}).get('execution_id')

        db.execute(
            "INSERT INTO results (execution_id, data) VALUES (?, ?)",
            (execution_id, data.get('data'))
        )
        return {"saved": True}
```

**run.py:**
```python
import asyncio
from nexusql import DatabaseManager
from ia_modules.pipeline.runner import run_pipeline_from_json
from ia_modules.pipeline.services import ServiceRegistry
from ia_modules.pipeline.core import ExecutionContext

async def main():
    # Setup database
    db = DatabaseManager({'database_url': 'sqlite:///pipeline.db'})

    # Create services
    services = ServiceRegistry()
    services.register('database', db)

    # Create execution context
    ctx = ExecutionContext(
        execution_id='job-001',
        pipeline_id='data_processor',
        user_id='user-123'
    )

    # Run pipeline
    result = await run_pipeline_from_json(
        'pipeline.json',
        input_data={},
        services=services,
        execution_context=ctx
    )

    print(f"Result: {result}")

asyncio.run(main())
```

## Next Steps

- **[Features Overview](FEATURES.md)** - Complete feature matrix
- **[Developer Guide](DEVELOPER_GUIDE.md)** - Detailed API documentation
- **[Reliability Guide](RELIABILITY_USAGE_GUIDE.md)** - Production reliability patterns
- **[Plugin System](PLUGIN_SYSTEM_DOCUMENTATION.md)** - Extending IA Modules
- **[CLI Documentation](CLI_TOOL_DOCUMENTATION.md)** - Command-line tools
- **[Pipeline Architecture](PIPELINE_ARCHITECTURE.md)** - Architecture overview
- **[Execution Architecture](EXECUTION_ARCHITECTURE.md)** - Execution patterns
- **[Testing Guide](TESTING_GUIDE.md)** - Testing pipelines

## Getting Help

- Documentation: See [docs/](.)
- Examples: Check tests/pipelines/ for working examples
