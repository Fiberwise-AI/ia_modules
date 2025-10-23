# Getting Started with IA Modules

Welcome to IA Modules - a production-ready pipeline framework for building reliable AI agent systems.

## Installation

### Basic Installation

```bash
pip install ia_modules
```

### With Optional Features

```bash
# CLI tools with visualization
pip install ia_modules[cli]

# Performance profiling
pip install ia_modules[profiling]

# Everything
pip install ia_modules[all]
```

### For Development

```bash
git clone <repository-url>
cd ia_modules
pip install -e ".[dev]"
```

## Quick Start (5 Minutes)

### 1. Your First Pipeline

Create a simple pipeline that processes data through multiple steps:

```python
from ia_modules.pipeline.core import PipelineStep, StepResult

# Define pipeline steps
class FetchDataStep(PipelineStep):
    async def execute(self, context):
        # Fetch data from your source
        data = {"user_input": "Hello, world!"}
        return StepResult(
            success=True,
            data=data,
            next_step="process"
        )

class ProcessDataStep(PipelineStep):
    async def execute(self, context):
        # Process the data
        input_data = context.get_data("user_input")
        processed = input_data.upper()
        return StepResult(
            success=True,
            data={"processed": processed},
            next_step="output"
        )

class OutputStep(PipelineStep):
    async def execute(self, context):
        # Output results
        result = context.get_data("processed")
        print(f"Result: {result}")
        return StepResult(success=True)

# Run the pipeline
from ia_modules.pipeline.runner import PipelineRunner

async def main():
    runner = PipelineRunner()

    # Register steps
    runner.register_step("fetch", FetchDataStep())
    runner.register_step("process", ProcessDataStep())
    runner.register_step("output", OutputStep())

    # Execute
    result = await runner.run(start_step="fetch")
    print(f"Pipeline completed: {result.success}")

import asyncio
asyncio.run(main())
```

### 2. Using the CLI

IA Modules includes a powerful CLI for managing pipelines:

```bash
# Validate a pipeline definition
ia-modules validate my_pipeline.json

# Run a pipeline
ia-modules run my_pipeline.json

# Visualize pipeline structure
ia-modules visualize my_pipeline.json --output graph.png

# List all pipelines
ia-modules list

# Show pipeline details
ia-modules info my_pipeline
```

## Core Concepts

### Pipeline Structure

Pipelines are defined in JSON format:

```json
{
  "name": "example_pipeline",
  "version": "1.0.0",
  "steps": {
    "start": {
      "module": "my_module.steps",
      "class": "StartStep",
      "transitions": {
        "success": "process",
        "error": "error_handler"
      }
    },
    "process": {
      "module": "my_module.steps",
      "class": "ProcessStep",
      "transitions": {
        "success": "end"
      }
    }
  }
}
```

### Context and Data Flow

The pipeline context manages data between steps:

```python
class MyStep(PipelineStep):
    async def execute(self, context):
        # Get data from previous steps
        user_input = context.get_data("user_input")

        # Process data
        result = process(user_input)

        # Store data for next steps
        context.set_data("result", result)

        return StepResult(success=True, next_step="next")
```

## Essential Features

### 1. Checkpointing (Resume Failed Pipelines)

Enable automatic checkpointing to resume from failures:

```python
from ia_modules.checkpoint.manager import CheckpointManager

# Initialize checkpoint manager
checkpoint_mgr = CheckpointManager(storage_path="./checkpoints")

# Create runner with checkpointing
runner = PipelineRunner(checkpoint_manager=checkpoint_mgr)

# Run with checkpointing enabled
result = await runner.run(
    start_step="fetch",
    checkpoint_enabled=True,
    thread_id="my_workflow_123"
)

# Resume from checkpoint after failure
result = await runner.resume(thread_id="my_workflow_123")
```

### 2. Reliability Metrics (Monitor Your Pipelines)

Track reliability metrics for production systems:

```python
from ia_modules.reliability.metrics import ReliabilityMetrics
from ia_modules.reliability.memory_storage import InMemoryMetricStorage

# Initialize metrics
storage = InMemoryMetricStorage()
metrics = ReliabilityMetrics(storage)

# Record step execution
await metrics.record_step(
    agent_name="data_processor",
    success=True,
    required_compensation=False
)

# Record workflow completion
await metrics.record_workflow(
    workflow_id="workflow_123",
    steps=5,
    retries=1,
    success=True
)

# Get reliability report
report = await metrics.get_report()
print(f"Success Rate: {report.sr:.2%}")
print(f"Compensation Rate: {report.cr:.2%}")
print(f"Human Intervention Rate: {report.hir:.2%}")
```

### 3. Multi-Agent Workflows

Coordinate multiple agents in parallel or sequence:

```python
from ia_modules.agents.orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator()

# Define agents
orchestrator.register_agent("researcher", ResearchAgent())
orchestrator.register_agent("writer", WriterAgent())
orchestrator.register_agent("reviewer", ReviewerAgent())

# Execute workflow
result = await orchestrator.execute_workflow(
    workflow_id="content_creation",
    agents=["researcher", "writer", "reviewer"],
    mode="sequential"
)
```

### 4. Conditional Routing

Route execution based on step results:

```python
class DataValidatorStep(PipelineStep):
    async def execute(self, context):
        data = context.get_data("input_data")

        if self.validate(data):
            return StepResult(
                success=True,
                next_step="process_valid"
            )
        else:
            return StepResult(
                success=True,
                next_step="handle_invalid",
                data={"error": "Validation failed"}
            )
```

### 5. Parallel Execution

Execute multiple steps concurrently:

```json
{
  "name": "parallel_pipeline",
  "steps": {
    "start": {
      "module": "steps",
      "class": "StartStep",
      "transitions": {
        "success": ["process_a", "process_b", "process_c"]
      }
    },
    "process_a": {
      "module": "steps",
      "class": "ProcessAStep",
      "transitions": {"success": "merge"}
    },
    "process_b": {
      "module": "steps",
      "class": "ProcessBStep",
      "transitions": {"success": "merge"}
    },
    "process_c": {
      "module": "steps",
      "class": "ProcessCStep",
      "transitions": {"success": "merge"}
    },
    "merge": {
      "module": "steps",
      "class": "MergeStep"
    }
  }
}
```

## Production Deployment

### Database Storage for Metrics

Use SQL storage for production reliability metrics:

```python
from ia_modules.reliability.sql_metric_storage import SQLMetricStorage
from ia_modules.database.interfaces import ConnectionConfig, DatabaseType

# PostgreSQL configuration
config = ConnectionConfig(
    database_type=DatabaseType.POSTGRESQL,
    database_url="postgresql://user:pass@localhost/metrics"
)

storage = SQLMetricStorage(config)
await storage.initialize()

metrics = ReliabilityMetrics(storage)
```

### Monitoring and Alerting

Monitor SLO compliance in production:

```python
from ia_modules.reliability.slo_monitor import SLOMonitor

monitor = SLOMonitor(metrics)

# Check SLO compliance
compliance = await monitor.check_compliance()

if not compliance.mtte_compliant:
    print(f"⚠️ MTTE exceeds target: {compliance.mtte_current}ms")

if not compliance.rsr_compliant:
    print(f"⚠️ RSR below target: {compliance.rsr_current:.2%}")
```

### Error Recovery

Implement automatic error recovery:

```python
from ia_modules.reliability.recovery import RecoveryStrategy

class MyStep(PipelineStep):
    async def execute(self, context):
        try:
            result = await self.process_data()
            return StepResult(success=True, data=result)
        except RetryableError as e:
            # Automatic retry with backoff
            return StepResult(
                success=False,
                retry=True,
                retry_delay=5.0
            )
        except FatalError as e:
            # Route to error handler
            return StepResult(
                success=False,
                next_step="error_handler",
                data={"error": str(e)}
            )
```

## Example: Complete Research Agent

Here's a complete example of a research agent with reliability monitoring:

```python
from ia_modules.pipeline.core import PipelineStep, StepResult
from ia_modules.pipeline.runner import PipelineRunner
from ia_modules.reliability.metrics import ReliabilityMetrics
from ia_modules.reliability.memory_storage import InMemoryMetricStorage
from ia_modules.checkpoint.manager import CheckpointManager

class ResearchQueryStep(PipelineStep):
    async def execute(self, context):
        query = context.get_data("query")
        # Generate search queries
        queries = self.generate_queries(query)
        return StepResult(
            success=True,
            data={"search_queries": queries},
            next_step="search"
        )

class SearchStep(PipelineStep):
    async def execute(self, context):
        queries = context.get_data("search_queries")
        # Execute searches (parallel)
        results = await self.search(queries)
        return StepResult(
            success=True,
            data={"search_results": results},
            next_step="synthesize"
        )

class SynthesizeStep(PipelineStep):
    async def execute(self, context):
        results = context.get_data("search_results")
        # Synthesize findings
        report = self.synthesize(results)
        return StepResult(
            success=True,
            data={"report": report}
        )

async def main():
    # Initialize components
    checkpoint_mgr = CheckpointManager()
    metrics_storage = InMemoryMetricStorage()
    metrics = ReliabilityMetrics(metrics_storage)

    # Create runner
    runner = PipelineRunner(
        checkpoint_manager=checkpoint_mgr,
        metrics=metrics
    )

    # Register steps
    runner.register_step("query", ResearchQueryStep())
    runner.register_step("search", SearchStep())
    runner.register_step("synthesize", SynthesizeStep())

    # Execute with monitoring
    result = await runner.run(
        start_step="query",
        checkpoint_enabled=True,
        thread_id="research_123",
        initial_data={"query": "What is quantum computing?"}
    )

    # Get reliability metrics
    report = await metrics.get_report()
    print(f"Success Rate: {report.sr:.2%}")
    print(f"Average TCL: {report.tcl:.2f}ms")

import asyncio
asyncio.run(main())
```

## Next Steps

- **[Features Overview](FEATURES.md)** - Complete feature matrix
- **[API Reference](API_REFERENCE.md)** - Detailed API documentation
- **[Reliability Guide](RELIABILITY_USAGE_GUIDE.md)** - Production reliability patterns
- **[Plugin System](PLUGIN_SYSTEM_DOCUMENTATION.md)** - Extending IA Modules
- **[CLI Documentation](CLI_TOOL_DOCUMENTATION.md)** - Command-line tools
- **[Examples](EXAMPLES.md)** - More complete examples
- **[Deployment Guide](DEPLOYMENT.md)** - Production deployment
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues and solutions

## Getting Help

- GitHub Issues: Report bugs or request features
- Documentation: Full documentation at docs/
- Examples: Check tests/pipelines/ for working examples

## License

See [LICENSE](../LICENSE) for details.
