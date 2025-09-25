# Pipeline Tests

This directory contains unit tests, integration tests, and end-to-end tests for the pipeline infrastructure components.

## Test Structure

- **Unit Tests** (`tests/unit/`): 
  - Fast-running tests for individual components
  - Focus on core functionality and edge cases
  - Use mocks and stubs where appropriate
  - Test individual classes and functions in isolation

- **Integration Tests** (`tests/integration/`):
  - Tests that verify components work together
  - Test full pipeline execution flows
  - Test database integration and service dependencies
  - Test real data scenarios and component interactions

- **End-to-End Tests** (`tests/e2e/`):
  - Complete system tests from start to finish
  - Test real-world scenarios with actual pipeline configurations
  - Verify complete pipeline execution including file I/O
  - Test full user workflows and system integration

## Running Tests

To run all tests:

```bash
pytest -v
```

To run specific test types:

```bash
# Run unit tests only
pytest tests/unit/ -v

# Run integration tests only
pytest tests/integration/ -v

# Run end-to-end tests only
pytest tests/e2e/ -v

# Run tests by component
pytest tests/unit/test_condition_functions.py -v
pytest tests/integration/test_condition_functions_integration.py -v
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
