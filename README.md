# IA Modules - Pipeline Framework

A Python package for building robust, graph-based pipelines with conditional routing, service injection, authentication, and database management.

## Table of Contents

- [Overview](#overview)

- [Pipeline Examples & Features](TEST_PIPELINES_GUIDE.md) **See framework capabilities in action with runnable examples**
- [Core Features](#core-features)
- [Quickstart: Your First Pipeline](#quickstart-your-first-pipeline)
- [Core Architecture Principles](#core-architecture-principles)
- [Package Structure](#package-structure)
- [Key Components](#key-components)
  - [Pipeline System](#pipeline-system)
  - [Authentication System](#authentication-system)
  - [Database Layer](#database-layer)
- [Defining Pipelines (JSON Format)](#defining-pipelines-json-format)
  - [Top-Level Fields](#top-level-fields)
  - [Step Definition](#step-definition)
  - [Flow & Routing](#flow--routing)
  - [Templating and Parameterization](#templating-and-parameterization)
  - [Full JSON Example](#full-json-example)
- [Running Pipelines](#running-pipelines)
  - [Application Responsibilities](#application-responsibilities)
  - [Option 1: DB-Backed Execution (Production)](#option-1-db-backed-execution-production)
  - [Option 2: File-Based Execution (Development)](#option-2-file-based-execution-development)
- [Human-in-the-Loop (HITL)](#human-in-the-loop-hitl)
- [Advanced Topics](#advanced-topics)
  - [The Pipeline Importer Service](#the-pipeline-importer-service)
  - [Database Schema](#database-schema)
- [Installation](#installation)
- [Contributing](#contributing)

## Overview

`ia_modules` provides a comprehensive framework for creating, managing, and executing complex workflows as Directed Acyclic Graphs (DAGs). It is designed to decouple business logic into modular `Steps`, orchestrate their execution with intelligent routing, and provide common services like database access and authentication out-of-the-box.

## Core Features

- **Graph-Based Execution**: Define complex workflows as DAGs in simple JSON.
- **Conditional Routing**: Control the pipeline's path using simple expressions, with future support for AI and function-based decisions.
- **Service Injection**: Steps have secure and easy access to shared services like databases (`self.get_db()`) and HTTP clients (`self.get_http()`).
- **Human-in-the-Loop**: Pause a pipeline and wait for external human input before resuming.
- **Database Management**: Includes a database manager with multi-provider support and a built-in migration system.
- **Authentication**: FastAPI-compatible authentication middleware and session management.

## Documentation

Comprehensive documentation is available in the `docs/` folder:

### 📋 **Guides & Tutorials**
- **[Pipeline Examples & Features](TEST_PIPELINES_GUIDE.md)** - Runnable examples showcasing each framework feature
- **[Developer Guide](docs/DEVELOPER_GUIDE.md)** - Complete development workflow and best practices
- **[Testing Guide](docs/TESTING_GUIDE.md)** - Testing patterns, fixtures, and validation strategies

### 🏗️ **Architecture & Design**
- **[Pipeline Architecture](docs/PIPELINE_ARCHITECTURE.md)** - Core system design and execution patterns
- **[Database Interfaces](docs/DATABASE_INTERFACES.md)** - Database layer design and integration patterns

### 🔍 **References**
- **[API Reference](docs/API_REFERENCE.md)** - Complete API documentation for all modules
- **[Human-in-the-Loop Guide](docs/HUMAN_IN_LOOP_COMPREHENSIVE.md)** - Interactive workflow patterns and implementation

### 🎯 **Quick Navigation**
| Topic | Documentation | Purpose |
|-------|---------------|---------|
| **Getting Started** | [Pipeline Examples](TEST_PIPELINES_GUIDE.md) | See working examples |
| **Development** | [Developer Guide](docs/DEVELOPER_GUIDE.md) | Build and contribute |
| **Architecture** | [Pipeline Architecture](docs/PIPELINE_ARCHITECTURE.md) | Understand the system |
| **Testing** | [Testing Guide](docs/TESTING_GUIDE.md) | Quality assurance |
| **API Reference** | [API Reference](docs/API_REFERENCE.md) | Technical details |
| **Advanced Features** | [Human-in-the-Loop](docs/HUMAN_IN_LOOP_COMPREHENSIVE.md) | Interactive workflows |

## Quickstart: Your First Pipeline

Let's create and run a simple two-step pipeline.

1. Create your project structure:

```bash
mkdir my_pipeline_app
cd my_pipeline_app
mkdir pipelines
touch main.py
touch steps.py
```

2. Define the pipeline (`pipelines/hello_world.json`):

```json
{
  "name": "Hello World Pipeline",
  "version": "1.0",
  "steps": [
    {
      "id": "step_one",
      "step_class": "GenerateMessageStep",
      "module": "steps"
    },
    {
      "id": "step_two",
      "step_class": "PrintMessageStep",
      "module": "steps",
      "inputs": {
        "message": "{{ step_one.greeting }}"
      }
    }
  ],
  "flow": {
    "start_at": "step_one",
    "transitions": [
      { "from": "step_one", "to": "step_two" }
    ]
  }
}
```

3. Implement the steps (`steps.py`):

```python
# steps.py
from ia_modules.pipeline import Step

class GenerateMessageStep(Step):
    async def work(self, data):
        name = data.get("name", "World")
        return {"greeting": f"Hello, {name}!"}

class PrintMessageStep(Step):
    async def work(self, data):
        message = data.get("message")
        print(message)
        return {"status": "Message printed successfully"}
```

4. Create the runner (`main.py`):

```python
# main.py
import asyncio
from ia_modules.pipeline.runner import run_pipeline_from_json

async def main():
    print("Running the pipeline...")
    result = await run_pipeline_from_json(
        pipeline_file="pipelines/hello_world.json",
        input_data={"name": "Developer"}
    )
    print("\nPipeline finished with result:")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

5. Run it!

Make sure `ia_modules` is installed and in your `PYTHONPATH`.

```bash
python main.py
```

Expected Output:

```
Running the pipeline...
Hello, Developer!

Pipeline finished with result:
{'status': 'Message printed successfully'}
```

## Core Architecture Principles

1. **Graph-First Design**: All pipelines are defined as **directed acyclic graphs (DAGs)**, ensuring a clear, predictable, and finite execution flow.
2. **Intelligent Flow Control**:
   - **Expression-based routing**: Simple logical conditions on step outputs (e.g., `result.score > 0.8`).
   - **AI-driven routing**: (Future) Use an agent to make complex routing decisions.
   - **Function-based routing**: (Future) Define routing logic with custom Python functions.
3. **Service Injection & Dependency Management**: A clean `ServiceRegistry` pattern provides steps with managed access to shared resources like database connections and HTTP clients.

## Package Structure

```
ia_modules/
├── pipeline/            # Core pipeline execution engine
├── auth/                # Authentication and session management
├── database/            # Database abstraction and management
├── web/                 # Web utilities (execution tracking, etc.)
└── data/                # Shared data models
```

## Key Components

#### Pipeline System

- **Step**: The base class for all pipeline steps. Implements the `work` method containing business logic.
- **Pipeline**: The orchestrator that executes the graph, manages state, and handles routing.
- **HumanInputStep**: A specialized step that pauses execution to wait for human interaction.
- **ServiceRegistry**: A dependency injection container for services (DB, HTTP, etc.).

#### Authentication System

- **AuthMiddleware**: FastAPI-compatible middleware for protecting endpoints.
- **SessionManager**: Manages secure user sessions.

#### Database Layer

- **DatabaseManager**: Handles connections to multiple database backends (e.g., SQLite, PostgreSQL).
- **Migration System**: Manages database schema versioning and updates.

## Defining Pipelines (JSON Format)

Pipelines are defined as JSON documents, making them language-agnostic and easy to store, version, and edit.

### Top-Level Fields

- `name` (string): Human-readable name for the pipeline.
- `version` (string): Version of the pipeline (e.g., "1.0").
- `parameters` (object, optional): Default input parameters for a pipeline run.
- `steps` (array): A list of all step objects in the graph.
- `flow` (object): Defines the execution order and conditional transitions.
- `error_handling` (object, optional): Global configuration for retries, timeouts, etc.
- `outputs` (object, optional): Defines the final output of the entire pipeline, often templated from step results.

### Step Definition

- `id` (string): A unique identifier for the step within the pipeline.
- `step_class` (string): The Python class name that implements the step's logic.
- `module` (string): The Python module path where the `step_class` can be found.
- `inputs` (object, optional): Maps required inputs for the step to outputs from other steps or pipeline parameters.
- `config` (object, optional): Static configuration passed to the step instance.

### Flow & Routing

- `flow.start_at` (string): The `id` of the first step to execute.
- `flow.transitions` (array): A list of directed edges in the graph. Each transition includes:
  - `from` (string): The source step `id`.
  - `to` (string): The destination step `id`.
  - `condition` (string | object, optional): The routing rule. Defaults to `"always"`.
    - **Simple Conditions**: `"always"` or `"parameter:my_param"` (checks for truthiness of a pipeline parameter).
    - **Expression Conditions**: An object for more complex logic, e.g., `{"type": "expression", "config": {"source": "step_id.output_name", "operator": "gt", "value": 100}}`.

### Templating and Parameterization

The framework uses a simple templating syntax to pass data between steps and from pipeline parameters.

- **Reference a step's output**: `{{ step_id.output_name }}`
- **Reference a pipeline parameter**: `{{ parameters.param_name }}`

This syntax is used within the `inputs` mapping of a step.

> Note: Consolidating to `{{ ... }}` is recommended for clarity. If backwards compatibility is required for the single-brace style, document it as deprecated.

### Full JSON Example

This pipeline scrapes a Wikipedia page and processes its content.

```json
{
  "name": "AI Notebook Creation Pipeline",
  "description": "Creates research notebooks with AI-enhanced content.",
  "version": "1.0",
  "parameters": {
    "topic": "artificial intelligence",
    "enable_ai_processing": true
  },
  "steps": [
    {
      "id": "wikipedia_scraper",
      "name": "Wikipedia Content Scraper",
      "step_class": "WikipediaScraperStep",
      "module": "knowledge_processing.wikipedia_scraper_step",
      "inputs": {
        "topic": "{{ parameters.topic }}"
      },
      "outputs": ["scraped_content", "article_title"]
    },
    {
      "id": "structured_processor",
      "name": "Structured Content Processor",
      "step_class": "StructuredWikipediaProcessorStep",
      "module": "knowledge_processing.structured_wikipedia_processor_step",
      "inputs": {
        "scraped_html": "{{ wikipedia_scraper.scraped_content }}",
        "title": "{{ wikipedia_scraper.article_title }}"
      },
      "outputs": ["structured_content"]
    }
  ],
  "flow": {
    "start_at": "wikipedia_scraper",
    "transitions": [
      {
        "from": "wikipedia_scraper",
        "to": "structured_processor"
      }
    ]
  }
}
```

## Running Pipelines

There are two primary ways to execute a pipeline: backed by a database (for production) or directly from a file (for development).

### Application Responsibilities

The `ia_modules` library provides the execution engine. The consuming application is responsible for:

1. **Storing Pipeline JSON**: Keeping pipeline definitions in a local directory (e.g., `pipelines/`).
2. **Providing Services**: Creating and configuring the `ServiceRegistry` with application-specific database connections, HTTP clients, and other services.
3. **Database Integration**: (For production) Running the necessary database migrations and using the `PipelineImportService` to load JSON definitions into the database.

### Option 1: DB-Backed Execution (Production)

This is the standard approach for a deployed application.

```python
from ia_modules.pipeline.importer import PipelineImportService
from ia_modules.pipeline.runner import create_pipeline_from_json
from ia_modules.pipeline.services import ServiceRegistry

# 1. Setup application services
services = ServiceRegistry(db=app_db_manager, http=http_client)

# 2. Load pipeline configuration from the database
importer = PipelineImportService(db_provider, pipelines_dir='/path/to/pipelines')
pipeline_row = await importer.get_pipeline_by_slug('ai-notebook-creation-pipeline-v1')
pipeline_config = pipeline_row['pipeline_config'] # This is the parsed JSON

# 3. Create and run the pipeline instance
pipeline = create_pipeline_from_json(pipeline_config, services=services)
result = await pipeline.run({'topic': 'machine learning'})
```

### Option 2: File-Based Execution (Development)

Ideal for local development, testing, and ad-hoc runs.

```python
from ia_modules.pipeline.runner import run_pipeline_from_json
from ia_modules.pipeline.services import ServiceRegistry

# 1. Setup services (can be mocked for testing)
services = ServiceRegistry(db=mock_db, http=mock_http)

# 2. Run directly from the JSON file
result = await run_pipeline_from_json(
    pipeline_file="pipelines/ai_notebook_creation_pipeline.json",
    input_data={"topic": "machine learning"},
    services=services
)
```

## Human-in-the-Loop (HITL)

Pipelines can include a `HumanInputStep` to pause execution and wait for external input. When the runner encounters this step type, it will halt and return a payload indicating that human action is required.

**Example HITL Step Definition:**

```json
{
  "id": "human_approval",
  "type": "human_input",
  "step_class": "HumanInputStep",
  "config": {
    "ui_schema": {
      "title": "Review Content for Approval",
      "fields": [
        {"name": "decision", "type": "radio", "options": ["Approve", "Reject"]},
        {"name": "notes", "type": "textarea", "label": "Reasoning"}
      ]
    }
  }
}
```

The application's UI can use the `ui_schema` to dynamically render a form. Once the user submits the form, the application can resume the pipeline run, providing the user's data as input to the `human_approval` step.

## Advanced Topics

### The Pipeline Importer Service

In a production environment, pipelines are loaded from the filesystem into a database table for fast and reliable access. The `PipelineImportService` handles this process.

- **Location**: `ia_modules/pipeline/importer.py`
- **Purpose**: Scans a directory for `*.json` files, validates them, and upserts them into the `pipelines` database table.
- **Change Detection**: It computes a hash of the file content to avoid redundant database writes if a pipeline definition hasn't changed.

The consuming application typically calls `importer.import_all_pipelines()` on startup.

### Database Schema

The importer and runner expect a specific database schema for storing pipelines, execution jobs, and logs. The SQL migration scripts are located in `ia_modules/pipeline/migrations/`.

**It is the consuming application's responsibility to run these migrations.**

You can either:
1. Copy the migration files into your application's migration directory.
2. Configure your migration tool (e.g., Alembic) to discover migrations within `ia_modules`.

The core table is `pipelines`, which stores the JSON definition and metadata for each imported pipeline.

## Installation

This package is intended for private use. Install it directly from the Git repository.

```bash
# For development (allows editing the code locally)
git clone <repository-url>
cd ia_modules
pip install -e .

# For deployment
pip install git+<repository-url>
```

## Contributing

1. **Branching**: Create a feature branch from `develop` (e.g., `feature/add-new-step`).
2. **Development**: Make your changes. Ensure you add unit tests for new functionality.
3. **Linting & Formatting**: Run `black .` and `flake8` to ensure code quality.
4. **Pull Request**: Open a PR against the `develop` branch. Provide a clear description of the changes.

---