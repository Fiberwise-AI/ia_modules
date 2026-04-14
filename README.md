# IA Modules

**Intelligent Application Modules - Python framework for building reliable AI workflows**

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://github.com/yourusername/ia_modules/actions/workflows/test.yml/badge.svg)](https://github.com/yourusername/ia_modules/actions/workflows/test.yml)
[![Coverage](https://img.shields.io/badge/coverage-dynamic-lightgrey.svg)](htmlcov/index.html)
[![EARF Compliant](https://img.shields.io/badge/EARF-compliant-success.svg)](docs/RELIABILITY_USAGE_GUIDE.md)


## Table of Contents

- [What is this?](#what-is-this)
- [Core Features](#core-features)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Your First Pipeline](#quickstart-your-first-pipeline)
- [Built-in Step Types](#built-in-step-types)
- [Agent Authentication](#agent-authentication)
- [Core Architecture](#core-architecture-principles)
- [Pipeline Definition (JSON)](#defining-pipelines-json-format)
- [Running Pipelines](#running-pipelines)
- [AI/LLM Integration](#aillm-integration)
- [Human-in-the-Loop](#human-in-the-loop-hitl)
- [Parallel Processing](#parallel-processing)
- [Comparison vs LangChain/LangGraph](#why-ia-modules)
- [Roadmap](#roadmap)
- [Contributing](#contributing)

## What is this?

IA Modules runs AI workflows as directed graphs. You define steps (call an LLM, transform data, wait for human input) and connect them. The framework executes your graph, handles routing and parallelization, and tracks reliability metrics.

**What it does:**
- Executes workflows as directed graphs (DAGs) with conditional routing
- Tracks 7 reliability metrics: Success Rate, Compensation Rate, Pass Confidence, Human Intervention Rate, Model Accuracy, Tool Call Latency, Workflow Completion Time
- Checkpoints state so you can resume failed workflows
- Stores data in SQLite, PostgreSQL, MySQL, or MSSQL via [nexusql](https://github.com/Fiberwise-AI/nexusql)
- Exports metrics to Prometheus, CloudWatch, or Datadog
- Includes web UI for building and monitoring workflows

**Use it if:**
- You need to chain multiple LLM calls with conditional logic
- You want metrics on workflow reliability
- You need human-in-the-loop approval steps
- You want to checkpoint and resume long-running processes

**Try it:** Clone the repo and run `cd showcase_app && docker-compose up` to see the web UI.

## Core Features

### 🚀 **Pipeline & Workflow Execution**
- **Graph-Based Workflows**: Directed graphs with conditional routing, cycle detection, and loop execution
- **Conditional Routing**: Expression-based routing with dynamic step selection
- **Parallel Execution**: Automatic concurrent execution of independent pipeline branches
- **Checkpointing & Recovery**: Resume failed pipelines from last checkpoint with state serialization
- **Human-in-the-Loop (HITL)**: Pause execution for human approval with UI schema definitions
- **JSON Pipeline Definitions**: Declarative pipeline config with dynamic step loading and templating
- **Context Management**: Thread-safe execution context for step data sharing

### 📊 **Reliability & Observability (EARF)**
- **Reliability Metrics**: SR, CR, PC, HIR, MA, TCL, WCT tracking across executions
- **Storage Backends**: In-memory, SQL (PostgreSQL, MySQL, SQLite, DuckDB), Redis
- **SLO Monitoring**: Define and track Service Level Objectives with automated alerts
- **Event Replay**: Replay step executions for debugging and analysis
- **Evidence Collection**: Automatic evidence capture for compliance audits
- **Compensation Tracking**: Track and analyze compensation/rollback events
- **Performance Metrics**: MTTE (Mean Time to Error) and RSR (Retry Success Rate)

### 🤖 **AI & LLM Integration**
- **LLM Providers**: OpenAI, Anthropic, Google Gemini with unified interface
- **Built-in Step Types**: `LLMStep`, `FunctionStep`, `AgentStep`, `A2AStep`, `ParallelStep`, `OrchestratorStep` — no subclassing needed for common patterns
- **CLI Agent Execution**: Run Claude Code SDK or OpenCode as pipeline steps with workspace, mode, and tool constraints
- **A2A Remote Dispatch**: Send work to remote Agent-to-Agent servers via `A2AStep` with JWT authentication
- **Multi-Agent Orchestration**: Sequential, parallel, and hierarchical agent workflows
- **Agent State Sharing**: Share context and state between agents in workflows
- **Memory System**: Conversation history, session management, vector search, summarization
- **RAG Support**: Retrieval-Augmented Generation pipelines
- **Grounding & Validation**: Citation tracking, fact verification, grounding metrics
- **Agentic Patterns**: Debate, reflection, planning, agentic RAG, Chain-of-Thought, ReAct, Tree-of-Thoughts

### 🔑 **Agent Authentication & Permissions**
- **Pluggable OIDC**: Built-in mini OIDC provider for local dev, Keycloak/Auth0/Cognito for production
- **Custom `a2a` JWT Claim**: Scope agent execution to specific working directories, modes, and tools
- **Zero-Trust Enforcement**: `enforce_agent_claims()` validates JWT claims against the actual CWD/mode/tools before any executor runs
- **Default Permissions**: Sane defaults for research vs execute modes with tool allow-lists

### 🎨 **Web UI (Showcase App)**
- Visual workflow builder with drag-and-drop
- JSON editor with validation
- Real-time execution monitoring via WebSocket
- 12 pre-built workflow templates
- Live demos of 5 agentic patterns
- Save/load/export workflows

### 🛠️ **Developer Tools**
- **CLI**: Pipeline run, validate, visualize, format commands
- **Benchmarking Framework**: Performance and accuracy benchmarks with statistical analysis
- **Plugin System**: Custom steps, storage backends, hooks, automatic discovery
- **Scheduler**: Cron-based job scheduling with async execution and history
- **Validation**: Pydantic schema validation for pipeline inputs/outputs
- **Service Registry**: Dependency injection for database, HTTP, and custom services
- **Pipeline Importer**: JSON import with hash-based change detection

### 📦 **Database & Storage**
- **Database Layer**: [nexusql](https://github.com/Fiberwise-AI/nexusql) with pluggable backends (NexusQL, SQLAlchemy)
- **Supported Databases**: PostgreSQL, MySQL, SQLite, DuckDB, MSSQL
- **Migration System**: Built-in V001__*.sql migration runner with cross-DB syntax translation
- **Telemetry Exporters**: Prometheus, CloudWatch, Datadog

### 🔐 **Security & Validation**
- **Web Authentication**: FastAPI middleware and session management for HTTP endpoints
- **Agent Authentication**: OIDC JWT validation with pluggable IDP adapters (local mini OIDC, Keycloak, etc.)
- **Schema Validation**: Pydantic-based runtime type checking
- **Grounding**: Citation tracking and fact verification for AI outputs

### 📈 **Testing**
- Unit, integration, showcase, and e2e suites
- **Run**: `pytest tests/unit/` (unit), `pytest tests/integration/` (integration)
- Showcase app tests and Playwright e2e run in CI

---

## Quick Start

### Option 1: Try the Showcase App (Recommended) ⭐

The fastest way to see IA Modules in action with a full interactive demo:

```bash
# Clone the repository
git clone https://github.com/Fiberwise-AI/ia_modules.git
cd ia_modules

# Option A: Docker (Easiest)
cd showcase_app
docker-compose up

# Option B: Local Development
# Terminal 1 - Backend
cd showcase_app/backend
pip install -r requirements.txt
python main.py

# Terminal 2 - Frontend  
cd showcase_app/frontend
npm install
npm run dev
```

Open **http://localhost:3000** and explore:

#### 🚀 **What You'll See:**

1. **Agentic Patterns Page** (`/patterns`)
   - Interactive demos of 5 core patterns
   - Real-time execution with step-by-step visualization
   - Live code examples you can modify and run

2. **Multi-Agent Dashboard** (`/multi-agent`)
   - 12 pre-built workflow templates
   - Visual workflow builder with drag-and-drop
   - Real-time execution monitoring via WebSocket
   - Agent communication logs and performance metrics

3. **Pipeline Editor** (`/editor`)
   - Monaco code editor with JSON schema validation
   - Pipeline visualization with React Flow
   - Test execution with live output

4. **Metrics & Monitoring** (`/metrics`)
   - Real-time reliability metrics (SR, CR, PC, HIR)
   - SLO compliance monitoring
   - Execution history and logs

#### � **Enable Real LLM Integration:**

The showcase app works without API keys (demonstration mode), but for **real AI-powered workflows**:

```bash
# Copy environment template
cd showcase_app
cp .env.example .env

# Add your API key (choose one)
# OpenAI (recommended)
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o

# OR Anthropic
ANTHROPIC_API_KEY=sk-ant-your-key-here
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# OR Google Gemini
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-2.0-flash-exp

# Set default provider
DEFAULT_LLM_PROVIDER=openai
```

See [showcase_app/README.md](showcase_app/README.md) for LLM configuration details.

---

### Option 2: Package Installation

For integrating IA Modules into your own project:

```bash
# Basic installation
pip install ia_modules

# With reliability SQL backends
pip install ia_modules[sql]

# With Redis support
pip install ia_modules[redis]

# Everything (recommended for production)
pip install ia_modules[all]
```

---

### Your First Pipeline with Reliability Metrics

```python
from ia_modules.pipeline.core import Step, ExecutionContext
from ia_modules.pipeline.runner import run_pipeline_from_json
from ia_modules.pipeline.services import ServiceRegistry
from ia_modules.reliability import ReliabilityMetrics, MemoryMetricStorage
from ia_modules.database import get_database

# Define a simple step
class ProcessDataStep(Step):
    async def execute(self, data: dict) -> dict:
        input_value = data.get("input", "")
        result = input_value.upper()
        return {"output": result}

# Set up reliability tracking
async def main():
    # Setup database
    db = get_database('sqlite:///app.db')
    db.connect()

    # Setup reliability metrics
    storage = MemoryMetricStorage()
    metrics = ReliabilityMetrics(storage)

    # Setup services
    services = ServiceRegistry()
    services.register('database', db)
    services.register('metrics', metrics)

    # Create execution context
    ctx = ExecutionContext(
        execution_id='demo-001',
        pipeline_id='hello-pipeline',
        user_id='user-123'
    )

    # Run pipeline (from JSON file with steps defined)
    result = await run_pipeline_from_json(
        'pipeline.json',
        input_data={"input": "hello world"},
        services=services,
        execution_context=ctx
    )

    # Get reliability report
    report = await metrics.get_report()
    print(f"Success Rate: {report.sr:.2%}")
    print(f"Compensation Rate: {report.cr:.2%}")

    # Cleanup
    db.disconnect()

import asyncio
asyncio.run(main())
```

## Documentation

### 📚 **Getting Started**
- **[Showcase App Guide](showcase_app/README.md)** ⭐ Interactive demos and tutorials
- **[Patterns Guide](showcase_app/PATTERNS_GUIDE.md)** - Agentic patterns explained
- **[Quick Reference](showcase_app/QUICK_REFERENCE.md)** - Fast lookup guide
- **[Getting Started](docs/GETTING_STARTED.md)** - 5-minute framework quickstart
- **[Features Overview](docs/FEATURES.md)** - Complete feature matrix
- **[Migration Guide](docs/MIGRATION.md)** - Upgrade from v0.0.2

### 🔧 **Production Deployment**
- **[Reliability Usage Guide](docs/RELIABILITY_USAGE_GUIDE.md)** - EARF compliance and monitoring
- **[CLI Documentation](docs/CLI_TOOL_DOCUMENTATION.md)** - Command-line tools
- **[Plugin System](docs/PLUGIN_SYSTEM_DOCUMENTATION.md)** - Extending IA Modules
- **[API Reference](docs/API_REFERENCE.md)** - Detailed API documentation

### 🏗️ **Architecture & Development**
- **[Pipeline Architecture](docs/PIPELINE_ARCHITECTURE.md)** - Core system design
- **[Execution Architecture](docs/EXECUTION_ARCHITECTURE.md)** - How pipelines execute, step loading, tracking explained
- **[Developer Guide](docs/DEVELOPER_GUIDE.md)** - Development workflow
- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute
- **[Testing Guide](docs/TESTING_GUIDE.md)** - Testing strategies

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
from ia_modules.pipeline.core import Step

class GenerateMessageStep(Step):
    async def execute(self, data: dict) -> dict:
        name = data.get("name", "World")
        return {"greeting": f"Hello, {name}!"}

class PrintMessageStep(Step):
    async def execute(self, data: dict) -> dict:
        message = data.get("message")
        print(message)
        return {"status": "Message printed successfully"}
```

4. Create the runner (`main.py`):

```python
# main.py
import asyncio
from ia_modules.pipeline.runner import run_pipeline_from_json
from ia_modules.pipeline.services import ServiceRegistry
from ia_modules.pipeline.core import ExecutionContext

async def main():
    print("Running the pipeline...")

    # Setup services
    services = ServiceRegistry()

    # Create execution context
    ctx = ExecutionContext(
        execution_id='hello-001',
        pipeline_id='hello-world',
        user_id='developer'
    )

    result = await run_pipeline_from_json(
        pipeline_file="pipelines/hello_world.json",
        input_data={"name": "Developer"},
        services=services,
        execution_context=ctx
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

## Built-in Step Types

For most workflows you don't need to subclass `Step` — use the built-ins:

```python
from ia_modules.pipeline import LLMStep, FunctionStep, ParallelStep, A2AStep

# LLM step — prompt in, text out
summarizer = LLMStep(
    name="summarize",
    system_prompt="Summarize the following text concisely.",
    model="claude-sonnet-4-20250514",
)

# Function step — wrap any async callable
async def tally_votes(context):
    votes = context.get_data("votes", [])
    return {"winner": max(set(votes), key=votes.count)}

tally = FunctionStep(name="tally", fn=tally_votes)

# Parallel step — fan out to multiple workers
reviewers = ParallelStep(
    name="all_reviewers",
    children=[reviewer_1, reviewer_2, reviewer_3],
)

# A2A step — dispatch to a remote agent server
remote = A2AStep(
    name="remote_analysis",
    server_url="http://a2a-server:3008",
    mode="research",
    tools=["Read", "Glob", "Grep"],
)
```

## Agent Authentication

For multi-tenant or production deployments, agents authenticate via OIDC JWTs. The auth system is pluggable — use the built-in mini OIDC provider for local dev, or connect to Keycloak/Auth0/Cognito for production.

```python
from ia_modules.agents.auth import get_adapter
from ia_modules.agents.permissions import enforce_agent_claims

# Get configured IDP adapter (AGENT_AUTH_MODE=local or oidc)
adapter = get_adapter(db=my_db)

# Validate a Bearer token and get claims
claims = await adapter.validate_token(token)

# Enforce permissions against the actual execution context before running
enforce_agent_claims(
    claims,
    cwd="/data/apps/app-123/workspace",
    mode="research",
    tools=["Read", "Glob"],
)
```

Set `AGENT_AUTH_MODE=oidc` and configure `OIDC_DISCOVERY_URL` for production OIDC. Default is `local` (built-in mini OIDC — no external IDP needed). The custom `a2a` JWT claim scopes each token to specific working directories, modes, and tool allow-lists, and `enforce_agent_claims()` raises `ClaimsViolation` if the requested action doesn't match the token.

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
├── pipeline/            # Core pipeline execution engine + built-in step types
│   ├── llm_step.py      # LLMStep — prompt → text
│   ├── function_step.py # FunctionStep — wrap any async callable
│   ├── agent_step.py    # AgentStep — run a CLI agent locally
│   ├── a2a_step.py      # A2AStep — dispatch to remote A2A server
│   ├── parallel_step.py # ParallelStep — fan-out
│   └── orchestrator_step.py  # OrchestratorStep — atomic pattern execution
├── agents/              # Agent execution, auth, permissions
│   ├── auth/            # Pluggable OIDC adapters (local, Keycloak)
│   └── permissions.py   # Token claim enforcement (CWD, modes, tools)
├── auth/                # Web authentication and session management
├── database/            # Database abstraction and management
├── web/                 # Web utilities (execution tracking, etc.)
└── data/                # Shared data models
```

## Key Components

#### Pipeline System

- **Step**: The base class for all pipeline steps. Implements the `execute()` method containing business logic.
- **Pipeline**: The orchestrator that executes the graph, manages state, and handles routing.
- **ExecutionContext**: Tracks execution metadata (execution_id, pipeline_id, user_id, thread_id).
- **HumanInputStep**: A specialized step that pauses execution to wait for human interaction.
- **ServiceRegistry**: A dependency injection container for services (DB, HTTP, etc.).

#### Built-in Step Types

Ready-to-use step types so you don't need to subclass `Step` for common patterns:

- **LLMStep**: Send a prompt to a CLI agent, get text back.
- **FunctionStep**: Wrap any async callable (vote tallying, routing, aggregation).
- **AgentStep**: Run a CLI agent (Claude Code SDK, OpenCode) with workspace and tools.
- **A2AStep**: Dispatch work to a remote A2A agent server via JSON-RPC.
- **ParallelStep**: Fan-out — run child steps concurrently.
- **OrchestratorStep**: Run a collaboration pattern (debate, reflection, planning, agentic RAG) as one atomic step.

#### Agent Authentication & Permissions

- **IDPAdapter**: Abstract interface for OIDC identity providers.
- **MiniOIDCAdapter**: Built-in mini OIDC provider for local development (no external IDP needed).
- **KeycloakAdapter**: Production adapter for Keycloak/Auth0/Cognito/any OIDC-compliant IDP.
- **`get_adapter()`**: Factory that switches on `AGENT_AUTH_MODE` env var (`local` or `oidc`).
- **`enforce_agent_claims()`**: Zero-trust gate that validates JWT `a2a` claims against the requested CWD, mode, and tools before execution.
- **`DEFAULT_A2A_PERMISSIONS`**: Sane defaults for research vs execute modes.

#### Web Authentication System

- **AuthMiddleware**: FastAPI-compatible middleware for protecting endpoints.
- **SessionManager**: Manages secure user sessions.

#### Database Layer

- **get_database()**: Factory function for database connections with pluggable backends
- **DatabaseInterface**: Abstract base class defining database operations (connect, execute, fetch_one, fetch_all)
- **NexuSQLAdapter**: Default backend using [nexusql](https://github.com/Fiberwise-AI/nexusql) standalone package
- **SQLAlchemyAdapter**: Alternative backend with connection pooling support
- **Multi-Database Support**: PostgreSQL, MySQL, SQLite, MSSQL
- **Simplified Architecture**: No migration system - use your own migration tools

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
from ia_modules.pipeline.core import ExecutionContext
from ia_modules.database import get_database

# 1. Setup database
db = get_database('postgresql://user:pass@localhost/db')
db.connect()

# 2. Setup application services
services = ServiceRegistry()
services.register('database', db)

# 3. Load pipeline configuration from the database
importer = PipelineImportService(db, pipelines_dir='/path/to/pipelines')
pipeline_row = await importer.get_pipeline_by_slug('ai-notebook-creation-pipeline-v1')
pipeline_config = pipeline_row['pipeline_config'] # This is the parsed JSON

# 4. Create execution context
ctx = ExecutionContext(
    execution_id='exec-001',
    pipeline_id='ai-notebook',
    user_id='user-123'
)

# 5. Create and run the pipeline instance
pipeline = create_pipeline_from_json(pipeline_config, services=services)
result = await pipeline.run({'topic': 'machine learning'}, execution_context=ctx)
```

### Option 2: File-Based Execution (Development)

Ideal for local development, testing, and ad-hoc runs.

```python
from ia_modules.pipeline.runner import run_pipeline_from_json
from ia_modules.pipeline.services import ServiceRegistry
from ia_modules.pipeline.core import ExecutionContext
from ia_modules.database import get_database

# 1. Setup database (or use mock for testing)
db = get_database('sqlite:///test.db')
db.connect()

# 2. Setup services
services = ServiceRegistry()
services.register('database', db)

# 3. Create execution context
ctx = ExecutionContext(
    execution_id='dev-001',
    pipeline_id='ai-notebook',
    user_id='developer'
)

# 4. Run directly from the JSON file
result = await run_pipeline_from_json(
    pipeline_file="pipelines/ai_notebook_creation_pipeline.json",
    input_data={"topic": "machine learning"},
    services=services,
    execution_context=ctx
)

# Cleanup
db.disconnect()
```

## AI/LLM Integration

LLM calls happen through the built-in step types — use `LLMStep` for prompt-in/text-out and `AgentStep` for full CLI agents (Claude Code SDK, OpenCode) with workspace and tool constraints. See [Built-in Step Types](#built-in-step-types) above.

### Provider Configuration

`LLMProviderService` is a lightweight registry that maps `provider_id → api_key / model` so steps can look up credentials without hardcoding them. Register it once in your `ServiceRegistry` and any `LLMStep` / `AgentStep` can resolve its provider by id.

```python
from ia_modules.pipeline.llm_provider_service import LLMProviderService
from ia_modules.pipeline.services import ServiceRegistry
import os

provider_service = LLMProviderService()
provider_service.register_provider("openai", model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
provider_service.register_provider("anthropic", model="claude-sonnet-4-5-20250929", api_key=os.getenv("ANTHROPIC_API_KEY"))

services = ServiceRegistry()
services.register("llm_provider", provider_service)
```

Steps reference the provider by id in their config — the service layer is *only* for key/model lookup, not for making completion calls. Actual inference happens inside `LLMStep` / `AgentStep` via the configured CLI agent.

### Environment Variables

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
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

## Parallel Processing

The framework automatically executes steps in parallel when they have no dependencies on each other.

### Automatic Parallelization

```json
{
  "flow": {
    "start_at": "data_loader",
    "transitions": [
      {"from": "data_loader", "to": "processor_1"},
      {"from": "data_loader", "to": "processor_2"}, 
      {"from": "data_loader", "to": "processor_3"},
      {"from": "processor_1", "to": "merger"},
      {"from": "processor_2", "to": "merger"},
      {"from": "processor_3", "to": "merger"}
    ]
  }
}
```

**Execution Flow:**
1. `data_loader` runs first
2. `processor_1`, `processor_2`, `processor_3` run **simultaneously** 
3. `merger` waits for all three processors to complete

### Performance Benefits

- **Faster I/O operations**: Multiple API calls or database queries run concurrently instead of sequentially
- **Reduced total runtime**: Three 1-second API calls complete in ~1 second instead of 3 seconds
- **Better resource usage**: While one step waits for I/O, others can execute

## Advanced Topics

### The Pipeline Importer Service

In a production environment, pipelines are loaded from the filesystem into a database table for fast and reliable access. The `PipelineImportService` handles this process.

- **Location**: `ia_modules/pipeline/importer.py`
- **Purpose**: Scans a directory for `*.json` files, validates them, and upserts them into the `pipelines` database table.
- **Change Detection**: It computes a hash of the file content to avoid redundant database writes if a pipeline definition hasn't changed.

The consuming application typically calls `importer.import_all_pipelines()` on startup.

### Database Setup

IA Modules uses [nexusql](https://github.com/Fiberwise-AI/nexusql) for database operations. nexusql provides two backend options:
- **NexusQL backend** (default): Lightweight, built-in SQL execution
- **SQLAlchemy backend**: Connection pooling for production workloadse

**Basic setup with SQLite:**

```python
from ia_modules.database import get_database

# Create database connection
db = get_database('sqlite:///app.db')

# Run migrations to create tables
await db.initialize(
    apply_schema=True,
    app_migration_paths=["database/migrations"]
)
```

**Production setup with PostgreSQL:**

```python
# Use SQLAlchemy backend for connection pooling
db = get_database(
    'postgresql://user:pass@localhost/db',
    backend='sqlalchemy',
    pool_size=10,
    max_overflow=20
)

# Run migrations
await db.initialize(
    apply_schema=True,
    app_migration_paths=["database/migrations"]
)
```

**About migrations**: nexusql includes a migration runner that executes `V001__description.sql` files from `database/migrations/`. It automatically translates SQL syntax for your target database (PostgreSQL, MySQL, SQLite, etc.). See [nexusql docs](https://github.com/Fiberwise-AI/nexusql) for details.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Quick Start:**
```bash
git clone <repository-url>
cd ia_modules
pip install -e ".[dev,all]"
pytest tests/ -v
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

- **Documentation**: [docs/](docs/) and [showcase_app/](showcase_app/)
- **Issues**: [GitHub Issues](https://github.com/Fiberwise-AI/ia_modules/issues)
- **Examples**: [tests/pipelines/](tests/pipelines/) and [showcase_app/](showcase_app/)
- **Live Demo**: Run `showcase_app` locally to explore all features

---

**Built with ❤️ for production AI systems**

*Experience it live: Start the [Showcase App](#quick-start) in 2 minutes!*