# IA Modules - Pipeline Framework

## Overview

A Python package for building graph-based pipelines with conditional routing, service injection, authentication, and database management.

## Core Architecture Principles

### 1. **Graph-First Design**
- All pipelines are defined as **directed acyclic graphs (DAGs)**
- Conditional branching based on step outputs

### 2. **Intelligent Flow Control**
- **Expression-based routing**: Simple logical conditions
- **AI-driven routing**: Agent-based decision making (future)
- **Function-based routing**: Custom business logic (future)

### 3. **Service Injection & Dependency Management**
- Preserved from Architecture 1.0
- Clean `ServiceRegistry` pattern
- Database and HTTP service access via `self.get_db()`, `self.get_http()`

## Package Structure

```
ia_modules/
├── pipeline/            # Core pipeline execution engine
│   ├── core.py         # Pipeline, Step, HumanInputStep, TemplateParameterResolver
│   ├── services.py     # ServiceRegistry, CentralLoggingService
│   ├── runner.py       # JSON pipeline loader and execution
│   ├── routing.py      # Conditional routing logic
│   └── pipeline_models.py  # Data models for pipeline execution
├── auth/               # Authentication and session management
│   ├── middleware.py   # FastAPI authentication middleware
│   ├── session.py      # SessionManager
│   ├── models.py       # User models and roles
│   └── security.py     # Token generation and security
├── database/           # Database abstraction and management
│   ├── manager.py      # DatabaseManager with multi-database support
│   ├── migrations.py   # Database migration system
│   ├── interfaces.py   # Database connection interfaces
│   └── providers.py    # Database provider implementations
├── web/                # Web utilities
│   ├── database.py     # Web-optimized database operations
│   └── execution_tracker.py  # Pipeline execution tracking
└── data/               # Data models and compatibility shims
    └── pipeline_models.py  # Pipeline model re-exports
```

## Key Components

### Pipeline System
- **Step Class**: Base class for pipeline steps with service injection
- **Pipeline Class**: Graph-based pipeline orchestrator with conditional routing
- **HumanInputStep**: Base class for human-in-the-loop interactions
- **TemplateParameterResolver**: Dynamic parameter substitution system
- **ServiceRegistry**: Dependency injection for database and HTTP services

### Authentication System
- **AuthMiddleware**: FastAPI-compatible authentication middleware
- **SessionManager**: Secure session handling and user state management
- **User Models**: Type-safe user data structures and role management
- **Security**: Token generation and session cookie management

### Database Layer
- **DatabaseManager**: Multi-database connection and transaction management
- **Migration System**: Schema versioning and automated database updates
- **Provider Interfaces**: Abstract database operations for different providers
- **Connection Management**: Type-safe database configuration and pooling

## Usage Examples

### Basic Step Implementation

```python
from ia_modules.pipeline import Step

class MyStep(Step):
    async def work(self, data: Dict[str, Any]) -> Any:
        # Access injected services
        db = self.get_db()
        http = self.get_http()

        # Your business logic here
        result = await process_data(data)
        return {"processed": result}
```

### Pipeline Execution

```python
from ia_modules.pipeline import run_pipeline_from_json

# Execute pipeline from JSON configuration
result = await run_pipeline_from_json(
    pipeline_config,
    input_data,
    services
)
```

## Graph Pipeline Definition Format

### Enhanced JSON Schema

```json
{
  "name": "Advanced Travel Pipeline",
  "description": "Graph-based pipeline with conditional flows",
  "version": "2.0",
  "steps": [
    {
      "id": "step_id",                    // Required: unique step identifier
      "name": "Human Readable Name",      // Required: display name
      "type": "task",                     // Required: task|human_input
      "step_class": "StepClassName",      // Required: Python class name
      "module": "module.path",            // Required: import path
      "config": {...},                    // Optional: step configuration
      "input_schema": {...},              // Optional: JSON schema validation
      "output_schema": {...}              // Optional: output contract
    }
  ],
  "flow": {
    "start_at": "first_step_id",          // Required: entry point
    "paths": [
      {
        "from": "step_a",                 // Required: source step
        "to": "step_b",                   // Required: target step
        "condition": {                    // Required: routing condition
          "type": "expression",           // expression|always|agent|function
          "config": {
            "source": "result.score",     // Data path to evaluate
            "operator": "greater_than",   // Comparison operator
            "value": 0.7                  // Threshold value
          }
        }
      }
    ]
  }
}
```

### Supported Condition Types

#### 1. Always Condition
```json
{"type": "always"}
```
- Always takes this path (fallback/default)

#### 2. Expression Condition
```json
TODO: Create a doocument of all possible expressions IA Pipelines can/should perform.
{
  "type": "expression",
  "config": {
    "source": "result.quality_score",
    "operator": "greater_than_or_equal",
    "value": 75
  }
}
```

**Supported Operators:**
- `equals`, `greater_than`, `greater_than_or_equal`
- `less_than`, `equals_ignore_case`

#### 3. Future: Agent Condition (AI-Driven Routing)
```json
{
  "type": "agent",
  "config": {
    "agent_id": "quality_assessor_agent",
    "input": {"prompt": "Assess content quality: {result}"},
    "output": {"evaluation": {"operator": "equals", "value": "high_quality"}}
  }
}
```

#### 4. Future: Function Condition (Custom Business Logic)
```json
{
  "type": "function",
  "config": {
    "function_class": "BusinessRulesValidator",
    "function_method": "validate_lead_quality",
    "expected_result": true
  }
}
```

## Human-in-the-Loop (HITL) Architecture

### HITL Step Definition

```json
{
  "id": "human_approval",
  "name": "Human Review",
  "type": "human_input",
  "step_class": "HumanInputStep",
  "config": {
    "ui_schema": {
      "title": "Review Content",
      "fields": [
        {"name": "decision", "type": "radio", "options": ["Approve", "Reject"]},
        {"name": "notes", "type": "textarea"}
      ]
    }
  }
}
```

### HITL Base Implementation

```python
class HumanInputStep(Step):
    """Base class for human-in-the-loop steps"""

    async def work(self, data: Dict[str, Any]) -> Any:
        ui_schema = self.config.get('ui_schema', {})

        # Base implementation - override with specific HITL integration
        return {
            'human_input_required': True,
            'ui_schema': ui_schema,
            'message': f'Human input step {self.name}'
        }
```

## Real-World Examples

### Example 1: Travel Content Pipeline

**Intelligent Flow Logic:**
1. **High-confidence location** → Detailed POI enrichment → Advanced story
2. **Low-confidence location** → Basic POI enrichment → Simple summary

```json
{
  "flow": {
    "start_at": "location_parser",
    "paths": [
      {
        "from": "osm_fetcher",
        "to": "poi_enricher_detailed",
        "condition": {
          "type": "expression",
          "config": {"source": "result.poi_potential", "operator": "greater_than", "value": 0.7}
        }
      },
      {
        "from": "osm_fetcher",
        "to": "poi_enricher_basic",
        "condition": {"type": "always"}
      }
    ]
  }
}
```

### Example 2: Lead Processing Pipeline

**Quality-Based Routing:**
1. **High-quality leads** → Human review → CRM integration
2. **Low-quality leads** → Automated notification only

```json
{
  "flow": {
    "paths": [
      {
        "from": "lead_scorer",
        "to": "human_review",
        "condition": {
          "type": "expression",
          "config": {"source": "result.score", "operator": "greater_than_or_equal", "value": 75}
        }
      },
      {
        "from": "lead_scorer",
        "to": "automated_notification",
        "condition": {"type": "always"}
      }
    ]
  }
}
```

## Installation

```bash
# Install in development mode
pip install -e .

# Or install from source
pip install git+<repository-url>
```

## Configuration

The package supports environment-based configuration and includes built-in database migration support.

```python
from ia_modules.database import DatabaseManager
from ia_modules.auth import AuthMiddleware

# Initialize database
db_manager = DatabaseManager("sqlite:///app.db")
await db_manager.run_migrations()

# Set up authentication
auth = AuthMiddleware(secret_key="your-secret-key")
```