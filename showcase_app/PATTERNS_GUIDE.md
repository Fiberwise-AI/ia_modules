# Agentic Design Patterns Guide

This guide covers three AI agent patterns implemented as pipeline steps for the ia_modules framework.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Reflection Pattern](#reflection-pattern)
- [Planning Pattern](#planning-pattern)
- [Tool Use Pattern](#tool-use-pattern)
- [Creating Pipelines](#creating-pipelines)
- [API Reference](#api-reference)

## Overview

The ia_modules framework includes three fundamental AI agent patterns:

| Pattern | Purpose | Research |
|---------|---------|----------|
| **Reflection** | Self-critique and iterative improvement | [Bai et al., 2022](https://arxiv.org/abs/2212.08073) |
| **Planning** | Multi-step goal decomposition | [Wei et al., 2022](https://arxiv.org/abs/2201.11903) |
| **Tool Use** | Dynamic capability selection | [Schick et al., 2023](https://arxiv.org/abs/2302.04761) |

All patterns are implemented as `Step` subclasses and integrate seamlessly with the pipeline system.

## Quick Start

### Prerequisites

```bash
# Set at least one LLM API key
export OPENAI_API_KEY=sk-...
# OR
export ANTHROPIC_API_KEY=sk-ant-...
# OR
export GEMINI_API_KEY=...
```

### Run Example

```bash
cd ia_modules
python -m ia_modules.pipeline.graph_pipeline_runner \
  showcase_app/backend/pipelines/agentic_patterns_demo.json
```

### Run Tests

```bash
cd showcase_app
pytest tests/test_pattern_steps.py -v
```

## Reflection Pattern

### Description

The Reflection pattern implements iterative improvement through self-critique. The agent generates output, evaluates it against defined criteria, and refines until quality thresholds are met.

### Configuration

```python
{
  "initial_output": str,      # Text to improve
  "criteria": {               # Quality criteria
    "name": "description"
  },
  "max_iterations": int       # Maximum refinement cycles
}
```

### Example

```json
{
  "id": "improve_text",
  "name": "Improve Content",
  "step_class": "ReflectionStep",
  "module": "showcase_app.backend.pipelines.pattern_steps",
  "config": {
    "initial_output": "Draft article text...",
    "criteria": {
      "clarity": "Text should be clear and concise",
      "accuracy": "Facts must be verifiable",
      "completeness": "Cover all key points"
    },
    "max_iterations": 3
  }
}
```

### Output

```python
{
  "final_output": str,        # Improved text
  "final_score": float,       # Quality score (0-10)
  "iterations": list,         # Iteration history
  "total_iterations": int,    # Number of iterations
  "improved": bool            # Whether quality improved
}
```

### Python Usage

```python
from showcase_app.backend.pipelines.pattern_steps import ReflectionStep

step = ReflectionStep(
    name="improve",
    config={
        "initial_output": "Draft text",
        "criteria": {"clarity": "Must be clear"},
        "max_iterations": 3
    }
)

result = await step.run({})
print(result["final_output"])
```

## Planning Pattern

### Description

The Planning pattern breaks down complex goals into structured, executable sub-tasks with dependencies and expected outcomes.

### Configuration

```python
{
  "goal": str,               # Objective to achieve
  "constraints": list,       # Limitations/requirements
  "context": dict           # Additional information
}
```

### Example

```json
{
  "id": "create_plan",
  "name": "Create Research Plan",
  "step_class": "PlanningStep",
  "module": "showcase_app.backend.pipelines.pattern_steps",
  "config": {
    "goal": "Research renewable energy solutions",
    "constraints": [
      "Budget under $20,000",
      "Focus on solar power",
      "Complete within 2 weeks"
    ],
    "context": {
      "location": "Northeast US",
      "property_size": "2000 sq ft"
    }
  }
}
```

### Output

```python
{
  "goal": str,                # Original goal
  "plan": list,               # Structured steps
  "is_valid": bool,           # Plan validation result
  "validation_feedback": str, # Validation commentary
  "total_steps": int,         # Number of steps
  "estimated_time": str       # Time estimate
}
```

Each plan step includes:
```python
{
  "step_number": int,
  "action": str,
  "expected_outcome": str,
  "dependencies": list
}
```

### Python Usage

```python
from showcase_app.backend.pipelines.pattern_steps import PlanningStep

step = PlanningStep(
    name="plan",
    config={
        "goal": "Launch product feature",
        "constraints": ["3-month timeline", "Team of 5"]
    }
)

result = await step.run({})
for step_item in result["plan"]:
    print(f"{step_item['step_number']}: {step_item['action']}")
```

## Tool Use Pattern

### Description

The Tool Use pattern analyzes task requirements and dynamically selects appropriate tools for execution.

### Configuration

```python
{
  "task": str,              # Task description
  "available_tools": list   # Available tool names
}
```

### Example

```json
{
  "id": "solve_problem",
  "name": "Solve with Tools",
  "step_class": "ToolUseStep",
  "module": "showcase_app.backend.pipelines.pattern_steps",
  "config": {
    "task": "Calculate compound interest for $1000 at 5% over 10 years",
    "available_tools": [
      "calculator",
      "search",
      "code_executor"
    ]
  }
}
```

### Output

```python
{
  "task": str,              # Original task
  "selected_tools": list,   # Tools selected
  "tool_results": list,     # Results from each tool
  "final_answer": str,      # Synthesized answer
  "tools_used": int         # Number of tools used
}
```

### Python Usage

```python
from showcase_app.backend.pipelines.pattern_steps import ToolUseStep

step = ToolUseStep(
    name="solve",
    config={
        "task": "Calculate ROI",
        "available_tools": ["calculator", "search"]
    }
)

result = await step.run({})
print(result["final_answer"])
```

## Creating Pipelines

### Basic Structure

```json
{
  "name": "pipeline_name",
  "version": "1.0.0",
  "steps": [
    {
      "id": "unique_id",
      "name": "Display Name",
      "step_class": "PatternStepClass",
      "module": "showcase_app.backend.pipelines.pattern_steps",
      "config": {}
    }
  ],
  "flow": {
    "start_at": "unique_id",
    "paths": [
      {
        "from": "source_id",
        "to": "target_id",
        "condition": {"type": "always"}
      }
    ]
  }
}
```

### Multi-Pattern Pipeline

```json
{
  "name": "content_creation",
  "version": "1.0.0",
  "steps": [
    {
      "id": "plan",
      "name": "Plan Content",
      "step_class": "PlanningStep",
      "module": "showcase_app.backend.pipelines.pattern_steps",
      "config": {
        "goal": "Write article about AI",
        "constraints": ["1000 words", "Beginner audience"]
      }
    },
    {
      "id": "research",
      "name": "Research Topics",
      "step_class": "ToolUseStep",
      "module": "showcase_app.backend.pipelines.pattern_steps",
      "config": {
        "task": "Find latest AI developments",
        "available_tools": ["search"]
      }
    },
    {
      "id": "refine",
      "name": "Refine Quality",
      "step_class": "ReflectionStep",
      "module": "showcase_app.backend.pipelines.pattern_steps",
      "config": {
        "initial_output": "Draft text...",
        "criteria": {
          "accuracy": "Facts must be correct",
          "clarity": "Easy to understand"
        },
        "max_iterations": 3
      }
    }
  ],
  "flow": {
    "start_at": "plan",
    "paths": [
      {
        "from": "plan",
        "to": "research",
        "condition": {"type": "always"}
      },
      {
        "from": "research",
        "to": "refine",
        "condition": {"type": "always"}
      },
      {
        "from": "refine",
        "to": "end_with_success",
        "condition": {"type": "always"}
      }
    ]
  }
}
```

### Running Pipelines

```bash
# From file
python -m ia_modules.pipeline.graph_pipeline_runner your_pipeline.json

# With input data
echo '{"topic": "AI"}' | python -m ia_modules.pipeline.graph_pipeline_runner your_pipeline.json
```

### Programmatic Execution

```python
from ia_modules.pipeline.graph_pipeline_runner import run_graph_pipeline_from_file

result = await run_graph_pipeline_from_file(
    pipeline_file="content_creation.json",
    input_data={}
)

print(result["plan"]["plan"])
print(result["research"]["final_answer"])
print(result["refine"]["final_output"])
```

## API Reference

### ReflectionStep

```python
class ReflectionStep(Step):
    def __init__(self, name: str, config: dict):
        """
        Args:
            name: Step name
            config: {
                "initial_output": str,
                "criteria": dict,
                "max_iterations": int
            }
        """

    async def run(self, data: dict) -> dict:
        """
        Returns:
            {
                "final_output": str,
                "final_score": float,
                "iterations": list,
                "total_iterations": int,
                "improved": bool
            }
        """
```

### PlanningStep

```python
class PlanningStep(Step):
    def __init__(self, name: str, config: dict):
        """
        Args:
            name: Step name
            config: {
                "goal": str,
                "constraints": list,
                "context": dict
            }
        """

    async def run(self, data: dict) -> dict:
        """
        Returns:
            {
                "goal": str,
                "plan": list,
                "is_valid": bool,
                "validation_feedback": str,
                "total_steps": int,
                "estimated_time": str
            }
        """
```

### ToolUseStep

```python
class ToolUseStep(Step):
    def __init__(self, name: str, config: dict):
        """
        Args:
            name: Step name
            config: {
                "task": str,
                "available_tools": list
            }
        """

    async def run(self, data: dict) -> dict:
        """
        Returns:
            {
                "task": str,
                "selected_tools": list,
                "tool_results": list,
                "final_answer": str,
                "tools_used": int
            }
        """
```

## Files

- **Implementation**: `backend/pipelines/pattern_steps.py`
- **Example Pipeline**: `backend/pipelines/agentic_patterns_demo.json`
- **Tests**: `tests/test_pattern_steps.py`
- **Quick Reference**: `QUICK_REFERENCE.md`
