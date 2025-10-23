# IA Modules Pipeline & Agent Guide for Showcase App

This guide explains how to use IA Modules pipelines and agents in the showcase app. Focus on **pipeline patterns**, **agent orchestration**, and **best practices** - not the FastAPI/React implementation details.

---

## Core Concepts

### What is a Pipeline?

A **pipeline** is a sequence of steps that process data. Each step:
- Takes input data
- Performs work (LLM calls, data transforms, API calls, etc.)
- Returns output data
- Can fail and retry automatically
- Has its state saved (checkpoints)

**Think of it like a Unix pipeline:**
```bash
cat data.txt | grep "error" | sort | uniq -c
```

But for AI/LLM workflows:
```
UserInput → ExtractInfo → GenerateCode → Review → Deploy
```

### What is an Agent?

An **agent** is an autonomous component that makes decisions and executes tools. Agents:
- Have a specific role (e.g., "Planner", "Executor", "Reviewer")
- Can call tools (search, calculator, API calls)
- Make decisions based on context
- Can be orchestrated together in workflows

**Think of agents as specialized workers:**
- **Planner Agent**: Breaks down tasks
- **Researcher Agent**: Searches and gathers info
- **Coder Agent**: Writes code
- **Reviewer Agent**: Checks quality

---

## Pipeline Patterns

### 1. Linear Pipeline (Sequential Steps)

**Use Case:** Simple workflows where each step depends on the previous one.

```json
{
  "name": "data_processing",
  "steps": [
    {
      "name": "load_data",
      "module": "steps.load_step",
      "class": "LoadDataStep"
    },
    {
      "name": "validate",
      "module": "steps.validate_step",
      "class": "ValidateStep"
    },
    {
      "name": "transform",
      "module": "steps.transform_step",
      "class": "TransformStep"
    },
    {
      "name": "export",
      "module": "steps.export_step",
      "class": "ExportStep"
    }
  ],
  "flow": {
    "start_at": "load_data",
    "transitions": [
      {"from": "load_data", "to": "validate"},
      {"from": "validate", "to": "transform"},
      {"from": "transform", "to": "export"}
    ]
  }
}
```

**Step Implementation:**
```python
from ia_modules.pipeline import Step

class LoadDataStep(Step):
    async def execute(self, input_data):
        # Load data from source
        data = await self.load_from_source(input_data["source"])

        return {
            "data": data,
            "row_count": len(data)
        }
```

### 2. Conditional Pipeline (Branching Logic)

**Use Case:** Different paths based on conditions (e.g., data quality, user choice).

```json
{
  "name": "content_moderation",
  "steps": [
    {
      "name": "check_content",
      "module": "steps.moderation_step",
      "class": "ModerationStep"
    },
    {
      "name": "auto_approve",
      "module": "steps.approve_step",
      "class": "ApproveStep"
    },
    {
      "name": "human_review",
      "module": "steps.human_review_step",
      "class": "HumanReviewStep"
    },
    {
      "name": "publish",
      "module": "steps.publish_step",
      "class": "PublishStep"
    }
  ],
  "flow": {
    "start_at": "check_content",
    "transitions": [
      {
        "from": "check_content",
        "to": "auto_approve",
        "condition": {
          "type": "field_equals",
          "field": "confidence",
          "value": "high"
        }
      },
      {
        "from": "check_content",
        "to": "human_review",
        "condition": {
          "type": "field_equals",
          "field": "confidence",
          "value": "low"
        }
      },
      {"from": "auto_approve", "to": "publish"},
      {"from": "human_review", "to": "publish"}
    ]
  }
}
```

**Condition Functions:**
```python
# Built-in conditions in IA Modules
from ia_modules.pipeline.condition_functions import (
    business_hours_condition,
    threshold_condition,
    data_quality_condition
)

# Custom condition
def content_safe_condition(state, config):
    """Check if content is safe to publish"""
    toxicity_score = state.get("toxicity_score", 0)
    return toxicity_score < config.get("threshold", 0.3)
```

### 3. Parallel Pipeline (Concurrent Execution)

**Use Case:** Independent tasks that can run simultaneously.

```json
{
  "name": "multi_source_research",
  "steps": [
    {
      "name": "search_web",
      "module": "steps.web_search_step",
      "class": "WebSearchStep",
      "config": {"source": "web"}
    },
    {
      "name": "search_database",
      "module": "steps.db_search_step",
      "class": "DatabaseSearchStep",
      "config": {"source": "internal_db"}
    },
    {
      "name": "search_papers",
      "module": "steps.academic_search_step",
      "class": "AcademicSearchStep",
      "config": {"source": "arxiv"}
    },
    {
      "name": "aggregate_results",
      "module": "steps.aggregate_step",
      "class": "AggregateStep"
    }
  ],
  "flow": {
    "start_at": "search_web",
    "parallel": [
      ["search_web", "search_database", "search_papers"]
    ],
    "transitions": [
      {"from": "search_web", "to": "aggregate_results"},
      {"from": "search_database", "to": "aggregate_results"},
      {"from": "search_papers", "to": "aggregate_results"}
    ]
  }
}
```

### 4. Loop Pipeline (Iterative Processing)

**Use Case:** Retry until success, iterative refinement, batch processing.

```json
{
  "name": "iterative_refinement",
  "steps": [
    {
      "name": "generate_content",
      "module": "steps.generate_step",
      "class": "GenerateStep"
    },
    {
      "name": "review_quality",
      "module": "steps.quality_check_step",
      "class": "QualityCheckStep"
    },
    {
      "name": "refine",
      "module": "steps.refine_step",
      "class": "RefineStep"
    },
    {
      "name": "finalize",
      "module": "steps.finalize_step",
      "class": "FinalizeStep"
    }
  ],
  "flow": {
    "start_at": "generate_content",
    "transitions": [
      {"from": "generate_content", "to": "review_quality"},
      {
        "from": "review_quality",
        "to": "refine",
        "condition": {
          "type": "field_equals",
          "field": "quality_passed",
          "value": false
        }
      },
      {
        "from": "review_quality",
        "to": "finalize",
        "condition": {
          "type": "field_equals",
          "field": "quality_passed",
          "value": true
        }
      },
      {"from": "refine", "to": "generate_content"}
    ],
    "loops": [
      {
        "steps": ["generate_content", "review_quality", "refine"],
        "max_iterations": 3,
        "exit_condition": {
          "type": "field_equals",
          "field": "quality_passed",
          "value": true
        }
      }
    ]
  }
}
```

### 5. Human-in-the-Loop Pipeline (HITL)

**Use Case:** Require human approval, editing, or decision at specific points.

```json
{
  "name": "content_creation_hitl",
  "steps": [
    {
      "name": "generate_draft",
      "module": "steps.generate_step",
      "class": "GenerateDraftStep"
    },
    {
      "name": "human_review",
      "module": "ia_modules.pipeline.hitl",
      "class": "HumanReviewStep",
      "config": {
        "review_type": "content_approval",
        "allow_edit": true,
        "timeout_seconds": 3600
      }
    },
    {
      "name": "publish",
      "module": "steps.publish_step",
      "class": "PublishStep"
    }
  ],
  "flow": {
    "start_at": "generate_draft",
    "transitions": [
      {"from": "generate_draft", "to": "human_review"},
      {
        "from": "human_review",
        "to": "publish",
        "condition": {
          "type": "field_equals",
          "field": "approval_status",
          "value": "approved"
        }
      },
      {
        "from": "human_review",
        "to": "generate_draft",
        "condition": {
          "type": "field_equals",
          "field": "approval_status",
          "value": "rejected"
        }
      }
    ]
  }
}
```

**HITL Step Implementation:**
```python
from ia_modules.pipeline.hitl import HumanReviewStep

class ContentApprovalStep(HumanReviewStep):
    async def execute(self, input_data):
        # Present content to human
        content = input_data["generated_content"]

        # Request human input (pipeline pauses here)
        response = await self.request_human_input({
            "type": "approval",
            "content": content,
            "question": "Approve this content for publishing?",
            "options": ["approve", "reject", "edit"]
        })

        if response["action"] == "edit":
            # Human edited the content
            content = response["edited_content"]

        return {
            "approval_status": response["action"],
            "final_content": content,
            "reviewer": response["user_id"]
        }
```

---

## Agent Orchestration Patterns

### 1. Single Agent Pipeline

**Use Case:** One agent handles the entire workflow.

```python
from ia_modules.agents import BaseAgent, AgentRole

# Define agent role
planner_role = AgentRole(
    name="Planner",
    description="Plans and breaks down tasks",
    capabilities=["task_breakdown", "prioritization"],
    tools=["search", "calculator"]
)

# Create agent
planner = BaseAgent(
    role=planner_role,
    llm_config={"model": "gpt-4"}
)

# Use in pipeline step
class PlanningStep(Step):
    async def execute(self, input_data):
        # Agent processes the task
        result = await planner.execute({
            "task": "Break down building a web app",
            "context": input_data
        })

        return {
            "subtasks": result["subtasks"],
            "timeline": result["timeline"]
        }
```

### 2. Multi-Agent Collaboration

**Use Case:** Multiple agents work together, each with specialized skills.

```python
from ia_modules.agents import AgentOrchestrator, BaseAgent, AgentRole

# Define roles
researcher_role = AgentRole(
    name="Researcher",
    description="Gathers and analyzes information",
    capabilities=["web_search", "data_analysis"],
    tools=["search", "scraper", "analyzer"]
)

writer_role = AgentRole(
    name="Writer",
    description="Creates clear, engaging content",
    capabilities=["content_creation", "editing"],
    tools=["grammar_check", "style_guide"]
)

reviewer_role = AgentRole(
    name="Reviewer",
    description="Ensures quality and accuracy",
    capabilities=["fact_checking", "quality_assurance"],
    tools=["fact_checker", "plagiarism_detector"]
)

# Create agents
researcher = BaseAgent(role=researcher_role, llm_config={"model": "gpt-4"})
writer = BaseAgent(role=writer_role, llm_config={"model": "gpt-4"})
reviewer = BaseAgent(role=reviewer_role, llm_config={"model": "gpt-4"})

# Orchestrate agents
orchestrator = AgentOrchestrator()
orchestrator.add_agent("researcher", researcher)
orchestrator.add_agent("writer", writer)
orchestrator.add_agent("reviewer", reviewer)

# Define workflow
orchestrator.add_edge("researcher", "writer")
orchestrator.add_edge("writer", "reviewer")
orchestrator.add_edge("reviewer", "writer", condition=lambda state: not state.get("approved"))

# Execute in pipeline
class ContentCreationStep(Step):
    async def execute(self, input_data):
        # Run agent orchestration
        result = await orchestrator.run(
            start_agent="researcher",
            initial_state={"topic": input_data["topic"]},
            max_steps=10
        )

        return {
            "research": result["research"],
            "content": result["content"],
            "review_status": result["approved"]
        }
```

### 3. Agent with Tools

**Use Case:** Agent needs to call external services/APIs.

```python
from ia_modules.agents import BaseAgent
from ia_modules.tools import Tool, ToolRegistry

# Define tools
@Tool(
    name="web_search",
    description="Search the web for information",
    parameters={
        "query": {"type": "string", "description": "Search query"}
    }
)
async def web_search(query: str):
    # Call search API
    results = await search_api.search(query)
    return {"results": results}

@Tool(
    name="calculator",
    description="Perform mathematical calculations",
    parameters={
        "expression": {"type": "string", "description": "Math expression"}
    }
)
async def calculator(expression: str):
    result = eval(expression)  # Use safe eval in production!
    return {"result": result}

# Register tools
registry = ToolRegistry()
registry.register(web_search)
registry.register(calculator)

# Create agent with tools
agent = BaseAgent(
    role=researcher_role,
    tools=registry.get_all(),
    llm_config={"model": "gpt-4"}
)

# Agent will automatically call tools when needed
result = await agent.execute({
    "task": "Find the population of NYC and calculate density per sq mile"
})
```

### 4. Agent with Memory

**Use Case:** Agent needs to remember previous interactions.

```python
from ia_modules.agents import BaseAgent
from ia_modules.memory import MemoryConversationMemory

# Create memory
memory = MemoryConversationMemory()
# Create agent with memory
agent = BaseAgent(
    role=assistant_role,
    memory=memory,
    llm_config={"model": "gpt-4"}
)

# Use in pipeline
class ConversationalStep(Step):
    async def execute(self, input_data):
        thread_id = input_data.get("thread_id", "default")
        user_message = input_data["message"]

        # Add user message to memory
        await agent.memory.add_message(
            thread_id=thread_id,
            role="user",
            content=user_message
        )

        # Get conversation history
        history = await agent.memory.get_messages(thread_id, limit=10)

        # Agent generates response with context
        response = await agent.execute({
            "message": user_message,
            "history": history
        })

        # Save assistant response
        await agent.memory.add_message(
            thread_id=thread_id,
            role="assistant",
            content=response["content"]
        )

        return {
            "response": response["content"],
            "thread_id": thread_id
        }
```

---

## Best Practices

### 1. State Management

**Keep state clean and well-structured:**

```python
class MyStep(Step):
    async def execute(self, input_data):
        # Good: Clear state structure
        return {
            "status": "success",
            "data": {
                "processed_items": 42,
                "failed_items": 3
            },
            "metadata": {
                "processing_time_ms": 1500,
                "model_used": "gpt-4"
            }
        }

        # Bad: Flat, unclear structure
        return {
            "result": "done",
            "count": 42,
            "failures": 3,
            "time": 1500
        }
```

### 2. Error Handling

**Use compensation logic for cleanup:**

```python
class DatabaseWriteStep(Step):
    async def execute(self, input_data):
        try:
            # Write to database
            record_id = await db.insert(input_data["data"])

            return {
                "status": "success",
                "record_id": record_id
            }
        except Exception as e:
            # Step failed - will trigger compensation
            raise

    async def compensate(self, input_data, original_output):
        """Called if later steps fail"""
        # Undo the database write
        if original_output and "record_id" in original_output:
            await db.delete(original_output["record_id"])

        return {"compensated": True}
```

### 3. Retry Strategy

**Configure retry for transient failures:**

```json
{
  "name": "api_call_step",
  "module": "steps.api_step",
  "class": "APICallStep",
  "config": {
    "retry_config": {
      "max_retries": 3,
      "backoff_factor": 2,
      "retry_on": ["APIError", "NetworkError"],
      "timeout": 30
    }
  }
}
```

```python
from ia_modules.pipeline.retry import with_retry, RetryConfig

class APICallStep(Step):
    @with_retry(RetryConfig(
        max_retries=3,
        backoff_factor=2,
        retry_on=[APIError, NetworkError]
    ))
    async def execute(self, input_data):
        # This will automatically retry on failure
        response = await api.call(input_data["endpoint"])
        return {"data": response}
```

### 4. Checkpointing

**Enable checkpoints for long-running pipelines:**

```python
from ia_modules.checkpoint import SQLCheckpointer
from ia_modules.database import DatabaseManager

# Create checkpointer
db = DatabaseManager("postgresql://...")
await db.connect()
checkpointer = SQLCheckpointer(db)


# Run pipeline with checkpointing
from ia_modules.pipeline import GraphPipelineRunner

runner = GraphPipelineRunner(
    pipeline_config=pipeline_config,
    checkpointer=checkpointer
)

# Execute (can resume if interrupted)
result = await runner.run(
    input_data=input_data,
    thread_id="user-123-session-456"
)

# Later: Resume from checkpoint
result = await runner.resume(thread_id="user-123-session-456")
```

### 5. Metrics & Monitoring

**Track reliability metrics:**

```python
from ia_modules.reliability import ReliabilityMetrics, SQLMetricStorage

# Create storage
storage = SQLMetricStorage(db_manager)
# Create metrics tracker
metrics = ReliabilityMetrics(storage)

# Record step execution
await metrics.record_step({
    "agent": "researcher",
    "success": True,
    "required_compensation": False,
    "timestamp": datetime.now(timezone.utc)
})

# Get metrics report
report = await metrics.get_report()
print(f"Success Rate: {report.success_rate}%")
print(f"Compensation Rate: {report.compensation_rate}%")
```

### 6. Testing Pipelines

**Write unit tests for steps:**

```python
import pytest
from steps.my_step import MyStep

@pytest.mark.asyncio
async def test_my_step_success():
    step = MyStep(config={"some": "config"})

    result = await step.execute({
        "input_field": "test_value"
    })

    assert result["status"] == "success"
    assert "output_field" in result

@pytest.mark.asyncio
async def test_my_step_handles_error():
    step = MyStep(config={})

    with pytest.raises(ValidationError):
        await step.execute({})  # Missing required field
```

---

## Showcase App Pipeline Examples

### Example 1: Simple Hello World

```python
# steps/hello_step.py
from ia_modules.pipeline import Step

class HelloStep(Step):
    async def execute(self, input_data):
        name = input_data.get("name", "World")
        return {
            "message": f"Hello, {name}!",
            "timestamp": datetime.now().isoformat()
        }
```

### Example 2: Data Processing with Validation

```python
# steps/validate_step.py
from ia_modules.pipeline import Step

class ValidateDataStep(Step):
    async def execute(self, input_data):
        data = input_data["data"]
        errors = []

        # Validation logic
        for row in data:
            if not row.get("id"):
                errors.append(f"Missing ID in row: {row}")
            if not row.get("value"):
                errors.append(f"Missing value in row: {row}")

        if errors:
            return {
                "status": "failed",
                "errors": errors,
                "valid_count": 0
            }

        return {
            "status": "passed",
            "valid_count": len(data),
            "data": data
        }
```

### Example 3: LLM-Powered Content Generation

```python
# steps/generate_content_step.py
from ia_modules.pipeline import Step
from ia_modules.agents import BaseAgent, AgentRole

class GenerateContentStep(Step):
    async def execute(self, input_data):
        # Create writer agent
        writer_role = AgentRole(
            name="ContentWriter",
            description="Creates engaging content",
            capabilities=["writing", "editing"]
        )

        writer = BaseAgent(
            role=writer_role,
            llm_config={"model": "gpt-4", "temperature": 0.7}
        )

        # Generate content
        result = await writer.execute({
            "task": f"Write a blog post about {input_data['topic']}",
            "style": input_data.get("style", "professional"),
            "length": input_data.get("length", "medium")
        })

        return {
            "content": result["text"],
            "word_count": len(result["text"].split()),
            "model_used": "gpt-4"
        }
```

---

## Summary

### Key Takeaways

1. **Pipelines** = Workflow orchestration with steps
2. **Agents** = Autonomous decision-makers with tools
3. **Use pipelines for**: Process flow, state management, retry, checkpoints
4. **Use agents for**: LLM interactions, tool calling, decision making
5. **Combine both**: Agents as steps in pipelines

### When to Use What

| Use Case | Solution |
|----------|----------|
| Sequential data processing | Linear Pipeline |
| Conditional logic | Conditional Pipeline |
| Parallel tasks | Parallel Pipeline |
| Iterative refinement | Loop Pipeline |
| Human approval needed | HITL Pipeline |
| Single AI task | Single Agent |
| Complex multi-step AI | Multi-Agent Orchestration |
| Agent needs APIs | Agent with Tools |
| Conversational AI | Agent with Memory |

### Next Steps

1. Review example pipelines in `showcase_app/pipelines/`
2. Try building a simple pipeline
3. Add your own step implementations
4. Experiment with agent orchestration
5. Monitor metrics and iterate

---

**Remember**: Pipelines manage the workflow. Agents make the decisions. Together, they create powerful, reliable AI systems. 🚀
