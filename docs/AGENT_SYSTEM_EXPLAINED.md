# Agent System Complete Breakdown

## What Are Agents in IA Modules?

**Agents** are specialized, single-purpose components that work together to accomplish complex tasks. Unlike traditional pipeline steps, agents:

1. **Are role-based** - Each agent has a specific job (planner, researcher, coder, critic)
2. **Share centralized state** - All agents read/write to the same StateManager
3. **Can form feedback loops** - Agents can review each other's work iteratively
4. **Have guardrails** - Max iterations, allowed tools, explicit system prompts

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    AgentOrchestrator                        │
│  (Manages workflow: which agent runs when, in what order)  │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
   ┌─────────┐         ┌─────────┐        ┌─────────┐
   │ Planner │         │  Coder  │        │ Critic  │
   │ Agent   │────────▶│ Agent   │◀──────▶│ Agent   │
   └─────────┘         └─────────┘        └─────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                   ┌──────────────────┐
                   │  StateManager    │
                   │  (Shared State)  │
                   └──────────────────┘
```

## Core Components

### 1. AgentRole (Definition)

**What it is**: A dataclass that defines an agent's capabilities and constraints.

**Key Fields**:
- `name` - Unique identifier (e.g., "planner", "coder")
- `description` - What this agent does
- `allowed_tools` - Which tools it can use (enforces permissions)
- `system_prompt` - Behavioral instructions (for LLM-based agents)
- `max_iterations` - How many times it can retry (prevents infinite loops)

**Example**:
```python
planner_role = AgentRole(
    name="planner",
    description="Breaks down complex tasks into actionable steps",
    allowed_tools=["search", "read_file"],
    system_prompt="You are a planning agent. Create detailed step-by-step plans.",
    max_iterations=5
)
```

**Purpose**: Separates agent **definition** (what it can do) from **implementation** (how it does it).

### 2. BaseAgent (Implementation)

**What it is**: Abstract base class that all agents inherit from.

**Contract**:
```python
class MyAgent(BaseAgent):
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # 1. Read from shared state
        previous_result = await self.read_state("key")

        # 2. Do specialized work
        result = await self.do_my_job(previous_result)

        # 3. Write back to state
        await self.write_state("my_result", result)

        # 4. Return summary
        return {"status": "success", "result": result}
```

**Key Methods**:
- `execute(input_data)` - Main entry point (abstract - must implement)
- `read_state(key, default)` - Read from centralized state
- `write_state(key, value)` - Write to centralized state
- `increment_iteration()` - Track retry attempts
- `validate_iteration()` - Check if under max_iterations

**Purpose**: Provides consistent interface for all agents while allowing specialization.

### 3. StateManager (Shared Memory)

**What it is**: Centralized, thread-scoped state that all agents read from and write to.

**Key Features**:
- **Thread-scoped** - Each workflow has isolated state (multi-user safe)
- **Versioned** - Every state change creates a new version (history)
- **Rollback** - Can revert to previous states
- **Async locks** - Prevents race conditions
- **Checkpoint integration** - Can persist to database

**API**:
```python
state = StateManager(thread_id="workflow-123")

# Basic operations
await state.set("key", "value")
value = await state.get("key", default=None)
await state.update({"k1": "v1", "k2": "v2"})
await state.delete("key")

# Versioning
snapshot = await state.snapshot()  # Get complete state
await state.rollback(steps=2)      # Revert 2 changes

# Persistence
await state.persist()              # Save to checkpoint
```

**Purpose**: Single source of truth for agent collaboration + debugging through versioning.

### 4. AgentOrchestrator (Workflow Manager)

**What it is**: Graph-based executor that decides which agent runs when.

**Capabilities**:
- **Sequential edges** - Agent A → Agent B
- **Conditional edges** - Agent A → Agent B (only if condition met)
- **Feedback loops** - Agent A → Agent B → Agent A (until approved)
- **Cycle detection** - Prevents infinite graphs
- **Max steps** - Bounds execution (safety)

**API**:
```python
orchestrator = AgentOrchestrator(state)

# Add agents
orchestrator.add_agent("planner", planner_agent)
orchestrator.add_agent("coder", coder_agent)
orchestrator.add_agent("critic", critic_agent)

# Define workflow
orchestrator.add_edge("planner", "coder")  # Sequential

# Feedback loop: coder → critic → coder (max 3 iterations)
is_complete = orchestrator.add_feedback_loop(
    worker_agent="coder",
    critic_agent="critic",
    max_iterations=3
)

# Conditional edge: critic → formatter (only when approved)
orchestrator.add_edge("critic", "formatter", condition=is_complete)

# Execute
result = await orchestrator.run(
    start_agent="planner",
    input_data={"task": "Create a Python function"},
    max_steps=100  # Safety limit
)
```

**Purpose**: Declarative workflow definition with automatic safety mechanisms.

### 5. Built-in Agents (Examples)

We provide 5 specialized agents:

**PlannerAgent**:
```python
# Breaks down tasks into steps
await state.set("plan", ["step1", "step2", "step3"])
```

**ResearcherAgent**:
```python
# Gathers information
await state.set("research_findings", ["fact1", "fact2"])
```

**CoderAgent**:
```python
# Generates code
await state.set("code_snippets", ["def foo(): ..."])
```

**CriticAgent**:
```python
# Reviews work
await state.set("critique", "Missing error handling")
await state.set("approved", False)  # Triggers retry
```

**FormatterAgent**:
```python
# Formats final output
await state.set("final_answer", "Well-formatted result")
```

**Purpose**: Ready-to-use agents demonstrating the pattern.

## How Agents Fit Into Pipelines

### Traditional Pipeline Steps vs Agents

**Traditional Pipeline Step**:
```python
class MyStep(PipelineStep):
    async def execute(self, input_data: Dict, context: PipelineContext):
        # Input → Process → Output (linear, isolated)
        result = process(input_data)
        return result
```

**Agent**:
```python
class MyAgent(BaseAgent):
    async def execute(self, input_data: Dict):
        # Read state → Process → Write state (collaborative)
        prev = await self.read_state("previous_result")
        result = process(prev, input_data)
        await self.write_state("my_result", result)
        return {"status": "success"}
```

### Key Differences

| Aspect | Pipeline Steps | Agents |
|--------|---------------|--------|
| **State** | Passed through steps linearly | Centralized, shared state |
| **Collaboration** | Steps don't know about each other | Agents read each other's outputs |
| **Loops** | Requires explicit loop configuration | Built-in feedback loop support |
| **Retry** | Step-level error handling | Agent-level iteration tracking |
| **Debugging** | Checkpoint after each step | State versioning (every change) |

### Integration Options

**Option 1: Agents as Pipeline Steps**

You can wrap an agent orchestrator as a pipeline step:

```python
class AgentWorkflowStep(PipelineStep):
    """Run multi-agent workflow within a pipeline."""

    async def execute(self, input_data: Dict, context: PipelineContext):
        # Create state
        state = StateManager(thread_id=context.thread_id)

        # Create orchestrator
        orchestrator = AgentOrchestrator(state)
        # ... add agents and edges ...

        # Run agents
        result = await orchestrator.run("planner", input_data)

        # Return final state as step output
        return await state.snapshot()
```

**Option 2: Pure Agent Workflows**

Skip pipelines entirely, use agents directly:

```python
# Create state
state = StateManager(thread_id="task-123")

# Create orchestrator
orchestrator = AgentOrchestrator(state)
orchestrator.add_agent("planner", PlannerAgent(state))
orchestrator.add_agent("coder", CoderAgent(state))
orchestrator.add_agent("critic", CriticAgent(state))

# Define workflow
orchestrator.add_edge("planner", "coder")
is_complete = orchestrator.add_feedback_loop("coder", "critic", max_iterations=3)

# Execute
result = await orchestrator.run("planner", {"task": "Build API"})
```

**Option 3: Hybrid - Agents + Traditional Steps**

Mix agents and traditional steps:

```python
pipeline = Pipeline([
    LoadDataStep(),                    # Traditional step
    AgentWorkflowStep(),              # Agents for complex reasoning
    SaveResultsStep(),                # Traditional step
])
```

## When to Use Agents vs Traditional Steps

**Use Traditional Pipeline Steps When**:
- ✅ Task is simple and linear (fetch → process → save)
- ✅ Steps are independent and don't need to collaborate
- ✅ No retry or feedback loops needed
- ✅ Performance is critical (less overhead)

**Use Agents When**:
- ✅ Task requires multi-step reasoning
- ✅ Need feedback loops (generate → review → revise)
- ✅ Agents need to collaborate (share findings)
- ✅ Task involves LLM-based decision making
- ✅ Need detailed debugging (state versioning)

**Example Use Cases**:

**Traditional Pipeline**:
- ETL workflows (Extract → Transform → Load)
- API request chains (Auth → Fetch → Parse)
- Batch processing (Read files → Process → Write)

**Agent Workflow**:
- Content generation (Plan → Write → Review → Revise)
- Code generation (Design → Code → Test → Fix)
- Research (Search → Analyze → Synthesize)

## Complete Example: Content Generation

```python
from ia_modules.agents import (
    AgentRole, BaseAgent, StateManager, AgentOrchestrator
)

# 1. Define custom agent
class WriterAgent(BaseAgent):
    async def execute(self, input_data: Dict[str, Any]):
        # Read plan from planner
        plan = await self.read_state("plan", [])
        topic = input_data.get("topic", "")

        # Generate content
        content = f"Article about {topic}\n"
        for step in plan:
            content += f"- {step}\n"

        # Write to state
        await self.write_state("draft", content)
        await self.write_state("approved", False)  # Needs review

        return {"status": "draft_created"}

# 2. Create state manager
state = StateManager(thread_id="article-gen-123")

# 3. Create agents
planner_role = AgentRole(
    name="planner",
    description="Creates article outline",
    max_iterations=3
)
writer_role = AgentRole(
    name="writer",
    description="Writes article content",
    max_iterations=5
)
critic_role = AgentRole(
    name="critic",
    description="Reviews article quality",
    max_iterations=3
)

from ia_modules.agents.roles import PlannerAgent, CriticAgent

planner = PlannerAgent(planner_role, state)
writer = WriterAgent(writer_role, state)
critic = CriticAgent(critic_role, state)

# 4. Create orchestrator
orchestrator = AgentOrchestrator(state)
orchestrator.add_agent("planner", planner)
orchestrator.add_agent("writer", writer)
orchestrator.add_agent("critic", critic)

# 5. Define workflow
orchestrator.add_edge("planner", "writer")

# Feedback loop: writer → critic → writer (until approved)
is_approved = orchestrator.add_feedback_loop(
    worker_agent="writer",
    critic_agent="critic",
    max_iterations=3
)

# 6. Execute
result = await orchestrator.run(
    start_agent="planner",
    input_data={"topic": "AI Agent Systems"},
    max_steps=50
)

# 7. Get final result
final_draft = await state.get("draft")
was_approved = await state.get("approved")
```

## State Versioning Example

```python
# Initial state
await state.set("draft", "First draft")
# Version 1: {"draft": "First draft"}

# Critic suggests changes
await state.set("critique", "Add more details")
# Version 2: {"draft": "First draft", "critique": "Add more details"}

# Writer revises
await state.set("draft", "Second draft with more details")
# Version 3: {"draft": "Second draft...", "critique": "Add more details"}

# Oh no, second draft is worse!
await state.rollback(steps=1)
# Back to Version 2: {"draft": "First draft", "critique": "Add more details"}

# Try different approach
await state.set("draft", "Better second draft")
# Version 3 (new): {"draft": "Better second draft", "critique": ...}
```

## Test Coverage

**Week 4 Tests: 42/42 passing (100%)**

1. **Agent Core Tests** (11 tests):
   - AgentRole creation
   - BaseAgent initialization
   - State read/write operations
   - Iteration tracking and validation

2. **State Manager Tests** (18 tests):
   - Get/set/update/delete operations
   - State versioning and history
   - Rollback functionality
   - Checkpoint integration
   - Thread isolation

3. **Orchestrator Tests** (13 tests):
   - Agent registration
   - Edge creation (sequential, conditional)
   - Workflow execution
   - Feedback loops
   - Cycle detection
   - Max steps enforcement

## Summary

**What We Built**:
- ✅ Role-based agent system (AgentRole)
- ✅ Agent base class with state access (BaseAgent)
- ✅ Centralized state with versioning (StateManager)
- ✅ Graph-based orchestration (AgentOrchestrator)
- ✅ Feedback loop support
- ✅ 5 built-in agents
- ✅ 42 tests, all passing

**How It Fits**:
- Agents are a **specialized workflow pattern** for complex, multi-step reasoning
- Can be used **standalone** or **within pipelines**
- Complementary to traditional steps (not replacement)
- Ideal for LLM-based tasks requiring collaboration and iteration

**Next Steps (Week 6)**:
- Add reliability metrics (SVR, CR, PC, HIR, MA)
- Add replay system for debugging
- Add explicit modes (explore/execute/escalate)
- Add evidence tracking (separate facts from confidence)
