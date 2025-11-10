# Agent Collaboration Pattern Examples

This directory contains practical, runnable examples demonstrating the four collaboration patterns implemented in ia_modules.

## Overview

The ia_modules agent system supports four distinct collaboration patterns, each suited to different types of multi-agent tasks:

| Pattern | Use Case | Example |
|---------|----------|---------|
| **Hierarchical** | Leader-worker delegation | Research team coordination |
| **Debate** | Adversarial argumentation | Code review discussions |
| **Consensus** | Agreement-based decisions | Product feature decisions |
| **Peer-to-Peer** | Equal collaboration | Knowledge sharing network |

## Examples

### 1. Hierarchical: Research Team

**File:** [`hierarchical_research_team.py`](./hierarchical_research_team.py)

**Scenario:** A lead researcher coordinates specialist researchers to analyze a complex topic.

**Structure:**
- 1 Leader Agent (Lead Researcher)
- 3 Worker Agents (Algorithm Specialist, Hardware Specialist, Application Specialist)

**Process:**
1. Leader receives research topic
2. Leader decomposes into specialized subtasks
3. Workers execute in parallel
4. Leader synthesizes comprehensive report

**Run:**
```bash
cd ia_modules/agents/examples
python hierarchical_research_team.py
```

**Key Concepts:**
- Task decomposition and delegation
- Worker specialization
- Result synthesis
- Leader-worker communication

---

### 2. Debate: Code Review

**File:** [`debate_code_review.py`](./debate_code_review.py)

**Scenario:** Agents argue for and against accepting code changes from multiple perspectives.

**Structure:**
- 2 Proponent Agents (Security, Performance)
- 2 Opponent Agents (Security, Performance)
- 1 Moderator Agent

**Process:**
1. Opening statements (for/against)
2. Multiple debate rounds with rebuttals
3. Closing arguments
4. Synthesis of best arguments

**Run:**
```bash
cd ia_modules/agents/examples
python debate_code_review.py
```

**Key Concepts:**
- Adversarial argumentation
- Multiple perspectives
- Structured debate rounds
- Critical analysis

---

### 3. Consensus: Product Decision

**File:** [`consensus_product_decision.py`](./consensus_product_decision.py)

**Scenario:** Stakeholders from different departments reach consensus on product features through voting and discussion.

**Structure:**
- 5 Stakeholder Agents (Engineering, Design, Sales, Support, Product)

**Process:**
1. Initial proposal presentation
2. Stakeholder discussion (each perspective)
3. Voting round
4. Refinement based on feedback
5. Iteration until consensus

**Run:**
```bash
cd ia_modules/agents/examples
python consensus_product_decision.py
```

**Key Concepts:**
- Multi-stakeholder decision making
- Voting mechanisms (majority/unanimous/weighted)
- Iterative refinement
- Consensus building

---

### 4. Peer-to-Peer: Knowledge Sharing

**File:** [`peer_knowledge_sharing.py`](./peer_knowledge_sharing.py)

**Scenario:** Expert agents from different domains share knowledge as equals to build comprehensive understanding.

**Structure:**
- 4 Expert Peer Agents (Computer Science, Mathematics, Physics, Biology)

**Process:**
1. Round 1: Each expert contributes domain perspective
2. Sharing: All contributions distributed to all peers
3. Round 2: Experts build on each other's insights
4. Synthesis of collective knowledge

**Run:**
```bash
cd ia_modules/agents/examples
python peer_knowledge_sharing.py
```

**Key Concepts:**
- Equal collaboration (no hierarchy)
- Cross-domain knowledge integration
- Collective intelligence
- Emergent understanding

---

## Pattern Selection Guide

### When to Use Each Pattern

**Hierarchical Pattern:**
- ✅ Clear task decomposition possible
- ✅ Need centralized coordination
- ✅ Workers have specialized skills
- ✅ Leader can synthesize results
- ❌ Tasks are too interconnected
- ❌ No clear authority structure

**Debate Pattern:**
- ✅ Need to explore multiple perspectives
- ✅ Critical analysis required
- ✅ Identify weaknesses in reasoning
- ✅ Stress-test ideas
- ❌ Need quick consensus
- ❌ Issue is not controversial

**Consensus Pattern:**
- ✅ Democratic decision making needed
- ✅ Multiple stakeholders must agree
- ✅ Risk mitigation through diverse input
- ✅ Shared understanding required
- ❌ Urgent decision needed
- ❌ Clear authority exists

**Peer-to-Peer Pattern:**
- ✅ Equal expertise among agents
- ✅ Knowledge sharing beneficial
- ✅ Collaborative problem solving
- ✅ Building on each other's work
- ❌ Need clear leadership
- ❌ Tasks are independent

---

## Architecture

All examples follow the same architecture:

```
┌─────────────────┐
│   StateManager  │  ← Centralized state for all agents
└─────────────────┘
         ↑
         │
    ┌────┴────┐
    │         │
┌───▼─────┐   │
│ Agents  │◄──┼─► MessageBus ← Inter-agent communication
└─────────┘   │
    │         │
    └────┬────┘
         ↓
┌─────────────────┐
│  Collaboration  │  ← Pattern-specific orchestration
│     Pattern     │
└─────────────────┘
```

### Key Components

1. **StateManager**: Shared state for agent communication
2. **MessageBus**: Message passing between agents
3. **AgentRole**: Defines agent capabilities and behavior
4. **BaseCollaborativeAgent**: Base class with messaging
5. **Collaboration Pattern**: Orchestrates multi-agent workflow

---

## Customization

Each example can be customized:

### 1. Change Agent Count
```python
# Hierarchical: Add more workers
workers = [
    SpecialistResearcherAgent(..., specialty="algorithms"),
    SpecialistResearcherAgent(..., specialty="hardware"),
    SpecialistResearcherAgent(..., specialty="applications"),
    SpecialistResearcherAgent(..., specialty="theory"),  # NEW
]
```

### 2. Modify Agent Behavior
```python
class CustomResearcherAgent(SpecialistResearcherAgent):
    async def _process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        # Custom processing logic
        result = await super()._process_task(task_data)
        result["custom_analysis"] = self._analyze(task_data)
        return result
```

### 3. Adjust Collaboration Parameters
```python
# Consensus: Change voting strategy
consensus = ConsensusCollaboration(
    agents=stakeholders,
    strategy=ConsensusStrategy.UNANIMOUS,  # Require all to agree
    max_iterations=5  # More refinement rounds
)
```

### 4. Add Real LLM Integration
```python
# Replace simulated responses with actual LLM calls
class LLMResearcherAgent(SpecialistResearcherAgent):
    async def _process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        # Use ia_modules LLM service
        from ia_modules.pipeline.llm_provider_service import LLMProviderService

        llm = LLMProviderService()
        response = await llm.generate(
            prompt=self._create_research_prompt(task_data),
            model="gpt-4"
        )
        return self._parse_llm_response(response)
```

---

## Integration with Pipelines

These collaboration patterns can be integrated into ia_modules pipelines:

```python
from ia_modules.pipeline import Pipeline, PipelineStep
from ia_modules.agents.examples import run_research_team_example

class MultiAgentResearchStep(PipelineStep):
    """Pipeline step using hierarchical collaboration."""

    async def process(self, input_data: Dict[str, Any], context: ExecutionContext):
        # Run hierarchical research collaboration
        result = await run_research_team_example()

        # Extract results for pipeline
        return {
            "research_findings": result["worker_outputs"],
            "synthesis": result["summary"]
        }

# Use in pipeline
pipeline = Pipeline([
    MultiAgentResearchStep(),
    # ... other steps
])
```

---

## Testing

Each example includes simulated agent responses for demonstration purposes. To test with real agents:

1. **Unit Tests**: Test individual agent behavior
```python
import pytest

@pytest.mark.asyncio
async def test_researcher_processes_task():
    state = StateManager(thread_id="test")
    bus = MessageBus()

    researcher = SpecialistResearcherAgent(
        role=AgentRole(name="test_researcher"),
        state_manager=state,
        message_bus=bus,
        specialty="algorithms"
    )

    result = await researcher._process_task({
        "task_id": "test",
        "description": "Analyze quantum algorithms"
    })

    assert result["status"] == "success"
    assert "findings" in result
```

2. **Integration Tests**: Test full collaboration pattern
```python
@pytest.mark.asyncio
async def test_research_team_collaboration():
    result = await run_research_team_example()

    assert result["status"] == "success"
    assert result["successful_workers"] == 3
    assert "summary" in result
```

---

## Troubleshooting

### Common Issues

**1. Agents not communicating:**
- Ensure `await collaboration.initialize()` is called
- Check all agents are subscribed to message bus
- Verify agent IDs are unique

**2. Timeout errors:**
- Increase timeout in `send_task_request(timeout=60.0)`
- Check for deadlocks in agent logic
- Ensure agents respond to task requests

**3. State not persisting:**
- Verify StateManager is shared across agents
- Check `thread_id` is consistent
- Use `await state.set()` not direct assignment

**4. Missing dependencies:**
```bash
pip install ia_modules
# Or from source:
cd ia_modules
pip install -e .
```

---

## Next Steps

1. **Run Examples**: Try each example to understand the patterns
2. **Customize**: Modify agent behavior for your use case
3. **Integrate**: Add collaboration patterns to your pipelines
4. **Extend**: Create new agent types and specializations

## Advanced Topics

### Combining Patterns

Patterns can be combined for complex workflows:

```python
# Hierarchical coordination of debate teams
class DebateCoordinator(LeaderAgent):
    async def execute(self, input_data):
        # Run multiple debates in parallel
        debate_results = await self._run_parallel_debates([
            {"topic": "Security vs Usability"},
            {"topic": "Performance vs Maintainability"}
        ])

        # Synthesize debate conclusions
        return self._synthesize_debates(debate_results)
```

### Agent Learning

Agents can improve over time:

```python
class LearningAgent(BaseCollaborativeAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.performance_history = []

    async def execute(self, input_data):
        result = await super().execute(input_data)

        # Track performance
        self.performance_history.append({
            "input": input_data,
            "output": result,
            "timestamp": datetime.now()
        })

        # Adjust behavior based on history
        if len(self.performance_history) > 10:
            self._optimize_strategy()

        return result
```

### Human-in-the-Loop

Add human oversight to collaboration:

```python
class HumanApprovalAgent(BaseCollaborativeAgent):
    async def execute(self, input_data):
        # Agent processes normally
        result = await self._process(input_data)

        # Request human approval
        if self._needs_approval(result):
            approval = await self._request_human_approval(result)
            if not approval["approved"]:
                result = await self._revise(result, approval["feedback"])

        return result
```

---

## Resources

- **Agent System Documentation**: [`../AGENT_FOLDERS_GUIDE.md`](../AGENT_FOLDERS_GUIDE.md)
- **Core Implementation**: [`../collaboration_patterns/`](../collaboration_patterns/)
- **Base Classes**: [`../core.py`](../core.py), [`../base_agent.py`](../base_agent.py)
- **Communication**: [`../communication.py`](../communication.py)
- **State Management**: [`../state.py`](../state.py)

---

## Contributing

To add new examples:

1. Create new file: `{pattern}_{use_case}.py`
2. Follow existing structure
3. Include docstring with scenario description
4. Add `run_{example_name}_example()` function
5. Update this README

---

## License

MIT License - See ia_modules root LICENSE file

---

**Last Updated**: 2025-11-05
**ia_modules Version**: 1.0.0+
