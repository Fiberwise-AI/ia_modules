# IA Modules - Quick Start Guide

Get started with IA Modules in under 5 minutes!

## 🚀 Fastest Start: Showcase App

**See IA Modules in action with our interactive showcase:**

### Linux/Mac

```bash
# Install everything
./install.sh

# Start the app
./start.sh
```

### Windows

```bash
# Install everything
install.bat

# Start the app
start.bat
```

**Then open:** http://localhost:3000

### What You'll See

- ✅ **Example Pipelines** - Pre-built pipelines ready to execute
- ✅ **Agent Collaboration Patterns** - Debate, reflection, planning, agentic RAG
- ✅ **Real-Time Monitoring** - Watch pipelines execute live
- ✅ **Reliability Dashboard** - SR, CR, HIR, TCL, WCT metrics
- ✅ **SLO Compliance** - Visual compliance tracking
- ✅ **Execution History** - All runs with detailed logs

## 📦 Framework-Only Installation

If you just want to use IA Modules in your own project:

```bash
# Install with all features
pip install ia_modules[all]

# Or minimal install
pip install ia_modules
```

## 💻 Your First Pipeline

```python
from ia_modules.pipeline.core import PipelineStep, StepResult
from ia_modules.pipeline.runner import PipelineRunner
from ia_modules.reliability.metrics import ReliabilityMetrics
from ia_modules.reliability.memory_storage import InMemoryMetricStorage

# 1. Define a step
class GreetingStep(PipelineStep):
    async def execute(self, context):
        name = context.get_data("name", "World")
        return StepResult(
            success=True,
            data={"greeting": f"Hello, {name}!"}
        )

# 2. Set up reliability tracking
async def main():
    storage = InMemoryMetricStorage()
    metrics = ReliabilityMetrics(storage)

    runner = PipelineRunner(metrics=metrics)
    runner.register_step("greet", GreetingStep())

    # 3. Run the pipeline
    result = await runner.run(
        start_step="greet",
        initial_data={"name": "Developer"}
    )

    # 4. Check metrics
    report = await metrics.get_report()
    print(f"Success Rate: {report.sr:.2%}")
    print(f"Result: {result.data}")

import asyncio
asyncio.run(main())
```

## 📚 Next Steps

### Learn the Framework

1. **[Getting Started Guide](docs/GETTING_STARTED.md)** - Comprehensive tutorial
2. **[Features Overview](docs/FEATURES.md)** - All capabilities
3. **[API Reference](docs/API_REFERENCE.md)** - Detailed API docs

### Explore Examples

1. **Showcase App** - Interactive demo at http://localhost:3000
2. **Example Pipelines** - See [tests/pipelines/](tests/pipelines/)
3. **Test Suite** - Check [tests/](tests/) for working examples

### Production Deployment

1. **[Reliability Guide](docs/RELIABILITY_USAGE_GUIDE.md)** - EARF compliance
2. **[Migration Guide](MIGRATION.md)** - Upgrading from older versions
3. **[Contributing](CONTRIBUTING.md)** - How to contribute

## Built-in Step Types

IA Modules provides ready-to-use step types so you don't need to subclass `PipelineStep` for common patterns:

| Step Type | Module | Use Case |
|-----------|--------|----------|
| `LLMStep` | `pipeline.llm_step` | Send a prompt to a CLI agent, get text back |
| `FunctionStep` | `pipeline.function_step` | Wrap an async callable (vote tallying, routing, aggregation) |
| `AgentStep` | `pipeline.agent_step` | Run a CLI agent with workspace and tools |
| `A2AStep` | `pipeline.a2a_step` | Dispatch work to a remote A2A agent server |
| `ParallelStep` | `pipeline.parallel_step` | Fan-out: run child steps concurrently |
| `OrchestratorStep` | `pipeline.orchestrator_step` | Run an orchestrator pattern as one atomic step |

```python
from ia_modules.pipeline import LLMStep, FunctionStep, ParallelStep

# LLM step — just a prompt, no subclassing needed
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
parallel = ParallelStep(
    name="all_reviewers",
    children=[reviewer_1, reviewer_2, reviewer_3],
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

# Enforce permissions before execution
enforce_agent_claims(claims, cwd="/data/workspace", mode="research", tools=["Read", "Glob"])
```

Set `AGENT_AUTH_MODE=oidc` and configure `OIDC_DISCOVERY_URL` for production Keycloak/OIDC. Default is `local` (built-in mini OIDC, no external IDP needed).

## 🎯 Common Use Cases

### Data Processing Pipeline

```python
# Multi-step data transformation
pipeline_config = {
    "steps": {
        "load": {"type": "function", "fn": "load_data"},
        "validate": {"type": "function", "fn": "validate_data"},
        "transform": {"type": "llm", "system_prompt": "Transform this data..."},
        "export": {"type": "function", "fn": "export_data"}
    }
}
```

### AI Agent Workflow

```python
# LLM-powered content generation
pipeline_config = {
    "steps": {
        "research": {"type": "agent", "mode": "research"},
        "draft": {"type": "llm", "system_prompt": "Draft content based on research..."},
        "review": {"type": "llm", "system_prompt": "Review and improve..."},
        "publish": {"type": "function", "fn": "publish"}
    }
}
```

### Remote Agent Dispatch (A2A)

```python
# Dispatch to a remote A2A server
from ia_modules.pipeline import A2AStep

remote_agent = A2AStep(
    name="remote_analysis",
    server_url="http://a2a-server:3008",
    mode="research",
    tools=["Read", "Glob", "Grep"],
)
```

### Human-in-the-Loop

```python
# Interactive approval workflow
pipeline_config = {
    "steps": {
        "prepare": {"type": "function", "fn": "prepare"},
        "human_review": {"type": "hitl", "prompt": "Approve this?"},
        "process": {"type": "function", "fn": "process_decision"}
    }
}
```

## 🛠️ CLI Tools

```bash
# Validate pipeline definition
ia-modules validate pipeline.json

# Run pipeline
ia-modules run pipeline.json

# Visualize pipeline
ia-modules visualize pipeline.json --output graph.png

# Run benchmarks
ia-modules benchmark config.json
```

## 💡 Tips

### Enable Reliability Tracking

Always use reliability metrics in production:

```python
from ia_modules.reliability.sql_metric_storage import SQLMetricStorage
from ia_modules.database.interfaces import ConnectionConfig, DatabaseType

# PostgreSQL for production
config = ConnectionConfig(
    database_type=DatabaseType.POSTGRESQL,
    database_url="postgresql://user:pass@localhost/metrics"
)
storage = SQLMetricStorage(config)
metrics = ReliabilityMetrics(storage)
```

### Monitor SLO Compliance

```python
from ia_modules.reliability.slo_monitor import SLOMonitor

monitor = SLOMonitor(metrics)
compliance = await monitor.check_compliance()

if not compliance.sr_compliant:
    alert(f"SLO violation: SR={compliance.sr_current:.2%}")
```

### Use Checkpointing

```python
from ia_modules.checkpoint.manager import CheckpointManager

checkpoint_mgr = CheckpointManager()
runner = PipelineRunner(
    checkpoint_manager=checkpoint_mgr,
    metrics=metrics
)

# Automatically saves state, can resume on failure
result = await runner.run(
    start_step="fetch",
    checkpoint_enabled=True,
    thread_id="workflow_123"
)
```

## 🆘 Troubleshooting

### Backend Won't Start

```bash
# Check if port is in use
netstat -an | grep 8000  # Linux/Mac
netstat -an | findstr 8000  # Windows

# Try different port
cd showcase_app/backend
uvicorn main:app --port 8001
```

### Frontend Build Errors

```bash
cd showcase_app/frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### Import Errors

```bash
# Make sure IA Modules is installed
pip install -e .

# Verify installation
python -c "import ia_modules; print(ia_modules.__version__)"
```

## 📖 Documentation

- **Getting Started**: [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)
- **Features**: [docs/FEATURES.md](docs/FEATURES.md)
- **API Reference**: [docs/API_REFERENCE.md](docs/API_REFERENCE.md)
- **Reliability**: [docs/RELIABILITY_USAGE_GUIDE.md](docs/RELIABILITY_USAGE_GUIDE.md)
- **Showcase App**: [showcase_app/README.md](showcase_app/README.md)

## 🔗 Links

- **Main README**: [README.md](README.md)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)
- **Roadmap**: [ROADMAP.md](ROADMAP.md)
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)
- **License**: [LICENSE](LICENSE)

## 🎉 You're Ready!

Choose your path:

1. **Try the showcase app** → Run `./start.sh` or `start.bat`
2. **Build with the framework** → Read [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)
3. **Deploy to production** → See [docs/RELIABILITY_USAGE_GUIDE.md](docs/RELIABILITY_USAGE_GUIDE.md)

**Questions?** Check the docs or open an issue on GitHub.

**Happy building!** 🚀
