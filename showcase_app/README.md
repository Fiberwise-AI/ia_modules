# IA Modules Showcase App

A web application demonstrating the capabilities of the IA Modules framework through interactive examples and real-time visualizations.

## What It Demonstrates

### 🚀 Pipeline Execution
- **Sequential Processing** - Steps execute in order with data passing between them
- **Real-time Monitoring** - Watch pipeline execution with live updates
- **Data Flow Visualization** - See how data transforms through each step
- **Multiple Patterns** - Examples of different pipeline architectures

### 📊 Reliability Metrics
- **Success Rate (SR)** - Track pipeline completion rates
- **Checkpoint Recovery (CR)** - Monitor checkpoint/recovery operations
- **Human Intervention Rate (HIR)** - Measure human-in-the-loop frequency
- **SLO Compliance** - Visualize service level objectives
- **Historical Analysis** - Trend charts and performance over time

### 🤖 LLM Integration
- **Multi-Provider Support** - OpenAI, Anthropic, Google Gemini
- **Agentic Patterns** - Reflection, Planning, Tool Use, RAG, Metacognition
- **Token Tracking** - Real-time usage monitoring per request
- **Cost Calculation** - Automatic USD cost tracking
- **Rate Limiting** - Configurable requests/minute and tokens/minute limits

### ✨ Modern UI/UX
- **Dark Mode** - System-aware theme with manual toggle
- **Loading States** - Skeleton loaders and spinners
- **Toast Notifications** - Success/error feedback
- **Error Boundaries** - Graceful error recovery
- **Keyboard Shortcuts** - Cmd+D (theme), Cmd+B (sidebar), Cmd+/ (help), Esc (close)
- **Mobile Responsive** - Works on all screen sizes

## Architecture

The showcase app is built with:

**Backend (FastAPI)**
- API routes for pipelines, execution, and metrics
- IA Modules integration for pipeline execution
- WebSocket support for real-time updates
- Service layer for business logic

**Frontend (React)**
- Component-based UI with React 18
- TanStack Query for server state
- Recharts for metrics visualization
- React Flow for pipeline graphs
- Tailwind CSS for styling

```
showcase_app/
├── backend/
│   ├── api/          # REST endpoints
│   ├── services/     # Business logic
│   ├── models/       # Data models
│   └── pipelines/    # Example pipelines
└── frontend/
    └── src/
        ├── components/  # UI components
        ├── pages/       # Page views
        └── services/    # API clients
```

## Example Pipelines

### Simple Three-Step Pipeline
Sequential data transformation demonstrating basic pipeline flow.
- Step 1: Uppercase + add prefix
- Step 2: Lowercase + add suffix  
- Step 3: Reverse + add prefix

### Conditional Processing
Branching logic based on data quality scores.
- Filters data by quality threshold
- Routes to different processing paths
- Demonstrates conditional execution

### Parallel Processing
Concurrent execution of independent steps.
- Splits data across multiple processors
- Executes in parallel
- Merges results

### Human-in-the-Loop
Interactive approval workflow.
- Pauses execution for human review
- Supports approve/reject decisions
- Resumes based on decision

### Error Recovery
Automatic retry and compensation patterns.
- Retries failed operations
- Applies compensation logic
- Demonstrates resilience patterns

## IA Modules Concepts Demonstrated

### Pipeline Architecture
- **Sequential Flow** - Steps execute one after another
- **Conditional Branching** - Route execution based on conditions
- **Parallel Execution** - Run independent steps concurrently
- **Data Passing** - Automatic output-to-input binding

### Reliability Features
- **Checkpointing** - Save state and resume from failures
- **Event Replay** - Debug and replay failed executions
- **Metrics Tracking** - Real-time monitoring of SR, CR, HIR
- **SLO Management** - Define and track service objectives

### Step Patterns
- **Basic Steps** - Simple data transformation
- **LLM Steps** - AI-powered processing
- **Human Steps** - Interactive approval gates
- **Parallel Steps** - Concurrent execution

---

For setup and usage instructions, see the main [IA Modules README](../README.md).
