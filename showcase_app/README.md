# IA Modules Showcase App

A modern web application demonstrating the capabilities of IA Modules framework with real-time pipeline execution, reliability metrics, and interactive visualizations.

## Features

### 🚀 Pipeline Execution
- **Interactive Pipeline Builder** - Create and edit pipelines visually
- **Real-time Execution** - Watch pipelines execute with live updates
- **Step-by-Step Visualization** - See data flow between steps
- **Multiple Example Pipelines** - Pre-built examples showcasing features

### 📊 Reliability Dashboard
- **Live Metrics** - Real-time SR, CR, PC, HIR, MA, TCL, WCT tracking
- **Historical Charts** - Trend analysis and performance over time
- **SLO Monitoring** - Visual SLO compliance indicators
- **Event Replay** - Debug failed executions

### 🎯 Demo Features
- **Data Processing Pipeline** - Multi-step data transformation
- **AI Agent Workflow** - LLM-powered content generation
- **Human-in-the-Loop** - Interactive approval workflows
- **Parallel Processing** - Concurrent step execution demo
- **Error Recovery** - Automatic retry and compensation demo

## Architecture

```
showcase_app/
├── backend/                 # FastAPI backend
│   ├── api/                # API routes
│   │   ├── pipelines.py   # Pipeline management
│   │   ├── execution.py   # Pipeline execution
│   │   ├── metrics.py     # Reliability metrics
│   │   └── websocket.py   # Real-time updates
│   ├── models/             # Pydantic models
│   ├── services/           # Business logic
│   ├── pipelines/          # Example pipelines
│   └── main.py            # FastAPI app entry
│
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/    # Reusable UI components
│   │   ├── pages/         # Page components
│   │   ├── services/      # API clients
│   │   └── App.jsx        # Main app component
│   ├── public/
│   └── package.json
│
└── README.md              # This file
```

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- npm or yarn
- Docker & Docker Compose (for PostgreSQL and Redis)

### Start Services with Docker

```bash
# Start PostgreSQL and Redis
cd showcase_app
docker-compose up -d

# Wait for services to be ready
docker-compose ps
```

This starts:
- PostgreSQL on port 5433
- Redis on port 6379

### Backend Setup

```bash
# Navigate to showcase app
cd showcase_app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -e "..[all]"
pip install fastapi uvicorn websockets python-multipart

# Run backend
cd backend
uvicorn main:app --reload --port 5555
```

Backend will be available at http://localhost:5555

**API Documentation**: http://localhost:5555/docs

### Frontend Setup

```bash
# In a new terminal, navigate to frontend
cd showcase_app/frontend

# Install dependencies
npm install

# Start development server
npm start
```

Frontend will be available at http://localhost:5173

## API Endpoints

### Pipeline Management

```
GET    /api/pipelines              # List all pipelines
GET    /api/pipelines/{id}         # Get pipeline details
POST   /api/pipelines              # Create new pipeline
PUT    /api/pipelines/{id}         # Update pipeline
DELETE /api/pipelines/{id}         # Delete pipeline
```

### Pipeline Execution

```
POST   /api/execute/{pipeline_id}  # Start pipeline execution
GET    /api/execution/{job_id}     # Get execution status
POST   /api/execution/{job_id}/pause   # Pause execution
POST   /api/execution/{job_id}/resume  # Resume execution
DELETE /api/execution/{job_id}     # Cancel execution
```

### Reliability Metrics

```
GET    /api/metrics/report         # Get reliability report
GET    /api/metrics/history        # Get metric history
GET    /api/metrics/slo            # Get SLO compliance
GET    /api/metrics/events         # Get event log
POST   /api/metrics/replay/{event_id}  # Replay event
```

### WebSocket

```
WS     /ws/execution/{job_id}      # Real-time execution updates
WS     /ws/metrics                 # Real-time metrics stream
```

## Example Pipelines

### 1. Data Processing Pipeline
Demonstrates multi-step data transformation with validation and error handling.

```python
# Steps: Load → Validate → Transform → Analyze → Export
```

### 2. AI Content Generator
LLM-powered content generation with human review.

```python
# Steps: Topic Selection → Research → Draft → Human Review → Publish
```

### 3. Parallel Processing Demo
Showcases concurrent execution of independent steps.

```python
# Steps: Split → [Process A, Process B, Process C] → Merge → Output
```

### 4. Error Recovery Demo
Demonstrates automatic retry and compensation patterns.

```python
# Steps: Attempt → Retry (on failure) → Compensate → Report
```

### 5. Human-in-the-Loop Workflow
Interactive approval process with UI integration.

```python
# Steps: Prepare → Submit for Approval → Wait for Human → Process Decision
```

## Frontend Components

### Pipeline Viewer
- Visual graph representation of pipeline structure
- Interactive node selection
- Data flow visualization
- Execution progress overlay

### Metrics Dashboard
- Real-time metric cards (SR, CR, HIR, etc.)
- Line charts for trend analysis
- SLO compliance indicators
- Performance histograms

### Execution Monitor
- Live execution log
- Step-by-step progress
- Output preview
- Error handling visualization

### Pipeline Editor
- JSON editor with syntax highlighting
- Visual pipeline builder (drag-and-drop)
- Step configuration forms
- Validation and error checking

## Technology Stack

### Backend
- **FastAPI** - Modern async web framework
- **IA Modules** - Pipeline execution engine
- **Pydantic** - Data validation
- **WebSockets** - Real-time updates
- **SQLite** - Metrics storage (development)
- **PostgreSQL** - Production metrics storage (optional)

### Frontend
- **React 18** - UI framework
- **React Router** - Navigation
- **TanStack Query** - Server state management
- **Recharts** - Data visualization
- **Tailwind CSS** - Styling
- **React Flow** - Pipeline visualization
- **Axios** - HTTP client
- **Socket.io Client** - WebSocket client

## Development

### Running Tests

```bash
# Backend tests
cd backend
pytest tests/ -v

# Frontend tests
cd frontend
npm test
```

### Building for Production

```bash
# Backend
cd backend
pip install build
python -m build

# Frontend
cd frontend
npm run build
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d
```

Access at http://localhost:5173

## Configuration

### Environment Variables

Create `.env` file in backend directory:

```env
# Database
DATABASE_URL=sqlite:///./metrics.db
# or for PostgreSQL:
# DATABASE_URL=postgresql://user:pass@localhost/ia_modules

# Redis (optional)
REDIS_URL=redis://localhost:6379

# CORS
CORS_ORIGINS=http://localhost:5173,http://localhost:3001

# LLM API Keys (optional, for AI demos)
GEMINI_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here

# App Settings
DEBUG=true
LOG_LEVEL=info
```

Create `.env` file in frontend directory:

```env
REACT_APP_API_URL=http://localhost:5555
REACT_APP_WS_URL=ws://localhost:5555
```

## Demo Scenarios

### Scenario 1: Basic Pipeline Execution
1. Open app at http://localhost:5173
2. Navigate to "Examples" → "Data Processing"
3. Click "Run Pipeline"
4. Watch real-time execution in the monitor
5. View results and metrics

### Scenario 2: Reliability Monitoring
1. Navigate to "Metrics Dashboard"
2. Run multiple pipelines from different tabs
3. Watch metrics update in real-time
4. Check SLO compliance indicators
5. View historical trends

### Scenario 3: Human-in-the-Loop
1. Navigate to "Examples" → "Approval Workflow"
2. Click "Run Pipeline"
3. Watch pipeline pause at approval step
4. Review content in approval dialog
5. Approve or reject
6. Watch pipeline resume

### Scenario 4: Error Recovery
1. Navigate to "Examples" → "Error Recovery"
2. Configure failure probability
3. Run pipeline
4. Watch automatic retry attempts
5. View compensation logic
6. Check final status

## Features Showcase

### Real-time Updates
- WebSocket connection for live execution updates
- Streaming metrics to dashboard
- Live log tailing
- Progress indicators

### Interactive Visualizations
- Animated pipeline graph
- Step execution highlighting
- Data flow arrows
- Error state visualization

### Responsive Design
- Mobile-friendly interface
- Adaptive layouts
- Touch-friendly controls
- Progressive Web App (PWA) ready

## Troubleshooting

### Backend Issues

**Port 5555 already in use:**
```bash
# Use different port
uvicorn main:app --reload --port 8001
```

**Database connection errors:**
```bash
# Reset database
rm metrics.db
# Restart backend
```

### Frontend Issues

**Port 3000 already in use:**
```bash
# Use different port
PORT=3001 npm start
```

**API connection errors:**
```bash
# Check backend is running
curl http://localhost:5555/health

# Update .env with correct API URL
```

### WebSocket Issues

**Connection refused:**
- Ensure backend WebSocket endpoint is accessible
- Check CORS configuration
- Verify firewall settings

## Contributing

See main [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](../LICENSE)

## Support

- **Documentation**: [docs/](../docs/)
- **Main README**: [../README.md](../README.md)
- **Issues**: GitHub Issues

---

**Built to showcase IA Modules capabilities** 🚀
