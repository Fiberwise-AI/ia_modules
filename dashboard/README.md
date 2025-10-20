# IA Modules Dashboard

Web-based dashboard for visual pipeline design, real-time monitoring, and debugging.

## Overview

The IA Modules Dashboard provides a modern web interface for:

1. **Visual Pipeline Designer** - Drag-and-drop pipeline creation
2. **Real-Time Monitoring** - Live execution tracking with WebSockets
3. **Pipeline Debugger** - Step-by-step debugging with variable inspection
4. **Metrics Dashboard** - Performance and cost monitoring
5. **Plugin Management** - Browse and configure plugins

## Architecture

```
dashboard/
├── api/                    # FastAPI backend
│   ├── main.py            # FastAPI app and routes
│   ├── models.py          # Pydantic models
│   ├── services.py        # Business logic
│   └── __init__.py
├── frontend/               # React frontend (coming soon)
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── hooks/
│   │   └── utils/
│   ├── package.json
│   └── vite.config.js
└── requirements.txt
```

## Backend API

### Quick Start

```bash
# Install dependencies
cd ia_modules/dashboard
pip install -r requirements.txt

# Run the API server
python -m api.main

# Or with uvicorn
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs (Swagger UI)
- **ReDoc**: http://localhost:8000/redoc

### API Endpoints

#### Pipeline Management

```http
GET    /api/pipelines              # List pipelines
POST   /api/pipelines              # Create pipeline
GET    /api/pipelines/{id}         # Get pipeline
PUT    /api/pipelines/{id}         # Update pipeline
DELETE /api/pipelines/{id}         # Delete pipeline
POST   /api/pipelines/validate     # Validate config
```

#### Execution

```http
POST   /api/pipelines/{id}/execute      # Execute pipeline
GET    /api/executions/{id}/status      # Get status
GET    /api/executions/{id}/logs        # Get logs
POST   /api/executions/{id}/cancel      # Cancel execution
```

#### Telemetry

```http
GET    /api/metrics                # Get metrics
GET    /api/metrics/prometheus     # Prometheus format
GET    /api/benchmarks             # Benchmark history
```

#### Plugins

```http
GET    /api/plugins                # List plugins
```

#### WebSocket

```
WS     /ws/pipeline/{execution_id} # Real-time updates
```

### Example Usage

#### Create a Pipeline

```python
import httpx
import asyncio

async def create_pipeline():
    async with httpx.AsyncClient() as client:
        pipeline = {
            "name": "Data Processing Pipeline",
            "description": "Fetch and process data",
            "config": {
                "name": "data_pipeline",
                "steps": [
                    {
                        "id": "fetch",
                        "name": "fetch_data",
                        "module": "steps.fetch_data",
                        "class": "FetchDataStep",
                        "config": {}
                    }
                ],
                "flow": {
                    "start_at": "fetch",
                    "paths": [
                        {
                            "from_step": "fetch",
                            "to_step": "end_with_success"
                        }
                    ]
                }
            },
            "tags": ["data", "processing"]
        }

        response = await client.post(
            "http://localhost:8000/api/pipelines",
            json=pipeline
        )

        print(response.json())

asyncio.run(create_pipeline())
```

#### Execute a Pipeline

```python
import httpx
import asyncio

async def execute_pipeline(pipeline_id: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"http://localhost:8000/api/pipelines/{pipeline_id}/execute",
            json={
                "input_data": {
                    "query": "test data"
                }
            }
        )

        result = response.json()
        execution_id = result["execution_id"]
        print(f"Execution started: {execution_id}")

        return execution_id

asyncio.run(execute_pipeline("your-pipeline-id"))
```

#### Real-Time Monitoring (WebSocket)

```python
import asyncio
import websockets
import json

async def monitor_execution(execution_id: str):
    uri = f"ws://localhost:8000/ws/pipeline/{execution_id}"

    async with websockets.connect(uri) as websocket:
        print(f"Connected to execution {execution_id}")

        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)

                print(f"\n[{data['type']}] {data['timestamp']}")
                print(json.dumps(data['data'], indent=2))

                # Break if execution completed or failed
                if data['type'] in ['execution_completed', 'execution_failed']:
                    break

            except websockets.exceptions.ConnectionClosed:
                print("Connection closed")
                break

asyncio.run(monitor_execution("your-execution-id"))
```

### WebSocket Message Types

The WebSocket sends real-time updates during pipeline execution:

```python
# Execution lifecycle
{
    "type": "execution_started",
    "execution_id": "uuid",
    "timestamp": "2025-10-19T...",
    "data": {"pipeline_id": "uuid"}
}

{
    "type": "step_started",
    "execution_id": "uuid",
    "timestamp": "2025-10-19T...",
    "data": {"step_name": "fetch_data"}
}

{
    "type": "step_completed",
    "execution_id": "uuid",
    "timestamp": "2025-10-19T...",
    "data": {
        "step_name": "fetch_data",
        "duration_seconds": 1.23,
        "output": {...}
    }
}

{
    "type": "log_message",
    "execution_id": "uuid",
    "timestamp": "2025-10-19T...",
    "data": {
        "level": "info",
        "message": "Processing batch 1/10",
        "step_name": "process_data"
    }
}

{
    "type": "progress_update",
    "execution_id": "uuid",
    "timestamp": "2025-10-19T...",
    "data": {
        "progress_percent": 45.5,
        "current_step": "process_data",
        "items_processed": 450,
        "total_items": 1000
    }
}

{
    "type": "metrics_update",
    "execution_id": "uuid",
    "timestamp": "2025-10-19T...",
    "data": {
        "duration_seconds": 5.23,
        "memory_mb": 234.5,
        "cpu_percent": 45.2,
        "cost_usd": 0.15
    }
}

{
    "type": "execution_completed",
    "execution_id": "uuid",
    "timestamp": "2025-10-19T...",
    "data": {
        "duration_seconds": 10.5,
        "output": {...}
    }
}
```

## Frontend (Coming Soon)

The React frontend is planned with the following features:

### Pipeline Designer

- Drag-and-drop interface with React Flow
- Visual step connector
- Step configuration forms
- Condition builder
- Live JSON preview
- Real-time validation

### Monitoring Dashboard

- Live execution viewer
- Step progress bars
- Real-time logs
- Performance charts (Chart.js)
- Cost tracking
- Error alerts

### Debugger

- Step-by-step execution
- Variable inspection
- Breakpoints
- Time-travel debugging
- Mock data injection

## Development

### Running Tests

```bash
cd ia_modules
python -m pytest dashboard/tests/ -v
```

### API Documentation

The API is self-documenting via OpenAPI/Swagger:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### Adding New Endpoints

1. Add model to `models.py`:
```python
class MyRequest(BaseModel):
    field: str
```

2. Add service method to `services.py`:
```python
async def my_service_method(self, request: MyRequest):
    # Implementation
    pass
```

3. Add endpoint to `main.py`:
```python
@app.post("/api/my-endpoint")
async def my_endpoint(request: MyRequest):
    result = await service.my_service_method(request)
    return result
```

## Configuration

### Environment Variables

```bash
# .env file
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8000
DASHBOARD_RELOAD=true
LOG_LEVEL=info
PIPELINES_DIR=./pipelines
```

### CORS Configuration

Update `main.py` to allow your frontend origin:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  dashboard-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./pipelines:/app/pipelines
    environment:
      - LOG_LEVEL=info
```

## Roadmap

### Phase 1: Backend API ✅ (Current)
- [x] RESTful API for pipelines
- [x] WebSocket for real-time updates
- [x] Execution management
- [x] Telemetry integration
- [x] Plugin listing

### Phase 2: Frontend UI (Week 2)
- [ ] React application setup
- [ ] Visual pipeline designer
- [ ] Real-time monitoring
- [ ] Metrics dashboard

### Phase 3: Advanced Features (Week 3)
- [ ] Pipeline debugger
- [ ] Variable inspection
- [ ] Breakpoints
- [ ] Mock data injection
- [ ] Performance profiling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

Part of IA Modules - See main LICENSE file
