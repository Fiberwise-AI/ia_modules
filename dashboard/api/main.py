"""
IA Modules Dashboard API

FastAPI backend for the visual pipeline designer and monitoring dashboard.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
import uuid

from .models import (
    Pipeline, PipelineCreate, PipelineUpdate,
    ExecutionRequest, ExecutionResponse, ExecutionStatus,
    MetricsResponse, BenchmarkResponse
)
from .services import (
    PipelineService,
    ExecutionService,
    MetricsService,
    WebSocketManager
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="IA Modules Dashboard API",
    description="Visual pipeline designer and monitoring dashboard",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # Vite default
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Services
pipeline_service = PipelineService()
execution_service = ExecutionService()
metrics_service = MetricsService()
ws_manager = WebSocketManager()


# ============================================================================
# Pipeline Management Endpoints
# ============================================================================

@app.get("/api/pipelines", response_model=List[Pipeline])
async def list_pipelines(
    skip: int = 0,
    limit: int = 100,
    search: Optional[str] = None
):
    """List all pipelines with optional search"""
    pipelines = await pipeline_service.list_pipelines(skip=skip, limit=limit, search=search)
    return pipelines


@app.post("/api/pipelines", response_model=Pipeline, status_code=201)
async def create_pipeline(pipeline: PipelineCreate):
    """Create a new pipeline"""
    try:
        created = await pipeline_service.create_pipeline(pipeline)
        return created
    except Exception as e:
        logger.error(f"Failed to create pipeline: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/pipelines/{pipeline_id}", response_model=Pipeline)
async def get_pipeline(pipeline_id: str):
    """Get a specific pipeline by ID"""
    pipeline = await pipeline_service.get_pipeline(pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    return pipeline


@app.put("/api/pipelines/{pipeline_id}", response_model=Pipeline)
async def update_pipeline(pipeline_id: str, pipeline: PipelineUpdate):
    """Update an existing pipeline"""
    try:
        updated = await pipeline_service.update_pipeline(pipeline_id, pipeline)
        if not updated:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        return updated
    except Exception as e:
        logger.error(f"Failed to update pipeline: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/pipelines/{pipeline_id}", status_code=204)
async def delete_pipeline(pipeline_id: str):
    """Delete a pipeline"""
    success = await pipeline_service.delete_pipeline(pipeline_id)
    if not success:
        raise HTTPException(status_code=404, detail="Pipeline not found")


@app.post("/api/pipelines/validate")
async def validate_pipeline(pipeline: dict):
    """Validate a pipeline configuration without saving"""
    try:
        result = await pipeline_service.validate_pipeline(pipeline)
        return result
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return {"valid": False, "errors": [str(e)]}


# ============================================================================
# Pipeline Execution Endpoints
# ============================================================================

@app.post("/api/pipelines/{pipeline_id}/execute", response_model=ExecutionResponse)
async def execute_pipeline(pipeline_id: str, request: ExecutionRequest):
    """Execute a pipeline with given input data"""
    try:
        execution_id = str(uuid.uuid4())

        # Start execution in background
        asyncio.create_task(
            execution_service.execute_pipeline(
                pipeline_id=pipeline_id,
                execution_id=execution_id,
                input_data=request.input_data,
                ws_manager=ws_manager
            )
        )

        return ExecutionResponse(
            execution_id=execution_id,
            pipeline_id=pipeline_id,
            status="started",
            started_at=datetime.now(timezone.utc).isoformat()
        )
    except Exception as e:
        logger.error(f"Failed to start execution: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/executions/{execution_id}/status", response_model=ExecutionStatus)
async def get_execution_status(execution_id: str):
    """Get the status of a pipeline execution"""
    status = await execution_service.get_execution_status(execution_id)
    if not status:
        raise HTTPException(status_code=404, detail="Execution not found")
    return status


@app.get("/api/executions/{execution_id}/logs")
async def get_execution_logs(execution_id: str):
    """Get logs for a pipeline execution"""
    logs = await execution_service.get_execution_logs(execution_id)
    if logs is None:
        raise HTTPException(status_code=404, detail="Execution not found")
    return {"execution_id": execution_id, "logs": logs}


@app.post("/api/executions/{execution_id}/cancel")
async def cancel_execution(execution_id: str):
    """Cancel a running pipeline execution"""
    success = await execution_service.cancel_execution(execution_id)
    if not success:
        raise HTTPException(status_code=404, detail="Execution not found or already completed")
    return {"execution_id": execution_id, "status": "cancelled"}


# ============================================================================
# Telemetry & Metrics Endpoints
# ============================================================================

@app.get("/api/metrics", response_model=MetricsResponse)
async def get_metrics(
    pipeline_id: Optional[str] = None,
    time_range: str = "1h"  # 1h, 24h, 7d, 30d
):
    """Get telemetry metrics"""
    metrics = await metrics_service.get_metrics(pipeline_id=pipeline_id, time_range=time_range)
    return metrics


@app.get("/api/metrics/prometheus")
async def get_prometheus_metrics():
    """Get metrics in Prometheus text format"""
    metrics_text = await metrics_service.get_prometheus_metrics()
    return Response(content=metrics_text, media_type="text/plain; version=0.0.4")


@app.get("/api/benchmarks", response_model=List[BenchmarkResponse])
async def get_benchmarks(
    pipeline_id: Optional[str] = None,
    limit: int = 50
):
    """Get benchmark history"""
    benchmarks = await metrics_service.get_benchmarks(pipeline_id=pipeline_id, limit=limit)
    return benchmarks


# ============================================================================
# Plugin Endpoints
# ============================================================================

@app.get("/api/plugins")
async def list_plugins():
    """List all available plugins"""
    from ia_modules.plugins import get_registry

    registry = get_registry()
    plugins = []

    for plugin in registry.list_plugins():
        plugins.append({
            "name": plugin.name,
            "version": plugin.version,
            "type": plugin.plugin_type.value,
            "description": plugin.description,
            "author": plugin.author,
            "config_schema": plugin.config_schema
        })

    return {"plugins": plugins}


# ============================================================================
# WebSocket Endpoint for Real-Time Updates
# ============================================================================

@app.websocket("/ws/pipeline/{execution_id}")
async def websocket_endpoint(websocket: WebSocket, execution_id: str):
    """WebSocket endpoint for real-time pipeline execution updates"""
    await ws_manager.connect(execution_id, websocket)

    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()

            # Handle ping/pong
            if data == "ping":
                await websocket.send_text("pong")

    except WebSocketDisconnect:
        await ws_manager.disconnect(execution_id)
        logger.info(f"WebSocket disconnected for execution {execution_id}")


# ============================================================================
# Health & Status Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "0.1.0"
    }


@app.get("/api/stats")
async def get_stats():
    """Get dashboard statistics"""
    stats = {
        "total_pipelines": await pipeline_service.count_pipelines(),
        "active_executions": await execution_service.count_active_executions(),
        "total_executions_today": await execution_service.count_executions_today(),
        "telemetry_enabled": True
    }
    return stats


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "timestamp": datetime.now(timezone.utc).isoformat()}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )


# ============================================================================
# Startup & Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting IA Modules Dashboard API...")
    await pipeline_service.initialize()
    await execution_service.initialize()
    logger.info("Dashboard API ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down IA Modules Dashboard API...")
    await ws_manager.disconnect_all()
    logger.info("Shutdown complete")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
