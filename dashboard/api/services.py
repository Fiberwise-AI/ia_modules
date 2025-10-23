"""
Service layer for Dashboard API
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import uuid

from fastapi import WebSocket
from .models import (
    Pipeline, PipelineCreate, PipelineUpdate,
    ExecutionStatus, ExecutionStatusEnum,
    StepStatus, MetricsResponse, BenchmarkResponse,
    WSMessage, WSMessageType, LogMessage, LogLevel,
    ValidationResult, ValidationError
)

logger = logging.getLogger(__name__)


# ============================================================================
# Pipeline Service
# ============================================================================

class PipelineService:
    """Service for managing pipelines"""

    def __init__(self):
        self.pipelines: Dict[str, Pipeline] = {}
        self.pipelines_dir = Path("pipelines")
        self.pipelines_dir.mkdir(exist_ok=True)

    async def initialize(self):
        """Initialize service - load existing pipelines"""
        logger.info("Initializing PipelineService...")
        await self._load_pipelines_from_disk()
        logger.info(f"Loaded {len(self.pipelines)} pipelines")

    async def _load_pipelines_from_disk(self):
        """Load pipelines from JSON files"""
        for file_path in self.pipelines_dir.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    pipeline = Pipeline(**data)
                    self.pipelines[pipeline.id] = pipeline
            except Exception as e:
                logger.error(f"Failed to load pipeline from {file_path}: {e}")

    async def _save_pipeline_to_disk(self, pipeline: Pipeline):
        """Save pipeline to disk"""
        file_path = self.pipelines_dir / f"{pipeline.id}.json"
        with open(file_path, 'w') as f:
            json.dump(pipeline.dict(), f, indent=2, default=str)

    async def list_pipelines(
        self,
        skip: int = 0,
        limit: int = 100,
        search: Optional[str] = None
    ) -> List[Pipeline]:
        """List pipelines with pagination and search"""
        pipelines = list(self.pipelines.values())

        if search:
            search_lower = search.lower()
            pipelines = [
                p for p in pipelines
                if search_lower in p.name.lower() or
                   (p.description and search_lower in p.description.lower())
            ]

        # Sort by updated_at descending
        pipelines.sort(key=lambda p: p.updated_at, reverse=True)

        return pipelines[skip:skip + limit]

    async def get_pipeline(self, pipeline_id: str) -> Optional[Pipeline]:
        """Get pipeline by ID"""
        return self.pipelines.get(pipeline_id)

    async def create_pipeline(self, pipeline_create: PipelineCreate) -> Pipeline:
        """Create a new pipeline"""
        pipeline_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        pipeline = Pipeline(
            id=pipeline_id,
            name=pipeline_create.name,
            description=pipeline_create.description,
            config=pipeline_create.config,
            tags=pipeline_create.tags,
            created_at=now,
            updated_at=now
        )

        self.pipelines[pipeline_id] = pipeline
        await self._save_pipeline_to_disk(pipeline)

        logger.info(f"Created pipeline: {pipeline.name} ({pipeline_id})")
        return pipeline

    async def update_pipeline(
        self,
        pipeline_id: str,
        pipeline_update: PipelineUpdate
    ) -> Optional[Pipeline]:
        """Update an existing pipeline"""
        pipeline = self.pipelines.get(pipeline_id)
        if not pipeline:
            return None

        # Update fields
        if pipeline_update.name is not None:
            pipeline.name = pipeline_update.name
        if pipeline_update.description is not None:
            pipeline.description = pipeline_update.description
        if pipeline_update.config is not None:
            pipeline.config = pipeline_update.config
        if pipeline_update.tags is not None:
            pipeline.tags = pipeline_update.tags
        if pipeline_update.enabled is not None:
            pipeline.enabled = pipeline_update.enabled

        pipeline.updated_at = datetime.now(timezone.utc)

        await self._save_pipeline_to_disk(pipeline)

        logger.info(f"Updated pipeline: {pipeline.name} ({pipeline_id})")
        return pipeline

    async def delete_pipeline(self, pipeline_id: str) -> bool:
        """Delete a pipeline"""
        if pipeline_id not in self.pipelines:
            return False

        # Remove from memory
        pipeline = self.pipelines.pop(pipeline_id)

        # Remove from disk
        file_path = self.pipelines_dir / f"{pipeline_id}.json"
        if file_path.exists():
            file_path.unlink()

        logger.info(f"Deleted pipeline: {pipeline.name} ({pipeline_id})")
        return True

    async def validate_pipeline(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate pipeline configuration"""
        from ia_modules.cli.validate import validate_pipeline_json

        errors = []
        warnings = []

        try:
            # Use existing validation from CLI
            issues = validate_pipeline_json(config)

            for issue in issues:
                error = ValidationError(
                    field=issue.get("field", "general"),
                    message=issue.get("message", "Unknown error"),
                    severity=issue.get("severity", "error")
                )

                if error.severity == "error":
                    errors.append(error)
                else:
                    warnings.append(error)

            return ValidationResult(
                valid=len(errors) == 0,
                errors=errors,
                warnings=warnings
            )

        except Exception as e:
            return ValidationResult(
                valid=False,
                errors=[ValidationError(
                    field="general",
                    message=str(e),
                    severity="error"
                )]
            )

    async def count_pipelines(self) -> int:
        """Count total pipelines"""
        return len(self.pipelines)


# ============================================================================
# Execution Service
# ============================================================================

class ExecutionService:
    """Service for managing pipeline executions"""

    def __init__(self):
        self.executions: Dict[str, ExecutionStatus] = {}
        self.logs: Dict[str, List[LogMessage]] = {}

    async def initialize(self):
        """Initialize service"""
        logger.info("Initializing ExecutionService...")

    async def execute_pipeline(
        self,
        pipeline_id: str,
        execution_id: str,
        input_data: Dict[str, Any],
        ws_manager: 'WebSocketManager'
    ):
        """Execute a pipeline"""
        from ia_modules.pipeline.runner import run_pipeline_from_json

        # Create execution status
        execution = ExecutionStatus(
            execution_id=execution_id,
            pipeline_id=pipeline_id,
            status=ExecutionStatusEnum.RUNNING,
            started_at=datetime.now(timezone.utc),
            input_data=input_data,
            steps=[],
            progress_percent=0.0
        )

        self.executions[execution_id] = execution
        self.logs[execution_id] = []

        # Send start message
        await ws_manager.send_message(execution_id, WSMessage(
            type=WSMessageType.EXECUTION_STARTED,
            execution_id=execution_id,
            timestamp=datetime.now(timezone.utc),
            data={"pipeline_id": pipeline_id}
        ))

        try:
            # Load pipeline config
            pipeline_file = Path("pipelines") / f"{pipeline_id}.json"

            # Execute pipeline
            result = await run_pipeline_from_json(
                str(pipeline_file),
                input_data=input_data
            )

            # Update execution status
            execution.status = ExecutionStatusEnum.COMPLETED
            execution.completed_at = datetime.now(timezone.utc)
            execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
            execution.output_data = result
            execution.progress_percent = 100.0

            # Send completion message
            await ws_manager.send_message(execution_id, WSMessage(
                type=WSMessageType.EXECUTION_COMPLETED,
                execution_id=execution_id,
                timestamp=datetime.now(timezone.utc),
                data={
                    "duration_seconds": execution.duration_seconds,
                    "output": result
                }
            ))

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)

            execution.status = ExecutionStatusEnum.FAILED
            execution.completed_at = datetime.now(timezone.utc)
            execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
            execution.error = str(e)

            # Send failure message
            await ws_manager.send_message(execution_id, WSMessage(
                type=WSMessageType.EXECUTION_FAILED,
                execution_id=execution_id,
                timestamp=datetime.now(timezone.utc),
                data={"error": str(e)}
            ))

    async def get_execution_status(self, execution_id: str) -> Optional[ExecutionStatus]:
        """Get execution status"""
        return self.executions.get(execution_id)

    async def get_execution_logs(self, execution_id: str) -> Optional[List[LogMessage]]:
        """Get execution logs"""
        return self.logs.get(execution_id)

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution"""
        execution = self.executions.get(execution_id)
        if not execution or execution.status not in [ExecutionStatusEnum.PENDING, ExecutionStatusEnum.RUNNING]:
            return False

        execution.status = ExecutionStatusEnum.CANCELLED
        execution.completed_at = datetime.now(timezone.utc)
        execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()

        return True

    async def count_active_executions(self) -> int:
        """Count active executions"""
        return sum(
            1 for e in self.executions.values()
            if e.status in [ExecutionStatusEnum.PENDING, ExecutionStatusEnum.RUNNING]
        )

    async def count_executions_today(self) -> int:
        """Count executions started today"""
        today = datetime.now(timezone.utc).date()
        return sum(
            1 for e in self.executions.values()
            if e.started_at.date() == today
        )


# ============================================================================
# Metrics Service
# ============================================================================

class MetricsService:
    """Service for telemetry metrics"""

    async def get_metrics(
        self,
        pipeline_id: Optional[str] = None,
        time_range: str = "1h"
    ) -> MetricsResponse:
        """Get telemetry metrics"""
        from ia_modules.telemetry import get_telemetry

        telemetry = get_telemetry()
        metrics = telemetry.get_metrics()

        # Filter by pipeline_id if provided
        if pipeline_id:
            metrics = [
                m for m in metrics
                if pipeline_id in str(m)
            ]

        # Convert to response format
        from .models import Metric, MetricPoint

        response_metrics = []
        for metric in metrics:
            metric_obj = Metric(
                name=metric.name,
                type=metric.metric_type.value,
                help_text=metric.help_text,
                points=[]
            )
            response_metrics.append(metric_obj)

        return MetricsResponse(
            metrics=response_metrics,
            time_range=time_range,
            pipeline_id=pipeline_id
        )

    async def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        from ia_modules.telemetry import get_telemetry, PrometheusExporter

        telemetry = get_telemetry()
        exporter = PrometheusExporter(prefix="ia_modules")
        exporter.export(telemetry.get_metrics())

        return exporter.get_metrics_text()

    async def get_benchmarks(
        self,
        pipeline_id: Optional[str] = None,
        limit: int = 50
    ) -> List[BenchmarkResponse]:
        """Get benchmark history"""
        # Note: Benchmark history storage will be implemented in v0.0.4
        # For now, benchmarks are run on-demand only
        return []


# ============================================================================
# WebSocket Manager
# ============================================================================

class WebSocketManager:
    """Manage WebSocket connections for real-time updates"""

    def __init__(self):
        self.connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, execution_id: str, websocket: WebSocket):
        """Connect a WebSocket for an execution"""
        await websocket.accept()

        if execution_id not in self.connections:
            self.connections[execution_id] = []

        self.connections[execution_id].append(websocket)
        logger.info(f"WebSocket connected for execution {execution_id}")

    async def disconnect(self, execution_id: str):
        """Disconnect WebSocket"""
        if execution_id in self.connections:
            del self.connections[execution_id]
            logger.info(f"WebSocket disconnected for execution {execution_id}")

    async def send_message(self, execution_id: str, message: WSMessage):
        """Send message to all connected clients for an execution"""
        if execution_id not in self.connections:
            return

        # Convert to JSON
        message_json = message.json()

        # Send to all connected clients
        disconnected = []
        for websocket in self.connections[execution_id]:
            try:
                await websocket.send_text(message_json)
            except Exception as e:
                logger.error(f"Failed to send message: {e}")
                disconnected.append(websocket)

        # Remove disconnected clients
        for websocket in disconnected:
            self.connections[execution_id].remove(websocket)

    async def disconnect_all(self):
        """Disconnect all WebSockets"""
        for execution_id in list(self.connections.keys()):
            for websocket in self.connections[execution_id]:
                try:
                    await websocket.close()
                except:
                    pass

        self.connections.clear()
        logger.info("All WebSockets disconnected")
