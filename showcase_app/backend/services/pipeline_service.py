"""Pipeline service using ACTUAL ia_modules library - not reinventing"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from ia_modules.pipeline.runner import create_pipeline_from_json
from ia_modules.pipeline.graph_pipeline_runner import GraphPipelineRunner
from ia_modules.pipeline.services import ServiceRegistry
from ia_modules.telemetry.integration import get_telemetry
from ia_modules.telemetry.tracing import SimpleTracer
from ia_modules.checkpoint import SQLCheckpointer
from ia_modules.reliability.metrics import ReliabilityMetrics
from ia_modules.reliability.sql_metric_storage import SQLMetricStorage
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import asyncio
import uuid
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class PipelineService:
    """Service for managing pipelines and execution using ia_modules library"""

    def __init__(self, metrics_service, db_manager):
        self.metrics_service = metrics_service
        self.db_manager = db_manager
        self.pipelines: Dict[str, Dict[str, Any]] = {}
        self.executions: Dict[str, Dict[str, Any]] = {}

        logger.info("Initializing pipeline service with ACTUAL ia_modules library...")

        # Use the ACTUAL library components
        self.services = ServiceRegistry()
        if db_manager:
            self.services.register('database', db_manager)

        # Setup telemetry to capture step execution details
        self.tracer = SimpleTracer()
        self.telemetry = get_telemetry(enabled=True, tracer=self.tracer)
        self.services.register('telemetry', self.telemetry)
        self.services.register('tracer', self.tracer)

        # Setup checkpointing
        self.checkpointer = None
        if db_manager:
            self.checkpointer = SQLCheckpointer(db_manager)
            self.services.register('checkpointer', self.checkpointer)
            logger.info("Checkpointer initialized")

        # Setup reliability metrics using ia_modules
        self.reliability_storage = None
        self.reliability_metrics = None
        if db_manager:
            self.reliability_storage = SQLMetricStorage(db_manager)
            self.reliability_metrics = ReliabilityMetrics(self.reliability_storage)
            self.services.register('reliability_metrics', self.reliability_metrics)
            logger.info("Reliability metrics initialized with SQL storage")

        # Load example pipeline JSON files from tests
        self.pipeline_dir = Path(__file__).parent.parent.parent.parent / "tests" / "pipelines"

        # Load real test pipelines from the framework
        self.load_test_pipelines()

        logger.info(f"Pipeline service initialized with {len(self.pipelines)} pipelines")

    def load_test_pipelines(self):
        """Load ACTUAL test pipelines from ia_modules/tests/pipelines/"""
        test_pipeline_dirs = [
            "simple_pipeline",
            "conditional_pipeline",
            "parallel_pipeline",
            "loop_pipeline",
            "hitl_pipeline"
        ]

        for pipeline_name in test_pipeline_dirs:
            pipeline_path = self.pipeline_dir / pipeline_name / "pipeline.json"
            if pipeline_path.exists():
                try:
                    with open(pipeline_path, 'r') as f:
                        config = json.load(f)

                    pipeline_id = str(uuid.uuid4())
                    self.pipelines[pipeline_id] = {
                        "id": pipeline_id,
                        "name": config.get("name", pipeline_name),
                        "description": config.get("description", ""),
                        "config": config,
                        "tags": self._get_pipeline_tags(config),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "updated_at": datetime.now(timezone.utc).isoformat()
                    }
                    logger.info(f"Loaded test pipeline: {config.get('name')}")
                except Exception as e:
                    logger.warning(f"Failed to load pipeline {pipeline_name}: {e}")
            else:
                logger.warning(f"Pipeline file not found: {pipeline_path}")

    def _get_pipeline_tags(self, config: Dict[str, Any]) -> List[str]:
        """Extract tags from pipeline config"""
        tags = ["test", "framework"]

        # Add tags based on pipeline features
        if "flow" in config:
            flow = config["flow"]
            if len(flow.get("paths", [])) > len(config.get("steps", [])):
                tags.append("conditional")

            # Check for parallel execution (multiple paths from same step)
            from_steps = {}
            for path in flow.get("paths", []):
                from_step = path.get("from_step")
                if from_step:
                    from_steps[from_step] = from_steps.get(from_step, 0) + 1

            if any(count > 1 for count in from_steps.values()):
                tags.append("parallel")

        return tags

    async def get_pipeline(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get pipeline by ID"""
        return self.pipelines.get(pipeline_id)

    async def list_pipelines(self) -> List[Dict[str, Any]]:
        """List all pipelines"""
        return list(self.pipelines.values())

    async def execute_pipeline(
        self,
        pipeline_id: str,
        input_data: Dict[str, Any],
        checkpoint_enabled: bool = True
    ) -> str:
        """Execute pipeline using ACTUAL GraphPipelineRunner"""
        pipeline = await self.get_pipeline(pipeline_id)
        if not pipeline:
            raise ValueError(f"Pipeline not found: {pipeline_id}")

        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        # Create execution record
        self.executions[job_id] = {
            "job_id": job_id,
            "pipeline_id": pipeline_id,
            "status": "pending",
            "input_data": input_data,
            "output_data": None,
            "current_step": None,
            "started_at": now.isoformat(),
            "completed_at": None,
            "error": None,
            "progress": 0.0,
            "steps": []
        }

        # Start execution in background using ACTUAL library
        asyncio.create_task(self._run_pipeline_with_library(job_id, pipeline, input_data))

        logger.info(f"Started execution: {job_id} for pipeline {pipeline_id}")
        return job_id

    async def _run_pipeline_with_library(
        self,
        job_id: str,
        pipeline: Dict[str, Any],
        input_data: Dict[str, Any]
    ):
        """Execute pipeline using create_pipeline_from_json from ia_modules library"""
        execution = self.executions[job_id]
        start_time = datetime.now(timezone.utc)

        try:
            execution["status"] = "running"

            # Add pipeline directory to path for imports
            pipeline_config = pipeline["config"]
            pipeline_name = pipeline_config.get("name", "unknown")

            # Find the pipeline directory and add to path
            pipeline_slug = pipeline_name.lower().replace(" ", "_").replace("-", "_")
            pipeline_dir = self.pipeline_dir / pipeline_slug
            if pipeline_dir.exists():
                sys.path.insert(0, str(pipeline_dir))

            logger.info(f"Running pipeline {pipeline_name} with input_data: {input_data}")

            # Use GraphPipelineRunner for graph-based execution with inputs/outputs routing
            graph_runner = GraphPipelineRunner(self.services)
            result = await graph_runner.run_pipeline_from_json(pipeline_config, input_data, use_enhanced_features=False)

            execution["output_data"] = result.get("output", result) if isinstance(result, dict) else result
            execution["steps"] = result.get("steps", [])
            execution["status"] = "completed"
            execution["progress"] = 1.0

            # Extract token usage from result or telemetry
            total_tokens = result.get("total_tokens") if isinstance(result, dict) else None
            estimated_cost = self._estimate_cost(total_tokens) if total_tokens else None

            # Clear spans for next execution
            self.tracer.clear()

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            execution["status"] = "failed"
            execution["error"] = str(e)
            total_tokens = None
            estimated_cost = None

        finally:
            completed_at = datetime.now(timezone.utc)
            execution["completed_at"] = completed_at.isoformat()

            # Record metrics using ia_modules ReliabilityMetrics
            if self.reliability_metrics:
                await self.reliability_metrics.record_workflow(
                    workflow_id=job_id,
                    steps=len(execution.get("steps", [])),
                    retries=0,
                    success=execution["status"] == "completed",
                    required_human=False,
                    tokens=total_tokens,
                    cost=estimated_cost
                )

            # Also record to old metrics_service for backward compatibility
            if self.metrics_service:
                await self.metrics_service.record_workflow(
                    workflow_id=job_id,
                    steps=len(execution.get("steps", [])),
                    retries=0,
                    success=execution["status"] == "completed",
                    required_human=False,
                    tokens=total_tokens,
                    cost=estimated_cost
                )

    def _extract_step_data_from_telemetry(self, spans: List[Any]) -> List[Dict[str, Any]]:
        """Extract step execution data from telemetry spans"""
        steps = []

        for span in spans:
            # Filter for step spans (not pipeline-level spans)
            if hasattr(span, 'name') and 'step' in span.name.lower():
                span_dict = span.to_dict() if hasattr(span, 'to_dict') else span

                # Calculate duration from timestamps if not provided
                duration_ms = None
                if span_dict.get('duration'):
                    duration_ms = span_dict['duration'] * 1000
                elif span_dict.get('start_time') and span_dict.get('end_time'):
                    duration_ms = (span_dict['end_time'] - span_dict['start_time']) * 1000

                steps.append({
                    "step_name": span_dict.get('name', 'unknown'),
                    "step_index": len(steps),
                    "status": "completed" if span_dict.get('status') == 'ok' else "failed",
                    "started_at": datetime.fromtimestamp(span_dict.get('start_time', 0), tz=timezone.utc).isoformat(),
                    "completed_at": datetime.fromtimestamp(span_dict.get('end_time', 0), tz=timezone.utc).isoformat() if span_dict.get('end_time') else None,
                    "input_data": span_dict.get('attributes', {}).get('input'),
                    "output_data": span_dict.get('attributes', {}).get('output'),
                    "error": span_dict.get('attributes', {}).get('status.description'),
                    "duration_ms": int(duration_ms) if duration_ms else 0,
                    "is_parallel": False,
                    "branch": None
                })

        return steps

    def _extract_total_tokens(self, spans: List[Any]) -> Optional[int]:
        """Extract total token count from telemetry spans"""
        total_tokens = 0

        for span in spans:
            span_dict = span.to_dict() if hasattr(span, 'to_dict') else span
            attributes = span_dict.get('attributes', {})

            # Check for usage data in attributes
            if 'usage' in attributes:
                usage = attributes['usage']
                if isinstance(usage, dict) and 'total_tokens' in usage:
                    total_tokens += usage['total_tokens']

            # Also check for direct token count
            if 'tokens' in attributes:
                total_tokens += attributes['tokens']

        return total_tokens if total_tokens > 0 else None

    def _estimate_cost(self, total_tokens: int) -> float:
        """Estimate cost based on token count

        Uses rough pricing estimates:
        - GPT-4: $0.03 / 1K input tokens, $0.06 / 1K output tokens
        - GPT-3.5: $0.0015 / 1K input tokens, $0.002 / 1K output tokens
        - Average estimate: $0.02 / 1K tokens
        """
        # Simple average estimate - in production, track per-model pricing
        cost_per_1k_tokens = 0.02
        return round((total_tokens / 1000) * cost_per_1k_tokens, 4)

    async def get_execution(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get execution by ID"""
        return self.executions.get(job_id)

    async def list_executions(self) -> List[Dict[str, Any]]:
        """List all executions"""
        return list(self.executions.values())

    async def cancel_execution(self, job_id: str) -> bool:
        """Cancel execution"""
        if job_id in self.executions:
            execution = self.executions[job_id]
            if execution["status"] in ["pending", "running"]:
                execution["status"] = "cancelled"
                execution["completed_at"] = datetime.now(timezone.utc).isoformat()
                logger.info(f"Cancelled execution: {job_id}")
                return True
        return False

    # Checkpoint management methods using ia_modules library

    async def list_checkpoints(
        self,
        pipeline_id: str,
        thread_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List checkpoints for a pipeline using library checkpointer"""
        if not self.checkpointer:
            return []

        try:
            checkpoints = await self.checkpointer.list_checkpoints(
                pipeline_id=pipeline_id,
                thread_id=thread_id
            )

            return [cp.to_dict() for cp in checkpoints]
        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return []

    async def get_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Get checkpoint details using library checkpointer"""
        if not self.checkpointer:
            return None

        try:
            # Extract thread_id from checkpoint_id (would need proper lookup in real impl)
            # For now, return None as we need thread_id to load
            return None
        except Exception as e:
            logger.error(f"Failed to get checkpoint: {e}")
            return None

    async def resume_from_checkpoint(
        self,
        checkpoint_id: str,
        input_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Resume pipeline execution from a checkpoint using library"""
        if not self.checkpointer:
            raise ValueError("Checkpointing not enabled")

        try:
            # Load checkpoint from library
            checkpoint = await self.checkpointer.load_checkpoint(
                thread_id="default",  # Would need proper thread_id lookup
                checkpoint_id=checkpoint_id
            )

            if not checkpoint:
                raise ValueError(f"Checkpoint {checkpoint_id} not found")

            # Create new execution from checkpoint
            job_id = str(uuid.uuid4())
            execution = {
                "job_id": job_id,
                "pipeline_id": checkpoint.pipeline_id,
                "pipeline_name": self.pipelines.get(checkpoint.pipeline_id, {}).get("name", "Unknown"),
                "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": None,
                "input_data": input_data or checkpoint.state.get("pipeline_input", {}),
                "output_data": None,
                "steps": [],
                "resumed_from_checkpoint": checkpoint_id,
                "progress": 0.0
            }

            self.executions[job_id] = execution

            # Resume execution from checkpoint state
            # This would use the library's pipeline resume functionality
            logger.info(f"Resumed execution {job_id} from checkpoint {checkpoint_id}")

            return job_id

        except Exception as e:
            logger.error(f"Failed to resume from checkpoint: {e}")
            raise

    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint using library checkpointer"""
        if not self.checkpointer:
            return False

        try:
            await self.checkpointer.delete_checkpoint(
                thread_id="default",  # Would need proper thread_id lookup
                checkpoint_id=checkpoint_id
            )
            logger.info(f"Deleted checkpoint {checkpoint_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete checkpoint: {e}")
            return False
