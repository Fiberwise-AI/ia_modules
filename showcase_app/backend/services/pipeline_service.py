"""Pipeline service using ACTUAL ia_modules library - not reinventing"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from ia_modules.pipeline.runner import create_pipeline_from_json
from ia_modules.pipeline.graph_pipeline_runner import GraphPipelineRunner
from ia_modules.pipeline.services import ServiceRegistry
from ia_modules.pipeline.execution_tracker import ExecutionTracker, ExecutionStatus
from ia_modules.telemetry.integration import get_telemetry
from ia_modules.telemetry.tracing import SimpleTracer
from ia_modules.checkpoint import SQLCheckpointer
from ia_modules.reliability.metrics import ReliabilityMetrics
from ia_modules.reliability.sql_metric_storage import SQLMetricStorage
from ia_modules.database.manager import DatabaseManager
from typing import Dict, Any, Optional, List
import asyncio
import uuid
import logging
import json
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class PipelineService:
    """Service for managing pipelines and execution using ia_modules library"""

    def __init__(self, metrics_service, db_manager):
        self.metrics_service = metrics_service
        self.db_manager = db_manager
        self.pipelines: Dict[str, Dict[str, Any]] = {}
        self.executions: Dict[str, Dict[str, Any]] = {}

        # Set pipelines base path
        self.pipelines_base_path = Path(__file__).parent.parent.parent / 'tests' / 'pipelines'

        # Use database execution tracker
        self.tracker = ExecutionTracker(db_manager)
        logger.info("Using database execution tracker")

        logger.info("Initializing pipeline service with ACTUAL ia_modules library...")

        # Use the ACTUAL library components
        self.services = ServiceRegistry()
        if db_manager:
            self.services.register('database', db_manager)

        # Register execution tracker in services
        self.services.register('execution_tracker', self.tracker)

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

        # Setup HITL manager
        if db_manager:
            from ia_modules.pipeline.hitl_manager import HITLManager
            self.hitl_manager = HITLManager(db_manager)
            self.services.register('hitl_manager', self.hitl_manager)
            logger.info("HITL manager initialized")

        # Load example pipeline JSON files from tests
        self.pipeline_dir = Path(__file__).parent.parent.parent.parent / "tests" / "pipelines"

        logger.info(f"Pipeline service initialized")

    async def _save_execution_to_db(self, execution: Dict[str, Any]):
        """Save execution to PostgreSQL with error handling"""
        if not self.db_manager:
            return

        try:
            # Prepare common parameters
            params = {
                "execution_id": execution["job_id"],
                "pipeline_id": execution["pipeline_id"],
                "pipeline_name": execution.get("pipeline_name", "Unknown"),
                "status": execution["status"],
                "started_at": execution["started_at"],
                "completed_at": execution.get("completed_at"),
                "input_data": json.dumps(execution["input_data"]) if execution.get("input_data") else None,
                "output_data": json.dumps(execution["output_data"]) if execution.get("output_data") else None,
                "error_message": execution.get("error"),
                "total_steps": await self._count_total_steps(execution["job_id"]),
                "completed_steps": await self._count_completed_steps(execution["job_id"]),
                "failed_steps": await self._count_failed_steps(execution["job_id"]),
                "execution_time_ms": self._calculate_duration_ms(execution),
                "metadata_json": json.dumps({"progress": execution.get("progress", 0.0)})
            }

            # Check if record exists
            check_query = "SELECT COUNT(*) as count FROM pipeline_executions WHERE execution_id = :execution_id"
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.db_manager.fetch_one, check_query, {"execution_id": execution["job_id"]})
            
            exists = result and result.get("count", 0) > 0

            if exists:
                # Update existing record
                update_query = """
                    UPDATE pipeline_executions SET
                        status = :status,
                        completed_at = :completed_at,
                        output_data = :output_data,
                        error_message = :error_message,
                        total_steps = :total_steps,
                        completed_steps = :completed_steps,
                        failed_steps = :failed_steps,
                        execution_time_ms = :execution_time_ms,
                        metadata_json = :metadata_json
                    WHERE execution_id = :execution_id
                """
                await self.db_manager.execute_async(update_query, params)
            else:
                # Insert new record
                insert_query = """
                    INSERT INTO pipeline_executions
                    (execution_id, pipeline_id, pipeline_name, status, started_at, completed_at,
                     input_data, output_data, error_message, total_steps, completed_steps, failed_steps,
                     execution_time_ms, metadata_json)
                    VALUES (:execution_id, :pipeline_id, :pipeline_name, :status, :started_at, :completed_at,
                            :input_data, :output_data, :error_message, :total_steps, :completed_steps, :failed_steps,
                            :execution_time_ms, :metadata_json)
                """
                await self.db_manager.execute_async(insert_query, params)

        except Exception as e:
            logger.error(f"Failed to save execution to database: {e}")
            # DatabaseManager doesn't have rollback - connection auto-rolls back on error
            raise

    def _calculate_duration_ms(self, execution: Dict[str, Any]) -> Optional[int]:
        """Calculate execution duration in milliseconds"""
        if not execution.get("started_at") or not execution.get("completed_at"):
            return None
        try:
            start = datetime.fromisoformat(execution["started_at"].replace('Z', '+00:00'))
            end = datetime.fromisoformat(execution["completed_at"].replace('Z', '+00:00'))
            return int((end - start).total_seconds() * 1000)
        except:
            return None

    async def _count_total_steps(self, execution_id: str) -> int:
        """Count total steps from ExecutionTracker"""
        if not self.tracker:
            return 0
        try:
            steps = await self.tracker.get_execution_steps(execution_id)
            return len(steps)
        except Exception as e:
            logger.error(f"Error counting total steps: {e}")
            return 0

    async def _count_completed_steps(self, execution_id: str) -> int:
        """Count completed steps from ExecutionTracker"""
        if not self.tracker:
            return 0
        try:
            steps = await self.tracker.get_execution_steps(execution_id)
            # StepExecutionRecord.status is a StepStatus enum, need to check the value
            return len([s for s in steps if s.status.value == "completed"])
        except Exception as e:
            logger.error(f"Error counting completed steps: {e}")
            return 0

    async def _count_failed_steps(self, execution_id: str) -> int:
        """Count failed steps from ExecutionTracker"""
        if not self.tracker:
            return 0
        try:
            steps = await self.tracker.get_execution_steps(execution_id)
            # StepExecutionRecord.status is a StepStatus enum, need to check the value
            return len([s for s in steps if s.status.value == "failed"])
        except Exception as e:
            logger.error(f"Error counting failed steps: {e}")
            return 0

    async def _load_executions_from_db(self) -> List[Dict[str, Any]]:
        """Load recent executions from PostgreSQL with error handling"""
        if not self.db_manager:
            return []

        try:
            query = """
                SELECT execution_id, pipeline_id, pipeline_name, status, started_at, completed_at,
                       input_data, output_data, error_message, total_steps, completed_steps, failed_steps,
                       execution_time_ms, metadata_json
                FROM pipeline_executions
                ORDER BY started_at DESC
                LIMIT 100
            """

            # fetch_all is sync, so run in thread pool
            loop = asyncio.get_event_loop()
            rows = await loop.run_in_executor(None, self.db_manager.fetch_all, query)

            executions = []
            for row in rows:
                exec_dict = {
                    "job_id": row["execution_id"],
                    "pipeline_id": row["pipeline_id"],
                    "pipeline_name": row["pipeline_name"],
                    "status": row["status"],
                    "started_at": row["started_at"],
                    "completed_at": row["completed_at"],
                    "input_data": json.loads(row["input_data"]) if row.get("input_data") else None,
                    "output_data": json.loads(row["output_data"]) if row.get("output_data") else None,
                    "error": row["error_message"],
                    "progress": json.loads(row["metadata_json"]).get("progress", 0.0) if row.get("metadata_json") else 0.0,
                    "steps": [],
                    "current_step": None
            }
            executions.append(exec_dict)

            return executions
        except Exception as e:
            logger.error(f"Failed to load executions from database: {e}")
            return []

    def load_test_pipelines(self):
        """Load ACTUAL test pipelines from ia_modules/tests/pipelines/ and save to database"""
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
                    pipeline_slug = config.get("name", pipeline_name).lower().replace(" ", "_").replace("-", "_")
                    
                    # Save to database
                    self._save_pipeline_to_db(
                        pipeline_id=pipeline_id,
                        slug=pipeline_slug,
                        name=config.get("name", pipeline_name),
                        description=config.get("description", ""),
                        pipeline_json=config,
                        file_path=str(pipeline_path),
                        is_system=True
                    )
                    
                    # Also keep in memory for backward compatibility
                    self.pipelines[pipeline_id] = {
                        "id": pipeline_id,
                        "name": config.get("name", pipeline_name),
                        "description": config.get("description", ""),
                        "config": config,
                        "tags": self._get_pipeline_tags(config),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "updated_at": datetime.now(timezone.utc).isoformat()
                    }
                    logger.info(f"Loaded and saved test pipeline: {config.get('name')}")
                except Exception as e:
                    logger.warning(f"Failed to load pipeline {pipeline_name}: {e}")
            else:
                logger.warning(f"Pipeline file not found: {pipeline_path}")

    def _save_pipeline_to_db(self, pipeline_id: str, slug: str, name: str, description: str, 
                           pipeline_json: Dict[str, Any], file_path: str, is_system: bool = False):
        """Save pipeline to database"""
        if not self.db_manager:
            logger.warning("No database manager available, skipping database save")
            return
            
        try:
            # Check if pipeline already exists
            check_query = "SELECT id FROM pipelines WHERE id = :id OR slug = :slug"
            check_params = {"id": pipeline_id, "slug": slug}
            
            existing = self.db_manager.fetch_one(check_query, check_params)
            
            if existing:
                logger.info(f"Pipeline {name} already exists in database, skipping import")
                return
            
            # Pipeline doesn't exist, insert it
            insert_query = """
                INSERT INTO pipelines (id, slug, name, description, pipeline_json, file_path, is_system, is_active)
                VALUES (:id, :slug, :name, :description, :pipeline_json, :file_path, :is_system, :is_active)
            """

            params = {
                "id": pipeline_id,
                "slug": slug,
                "name": name,
                "description": description,
                "pipeline_json": json.dumps(pipeline_json),
                "file_path": file_path,
                "is_system": is_system,
                "is_active": True
            }
            
            result = self.db_manager.execute(insert_query, params)
            
            if result:
                logger.info(f"Saved pipeline {name} to database")
            else:
                logger.error(f"Failed to save pipeline {name} to database")
                
        except Exception as e:
            logger.error(f"Error saving pipeline {name} to database: {e}")

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
        """Get pipeline by ID from database"""
        # First check memory cache
        if pipeline_id in self.pipelines:
            return self.pipelines[pipeline_id]
        
        # If not in cache, load from database
        if self.db_manager:
            try:
                query = """
                    SELECT id, slug, name, description, pipeline_json, file_path, is_system, created_at, updated_at
                    FROM pipelines 
                    WHERE id = :id AND is_active = true
                """
                
                import asyncio
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, self.db_manager.fetch_one, query, {"id": pipeline_id})
                
                if result:
                    config = json.loads(result["pipeline_json"])
                    pipeline_data = {
                        "id": result["id"],
                        "name": result["name"],
                        "description": result["description"],
                        "config": config,
                        "tags": self._get_pipeline_tags(config),
                        "created_at": result["created_at"],
                        "updated_at": result["updated_at"],
                        "file_path": result["file_path"],
                        "is_system": result["is_system"]
                    }
                    
                    # Cache in memory for future use
                    self.pipelines[pipeline_id] = pipeline_data
                    return pipeline_data
                    
            except Exception as e:
                logger.error(f"Error loading pipeline {pipeline_id} from database: {e}")
        
        return None

    async def list_pipelines(self) -> List[Dict[str, Any]]:
        """List all active pipelines from database"""
        pipelines = []
        
        # Load from database if available
        if self.db_manager:
            try:
                query = """
                    SELECT id, slug, name, description, pipeline_json, file_path, is_system, created_at, updated_at
                    FROM pipelines 
                    WHERE is_active = true
                    ORDER BY name
                """
                
                import asyncio
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(None, self.db_manager.fetch_all, query)
                
                for result in results:
                    config = json.loads(result["pipeline_json"])
                    pipeline_data = {
                        "id": result["id"],
                        "name": result["name"],
                        "description": result["description"],
                        "config": config,
                        "tags": self._get_pipeline_tags(config),
                        "created_at": result["created_at"],
                        "updated_at": result["updated_at"],
                        "file_path": result["file_path"],
                        "is_system": result["is_system"]
                    }
                    
                    # Cache in memory
                    self.pipelines[result["id"]] = pipeline_data
                    pipelines.append(pipeline_data)
                    
            except Exception as e:
                logger.error(f"Error loading pipelines from database: {e}")
                # Fall back to memory cache
                pipelines = list(self.pipelines.values())
        else:
            # Fall back to memory cache
            pipelines = list(self.pipelines.values())
        
        return pipelines

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

        now = datetime.now(timezone.utc)

        # Track execution start in database if tracker available
        if self.tracker:
            job_id = await self.tracker.start_execution(
                pipeline_id=pipeline_id,
                pipeline_name=pipeline.get("name", "Unknown Pipeline"),
                input_data=input_data,
                total_steps=len(pipeline.get("config", {}).get("steps", []))
            )
        else:
            job_id = str(uuid.uuid4())

        # Create in-memory execution record for backward compatibility
        self.executions[job_id] = {
            "job_id": job_id,
            "pipeline_id": pipeline_id,
            "pipeline_name": pipeline.get("name", "Unknown Pipeline"),
            "status": "running",
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

        # Get WebSocket manager for real-time updates
        from api.websocket import get_ws_manager
        ws_manager = get_ws_manager()

        try:
            execution["status"] = "running"
            
            # Notify WebSocket: Execution started
            await ws_manager.broadcast_execution(job_id, {
                "type": "execution_started",
                "job_id": job_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            # Add pipeline directory to path for imports
            pipeline_config = pipeline["config"]
            pipeline_name = pipeline_config.get("name", "unknown")

            # Find the pipeline directory and add to path
            pipeline_slug = pipeline_name.lower().replace(" ", "_").replace("-", "_")
            pipeline_dir = self.pipeline_dir / pipeline_slug
            if pipeline_dir.exists():
                sys.path.insert(0, str(pipeline_dir))

            logger.info(f"Running pipeline {pipeline_name} with input_data: {input_data}")

            # Create step execution callback for WebSocket notifications
            async def step_callback(step_name: str, event: str, data: dict = None):
                """Callback to notify WebSocket of step events"""
                step_data = {
                    "type": f"step_{event}",
                    "job_id": job_id,
                    "step_name": step_name,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                if data:
                    step_data.update(data)
                
                await ws_manager.broadcast_execution(job_id, step_data)
                
                # Update execution steps list
                if event == "started":
                    execution["steps"].append({
                        "step_name": step_name,
                        "status": "running",
                        "started_at": step_data["timestamp"],
                        "input_data": data.get("input") if data else None
                    })
                elif event in ("completed", "failed"):
                    # Find and update the step
                    for step in execution["steps"]:
                        if step["step_name"] == step_name:
                            step["status"] = "completed" if event == "completed" else "failed"
                            step["completed_at"] = step_data["timestamp"]
                            if data:
                                step["output_data"] = data.get("output")
                                step["error"] = data.get("error")
                                step["duration_ms"] = data.get("duration_ms")
                            break

            # Create execution context with the job_id
            from ia_modules.pipeline.core import ExecutionContext
            execution_context = ExecutionContext(
                execution_id=job_id,
                pipeline_id=execution["pipeline_id"]
            )

            # Use shared services
            graph_runner = GraphPipelineRunner(self.services)

            result = await graph_runner.run_pipeline_from_json(
                pipeline_config,
                input_data,
                execution_context=execution_context
            )

            execution["output_data"] = result.get("output", result)
            execution["steps"] = result.get("steps", [])

            # Check if pipeline is waiting for human input
            if result.get("status") == "waiting_for_human":
                execution["status"] = "waiting_for_human"
                execution["progress"] = len(result.get("steps", [])) / max(len(pipeline_config.get("steps", [])), 1)

                # Notify WebSocket: Execution paused for human input
                pause_message = {
                    "type": "execution_paused",
                    "job_id": job_id,
                    "pipeline_id": execution["pipeline_id"],
                    "status": "waiting_for_human",
                    "waiting_step": result.get("waiting_step"),
                    "interaction_id": result.get("interaction_id"),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                await ws_manager.broadcast_execution(job_id, pause_message)
                await ws_manager.broadcast_pipeline(execution["pipeline_id"], pause_message)
            else:
                execution["status"] = "completed"
                execution["progress"] = 1.0

                # Notify WebSocket: Execution completed
                complete_message = {
                    "type": "execution_completed",
                    "job_id": job_id,
                    "pipeline_id": execution["pipeline_id"],
                    "status": "completed",
                    "output_data": execution["output_data"],
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                await ws_manager.broadcast_execution(job_id, complete_message)
                await ws_manager.broadcast_pipeline(execution["pipeline_id"], complete_message)

            # Extract token usage from result or telemetry
            total_tokens = result.get("total_tokens") if isinstance(result, dict) else None
            estimated_cost = self._estimate_cost(total_tokens) if total_tokens else None

            # Clear spans for next execution
            self.tracer.clear()

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            execution["status"] = "failed"
            execution["error"] = str(e)
            
            # Notify WebSocket: Execution failed
            failed_message = {
                "type": "execution_failed",
                "job_id": job_id,
                "pipeline_id": execution["pipeline_id"],
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            await ws_manager.broadcast_execution(job_id, failed_message)
            await ws_manager.broadcast_pipeline(execution["pipeline_id"], failed_message)
            
            total_tokens = None
            estimated_cost = None

        finally:
            completed_at = datetime.now(timezone.utc)
            execution["completed_at"] = completed_at.isoformat()

            # Update execution tracker
            if self.tracker:
                # Count actual step executions from database
                step_records = await self.tracker.get_execution_steps(job_id)
                completed_steps = len([s for s in step_records if s.status.value == 'completed'])
                failed_steps = len([s for s in step_records if s.status.value == 'failed'])

                # Determine final status
                if execution["status"] == "waiting_for_human":
                    final_status = ExecutionStatus.WAITING_FOR_HUMAN
                elif execution["status"] == "completed":
                    final_status = ExecutionStatus.COMPLETED
                else:
                    final_status = ExecutionStatus.FAILED

                await self.tracker.update_execution_status(
                    execution_id=job_id,
                    status=final_status,
                    completed_steps=completed_steps,
                    failed_steps=failed_steps,
                    output_data=execution.get("output_data"),
                    error_message=execution.get("error")
                )

            # Record metrics using ia_modules ReliabilityMetrics
            if self.reliability_metrics:
                await self.reliability_metrics.record_workflow(
                    workflow_id=job_id,
                    steps=len(execution.get("steps", [])),
                    retries=0,
                    success=execution["status"] == "completed",
                    required_human=False
                )

            # Also record to old metrics_service for backward compatibility (includes cost tracking)
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
        # Always load detailed step information from database for completed executions
        if self.tracker:
            record = await self.tracker.get_execution(job_id)
            if record:
                # Load step executions from database
                step_records = await self.tracker.get_execution_steps(job_id)
                steps = []
                for step_record in step_records:
                    steps.append({
                        "step_id": step_record.step_id,
                        "step_name": step_record.step_name,
                        "step_type": step_record.step_type,
                        "status": step_record.status.value,
                        "started_at": step_record.started_at,  # Already a string
                        "completed_at": step_record.completed_at,  # Already a string
                        "execution_time_ms": step_record.execution_time_ms,  # Use correct field name
                        "input_data": step_record.input_data,
                        "output_data": step_record.output_data,
                        "error_message": step_record.error_message,  # Use correct field name
                        "retry_count": step_record.retry_count,
                        "metadata": step_record.metadata
                    })
                
                return {
                    "job_id": record.execution_id,
                    "pipeline_id": record.pipeline_id,
                    "pipeline_name": record.pipeline_name,
                    "status": record.status.value,
                    "started_at": record.started_at,
                    "completed_at": record.completed_at,
                    "execution_time_ms": record.execution_time_ms,
                    "total_steps": record.total_steps,
                    "completed_steps": record.completed_steps,
                    "failed_steps": record.failed_steps,
                    "input_data": record.input_data,
                    "output_data": record.output_data,
                    "error_message": record.error_message,
                    "error": record.error_message,  # Keep for backwards compatibility
                    "progress": 1.0 if record.status == ExecutionStatus.COMPLETED else 0.5,
                    "steps": steps,
                    "current_step": steps[-1]["step_name"] if steps else None
                }
        
        return None

    async def list_executions(self) -> List[Dict[str, Any]]:
        """List all executions"""
        if self.tracker:
            # Load from database
            exec_records = await self.tracker.get_recent_executions(limit=100)
            executions = []
            for record in exec_records:
                executions.append({
                    "job_id": record.execution_id,
                    "pipeline_id": record.pipeline_id,
                    "pipeline_name": record.pipeline_name,
                    "status": record.status.value,
                    "started_at": record.started_at,
                    "completed_at": record.completed_at,
                    "input_data": record.input_data,
                    "output_data": record.output_data,
                    "error": record.error_message,
                    "progress": 1.0 if record.status == ExecutionStatus.COMPLETED else 0.5,
                    "steps": [],
                    "current_step": None
                })
            return executions
        else:
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

    async def resume_from_hitl(
        self,
        execution_id: str,
        interaction_id: str,
        human_input: Dict[str, Any]
    ):
        """
        Resume a pipeline execution that was paused for HITL interaction.

        Args:
            execution_id: The execution ID that was paused
            interaction_id: The HITL interaction ID
            human_input: The human's response data

        This will continue the pipeline from where it paused, injecting the human
        input into the waiting step and continuing execution.
        """
        logger.info(f"Resuming execution {execution_id} from HITL interaction {interaction_id}")

        # Get the execution record to find which pipeline and step was waiting
        execution_record = await self.tracker.get_execution(execution_id)
        if not execution_record:
            raise ValueError(f"Execution {execution_id} not found")

        pipeline_id = execution_record.pipeline_id

        # Load the pipeline from database
        pipeline_record = self.db_manager.fetch_one(
            "SELECT pipeline_json FROM pipelines WHERE id = :id",
            {"id": pipeline_id}
        )

        if not pipeline_record:
            raise ValueError(f"Pipeline {pipeline_id} not found in database")

        config = json.loads(pipeline_record['pipeline_json']) if isinstance(pipeline_record['pipeline_json'], str) else pipeline_record['pipeline_json']

        # Get the saved execution state from HITL interaction
        from ia_modules.pipeline.hitl_manager import HITLManager
        hitl_manager = HITLManager(self.db_manager)
        execution_state = await hitl_manager.get_execution_state(interaction_id)

        if not execution_state:
            raise ValueError(f"No execution state found for interaction {interaction_id}")

        logger.info(f"Restoring execution from step {execution_state['current_step']}")

        # Restore execution context
        exec_ctx_data = execution_state['execution_context']
        from ia_modules.pipeline.core import ExecutionContext
        execution_context = ExecutionContext(
            execution_id=exec_ctx_data['execution_id'],
            pipeline_id=exec_ctx_data['pipeline_id'],
            user_id=exec_ctx_data.get('user_id'),
            thread_id=exec_ctx_data.get('thread_id'),
            metadata=exec_ctx_data.get('metadata', {})
        )

        # Merge human input into current_data for the next step
        current_data = execution_state['current_data'].copy()
        current_data.update(human_input)

        # Create a new runner with the services (includes db, tracker, hitl_manager)
        runner = GraphPipelineRunner(self.services)

        # Use the config loaded from database instead of execution_state's partial config
        config_dict = config

        # Find the next step after the HITL step
        hitl_step_name = execution_state['current_step']
        flow_paths = config_dict.get('flow', {}).get('paths', [])

        next_step = None
        for path in flow_paths:
            if path.get('from') == hitl_step_name:
                next_step = path.get('to')
                break

        if not next_step:
            logger.info(f"No next step after HITL step {hitl_step_name} - execution complete")
            # Mark execution as completed
            from ia_modules.pipeline.execution_tracker import ExecutionStatus
            await self.tracker.update_execution_status(
                execution_id=execution_id,
                status=ExecutionStatus.COMPLETED
            )

            # Broadcast completion via WebSocket
            from ia_modules.showcase_app.backend.api.websockets import ws_manager
            complete_message = {
                "type": "execution_completed",
                "job_id": execution_id,
                "pipeline_id": pipeline_id,
                "status": "completed",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            await ws_manager.broadcast_execution(execution_id, complete_message)
            await ws_manager.broadcast_pipeline(pipeline_id, complete_message)
            return

        # Create a continuation config starting from next_step
        continuation_config = config_dict.copy()
        continuation_config['flow']['start_at'] = next_step

        # Execute the continuation
        logger.info(f"Continuing execution from step {next_step} with human input")

        try:
            result = await runner.run_pipeline_from_json(
                pipeline_config=continuation_config,
                input_data=current_data,
                execution_context=execution_context,
                use_enhanced_features=True
            )

            # Check if this continuation also paused for HITL
            if result.get('status') == 'waiting_for_human':
                logger.info(f"Execution paused again at step {result.get('waiting_step')}")
                # Broadcast pause via WebSocket
                from ia_modules.showcase_app.backend.api.websockets import ws_manager
                pause_message = {
                    "type": "execution_paused",
                    "job_id": execution_id,
                    "pipeline_id": pipeline_id,
                    "status": "waiting_for_human",
                    "waiting_step": result.get("waiting_step"),
                    "interaction_id": result.get("interaction_id"),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                await ws_manager.broadcast_execution(execution_id, pause_message)
                await ws_manager.broadcast_pipeline(pipeline_id, pause_message)
            else:
                # Mark execution as completed or failed
                from ia_modules.pipeline.execution_tracker import ExecutionStatus
                final_status = ExecutionStatus.COMPLETED if result.get('status') != 'failed' else ExecutionStatus.FAILED
                await self.tracker.update_execution_status(
                    execution_id=execution_id,
                    status=final_status
                )

                # Broadcast completion via WebSocket
                from ia_modules.showcase_app.backend.api.websockets import ws_manager
                complete_message = {
                    "type": "execution_completed" if final_status == ExecutionStatus.COMPLETED else "execution_failed",
                    "job_id": execution_id,
                    "pipeline_id": pipeline_id,
                    "status": "completed" if final_status == ExecutionStatus.COMPLETED else "failed",
                    "output_data": result.get("output"),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                await ws_manager.broadcast_execution(execution_id, complete_message)
                await ws_manager.broadcast_pipeline(pipeline_id, complete_message)

                logger.info(f"Execution {execution_id} completed after HITL resume")

        except Exception as e:
            logger.error(f"Error during HITL resume: {e}")
            from ia_modules.pipeline.execution_tracker import ExecutionStatus
            await self.tracker.update_execution_status(
                execution_id=execution_id,
                status=ExecutionStatus.FAILED,
                error_message=str(e)
            )

            # Broadcast failure via WebSocket
            from ia_modules.showcase_app.backend.api.websockets import ws_manager
            failed_message = {
                "type": "execution_failed",
                "job_id": execution_id,
                "pipeline_id": pipeline_id,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            await ws_manager.broadcast_execution(execution_id, failed_message)
            await ws_manager.broadcast_pipeline(pipeline_id, failed_message)

            raise
