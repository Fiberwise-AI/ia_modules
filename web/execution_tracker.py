"""
Pipeline Execution Tracking System

Provides database persistence and real-time monitoring of pipeline executions
with WebSocket updates for live viewing.
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from ..database.manager import DatabaseManager

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Pipeline execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(Enum):
    """Pipeline step status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ExecutionRecord:
    """Pipeline execution record"""
    execution_id: str
    pipeline_id: str
    pipeline_name: str
    status: ExecutionStatus
    started_at: str
    completed_at: Optional[str] = None
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time_ms: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['status'] = self.status.value
        return result


@dataclass
class StepExecutionRecord:
    """Pipeline step execution record"""
    step_execution_id: str
    execution_id: str
    step_id: str
    step_name: str
    step_type: str
    status: StepStatus
    started_at: str
    completed_at: Optional[str] = None
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time_ms: Optional[int] = None
    retry_count: int = 0
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['status'] = self.status.value
        return result


class ExecutionTracker:
    """Tracks pipeline executions with database persistence and real-time updates"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.active_executions: Dict[str, ExecutionRecord] = {}
        self.websocket_connections: List[Any] = []  # WebSocket connections for real-time updates
        self._initialized = False

    async def initialize(self):
        """Initialize the execution tracker and create database tables"""
        if self._initialized:
            return True

        try:
            # Create execution tracking tables
            await self._create_tables()

            # Load active executions from database
            await self._load_active_executions()

            self._initialized = True
            logger.info("Execution tracker initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize execution tracker: {e}")
            return False

    async def _create_tables(self):
        """Create database tables for execution tracking"""

        # Check if enhanced tracking columns exist, add them if not
        try:
            # Check if total_steps column exists
            result = self.db.fetch_one(
                "SELECT name FROM pragma_table_info('pipeline_executions') WHERE name='total_steps'"
            )

            if not result:
                # Add enhanced tracking columns to existing table
                self.db.execute("ALTER TABLE pipeline_executions ADD COLUMN total_steps INTEGER DEFAULT 0")
                self.db.execute("ALTER TABLE pipeline_executions ADD COLUMN completed_steps INTEGER DEFAULT 0")
                self.db.execute("ALTER TABLE pipeline_executions ADD COLUMN failed_steps INTEGER DEFAULT 0")
                self.db.execute("ALTER TABLE pipeline_executions ADD COLUMN execution_time_ms INTEGER")
                self.db.execute("ALTER TABLE pipeline_executions ADD COLUMN metadata_json TEXT")

        except Exception as e:
            # If pipeline_executions doesn't exist, create full enhanced table
            execution_schema = """
                execution_id TEXT PRIMARY KEY,
                pipeline_id TEXT NOT NULL,
                pipeline_name TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                total_steps INTEGER DEFAULT 0,
                completed_steps INTEGER DEFAULT 0,
                failed_steps INTEGER DEFAULT 0,
                input_data TEXT,
                output_data TEXT,
                error_message TEXT,
                execution_time_ms INTEGER,
                metadata_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            """
            self.db.create_table("pipeline_executions", execution_schema)

        # Check if enhanced step executions table exists
        try:
            result = self.db.fetch_one(
                "SELECT name FROM pragma_table_info('step_executions') WHERE name='step_execution_id'"
            )

            if not result:
                # Create enhanced step executions table
                step_schema = """
                    step_execution_id TEXT PRIMARY KEY,
                    execution_id TEXT NOT NULL,
                    step_id TEXT NOT NULL,
                    step_name TEXT NOT NULL,
                    step_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    input_data TEXT,
                    output_data TEXT,
                    error_message TEXT,
                    execution_time_ms INTEGER,
                    retry_count INTEGER DEFAULT 0,
                    metadata_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (execution_id) REFERENCES pipeline_executions (execution_id)
                """
                self.db.create_table("step_executions_enhanced", step_schema)

        except Exception:
            # Create table if it doesn't exist
            step_schema = """
                step_execution_id TEXT PRIMARY KEY,
                execution_id TEXT NOT NULL,
                step_id TEXT NOT NULL,
                step_name TEXT NOT NULL,
                step_type TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                input_data TEXT,
                output_data TEXT,
                error_message TEXT,
                execution_time_ms INTEGER,
                retry_count INTEGER DEFAULT 0,
                metadata_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (execution_id) REFERENCES pipeline_executions (execution_id)
            """
            self.db.create_table("step_executions_enhanced", step_schema)

        # Create indexes for better query performance
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_executions_pipeline_id ON pipeline_executions(pipeline_id)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_executions_status ON pipeline_executions(status)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_executions_started_at ON pipeline_executions(started_at)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_step_executions_enhanced_execution_id ON step_executions_enhanced(execution_id)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_step_executions_enhanced_status ON step_executions_enhanced(status)")

    async def _load_active_executions(self):
        """Load active executions from database"""
        try:
            active_statuses = [ExecutionStatus.PENDING.value, ExecutionStatus.RUNNING.value]
            query = "SELECT * FROM pipeline_executions WHERE status IN (?, ?) ORDER BY started_at DESC"

            rows = self.db.fetch_all(query, tuple(active_statuses))

            for row in rows:
                execution = ExecutionRecord(
                    execution_id=row['execution_id'],
                    pipeline_id=row['pipeline_id'],
                    pipeline_name=row['pipeline_name'],
                    status=ExecutionStatus(row['status']),
                    started_at=row['started_at'],
                    completed_at=row['completed_at'],
                    total_steps=row.get('total_steps', 0),
                    completed_steps=row.get('completed_steps', 0),
                    failed_steps=row.get('failed_steps', 0),
                    input_data=json.loads(row['input_data']) if row.get('input_data') else None,
                    output_data=json.loads(row['output_data']) if row.get('output_data') else None,
                    error_message=row['error_message'],
                    execution_time_ms=row.get('execution_time_ms'),
                    metadata=json.loads(row['metadata_json']) if row.get('metadata_json') else None
                )
                self.active_executions[execution.execution_id] = execution

            logger.info(f"Loaded {len(self.active_executions)} active executions")

        except Exception as e:
            logger.error(f"Failed to load active executions: {e}")

    async def start_execution(self,
                            pipeline_id: str,
                            pipeline_name: str,
                            input_data: Dict[str, Any],
                            total_steps: int = 0,
                            metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start tracking a new pipeline execution"""

        execution_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        execution = ExecutionRecord(
            execution_id=execution_id,
            pipeline_id=pipeline_id,
            pipeline_name=pipeline_name,
            status=ExecutionStatus.RUNNING,
            started_at=now,
            total_steps=total_steps,
            input_data=input_data,
            metadata=metadata or {}
        )

        # Save to database
        await self._save_execution(execution)

        # Track in memory
        self.active_executions[execution_id] = execution

        # Broadcast update
        await self._broadcast_execution_update(execution)

        logger.info(f"Started execution tracking: {execution_id} for pipeline {pipeline_name}")
        return execution_id

    async def update_execution_status(self,
                                    execution_id: str,
                                    status: ExecutionStatus,
                                    completed_steps: Optional[int] = None,
                                    failed_steps: Optional[int] = None,
                                    output_data: Optional[Dict[str, Any]] = None,
                                    error_message: Optional[str] = None):
        """Update execution status"""

        if execution_id not in self.active_executions:
            logger.warning(f"Execution {execution_id} not found in active executions")
            return

        execution = self.active_executions[execution_id]
        execution.status = status

        if completed_steps is not None:
            execution.completed_steps = completed_steps

        if failed_steps is not None:
            execution.failed_steps = failed_steps

        if output_data is not None:
            execution.output_data = output_data

        if error_message is not None:
            execution.error_message = error_message

        # Set completion time for terminal states
        if status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED]:
            execution.completed_at = datetime.now(timezone.utc).isoformat()

            # Calculate execution time
            if execution.started_at and execution.completed_at:
                start_time = datetime.fromisoformat(execution.started_at.replace('Z', '+00:00'))
                end_time = datetime.fromisoformat(execution.completed_at.replace('Z', '+00:00'))
                execution.execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

            # Remove from active executions if completed
            self.active_executions.pop(execution_id, None)

        # Save to database
        await self._save_execution(execution)

        # Broadcast update
        await self._broadcast_execution_update(execution)

        logger.info(f"Updated execution {execution_id} status to {status.value}")

    async def start_step_execution(self,
                                 execution_id: str,
                                 step_id: str,
                                 step_name: str,
                                 step_type: str,
                                 input_data: Optional[Dict[str, Any]] = None,
                                 metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start tracking a step execution"""

        step_execution_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        step_execution = StepExecutionRecord(
            step_execution_id=step_execution_id,
            execution_id=execution_id,
            step_id=step_id,
            step_name=step_name,
            step_type=step_type,
            status=StepStatus.RUNNING,
            started_at=now,
            input_data=input_data,
            metadata=metadata or {}
        )

        # Save to database
        await self._save_step_execution(step_execution)

        # Broadcast update
        await self._broadcast_step_update(step_execution)

        logger.info(f"Started step execution: {step_execution_id} for step {step_name}")
        return step_execution_id

    async def complete_step_execution(self,
                                    step_execution_id: str,
                                    status: StepStatus,
                                    output_data: Optional[Dict[str, Any]] = None,
                                    error_message: Optional[str] = None,
                                    retry_count: int = 0):
        """Complete a step execution"""

        # Get step execution from database
        step_execution = await self._get_step_execution(step_execution_id)
        if not step_execution:
            logger.warning(f"Step execution {step_execution_id} not found")
            return

        # Update status and completion time
        step_execution.status = status
        step_execution.completed_at = datetime.now(timezone.utc).isoformat()
        step_execution.retry_count = retry_count

        if output_data is not None:
            step_execution.output_data = output_data

        if error_message is not None:
            step_execution.error_message = error_message

        # Calculate execution time
        if step_execution.started_at and step_execution.completed_at:
            start_time = datetime.fromisoformat(step_execution.started_at.replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(step_execution.completed_at.replace('Z', '+00:00'))
            step_execution.execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

        # Save to database
        await self._save_step_execution(step_execution)

        # Update execution step counts
        await self._update_execution_step_counts(step_execution.execution_id)

        # Broadcast update
        await self._broadcast_step_update(step_execution)

        logger.info(f"Completed step execution: {step_execution_id} with status {status.value}")

    async def get_execution(self, execution_id: str) -> Optional[ExecutionRecord]:
        """Get execution record by ID"""

        # Check active executions first
        if execution_id in self.active_executions:
            return self.active_executions[execution_id]

        # Query database
        query = "SELECT * FROM pipeline_executions WHERE execution_id = ?"
        row = self.db.fetch_one(query, (execution_id,))

        if not row:
            return None

        return ExecutionRecord(
            execution_id=row['execution_id'],
            pipeline_id=row['pipeline_id'],
            pipeline_name=row['pipeline_name'],
            status=ExecutionStatus(row['status']),
            started_at=row['started_at'],
            completed_at=row['completed_at'],
            total_steps=row.get('total_steps', 0),
            completed_steps=row.get('completed_steps', 0),
            failed_steps=row.get('failed_steps', 0),
            input_data=json.loads(row['input_data']) if row.get('input_data') else None,
            output_data=json.loads(row['output_data']) if row.get('output_data') else None,
            error_message=row['error_message'],
            execution_time_ms=row.get('execution_time_ms'),
            metadata=json.loads(row['metadata_json']) if row.get('metadata_json') else None
        )

    async def get_execution_steps(self, execution_id: str) -> List[StepExecutionRecord]:
        """Get all step executions for a pipeline execution"""

        query = """
            SELECT * FROM step_executions_enhanced
            WHERE execution_id = ?
            ORDER BY started_at ASC
        """

        rows = self.db.fetch_all(query, (execution_id,))

        steps = []
        for row in rows:
            step = StepExecutionRecord(
                step_execution_id=row['step_execution_id'],
                execution_id=row['execution_id'],
                step_id=row['step_id'],
                step_name=row['step_name'],
                step_type=row['step_type'],
                status=StepStatus(row['status']),
                started_at=row['started_at'],
                completed_at=row['completed_at'],
                input_data=json.loads(row['input_data']) if row['input_data'] else None,
                output_data=json.loads(row['output_data']) if row['output_data'] else None,
                error_message=row['error_message'],
                execution_time_ms=row['execution_time_ms'],
                retry_count=row['retry_count'],
                metadata=json.loads(row['metadata_json']) if row['metadata_json'] else None
            )
            steps.append(step)

        return steps

    async def get_recent_executions(self, limit: int = 50, pipeline_id: Optional[str] = None) -> List[ExecutionRecord]:
        """Get recent pipeline executions"""

        if pipeline_id:
            query = """
                SELECT * FROM pipeline_executions
                WHERE pipeline_id = ?
                ORDER BY started_at DESC
                LIMIT ?
            """
            params = (pipeline_id, limit)
        else:
            query = """
                SELECT * FROM pipeline_executions
                ORDER BY started_at DESC
                LIMIT ?
            """
            params = (limit,)

        rows = self.db.fetch_all(query, params)

        executions = []
        for row in rows:
            execution = ExecutionRecord(
                execution_id=row['execution_id'],
                pipeline_id=row['pipeline_id'],
                pipeline_name=row['pipeline_name'],
                status=ExecutionStatus(row['status']),
                started_at=row['started_at'],
                completed_at=row['completed_at'],
                total_steps=row.get('total_steps', 0),
                completed_steps=row.get('completed_steps', 0),
                failed_steps=row.get('failed_steps', 0),
                input_data=json.loads(row['input_data']) if row.get('input_data') else None,
                output_data=json.loads(row['output_data']) if row.get('output_data') else None,
                error_message=row['error_message'],
                execution_time_ms=row.get('execution_time_ms'),
                metadata=json.loads(row['metadata_json']) if row.get('metadata_json') else None
            )
            executions.append(execution)

        return executions

    async def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics"""

        stats = {}

        # Total executions
        total_result = self.db.fetch_one("SELECT COUNT(*) as count FROM pipeline_executions")
        stats['total_executions'] = total_result['count'] if total_result else 0

        # Executions by status
        status_query = """
            SELECT status, COUNT(*) as count
            FROM pipeline_executions
            GROUP BY status
        """
        status_rows = self.db.fetch_all(status_query)
        stats['by_status'] = {row['status']: row['count'] for row in status_rows}

        # Active executions
        stats['active_executions'] = len(self.active_executions)

        # Recent execution times (last 10 completed)
        time_query = """
            SELECT execution_time_ms
            FROM pipeline_executions
            WHERE execution_time_ms IS NOT NULL
            ORDER BY completed_at DESC
            LIMIT 10
        """
        time_rows = self.db.fetch_all(time_query)
        if time_rows:
            times = [row['execution_time_ms'] for row in time_rows]
            stats['avg_execution_time_ms'] = sum(times) / len(times)
            stats['min_execution_time_ms'] = min(times)
            stats['max_execution_time_ms'] = max(times)

        return stats

    def add_websocket_connection(self, websocket):
        """Add WebSocket connection for real-time updates"""
        self.websocket_connections.append(websocket)

    def remove_websocket_connection(self, websocket):
        """Remove WebSocket connection"""
        if websocket in self.websocket_connections:
            self.websocket_connections.remove(websocket)

    async def _save_execution(self, execution: ExecutionRecord):
        """Save execution record to database"""

        query = """
            INSERT OR REPLACE INTO pipeline_executions
            (execution_id, pipeline_id, pipeline_name, status, started_at, completed_at,
             total_steps, completed_steps, failed_steps, input_data, output_data,
             error_message, execution_time_ms, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        params = (
            execution.execution_id,
            execution.pipeline_id,
            execution.pipeline_name,
            execution.status.value,
            execution.started_at,
            execution.completed_at,
            execution.total_steps,
            execution.completed_steps,
            execution.failed_steps,
            json.dumps(execution.input_data) if execution.input_data else None,
            json.dumps(execution.output_data) if execution.output_data else None,
            execution.error_message,
            execution.execution_time_ms,
            json.dumps(execution.metadata) if execution.metadata else None
        )

        self.db.execute(query, params)

    async def _save_step_execution(self, step_execution: StepExecutionRecord):
        """Save step execution record to database"""

        query = """
            INSERT OR REPLACE INTO step_executions_enhanced
            (step_execution_id, execution_id, step_id, step_name, step_type, status,
             started_at, completed_at, input_data, output_data, error_message,
             execution_time_ms, retry_count, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        params = (
            step_execution.step_execution_id,
            step_execution.execution_id,
            step_execution.step_id,
            step_execution.step_name,
            step_execution.step_type,
            step_execution.status.value,
            step_execution.started_at,
            step_execution.completed_at,
            json.dumps(step_execution.input_data) if step_execution.input_data else None,
            json.dumps(step_execution.output_data) if step_execution.output_data else None,
            step_execution.error_message,
            step_execution.execution_time_ms,
            step_execution.retry_count,
            json.dumps(step_execution.metadata) if step_execution.metadata else None
        )

        self.db.execute(query, params)

    async def _get_step_execution(self, step_execution_id: str) -> Optional[StepExecutionRecord]:
        """Get step execution from database"""

        query = "SELECT * FROM step_executions_enhanced WHERE step_execution_id = ?"
        row = self.db.fetch_one(query, (step_execution_id,))

        if not row:
            return None

        return StepExecutionRecord(
            step_execution_id=row['step_execution_id'],
            execution_id=row['execution_id'],
            step_id=row['step_id'],
            step_name=row['step_name'],
            step_type=row['step_type'],
            status=StepStatus(row['status']),
            started_at=row['started_at'],
            completed_at=row['completed_at'],
            input_data=json.loads(row['input_data']) if row['input_data'] else None,
            output_data=json.loads(row['output_data']) if row['output_data'] else None,
            error_message=row['error_message'],
            execution_time_ms=row['execution_time_ms'],
            retry_count=row['retry_count'],
            metadata=json.loads(row['metadata_json']) if row['metadata_json'] else None
        )

    async def _update_execution_step_counts(self, execution_id: str):
        """Update step counts for an execution"""

        # Count completed and failed steps
        count_query = """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
            FROM step_executions_enhanced
            WHERE execution_id = ?
        """

        result = self.db.fetch_one(count_query, (execution_id,))

        if result and execution_id in self.active_executions:
            execution = self.active_executions[execution_id]
            execution.completed_steps = result['completed']
            execution.failed_steps = result['failed']

            # Update database
            await self._save_execution(execution)

    async def _broadcast_execution_update(self, execution: ExecutionRecord):
        """Broadcast execution update to WebSocket connections"""

        message = {
            "type": "execution_update",
            "data": execution.to_dict()
        }

        await self._broadcast_message(message)

    async def _broadcast_step_update(self, step_execution: StepExecutionRecord):
        """Broadcast step update to WebSocket connections"""

        message = {
            "type": "step_update",
            "data": step_execution.to_dict()
        }

        await self._broadcast_message(message)

    async def _broadcast_message(self, message: Dict[str, Any]):
        """Broadcast message to all WebSocket connections"""

        if not self.websocket_connections:
            return

        message_json = json.dumps(message)
        disconnected = []

        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(message_json)
            except Exception as e:
                logger.warning(f"Failed to send WebSocket message: {e}")
                disconnected.append(websocket)

        # Remove disconnected connections
        for websocket in disconnected:
            self.remove_websocket_connection(websocket)


# Global execution tracker instance
execution_tracker: Optional[ExecutionTracker] = None


def get_execution_tracker() -> ExecutionTracker:
    """Get the global execution tracker instance"""
    global execution_tracker
    if execution_tracker is None:
        raise RuntimeError("Execution tracker not initialized")
    return execution_tracker


async def initialize_execution_tracker(db_manager: DatabaseManager) -> ExecutionTracker:
    """Initialize the global execution tracker"""
    global execution_tracker
    execution_tracker = ExecutionTracker(db_manager)
    await execution_tracker.initialize()
    return execution_tracker