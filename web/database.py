"""
Database Service for Pipeline Web Interface

Handles storage and retrieval of pipeline definitions, executions, and results
using the ia_modules database system.
"""



import json# This file has been refactored to use the PipelineService from the app layer

import uuid# The PipelineDatabase class has been removed in favor of proper service layer architecture

import logging# All pipeline operations now go through the PipelineService in app_knowledge_base/api/services/pipeline_service.py

from datetime import datetime

from typing import Dict, Any, List, Optional# Legacy PipelineDatabase class removed - use PipelineService instead

from ia_modules.database.manager import DatabaseManager
from .models import PipelineModel, PipelineExecution, ExecutionStatus

logger = logging.getLogger(__name__)


class PipelineDatabase:
    """Database service for pipeline operations"""

    def __init__(self, database_path: str = "pipeline_visual_editor.db"):
        """Initialize database connection"""
        self.db = DatabaseManager(f"sqlite:///{database_path}")
        self.db.connect()
        self._create_tables()

    def _create_tables(self):
        """Create necessary database tables"""

        # Pipelines table
        self.db.create_table("pipelines", """
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            version TEXT DEFAULT '2.0',
            pipeline_data TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        """)

        # Pipeline executions table
        self.db.create_table("pipeline_executions", """
            id TEXT PRIMARY KEY,
            execution_id TEXT UNIQUE NOT NULL,
            pipeline_name TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT DEFAULT CURRENT_TIMESTAMP,
            completed_at TEXT,
            duration_seconds REAL,
            input_data TEXT,
            output_data TEXT,
            current_step TEXT,
            error_message TEXT
        """)

        # Execution logs table
        self.db.create_table("execution_logs", """
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            execution_id TEXT NOT NULL,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            level TEXT DEFAULT 'INFO',
            message TEXT NOT NULL,
            FOREIGN KEY (execution_id) REFERENCES pipeline_executions(execution_id)
        """)

        # Pipeline templates table
        self.db.create_table("pipeline_templates", """
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            category TEXT DEFAULT 'user',
            pipeline_data TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        """)

        logger.info("Pipeline database tables created")

    def save_pipeline(self, pipeline: PipelineModel) -> str:
        """Save a pipeline definition"""
        pipeline_id = str(uuid.uuid4())
        pipeline_data = pipeline.json()

        query = """
        INSERT INTO pipelines (id, name, description, version, pipeline_data, updated_at)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """

        self.db.execute(query, (
            pipeline_id,
            pipeline.name,
            pipeline.description,
            pipeline.version,
            pipeline_data
        ))

        logger.info(f"Pipeline saved: {pipeline.name} ({pipeline_id})")
        return pipeline_id

    def list_pipelines(self) -> List[Dict[str, str]]:
        """List all saved pipelines"""
        query = "SELECT id, name, description FROM pipelines ORDER BY updated_at DESC"
        rows = self.db.fetch_all(query)

        return [
            {
                "id": row["id"],
                "name": row["name"],
                "description": row["description"] or ""
            }
            for row in rows
        ]

    def get_pipeline(self, pipeline_id: str) -> Optional[PipelineModel]:
        """Get a specific pipeline"""
        query = "SELECT pipeline_data FROM pipelines WHERE id = ?"
        row = self.db.fetch_one(query, (pipeline_id,))

        if not row:
            return None

        try:
            pipeline_data = json.loads(row["pipeline_data"])
            return PipelineModel(**pipeline_data)
        except Exception as e:
            logger.error(f"Failed to parse pipeline data for {pipeline_id}: {e}")
            return None

    def delete_pipeline(self, pipeline_id: str) -> bool:
        """Delete a pipeline"""
        query = "DELETE FROM pipelines WHERE id = ?"
        cursor = self.db.execute(query, (pipeline_id,))
        success = cursor.rowcount > 0

        if success:
            logger.info(f"Pipeline deleted: {pipeline_id}")
        else:
            logger.warning(f"Pipeline not found for deletion: {pipeline_id}")

        return success

    def create_execution(self, execution: PipelineExecution):
        """Create a new execution record"""
        input_data = json.dumps(execution.input_data) if execution.input_data else None

        query = """
        INSERT INTO pipeline_executions
        (id, execution_id, pipeline_name, status, started_at, input_data, current_step)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """

        execution_id = str(uuid.uuid4())

        self.db.execute(query, (
            execution_id,
            execution.execution_id,
            execution.pipeline_name,
            execution.status.value,
            execution.started_at.isoformat(),
            input_data,
            execution.current_step
        ))

        logger.info(f"Execution created: {execution.execution_id}")

    def update_execution_status(
        self,
        execution_id: str,
        status: ExecutionStatus,
        current_step: Optional[str] = None,
        completed_at: Optional[datetime] = None,
        output_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ):
        """Update execution status"""
        output_data_str = json.dumps(output_data) if output_data else None
        completed_at_str = completed_at.isoformat() if completed_at else None

        query = """
        UPDATE pipeline_executions
        SET status = ?, current_step = ?, completed_at = ?, output_data = ?, error_message = ?
        WHERE execution_id = ?
        """

        self.db.execute(query, (
            status.value,
            current_step,
            completed_at_str,
            output_data_str,
            error_message,
            execution_id
        ))

        logger.info(f"Execution {execution_id} status updated to {status.value}")

    def list_executions(self, limit: int = 50) -> List[PipelineExecution]:
        """List pipeline executions"""
        query = """
        SELECT execution_id, pipeline_name, status, started_at, completed_at,
               duration_seconds, input_data, output_data, current_step, error_message
        FROM pipeline_executions
        ORDER BY started_at DESC
        LIMIT ?
        """

        rows = self.db.fetch_all(query, (limit,))
        executions = []

        for row in rows:
            try:
                input_data = json.loads(row["input_data"]) if row["input_data"] else None
                output_data = json.loads(row["output_data"]) if row["output_data"] else None

                execution = PipelineExecution(
                    execution_id=row["execution_id"],
                    pipeline_name=row["pipeline_name"],
                    status=ExecutionStatus(row["status"]),
                    started_at=datetime.fromisoformat(row["started_at"]),
                    completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
                    duration_seconds=row["duration_seconds"],
                    input_data=input_data or {},
                    output_data=output_data,
                    current_step=row["current_step"],
                    error=row["error_message"]
                )
                executions.append(execution)
            except Exception as e:
                logger.error(f"Failed to parse execution {row['execution_id']}: {e}")
                continue

        return executions

    def get_execution(self, execution_id: str) -> Optional[PipelineExecution]:
        """Get specific execution details"""
        query = """
        SELECT execution_id, pipeline_name, status, started_at, completed_at,
               duration_seconds, input_data, output_data, current_step, error_message
        FROM pipeline_executions
        WHERE execution_id = ?
        """

        row = self.db.fetch_one(query, (execution_id,))

        if not row:
            return None

        try:
            input_data = json.loads(row["input_data"]) if row["input_data"] else None
            output_data = json.loads(row["output_data"]) if row["output_data"] else None

            return PipelineExecution(
                execution_id=row["execution_id"],
                pipeline_name=row["pipeline_name"],
                status=ExecutionStatus(row["status"]),
                started_at=datetime.fromisoformat(row["started_at"]),
                completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
                duration_seconds=row["duration_seconds"],
                input_data=input_data or {},
                output_data=output_data,
                current_step=row["current_step"],
                error=row["error_message"]
            )
        except Exception as e:
            logger.error(f"Failed to parse execution {execution_id}: {e}")
            return None

    def add_execution_log(self, execution_id: str, message: str, log_level: str = "INFO"):
        """Add a log entry for an execution"""
        query = """
        INSERT INTO execution_logs (execution_id, level, message)
        VALUES (?, ?, ?)
        """

        self.db.execute(query, (execution_id, log_level, message))

    def get_execution_logs(self, execution_id: str) -> List[Dict[str, Any]]:
        """Get execution logs"""
        query = """
        SELECT timestamp, level, message
        FROM execution_logs
        WHERE execution_id = ?
        ORDER BY timestamp ASC
        """

        rows = self.db.fetch_all(query, (execution_id,))

        return [
            {
                "timestamp": row["timestamp"],
                "level": row["level"],
                "message": row["message"]
            }
            for row in rows
        ]

    def save_template(self, pipeline: PipelineModel, category: str = "user") -> str:
        """Save a pipeline as a template"""
        template_id = str(uuid.uuid4())
        pipeline_data = pipeline.json()

        query = """
        INSERT INTO pipeline_templates (id, name, description, category, pipeline_data)
        VALUES (?, ?, ?, ?, ?)
        """

        self.db.execute(query, (
            template_id,
            pipeline.name,
            pipeline.description,
            category,
            pipeline_data
        ))

        logger.info(f"Template saved: {pipeline.name} ({template_id})")
        return template_id

    def get_templates(self, category: Optional[str] = None) -> List[Dict[str, str]]:
        """List pipeline templates"""
        if category:
            query = "SELECT id, name, description, category FROM pipeline_templates WHERE category = ? ORDER BY name"
            rows = self.db.fetch_all(query, (category,))
        else:
            query = "SELECT id, name, description, category FROM pipeline_templates ORDER BY category, name"
            rows = self.db.fetch_all(query)

        return [
            {
                "id": row["id"],
                "name": row["name"],
                "description": row["description"] or "",
                "category": row["category"]
            }
            for row in rows
        ]

    def get_template(self, template_id: str) -> Optional[PipelineModel]:
        """Get a specific template"""
        query = "SELECT pipeline_data FROM pipeline_templates WHERE id = ?"
        row = self.db.fetch_one(query, (template_id,))

        if not row:
            return None

        try:
            pipeline_data = json.loads(row["pipeline_data"])
            return PipelineModel(**pipeline_data)
        except Exception as e:
            logger.error(f"Failed to parse template data for {template_id}: {e}")
            return None

    def close(self):
        """Close database connection"""
        if self.db:
            self.db.disconnect()