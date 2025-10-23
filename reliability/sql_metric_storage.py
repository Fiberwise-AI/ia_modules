"""
SQL Metric Storage

Persistent storage for reliability metrics using SQL databases.
Supports PostgreSQL, MySQL, SQLite, and DuckDB.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import json

from ia_modules.reliability.metrics import MetricStorage
from ia_modules.database import DatabaseManager


class SQLMetricStorage(MetricStorage):
    """
    SQL-based metric storage for production use.

    Supports multiple database backends through DatabaseManager.

    Example:
        >>> from ia_modules.database import DatabaseManager
        >>> db_manager = DatabaseManager(config)
        >>> await db_manager.connect()
        >>> storage = SQLMetricStorage(db_manager)
        >>> await storage.record_step({"agent": "planner", "success": True})
    """

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize SQL metric storage.

        Args:
            db_manager: DatabaseManager instance
        """
        self.db = db_manager
        self.logger = logging.getLogger("SQLMetricStorage")

        # Initialize - schema created by migrations (V007__reliability_metrics.sql)
        if not self.db.table_exists("reliability_steps"):
            raise RuntimeError(
                "reliability_steps table not found. Run database migrations first."
            )

        self._initialized = True
        self.logger.info(f"Initialized SQL metric storage ({self.db.config.database_type.value})")

    async def record_step(self, record: Dict[str, Any]):
        """
        Record a step metric.

        Args:
            record: Step metric data
        """
        if not self._initialized:
            await self.initialize()

        from datetime import timezone

        query = """
        INSERT INTO reliability_steps (
            agent_name, success, required_compensation, required_human,
            mode, declared_mode, mode_violation, timestamp
        ) VALUES (:agent_name, :success, :required_compensation, :required_human,
                  :mode, :declared_mode, :mode_violation, :timestamp)
        """

        timestamp = record.get("timestamp", datetime.now(timezone.utc))
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        params = {
            "agent_name": record["agent"],
            "success": record["success"],
            "required_compensation": record.get("required_compensation", False),
            "required_human": record.get("required_human", False),
            "mode": record.get("mode"),
            "declared_mode": record.get("declared_mode"),
            "mode_violation": record.get("mode_violation", False),
            "timestamp": timestamp
        }

        await self.db.execute_async(query, params)

    async def record_workflow(self, record: Dict[str, Any]):
        """
        Record a workflow metric.

        Args:
            record: Workflow metric data
        """
        if not self._initialized:
            await self.initialize()

        from datetime import timezone

        query = """
        INSERT INTO reliability_workflows (
            workflow_id, steps, retries, success, required_human, timestamp
        ) VALUES (:workflow_id, :steps, :retries, :success, :required_human, :timestamp)
        """

        timestamp = record.get("timestamp", datetime.now(timezone.utc))
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        params = {
            "workflow_id": record["workflow_id"],
            "steps": record["steps"],
            "retries": record.get("retries", 0),
            "success": record["success"],
            "required_human": record.get("required_human", False),
            "timestamp": timestamp
        }

        await self.db.execute_async(query, params)

    async def get_steps(
        self,
        agent: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get step records with filtering.

        Args:
            agent: Filter by agent name (optional)
            since: Filter by timestamp (optional)

        Returns:
            List of step records
        """
        if not self._initialized:
            await self.initialize()

        query = "SELECT * FROM reliability_steps WHERE 1=1"
        params = {}

        if agent:
            query += " AND agent_name = :agent"
            params["agent"] = agent

        if since:
            query += " AND timestamp >= :since"
            params["since"] = since

        query += " ORDER BY timestamp ASC"

        result = self.db.fetch_all(query, params if params else None)

        # Convert to expected format
        steps = []
        for row in result:
            step = {
                "agent": row["agent_name"],
                "success": bool(row["success"]),
                "required_compensation": bool(row.get("required_compensation", False)),
                "required_human": bool(row.get("required_human", False)),
                "mode": row.get("mode"),
                "declared_mode": row.get("declared_mode"),
                "mode_violation": bool(row.get("mode_violation", False)),
                "timestamp": row["timestamp"]
            }
            steps.append(step)

        return steps

    async def get_workflows(
        self,
        since: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get workflow records.

        Args:
            since: Filter by timestamp (optional)

        Returns:
            List of workflow records
        """
        if not self._initialized:
            await self.initialize()

        query = "SELECT * FROM reliability_workflows WHERE 1=1"
        params = {}

        if since:
            query += " AND timestamp >= :since"
            params["since"] = since

        query += " ORDER BY timestamp ASC"

        result = self.db.fetch_all(query, params if params else None)

        # Convert to expected format
        workflows = []
        for row in result:
            workflow = {
                "workflow_id": row["workflow_id"],
                "steps": row["steps"],
                "retries": row.get("retries", 0),
                "success": bool(row["success"]),
                "required_human": bool(row.get("required_human", False)),
                "timestamp": row["timestamp"]
            }
            workflows.append(workflow)

        return workflows

    async def record_slo_measurement(
        self,
        measurement_type: str,
        thread_id: str,
        checkpoint_id: Optional[str] = None,
        duration_ms: Optional[int] = None,
        replay_mode: Optional[str] = None,
        success: bool = True,
        error: Optional[str] = None
    ):
        """
        Record an SLO measurement (MTTE or RSR).

        Args:
            measurement_type: 'mtte' or 'rsr'
            thread_id: Thread ID
            checkpoint_id: Checkpoint ID (optional)
            duration_ms: Duration in milliseconds (for MTTE)
            replay_mode: Replay mode (for RSR)
            success: Whether measurement succeeded
            error: Error message if failed
        """
        if not self._initialized:
            await self.initialize()

        from datetime import timezone

        query = """
        INSERT INTO reliability_slo_measurements (
            measurement_type, thread_id, checkpoint_id,
            duration_ms, replay_mode, success, error, timestamp
        ) VALUES (:measurement_type, :thread_id, :checkpoint_id,
                  :duration_ms, :replay_mode, :success, :error, :timestamp)
        """

        params = {
            "measurement_type": measurement_type,
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
            "duration_ms": duration_ms,
            "replay_mode": replay_mode,
            "success": success,
            "error": error,
            "timestamp": datetime.now(timezone.utc)
        }

        await self.db.execute_async(query, params)

    async def get_slo_measurements(
        self,
        measurement_type: str,
        since: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get SLO measurements.

        Args:
            measurement_type: 'mtte' or 'rsr'
            since: Filter by timestamp (optional)

        Returns:
            List of measurements
        """
        if not self._initialized:
            await self.initialize()

        query = "SELECT * FROM reliability_slo_measurements WHERE measurement_type = :measurement_type"
        params = {"measurement_type": measurement_type}

        if since:
            query += " AND timestamp >= :since"
            params["since"] = since

        query += " ORDER BY timestamp ASC"

        result = self.db.fetch_all(query, params)

        # Convert success field to boolean (SQLite stores as 0/1)
        measurements = []
        for row in result:
            measurement = dict(row)
            measurement["success"] = bool(measurement["success"])
            measurements.append(measurement)

        return measurements

    async def close(self):
        """Close database connection."""
        if self.db:
            self.db.disconnect()
            self._initialized = False
