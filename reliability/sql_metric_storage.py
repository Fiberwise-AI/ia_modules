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
from ia_modules.database.interfaces import DatabaseInterface, ConnectionConfig


class SQLMetricStorage(MetricStorage):
    """
    SQL-based metric storage for production use.

    Supports multiple database backends through DatabaseConnection abstraction.

    Example:
        >>> from ia_modules.database.interfaces import ConnectionConfig, DatabaseType
        >>> config = ConnectionConfig(
        ...     database_type=DatabaseType.POSTGRESQL,
        ...     database_url="postgresql://localhost/metrics"
        ... )
        >>> storage = SQLMetricStorage(config)
        >>> await storage.record_step({"agent": "planner", "success": True})
    """

    def __init__(
        self,
        config: ConnectionConfig,
        connection: Optional[DatabaseInterface] = None
    ):
        """
        Initialize SQL metric storage.

        Args:
            config: Database configuration
            connection: Optional existing database connection
        """
        self.config = config
        self.connection = connection
        self.logger = logging.getLogger("SQLMetricStorage")
        self._initialized = False

    async def initialize(self):
        """Initialize database connection and ensure tables exist."""
        if self._initialized:
            return

        if self.connection is None:
            # Create connection if not provided
            from ia_modules.database.sqlite_simple import SimpleSQLite

            # Parse database URL to extract path for SQLite
            db_path = self.config.database_url
            if db_path.startswith("sqlite:"):
                # Remove sqlite: prefix
                db_path = db_path.replace("sqlite:", "")
                # Handle sqlite::memory: -> :memory:
                if db_path.startswith(":"):
                    db_path = db_path  # Already correct format
                elif db_path.startswith("///"):
                    db_path = db_path[3:]  # Remove ///
                elif db_path.startswith("//"):
                    db_path = db_path[2:]  # Remove //

            self.connection = SimpleSQLite(db_path)
            await self.connection.connect()

        # Ensure tables exist (run migration if needed)
        await self._ensure_tables()

        self._initialized = True
        self.logger.info(f"Initialized SQL metric storage ({self.config.database_type.value})")

    async def _ensure_tables(self):
        """Ensure reliability metrics tables exist."""
        # Check if tables exist
        check_query = """
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='reliability_steps'
        """

        if self.config.database_type.value == "postgresql":
            check_query = """
            SELECT tablename FROM pg_tables
            WHERE tablename='reliability_steps'
            """

        result = await self.connection.execute_query(check_query)

        if not result.data:
            # Tables don't exist - create them
            self.logger.info("Creating reliability metrics tables...")

            # Create tables (SQLite syntax)
            create_steps = """
            CREATE TABLE IF NOT EXISTS reliability_steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                success BOOLEAN NOT NULL,
                required_compensation BOOLEAN DEFAULT FALSE,
                required_human BOOLEAN DEFAULT FALSE,
                mode TEXT,
                declared_mode TEXT,
                mode_violation BOOLEAN DEFAULT FALSE,
                timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """

            create_workflows = """
            CREATE TABLE IF NOT EXISTS reliability_workflows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workflow_id TEXT NOT NULL UNIQUE,
                steps INTEGER NOT NULL,
                retries INTEGER DEFAULT 0,
                success BOOLEAN NOT NULL,
                required_human BOOLEAN DEFAULT FALSE,
                timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """

            create_slo = """
            CREATE TABLE IF NOT EXISTS reliability_slo_measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                measurement_type TEXT NOT NULL CHECK (measurement_type IN ('mtte', 'rsr')),
                thread_id TEXT NOT NULL,
                checkpoint_id TEXT,
                duration_ms INTEGER,
                replay_mode TEXT,
                success BOOLEAN NOT NULL,
                error TEXT,
                timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """

            # Execute table creation
            await self.connection.execute_query(create_steps)
            await self.connection.execute_query(create_workflows)
            await self.connection.execute_query(create_slo)

            self.logger.info("Reliability metrics tables created successfully")

    async def record_step(self, record: Dict[str, Any]):
        """
        Record a step metric.

        Args:
            record: Step metric data
        """
        if not self._initialized:
            await self.initialize()

        query = """
        INSERT INTO reliability_steps (
            agent_name, success, required_compensation, required_human,
            mode, declared_mode, mode_violation, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """

        timestamp = record.get("timestamp", datetime.utcnow())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        params = (
            record["agent"],
            record["success"],
            record.get("required_compensation", False),
            record.get("required_human", False),
            record.get("mode"),
            record.get("declared_mode"),
            record.get("mode_violation", False),
            timestamp
        )

        await self.connection.execute_query(query, params)

    async def record_workflow(self, record: Dict[str, Any]):
        """
        Record a workflow metric.

        Args:
            record: Workflow metric data
        """
        if not self._initialized:
            await self.initialize()

        query = """
        INSERT INTO reliability_workflows (
            workflow_id, steps, retries, success, required_human, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?)
        """

        timestamp = record.get("timestamp", datetime.utcnow())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        params = (
            record["workflow_id"],
            record["steps"],
            record.get("retries", 0),
            record["success"],
            record.get("required_human", False),
            timestamp
        )

        await self.connection.execute_query(query, params)

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
        params = []

        if agent:
            query += " AND agent_name = ?"
            params.append(agent)

        if since:
            query += " AND timestamp >= ?"
            params.append(since)

        query += " ORDER BY timestamp ASC"

        result = await self.connection.execute_query(query, tuple(params))

        # Convert to expected format
        steps = []
        for row in result.data:
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
        params = []

        if since:
            query += " AND timestamp >= ?"
            params.append(since)

        query += " ORDER BY timestamp ASC"

        result = await self.connection.execute_query(query, tuple(params) if params else ())

        # Convert to expected format
        workflows = []
        for row in result.data:
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

        query = """
        INSERT INTO reliability_slo_measurements (
            measurement_type, thread_id, checkpoint_id,
            duration_ms, replay_mode, success, error, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """

        params = (
            measurement_type,
            thread_id,
            checkpoint_id,
            duration_ms,
            replay_mode,
            success,
            error,
            datetime.utcnow()
        )

        await self.connection.execute_query(query, params)

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

        query = "SELECT * FROM reliability_slo_measurements WHERE measurement_type = ?"
        params = [measurement_type]

        if since:
            query += " AND timestamp >= ?"
            params.append(since)

        query += " ORDER BY timestamp ASC"

        result = await self.connection.execute_query(query, tuple(params))

        # Convert success field to boolean (SQLite stores as 0/1)
        measurements = []
        for row in result.data:
            measurement = dict(row)
            measurement["success"] = bool(measurement["success"])
            measurements.append(measurement)

        return measurements

    async def close(self):
        """Close database connection."""
        if self.connection:
            await self.connection.disconnect()
            self._initialized = False
