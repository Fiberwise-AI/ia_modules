"""
DB-backed pipeline runner helper

Provides a small convenience function to run a pipeline that was
imported into the application's `pipelines` table. The function loads
the pipeline config by slug, creates a minimal ServiceRegistry (with
CentralLoggingService), runs the pipeline, and writes a lightweight
job record into the database. It returns a job document describing the run.

This helper assumes the `db_provider` implements the same DatabaseInterface
used by the importer (methods like `execute_query`, `fetch_one`, etc.).
"""

import json
import uuid
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from .importer import PipelineImportService
from .runner import create_pipeline_from_json
from .services import ServiceRegistry, CentralLoggingService

logger = logging.getLogger(__name__)


async def run_pipeline_by_slug(
    db_provider,
    slug: str,
    input_data: Optional[Dict[str, Any]] = None,
    services: Optional[ServiceRegistry] = None,
    create_job_record: bool = True
) -> Dict[str, Any]:
    """Run a pipeline imported into the DB by its slug.

    Returns a job document with run metadata and final result.
    """

    importer = PipelineImportService(db_provider)
    pipeline_row = await importer.get_pipeline_by_slug(slug)
    if not pipeline_row:
        raise ValueError(f"Pipeline not found for slug: {slug}")

    pipeline_config = pipeline_row.get('pipeline_config')
    if not pipeline_config:
        raise ValueError("Pipeline configuration missing in DB row")

    if input_data is None:
        input_data = {}

    # Prepare services and logger
    own_services = False
    if services is None:
        services = ServiceRegistry()
        own_services = True

    central_logger: CentralLoggingService = services.get('central_logger')
    if not central_logger:
        central_logger = CentralLoggingService()
        services.register('central_logger', central_logger)

    # Create a job/execution id and optionally persist a job record
    job_id = str(uuid.uuid4())
    started_at = datetime.utcnow().isoformat() + 'Z'

    if create_job_record:
        try:
            insert_query = """
            INSERT INTO pipeline_executions (execution_id, pipeline_id, pipeline_name, status, started_at, input_data)
            VALUES (?, ?, ?, ?, ?, ?)
            """
            pipeline_name = pipeline_row.get('name', 'Unknown Pipeline')
            await db_provider.execute_query(insert_query, (job_id, slug, pipeline_name, 'running', started_at, json.dumps(input_data)))
        except Exception as e:
            logger.warning(f"Failed to create pipeline_executions row: {e}")

    # Set execution id on logger
    central_logger.set_execution_id(job_id)

    # Register a DB adapter into services so steps can call db_service.log_step_execution
    class _DBAdapter:
        def __init__(self, dbp, job_id):
            self._db = dbp
            self._job_id = job_id
            self._step_statuses = {}

        def log_step_execution(self, execution_id, step_name, success, duration, error=None, result=None):
            """Called by Step._log_step_execution_to_database (synchronous). Schedule an async update."""
            try:
                # update in background
                import asyncio

                async def _update():
                    try:
                        # Update pipeline_executions status
                        self._step_statuses[step_name] = 'succeeded' if success else 'failed'
                        completed_steps = len([s for s in self._step_statuses.values() if s == 'succeeded'])
                        failed_steps = len([s for s in self._step_statuses.values() if s == 'failed'])

                        update_q = """
                        UPDATE pipeline_executions
                        SET completed_steps = ?, failed_steps = ?
                        WHERE execution_id = ?
                        """
                        await self._db.execute_query(update_q, (completed_steps, failed_steps, self._job_id))

                        # Insert into pipeline_logs if table exists
                        try:
                            insert_log = """
                            INSERT INTO pipeline_logs (job_id, step_name, event_type, timestamp, data, duration)
                            VALUES (?, ?, ?, ?, ?, ?)
                            """
                            event_type = 'complete' if success else 'error'
                            data = json.dumps(result) if result is not None else (json.dumps({'error': error}) if error else None)
                            await self._db.execute_query(insert_log, (self._job_id, step_name, event_type, datetime.utcnow().isoformat(), data, duration))
                        except Exception:
                            pass

                    except Exception:
                        pass

                asyncio.create_task(_update())
            except Exception:
                pass

        async def log_execution_event(self, execution_id, event_type, message, step_name, data, timestamp):
            """Used by CentralLoggingService.write_to_database (async)."""
            try:
                insert_log = """
                INSERT INTO pipeline_logs (job_id, step_name, event_type, timestamp, data)
                VALUES (?, ?, ?, ?, ?)
                """
                await self._db.execute_query(insert_log, (execution_id, step_name, event_type, timestamp, data))
            except Exception:
                pass

    # Register DB adapter into services so Step._log_step_execution_to_database finds db_service
    try:
        db_adapter = _DBAdapter(db_provider, job_id)
        services.register('database', db_adapter)
    except Exception:
        pass

    # Build and run pipeline
    pipeline = create_pipeline_from_json(pipeline_config, services=services)

    job_doc: Dict[str, Any] = {
        'job_id': job_id,
        'pipeline_slug': slug,
        'status': 'running',
        'started_at': started_at,
        'finished_at': None,
        'current_step': None,
        'step_statuses': {},
        'result': None
    }

    try:
        result = await pipeline.run(input_data)

        job_doc['status'] = 'succeeded'
        job_doc['finished_at'] = datetime.utcnow().isoformat() + 'Z'
        job_doc['result'] = result

        # Persist final job status
        if create_job_record:
            try:
                update_query = """
                UPDATE pipeline_executions
                SET status = ?, completed_at = ?, output_data = ?
                WHERE execution_id = ?
                """
                await db_provider.execute_query(update_query, (job_doc['status'], job_doc['finished_at'], json.dumps(result), job_id))
            except Exception as e:
                logger.warning(f"Failed to update pipeline_executions row: {e}")

    except Exception as e:
        logger.exception(f"Pipeline run failed for slug={slug}: {e}")
        job_doc['status'] = 'failed'
        job_doc['finished_at'] = datetime.utcnow().isoformat() + 'Z'
        job_doc['result'] = {'error': str(e)}

        if create_job_record:
            try:
                update_query = """
                UPDATE pipeline_executions
                SET status = ?, completed_at = ?, output_data = ?, error_message = ?
                WHERE execution_id = ?
                """
                await db_provider.execute_query(update_query, (job_doc['status'], job_doc['finished_at'], json.dumps(job_doc['result']), str(e), job_id))
            except Exception as e2:
                logger.warning(f"Failed to update pipeline_executions failure row: {e2}")

    finally:
        # Attempt to write logs to DB via central logger
        try:
            await central_logger.write_to_database(db_provider)
        except Exception:
            pass

        # Cleanup service registry if we created it
        if own_services:
            try:
                await services.cleanup_all()
            except Exception:
                pass

    return job_doc
