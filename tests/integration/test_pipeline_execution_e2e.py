"""
End-to-end pipeline execution tests with real database.

Tests the full execution flow from execute_pipeline through GraphPipelineRunner
with actual database operations including step tracking.

NOTE: These tests are currently disabled due to showcase_app dependencies.
"""

import pytest
from pathlib import Path
import uuid
import json
import os

try:
    from core.database import DatabaseManager
    from core.service_registry import ServiceRegistry
    from pipeline.execution_tracker import ExecutionTracker
    from pipeline.service import PipelineService
    from pipeline.import_export import PipelineImportService
except ImportError:
    DatabaseManager = None
    ServiceRegistry = None
    ExecutionTracker = None
    PipelineService = None
    PipelineImportService = None

# Skip all tests in this module until showcase_app dependencies are resolved
pytestmark = pytest.mark.skip(reason="Tests need showcase_app which has dependency issues")


@pytest.fixture
def pipelines_dir():
    """Get path to test pipelines"""
    return Path(__file__).parent.parent / "pipelines"


@pytest.fixture
async def db_manager(request):
    """Database manager for different backends"""
    # Use SQLite for basic tests
    db_url = 'sqlite:///test_e2e_execution.db'

    db = DatabaseManager(db_url)
    await db.initialize()

    # Run migrations
    from nexusql import MigrationRunner
    runner = MigrationRunner(db)
    await runner.run_pending_migrations()

    yield db

    await db.close()

    # Cleanup SQLite file
    db_path = Path('test_e2e_execution.db')
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
async def pipeline_service(db_manager, pipelines_dir):
    """Create PipelineService with real database"""
    services = ServiceRegistry()
    services.register('database', db_manager)

    tracker = ExecutionTracker(db_manager)
    services.register('execution_tracker', tracker)

    # Create metrics service (mock)
    class MockMetricsService:
        def track_metric(self, *args, **kwargs):
            pass

    metrics_service = MockMetricsService()

    service = PipelineService(
        metrics_service=metrics_service,
        db_manager=db_manager
    )
    service.pipeline_dir = pipelines_dir

    # Import test pipelines
    importer = PipelineImportService(db_manager, pipelines_dir)
    await importer.import_all_pipelines()

    return service


@pytest.mark.asyncio
class TestPipelineExecutionE2E:
    """End-to-end pipeline execution tests"""

    async def test_simple_pipeline_execution_tracks_steps(self, pipeline_service, db_manager):
        """Execute simple pipeline and verify all steps tracked in database"""
        # Create a test pipeline directly
        test_pipeline = {
            "name": "Test E2E Pipeline",
            "version": "1.0.0",
            "description": "Test pipeline for e2e execution",
            "config": {
                "name": "Test E2E Pipeline",
                "version": "1.0.0",
                "steps": [
                    {
                        "id": "step1",
                        "name": "Step 1",
                        "step_class": "DataTransformStep",
                        "module": "tests.pipelines.simple_pipeline.steps.simple_steps",
                        "config": {}
                    }
                ],
                "flow": {
                    "start_at": "step1",
                    "paths": []
                }
            }
        }

        # Insert pipeline into database
        from ia_modules.pipeline.importer import PipelineImportService
        PipelineImportService(db_manager, Path(__file__).parent.parent / "pipelines")

        # Insert directly
        pipeline_id = str(uuid.uuid4())
        query = """
        INSERT INTO pipelines (id, name, slug, version, description, config_json, is_active)
        VALUES (:id, :name, :slug, :version, :description, :config_json, :is_active)
        """
        db_manager.execute(query, {
            'id': pipeline_id,
            'name': test_pipeline['name'],
            'slug': 'test_e2e_pipeline',
            'version': test_pipeline['version'],
            'description': test_pipeline['description'],
            'config_json': json.dumps(test_pipeline['config']),
            'is_active': True
        })

        # Verify insert
        verify_query = "SELECT * FROM pipelines WHERE id = :id"
        pipeline_record = await db_manager.fetch_one(verify_query, {'id': pipeline_id})
        assert pipeline_record is not None, f"Pipeline {pipeline_id} not inserted"

        # Execute pipeline
        job_id = await pipeline_service.execute_pipeline(
            pipeline_id=pipeline_id,
            input_data={"value": 10}
        )

        assert job_id is not None

        # Wait for execution to complete
        import asyncio
        max_wait = 30
        waited = 0
        while waited < max_wait:
            status = await pipeline_service.get_execution_status(job_id)
            if status['status'] in ('completed', 'failed'):
                break
            await asyncio.sleep(0.5)
            waited += 0.5

        # Verify execution in database
        query = "SELECT * FROM pipeline_executions WHERE id = :execution_id"
        execution = await db_manager.fetch_one(query, {'execution_id': job_id})

        assert execution is not None
        assert execution['status'] in ('completed', 'failed')

        # Verify step executions were tracked
        step_query = "SELECT * FROM step_executions WHERE execution_id = :execution_id ORDER BY started_at"
        steps = await db_manager.fetch_all(step_query, {'execution_id': job_id})

        assert len(steps) > 0, "No step executions tracked"

        # Verify each step has required data
        for step in steps:
            assert step['step_id'] is not None
            assert step['step_name'] is not None
            assert step['status'] is not None
            assert step['started_at'] is not None

    async def test_pipeline_execution_with_error_tracking(self, pipeline_service, db_manager):
        """Execute pipeline that fails and verify error tracked"""
        pipelines = await pipeline_service.list_pipelines()

        # Use any pipeline and pass invalid input to cause failure
        if not pipelines:
            pytest.skip("No pipelines available")

        pipeline_id = pipelines[0]['id']

        # Execute with bad input
        job_id = await pipeline_service.execute_pipeline(
            pipeline_id=pipeline_id,
            input_data={}  # Empty input may cause some pipelines to fail
        )

        # Wait for execution
        import asyncio
        max_wait = 30
        waited = 0
        while waited < max_wait:
            status = await pipeline_service.get_execution_status(job_id)
            if status['status'] in ('completed', 'failed'):
                break
            await asyncio.sleep(0.5)
            waited += 0.5

        # Check execution exists in database
        query = "SELECT * FROM pipeline_executions WHERE id = :execution_id"
        execution = await db_manager.fetch_one(query, {'execution_id': job_id})

        assert execution is not None

        # If it failed, verify error message exists
        if execution['status'] == 'failed':
            assert execution['error_message'] is not None or execution['error_message'] != ''

    async def test_parallel_executions_tracked_separately(self, pipeline_service, db_manager):
        """Execute same pipeline twice in parallel and verify separate tracking"""
        pipelines = await pipeline_service.list_pipelines()

        if not pipelines:
            pytest.skip("No pipelines available")

        pipeline_id = pipelines[0]['id']

        # Start two executions in parallel
        job_id_1 = await pipeline_service.execute_pipeline(
            pipeline_id=pipeline_id,
            input_data={"value": 1}
        )

        job_id_2 = await pipeline_service.execute_pipeline(
            pipeline_id=pipeline_id,
            input_data={"value": 2}
        )

        assert job_id_1 != job_id_2

        # Wait for both to complete
        import asyncio
        max_wait = 30

        async def wait_for_execution(job_id):
            waited = 0
            while waited < max_wait:
                status = await pipeline_service.get_execution_status(job_id)
                if status['status'] in ('completed', 'failed'):
                    return
                await asyncio.sleep(0.5)
                waited += 0.5

        await asyncio.gather(
            wait_for_execution(job_id_1),
            wait_for_execution(job_id_2)
        )

        # Verify both executions in database
        query = "SELECT * FROM pipeline_executions WHERE id = :execution_id"

        execution_1 = await db_manager.fetch_one(query, {'execution_id': job_id_1})
        execution_2 = await db_manager.fetch_one(query, {'execution_id': job_id_2})

        assert execution_1 is not None
        assert execution_2 is not None
        assert execution_1['id'] != execution_2['id']

        # Verify step executions are separate
        step_query = "SELECT COUNT(*) as count FROM step_executions WHERE execution_id = :execution_id"

        steps_1 = await db_manager.fetch_one(step_query, {'execution_id': job_id_1})
        steps_2 = await db_manager.fetch_one(step_query, {'execution_id': job_id_2})

        # Both should have tracked steps
        assert steps_1['count'] > 0
        assert steps_2['count'] > 0


@pytest.mark.postgres
@pytest.mark.asyncio
class TestPipelineExecutionPostgreSQL:
    """PostgreSQL-specific execution tests"""

    async def test_postgresql_execution_with_concurrent_writes(self, pipelines_dir):
        """Test PostgreSQL handles concurrent pipeline executions"""
        db_url = os.environ.get('TEST_POSTGRESQL_URL')
        if not db_url:
            pytest.skip("PostgreSQL not configured")

        db = DatabaseManager(db_url)
        await db.initialize()

        try:
            services = ServiceRegistry()
            services.register('database', db)

            tracker = ExecutionTracker(db)
            services.register('execution_tracker', tracker)

            service = PipelineService(
                db=db,
                pipeline_dir=pipelines_dir,
                services=services
            )

            # Import pipelines
            importer = PipelineImportService(db, pipelines_dir)
            await importer.import_all_pipelines()

            pipelines = await service.list_pipelines()
            if not pipelines:
                pytest.skip("No pipelines available")

            pipeline_id = pipelines[0]['id']

            # Start 5 concurrent executions
            import asyncio
            jobs = []
            for i in range(5):
                job_id = await service.execute_pipeline(
                    pipeline_id=pipeline_id,
                    input_data={"value": i}
                )
                jobs.append(job_id)

            # Wait for all to complete
            async def wait_for_execution(job_id):
                max_wait = 30
                waited = 0
                while waited < max_wait:
                    status = await service.get_execution_status(job_id)
                    if status['status'] in ('completed', 'failed'):
                        return
                    await asyncio.sleep(0.5)
                    waited += 0.5

            await asyncio.gather(*[wait_for_execution(job_id) for job_id in jobs])

            # Verify all executions in database
            query = "SELECT COUNT(*) as count FROM pipeline_executions WHERE id = ANY(:execution_ids)"
            result = await db.fetch_one(query, {'execution_ids': jobs})

            assert result['count'] == 5

        finally:
            await db.close()


@pytest.mark.mysql
@pytest.mark.asyncio
class TestPipelineExecutionMySQL:
    """MySQL-specific execution tests"""

    async def test_mysql_execution_tracking(self, pipelines_dir):
        """Test MySQL execution tracking"""
        db_url = os.environ.get('TEST_MYSQL_URL')
        if not db_url:
            pytest.skip("MySQL not configured")

        db = DatabaseManager(db_url)
        await db.initialize()

        try:
            services = ServiceRegistry()
            services.register('database', db)

            tracker = ExecutionTracker(db)
            services.register('execution_tracker', tracker)

            service = PipelineService(
                db=db,
                pipeline_dir=pipelines_dir,
                services=services
            )

            # Import and execute
            importer = PipelineImportService(db, pipelines_dir)
            await importer.import_all_pipelines()

            pipelines = await service.list_pipelines()
            if not pipelines:
                pytest.skip("No pipelines available")

            pipeline_id = pipelines[0]['id']

            job_id = await service.execute_pipeline(
                pipeline_id=pipeline_id,
                input_data={"value": 42}
            )

            # Wait for completion
            import asyncio
            max_wait = 30
            waited = 0
            while waited < max_wait:
                status = await service.get_execution_status(job_id)
                if status['status'] in ('completed', 'failed'):
                    break
                await asyncio.sleep(0.5)
                waited += 0.5

            # Verify in database
            query = "SELECT * FROM pipeline_executions WHERE id = :execution_id"
            execution = await db.fetch_one(query, {'execution_id': job_id})

            assert execution is not None

        finally:
            await db.close()


@pytest.mark.mssql
@pytest.mark.asyncio
class TestPipelineExecutionMSSQL:
    """MSSQL-specific execution tests"""

    async def test_mssql_execution_tracking(self, pipelines_dir):
        """Test MSSQL execution tracking"""
        db_url = os.environ.get('TEST_MSSQL_URL')
        if not db_url:
            pytest.skip("MSSQL not configured")

        db = DatabaseManager(db_url)
        await db.initialize()

        try:
            services = ServiceRegistry()
            services.register('database', db)

            tracker = ExecutionTracker(db)
            services.register('execution_tracker', tracker)

            service = PipelineService(
                db=db,
                pipeline_dir=pipelines_dir,
                services=services
            )

            # Import and execute
            importer = PipelineImportService(db, pipelines_dir)
            await importer.import_all_pipelines()

            pipelines = await service.list_pipelines()
            if not pipelines:
                pytest.skip("No pipelines available")

            pipeline_id = pipelines[0]['id']

            job_id = await service.execute_pipeline(
                pipeline_id=pipeline_id,
                input_data={"value": 99}
            )

            # Wait for completion
            import asyncio
            max_wait = 30
            waited = 0
            while waited < max_wait:
                status = await service.get_execution_status(job_id)
                if status['status'] in ('completed', 'failed'):
                    break
                await asyncio.sleep(0.5)
                waited += 0.5

            # Verify in database
            query = "SELECT * FROM pipeline_executions WHERE id = :execution_id"
            execution = await db.fetch_one(query, {'execution_id': job_id})

            assert execution is not None

        finally:
            await db.close()
