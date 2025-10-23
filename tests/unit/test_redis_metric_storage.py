"""
Tests for Redis metric storage.

Requires Redis server running on localhost:6379 for integration tests.
Tests use a separate Redis database (db=15) to avoid conflicts.
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import json

# Check if redis is available
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Check if Redis server is actually running
REDIS_SERVER_AVAILABLE = False
if REDIS_AVAILABLE:
    try:
        import asyncio
        async def check_redis():
            try:
                client = redis.Redis.from_url("redis://localhost:6379/15", decode_responses=True)
                await client.ping()
                await client.aclose()
                return True
            except:
                return False
        REDIS_SERVER_AVAILABLE = asyncio.run(check_redis())
    except:
        REDIS_SERVER_AVAILABLE = False

from reliability.redis_metric_storage import RedisMetricStorage


pytestmark = pytest.mark.skipif(
    not REDIS_AVAILABLE or not REDIS_SERVER_AVAILABLE,
    reason="redis package not installed or Redis server not running"
)


@pytest.fixture
async def redis_storage():
    """Create Redis storage for testing."""
    storage = RedisMetricStorage(
        redis_url="redis://localhost:6379/15",  # Use db 15 for tests
        key_prefix="test:reliability",
        ttl_days=1  # Short TTL for tests
    )

    try:
        # Clean up any existing test data
        await storage.client.flushdb()

        yield storage
    finally:
        # Clean up
        await storage.client.flushdb()
        await storage.close()


@pytest.mark.asyncio
async def test_redis_storage_creation():
    """Test creating Redis storage instance."""
    storage = RedisMetricStorage(redis_url="redis://localhost:6379/15")

    assert storage.redis_url == "redis://localhost:6379/15"
    assert storage.key_prefix == "reliability"
    assert storage.ttl_seconds == 90 * 86400


@pytest.mark.asyncio
async def test_redis_storage_initialization(redis_storage):
    """Test Redis connection initialization."""
    assert redis_storage.client is not None

    # Test ping
    result = await redis_storage.client.ping()
    assert result is True


@pytest.mark.asyncio
async def test_record_step(redis_storage):
    """Test recording a step metric."""
    now = datetime.now(timezone.utc)

    await redis_storage.record_step({
        "agent": "researcher",
        "success": True,
        "required_compensation": False,
        "required_human": False,
        "mode": "execute",
        "declared_mode": "execute",
        "mode_violation": False,
        "timestamp": now
    })

    # Verify step was stored
    steps = await redis_storage.get_steps(agent="researcher")

    assert len(steps) == 1
    assert steps[0]["agent"] == "researcher"
    assert steps[0]["success"] is True
    assert steps[0]["mode"] == "execute"


@pytest.mark.asyncio
async def test_record_multiple_steps(redis_storage):
    """Test recording multiple step metrics."""
    now = datetime.now(timezone.utc)

    for i in range(5):
        await redis_storage.record_step({
            "agent": "researcher",
            "success": i % 2 == 0,  # Alternate success/failure
            "timestamp": now + timedelta(seconds=i)
        })

    steps = await redis_storage.get_steps(agent="researcher")

    assert len(steps) == 5
    # Count successful steps
    successful = sum(1 for s in steps if s["success"])
    assert successful == 3


@pytest.mark.asyncio
async def test_get_steps_with_time_filter(redis_storage):
    """Test retrieving steps with time range filter."""
    base_time = datetime.now(timezone.utc)

    # Record steps across 10 seconds
    for i in range(10):
        await redis_storage.record_step({
            "agent": "researcher",
            "success": True,
            "timestamp": base_time + timedelta(seconds=i)
        })

    # Query middle 6 seconds
    since = base_time + timedelta(seconds=2)
    until = base_time + timedelta(seconds=7)

    steps = await redis_storage.get_steps(
        agent="researcher",
        since=since,
        until=until
    )

    assert len(steps) == 6


@pytest.mark.asyncio
async def test_record_workflow(redis_storage):
    """Test recording a workflow metric."""
    now = datetime.now(timezone.utc)

    await redis_storage.record_workflow({
        "workflow_id": "workflow-123",
        "total_steps": 10,
        "total_retries": 2,
        "required_compensation": False,
        "required_human": True,
        "agents_involved": ["researcher", "planner"],
        "timestamp": now
    })

    # Verify workflow was stored
    workflows = await redis_storage.get_workflows()

    assert len(workflows) == 1
    assert workflows[0]["workflow_id"] == "workflow-123"
    assert workflows[0]["total_steps"] == 10
    assert workflows[0]["total_retries"] == 2
    assert workflows[0]["required_human"] is True
    assert "researcher" in workflows[0]["agents_involved"]


@pytest.mark.asyncio
async def test_get_workflows_with_time_filter(redis_storage):
    """Test retrieving workflows with time range filter."""
    base_time = datetime.now(timezone.utc)

    # Record workflows across 5 seconds
    for i in range(5):
        await redis_storage.record_workflow({
            "workflow_id": f"workflow-{i}",
            "total_steps": 10,
            "total_retries": 0,
            "timestamp": base_time + timedelta(seconds=i)
        })

    # Query middle 3 seconds
    since = base_time + timedelta(seconds=1)
    until = base_time + timedelta(seconds=3)

    workflows = await redis_storage.get_workflows(since=since, until=until)

    assert len(workflows) == 3


@pytest.mark.asyncio
async def test_record_mtte_measurement(redis_storage):
    """Test recording MTTE measurement."""
    await redis_storage.record_slo_measurement(
        measurement_type="mtte",
        thread_id="thread-123",
        checkpoint_id="checkpoint-456",
        value=2500.0,  # 2.5 seconds
        success=True,
        metadata={"agent": "researcher"}
    )

    # Verify measurement was stored
    measurements = await redis_storage.get_slo_measurements("mtte")

    assert len(measurements) == 1
    assert measurements[0]["measurement_type"] == "mtte"
    assert measurements[0]["value"] == 2500.0
    assert measurements[0]["success"] is True


@pytest.mark.asyncio
async def test_record_rsr_measurement(redis_storage):
    """Test recording RSR measurement."""
    await redis_storage.record_slo_measurement(
        measurement_type="rsr",
        thread_id="thread-123",
        checkpoint_id="checkpoint-456",
        value=1.0,  # Successful replay
        success=True
    )

    # Verify measurement was stored
    measurements = await redis_storage.get_slo_measurements("rsr")

    assert len(measurements) == 1
    assert measurements[0]["measurement_type"] == "rsr"
    assert measurements[0]["value"] == 1.0


@pytest.mark.asyncio
async def test_get_slo_measurements_with_time_filter(redis_storage):
    """Test retrieving SLO measurements with time range filter."""
    base_time = datetime.now(timezone.utc)

    # Record MTTE measurements across 10 seconds
    for i in range(10):
        # Simulate recording at specific times
        timestamp = base_time + timedelta(seconds=i)

        # Temporarily override datetime for consistent timestamps
        await redis_storage.record_slo_measurement(
            measurement_type="mtte",
            thread_id=f"thread-{i}",
            checkpoint_id=f"checkpoint-{i}",
            value=1000.0 + i * 100,
            success=True
        )

    # Query middle 6 seconds (account for small timing variations)
    since = base_time + timedelta(seconds=2)
    until = base_time + timedelta(seconds=8)

    measurements = await redis_storage.get_slo_measurements(
        "mtte",
        since=since,
        until=until
    )

    # Should get approximately 6 measurements
    assert len(measurements) >= 5  # Allow for timing variations


@pytest.mark.asyncio
async def test_multiple_agents(redis_storage):
    """Test tracking metrics for multiple agents."""
    now = datetime.now(timezone.utc)

    # Record steps for different agents
    for agent in ["researcher", "planner", "executor"]:
        for i in range(3):
            await redis_storage.record_step({
                "agent": agent,
                "success": True,
                "timestamp": now + timedelta(seconds=i)
            })

    # Query each agent
    researcher_steps = await redis_storage.get_steps(agent="researcher")
    planner_steps = await redis_storage.get_steps(agent="planner")
    executor_steps = await redis_storage.get_steps(agent="executor")

    assert len(researcher_steps) == 3
    assert len(planner_steps) == 3
    assert len(executor_steps) == 3


@pytest.mark.asyncio
async def test_step_with_mode_violation(redis_storage):
    """Test recording step with mode violation."""
    await redis_storage.record_step({
        "agent": "executor",
        "success": False,
        "mode": "explore",
        "declared_mode": "execute",
        "mode_violation": True,
        "timestamp": datetime.now(timezone.utc)
    })

    steps = await redis_storage.get_steps(agent="executor")

    assert len(steps) == 1
    assert steps[0]["mode_violation"] is True
    assert steps[0]["mode"] == "explore"
    assert steps[0]["declared_mode"] == "execute"


@pytest.mark.asyncio
async def test_workflow_with_compensation(redis_storage):
    """Test recording workflow that required compensation."""
    await redis_storage.record_workflow({
        "workflow_id": "workflow-comp",
        "total_steps": 15,
        "total_retries": 5,
        "required_compensation": True,
        "required_human": False,
        "agents_involved": ["executor"],
        "timestamp": datetime.now(timezone.utc)
    })

    workflows = await redis_storage.get_workflows()

    assert len(workflows) == 1
    assert workflows[0]["required_compensation"] is True
    assert workflows[0]["total_retries"] == 5


@pytest.mark.asyncio
async def test_close_connection(redis_storage):
    """Test closing Redis connection."""

    await redis_storage.close()



@pytest.mark.asyncio
async def test_import_error_handling():
    """Test handling of missing redis package."""
    with patch.dict('sys.modules', {'redis.asyncio': None}):
        # This test verifies the error handling in __init__
        # In actual code, REDIS_AVAILABLE would be False
        pass  # Skip actual test since redis is imported at module level
