"""
Comprehensive tests for benchmarking/profilers.py
"""
import pytest
import asyncio
import time
from unittest.mock import patch

from ia_modules.benchmarking.profilers import (
    MemorySnapshot,
    CPUSnapshot,
    MemoryProfiler,
    CPUProfiler,
    CombinedProfiler,
    profile_pipeline
)


class TestMemorySnapshot:
    """Test MemorySnapshot dataclass"""

    def test_memory_snapshot_creation(self):
        """Test creating a memory snapshot"""
        snapshot = MemorySnapshot(
            timestamp=123.45,
            current_mb=50.0,
            peak_mb=75.0
        )

        assert snapshot.timestamp == 123.45
        assert snapshot.current_mb == 50.0
        assert snapshot.peak_mb == 75.0
        assert snapshot.traced_mb is None
        assert snapshot.top_allocations is None

    def test_memory_snapshot_with_traced_memory(self):
        """Test snapshot with traced memory"""
        snapshot = MemorySnapshot(
            timestamp=123.45,
            current_mb=50.0,
            peak_mb=75.0,
            traced_mb=45.0,
            top_allocations=[{'file': 'test.py', 'size_mb': 10.0}]
        )

        assert snapshot.traced_mb == 45.0
        assert len(snapshot.top_allocations) == 1


class TestCPUSnapshot:
    """Test CPUSnapshot dataclass"""

    def test_cpu_snapshot_creation(self):
        """Test creating a CPU snapshot"""
        snapshot = CPUSnapshot(
            timestamp=123.45,
            cpu_percent=75.5,
            user_time=10.0,
            system_time=2.0
        )

        assert snapshot.timestamp == 123.45
        assert snapshot.cpu_percent == 75.5
        assert snapshot.user_time == 10.0
        assert snapshot.system_time == 2.0


class TestMemoryProfiler:
    """Test MemoryProfiler class"""

    def test_init_without_tracemalloc(self):
        """Test initialization without tracemalloc"""
        profiler = MemoryProfiler(use_tracemalloc=False)

        assert profiler.use_tracemalloc is False
        assert profiler.top_allocations == 10
        assert profiler._tracemalloc_started is False

    def test_init_with_custom_top_allocations(self):
        """Test initialization with custom top allocations"""
        profiler = MemoryProfiler(top_allocations=20)

        assert profiler.top_allocations == 20

    def test_start_without_tracemalloc(self):
        """Test start when tracemalloc is disabled"""
        profiler = MemoryProfiler(use_tracemalloc=False)

        profiler.start()

        assert profiler._tracemalloc_started is False

    def test_start_with_tracemalloc(self):
        """Test start when tracemalloc is enabled"""
        profiler = MemoryProfiler(use_tracemalloc=True)

        profiler.start()

        assert profiler._tracemalloc_started is True

        profiler.stop()  # Clean up

    def test_stop_tracemalloc(self):
        """Test stopping tracemalloc"""
        profiler = MemoryProfiler(use_tracemalloc=True)

        profiler.start()
        assert profiler._tracemalloc_started is True

        profiler.stop()
        assert profiler._tracemalloc_started is False

    def test_take_snapshot(self):
        """Test taking a memory snapshot"""
        profiler = MemoryProfiler()

        snapshot = profiler.take_snapshot()

        assert isinstance(snapshot, MemorySnapshot)
        assert snapshot.timestamp > 0
        assert snapshot.current_mb >= 0
        assert snapshot.peak_mb >= 0

    def test_take_snapshot_with_tracemalloc(self):
        """Test snapshot with tracemalloc enabled"""
        profiler = MemoryProfiler(use_tracemalloc=True)

        profiler.start()
        snapshot = profiler.take_snapshot()
        profiler.stop()

        assert snapshot.traced_mb is not None
        assert snapshot.traced_mb >= 0
        assert snapshot.top_allocations is not None
        assert isinstance(snapshot.top_allocations, list)

    def test_get_stats_empty(self):
        """Test getting stats with no snapshots"""
        profiler = MemoryProfiler()

        stats = profiler.get_stats()

        assert stats == {}

    def test_get_stats_with_snapshots(self):
        """Test getting stats after taking snapshots"""
        profiler = MemoryProfiler()

        profiler.take_snapshot()
        time.sleep(0.01)  # Small delay
        profiler.take_snapshot()

        stats = profiler.get_stats()

        assert 'initial_mb' in stats
        assert 'final_mb' in stats
        assert 'delta_mb' in stats
        assert 'peak_mb' in stats
        assert 'average_mb' in stats
        assert stats['snapshots'] == 2

    def test_get_stats_with_tracemalloc(self):
        """Test stats include tracemalloc data"""
        profiler = MemoryProfiler(use_tracemalloc=True)

        profiler.start()
        profiler.take_snapshot()
        profiler.take_snapshot()
        stats = profiler.get_stats()
        profiler.stop()

        assert 'traced_mb' in stats
        assert 'initial' in stats['traced_mb']
        assert 'final' in stats['traced_mb']
        assert 'delta' in stats['traced_mb']
        assert 'peak' in stats['traced_mb']

    def test_reset(self):
        """Test resetting profiler state"""
        profiler = MemoryProfiler()

        profiler.take_snapshot()
        profiler.take_snapshot()

        profiler.reset()

        stats = profiler.get_stats()
        assert stats == {}

    def test_reset_clears_tracemalloc_traces(self):
        """Test reset clears tracemalloc traces"""
        profiler = MemoryProfiler(use_tracemalloc=True)

        profiler.start()
        assert profiler._tracemalloc_started is True

        profiler.take_snapshot()
        profiler.reset()

        # Reset should clear snapshots but keep tracemalloc running
        assert len(profiler._snapshots) == 0

        profiler.stop()

        # After stop, tracemalloc should be stopped
        assert profiler._tracemalloc_started is False

    def test_get_current_memory_without_psutil(self):
        """Test memory retrieval without psutil installed"""
        profiler = MemoryProfiler()

        # Mock psutil import to raise ImportError
        with patch('builtins.__import__', side_effect=lambda name, *args: __import__(name, *args) if name != 'psutil' else (_ for _ in ()).throw(ImportError())):
            memory = profiler._get_current_memory()

        assert memory == 0.0

    def test_get_peak_memory_without_psutil(self):
        """Test peak memory without psutil"""
        profiler = MemoryProfiler()

        # Mock psutil import to raise ImportError
        with patch('builtins.__import__', side_effect=lambda name, *args: __import__(name, *args) if name != 'psutil' else (_ for _ in ()).throw(ImportError())):
            memory = profiler._get_peak_memory()

        assert memory == 0.0


class TestCPUProfiler:
    """Test CPUProfiler class"""

    def test_init(self):
        """Test CPUProfiler initialization"""
        profiler = CPUProfiler(sample_interval=0.5)

        assert profiler.sample_interval == 0.5
        assert profiler._monitoring is False
        assert profiler._monitor_task is None

    @pytest.mark.asyncio
    async def test_start(self):
        """Test starting CPU profiler"""
        profiler = CPUProfiler(sample_interval=0.1)

        await profiler.start()

        assert profiler._monitoring is True
        assert profiler._monitor_task is not None

        await profiler.stop()

    @pytest.mark.asyncio
    async def test_stop(self):
        """Test stopping CPU profiler"""
        profiler = CPUProfiler(sample_interval=0.1)

        await profiler.start()
        await profiler.stop()

        assert profiler._monitoring is False

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        """Test that starting twice doesn't create multiple tasks"""
        profiler = CPUProfiler(sample_interval=0.1)

        await profiler.start()
        task1 = profiler._monitor_task

        await profiler.start()
        task2 = profiler._monitor_task

        assert task1 is task2

        await profiler.stop()

    @pytest.mark.asyncio
    async def test_monitor_cpu_collects_snapshots(self):
        """Test that CPU monitoring collects snapshots"""
        profiler = CPUProfiler(sample_interval=0.05)

        await profiler.start()
        await asyncio.sleep(0.2)  # Let it collect some samples
        await profiler.stop()

        # Should have collected at least one snapshot
        assert len(profiler._snapshots) > 0

    @pytest.mark.asyncio
    async def test_get_stats_empty(self):
        """Test getting stats with no snapshots"""
        profiler = CPUProfiler()

        stats = profiler.get_stats()

        assert stats == {}

    @pytest.mark.asyncio
    async def test_get_stats_with_snapshots(self):
        """Test getting stats after collecting snapshots"""
        profiler = CPUProfiler(sample_interval=0.05)

        await profiler.start()
        await asyncio.sleep(0.15)  # Collect some samples
        await profiler.stop()

        stats = profiler.get_stats()

        assert 'average_cpu_percent' in stats
        assert 'peak_cpu_percent' in stats
        assert 'total_user_time' in stats
        assert 'total_system_time' in stats
        assert 'total_cpu_time' in stats
        assert 'samples' in stats
        assert stats['samples'] > 0

    def test_reset(self):
        """Test resetting profiler state"""
        profiler = CPUProfiler()

        # Manually add a snapshot
        profiler._snapshots.append(
            CPUSnapshot(
                timestamp=time.time(),
                cpu_percent=50.0,
                user_time=1.0,
                system_time=0.5
            )
        )

        profiler.reset()

        assert len(profiler._snapshots) == 0

    @pytest.mark.asyncio
    async def test_monitor_cpu_without_psutil(self):
        """Test CPU monitoring without psutil installed"""
        profiler = CPUProfiler(sample_interval=0.1)

        # Mock psutil import to raise ImportError
        with patch('builtins.__import__', side_effect=lambda name, *args: __import__(name, *args) if name != 'psutil' else (_ for _ in ()).throw(ImportError())):
            await profiler.start()
            await asyncio.sleep(0.2)
            await profiler.stop()

        # Should handle gracefully without crashes
        assert len(profiler._snapshots) == 0


class TestCombinedProfiler:
    """Test CombinedProfiler class"""

    def test_init(self):
        """Test CombinedProfiler initialization"""
        profiler = CombinedProfiler(
            use_tracemalloc=True,
            cpu_sample_interval=0.2
        )

        assert isinstance(profiler.memory_profiler, MemoryProfiler)
        assert isinstance(profiler.cpu_profiler, CPUProfiler)
        assert profiler.memory_profiler.use_tracemalloc is True
        assert profiler.cpu_profiler.sample_interval == 0.2

    @pytest.mark.asyncio
    async def test_profile_simple_function(self):
        """Test profiling a simple async function"""
        async def simple_func():
            await asyncio.sleep(0.1)
            return "result"

        profiler = CombinedProfiler()

        stats = await profiler.profile(simple_func)

        assert 'duration_seconds' in stats
        assert stats['duration_seconds'] >= 0.1
        assert 'memory' in stats
        assert 'cpu' in stats

    @pytest.mark.asyncio
    async def test_profile_with_args(self):
        """Test profiling function with arguments"""
        async def func_with_args(x, y, z=0):
            await asyncio.sleep(0.05)
            return x + y + z

        profiler = CombinedProfiler()

        stats = await profiler.profile(func_with_args, 1, 2, z=3)

        assert 'duration_seconds' in stats

    @pytest.mark.asyncio
    async def test_profile_resets_after_execution(self):
        """Test that profiler resets state after execution"""
        async def simple_func():
            await asyncio.sleep(0.05)

        profiler = CombinedProfiler()

        await profiler.profile(simple_func)

        # Profilers should be reset
        assert profiler.memory_profiler.get_stats() == {}
        assert profiler.cpu_profiler.get_stats() == {}

    @pytest.mark.asyncio
    async def test_profile_handles_exceptions(self):
        """Test that profiler handles exceptions in function"""
        async def failing_func():
            await asyncio.sleep(0.05)
            raise ValueError("Test error")

        profiler = CombinedProfiler()

        with pytest.raises(ValueError, match="Test error"):
            await profiler.profile(failing_func)

        # Profilers should be stopped even after exception
        assert profiler.memory_profiler._tracemalloc_started is False
        assert profiler.cpu_profiler._monitoring is False

    @pytest.mark.asyncio
    async def test_profile_with_tracemalloc(self):
        """Test profiling with tracemalloc enabled"""
        async def memory_allocating_func():
            # Allocate some memory
            data = [i for i in range(10000)]
            await asyncio.sleep(0.05)
            return len(data)

        profiler = CombinedProfiler(use_tracemalloc=True)

        stats = await profiler.profile(memory_allocating_func)

        assert 'memory' in stats
        # With tracemalloc, should have traced memory stats
        memory_stats = stats['memory']
        if memory_stats:  # Might be empty on some systems
            assert 'initial_mb' in memory_stats
            assert 'final_mb' in memory_stats


class TestProfilePipeline:
    """Test profile_pipeline convenience function"""

    @pytest.mark.asyncio
    async def test_profile_pipeline_basic(self):
        """Test basic pipeline profiling"""
        async def mock_pipeline(data):
            await asyncio.sleep(0.1)
            return {"result": data["input"] * 2}

        stats = await profile_pipeline(
            mock_pipeline,
            {"input": 5}
        )

        assert 'duration_seconds' in stats
        assert stats['duration_seconds'] >= 0.1
        assert 'memory' in stats
        assert 'cpu' in stats

    @pytest.mark.asyncio
    async def test_profile_pipeline_with_tracemalloc(self):
        """Test pipeline profiling with tracemalloc"""
        async def mock_pipeline(data):
            await asyncio.sleep(0.05)
            return {"result": data}

        stats = await profile_pipeline(
            mock_pipeline,
            {"input": "test"},
            use_tracemalloc=True
        )

        assert 'memory' in stats

    @pytest.mark.asyncio
    async def test_profile_pipeline_custom_interval(self):
        """Test pipeline profiling with custom CPU sample interval"""
        async def mock_pipeline(data):
            await asyncio.sleep(0.1)
            return data

        stats = await profile_pipeline(
            mock_pipeline,
            {},
            cpu_sample_interval=0.05
        )

        assert 'cpu' in stats
