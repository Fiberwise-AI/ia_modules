"""
Edge case tests for benchmarking/profilers.py to reach 100% coverage
"""
import pytest
import asyncio
from unittest.mock import MagicMock, patch, PropertyMock

from ia_modules.benchmarking.profilers import (
    MemoryProfiler,
    CPUProfiler,
)


class TestMemoryProfilerEdgeCases:
    """Test edge cases in MemoryProfiler"""

    def test_get_peak_memory_attribute_error_fallback(self):
        """Test that _get_peak_memory falls back to rss when memory_full_info() raises AttributeError"""
        profiler = MemoryProfiler()

        # Mock psutil (imported inside the method)
        with patch('psutil.Process') as mock_process_cls:
            mock_process = MagicMock()
            mock_process.memory_full_info.side_effect = AttributeError("uss not available")
            mock_process.memory_info.return_value.rss = 1024 * 1024 * 100  # 100 MB
            mock_process_cls.return_value = mock_process

            result = profiler._get_peak_memory()

            # Should fall back to rss
            assert result == 100.0  # 100 MB

    def test_get_peak_memory_access_denied_fallback(self):
        """Test that _get_peak_memory falls back to rss when memory_full_info() raises AccessDenied"""
        profiler = MemoryProfiler()

        # Mock psutil (imported inside the method)
        import psutil
        with patch('psutil.Process') as mock_process_cls:
            mock_process = MagicMock()
            mock_process.memory_full_info.side_effect = psutil.AccessDenied("Permission denied")
            mock_process.memory_info.return_value.rss = 1024 * 1024 * 150  # 150 MB
            mock_process_cls.return_value = mock_process

            result = profiler._get_peak_memory()

            # Should fall back to rss
            assert result == 150.0  # 150 MB

    def test_get_traced_memory_when_not_started(self):
        """Test that _get_traced_memory returns 0.0 when tracemalloc not started"""
        profiler = MemoryProfiler(use_tracemalloc=False)

        # Don't start profiler
        result = profiler._get_traced_memory()

        assert result == 0.0

    def test_get_top_allocations_when_not_started(self):
        """Test that _get_top_allocations returns empty list when tracemalloc not started"""
        profiler = MemoryProfiler(use_tracemalloc=False)

        # Don't start profiler
        result = profiler._get_top_allocations()

        assert result == []

    def test_get_stats_without_traced_mb_data(self):
        """Test get_stats when tracemalloc is started but no traced_mb data available"""
        profiler = MemoryProfiler(use_tracemalloc=False)

        # Take snapshot without tracemalloc
        profiler.take_snapshot()

        stats = profiler.get_stats()

        # Should not have traced_mb key since tracemalloc wasn't started
        assert 'traced_mb' not in stats
        assert 'initial_mb' in stats


class TestCPUProfilerEdgeCases:
    """Test edge cases in CPUProfiler"""

    @pytest.mark.asyncio
    async def test_stop_when_not_monitoring(self):
        """Test that stop() returns early when not monitoring"""
        profiler = CPUProfiler()

        # Stop without starting
        await profiler.stop()

        # Should return early without error
        assert profiler._monitoring is False
        assert profiler._monitor_task is None

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        """Test that calling start() twice doesn't create multiple tasks"""
        profiler = CPUProfiler(sample_interval=0.1)

        await profiler.start()
        first_task = profiler._monitor_task

        await profiler.start()  # Call again
        second_task = profiler._monitor_task

        # Should be the same task (idempotent)
        assert first_task is second_task
        assert profiler._monitoring is True

        await profiler.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_monitor_task(self):
        """Test that stop() properly cancels the monitoring task"""
        profiler = CPUProfiler(sample_interval=0.1)

        await profiler.start()
        assert profiler._monitoring is True
        assert profiler._monitor_task is not None

        await profiler.stop()

        assert profiler._monitoring is False
        # Task should be cancelled
        assert profiler._monitor_task.cancelled() or profiler._monitor_task.done()

    @pytest.mark.asyncio
    async def test_monitor_task_handles_cancellation(self):
        """Test that the monitoring task handles CancelledError gracefully"""
        profiler = CPUProfiler(sample_interval=0.05)

        await profiler.start()
        await asyncio.sleep(0.02)  # Let it run briefly

        # Stop should cancel the task and catch CancelledError
        await profiler.stop()

        # Should complete without raising exception
        assert profiler._monitoring is False
