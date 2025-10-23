"""
Performance Profilers

Memory and CPU profiling integration for detailed performance analysis.
"""

import time
import asyncio
import tracemalloc
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
import logging


@dataclass
class MemorySnapshot:
    """Memory usage snapshot"""
    timestamp: float
    current_mb: float
    peak_mb: float
    traced_mb: Optional[float] = None  # If using tracemalloc
    top_allocations: Optional[List[Dict[str, Any]]] = None


@dataclass
class CPUSnapshot:
    """CPU usage snapshot"""
    timestamp: float
    cpu_percent: float
    user_time: float
    system_time: float


class MemoryProfiler:
    """
    Memory profiling for pipeline execution

    Supports:
    - psutil for system-level memory
    - tracemalloc for Python-specific allocations
    - Snapshot-based tracking
    - Top allocation tracking
    """

    def __init__(self, use_tracemalloc: bool = False, top_allocations: int = 10):
        self.use_tracemalloc = use_tracemalloc
        self.top_allocations = top_allocations
        self.logger = logging.getLogger("MemoryProfiler")
        self._snapshots: List[MemorySnapshot] = []
        self._tracemalloc_started = False

    def start(self) -> None:
        """Start memory profiling"""
        if self.use_tracemalloc and not self._tracemalloc_started:
            tracemalloc.start()
            self._tracemalloc_started = True
            self.logger.debug("Started tracemalloc")

    def stop(self) -> None:
        """Stop memory profiling"""
        if self._tracemalloc_started:
            tracemalloc.stop()
            self._tracemalloc_started = False
            self.logger.debug("Stopped tracemalloc")

    def take_snapshot(self) -> MemorySnapshot:
        """Take a memory snapshot"""
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            current_mb=self._get_current_memory(),
            peak_mb=self._get_peak_memory()
        )

        if self._tracemalloc_started:
            snapshot.traced_mb = self._get_traced_memory()
            snapshot.top_allocations = self._get_top_allocations()

        self._snapshots.append(snapshot)
        return snapshot

    def get_stats(self) -> Dict[str, Any]:
        """Get memory profiling statistics"""
        if not self._snapshots:
            return {}

        current_mbs = [s.current_mb for s in self._snapshots]
        peak_mbs = [s.peak_mb for s in self._snapshots]

        stats = {
            'initial_mb': current_mbs[0],
            'final_mb': current_mbs[-1],
            'delta_mb': current_mbs[-1] - current_mbs[0],
            'peak_mb': max(peak_mbs),
            'average_mb': sum(current_mbs) / len(current_mbs),
            'snapshots': len(self._snapshots)
        }

        if self._tracemalloc_started:
            traced_mbs = [s.traced_mb for s in self._snapshots if s.traced_mb is not None]
            if traced_mbs:
                stats['traced_mb'] = {
                    'initial': traced_mbs[0],
                    'final': traced_mbs[-1],
                    'delta': traced_mbs[-1] - traced_mbs[0],
                    'peak': max(traced_mbs)
                }

        return stats

    def reset(self) -> None:
        """Reset profiler state"""
        self._snapshots.clear()
        if self._tracemalloc_started:
            tracemalloc.clear_traces()

    def _get_current_memory(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            self.logger.warning("psutil not installed, returning 0")
            return 0.0

    def _get_peak_memory(self) -> float:
        """Get peak memory usage in MB"""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            # Use memory_info for peak on Windows, memory_full_info on Unix
            try:
                return process.memory_full_info().uss / 1024 / 1024
            except (AttributeError, psutil.AccessDenied):
                return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def _get_traced_memory(self) -> float:
        """Get tracemalloc current memory in MB"""
        if not self._tracemalloc_started:
            return 0.0
        current, peak = tracemalloc.get_traced_memory()
        return current / 1024 / 1024

    def _get_top_allocations(self) -> List[Dict[str, Any]]:
        """Get top memory allocations"""
        if not self._tracemalloc_started:
            return []

        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')[:self.top_allocations]

        return [
            {
                'file': str(stat.traceback),
                'size_mb': stat.size / 1024 / 1024,
                'count': stat.count
            }
            for stat in top_stats
        ]


class CPUProfiler:
    """
    CPU profiling for pipeline execution

    Supports:
    - Process-level CPU monitoring
    - User/system time tracking
    - CPU percentage over time
    """

    def __init__(self, sample_interval: float = 0.1):
        self.sample_interval = sample_interval
        self.logger = logging.getLogger("CPUProfiler")
        self._snapshots: List[CPUSnapshot] = []
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start CPU profiling"""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_cpu())
        self.logger.debug("Started CPU monitoring")

    async def stop(self) -> None:
        """Stop CPU profiling"""
        if not self._monitoring:
            return

        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        self.logger.debug("Stopped CPU monitoring")

    async def _monitor_cpu(self) -> None:
        """Monitor CPU usage periodically"""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())

            while self._monitoring:
                cpu_times = process.cpu_times()
                snapshot = CPUSnapshot(
                    timestamp=time.time(),
                    cpu_percent=process.cpu_percent(interval=None),
                    user_time=cpu_times.user,
                    system_time=cpu_times.system
                )
                self._snapshots.append(snapshot)
                await asyncio.sleep(self.sample_interval)
        except ImportError:
            self.logger.warning("psutil not installed, CPU monitoring disabled")
        except asyncio.CancelledError:
            pass

    def get_stats(self) -> Dict[str, Any]:
        """Get CPU profiling statistics"""
        if not self._snapshots:
            return {}

        cpu_percents = [s.cpu_percent for s in self._snapshots]
        user_times = [s.user_time for s in self._snapshots]
        system_times = [s.system_time for s in self._snapshots]

        return {
            'average_cpu_percent': sum(cpu_percents) / len(cpu_percents),
            'peak_cpu_percent': max(cpu_percents),
            'total_user_time': user_times[-1] - user_times[0] if len(user_times) > 1 else 0,
            'total_system_time': system_times[-1] - system_times[0] if len(system_times) > 1 else 0,
            'total_cpu_time': (user_times[-1] - user_times[0]) + (system_times[-1] - system_times[0]) if len(user_times) > 1 else 0,
            'samples': len(self._snapshots)
        }

    def reset(self) -> None:
        """Reset profiler state"""
        self._snapshots.clear()


class CombinedProfiler:
    """
    Combined memory and CPU profiling

    Profiles both memory and CPU during function execution.
    """

    def __init__(
        self,
        use_tracemalloc: bool = False,
        cpu_sample_interval: float = 0.1
    ):
        self.memory_profiler = MemoryProfiler(use_tracemalloc=use_tracemalloc)
        self.cpu_profiler = CPUProfiler(sample_interval=cpu_sample_interval)
        self.logger = logging.getLogger("CombinedProfiler")

    async def profile(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Profile function execution for both memory and CPU

        Args:
            func: Async function to profile
            *args: Arguments to pass to func
            **kwargs: Keyword arguments to pass to func

        Returns:
            Dictionary with memory and CPU statistics
        """
        # Start profiling
        self.memory_profiler.start()
        await self.cpu_profiler.start()

        # Take initial snapshots
        self.memory_profiler.take_snapshot()

        try:
            # Execute function
            start_time = time.time()
            result = await func(*args, **kwargs)
            duration = time.time() - start_time

            # Take final snapshot
            self.memory_profiler.take_snapshot()

            # Stop profiling
            await self.cpu_profiler.stop()
            self.memory_profiler.stop()

            # Collect statistics
            stats = {
                'duration_seconds': duration,
                'memory': self.memory_profiler.get_stats(),
                'cpu': self.cpu_profiler.get_stats()
            }

            # Reset for next run
            self.memory_profiler.reset()
            self.cpu_profiler.reset()

            return stats

        except Exception as e:
            await self.cpu_profiler.stop()
            self.memory_profiler.stop()
            raise


async def profile_pipeline(
    pipeline_func: Callable,
    pipeline_data: Dict[str, Any],
    use_tracemalloc: bool = False,
    cpu_sample_interval: float = 0.1
) -> Dict[str, Any]:
    """
    Convenience function to profile a pipeline execution

    Args:
        pipeline_func: Async pipeline function to profile
        pipeline_data: Data to pass to pipeline
        use_tracemalloc: Enable tracemalloc for detailed Python allocations
        cpu_sample_interval: CPU sampling interval in seconds

    Returns:
        Profiling statistics
    """
    profiler = CombinedProfiler(
        use_tracemalloc=use_tracemalloc,
        cpu_sample_interval=cpu_sample_interval
    )
    return await profiler.profile(pipeline_func, pipeline_data)
