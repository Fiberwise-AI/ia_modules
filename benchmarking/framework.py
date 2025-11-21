"""
Benchmark Framework

Core benchmarking infrastructure for measuring pipeline performance.
"""

import time
import asyncio
import statistics
from typing import Dict, Any, List, Optional, Callable
from dataclasses import asdict
import logging

# Import from models to avoid circular imports
from .models import BenchmarkConfig, BenchmarkResult

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Runs performance benchmarks on async functions

    Provides:
    - Statistical analysis of execution times
    - Memory profiling (optional)
    - CPU profiling (optional)
    - Warmup iterations
    - Timeout handling
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.logger = logging.getLogger("BenchmarkRunner")

    async def run(
        self,
        name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> BenchmarkResult:
        """
        Run benchmark on an async function

        Args:
            name: Name for this benchmark
            func: Async function to benchmark
            *args: Arguments to pass to func
            **kwargs: Keyword arguments to pass to func

        Returns:
            BenchmarkResult with statistics
        """
        self.logger.info(f"Starting benchmark: {name}")

        # Warmup
        if self.config.warmup_iterations > 0:
            self.logger.debug(f"Running {self.config.warmup_iterations} warmup iterations")
            await self._run_iterations(func, self.config.warmup_iterations, *args, **kwargs)

        # Actual benchmark
        self.logger.debug(f"Running {self.config.iterations} benchmark iterations")
        times = await self._run_iterations(func, self.config.iterations, *args, **kwargs)

        # Calculate statistics
        result = self._calculate_statistics(name, times)
        result.config = asdict(self.config)

        # Optional profiling
        if self.config.profile_memory:
            result.memory_stats = await self._profile_memory(func, *args, **kwargs)
            # Calculate memory per operation
            if result.memory_stats and 'delta_mb' in result.memory_stats:
                result.memory_per_operation_mb = result.memory_stats['delta_mb'] / result.iterations

        if self.config.profile_cpu:
            result.cpu_stats = await self._profile_cpu(func, *args, **kwargs)
            # Calculate CPU per operation
            if result.cpu_stats and 'average_cpu_percent' in result.cpu_stats:
                result.cpu_per_operation_percent = result.cpu_stats['average_cpu_percent']

        if self.config.collect_intermediate:
            result.raw_times = times

        self.logger.info(f"Completed benchmark: {name}")
        return result

    async def _run_iterations(
        self,
        func: Callable,
        iterations: int,
        *args,
        **kwargs
    ) -> List[float]:
        """Run function for specified iterations and collect times"""
        times = []

        for i in range(iterations):
            start_time = time.perf_counter()

            try:
                if self.config.timeout:
                    await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=self.config.timeout
                    )
                else:
                    await func(*args, **kwargs)
            except asyncio.TimeoutError:
                self.logger.warning(f"Iteration {i+1} timed out after {self.config.timeout}s")
                times.append(self.config.timeout)
                continue
            except Exception as e:
                self.logger.error(f"Iteration {i+1} failed: {e}")
                raise

            end_time = time.perf_counter()
            times.append(end_time - start_time)

        return times

    def _calculate_statistics(self, name: str, times: List[float]) -> BenchmarkResult:
        """Calculate statistical metrics from timing data"""
        if not times:
            raise ValueError("No timing data collected")

        sorted_times = sorted(times)
        n = len(sorted_times)
        mean_time = statistics.mean(times)
        total_time = sum(times)

        # Calculate throughput (operations per second)
        ops_per_sec = n / total_time if total_time > 0 else 0.0

        return BenchmarkResult(
            name=name,
            iterations=n,
            mean_time=mean_time,
            median_time=statistics.median(times),
            std_dev=statistics.stdev(times) if n > 1 else 0.0,
            min_time=min(times),
            max_time=max(times),
            p95_time=sorted_times[int(n * 0.95)] if n > 1 else sorted_times[0],
            p99_time=sorted_times[int(n * 0.99)] if n > 1 else sorted_times[0],
            total_time=total_time,
            operations_per_second=ops_per_sec
        )

    async def _profile_memory(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Profile memory usage during execution"""
        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())

            # Measure before
            mem_before = process.memory_info().rss / 1024 / 1024  # MB

            # Run function
            await func(*args, **kwargs)

            # Measure after
            mem_after = process.memory_info().rss / 1024 / 1024  # MB

            return {
                'before_mb': mem_before,
                'after_mb': mem_after,
                'delta_mb': mem_after - mem_before,
                'peak_mb': mem_after
            }
        except ImportError:
            self.logger.warning("psutil not installed, memory profiling disabled")
            return {}

    async def _profile_cpu(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Profile CPU usage during execution"""
        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())

            # Measure before
            cpu_before = process.cpu_percent(interval=0.1)

            # Run function
            start_time = time.time()
            await func(*args, **kwargs)
            duration = time.time() - start_time

            # Measure after
            cpu_after = process.cpu_percent(interval=0.1)

            return {
                'cpu_percent_before': cpu_before,
                'cpu_percent_after': cpu_after,
                'duration_seconds': duration,
                'cpu_time_seconds': duration * (cpu_after / 100)
            }
        except ImportError:
            self.logger.warning("psutil not installed, CPU profiling disabled")
            return {}


class BenchmarkSuite:
    """
    Manages a collection of related benchmarks
    """

    def __init__(self, name: str, config: Optional[BenchmarkConfig] = None):
        self.name = name
        self.config = config or BenchmarkConfig()
        self.runner = BenchmarkRunner(self.config)
        self.results: List[BenchmarkResult] = []
        self.logger = logging.getLogger(f"BenchmarkSuite.{name}")

    async def add_benchmark(
        self,
        name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> BenchmarkResult:
        """Add and run a benchmark"""
        result = await self.runner.run(name, func, *args, **kwargs)
        self.results.append(result)
        return result

    def get_results(self) -> List[BenchmarkResult]:
        """Get all benchmark results"""
        return self.results.copy()

    def get_summary(self) -> str:
        """Get summary of all benchmarks"""
        if not self.results:
            return f"{self.name}: No benchmarks run"

        summary = [f"{self.name} Benchmark Suite"]
        summary.append("=" * 50)

        for result in self.results:
            summary.append("")
            summary.append(result.get_summary())

        return "\n".join(summary)

    def clear_results(self) -> None:
        """Clear all benchmark results"""
        self.results.clear()
