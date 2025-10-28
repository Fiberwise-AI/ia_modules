"""
Tests for Benchmark Framework

Tests core benchmarking functionality including:
- BenchmarkConfig validation
- BenchmarkRunner execution
- Statistical calculations
- BenchmarkSuite management
"""

import pytest
from ia_modules.pipeline.test_utils import create_test_execution_context
import asyncio
import time
from ia_modules.benchmarking.framework import (
    BenchmarkConfig,
    BenchmarkRunner,
    BenchmarkResult,
    BenchmarkSuite
)


class TestBenchmarkConfig:
    """Test BenchmarkConfig validation"""

    def test_default_config(self):
        """Test default configuration"""
        config = BenchmarkConfig()
        assert config.iterations == 100
        assert config.warmup_iterations == 5
        assert config.timeout is None
        assert config.profile_memory is False
        assert config.profile_cpu is False

    def test_custom_config(self):
        """Test custom configuration"""
        config = BenchmarkConfig(
            iterations=50,
            warmup_iterations=10,
            timeout=5.0,
            profile_memory=True
        )
        assert config.iterations == 50
        assert config.warmup_iterations == 10
        assert config.timeout == 5.0
        assert config.profile_memory is True

    def test_invalid_iterations(self):
        """Test error on invalid iterations"""
        with pytest.raises(ValueError, match="iterations must be >= 1"):
            BenchmarkConfig(iterations=0)

    def test_invalid_warmup(self):
        """Test error on invalid warmup iterations"""
        with pytest.raises(ValueError, match="warmup_iterations must be >= 0"):
            BenchmarkConfig(warmup_iterations=-1)

    def test_invalid_timeout(self):
        """Test error on invalid timeout"""
        with pytest.raises(ValueError, match="timeout must be > 0"):
            BenchmarkConfig(timeout=-1.0)


class TestBenchmarkRunner:
    """Test BenchmarkRunner execution"""

    @pytest.mark.asyncio
    async def test_simple_benchmark(self):
        """Test basic benchmark execution"""
        async def simple_func():
            await asyncio.sleep(0.01)  # 10ms

        config = BenchmarkConfig(iterations=10, warmup_iterations=2)
        runner = BenchmarkRunner(config)

        result = await runner.run("test_benchmark", simple_func)

        assert result.name == "test_benchmark"
        assert result.iterations == 10
        assert result.mean_time > 0
        assert result.median_time > 0
        assert result.min_time > 0
        assert result.max_time > 0

    @pytest.mark.asyncio
    async def test_benchmark_with_args(self):
        """Test benchmark with function arguments"""
        async def func_with_args(x, y):
            await asyncio.sleep(0.001)
            return x + y

        config = BenchmarkConfig(iterations=5, warmup_iterations=0)
        runner = BenchmarkRunner(config)

        result = await runner.run("test_args", func_with_args, 10, 20)

        assert result.name == "test_args"
        assert result.iterations == 5

    @pytest.mark.asyncio
    async def test_benchmark_statistics(self):
        """Test statistical calculations"""
        async def variable_func():
            # Variable execution time
            await asyncio.sleep(0.001 + (time.time() % 0.002))

        config = BenchmarkConfig(iterations=20, warmup_iterations=0)
        runner = BenchmarkRunner(config)

        result = await runner.run("test_stats", variable_func)

        # Check all statistics are calculated
        assert result.mean_time > 0
        assert result.median_time > 0
        assert result.std_dev >= 0
        assert result.min_time <= result.mean_time <= result.max_time
        assert result.p95_time >= result.median_time
        assert result.p99_time >= result.p95_time
        assert result.total_time == pytest.approx(sum([result.mean_time * result.iterations]), rel=0.5)

    @pytest.mark.asyncio
    async def test_benchmark_timeout(self):
        """Test benchmark with timeout"""
        async def slow_func():
            await asyncio.sleep(1.0)  # 1 second

        config = BenchmarkConfig(iterations=3, warmup_iterations=0, timeout=0.1)
        runner = BenchmarkRunner(config)

        # Should complete but with timeout values
        result = await runner.run("test_timeout", slow_func)

        assert result.iterations == 3
        # Some times should be the timeout value
        assert any(t == pytest.approx(0.1, abs=0.01) for t in [result.min_time, result.max_time])

    @pytest.mark.asyncio
    async def test_collect_intermediate_data(self):
        """Test collection of raw timing data"""
        async def simple_func():
            await asyncio.sleep(0.001)

        config = BenchmarkConfig(iterations=5, warmup_iterations=0, collect_intermediate=True)
        runner = BenchmarkRunner(config)

        result = await runner.run("test_raw", simple_func)

        assert result.raw_times is not None
        assert len(result.raw_times) == 5
        assert all(t > 0 for t in result.raw_times)


class TestBenchmarkResult:
    """Test BenchmarkResult functionality"""

    def test_result_creation(self):
        """Test creating benchmark result"""
        result = BenchmarkResult(
            name="test",
            iterations=10,
            mean_time=0.1,
            median_time=0.09,
            std_dev=0.02,
            min_time=0.08,
            max_time=0.15,
            p95_time=0.14,
            p99_time=0.15,
            total_time=1.0
        )

        assert result.name == "test"
        assert result.iterations == 10
        assert result.mean_time == 0.1

    def test_result_to_dict(self):
        """Test converting result to dictionary"""
        result = BenchmarkResult(
            name="test",
            iterations=10,
            mean_time=0.1,
            median_time=0.09,
            std_dev=0.02,
            min_time=0.08,
            max_time=0.15,
            p95_time=0.14,
            p99_time=0.15,
            total_time=1.0
        )

        d = result.to_dict()

        assert d['name'] == "test"
        assert d['iterations'] == 10
        assert d['mean_time'] == 0.1
        assert 'timestamp' in d

    def test_result_summary(self):
        """Test getting human-readable summary"""
        result = BenchmarkResult(
            name="test_benchmark",
            iterations=100,
            mean_time=0.05,
            median_time=0.048,
            std_dev=0.01,
            min_time=0.04,
            max_time=0.08,
            p95_time=0.07,
            p99_time=0.075,
            total_time=5.0
        )

        summary = result.get_summary()

        assert "test_benchmark" in summary
        assert "100" in summary
        assert "50.00ms" in summary  # mean_time * 1000


class TestBenchmarkSuite:
    """Test BenchmarkSuite functionality"""

    @pytest.mark.asyncio
    async def test_suite_creation(self):
        """Test creating benchmark suite"""
        suite = BenchmarkSuite("test_suite")

        assert suite.name == "test_suite"
        assert len(suite.results) == 0

    @pytest.mark.asyncio
    async def test_suite_add_benchmark(self):
        """Test adding benchmarks to suite"""
        async def func1():
            await asyncio.sleep(0.001)

        async def func2():
            await asyncio.sleep(0.002)

        config = BenchmarkConfig(iterations=5, warmup_iterations=0)
        suite = BenchmarkSuite("test_suite", config)

        result1 = await suite.add_benchmark("bench1", func1)
        result2 = await suite.add_benchmark("bench2", func2)

        assert len(suite.results) == 2
        assert suite.results[0].name == "bench1"
        assert suite.results[1].name == "bench2"

    @pytest.mark.asyncio
    async def test_suite_get_results(self):
        """Test getting suite results"""
        async def func():
            await asyncio.sleep(0.001)

        config = BenchmarkConfig(iterations=3, warmup_iterations=0)
        suite = BenchmarkSuite("test_suite", config)

        await suite.add_benchmark("bench1", func)
        await suite.add_benchmark("bench2", func)

        results = suite.get_results()

        assert len(results) == 2
        assert isinstance(results, list)
        # Should be a copy
        results.clear()
        assert len(suite.results) == 2

    @pytest.mark.asyncio
    async def test_suite_summary(self):
        """Test suite summary generation"""
        async def func():
            await asyncio.sleep(0.001)

        config = BenchmarkConfig(iterations=3, warmup_iterations=0)
        suite = BenchmarkSuite("test_suite", config)

        await suite.add_benchmark("bench1", func)
        await suite.add_benchmark("bench2", func)

        summary = suite.get_summary()

        assert "test_suite" in summary
        assert "bench1" in summary
        assert "bench2" in summary

    @pytest.mark.asyncio
    async def test_suite_clear_results(self):
        """Test clearing suite results"""
        async def func():
            await asyncio.sleep(0.001)

        config = BenchmarkConfig(iterations=3, warmup_iterations=0)
        suite = BenchmarkSuite("test_suite", config)

        await suite.add_benchmark("bench1", func)
        assert len(suite.results) == 1

        suite.clear_results()
        assert len(suite.results) == 0


class TestBenchmarkPerformance:
    """Test benchmark performance characteristics"""

    @pytest.mark.asyncio
    async def test_warmup_iterations_not_counted(self):
        """Test that warmup iterations are not included in results"""
        call_count = 0

        async def counting_func():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.001)

        config = BenchmarkConfig(iterations=10, warmup_iterations=5)
        runner = BenchmarkRunner(config)

        result = await runner.run("test_warmup", counting_func)

        # Should call 15 times total (5 warmup + 10 measured)
        assert call_count == 15
        # But result should only show 10 iterations
        assert result.iterations == 10

    @pytest.mark.asyncio
    async def test_benchmark_overhead_minimal(self):
        """Test that benchmarking overhead is minimal"""
        async def minimal_func():
            pass  # No-op function

        config = BenchmarkConfig(iterations=100, warmup_iterations=10)
        runner = BenchmarkRunner(config)

        result = await runner.run("test_overhead", minimal_func)

        # Overhead should be very small (< 1ms per iteration on average)
        assert result.mean_time < 0.001

    @pytest.mark.asyncio
    async def test_statistical_consistency(self):
        """Test that statistics are consistent across runs"""
        async def consistent_func():
            await asyncio.sleep(0.01)  # Consistent 10ms

        config = BenchmarkConfig(iterations=20, warmup_iterations=5)
        runner = BenchmarkRunner(config)

        result = await runner.run("test_consistency", consistent_func)

        # With consistent timing, std dev should be relatively small
        # (allowing for system variance)
        coefficient_of_variation = result.std_dev / result.mean_time
        assert coefficient_of_variation < 0.3  # Less than 30% variation (relaxed for system variance)
