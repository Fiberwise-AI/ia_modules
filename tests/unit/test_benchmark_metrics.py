"""
Tests for benchmark metrics (cost tracking, throughput, resource efficiency)
"""

import pytest
from ia_modules.pipeline.test_utils import create_test_execution_context
from ia_modules.benchmarking import BenchmarkResult, BenchmarkRunner, BenchmarkConfig


class TestCostTracking:
    """Test cost tracking metrics"""

    def test_set_cost_tracking(self):
        """Test setting cost tracking metrics"""
        result = BenchmarkResult(
            name="test",
            iterations=100,
            mean_time=0.1,
            median_time=0.1,
            std_dev=0.01,
            min_time=0.09,
            max_time=0.11,
            p95_time=0.105,
            p99_time=0.108,
            total_time=10.0
        )

        # Set cost tracking
        result.set_cost_tracking(api_calls=500, cost_usd=2.50)

        assert result.api_calls_count == 500
        assert result.estimated_cost_usd == 2.50
        assert result.cost_per_operation == 0.025  # $2.50 / 100 iterations

    def test_cost_tracking_zero_iterations(self):
        """Test cost tracking with zero iterations"""
        result = BenchmarkResult(
            name="test",
            iterations=0,
            mean_time=0.0,
            median_time=0.0,
            std_dev=0.0,
            min_time=0.0,
            max_time=0.0,
            p95_time=0.0,
            p99_time=0.0,
            total_time=0.0
        )

        result.set_cost_tracking(api_calls=100, cost_usd=1.00)

        assert result.api_calls_count == 100
        assert result.estimated_cost_usd == 1.00
        assert result.cost_per_operation == 0.0  # Should not divide by zero

    def test_cost_in_summary(self):
        """Test cost appears in summary"""
        result = BenchmarkResult(
            name="api_test",
            iterations=50,
            mean_time=0.2,
            median_time=0.2,
            std_dev=0.02,
            min_time=0.18,
            max_time=0.22,
            p95_time=0.21,
            p99_time=0.215,
            total_time=10.0
        )

        result.set_cost_tracking(api_calls=250, cost_usd=5.75)

        summary = result.get_summary()
        assert "API Calls: 250" in summary
        assert "Est. Cost: $5.7500" in summary
        assert "$0.115000/op" in summary


class TestThroughputMetrics:
    """Test throughput metrics"""

    def test_operations_per_second_calculated(self):
        """Test ops/sec is calculated automatically"""
        result = BenchmarkResult(
            name="test",
            iterations=100,
            mean_time=0.01,
            median_time=0.01,
            std_dev=0.001,
            min_time=0.009,
            max_time=0.011,
            p95_time=0.0105,
            p99_time=0.0108,
            total_time=1.0,
            operations_per_second=100.0
        )

        assert result.operations_per_second == 100.0

    def test_set_throughput(self):
        """Test setting throughput with items processed"""
        result = BenchmarkResult(
            name="data_processor",
            iterations=10,
            mean_time=0.5,
            median_time=0.5,
            std_dev=0.05,
            min_time=0.45,
            max_time=0.55,
            p95_time=0.52,
            p99_time=0.54,
            total_time=5.0
        )

        # Process 1000 items in 5 seconds
        result.set_throughput(items_processed=1000)

        assert result.items_processed == 1000
        assert result.items_per_second == 200.0  # 1000 / 5.0

    def test_throughput_in_summary(self):
        """Test throughput appears in summary"""
        result = BenchmarkResult(
            name="batch_processor",
            iterations=20,
            mean_time=0.1,
            median_time=0.1,
            std_dev=0.01,
            min_time=0.09,
            max_time=0.11,
            p95_time=0.105,
            p99_time=0.108,
            total_time=2.0,
            operations_per_second=10.0
        )

        result.set_throughput(items_processed=5000)

        summary = result.get_summary()
        assert "Throughput: 10.00 ops/sec" in summary
        assert "Items/sec: 2500.00" in summary


class TestResourceEfficiency:
    """Test resource efficiency metrics"""

    def test_memory_per_operation(self):
        """Test memory per operation calculation"""
        result = BenchmarkResult(
            name="memory_test",
            iterations=100,
            mean_time=0.01,
            median_time=0.01,
            std_dev=0.001,
            min_time=0.009,
            max_time=0.011,
            p95_time=0.0105,
            p99_time=0.0108,
            total_time=1.0,
            memory_per_operation_mb=2.5
        )

        assert result.memory_per_operation_mb == 2.5

    def test_cpu_per_operation(self):
        """Test CPU per operation metric"""
        result = BenchmarkResult(
            name="cpu_test",
            iterations=50,
            mean_time=0.02,
            median_time=0.02,
            std_dev=0.002,
            min_time=0.018,
            max_time=0.022,
            p95_time=0.021,
            p99_time=0.0215,
            total_time=1.0,
            cpu_per_operation_percent=45.5
        )

        assert result.cpu_per_operation_percent == 45.5

    def test_resource_efficiency_in_summary(self):
        """Test resource metrics appear in summary"""
        result = BenchmarkResult(
            name="resource_test",
            iterations=100,
            mean_time=0.01,
            median_time=0.01,
            std_dev=0.001,
            min_time=0.009,
            max_time=0.011,
            p95_time=0.0105,
            p99_time=0.0108,
            total_time=1.0,
            memory_per_operation_mb=3.25,
            cpu_per_operation_percent=62.8
        )

        summary = result.get_summary()
        assert "Memory/op: 3.25MB" in summary
        assert "CPU/op: 62.80%" in summary


class TestMethodChaining:
    """Test method chaining for setting metrics"""

    def test_chain_cost_and_throughput(self):
        """Test chaining cost and throughput methods"""
        result = BenchmarkResult(
            name="chain_test",
            iterations=100,
            mean_time=0.1,
            median_time=0.1,
            std_dev=0.01,
            min_time=0.09,
            max_time=0.11,
            p95_time=0.105,
            p99_time=0.108,
            total_time=10.0
        )

        # Chain methods
        result.set_cost_tracking(api_calls=500, cost_usd=2.50).set_throughput(items_processed=10000)

        assert result.api_calls_count == 500
        assert result.estimated_cost_usd == 2.50
        assert result.items_processed == 10000
        assert result.items_per_second == 1000.0

    def test_chain_returns_self(self):
        """Test that chaining methods return self"""
        result = BenchmarkResult(
            name="self_test",
            iterations=10,
            mean_time=0.1,
            median_time=0.1,
            std_dev=0.01,
            min_time=0.09,
            max_time=0.11,
            p95_time=0.105,
            p99_time=0.108,
            total_time=1.0
        )

        chained = result.set_cost_tracking(api_calls=10, cost_usd=0.5)
        assert chained is result


class TestCompleteExample:
    """Test complete example with all metrics"""

    @pytest.mark.asyncio
    async def test_complete_benchmark_with_all_metrics(self):
        """Test a complete benchmark with all metric types"""

        async def sample_task():
            """Sample async task"""
            await asyncio.sleep(0.01)
            return {"processed": 100}

        import asyncio

        config = BenchmarkConfig(iterations=10, warmup_iterations=2)
        runner = BenchmarkRunner(config)

        result = await runner.run("complete_test", sample_task)

        # Add all metric types
        result.set_cost_tracking(api_calls=50, cost_usd=1.25)
        result.set_throughput(items_processed=1000)

        # Verify all metrics
        assert result.iterations == 10
        assert result.operations_per_second > 0
        assert result.api_calls_count == 50
        assert result.estimated_cost_usd == 1.25
        assert result.items_processed == 1000
        assert result.items_per_second > 0

        # Verify summary includes all metrics
        summary = result.get_summary()
        assert "Throughput:" in summary
        assert "API Calls:" in summary
        assert "Est. Cost:" in summary
        assert "Items/sec:" in summary
