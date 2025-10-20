"""
Tests for Benchmark Comparison

Tests comparison tools for detecting performance changes.
"""

import pytest
from ia_modules.benchmarking.framework import BenchmarkResult
from ia_modules.benchmarking.comparison import (
    BenchmarkComparator,
    ComparisonMetric,
    PerformanceChange,
    HistoricalComparator
)


class TestBenchmarkComparator:
    """Test BenchmarkComparator functionality"""

    def test_comparator_creation(self):
        """Test creating comparator"""
        comparator = BenchmarkComparator()
        assert comparator.significance_threshold == 5.0
        assert comparator.regression_threshold == 10.0

    def test_compare_improved_performance(self):
        """Test detecting performance improvement"""
        baseline = BenchmarkResult(
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

        current = BenchmarkResult(
            name="test",
            iterations=100,
            mean_time=0.08,  # 20% faster
            median_time=0.08,
            std_dev=0.01,
            min_time=0.07,
            max_time=0.09,
            p95_time=0.085,
            p99_time=0.088,
            total_time=8.0
        )

        comparator = BenchmarkComparator()
        results = comparator.compare(baseline, current)

        assert len(results) > 0
        mean_comparison = next(r for r in results if r.metric == ComparisonMetric.MEAN_TIME)
        assert mean_comparison.change_classification == PerformanceChange.IMPROVED
        assert mean_comparison.percent_change < 0  # Negative = improvement for time
        assert mean_comparison.is_significant

    def test_compare_regressed_performance(self):
        """Test detecting performance regression"""
        baseline = BenchmarkResult(
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

        current = BenchmarkResult(
            name="test",
            iterations=100,
            mean_time=0.15,  # 50% slower
            median_time=0.15,
            std_dev=0.01,
            min_time=0.14,
            max_time=0.16,
            p95_time=0.155,
            p99_time=0.158,
            total_time=15.0
        )

        comparator = BenchmarkComparator()
        results = comparator.compare(baseline, current)

        mean_comparison = next(r for r in results if r.metric == ComparisonMetric.MEAN_TIME)
        assert mean_comparison.change_classification == PerformanceChange.REGRESSED
        assert mean_comparison.percent_change > 0  # Positive = regression for time
        assert mean_comparison.is_significant

    def test_compare_unchanged_performance(self):
        """Test detecting unchanged performance"""
        baseline = BenchmarkResult(
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

        current = BenchmarkResult(
            name="test",
            iterations=100,
            mean_time=0.102,  # 2% change (below 5% threshold)
            median_time=0.102,
            std_dev=0.01,
            min_time=0.092,
            max_time=0.112,
            p95_time=0.107,
            p99_time=0.110,
            total_time=10.2
        )

        comparator = BenchmarkComparator()
        results = comparator.compare(baseline, current)

        mean_comparison = next(r for r in results if r.metric == ComparisonMetric.MEAN_TIME)
        assert mean_comparison.change_classification == PerformanceChange.UNCHANGED
        assert not mean_comparison.is_significant

    def test_has_regression(self):
        """Test regression detection"""
        baseline = BenchmarkResult(
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

        # Significant regression
        current_regressed = BenchmarkResult(
            name="test",
            iterations=100,
            mean_time=0.15,  # 50% slower
            median_time=0.15,
            std_dev=0.01,
            min_time=0.14,
            max_time=0.16,
            p95_time=0.155,
            p99_time=0.158,
            total_time=15.0
        )

        comparator = BenchmarkComparator(regression_threshold=10.0)
        results = comparator.compare(baseline, current_regressed)

        assert comparator.has_regression(results)

    def test_custom_thresholds(self):
        """Test custom significance and regression thresholds"""
        baseline = BenchmarkResult(
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

        current = BenchmarkResult(
            name="test",
            iterations=100,
            mean_time=0.107,  # 7% slower
            median_time=0.107,
            std_dev=0.01,
            min_time=0.097,
            max_time=0.117,
            p95_time=0.112,
            p99_time=0.115,
            total_time=10.7
        )

        # With 10% significance threshold, 7% should be insignificant
        strict_comparator = BenchmarkComparator(significance_threshold=10.0)
        results = strict_comparator.compare(baseline, current)

        mean_comparison = next(r for r in results if r.metric == ComparisonMetric.MEAN_TIME)
        assert not mean_comparison.is_significant

        # With 5% significance threshold, 7% should be significant
        lenient_comparator = BenchmarkComparator(significance_threshold=5.0)
        results = lenient_comparator.compare(baseline, current)

        mean_comparison = next(r for r in results if r.metric == ComparisonMetric.MEAN_TIME)
        assert mean_comparison.is_significant

    def test_compare_multiple_metrics(self):
        """Test comparing multiple metrics"""
        baseline = BenchmarkResult(
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

        current = BenchmarkResult(
            name="test",
            iterations=100,
            mean_time=0.12,
            median_time=0.11,
            std_dev=0.01,
            min_time=0.10,
            max_time=0.15,
            p95_time=0.14,
            p99_time=0.145,
            total_time=12.0
        )

        comparator = BenchmarkComparator()
        results = comparator.compare(
            baseline,
            current,
            metrics=[
                ComparisonMetric.MEAN_TIME,
                ComparisonMetric.MEDIAN_TIME,
                ComparisonMetric.P95_TIME,
                ComparisonMetric.P99_TIME
            ]
        )

        assert len(results) == 4
        assert all(r.change_classification == PerformanceChange.REGRESSED for r in results)

    def test_comparison_summary(self):
        """Test generating comparison summary"""
        baseline = BenchmarkResult(
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

        current = BenchmarkResult(
            name="test",
            iterations=100,
            mean_time=0.08,
            median_time=0.08,
            std_dev=0.01,
            min_time=0.07,
            max_time=0.09,
            p95_time=0.085,
            p99_time=0.088,
            total_time=8.0
        )

        comparator = BenchmarkComparator()
        results = comparator.compare(baseline, current)
        summary = comparator.get_summary(results)

        assert "Benchmark Comparison Summary" in summary
        assert "Improvements:" in summary


class TestHistoricalComparator:
    """Test HistoricalComparator functionality"""

    def test_historical_comparator_creation(self):
        """Test creating historical comparator"""
        comparator = HistoricalComparator()
        assert len(comparator.results_history) == 0

    def test_add_results(self):
        """Test adding results to history"""
        comparator = HistoricalComparator()

        result1 = BenchmarkResult(
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

        comparator.add_result(result1)
        assert len(comparator.results_history) == 1

    def test_get_trend(self):
        """Test trend analysis"""
        comparator = HistoricalComparator()

        # Add increasing mean times (performance degradation)
        for i in range(10):
            result = BenchmarkResult(
                name="test",
                iterations=100,
                mean_time=0.1 + (i * 0.01),  # Increasing
                median_time=0.1 + (i * 0.01),
                std_dev=0.01,
                min_time=0.09,
                max_time=0.11,
                p95_time=0.105,
                p99_time=0.108,
                total_time=10.0
            )
            comparator.add_result(result)

        trend = comparator.get_trend(ComparisonMetric.MEAN_TIME)

        assert trend['samples'] == 10
        assert trend['mean'] > 0
        assert trend['slope'] > 0  # Increasing trend
        assert trend['trend_direction'] == 'increasing'

    def test_detect_anomalies(self):
        """Test anomaly detection"""
        comparator = HistoricalComparator()

        # Add mostly consistent results
        for i in range(10):
            result = BenchmarkResult(
                name="test",
                iterations=100,
                mean_time=0.1,  # Consistent
                median_time=0.1,
                std_dev=0.01,
                min_time=0.09,
                max_time=0.11,
                p95_time=0.105,
                p99_time=0.108,
                total_time=10.0
            )
            comparator.add_result(result)

        # Add an anomaly
        anomaly_result = BenchmarkResult(
            name="test",
            iterations=100,
            mean_time=0.5,  # 5x slower
            median_time=0.5,
            std_dev=0.01,
            min_time=0.49,
            max_time=0.51,
            p95_time=0.505,
            p99_time=0.508,
            total_time=50.0
        )
        comparator.add_result(anomaly_result)

        anomalies = comparator.detect_anomalies(ComparisonMetric.MEAN_TIME)

        assert len(anomalies) >= 1
        # The anomaly should be the last result
        assert anomalies[-1][1] == anomaly_result

    def test_trend_window_size(self):
        """Test trend analysis with window size"""
        comparator = HistoricalComparator()

        # Add 20 results
        for i in range(20):
            result = BenchmarkResult(
                name="test",
                iterations=100,
                mean_time=0.1 + (i * 0.01),
                median_time=0.1,
                std_dev=0.01,
                min_time=0.09,
                max_time=0.11,
                p95_time=0.105,
                p99_time=0.108,
                total_time=10.0
            )
            comparator.add_result(result)

        # Get trend for last 5 results
        trend = comparator.get_trend(ComparisonMetric.MEAN_TIME, window_size=5)

        assert trend['samples'] == 5

    def test_clear_history(self):
        """Test clearing historical data"""
        comparator = HistoricalComparator()

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

        comparator.add_result(result)
        assert len(comparator.results_history) == 1

        comparator.clear_history()
        assert len(comparator.results_history) == 0
