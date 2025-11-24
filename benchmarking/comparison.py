"""
Benchmark Comparison Tools

Compare benchmark results across runs, versions, and configurations.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import statistics

from .models import BenchmarkResult


class ComparisonMetric(Enum):
    """Metrics that can be compared"""
    MEAN_TIME = "mean_time"
    MEDIAN_TIME = "median_time"
    P95_TIME = "p95_time"
    P99_TIME = "p99_time"
    MAX_TIME = "max_time"
    MEMORY_DELTA = "memory_delta"
    CPU_AVERAGE = "cpu_average"


class PerformanceChange(Enum):
    """Classification of performance changes"""
    IMPROVED = "improved"
    REGRESSED = "regressed"
    UNCHANGED = "unchanged"


@dataclass
class ComparisonResult:
    """Result of comparing two benchmark results"""
    metric: ComparisonMetric
    baseline_value: float
    current_value: float
    delta: float
    percent_change: float
    change_classification: PerformanceChange
    is_significant: bool  # Based on threshold

    def get_summary(self) -> str:
        """Get human-readable summary"""
        symbol = "📈" if self.change_classification == PerformanceChange.IMPROVED else \
                 "📉" if self.change_classification == PerformanceChange.REGRESSED else "➡️"

        return (
            f"{symbol} {self.metric.value}: "
            f"{self.baseline_value:.4f} → {self.current_value:.4f} "
            f"({self.percent_change:+.2f}%)"
        )


class BenchmarkComparator:
    """
    Compare benchmark results to detect performance regressions or improvements

    Features:
    - Multi-metric comparison
    - Statistical significance testing
    - Regression detection
    - Historical trend analysis
    """

    def __init__(
        self,
        significance_threshold: float = 5.0,  # % change to be significant
        regression_threshold: float = 10.0  # % change to flag as regression
    ):
        self.significance_threshold = significance_threshold
        self.regression_threshold = regression_threshold

    def compare(
        self,
        baseline: BenchmarkResult,
        current: BenchmarkResult,
        metrics: Optional[List[ComparisonMetric]] = None
    ) -> List[ComparisonResult]:
        """
        Compare two benchmark results

        Args:
            baseline: Baseline benchmark result
            current: Current benchmark result to compare
            metrics: Metrics to compare (default: all timing metrics)

        Returns:
            List of comparison results
        """
        if metrics is None:
            metrics = [
                ComparisonMetric.MEAN_TIME,
                ComparisonMetric.MEDIAN_TIME,
                ComparisonMetric.P95_TIME,
                ComparisonMetric.P99_TIME,
            ]

        results = []
        for metric in metrics:
            result = self._compare_metric(baseline, current, metric)
            if result:
                results.append(result)

        return results

    def _compare_metric(
        self,
        baseline: BenchmarkResult,
        current: BenchmarkResult,
        metric: ComparisonMetric
    ) -> Optional[ComparisonResult]:
        """Compare a specific metric"""
        baseline_value = self._extract_metric(baseline, metric)
        current_value = self._extract_metric(current, metric)

        if baseline_value is None or current_value is None:
            return None

        # Calculate change
        delta = current_value - baseline_value

        # Avoid division by zero
        if baseline_value == 0:
            percent_change = 0.0 if delta == 0 else float('inf')
        else:
            percent_change = (delta / baseline_value) * 100

        # Classify change (lower is better for time metrics)
        is_time_metric = metric.value.endswith('_time')

        if is_time_metric:
            # For time metrics, decrease is improvement
            if abs(percent_change) < self.significance_threshold:
                classification = PerformanceChange.UNCHANGED
            elif percent_change < 0:  # Decreased time = improvement
                classification = PerformanceChange.IMPROVED
            else:  # Increased time = regression
                classification = PerformanceChange.REGRESSED
        else:
            # For other metrics, increase might be good or bad
            # For now, treat increase as regression
            if abs(percent_change) < self.significance_threshold:
                classification = PerformanceChange.UNCHANGED
            elif percent_change > 0:
                classification = PerformanceChange.REGRESSED
            else:
                classification = PerformanceChange.IMPROVED

        is_significant = abs(percent_change) >= self.significance_threshold

        return ComparisonResult(
            metric=metric,
            baseline_value=baseline_value,
            current_value=current_value,
            delta=delta,
            percent_change=percent_change,
            change_classification=classification,
            is_significant=is_significant
        )

    def _extract_metric(
        self,
        result: BenchmarkResult,
        metric: ComparisonMetric
    ) -> Optional[float]:
        """Extract metric value from benchmark result"""
        if metric == ComparisonMetric.MEAN_TIME:
            return result.mean_time
        elif metric == ComparisonMetric.MEDIAN_TIME:
            return result.median_time
        elif metric == ComparisonMetric.P95_TIME:
            return result.p95_time
        elif metric == ComparisonMetric.P99_TIME:
            return result.p99_time
        elif metric == ComparisonMetric.MAX_TIME:
            return result.max_time
        elif metric == ComparisonMetric.MEMORY_DELTA:
            if result.memory_stats and 'delta_mb' in result.memory_stats:
                return result.memory_stats['delta_mb']
        elif metric == ComparisonMetric.CPU_AVERAGE:
            if result.cpu_stats and 'average_cpu_percent' in result.cpu_stats:
                return result.cpu_stats['average_cpu_percent']

        return None

    def has_regression(self, comparisons: List[ComparisonResult]) -> bool:
        """Check if any comparisons show significant regression"""
        return any(
            c.change_classification == PerformanceChange.REGRESSED and
            abs(c.percent_change) >= self.regression_threshold
            for c in comparisons
        )

    def get_summary(self, comparisons: List[ComparisonResult]) -> str:
        """Get human-readable summary of comparisons"""
        if not comparisons:
            return "No comparison data"

        lines = ["Benchmark Comparison Summary", "=" * 50]

        for comp in comparisons:
            lines.append(comp.get_summary())

        # Overall assessment
        regressions = [c for c in comparisons if c.change_classification == PerformanceChange.REGRESSED]
        improvements = [c for c in comparisons if c.change_classification == PerformanceChange.IMPROVED]

        lines.append("")
        lines.append(f"Regressions: {len(regressions)}")
        lines.append(f"Improvements: {len(improvements)}")
        lines.append(f"Unchanged: {len(comparisons) - len(regressions) - len(improvements)}")

        if self.has_regression(comparisons):
            lines.append("")
            lines.append("⚠️  WARNING: Significant performance regression detected!")

        return "\n".join(lines)


class HistoricalComparator:
    """
    Compare benchmark results across multiple historical runs

    Provides trend analysis and statistical insights.
    """

    def __init__(self):
        self.results_history: List[BenchmarkResult] = []

    def add_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result to history"""
        self.results_history.append(result)

    def get_trend(
        self,
        metric: ComparisonMetric,
        window_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze trend for a specific metric

        Args:
            metric: Metric to analyze
            window_size: Number of recent results to consider (None = all)

        Returns:
            Trend analysis
        """
        if not self.results_history:
            return {}

        # Get relevant results
        results = self.results_history[-window_size:] if window_size else self.results_history

        # Extract values
        comparator = BenchmarkComparator()
        values = [
            comparator._extract_metric(r, metric)
            for r in results
        ]
        values = [v for v in values if v is not None]

        if not values:
            return {}

        # Calculate trend statistics
        trend = {
            'metric': metric.value,
            'samples': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
            'min': min(values),
            'max': max(values),
            'range': max(values) - min(values)
        }

        # Calculate linear trend (simple slope)
        if len(values) >= 2:
            x = list(range(len(values)))
            mean_x = statistics.mean(x)
            mean_y = statistics.mean(values)

            numerator = sum((x[i] - mean_x) * (values[i] - mean_y) for i in range(len(values)))
            denominator = sum((x[i] - mean_x) ** 2 for i in range(len(values)))

            if denominator != 0:
                slope = numerator / denominator
                trend['slope'] = slope
                trend['trend_direction'] = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'

        # Recent change
        if len(values) >= 2:
            recent_change = ((values[-1] - values[-2]) / values[-2]) * 100 if values[-2] != 0 else 0
            trend['recent_change_percent'] = recent_change

        return trend

    def detect_anomalies(
        self,
        metric: ComparisonMetric,
        std_dev_threshold: float = 2.0
    ) -> List[Tuple[int, BenchmarkResult, float]]:
        """
        Detect anomalous benchmark results

        Args:
            metric: Metric to check for anomalies
            std_dev_threshold: Number of standard deviations to consider anomalous

        Returns:
            List of (index, result, z_score) for anomalies
        """
        if len(self.results_history) < 3:
            return []

        comparator = BenchmarkComparator()
        values = [
            comparator._extract_metric(r, metric)
            for r in self.results_history
        ]
        values = [v for v in values if v is not None]

        if len(values) < 3:
            return []

        mean = statistics.mean(values)
        std_dev = statistics.stdev(values)

        if std_dev == 0:
            return []

        anomalies = []
        for i, (result, value) in enumerate(zip(self.results_history, values)):
            z_score = abs((value - mean) / std_dev)
            if z_score > std_dev_threshold:
                anomalies.append((i, result, z_score))

        return anomalies

    def clear_history(self) -> None:
        """Clear historical results"""
        self.results_history.clear()
