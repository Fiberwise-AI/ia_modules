"""
Benchmark Models

Shared dataclasses for benchmarking - separated to avoid circular imports.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution"""
    iterations: int = 100
    warmup_iterations: int = 5
    timeout: Optional[float] = None  # seconds
    profile_memory: bool = False
    profile_cpu: bool = False
    collect_intermediate: bool = False  # Collect data from each iteration

    def __post_init__(self):
        if self.iterations < 1:
            raise ValueError("iterations must be >= 1")
        if self.warmup_iterations < 0:
            raise ValueError("warmup_iterations must be >= 0")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("timeout must be > 0")


@dataclass
class BenchmarkResult:
    """Results from a benchmark run"""
    name: str
    iterations: int

    # Timing statistics (in seconds)
    mean_time: float
    median_time: float
    std_dev: float
    min_time: float
    max_time: float
    p95_time: float
    p99_time: float
    total_time: float

    # Throughput metrics
    operations_per_second: float = 0.0
    items_processed: int = 0
    items_per_second: float = 0.0

    # Cost tracking
    api_calls_count: int = 0
    estimated_cost_usd: float = 0.0
    cost_per_operation: float = 0.0

    # Resource efficiency
    memory_per_operation_mb: float = 0.0
    cpu_per_operation_percent: float = 0.0

    # Optional profiling data
    memory_stats: Optional[Dict[str, Any]] = None
    cpu_stats: Optional[Dict[str, Any]] = None

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Raw data (optional)
    raw_times: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def set_cost_tracking(self, api_calls: int = 0, cost_usd: float = 0.0) -> 'BenchmarkResult':
        """
        Set cost tracking metrics

        Args:
            api_calls: Number of API calls made
            cost_usd: Total estimated cost in USD

        Returns:
            Self for method chaining
        """
        self.api_calls_count = api_calls
        self.estimated_cost_usd = cost_usd
        if self.iterations > 0 and cost_usd > 0:
            self.cost_per_operation = cost_usd / self.iterations
        return self

    def set_throughput(self, items_processed: int) -> 'BenchmarkResult':
        """
        Set throughput metrics based on items processed

        Args:
            items_processed: Total number of items/records processed

        Returns:
            Self for method chaining
        """
        self.items_processed = items_processed
        if self.total_time > 0:
            self.items_per_second = items_processed / self.total_time
        return self

    def get_summary(self) -> str:
        """Get human-readable summary"""
        summary = (
            f"{self.name}:\n"
            f"  Iterations: {self.iterations}\n"
            f"  Mean: {self.mean_time*1000:.2f}ms\n"
            f"  Median: {self.median_time*1000:.2f}ms\n"
            f"  Std Dev: {self.std_dev*1000:.2f}ms\n"
            f"  Min: {self.min_time*1000:.2f}ms\n"
            f"  Max: {self.max_time*1000:.2f}ms\n"
            f"  P95: {self.p95_time*1000:.2f}ms\n"
            f"  P99: {self.p99_time*1000:.2f}ms\n"
            f"  Total: {self.total_time:.2f}s"
        )

        # Add throughput if available
        if self.operations_per_second > 0:
            summary += f"\n  Throughput: {self.operations_per_second:.2f} ops/sec"
        if self.items_per_second > 0:
            summary += f"\n  Items/sec: {self.items_per_second:.2f}"

        # Add cost tracking if available
        if self.api_calls_count > 0:
            summary += f"\n  API Calls: {self.api_calls_count}"
        if self.estimated_cost_usd > 0:
            summary += f"\n  Est. Cost: ${self.estimated_cost_usd:.4f}"
            if self.cost_per_operation > 0:
                summary += f" (${self.cost_per_operation:.6f}/op)"

        # Add resource efficiency
        if self.memory_per_operation_mb > 0:
            summary += f"\n  Memory/op: {self.memory_per_operation_mb:.2f}MB"
        if self.cpu_per_operation_percent > 0:
            summary += f"\n  CPU/op: {self.cpu_per_operation_percent:.2f}%"

        return summary
