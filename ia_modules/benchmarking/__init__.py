"""
Performance Benchmarking Suite

Comprehensive benchmarking framework for pipeline performance analysis.
"""

from .models import BenchmarkResult, BenchmarkConfig
from .framework import BenchmarkRunner
from .profilers import MemoryProfiler, CPUProfiler
from .comparison import BenchmarkComparator
from .reporters import ConsoleReporter, JSONReporter, HTMLReporter


def __getattr__(name):
    """Lazy import telemetry_bridge to avoid circular imports."""
    if name in ('BenchmarkTelemetryBridge', 'get_bridge', 'configure_bridge'):
        from .telemetry_bridge import BenchmarkTelemetryBridge, get_bridge, configure_bridge
        globals()['BenchmarkTelemetryBridge'] = BenchmarkTelemetryBridge
        globals()['get_bridge'] = get_bridge
        globals()['configure_bridge'] = configure_bridge
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Core framework
    'BenchmarkRunner',
    'BenchmarkResult',
    'BenchmarkConfig',

    # Profilers
    'MemoryProfiler',
    'CPUProfiler',

    # Analysis
    'BenchmarkComparator',

    # Reporting
    'ConsoleReporter',
    'JSONReporter',
    'HTMLReporter',

    # Telemetry integration (lazy loaded)
    'BenchmarkTelemetryBridge',
    'get_bridge',
    'configure_bridge',
]
