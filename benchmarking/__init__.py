"""
Performance Benchmarking Suite

Comprehensive benchmarking framework for pipeline performance analysis.
"""

from .framework import BenchmarkRunner, BenchmarkResult, BenchmarkConfig
from .profilers import MemoryProfiler, CPUProfiler
from .comparison import BenchmarkComparator
from .reporters import ConsoleReporter, JSONReporter, HTMLReporter
from .telemetry_bridge import BenchmarkTelemetryBridge, get_bridge, configure_bridge

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

    # Telemetry integration
    'BenchmarkTelemetryBridge',
    'get_bridge',
    'configure_bridge',
]
