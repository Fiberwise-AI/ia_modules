"""
Telemetry and Monitoring for IA Modules

Provides metrics collection, distributed tracing, and monitoring integrations
for production observability.
"""

from .metrics import (
    MetricsCollector,
    Metric,
    MetricType,
    Counter,
    Gauge,
    Histogram,
    Summary
)

from .exporters import (
    MetricsExporter,
    PrometheusExporter,
    CloudWatchExporter,
    DatadogExporter,
    StatsDExporter
)

from .tracing import (
    Span,
    Tracer,
    SimpleTracer,
    OpenTelemetryTracer,
    traced,
    trace_context
)

from .integration import (
    PipelineTelemetry,
    get_telemetry,
    configure_telemetry
)

__all__ = [
    # Metrics
    'MetricsCollector',
    'Metric',
    'MetricType',
    'Counter',
    'Gauge',
    'Histogram',
    'Summary',

    # Exporters
    'MetricsExporter',
    'PrometheusExporter',
    'CloudWatchExporter',
    'DatadogExporter',
    'StatsDExporter',

    # Tracing
    'Span',
    'Tracer',
    'SimpleTracer',
    'OpenTelemetryTracer',
    'traced',
    'trace_context',

    # Integration
    'PipelineTelemetry',
    'get_telemetry',
    'configure_telemetry',
]
