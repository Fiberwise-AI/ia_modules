# IA Modules Telemetry System

Production-ready telemetry and monitoring for IA Modules pipelines.

## Overview

The telemetry system provides automatic instrumentation, metrics collection, distributed tracing, and production exporters for monitoring pipeline execution in real-time.

## Features

### 🎯 Automatic Instrumentation

- **Zero Configuration**: Telemetry is enabled by default
- **Pipeline Metrics**: Execution count, duration, success rate
- **Step Metrics**: Individual step duration and error tracking
- **Performance Metrics**: Cost, throughput, memory, CPU

### 📊 Metrics Collection

- **Counter**: Monotonically increasing values
- **Gauge**: Values that can go up or down
- **Histogram**: Observations in buckets (with percentiles)
- **Summary**: Quantile calculations (P50, P90, P95, P99)

### 🔍 Distributed Tracing

- **SimpleTracer**: In-memory for development
- **OpenTelemetryTracer**: Production-ready OTLP integration
- **Automatic Spans**: Pipelines and steps automatically traced
- **Context Propagation**: Parent-child relationships maintained

### 📤 Production Exporters

- **Prometheus**: Text format for Prometheus scraping
- **CloudWatch**: AWS CloudWatch Metrics (boto3)
- **Datadog**: Datadog API integration
- **StatsD**: UDP protocol for StatsD/Graphite

### 📈 Dashboards & Alerts

- **Grafana Dashboard**: 11 pre-configured panels
- **Prometheus Alerts**: 10 production-ready alert rules
- **Real-time Monitoring**: Track execution, cost, errors
- **Cost Tracking**: Monitor API costs per pipeline

## Quick Start

### Basic Usage (Automatic)

```python
from ia_modules.pipeline import Pipeline

# Create and run pipeline - telemetry is automatic!
pipeline = Pipeline("my_pipeline", steps, flow, services)
result = await pipeline.run(input_data)

# Metrics and traces collected automatically
```

### Export Metrics

```python
from ia_modules.telemetry import get_telemetry, PrometheusExporter

# Get telemetry instance
telemetry = get_telemetry()

# Export to Prometheus
exporter = PrometheusExporter(prefix="myapp")
exporter.export(telemetry.get_metrics())

# Print metrics
print(exporter.get_metrics_text())
```

### View Traces

```python
from ia_modules.telemetry import get_telemetry

telemetry = get_telemetry()

# View all spans
for span in telemetry.get_spans():
    print(f"{span.name}: {span.duration:.3f}s - {span.status}")
```

## Metrics Collected Automatically

### Pipeline-Level Metrics

- `pipeline_executions_total{pipeline_name, status}` - Total executions
- `pipeline_duration_seconds{pipeline_name}` - Execution duration histogram
- `active_pipelines{pipeline_name}` - Currently executing pipelines

### Step-Level Metrics

- `step_duration_seconds{pipeline_name, step_name}` - Step duration
- `step_errors_total{pipeline_name, step_name, error_type}` - Step errors

### Performance Metrics

- `items_processed_total{pipeline_name}` - Items processed
- `api_calls_total{pipeline_name}` - API calls made
- `pipeline_cost_usd_total{pipeline_name}` - Cost in USD
- `pipeline_memory_bytes{pipeline_name}` - Memory usage
- `pipeline_cpu_percent{pipeline_name}` - CPU usage

## Integration with Benchmarking

```python
from ia_modules.benchmarking import BenchmarkRunner, get_bridge

# Run benchmark
runner = BenchmarkRunner(profile_memory=True, profile_cpu=True)
result = await runner.benchmark(my_pipeline, iterations=50)

# Add cost tracking
result.set_cost_tracking(api_calls=1000, cost_usd=5.00)
result.set_throughput(items_processed=10000)

# Export to telemetry
bridge = get_bridge()
bridge.export_result("my_pipeline", result)

# Now available in all exporters!
```

## Grafana Dashboard

Import the pre-configured dashboard:

```bash
# Import into Grafana
cp ia_modules/telemetry/dashboards/grafana_pipeline_dashboard.json \
   /path/to/grafana/provisioning/dashboards/
```

**Panels included**:
1. Pipeline execution rate
2. Success rate (%)
3. Active pipelines
4. Duration (P95)
5. Step duration by name
6. Error rate by type
7. API calls per pipeline
8. Cost per hour
9. Throughput (items/sec)
10. Memory usage
11. CPU usage

## Prometheus Alerts

Configure alert rules:

```bash
# Add to Prometheus
cp ia_modules/telemetry/dashboards/alert_rules.yml \
   /etc/prometheus/rules/
```

**Alerts included**:
- High error rate (>5% for 5min)
- Pipeline duration anomaly (P95 > 300s)
- High cost (>$10/hour)
- Low throughput (<100 items/sec)
- High memory usage (>1GB)
- Pipeline not executing (20min)
- High step failure rate (>10%)
- API rate limit approaching (>1000/sec)
- High CPU usage (>80%)
- Queue depth growing

## Configuration

### Disable Telemetry

```python
# Disable for specific pipeline
pipeline = Pipeline(
    name="my_pipeline",
    steps=steps,
    flow=flow,
    services=services,
    enable_telemetry=False
)

# Disable globally
from ia_modules.telemetry import configure_telemetry
configure_telemetry(enabled=False)
```

### Custom Configuration

```python
from ia_modules.telemetry import (
    configure_telemetry,
    MetricsCollector,
    OpenTelemetryTracer
)

# Production setup
collector = MetricsCollector()
tracer = OpenTelemetryTracer(
    service_name="my-service",
    endpoint="http://jaeger:4317"
)

telemetry = configure_telemetry(
    collector=collector,
    tracer=tracer,
    enabled=True
)
```

## Documentation

- **[TELEMETRY_GUIDE.md](TELEMETRY_GUIDE.md)** - Complete API reference (500+ lines)
- **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** - Integration examples (700+ lines)
- **[Benchmark Metrics Guide](../benchmarking/METRICS_GUIDE.md)** - Benchmark integration

## Performance

- **Overhead**: <20% measured in production tests
- **Thread-Safe**: All operations use proper locking
- **Minimal Allocation**: Efficient memory usage
- **Sampling**: Built-in support for high-frequency operations

## Testing

```bash
# Run all telemetry tests
cd ia_modules
python -m pytest tests/unit/test_telemetry*.py tests/integration/test_telemetry*.py -v

# 72 tests, 100% passing
```

## Production Checklist

- [ ] Configure Prometheus scraping endpoint
- [ ] Import Grafana dashboard
- [ ] Set up alert rules in Prometheus
- [ ] Test metric export to your monitoring system
- [ ] Verify distributed traces appear correctly
- [ ] Set up alerting notification channels
- [ ] Document runbooks for each alert
- [ ] Test telemetry overhead in production load
- [ ] Configure sampling for high-frequency operations
- [ ] Set up log aggregation for trace correlation

## Examples

See complete examples in:
- [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) - Full integration examples
- [TELEMETRY_GUIDE.md](TELEMETRY_GUIDE.md) - API usage examples

## Support

For issues or questions:
1. Check the documentation guides
2. Review test examples in `tests/unit/test_telemetry*.py`
3. See integration tests in `tests/integration/test_telemetry*.py`

## License

Part of IA Modules - See main LICENSE file
