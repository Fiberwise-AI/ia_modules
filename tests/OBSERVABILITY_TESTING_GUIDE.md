# Observability Testing Guide

This guide explains how to use the observability stack (Prometheus, Grafana, OpenTelemetry, Jaeger) for testing and development.

## Overview

The test environment includes a complete observability stack:

- **Prometheus** - Metrics collection and storage
- **Grafana** - Metrics visualization and dashboards
- **OpenTelemetry Collector** - Metrics and traces aggregation
- **Jaeger** - Distributed tracing visualization
- **MySQL, MSSQL, PostgreSQL, Redis** - Database backends for integration testing

## Quick Start

### 1. Start the Observability Stack

```bash
# Navigate to tests directory
cd ia_modules/tests

# Start all services
docker-compose -f docker-compose.test.yml up -d

# Check service health
docker-compose -f docker-compose.test.yml ps
```

### 2. Access the Services

Once running, access the services at:

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Jaeger UI**: http://localhost:16686
- **OpenTelemetry Collector Health**: http://localhost:13133

### 3. Run Integration Tests

```bash
# Install dependencies including observability extras
pip install -e ".[all]"

# Run all database integration tests
pytest tests/integration/test_database_mysql.py -v -m mysql
pytest tests/integration/test_database_mssql.py -v -m mssql

# Run observability integration tests
pytest tests/integration/test_observability_integration.py -v -m observability
```

## Database Integration Testing

### MySQL Tests

```bash
# Set MySQL connection URL (optional, defaults to docker-compose settings)
export TEST_MYSQL_URL="mysql://testuser:testpass@localhost:3306/ia_modules_test"

# Run MySQL tests
pytest tests/integration/test_database_mysql.py -v -m mysql

# Run specific test class
pytest tests/integration/test_database_mysql.py::TestMySQLBasicOperations -v
```

**Test Coverage:**
- Basic CRUD operations
- JSON data type support
- Timestamp handling
- Transaction support (commit, rollback)
- Bulk insert performance
- UTF-8 MB4 (emoji) support
- Case-insensitive collations

### MSSQL Tests

```bash
# Set MSSQL connection URL (optional)
export TEST_MSSQL_URL="mssql://sa:TestPass123!@localhost:1433/ia_modules_test"

# Run MSSQL tests
pytest tests/integration/test_database_mssql.py -v -m mssql
```

**Test Coverage:**
- Basic CRUD operations
- NVARCHAR (Unicode) support
- DATETIME2 high-precision timestamps
- JSON validation with ISJSON()
- Binary data (VARBINARY)
- Transaction savepoints
- Stored procedures
- Computed columns
- OUTPUT clause
- Window functions (ROW_NUMBER, RANK)

### PostgreSQL Tests

PostgreSQL tests are already included in the existing test suite. Run with:

```bash
export TEST_POSTGRESQL_URL="postgresql://testuser:testpass@localhost:5434/ia_modules_test"
pytest tests/integration/ -v -m postgres
```

## Observability Integration Testing

### Prometheus Integration

```bash
# Run Prometheus integration tests
pytest tests/integration/test_observability_integration.py::TestPrometheusIntegration -v
```

**Tests:**
- Health check
- Configuration validation
- Target discovery
- Metrics export in Prometheus format
- Label handling

### Grafana Integration

```bash
# Run Grafana integration tests
pytest tests/integration/test_observability_integration.py::TestGrafanaIntegration -v
```

**Tests:**
- Health check
- Datasource configuration
- Querying Prometheus through Grafana API

### OpenTelemetry Integration

```bash
# Run OpenTelemetry Collector tests
pytest tests/integration/test_observability_integration.py::TestOpenTelemetryIntegration -v
```

**Tests:**
- Collector health check
- Collector metrics endpoint
- OTLP HTTP export
- OTLP gRPC export

### Jaeger Tracing Integration

```bash
# Run Jaeger integration tests
pytest tests/integration/test_observability_integration.py::TestJaegerIntegration -v
```

**Tests:**
- Jaeger UI accessibility
- Jaeger API
- Trace span export via OTLP

## Using the Observability Stack

### Exporting Metrics

#### Option 1: Prometheus Format

```python
from ia_modules.telemetry.metrics import MetricsCollector
from ia_modules.telemetry.exporters import PrometheusExporter

# Create collector
collector = MetricsCollector(namespace="my_app")

# Record metrics
collector.counter("requests_total", labels={"method": "GET"}).inc()
collector.gauge("active_connections").set(42)

# Export to Prometheus format
exporter = PrometheusExporter(prefix="my_app")
metrics = collector.collect()
exporter.export(metrics)

# Get Prometheus text format
print(exporter.get_metrics_text())
```

#### Option 2: OpenTelemetry (OTLP)

```python
from ia_modules.telemetry.metrics import MetricsCollector
from ia_modules.telemetry.opentelemetry_exporter import OpenTelemetryExporter

# Create collector
collector = MetricsCollector(namespace="my_app")

# Record metrics
collector.counter("operations_total").inc(10)
collector.gauge("queue_size").set(25)

# Export via OTLP to OpenTelemetry Collector
exporter = OpenTelemetryExporter(
    endpoint="http://localhost:4318",
    protocol="http",
    service_name="my_app",
    deployment_environment="test"
)

metrics = collector.collect()
exporter.export(metrics)
exporter.shutdown()
```

### Querying Metrics in Prometheus

1. Open Prometheus UI: http://localhost:9090
2. Use PromQL to query metrics:

```promql
# View all ia_modules metrics
{__name__=~"ia_modules.*"}

# Counter query
rate(ia_modules_requests_total[5m])

# Gauge query
ia_modules_active_connections

# Aggregation
sum by (method) (ia_modules_requests_total)
```

### Creating Dashboards in Grafana

1. Open Grafana: http://localhost:3000 (admin/admin)
2. Create a new dashboard
3. Add a panel
4. Select Prometheus datasource
5. Enter PromQL query
6. Configure visualization

Example queries:
- Request rate: `rate(ia_modules_requests_total[5m])`
- Active connections: `ia_modules_active_connections`
- Error rate: `rate(ia_modules_errors_total[5m])`

### Viewing Traces in Jaeger

1. Open Jaeger UI: http://localhost:16686
2. Select service: `ia_modules_test`
3. Click "Find Traces"
4. Click on a trace to view spans and timing

## Environment Variables

Configure database and observability endpoints:

```bash
# Database URLs
export TEST_POSTGRESQL_URL="postgresql://testuser:testpass@localhost:5434/ia_modules_test"
export TEST_MYSQL_URL="mysql://testuser:testpass@localhost:3306/ia_modules_test"
export TEST_MSSQL_URL="mssql://sa:TestPass123!@localhost:1433/ia_modules_test"

# Observability endpoints
export PROMETHEUS_URL="http://localhost:9090"
export GRAFANA_URL="http://localhost:3000"
export OTEL_COLLECTOR_URL="http://localhost:4318"
export JAEGER_URL="http://localhost:16686"
```

## Troubleshooting

### Services Won't Start

```bash
# Check logs
docker-compose -f docker-compose.test.yml logs

# Check specific service
docker-compose -f docker-compose.test.yml logs prometheus

# Restart services
docker-compose -f docker-compose.test.yml restart
```

### Database Connection Issues

```bash
# Check database is ready
docker-compose -f docker-compose.test.yml exec postgres pg_isready
docker-compose -f docker-compose.test.yml exec mysql mysqladmin ping
docker-compose -f docker-compose.test.yml exec mssql /opt/mssql-tools/bin/sqlcmd -S localhost -U sa -P TestPass123! -Q "SELECT 1"
```

### Metrics Not Appearing

1. Check OpenTelemetry Collector is receiving data:
   ```bash
   curl http://localhost:8888/metrics
   ```

2. Check Prometheus is scraping OTel Collector:
   - Visit http://localhost:9090/targets
   - Verify `otel-collector` target is UP

3. Check Grafana datasource connection:
   - Login to Grafana
   - Go to Configuration > Data Sources
   - Test the Prometheus datasource

### Clean Up

```bash
# Stop all services
docker-compose -f docker-compose.test.yml down

# Stop and remove volumes (clean slate)
docker-compose -f docker-compose.test.yml down -v

# Remove everything including images
docker-compose -f docker-compose.test.yml down -v --rmi all
```

## Running Specific Test Markers

```bash
# Run only database tests
pytest tests/integration/ -m "postgres or mysql or mssql" -v

# Run only observability tests
pytest tests/integration/ -m observability -v

# Run integration tests excluding slow tests
pytest tests/integration/ -m "integration and not slow" -v

# Run all integration tests
pytest tests/integration/ -m integration -v
```

## Architecture

```
┌─────────────────┐
│   Application   │
│   (ia_modules)  │
└────────┬────────┘
         │ OTLP (gRPC/HTTP)
         ▼
┌─────────────────────────────┐
│  OpenTelemetry Collector    │
│  - Receives OTLP metrics    │
│  - Receives OTLP traces     │
│  - Processes & routes data  │
└──────┬──────────────┬───────┘
       │              │
       │ Prometheus   │ OTLP
       │ Remote Write │
       ▼              ▼
┌────────────┐  ┌──────────┐
│ Prometheus │  │  Jaeger  │
│  - Stores  │  │ - Stores │
│   metrics  │  │  traces  │
└─────┬──────┘  └──────────┘
      │
      │ Prometheus API
      ▼
┌────────────┐
│  Grafana   │
│ - Queries  │
│ - Displays │
│ - Alerts   │
└────────────┘
```

## Best Practices

1. **Use Labels Wisely**: Don't create high-cardinality labels (e.g., user IDs)
2. **Namespace Metrics**: Always use a namespace prefix for your metrics
3. **Instrument Critical Paths**: Focus on business-critical operations
4. **Set Up Alerts**: Configure alerts in Grafana for important metrics
5. **Monitor Resource Usage**: Keep an eye on database and service resources
6. **Use Tracing for Debugging**: Distributed tracing helps understand request flow
7. **Clean Up Test Data**: Always clean up test tables and metrics after tests

## Further Reading

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [MySQL Testing Best Practices](https://dev.mysql.com/doc/)
- [MSSQL Testing Guide](https://docs.microsoft.com/sql/relational-databases/testing/)
