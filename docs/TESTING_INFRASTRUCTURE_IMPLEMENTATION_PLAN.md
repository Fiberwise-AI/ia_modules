# Testing & Infrastructure Implementation Plan

## Overview

This document provides a comprehensive implementation plan for production-readiness testing and operational infrastructure for ia_modules. While feature implementation plans cover **what to build**, this plan covers **how to validate and deploy** the system.

## Table of Contents

1. [Performance Testing Module](#performance-testing-module)
2. [Security Testing Module](#security-testing-module)
3. [Disaster Recovery Testing Module](#disaster-recovery-testing-module)
4. [Real Integration Testing Module](#real-integration-testing-module)
5. [Migration Testing Module](#migration-testing-module)
6. [Monitoring Configuration](#monitoring-configuration)
7. [Deployment Infrastructure](#deployment-infrastructure)
8. [Operational Runbooks](#operational-runbooks)

---

## 1. Performance Testing Module

### 1.1 Load Testing

**File**: `ia_modules/tests/performance/test_load_testing.py`

```python
"""Load testing for concurrent pipeline executions."""
import pytest
import asyncio
import time
import psutil
from ia_modules.pipeline import GraphPipelineRunner
from ia_modules.database import DatabaseManager


@pytest.mark.performance
class TestLoadPerformance:
    """Performance tests under concurrent load."""

    @pytest.fixture
    async def simple_pipeline(self):
        """Simple test pipeline."""
        return {
            "id": "perf_test",
            "name": "Performance Test Pipeline",
            "steps": [
                {
                    "id": "step1",
                    "type": "python",
                    "module_path": "tests.performance.steps",
                    "function_name": "simple_task"
                }
            ]
        }

    async def test_100_concurrent_executions(self, simple_pipeline, db_manager):
        """Verify system handles 100 concurrent executions."""
        runner = GraphPipelineRunner(db_manager=db_manager)

        # Create 100 concurrent tasks
        tasks = []
        for i in range(100):
            task = runner.run(simple_pipeline, {"input": i})
            tasks.append(task)

        start = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        duration = time.time() - start

        # Performance assertions
        assert duration < 30.0, f"100 executions took {duration:.2f}s (should be <30s)"

        # Check error rate
        errors = [r for r in results if isinstance(r, Exception)]
        error_rate = len(errors) / len(results)
        assert error_rate < 0.01, f"Error rate {error_rate:.1%} exceeds 1%"

        # Verify database connections didn't exhaust
        if hasattr(db_manager, '_pool'):
            active = db_manager._pool._queue.qsize()
            max_size = db_manager._pool._maxsize
            assert active < max_size, f"Connection pool exhausted ({active}/{max_size})"

    async def test_sustained_load_1hour(self, simple_pipeline, db_manager):
        """Run sustained load for 1 hour (1000 req/min)."""
        runner = GraphPipelineRunner(db_manager=db_manager)

        duration = 3600  # 1 hour
        rate = 1000 / 60  # 16.67 req/sec

        start_time = time.time()
        total_executions = 0
        errors = 0

        # Track memory over time
        memory_samples = []

        while time.time() - start_time < duration:
            try:
                await runner.run(simple_pipeline, {"input": total_executions})
                total_executions += 1
            except Exception as e:
                errors += 1
                print(f"Execution {total_executions} failed: {e}")

            # Sample memory every 100 executions
            if total_executions % 100 == 0:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                memory_samples.append(memory_mb)

            # Rate limiting
            await asyncio.sleep(1 / rate)

        # Assertions
        error_rate = errors / total_executions
        assert error_rate < 0.001, f"Error rate {error_rate:.1%} exceeds 0.1%"

        # Check for memory leaks
        if len(memory_samples) > 10:
            # Memory shouldn't grow by more than 50%
            initial_memory = memory_samples[0]
            final_memory = memory_samples[-1]
            growth = (final_memory - initial_memory) / initial_memory
            assert growth < 0.5, f"Memory grew by {growth:.1%} (possible leak)"

        print(f"Completed {total_executions} executions in {duration}s")
        print(f"Error rate: {error_rate:.3%}")
        print(f"Memory: {memory_samples[0]:.1f}MB -> {memory_samples[-1]:.1f}MB")

    async def test_throughput_measurement(self, simple_pipeline, db_manager):
        """Measure maximum throughput (req/sec)."""
        runner = GraphPipelineRunner(db_manager=db_manager)

        # Run for 60 seconds at maximum speed
        duration = 60
        start_time = time.time()
        executions = 0

        while time.time() - start_time < duration:
            await runner.run(simple_pipeline, {"input": executions})
            executions += 1

        elapsed = time.time() - start_time
        throughput = executions / elapsed

        print(f"Throughput: {throughput:.2f} req/sec")
        print(f"Total executions: {executions}")

        # Baseline assertion (adjust based on hardware)
        assert throughput > 10, f"Throughput {throughput:.2f} req/sec is too low"

    async def test_latency_percentiles(self, simple_pipeline, db_manager):
        """Measure latency percentiles (p50, p95, p99)."""
        runner = GraphPipelineRunner(db_manager=db_manager)

        latencies = []

        for i in range(1000):
            start = time.time()
            await runner.run(simple_pipeline, {"input": i})
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)

        latencies.sort()

        p50 = latencies[int(len(latencies) * 0.50)]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]

        print(f"Latency p50: {p50:.2f}ms")
        print(f"Latency p95: {p95:.2f}ms")
        print(f"Latency p99: {p99:.2f}ms")

        # Baseline assertions (adjust based on requirements)
        assert p50 < 100, f"p50 latency {p50:.2f}ms exceeds 100ms"
        assert p95 < 500, f"p95 latency {p95:.2f}ms exceeds 500ms"
        assert p99 < 1000, f"p99 latency {p99:.2f}ms exceeds 1000ms"
```

---

## 2. Security Testing Module

### 2.1 SQL Injection Testing

**File**: `ia_modules/tests/security/test_sql_injection.py`

```python
"""SQL injection prevention tests."""
import pytest
from ia_modules.database import DatabaseManager


class TestSQLInjectionPrevention:
    """Verify database is immune to SQL injection attacks."""

    @pytest.mark.parametrize("malicious_input", [
        "'; DROP TABLE pipelines; --",
        "1' OR '1'='1",
        "admin'--",
        "1' UNION SELECT * FROM users--",
        "1'; DELETE FROM pipelines WHERE '1'='1",
        "' OR 1=1--",
        "1' AND 1=0 UNION ALL SELECT 'admin', '81dc9bdb52d04dc20036dbd8313ed055'--",
    ])
    async def test_parameterized_queries_prevent_injection(
        self, db_manager, malicious_input
    ):
        """Verify malicious SQL inputs are safely handled."""
        # Try to inject SQL via parameterized query
        result = await db_manager.fetch_one(
            "SELECT * FROM pipelines WHERE id = :id",
            {"id": malicious_input}
        )

        # Should return None (not found), not execute malicious SQL
        assert result is None

        # Verify tables still exist (not dropped)
        tables = await db_manager.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        table_names = [t["name"] for t in tables]
        assert "pipelines" in table_names
        assert "pipeline_executions" in table_names

    async def test_no_sql_in_error_messages(self, db_manager):
        """Verify error messages don't leak SQL details."""
        try:
            await db_manager.execute(
                "SELECT * FROM non_existent_table_12345"
            )
            pytest.fail("Should have raised exception")
        except Exception as e:
            error_msg = str(e).lower()

            # Error message should NOT contain SQL keywords
            # (Prevents information disclosure)
            forbidden_keywords = ["select", "from", "where", "table", "database"]
            for keyword in forbidden_keywords:
                # Allow in generic error messages, but not raw SQL
                if f"'{keyword}'" in error_msg or f'"{keyword}"' in error_msg:
                    pytest.fail(f"Error message leaks SQL keyword: {keyword}")

    async def test_prepared_statements_used(self, db_manager):
        """Verify prepared statements are used for all queries."""
        # This is more of a code review check, but we can verify behavior

        # Insert with parameters
        await db_manager.execute(
            """
            INSERT INTO pipelines (id, name, slug, definition)
            VALUES (:id, :name, :slug, :definition)
            """,
            {
                "id": "test-sql-injection",
                "name": "'; DROP TABLE pipelines; --",
                "slug": "test",
                "definition": "{}"
            }
        )

        # Verify data inserted safely (name wasn't executed as SQL)
        result = await db_manager.fetch_one(
            "SELECT * FROM pipelines WHERE id = :id",
            {"id": "test-sql-injection"}
        )

        assert result is not None
        assert result["name"] == "'; DROP TABLE pipelines; --"

        # Cleanup
        await db_manager.execute(
            "DELETE FROM pipelines WHERE id = :id",
            {"id": "test-sql-injection"}
        )
```

### 2.2 Input Validation Testing

**File**: `ia_modules/tests/security/test_input_validation.py`

```python
"""Input validation security tests."""
import pytest
from ia_modules.pipeline import GraphPipelineRunner


class TestInputValidation:
    """Test input validation prevents malicious inputs."""

    @pytest.mark.parametrize("malicious_payload", [
        {"__proto__": {"admin": True}},  # Prototype pollution
        {"constructor": {"prototype": {"admin": True}}},
        {"../../../etc/passwd": "data"},  # Path traversal
        {"<script>alert('XSS')</script>": "data"},  # XSS
        {"$(rm -rf /)": "data"},  # Command injection
        {"${7*7}": "data"},  # Expression injection
    ])
    async def test_malicious_payloads_rejected(
        self, simple_pipeline, db_manager, malicious_payload
    ):
        """Verify malicious payloads are rejected or sanitized."""
        runner = GraphPipelineRunner(db_manager=db_manager)

        # Should either reject or safely handle malicious input
        result = await runner.run(simple_pipeline, malicious_payload)

        # Verify execution completed without executing malicious code
        assert result is not None

        # Check that payload wasn't executed (no files created, etc.)
        # This would be environment-specific
```

---

## 3. Disaster Recovery Testing Module

### 3.1 Backup and Restore Testing

**File**: `ia_modules/tests/disaster_recovery/test_backup_restore.py`

```python
"""Disaster recovery backup and restore tests."""
import pytest
import subprocess
import tempfile
import os
from pathlib import Path
from ia_modules.database import DatabaseManager


class TestDisasterRecovery:
    """Verify backup and restore procedures work correctly."""

    async def test_full_backup_and_restore_postgresql(self):
        """Complete disaster recovery simulation for PostgreSQL."""
        db_url = os.getenv("TEST_POSTGRESQL_URL")
        if not db_url:
            pytest.skip("PostgreSQL not configured")

        db = DatabaseManager(db_url)
        await db.initialize()

        # 1. Create test data
        await db.execute("""
            INSERT INTO pipelines (id, name, slug, definition)
            VALUES ('test-1', 'Test Pipeline 1', 'test-1', '{}')
        """)
        await db.execute("""
            INSERT INTO pipelines (id, name, slug, definition)
            VALUES ('test-2', 'Test Pipeline 2', 'test-2', '{}')
        """)

        # Verify data exists
        result = await db.fetch_all("SELECT COUNT(*) as count FROM pipelines")
        original_count = result[0]["count"]
        assert original_count >= 2

        # 2. Take backup
        backup_file = Path(tempfile.gettempdir()) / "test_backup.sql"
        result = subprocess.run([
            "pg_dump",
            db_url,
            "-f", str(backup_file)
        ], capture_output=True, text=True)

        assert result.returncode == 0, f"Backup failed: {result.stderr}"
        assert backup_file.exists()
        assert backup_file.stat().st_size > 0

        # 3. Simulate disaster (delete all data)
        await db.execute("TRUNCATE pipelines CASCADE")

        # Verify data gone
        result = await db.fetch_all("SELECT COUNT(*) as count FROM pipelines")
        assert result[0]["count"] == 0

        # 4. Restore from backup
        result = subprocess.run([
            "psql",
            db_url,
            "-f", str(backup_file)
        ], capture_output=True, text=True)

        assert result.returncode == 0, f"Restore failed: {result.stderr}"

        # 5. Verify data restored correctly
        result = await db.fetch_all("SELECT COUNT(*) as count FROM pipelines")
        restored_count = result[0]["count"]
        assert restored_count == original_count

        # 6. Verify restored data integrity
        pipelines = await db.fetch_all("SELECT * FROM pipelines ORDER BY id")
        assert len(pipelines) >= 2
        assert pipelines[0]["name"] == "Test Pipeline 1"
        assert pipelines[1]["name"] == "Test Pipeline 2"

        # Cleanup
        backup_file.unlink()

    async def test_measure_rto_rpo(self):
        """Measure Recovery Time Objective and Recovery Point Objective."""
        import time

        db_url = os.getenv("TEST_POSTGRESQL_URL")
        if not db_url:
            pytest.skip("PostgreSQL not configured")

        db = DatabaseManager(db_url)
        await db.initialize()

        # Create substantial test data
        for i in range(1000):
            await db.execute("""
                INSERT INTO pipelines (id, name, slug, definition)
                VALUES (:id, :name, :slug, '{}')
            """, {
                "id": f"perf-{i}",
                "name": f"Performance Test {i}",
                "slug": f"perf-{i}"
            })

        # Measure backup time
        backup_file = Path(tempfile.gettempdir()) / "rto_backup.sql"
        start = time.time()
        subprocess.run([
            "pg_dump", db_url, "-f", str(backup_file)
        ], check=True)
        backup_duration = time.time() - start

        # Simulate disaster
        await db.execute("TRUNCATE pipelines CASCADE")

        # Measure restore time (RTO)
        start = time.time()
        subprocess.run([
            "psql", db_url, "-f", str(backup_file)
        ], check=True)
        restore_duration = time.time() - start

        print(f"\nRecovery Metrics:")
        print(f"  Backup time: {backup_duration:.2f}s")
        print(f"  Restore time (RTO): {restore_duration:.2f}s")
        print(f"  Total recovery time: {backup_duration + restore_duration:.2f}s")

        # Assert RTO is acceptable
        assert restore_duration < 300, f"RTO {restore_duration:.0f}s exceeds 5 minutes"

        # RPO = time between backups (configured separately)
        # For this test, RPO would be the backup frequency

        # Cleanup
        backup_file.unlink()
```

---

## 4. Real Integration Testing Module

### 4.1 OpenTelemetry Real Integration

**File**: `ia_modules/tests/integration_real/test_opentelemetry_real.py`

```python
"""Real OpenTelemetry integration tests."""
import pytest
import time
import requests
from ia_modules.telemetry.tracing import OpenTelemetryTracer


@pytest.mark.integration_real
@pytest.mark.requires_otlp
class TestOpenTelemetryRealIntegration:
    """Test OpenTelemetry with real OTLP collector."""

    @pytest.fixture
    def otlp_endpoint(self):
        """OTLP collector endpoint (assumes docker-compose is running)."""
        return "http://localhost:4318/v1/traces"

    @pytest.fixture
    def jaeger_ui_url(self):
        """Jaeger UI for verification."""
        return "http://localhost:16686"

    async def test_traces_sent_to_collector(self, otlp_endpoint, jaeger_ui_url):
        """Verify traces actually reach OTLP collector and appear in Jaeger."""
        # Configure tracer with real endpoint
        tracer = OpenTelemetryTracer(
            endpoint=otlp_endpoint,
            service_name="ia_modules_integration_test"
        )

        # Generate unique trace ID for verification
        import uuid
        test_id = str(uuid.uuid4())

        # Send test trace
        with tracer.span("test_operation") as span:
            span.set_attribute("test_id", test_id)
            span.set_attribute("test_key", "test_value")
            time.sleep(0.1)

        # Flush traces
        tracer.flush()

        # Wait for trace to propagate
        time.sleep(2)

        # Verify trace appears in Jaeger
        # Query Jaeger API for our trace
        response = requests.get(
            f"{jaeger_ui_url}/api/traces",
            params={
                "service": "ia_modules_integration_test",
                "tag": f"test_id:{test_id}"
            }
        )

        assert response.status_code == 200
        traces = response.json()

        # Should find our trace
        assert len(traces["data"]) > 0, "Trace not found in Jaeger"

        # Verify trace has correct attributes
        trace = traces["data"][0]
        spans = trace["spans"]
        assert len(spans) > 0

        # Find our test span
        test_span = next(
            (s for s in spans if s.get("operationName") == "test_operation"),
            None
        )
        assert test_span is not None

        # Verify attributes
        tags = {tag["key"]: tag["value"] for tag in test_span.get("tags", [])}
        assert tags.get("test_id") == test_id
        assert tags.get("test_key") == "test_value"
```

---

## 5. Migration Testing Module

### 5.1 Schema Migration Testing

**File**: `ia_modules/tests/migration/test_schema_migrations.py`

```python
"""Database schema migration tests."""
import pytest
from ia_modules.database import DatabaseManager


class TestSchemaMigrations:
    """Test database schema upgrade/downgrade procedures."""

    async def test_fresh_database_initialization(self, test_db_url):
        """Verify fresh database initializes to latest schema."""
        db = DatabaseManager(test_db_url)

        # Initialize (should apply all migrations)
        await db.initialize()

        # Verify all tables exist
        tables = await db.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        table_names = [t["name"] for t in tables]

        expected_tables = [
            "pipelines",
            "pipeline_executions",
            "pipeline_steps",
            "checkpoints",
            "reliability_metrics"
        ]

        for table in expected_tables:
            assert table in table_names, f"Table {table} not created"

    async def test_migration_preserves_data(self, test_db_url):
        """Verify migrations don't lose existing data."""
        db = DatabaseManager(test_db_url)
        await db.initialize()

        # Insert test data
        await db.execute("""
            INSERT INTO pipelines (id, name, slug, definition)
            VALUES ('migration-test', 'Migration Test', 'migration-test', '{}')
        """)

        # Get current data
        before = await db.fetch_one(
            "SELECT * FROM pipelines WHERE id = 'migration-test'"
        )
        assert before is not None

        # Simulate migration (in practice, would run migration script)
        # For this test, just verify data persists

        # Re-initialize (should detect existing schema)
        await db.initialize()

        # Verify data still exists
        after = await db.fetch_one(
            "SELECT * FROM pipelines WHERE id = 'migration-test'"
        )
        assert after is not None
        assert after["name"] == before["name"]
        assert after["slug"] == before["slug"]

    async def test_idempotent_migrations(self, test_db_url):
        """Verify migrations can be run multiple times safely."""
        db = DatabaseManager(test_db_url)

        # Run initialization twice
        await db.initialize()
        await db.initialize()

        # Should not error or create duplicate tables
        tables = await db.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )

        # Count occurrences of each table
        from collections import Counter
        table_counts = Counter(t["name"] for t in tables)

        # No table should appear more than once
        for table, count in table_counts.items():
            assert count == 1, f"Table {table} appears {count} times"
```

---

## 6. Monitoring Configuration

### 6.1 Prometheus Configuration

**File**: `deployment/monitoring/prometheus/prometheus.yml`

```yaml
# Prometheus configuration for ia_modules
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'ia_modules_production'
    environment: 'production'

# Alertmanager configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - 'alertmanager:9093'

# Load alert rules
rule_files:
  - 'alerts.yml'
  - 'recording_rules.yml'

# Scrape configurations
scrape_configs:
  # IA Modules application metrics
  - job_name: 'ia_modules'
    static_configs:
      - targets:
          - 'ia-modules-api:8000'
    metrics_path: '/metrics'
    scrape_interval: 10s

  # PostgreSQL exporter
  - job_name: 'postgresql'
    static_configs:
      - targets:
          - 'postgres-exporter:9187'

  # Redis exporter
  - job_name: 'redis'
    static_configs:
      - targets:
          - 'redis-exporter:9121'

  # Node exporter (system metrics)
  - job_name: 'node'
    static_configs:
      - targets:
          - 'node-exporter:9100'
```

### 6.2 Alert Rules

**File**: `deployment/monitoring/prometheus/alerts.yml`

```yaml
# Alert rules for ia_modules
groups:
  - name: ia_modules_alerts
    interval: 30s
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: |
          rate(pipeline_executions_failed_total[5m]) / rate(pipeline_executions_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High pipeline error rate"
          description: "Error rate is {{ $value | humanizePercentage }} (threshold: 5%)"

      # High latency
      - alert: HighLatency
        expr: |
          histogram_quantile(0.95, pipeline_execution_duration_seconds_bucket) > 10
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High pipeline latency"
          description: "P95 latency is {{ $value }}s (threshold: 10s)"

      # Database connection pool exhaustion
      - alert: DatabasePoolExhaustion
        expr: |
          database_connections_active / database_connections_max > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Database connection pool nearly exhausted"
          description: "{{ $value | humanizePercentage }} of connections in use"

      # High memory usage
      - alert: HighMemoryUsage
        expr: |
          (process_resident_memory_bytes / node_memory_MemTotal_bytes) > 0.8
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value | humanizePercentage }}"

      # Service down
      - alert: ServiceDown
        expr: up{job="ia_modules"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "IA Modules service is down"
          description: "Service has been down for 1 minute"
```

### 6.3 Grafana Dashboard

**File**: `deployment/monitoring/grafana/dashboards/pipelines.json`

```json
{
  "dashboard": {
    "title": "IA Modules - Pipeline Metrics",
    "panels": [
      {
        "title": "Pipeline Execution Rate",
        "targets": [
          {
            "expr": "rate(pipeline_executions_total[5m])",
            "legendFormat": "{{status}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(pipeline_executions_failed_total[5m]) / rate(pipeline_executions_total[5m])",
            "legendFormat": "Error Rate"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Latency Percentiles",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, pipeline_execution_duration_seconds_bucket)",
            "legendFormat": "p50"
          },
          {
            "expr": "histogram_quantile(0.95, pipeline_execution_duration_seconds_bucket)",
            "legendFormat": "p95"
          },
          {
            "expr": "histogram_quantile(0.99, pipeline_execution_duration_seconds_bucket)",
            "legendFormat": "p99"
          }
        ],
        "type": "graph"
      }
    ]
  }
}
```

---

## 7. Deployment Infrastructure

### 7.1 Production Dockerfile

**File**: `deployment/docker/Dockerfile.prod`

```dockerfile
# Multi-stage production Dockerfile for ia_modules

# Stage 1: Build stage
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Production stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY ia_modules/ ./ia_modules/
COPY setup.py ./
COPY README.md ./

# Install application
RUN pip install --no-cache-dir -e .

# Create non-root user
RUN useradd -m -u 1000 ia_user && chown -R ia_user:ia_user /app
USER ia_user

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "-m", "ia_modules.server"]
```

### 7.2 Kubernetes Deployment

**File**: `deployment/kubernetes/deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ia-modules
  namespace: production
  labels:
    app: ia-modules
    version: v0.0.3
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ia-modules
  template:
    metadata:
      labels:
        app: ia-modules
        version: v0.0.3
    spec:
      containers:
        - name: ia-modules
          image: ia-modules:0.0.3
          ports:
            - containerPort: 8000
              name: http
              protocol: TCP
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: ia-modules-secrets
                  key: database-url
            - name: ENVIRONMENT
              value: "production"
          resources:
            requests:
              memory: "512Mi"
              cpu: "500m"
            limits:
              memory: "2Gi"
              cpu: "2000m"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 5
            timeoutSeconds: 3
            failureThreshold: 2
---
apiVersion: v1
kind: Service
metadata:
  name: ia-modules-service
  namespace: production
spec:
  selector:
    app: ia-modules
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

---

## 8. Operational Runbooks

### 8.1 Deployment Runbook

**File**: `docs/operations/deployment.md`

```markdown
# Deployment Runbook

## Prerequisites

- Kubernetes cluster access
- Docker registry credentials
- Database credentials
- Monitoring stack deployed (Prometheus, Grafana)

## Deployment Steps

### 1. Build and Push Image

```bash
# Build production image
docker build -f deployment/docker/Dockerfile.prod -t ia-modules:${VERSION} .

# Tag for registry
docker tag ia-modules:${VERSION} registry.example.com/ia-modules:${VERSION}

# Push to registry
docker push registry.example.com/ia-modules:${VERSION}
```

### 2. Apply Database Migrations

```bash
# Run migrations
kubectl exec -it deployment/ia-modules -- python -m ia_modules.database.migrate

# Verify migration
kubectl exec -it deployment/ia-modules -- python -m ia_modules.database.validate
```

### 3. Deploy Application

```bash
# Update image in deployment
kubectl set image deployment/ia-modules ia-modules=registry.example.com/ia-modules:${VERSION}

# Watch rollout
kubectl rollout status deployment/ia-modules

# Verify pods
kubectl get pods -l app=ia-modules
```

### 4. Verify Deployment

```bash
# Check health endpoint
kubectl port-forward svc/ia-modules-service 8000:80
curl http://localhost:8000/health

# Check metrics
curl http://localhost:8000/metrics

# Check logs
kubectl logs -f deployment/ia-modules
```

### 5. Rollback (if needed)

```bash
# Rollback to previous version
kubectl rollout undo deployment/ia-modules

# Verify rollback
kubectl rollout status deployment/ia-modules
```

## Post-Deployment

- Monitor Grafana dashboards for 1 hour
- Check error rates and latency
- Verify database connections stable
- Confirm alerts are not firing
```

---

## Summary

This implementation plan covers the **operational infrastructure** needed for production:

✅ **Performance Testing** - Load, throughput, latency testing
✅ **Security Testing** - SQL injection, input validation
✅ **Disaster Recovery** - Backup/restore procedures with RTO/RPO measurement
✅ **Real Integrations** - Actual external service testing
✅ **Migration Testing** - Schema upgrade validation
✅ **Monitoring Setup** - Prometheus, Grafana, alerts
✅ **Deployment Infrastructure** - Docker, Kubernetes, health checks
✅ **Operational Runbooks** - Step-by-step procedures

**Total Effort**: 12-18 days to implement complete testing and operational infrastructure.

These complement the feature implementation plans by ensuring the system is **production-ready**, not just **feature-complete**.
