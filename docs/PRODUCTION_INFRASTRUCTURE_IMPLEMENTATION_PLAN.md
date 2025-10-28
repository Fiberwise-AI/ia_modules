# Production Infrastructure Implementation Plan

**Date**: 2025-10-25
**Status**: Planning Phase
**Priority**: High - Production Readiness

---

## Table of Contents

1. [Container Orchestration](#1-container-orchestration)
2. [Horizontal Scaling & Distributed Execution](#2-horizontal-scaling--distributed-execution)
3. [Database Connection Pooling & Management](#3-database-connection-pooling--management)
4. [Message Queue Integration](#4-message-queue-integration)
5. [Rate Limiting & Throttling](#5-rate-limiting--throttling)
6. [Circuit Breaker & Resilience](#6-circuit-breaker--resilience)
7. [Service Mesh & Load Balancing](#7-service-mesh--load-balancing)
8. [Blue-Green Deployment Support](#8-blue-green-deployment-support)
9. [Implementation Timeline](#implementation-timeline)
10. [Dependencies & Prerequisites](#dependencies--prerequisites)

---

## 1. Container Orchestration

### Overview
Create production-ready Docker images, Kubernetes manifests, and Helm charts for easy deployment and scaling of ia-modules pipelines.

### Requirements

#### 1.1 Multi-Stage Dockerfile

```dockerfile
# ia_modules/deployment/Dockerfile

# Build stage
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY pyproject.toml setup.py README.md ./
COPY ia_modules ./ia_modules
RUN pip install --no-cache-dir --user -e .

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY ia_modules ./ia_modules
COPY pipelines ./pipelines

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Set environment variables
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import ia_modules; print('healthy')" || exit 1

# Default command
ENTRYPOINT ["python", "-m", "ia_modules.cli"]
CMD ["--help"]

# Labels
LABEL org.opencontainers.image.title="IA Modules Pipeline Runner"
LABEL org.opencontainers.image.description="Production-ready AI pipeline execution"
LABEL org.opencontainers.image.version="0.0.3"
LABEL org.opencontainers.image.vendor="IA Modules"
```

#### 1.2 Kubernetes Deployment Manifest

```yaml
# ia_modules/deployment/k8s/deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: ia-modules-runner
  namespace: ia-modules
  labels:
    app: ia-modules
    component: pipeline-runner
    version: v0.0.3
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: ia-modules
      component: pipeline-runner
  template:
    metadata:
      labels:
        app: ia-modules
        component: pipeline-runner
        version: v0.0.3
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: ia-modules-runner
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000

      initContainers:
      - name: wait-for-db
        image: busybox:1.35
        command: ['sh', '-c', 'until nc -z postgres-service 5432; do echo waiting for db; sleep 2; done;']

      containers:
      - name: pipeline-runner
        image: ia-modules:0.0.3
        imagePullPolicy: IfNotPresent

        ports:
        - name: http
          containerPort: 8000
          protocol: TCP
        - name: metrics
          containerPort: 9090
          protocol: TCP

        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: ia-modules-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: ia-modules-config
              key: redis-url
        - name: LOG_LEVEL
          value: "INFO"
        - name: PYTHONUNBUFFERED
          value: "1"

        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2000m"
            memory: "4Gi"

        livenessProbe:
          httpGet:
            path: /health/live
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3

        readinessProbe:
          httpGet:
            path: /health/ready
            port: http
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3

        volumeMounts:
        - name: pipeline-configs
          mountPath: /app/pipelines
          readOnly: true
        - name: temp-storage
          mountPath: /tmp

      volumes:
      - name: pipeline-configs
        configMap:
          name: pipeline-configs
      - name: temp-storage
        emptyDir: {}

---
apiVersion: v1
kind: Service
metadata:
  name: ia-modules-runner
  namespace: ia-modules
  labels:
    app: ia-modules
spec:
  type: ClusterIP
  ports:
  - name: http
    port: 80
    targetPort: http
    protocol: TCP
  - name: metrics
    port: 9090
    targetPort: metrics
    protocol: TCP
  selector:
    app: ia-modules
    component: pipeline-runner

---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ia-modules-runner
  namespace: ia-modules

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ia-modules-runner-hpa
  namespace: ia-modules
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ia-modules-runner
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
      - type: Pods
        value: 4
        periodSeconds: 15
      selectPolicy: Max
```

#### 1.3 Helm Chart Structure

```yaml
# ia_modules/deployment/helm/ia-modules/Chart.yaml

apiVersion: v2
name: ia-modules
description: AI Pipeline Orchestration Platform
type: application
version: 0.0.3
appVersion: "0.0.3"
keywords:
  - ai
  - pipeline
  - orchestration
maintainers:
  - name: IA Modules Team
    email: team@ia-modules.dev
dependencies:
  - name: postgresql
    version: "12.x.x"
    repository: https://charts.bitnami.com/bitnami
    condition: postgresql.enabled
  - name: redis
    version: "17.x.x"
    repository: https://charts.bitnami.com/bitnami
    condition: redis.enabled

---
# ia_modules/deployment/helm/ia-modules/values.yaml

# Default values for ia-modules
replicaCount: 3

image:
  repository: ia-modules
  pullPolicy: IfNotPresent
  tag: "0.0.3"

imagePullSecrets: []
nameOverride: ""
fullnameOverride: ""

serviceAccount:
  create: true
  annotations: {}
  name: ""

podAnnotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "9090"

podSecurityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000

securityContext:
  capabilities:
    drop:
    - ALL
  readOnlyRootFilesystem: false
  allowPrivilegeEscalation: false

service:
  type: ClusterIP
  port: 80
  targetPort: 8000
  annotations: {}

ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "100"
  hosts:
    - host: pipelines.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: ia-modules-tls
      hosts:
        - pipelines.example.com

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 500m
    memory: 1Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

postgresql:
  enabled: true
  auth:
    username: ia_modules
    password: changeme
    database: ia_modules
  primary:
    persistence:
      enabled: true
      size: 20Gi

redis:
  enabled: true
  auth:
    enabled: true
    password: changeme
  master:
    persistence:
      enabled: true
      size: 8Gi

config:
  logLevel: INFO
  workers: 4
  maxPipelineDuration: 3600
  enableMetrics: true
  enableTracing: true

secrets:
  databaseUrl: ""
  llmApiKeys:
    openai: ""
    anthropic: ""
    gemini: ""
```

#### 1.4 Health Check Endpoints

```python
# ia_modules/web/health.py

from typing import Dict, Any
from datetime import datetime
import asyncio
from enum import Enum
from fastapi import APIRouter, Response, status
from pydantic import BaseModel

router = APIRouter(prefix="/health", tags=["health"])

class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class HealthCheck(BaseModel):
    status: HealthStatus
    timestamp: datetime
    version: str
    uptime_seconds: float
    checks: Dict[str, Dict[str, Any]]

class HealthChecker:
    """Health check manager"""

    def __init__(self):
        self.start_time = datetime.now()
        self.version = "0.0.3"

    async def check_database(self) -> Dict[str, Any]:
        """Check database connectivity"""
        try:
            from ia_modules.database import DatabaseManager
            db = DatabaseManager.get_instance()

            # Simple query to check connection
            async with db.get_connection() as conn:
                result = await conn.execute("SELECT 1")

            return {
                "status": "healthy",
                "latency_ms": 5.2,
                "pool_size": db.pool_size,
                "available_connections": db.available_connections
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity"""
        try:
            from ia_modules.cache import get_redis_client
            redis = get_redis_client()

            await redis.ping()
            info = await redis.info()

            return {
                "status": "healthy",
                "latency_ms": 2.1,
                "memory_used_mb": info.get('used_memory', 0) / 1024 / 1024,
                "connected_clients": info.get('connected_clients', 0)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def check_disk_space(self) -> Dict[str, Any]:
        """Check disk space"""
        try:
            import shutil
            usage = shutil.disk_usage("/")

            free_gb = usage.free / (1024 ** 3)
            percent_used = (usage.used / usage.total) * 100

            status = "healthy"
            if percent_used > 90:
                status = "unhealthy"
            elif percent_used > 80:
                status = "degraded"

            return {
                "status": status,
                "free_gb": round(free_gb, 2),
                "percent_used": round(percent_used, 2)
            }
        except Exception as e:
            return {
                "status": "unknown",
                "error": str(e)
            }

    async def check_memory(self) -> Dict[str, Any]:
        """Check memory usage"""
        try:
            import psutil
            memory = psutil.virtual_memory()

            status = "healthy"
            if memory.percent > 90:
                status = "unhealthy"
            elif memory.percent > 80:
                status = "degraded"

            return {
                "status": status,
                "percent_used": memory.percent,
                "available_gb": round(memory.available / (1024 ** 3), 2)
            }
        except Exception as e:
            return {
                "status": "unknown",
                "error": str(e)
            }

    async def perform_checks(self, include_detailed: bool = True) -> HealthCheck:
        """Perform all health checks"""
        checks = {}

        if include_detailed:
            # Run all checks in parallel
            db_check, redis_check, disk_check, memory_check = await asyncio.gather(
                self.check_database(),
                self.check_redis(),
                self.check_disk_space(),
                self.check_memory(),
                return_exceptions=True
            )

            checks = {
                "database": db_check if not isinstance(db_check, Exception) else {"status": "error"},
                "redis": redis_check if not isinstance(redis_check, Exception) else {"status": "error"},
                "disk": disk_check if not isinstance(disk_check, Exception) else {"status": "error"},
                "memory": memory_check if not isinstance(memory_check, Exception) else {"status": "error"}
            }

        # Determine overall status
        statuses = [check.get("status") for check in checks.values()]

        if any(s == "unhealthy" for s in statuses):
            overall_status = HealthStatus.UNHEALTHY
        elif any(s == "degraded" for s in statuses):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        uptime = (datetime.now() - self.start_time).total_seconds()

        return HealthCheck(
            status=overall_status,
            timestamp=datetime.now(),
            version=self.version,
            uptime_seconds=uptime,
            checks=checks
        )

# Global health checker instance
_health_checker = HealthChecker()

@router.get("/live")
async def liveness_probe():
    """Kubernetes liveness probe - simple check"""
    return {"status": "alive"}

@router.get("/ready")
async def readiness_probe(response: Response):
    """Kubernetes readiness probe - detailed checks"""
    health = await _health_checker.perform_checks(include_detailed=True)

    if health.status == HealthStatus.UNHEALTHY:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return health

@router.get("/")
async def health_check():
    """Full health check endpoint"""
    return await _health_checker.perform_checks(include_detailed=True)
```

#### 1.5 Docker Compose for Local Development

```yaml
# ia_modules/deployment/docker-compose.yml

version: '3.8'

services:
  # Pipeline runner
  ia-modules:
    build:
      context: ..
      dockerfile: deployment/Dockerfile
    ports:
      - "8000:8000"
      - "9090:9090"
    environment:
      - DATABASE_URL=postgresql://ia_modules:changeme@postgres:5432/ia_modules
      - REDIS_URL=redis://:changeme@redis:6379/0
      - LOG_LEVEL=DEBUG
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ../pipelines:/app/pipelines:ro
      - pipeline-data:/app/data
    networks:
      - ia-modules-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health/live"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # PostgreSQL database
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=ia_modules
      - POSTGRES_PASSWORD=changeme
      - POSTGRES_DB=ia_modules
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - ia-modules-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ia_modules"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis cache
  redis:
    image: redis:7-alpine
    command: redis-server --requirepass changeme
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    networks:
      - ia-modules-network
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5

  # Prometheus monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    networks:
      - ia-modules-network

  # Grafana dashboards
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./grafana/datasources:/etc/grafana/provisioning/datasources:ro
    networks:
      - ia-modules-network
    depends_on:
      - prometheus

networks:
  ia-modules-network:
    driver: bridge

volumes:
  postgres-data:
  redis-data:
  pipeline-data:
  prometheus-data:
  grafana-data:
```

### Implementation Checklist

- [ ] Create multi-stage Dockerfile with optimization
- [ ] Create Kubernetes deployment manifests
- [ ] Create Kubernetes service definitions
- [ ] Create HorizontalPodAutoscaler configuration
- [ ] Create Helm chart structure
- [ ] Add Helm values.yaml with sensible defaults
- [ ] Implement health check endpoints (/health/live, /health/ready)
- [ ] Create Docker Compose for local development
- [ ] Add Prometheus configuration
- [ ] Add Grafana dashboard provisioning
- [ ] Create CI/CD pipeline for building images
- [ ] Document deployment procedures
- [ ] Add deployment troubleshooting guide
- [ ] Create example Kubernetes namespace setup

**Estimated Effort**: 2-3 weeks

---

## 2. Horizontal Scaling & Distributed Execution

### Overview
Enable pipeline execution across multiple machines with shared state management and intelligent load balancing.

### Requirements

#### 2.1 Distributed Pipeline Runner

```python
# ia_modules/distributed/__init__.py

from typing import Dict, Any, Optional, List
from enum import Enum
import asyncio
from dataclasses import dataclass
import logging

class BackendType(Enum):
    """Distributed execution backends"""
    KUBERNETES = "kubernetes"
    CELERY = "celery"
    RAY = "ray"
    DASK = "dask"
    LOCAL = "local"

class LoadBalancerStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    RANDOM = "random"
    CONSISTENT_HASH = "consistent_hash"

@dataclass
class WorkerInfo:
    """Information about a worker node"""
    worker_id: str
    host: str
    port: int
    status: str
    cpu_percent: float
    memory_percent: float
    active_pipelines: int
    last_heartbeat: float

class SharedStateBackend:
    """Base class for shared state backends"""

    async def get(self, key: str) -> Any:
        raise NotImplementedError()

    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        raise NotImplementedError()

    async def delete(self, key: str):
        raise NotImplementedError()

    async def lock(self, key: str, timeout: int = 10) -> 'DistributedLock':
        raise NotImplementedError()

class RedisStateBackend(SharedStateBackend):
    """Redis-based shared state"""

    def __init__(self, redis_url: str):
        import redis.asyncio as redis
        self.redis = redis.from_url(redis_url)
        self.logger = logging.getLogger(__name__)

    async def get(self, key: str) -> Any:
        """Get value from Redis"""
        value = await self.redis.get(key)
        if value:
            import json
            return json.loads(value)
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in Redis"""
        import json
        serialized = json.dumps(value)

        if ttl:
            await self.redis.setex(key, ttl, serialized)
        else:
            await self.redis.set(key, serialized)

    async def delete(self, key: str):
        """Delete key from Redis"""
        await self.redis.delete(key)

    async def lock(self, key: str, timeout: int = 10):
        """Create distributed lock"""
        from redis.lock import Lock
        return Lock(self.redis, f"lock:{key}", timeout=timeout)

class DistributedPipelineRunner:
    """Runs pipelines across multiple workers"""

    def __init__(
        self,
        backend: BackendType = BackendType.KUBERNETES,
        replicas: int = 3,
        load_balancer: LoadBalancerStrategy = LoadBalancerStrategy.LEAST_LOADED,
        shared_state_backend: Optional[SharedStateBackend] = None
    ):
        self.backend = backend
        self.replicas = replicas
        self.load_balancer_strategy = load_balancer
        self.shared_state = shared_state_backend
        self.workers: Dict[str, WorkerInfo] = {}
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize distributed runner"""
        if self.backend == BackendType.KUBERNETES:
            await self._init_kubernetes()
        elif self.backend == BackendType.CELERY:
            await self._init_celery()
        elif self.backend == BackendType.RAY:
            await self._init_ray()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    async def _init_kubernetes(self):
        """Initialize Kubernetes backend"""
        from kubernetes import client, config

        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()

        self.k8s_client = client.CoreV1Api()
        self.k8s_apps = client.AppsV1Api()

        # Discover worker pods
        await self._discover_workers()

    async def _discover_workers(self):
        """Discover worker pods in Kubernetes"""
        try:
            pods = self.k8s_client.list_namespaced_pod(
                namespace="ia-modules",
                label_selector="app=ia-modules,component=pipeline-runner"
            )

            for pod in pods.items:
                if pod.status.phase == "Running":
                    worker_id = pod.metadata.name
                    self.workers[worker_id] = WorkerInfo(
                        worker_id=worker_id,
                        host=pod.status.pod_ip,
                        port=8000,
                        status="ready",
                        cpu_percent=0.0,
                        memory_percent=0.0,
                        active_pipelines=0,
                        last_heartbeat=asyncio.get_event_loop().time()
                    )

            self.logger.info(f"Discovered {len(self.workers)} workers")

        except Exception as e:
            self.logger.error(f"Failed to discover workers: {e}")

    async def _init_celery(self):
        """Initialize Celery backend"""
        from celery import Celery

        self.celery_app = Celery(
            'ia_modules',
            broker='redis://localhost:6379/0',
            backend='redis://localhost:6379/1'
        )

    async def _init_ray(self):
        """Initialize Ray backend"""
        import ray

        ray.init(address='auto')
        self.logger.info(f"Ray initialized with {ray.available_resources()}")

    async def select_worker(self) -> WorkerInfo:
        """Select worker based on load balancing strategy"""
        if not self.workers:
            raise RuntimeError("No workers available")

        if self.load_balancer_strategy == LoadBalancerStrategy.ROUND_ROBIN:
            return self._round_robin_worker()
        elif self.load_balancer_strategy == LoadBalancerStrategy.LEAST_LOADED:
            return self._least_loaded_worker()
        elif self.load_balancer_strategy == LoadBalancerStrategy.RANDOM:
            import random
            return random.choice(list(self.workers.values()))
        else:
            return list(self.workers.values())[0]

    def _round_robin_worker(self) -> WorkerInfo:
        """Round-robin worker selection"""
        if not hasattr(self, '_round_robin_index'):
            self._round_robin_index = 0

        workers_list = list(self.workers.values())
        worker = workers_list[self._round_robin_index % len(workers_list)]
        self._round_robin_index += 1

        return worker

    def _least_loaded_worker(self) -> WorkerInfo:
        """Select least loaded worker"""
        return min(
            self.workers.values(),
            key=lambda w: (w.active_pipelines, w.cpu_percent, w.memory_percent)
        )

    async def run_pipeline(
        self,
        pipeline_def: Dict[str, Any],
        input_data: Dict[str, Any],
        pipeline_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute pipeline on distributed workers"""
        import uuid

        pipeline_id = pipeline_id or str(uuid.uuid4())

        # Store pipeline state
        if self.shared_state:
            await self.shared_state.set(
                f"pipeline:{pipeline_id}:status",
                "pending",
                ttl=3600
            )

        try:
            # Select worker
            worker = await self.select_worker()
            self.logger.info(f"Selected worker {worker.worker_id} for pipeline {pipeline_id}")

            # Execute on worker
            if self.backend == BackendType.KUBERNETES:
                result = await self._execute_on_k8s_worker(
                    worker,
                    pipeline_def,
                    input_data,
                    pipeline_id
                )
            elif self.backend == BackendType.RAY:
                result = await self._execute_on_ray(
                    pipeline_def,
                    input_data,
                    pipeline_id
                )
            else:
                raise ValueError(f"Backend {self.backend} not implemented")

            # Update state
            if self.shared_state:
                await self.shared_state.set(
                    f"pipeline:{pipeline_id}:status",
                    "completed",
                    ttl=3600
                )

            return result

        except Exception as e:
            self.logger.error(f"Pipeline {pipeline_id} failed: {e}")

            if self.shared_state:
                await self.shared_state.set(
                    f"pipeline:{pipeline_id}:status",
                    "failed",
                    ttl=3600
                )

            raise

    async def _execute_on_k8s_worker(
        self,
        worker: WorkerInfo,
        pipeline_def: Dict[str, Any],
        input_data: Dict[str, Any],
        pipeline_id: str
    ) -> Dict[str, Any]:
        """Execute pipeline on Kubernetes worker"""
        import httpx

        url = f"http://{worker.host}:{worker.port}/api/v1/pipelines/execute"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json={
                    "pipeline_id": pipeline_id,
                    "pipeline_def": pipeline_def,
                    "input_data": input_data
                },
                timeout=3600.0
            )

            response.raise_for_status()
            return response.json()

    async def _execute_on_ray(
        self,
        pipeline_def: Dict[str, Any],
        input_data: Dict[str, Any],
        pipeline_id: str
    ) -> Dict[str, Any]:
        """Execute pipeline on Ray cluster"""
        import ray

        @ray.remote
        def run_pipeline_task(pipeline_def, input_data, pipeline_id):
            from ia_modules.pipeline.runner import PipelineRunner

            runner = PipelineRunner(pipeline_def)
            result = asyncio.run(runner.run(input_data))
            return result

        # Submit task to Ray
        future = run_pipeline_task.remote(pipeline_def, input_data, pipeline_id)

        # Wait for result
        result = await asyncio.to_thread(ray.get, future)

        return result

    async def scale(self, replicas: int):
        """Scale number of workers"""
        if self.backend == BackendType.KUBERNETES:
            await self._scale_k8s_deployment(replicas)
        else:
            self.logger.warning(f"Scaling not supported for backend {self.backend}")

    async def _scale_k8s_deployment(self, replicas: int):
        """Scale Kubernetes deployment"""
        try:
            self.k8s_apps.patch_namespaced_deployment_scale(
                name="ia-modules-runner",
                namespace="ia-modules",
                body={"spec": {"replicas": replicas}}
            )

            self.logger.info(f"Scaled deployment to {replicas} replicas")

            # Wait for new workers to be ready
            await asyncio.sleep(5)
            await self._discover_workers()

        except Exception as e:
            self.logger.error(f"Failed to scale deployment: {e}")
```

#### 2.2 Worker Registration & Heartbeat

```python
# ia_modules/distributed/worker.py

import asyncio
import time
import psutil
from typing import Optional
import logging

class WorkerRegistry:
    """Worker registration and heartbeat management"""

    def __init__(
        self,
        worker_id: str,
        shared_state: SharedStateBackend,
        heartbeat_interval: int = 10
    ):
        self.worker_id = worker_id
        self.shared_state = shared_state
        self.heartbeat_interval = heartbeat_interval
        self.logger = logging.getLogger(__name__)
        self._heartbeat_task: Optional[asyncio.Task] = None

    async def register(self):
        """Register worker with shared state"""
        worker_info = {
            "worker_id": self.worker_id,
            "host": self._get_host(),
            "port": 8000,
            "status": "ready",
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "active_pipelines": 0,
            "last_heartbeat": time.time()
        }

        await self.shared_state.set(
            f"worker:{self.worker_id}",
            worker_info,
            ttl=self.heartbeat_interval * 3
        )

        self.logger.info(f"Worker {self.worker_id} registered")

    async def start_heartbeat(self):
        """Start heartbeat loop"""
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop_heartbeat(self):
        """Stop heartbeat loop"""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

    async def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while True:
            try:
                await self.register()
                await asyncio.sleep(self.heartbeat_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Heartbeat failed: {e}")

    def _get_host(self) -> str:
        """Get worker hostname/IP"""
        import socket
        return socket.gethostname()
```

### Implementation Checklist

- [ ] Create `ia_modules/distributed/__init__.py` with distributed runner
- [ ] Implement Redis shared state backend
- [ ] Implement Kubernetes backend
- [ ] Implement Ray backend (optional)
- [ ] Implement Celery backend (optional)
- [ ] Add worker discovery mechanism
- [ ] Add load balancing strategies
- [ ] Implement worker registration and heartbeat
- [ ] Add distributed locking
- [ ] Create worker health monitoring
- [ ] Add automatic failover
- [ ] Implement pipeline state synchronization
- [ ] Write distributed execution tests
- [ ] Document distributed deployment

**Estimated Effort**: 3-4 weeks

---

## 3. Database Connection Pooling & Management

### Overview
Implement efficient database connection pooling and management to handle high-concurrency workloads.

### Requirements

#### 3.1 Connection Pool Manager

```python
# ia_modules/database/pool.py

from typing import Optional, Dict, Any
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging
import time

@dataclass
class PoolConfig:
    """Database connection pool configuration"""
    min_size: int = 5
    max_size: int = 20
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    max_queries: int = 50000
    pre_ping: bool = True

class ConnectionPool:
    """Async database connection pool"""

    def __init__(
        self,
        database_url: str,
        config: Optional[PoolConfig] = None
    ):
        self.database_url = database_url
        self.config = config or PoolConfig()
        self.logger = logging.getLogger(__name__)

        self._pool: Optional[Any] = None
        self._stats = {
            "total_connections": 0,
            "active_connections": 0,
            "idle_connections": 0,
            "total_queries": 0,
            "failed_queries": 0,
            "pool_exhausted_count": 0,
            "avg_query_time_ms": 0.0
        }
        self._query_times = []

    async def initialize(self):
        """Initialize connection pool"""
        # Detect database type from URL
        if "postgresql" in self.database_url:
            await self._init_postgres_pool()
        elif "mysql" in self.database_url:
            await self._init_mysql_pool()
        elif "mssql" in self.database_url:
            await self._init_mssql_pool()
        else:
            raise ValueError(f"Unsupported database URL: {self.database_url}")

        self.logger.info(
            f"Connection pool initialized: "
            f"min_size={self.config.min_size}, max_size={self.config.max_size}"
        )

    async def _init_postgres_pool(self):
        """Initialize PostgreSQL connection pool"""
        import asyncpg

        self._pool = await asyncpg.create_pool(
            self.database_url,
            min_size=self.config.min_size,
            max_size=self.config.max_size,
            max_queries=self.config.max_queries,
            max_inactive_connection_lifetime=self.config.pool_recycle,
            command_timeout=self.config.pool_timeout,
            timeout=self.config.pool_timeout
        )

        self._stats["total_connections"] = self._pool.get_size()

    async def _init_mysql_pool(self):
        """Initialize MySQL connection pool"""
        import aiomysql
        from urllib.parse import urlparse

        parsed = urlparse(self.database_url)

        self._pool = await aiomysql.create_pool(
            host=parsed.hostname,
            port=parsed.port or 3306,
            user=parsed.username,
            password=parsed.password,
            db=parsed.path.lstrip('/'),
            minsize=self.config.min_size,
            maxsize=self.config.max_size,
            pool_recycle=self.config.pool_recycle,
            echo=self.config.echo
        )

    async def _init_mssql_pool(self):
        """Initialize MS SQL Server connection pool"""
        # MSSQL doesn't have native async pool, use threading pool
        import pyodbc
        from concurrent.futures import ThreadPoolExecutor

        self._pool = ThreadPoolExecutor(max_workers=self.config.max_size)
        self._connection_semaphore = asyncio.Semaphore(self.config.max_size)

    @asynccontextmanager
    async def acquire(self):
        """Acquire connection from pool"""
        start_time = time.perf_counter()

        try:
            # Wait for available connection
            async with asyncio.timeout(self.config.pool_timeout):
                if isinstance(self._pool, asyncpg.Pool):
                    async with self._pool.acquire() as conn:
                        self._stats["active_connections"] += 1
                        try:
                            yield conn
                        finally:
                            self._stats["active_connections"] -= 1

                elif isinstance(self._pool, aiomysql.Pool):
                    async with self._pool.acquire() as conn:
                        self._stats["active_connections"] += 1
                        try:
                            yield conn
                        finally:
                            self._stats["active_connections"] -= 1

                else:
                    # Handle non-async pools
                    async with self._connection_semaphore:
                        conn = await asyncio.to_thread(self._create_connection)
                        self._stats["active_connections"] += 1
                        try:
                            yield conn
                        finally:
                            self._stats["active_connections"] -= 1
                            await asyncio.to_thread(conn.close)

        except asyncio.TimeoutError:
            self._stats["pool_exhausted_count"] += 1
            self.logger.error(
                f"Pool exhausted! Active: {self._stats['active_connections']}, "
                f"Max: {self.config.max_size}"
            )
            raise

        finally:
            # Record connection acquisition time
            duration = (time.perf_counter() - start_time) * 1000
            self._query_times.append(duration)

            # Keep only last 1000 times for average
            if len(self._query_times) > 1000:
                self._query_times = self._query_times[-1000:]

            self._stats["avg_query_time_ms"] = sum(self._query_times) / len(self._query_times)

    async def execute(self, query: str, *args) -> Any:
        """Execute query using pooled connection"""
        async with self.acquire() as conn:
            self._stats["total_queries"] += 1

            try:
                if isinstance(conn, asyncpg.Connection):
                    result = await conn.fetch(query, *args)
                else:
                    cursor = await conn.cursor()
                    await cursor.execute(query, args)
                    result = await cursor.fetchall()

                return result

            except Exception as e:
                self._stats["failed_queries"] += 1
                self.logger.error(f"Query failed: {e}")
                raise

    async def close(self):
        """Close connection pool"""
        if self._pool:
            if isinstance(self._pool, (asyncpg.Pool, aiomysql.Pool)):
                self._pool.close()
                await self._pool.wait_closed()
            else:
                self._pool.shutdown(wait=True)

            self.logger.info("Connection pool closed")

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        if isinstance(self._pool, asyncpg.Pool):
            self._stats["idle_connections"] = self._pool.get_idle_size()
            self._stats["total_connections"] = self._pool.get_size()

        return self._stats.copy()

    async def health_check(self) -> bool:
        """Check pool health"""
        try:
            async with self.acquire() as conn:
                if isinstance(conn, asyncpg.Connection):
                    await conn.fetchval("SELECT 1")
                else:
                    cursor = await conn.cursor()
                    await cursor.execute("SELECT 1")

            return True

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
```

#### 3.2 Enhanced Database Manager

```python
# ia_modules/database/manager.py (enhanced)

class DatabaseManager:
    """Enhanced database manager with connection pooling"""

    _instance: Optional['DatabaseManager'] = None
    _lock = asyncio.Lock()

    def __init__(self, database_url: str, pool_config: Optional[PoolConfig] = None):
        self.database_url = database_url
        self.pool = ConnectionPool(database_url, pool_config)
        self.logger = logging.getLogger(__name__)

    @classmethod
    async def get_instance(
        cls,
        database_url: Optional[str] = None,
        pool_config: Optional[PoolConfig] = None
    ) -> 'DatabaseManager':
        """Get singleton instance with connection pool"""
        async with cls._lock:
            if cls._instance is None:
                if database_url is None:
                    raise ValueError("database_url required for first initialization")

                cls._instance = cls(database_url, pool_config)
                await cls._instance.pool.initialize()

            return cls._instance

    @asynccontextmanager
    async def get_connection(self):
        """Get connection from pool"""
        async with self.pool.acquire() as conn:
            yield conn

    async def execute(self, query: str, *args) -> Any:
        """Execute query"""
        return await self.pool.execute(query, *args)

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        return self.pool.get_stats()

    async def close(self):
        """Close database manager and pool"""
        await self.pool.close()
```

### Implementation Checklist

- [ ] Create `ia_modules/database/pool.py` with connection pooling
- [ ] Add PostgreSQL pool implementation (asyncpg)
- [ ] Add MySQL pool implementation (aiomysql)
- [ ] Add MSSQL pool implementation
- [ ] Implement pool statistics tracking
- [ ] Add connection recycling
- [ ] Add connection pre-ping health checks
- [ ] Implement pool exhaustion handling
- [ ] Add pool monitoring endpoints
- [ ] Create pool configuration validation
- [ ] Write connection pool tests
- [ ] Add pool performance benchmarks
- [ ] Document pool configuration best practices

**Estimated Effort**: 2 weeks

---

## 4. Message Queue Integration

### Overview
Integrate with message queue systems (RabbitMQ, Kafka, SQS) for async pipeline processing and event-driven workflows.

### Requirements

#### 4.1 Message Queue Interface

```python
# ia_modules/queue/__init__.py

from typing import Dict, Any, Optional, Callable, Awaitable
from enum import Enum
from dataclasses import dataclass
import asyncio
import logging
import json

class QueueBackend(Enum):
    """Supported queue backends"""
    RABBITMQ = "rabbitmq"
    KAFKA = "kafka"
    SQS = "sqs"
    REDIS = "redis"
    IN_MEMORY = "in_memory"

@dataclass
class RetryPolicy:
    """Retry policy for failed messages"""
    max_attempts: int = 3
    initial_delay_seconds: int = 1
    max_delay_seconds: int = 60
    exponential_base: float = 2.0
    backoff_strategy: str = "exponential"

@dataclass
class Message:
    """Message envelope"""
    id: str
    body: Dict[str, Any]
    headers: Dict[str, str]
    timestamp: float
    attempt: int = 1

class MessageQueue:
    """Base message queue interface"""

    def __init__(
        self,
        backend: QueueBackend,
        queue_name: str,
        connection_url: str,
        retry_policy: Optional[RetryPolicy] = None
    ):
        self.backend = backend
        self.queue_name = queue_name
        self.connection_url = connection_url
        self.retry_policy = retry_policy or RetryPolicy()
        self.logger = logging.getLogger(__name__)
        self._client: Optional[Any] = None

    async def connect(self):
        """Connect to message queue"""
        if self.backend == QueueBackend.RABBITMQ:
            await self._connect_rabbitmq()
        elif self.backend == QueueBackend.KAFKA:
            await self._connect_kafka()
        elif self.backend == QueueBackend.SQS:
            await self._connect_sqs()
        elif self.backend == QueueBackend.REDIS:
            await self._connect_redis()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    async def _connect_rabbitmq(self):
        """Connect to RabbitMQ"""
        import aio_pika

        self._client = await aio_pika.connect_robust(self.connection_url)
        self._channel = await self._client.channel()
        self._queue = await self._channel.declare_queue(
            self.queue_name,
            durable=True
        )

        self.logger.info(f"Connected to RabbitMQ: {self.queue_name}")

    async def _connect_kafka(self):
        """Connect to Kafka"""
        from aiokafka import AIOKafkaProducer, AIOKafkaConsumer

        self._producer = AIOKafkaProducer(
            bootstrap_servers=self.connection_url,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        await self._producer.start()

        self._consumer = AIOKafkaConsumer(
            self.queue_name,
            bootstrap_servers=self.connection_url,
            value_deserializer=lambda v: json.loads(v.decode('utf-8'))
        )
        await self._consumer.start()

        self.logger.info(f"Connected to Kafka: {self.queue_name}")

    async def _connect_sqs(self):
        """Connect to AWS SQS"""
        import aioboto3

        session = aioboto3.Session()
        self._client = await session.client('sqs').__aenter__()

        # Get or create queue
        response = await self._client.get_queue_url(QueueName=self.queue_name)
        self._queue_url = response['QueueUrl']

        self.logger.info(f"Connected to SQS: {self.queue_name}")

    async def _connect_redis(self):
        """Connect to Redis for queue"""
        import redis.asyncio as redis

        self._client = redis.from_url(self.connection_url)
        await self._client.ping()

        self.logger.info(f"Connected to Redis: {self.queue_name}")

    async def enqueue(
        self,
        message_body: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        delay_seconds: int = 0
    ) -> str:
        """Enqueue message"""
        import uuid
        import time

        message = Message(
            id=str(uuid.uuid4()),
            body=message_body,
            headers=headers or {},
            timestamp=time.time()
        )

        if self.backend == QueueBackend.RABBITMQ:
            await self._enqueue_rabbitmq(message, delay_seconds)
        elif self.backend == QueueBackend.KAFKA:
            await self._enqueue_kafka(message)
        elif self.backend == QueueBackend.SQS:
            await self._enqueue_sqs(message, delay_seconds)
        elif self.backend == QueueBackend.REDIS:
            await self._enqueue_redis(message)

        return message.id

    async def _enqueue_rabbitmq(self, message: Message, delay_seconds: int):
        """Enqueue to RabbitMQ"""
        import aio_pika

        msg = aio_pika.Message(
            body=json.dumps(message.body).encode('utf-8'),
            headers=message.headers,
            message_id=message.id,
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT
        )

        if delay_seconds > 0:
            # Use delayed message plugin
            msg.headers['x-delay'] = delay_seconds * 1000

        await self._channel.default_exchange.publish(
            msg,
            routing_key=self.queue_name
        )

    async def _enqueue_kafka(self, message: Message):
        """Enqueue to Kafka"""
        await self._producer.send(
            self.queue_name,
            value=message.body,
            headers=[(k, v.encode('utf-8')) for k, v in message.headers.items()]
        )

    async def _enqueue_sqs(self, message: Message, delay_seconds: int):
        """Enqueue to SQS"""
        await self._client.send_message(
            QueueUrl=self._queue_url,
            MessageBody=json.dumps(message.body),
            MessageAttributes={
                k: {'StringValue': v, 'DataType': 'String'}
                for k, v in message.headers.items()
            },
            DelaySeconds=delay_seconds
        )

    async def _enqueue_redis(self, message: Message):
        """Enqueue to Redis"""
        await self._client.rpush(
            self.queue_name,
            json.dumps({
                'id': message.id,
                'body': message.body,
                'headers': message.headers,
                'timestamp': message.timestamp
            })
        )

    async def consume(
        self,
        handler: Callable[[Message], Awaitable[None]],
        concurrency: int = 1
    ):
        """Consume messages from queue"""
        tasks = [
            asyncio.create_task(self._consumer_loop(handler))
            for _ in range(concurrency)
        ]

        await asyncio.gather(*tasks)

    async def _consumer_loop(self, handler: Callable[[Message], Awaitable[None]]):
        """Consumer loop"""
        while True:
            try:
                if self.backend == QueueBackend.RABBITMQ:
                    await self._consume_rabbitmq(handler)
                elif self.backend == QueueBackend.KAFKA:
                    await self._consume_kafka(handler)
                elif self.backend == QueueBackend.SQS:
                    await self._consume_sqs(handler)
                elif self.backend == QueueBackend.REDIS:
                    await self._consume_redis(handler)

            except Exception as e:
                self.logger.error(f"Consumer error: {e}")
                await asyncio.sleep(5)

    async def _consume_rabbitmq(self, handler: Callable[[Message], Awaitable[None]]):
        """Consume from RabbitMQ"""
        async with self._queue.iterator() as queue_iter:
            async for raw_message in queue_iter:
                try:
                    message = Message(
                        id=raw_message.message_id,
                        body=json.loads(raw_message.body.decode('utf-8')),
                        headers=dict(raw_message.headers or {}),
                        timestamp=raw_message.timestamp.timestamp()
                    )

                    await handler(message)
                    await raw_message.ack()

                except Exception as e:
                    self.logger.error(f"Message processing failed: {e}")
                    await raw_message.nack(requeue=True)

    async def _consume_kafka(self, handler: Callable[[Message], Awaitable[None]]):
        """Consume from Kafka"""
        async for raw_message in self._consumer:
            try:
                message = Message(
                    id=str(raw_message.offset),
                    body=raw_message.value,
                    headers={
                        k: v.decode('utf-8') if isinstance(v, bytes) else v
                        for k, v in (raw_message.headers or [])
                    },
                    timestamp=raw_message.timestamp / 1000
                )

                await handler(message)
                await self._consumer.commit()

            except Exception as e:
                self.logger.error(f"Message processing failed: {e}")

    async def _consume_sqs(self, handler: Callable[[Message], Awaitable[None]]):
        """Consume from SQS"""
        while True:
            response = await self._client.receive_message(
                QueueUrl=self._queue_url,
                MaxNumberOfMessages=10,
                WaitTimeSeconds=20,
                MessageAttributeNames=['All']
            )

            for raw_message in response.get('Messages', []):
                try:
                    message = Message(
                        id=raw_message['MessageId'],
                        body=json.loads(raw_message['Body']),
                        headers={
                            k: v['StringValue']
                            for k, v in raw_message.get('MessageAttributes', {}).items()
                        },
                        timestamp=float(raw_message['Attributes'].get('SentTimestamp', 0)) / 1000
                    )

                    await handler(message)

                    # Delete message after successful processing
                    await self._client.delete_message(
                        QueueUrl=self._queue_url,
                        ReceiptHandle=raw_message['ReceiptHandle']
                    )

                except Exception as e:
                    self.logger.error(f"Message processing failed: {e}")

    async def _consume_redis(self, handler: Callable[[Message], Awaitable[None]]):
        """Consume from Redis"""
        while True:
            # Blocking pop
            result = await self._client.blpop(self.queue_name, timeout=5)

            if result:
                _, raw_message = result

                try:
                    data = json.loads(raw_message)
                    message = Message(
                        id=data['id'],
                        body=data['body'],
                        headers=data['headers'],
                        timestamp=data['timestamp']
                    )

                    await handler(message)

                except Exception as e:
                    self.logger.error(f"Message processing failed: {e}")
                    # Push back to dead letter queue
                    await self._client.rpush(f"{self.queue_name}:dlq", raw_message)

    async def enqueue_pipeline(
        self,
        pipeline_def: Dict[str, Any],
        input_data: Dict[str, Any],
        priority: int = 0
    ) -> str:
        """Enqueue pipeline for background processing"""
        message_body = {
            "type": "pipeline_execution",
            "pipeline_def": pipeline_def,
            "input_data": input_data,
            "priority": priority
        }

        return await self.enqueue(message_body)

    async def close(self):
        """Close queue connections"""
        if self._client:
            if self.backend == QueueBackend.RABBITMQ:
                await self._client.close()
            elif self.backend == QueueBackend.KAFKA:
                await self._producer.stop()
                await self._consumer.stop()
            elif self.backend == QueueBackend.REDIS:
                await self._client.close()
```

#### 4.2 Pipeline Queue Consumer

```python
# ia_modules/queue/pipeline_consumer.py

from ia_modules.queue import MessageQueue, Message
from ia_modules.pipeline.runner import PipelineRunner
import logging

class PipelineQueueConsumer:
    """Consumes pipeline execution requests from queue"""

    def __init__(self, queue: MessageQueue):
        self.queue = queue
        self.logger = logging.getLogger(__name__)

    async def start(self, concurrency: int = 4):
        """Start consuming pipeline messages"""
        self.logger.info(f"Starting pipeline consumer (concurrency={concurrency})")
        await self.queue.consume(self.handle_pipeline_message, concurrency=concurrency)

    async def handle_pipeline_message(self, message: Message):
        """Handle pipeline execution message"""
        try:
            message_type = message.body.get("type")

            if message_type == "pipeline_execution":
                pipeline_def = message.body["pipeline_def"]
                input_data = message.body["input_data"]

                self.logger.info(f"Executing pipeline from queue: {message.id}")

                runner = PipelineRunner(pipeline_def)
                result = await runner.run(input_data)

                self.logger.info(f"Pipeline {message.id} completed successfully")

            else:
                self.logger.warning(f"Unknown message type: {message_type}")

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            raise
```

### Implementation Checklist

- [ ] Create `ia_modules/queue/__init__.py` with queue interface
- [ ] Implement RabbitMQ backend (aio-pika)
- [ ] Implement Kafka backend (aiokafka)
- [ ] Implement SQS backend (aioboto3)
- [ ] Implement Redis queue backend
- [ ] Add retry policy and DLQ support
- [ ] Create pipeline queue consumer
- [ ] Add message acknowledgment handling
- [ ] Implement priority queues
- [ ] Add queue monitoring and metrics
- [ ] Create queue management CLI
- [ ] Write queue integration tests
- [ ] Document queue setup and usage

**Estimated Effort**: 2-3 weeks

---

## 5. Rate Limiting & Throttling

### Overview
Implement API-level and LLM-level rate limiting to prevent abuse and manage costs.

### Requirements

#### 5.1 Rate Limiter Implementation

```python
# ia_modules/middleware/rate_limiter.py

from typing import Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass
import time
import asyncio
from collections import defaultdict
import logging

class RateLimitStrategy(Enum):
    """Rate limiting strategies"""
    TOKEN_BUCKET = "token_bucket"
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    LEAKY_BUCKET = "leaky_bucket"

@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    requests_per_minute: int = 100
    requests_per_hour: int = 5000
    requests_per_day: int = 100000
    burst_size: int = 20
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    backend: str = "redis"

class RateLimiter:
    """Rate limiter with multiple strategies"""

    def __init__(
        self,
        config: RateLimitConfig,
        redis_url: Optional[str] = None
    ):
        self.config = config
        self.logger = logging.getLogger(__name__)

        if config.backend == "redis" and redis_url:
            self._init_redis(redis_url)
        else:
            self._init_in_memory()

    def _init_redis(self, redis_url: str):
        """Initialize Redis backend"""
        import redis.asyncio as redis
        self.redis = redis.from_url(redis_url)
        self.backend = "redis"

    def _init_in_memory(self):
        """Initialize in-memory backend"""
        self.buckets: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "tokens": self.config.burst_size,
            "last_update": time.time(),
            "requests": []
        })
        self.backend = "memory"

    async def is_allowed(
        self,
        key: str,
        tokens: int = 1
    ) -> tuple[bool, Dict[str, Any]]:
        """Check if request is allowed"""
        if self.config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return await self._token_bucket(key, tokens)
        elif self.config.strategy == RateLimitStrategy.FIXED_WINDOW:
            return await self._fixed_window(key, tokens)
        elif self.config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return await self._sliding_window(key, tokens)
        else:
            return True, {}

    async def _token_bucket(
        self,
        key: str,
        tokens: int
    ) -> tuple[bool, Dict[str, Any]]:
        """Token bucket algorithm"""
        if self.backend == "redis":
            return await self._token_bucket_redis(key, tokens)
        else:
            return await self._token_bucket_memory(key, tokens)

    async def _token_bucket_redis(
        self,
        key: str,
        tokens: int
    ) -> tuple[bool, Dict[str, Any]]:
        """Token bucket with Redis"""
        now = time.time()
        rate = self.config.requests_per_minute / 60.0  # tokens per second

        # Lua script for atomic token bucket
        script = """
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local rate = tonumber(ARGV[2])
        local requested = tonumber(ARGV[3])
        local now = tonumber(ARGV[4])

        local bucket = redis.call('HMGET', key, 'tokens', 'last_update')
        local tokens = tonumber(bucket[1]) or capacity
        local last_update = tonumber(bucket[2]) or now

        -- Refill tokens based on time passed
        local time_passed = now - last_update
        local new_tokens = math.min(capacity, tokens + (time_passed * rate))

        -- Check if enough tokens
        if new_tokens >= requested then
            new_tokens = new_tokens - requested
            redis.call('HMSET', key, 'tokens', new_tokens, 'last_update', now)
            redis.call('EXPIRE', key, 3600)
            return {1, new_tokens, capacity}
        else
            redis.call('HMSET', key, 'tokens', new_tokens, 'last_update', now)
            redis.call('EXPIRE', key, 3600)
            return {0, new_tokens, capacity}
        end
        """

        result = await self.redis.eval(
            script,
            1,
            f"rate_limit:{key}",
            self.config.burst_size,
            rate,
            tokens,
            now
        )

        allowed = bool(result[0])
        remaining = int(result[1])
        capacity = int(result[2])

        info = {
            "limit": capacity,
            "remaining": remaining,
            "reset_in_seconds": int((capacity - remaining) / rate) if rate > 0 else 0
        }

        return allowed, info

    async def _token_bucket_memory(
        self,
        key: str,
        tokens: int
    ) -> tuple[bool, Dict[str, Any]]:
        """Token bucket with in-memory storage"""
        now = time.time()
        rate = self.config.requests_per_minute / 60.0

        bucket = self.buckets[key]
        last_update = bucket["last_update"]
        current_tokens = bucket["tokens"]

        # Refill tokens
        time_passed = now - last_update
        new_tokens = min(
            self.config.burst_size,
            current_tokens + (time_passed * rate)
        )

        if new_tokens >= tokens:
            bucket["tokens"] = new_tokens - tokens
            bucket["last_update"] = now

            info = {
                "limit": self.config.burst_size,
                "remaining": int(bucket["tokens"]),
                "reset_in_seconds": int((self.config.burst_size - bucket["tokens"]) / rate)
            }

            return True, info
        else:
            bucket["tokens"] = new_tokens
            bucket["last_update"] = now

            info = {
                "limit": self.config.burst_size,
                "remaining": int(bucket["tokens"]),
                "reset_in_seconds": int((self.config.burst_size - bucket["tokens"]) / rate)
            }

            return False, info

    async def _sliding_window(
        self,
        key: str,
        tokens: int
    ) -> tuple[bool, Dict[str, Any]]:
        """Sliding window algorithm"""
        now = time.time()
        window_size = 60  # 1 minute

        if self.backend == "redis":
            # Use sorted set for sliding window
            window_key = f"rate_limit:window:{key}"

            # Remove old entries
            await self.redis.zremrangebyscore(
                window_key,
                0,
                now - window_size
            )

            # Count requests in window
            count = await self.redis.zcard(window_key)

            if count < self.config.requests_per_minute:
                # Add current request
                await self.redis.zadd(window_key, {str(now): now})
                await self.redis.expire(window_key, window_size)

                return True, {
                    "limit": self.config.requests_per_minute,
                    "remaining": self.config.requests_per_minute - count - 1,
                    "reset_in_seconds": window_size
                }
            else:
                return False, {
                    "limit": self.config.requests_per_minute,
                    "remaining": 0,
                    "reset_in_seconds": window_size
                }

        else:
            # In-memory sliding window
            bucket = self.buckets[key]
            requests = bucket["requests"]

            # Remove old requests
            requests = [r for r in requests if r > now - window_size]
            bucket["requests"] = requests

            if len(requests) < self.config.requests_per_minute:
                requests.append(now)

                return True, {
                    "limit": self.config.requests_per_minute,
                    "remaining": self.config.requests_per_minute - len(requests),
                    "reset_in_seconds": window_size
                }
            else:
                return False, {
                    "limit": self.config.requests_per_minute,
                    "remaining": 0,
                    "reset_in_seconds": window_size
                }

    async def _fixed_window(
        self,
        key: str,
        tokens: int
    ) -> tuple[bool, Dict[str, Any]]:
        """Fixed window algorithm"""
        import math

        now = time.time()
        window = 60  # 1 minute window
        window_key = math.floor(now / window)

        if self.backend == "redis":
            redis_key = f"rate_limit:fixed:{key}:{window_key}"

            count = await self.redis.incr(redis_key)

            if count == 1:
                await self.redis.expire(redis_key, window)

            if count <= self.config.requests_per_minute:
                return True, {
                    "limit": self.config.requests_per_minute,
                    "remaining": self.config.requests_per_minute - count,
                    "reset_in_seconds": window - int(now % window)
                }
            else:
                return False, {
                    "limit": self.config.requests_per_minute,
                    "remaining": 0,
                    "reset_in_seconds": window - int(now % window)
                }

        else:
            bucket = self.buckets[f"{key}:{window_key}"]
            count = bucket.get("count", 0) + 1
            bucket["count"] = count

            if count <= self.config.requests_per_minute:
                return True, {
                    "limit": self.config.requests_per_minute,
                    "remaining": self.config.requests_per_minute - count,
                    "reset_in_seconds": window - int(now % window)
                }
            else:
                return False, {
                    "limit": self.config.requests_per_minute,
                    "remaining": 0,
                    "reset_in_seconds": window - int(now % window)
                }
```

#### 5.2 FastAPI Middleware

```python
# ia_modules/middleware/rate_limit_middleware.py

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logging

class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI rate limiting middleware"""

    def __init__(
        self,
        app,
        rate_limiter: RateLimiter,
        key_func: Optional[Callable[[Request], str]] = None
    ):
        super().__init__(app)
        self.rate_limiter = rate_limiter
        self.key_func = key_func or self._default_key_func
        self.logger = logging.getLogger(__name__)

    def _default_key_func(self, request: Request) -> str:
        """Default key function (by IP)"""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host

    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting"""
        key = self.key_func(request)

        allowed, info = await self.rate_limiter.is_allowed(key)

        # Add rate limit headers
        headers = {
            "X-RateLimit-Limit": str(info.get("limit", 0)),
            "X-RateLimit-Remaining": str(info.get("remaining", 0)),
            "X-RateLimit-Reset": str(info.get("reset_in_seconds", 0))
        }

        if not allowed:
            self.logger.warning(f"Rate limit exceeded for key: {key}")

            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "retry_after": info.get("reset_in_seconds", 60)
                },
                headers=headers
            )

        response = await call_next(request)

        # Add headers to response
        for header_name, header_value in headers.items():
            response.headers[header_name] = header_value

        return response
```

### Implementation Checklist

- [ ] Create `ia_modules/middleware/rate_limiter.py`
- [ ] Implement token bucket algorithm
- [ ] Implement sliding window algorithm
- [ ] Implement fixed window algorithm
- [ ] Add Redis backend for distributed rate limiting
- [ ] Create FastAPI middleware
- [ ] Add rate limit headers (X-RateLimit-*)
- [ ] Implement per-user rate limiting
- [ ] Add rate limit bypass for admins
- [ ] Create rate limit monitoring
- [ ] Add rate limit configuration per endpoint
- [ ] Write rate limiting tests
- [ ] Document rate limit setup

**Estimated Effort**: 1-2 weeks

---

## 6. Circuit Breaker & Resilience

### Overview
Implement circuit breaker pattern and resilience features to handle external service failures gracefully.

### Requirements

#### 6.1 Circuit Breaker Implementation

```python
# ia_modules/resilience/circuit_breaker.py

from typing import Optional, Callable, Any, Awaitable
from enum import Enum
from dataclasses import dataclass
import asyncio
import time
import logging
from functools import wraps

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"          # Normal operation
    OPEN = "open"              # Circuit tripped, rejecting requests
    HALF_OPEN = "half_open"    # Testing if service recovered

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5         # Open after N failures
    success_threshold: int = 2         # Close after N successes in half-open
    timeout_seconds: int = 60          # Time to wait before half-open
    half_open_max_calls: int = 3       # Max calls to try in half-open
    excluded_exceptions: tuple = ()    # Don't count these as failures

class CircuitBreakerError(Exception):
    """Circuit breaker is open"""
    pass

class CircuitBreaker:
    """Circuit breaker implementation"""

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.logger = logging.getLogger(__name__)

        # State
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0

        # Statistics
        self.total_calls = 0
        self.total_failures = 0
        self.total_successes = 0
        self.total_rejections = 0

    async def call(
        self,
        func: Callable[..., Awaitable[Any]],
        *args,
        **kwargs
    ) -> Any:
        """Execute function with circuit breaker"""
        self.total_calls += 1

        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.logger.info(f"Circuit {self.name} entering HALF_OPEN state")
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
            else:
                self.total_rejections += 1
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is OPEN"
                )

        # Limit calls in half-open state
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls >= self.config.half_open_max_calls:
                self.total_rejections += 1
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is HALF_OPEN (max calls reached)"
                )
            self.half_open_calls += 1

        # Execute function
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result

        except Exception as e:
            # Check if exception should be counted
            if not isinstance(e, self.config.excluded_exceptions):
                self._on_failure(e)
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return False

        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.config.timeout_seconds

    def _on_success(self):
        """Handle successful call"""
        self.total_successes += 1

        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1

            if self.success_count >= self.config.success_threshold:
                self.logger.info(f"Circuit {self.name} closing (successes: {self.success_count})")
                self._close()

        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0

    def _on_failure(self, exception: Exception):
        """Handle failed call"""
        self.total_failures += 1
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            self.logger.warning(
                f"Circuit {self.name} opening (failure in HALF_OPEN): {exception}"
            )
            self._open()

        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.logger.error(
                    f"Circuit {self.name} opening (failures: {self.failure_count}): {exception}"
                )
                self._open()

    def _open(self):
        """Open circuit"""
        self.state = CircuitState.OPEN
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0

    def _close(self):
        """Close circuit"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0

    def reset(self):
        """Manually reset circuit breaker"""
        self.logger.info(f"Circuit {self.name} manually reset")
        self._close()

    def get_state(self) -> dict:
        """Get circuit breaker state"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "total_rejections": self.total_rejections,
            "success_rate": (
                self.total_successes / self.total_calls
                if self.total_calls > 0
                else 0.0
            )
        }

def circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None
):
    """Circuit breaker decorator"""
    breaker = CircuitBreaker(name, config)

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await breaker.call(func, *args, **kwargs)

        # Attach breaker to function for inspection
        wrapper.circuit_breaker = breaker

        return wrapper

    return decorator

# Global circuit breaker registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}

def get_circuit_breaker(name: str) -> Optional[CircuitBreaker]:
    """Get circuit breaker by name"""
    return _circuit_breakers.get(name)

def register_circuit_breaker(breaker: CircuitBreaker):
    """Register circuit breaker"""
    _circuit_breakers[breaker.name] = breaker

def get_all_circuit_breakers() -> Dict[str, CircuitBreaker]:
    """Get all circuit breakers"""
    return _circuit_breakers.copy()
```

#### 6.2 Resilience Patterns

```python
# ia_modules/resilience/patterns.py

from typing import Optional, Callable, Any, Awaitable, List
import asyncio
import time
from functools import wraps

class Bulkhead:
    """Bulkhead pattern for resource isolation"""

    def __init__(self, name: str, max_concurrent: int = 10):
        self.name = name
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_calls = 0
        self.total_calls = 0
        self.total_rejections = 0

    async def call(
        self,
        func: Callable[..., Awaitable[Any]],
        *args,
        **kwargs
    ) -> Any:
        """Execute with bulkhead"""
        self.total_calls += 1

        acquired = self.semaphore.locked()

        if not acquired:
            self.total_rejections += 1
            raise Exception(f"Bulkhead '{self.name}' is full")

        async with self.semaphore:
            self.active_calls += 1
            try:
                return await func(*args, **kwargs)
            finally:
                self.active_calls -= 1

class Timeout:
    """Timeout pattern"""

    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds

    async def call(
        self,
        func: Callable[..., Awaitable[Any]],
        *args,
        **kwargs
    ) -> Any:
        """Execute with timeout"""
        try:
            return await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.timeout_seconds
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Operation timed out after {self.timeout_seconds}s"
            )

class Retry:
    """Retry pattern with exponential backoff"""

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    async def call(
        self,
        func: Callable[..., Awaitable[Any]],
        *args,
        **kwargs
    ) -> Any:
        """Execute with retry"""
        last_exception = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                return await func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                if attempt == self.max_attempts:
                    raise

                delay = self._calculate_delay(attempt)
                await asyncio.sleep(delay)

        raise last_exception

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff"""
        delay = min(
            self.initial_delay * (self.exponential_base ** (attempt - 1)),
            self.max_delay
        )

        if self.jitter:
            import random
            delay *= (0.5 + random.random())

        return delay

class Fallback:
    """Fallback pattern"""

    def __init__(
        self,
        fallback_func: Callable[..., Awaitable[Any]]
    ):
        self.fallback_func = fallback_func

    async def call(
        self,
        func: Callable[..., Awaitable[Any]],
        *args,
        **kwargs
    ) -> Any:
        """Execute with fallback"""
        try:
            return await func(*args, **kwargs)
        except Exception:
            return await self.fallback_func(*args, **kwargs)

# Combining patterns
class ResilientCall:
    """Combine multiple resilience patterns"""

    def __init__(
        self,
        retry: Optional[Retry] = None,
        timeout: Optional[Timeout] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        bulkhead: Optional[Bulkhead] = None,
        fallback: Optional[Fallback] = None
    ):
        self.retry = retry
        self.timeout = timeout
        self.circuit_breaker = circuit_breaker
        self.bulkhead = bulkhead
        self.fallback = fallback

    async def call(
        self,
        func: Callable[..., Awaitable[Any]],
        *args,
        **kwargs
    ) -> Any:
        """Execute with all configured patterns"""

        async def execute():
            result_func = func

            # Apply patterns in order: timeout -> circuit breaker -> bulkhead
            if self.timeout:
                async def with_timeout(*a, **kw):
                    return await self.timeout.call(result_func, *a, **kw)
                result_func = with_timeout

            if self.circuit_breaker:
                async def with_circuit_breaker(*a, **kw):
                    return await self.circuit_breaker.call(result_func, *a, **kw)
                result_func = with_circuit_breaker

            if self.bulkhead:
                async def with_bulkhead(*a, **kw):
                    return await self.bulkhead.call(result_func, *a, **kw)
                result_func = with_bulkhead

            return await result_func(*args, **kwargs)

        # Apply retry
        if self.retry:
            result = await self.retry.call(execute)
        else:
            result = await execute()

        # Apply fallback on failure
        if self.fallback and result is None:
            result = await self.fallback.call(func, *args, **kwargs)

        return result
```

### Implementation Checklist

- [ ] Create `ia_modules/resilience/circuit_breaker.py`
- [ ] Implement circuit breaker with state machine
- [ ] Add bulkhead pattern
- [ ] Add timeout pattern
- [ ] Add fallback pattern
- [ ] Create combined resilience wrapper
- [ ] Add circuit breaker monitoring endpoints
- [ ] Implement circuit breaker reset API
- [ ] Add metrics for circuit breaker events
- [ ] Create circuit breaker configuration
- [ ] Write resilience pattern tests
- [ ] Document resilience patterns usage

**Estimated Effort**: 2 weeks

---

## 7. Service Mesh & Load Balancing

### Overview
Integrate with service mesh (Istio/Linkerd) and implement load balancing strategies for distributed deployments.

### Requirements

#### 7.1 Istio Integration

```yaml
# ia_modules/deployment/k8s/istio/virtual-service.yaml

apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: ia-modules-vs
  namespace: ia-modules
spec:
  hosts:
  - ia-modules-runner
  - pipelines.example.com
  http:
  - match:
    - uri:
        prefix: "/api/v1/pipelines"
    route:
    - destination:
        host: ia-modules-runner
        port:
          number: 80
    timeout: 3600s
    retries:
      attempts: 3
      perTryTimeout: 1200s
      retryOn: 5xx,reset,connect-failure,refused-stream
  - match:
    - uri:
        prefix: "/health"
    route:
    - destination:
        host: ia-modules-runner
        port:
          number: 80
    timeout: 10s

---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: ia-modules-dr
  namespace: ia-modules
spec:
  host: ia-modules-runner
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 1000
        http2MaxRequests: 1000
        maxRequestsPerConnection: 2
    loadBalancer:
      consistentHash:
        httpHeaderName: "X-Pipeline-ID"
    outlierDetection:
      consecutiveErrors: 5
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
      minHealthPercent: 50

---
apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: ia-modules-gateway
  namespace: ia-modules
spec:
  selector:
    istio: ingressgateway
  servers:
  - port:
      number: 80
      name: http
      protocol: HTTP
    hosts:
    - "pipelines.example.com"
    tls:
      httpsRedirect: true
  - port:
      number: 443
      name: https
      protocol: HTTPS
    hosts:
    - "pipelines.example.com"
    tls:
      mode: SIMPLE
      credentialName: ia-modules-tls
```

#### 7.2 Load Balancer Implementation

```python
# ia_modules/distributed/load_balancer.py

from typing import List, Optional, Dict, Any
from enum import Enum
import hashlib
import random
import logging

class LoadBalancingAlgorithm(Enum):
    """Load balancing algorithms"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    CONSISTENT_HASH = "consistent_hash"
    RANDOM = "random"
    IP_HASH = "ip_hash"

class LoadBalancer:
    """Load balancer implementation"""

    def __init__(
        self,
        algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.ROUND_ROBIN
    ):
        self.algorithm = algorithm
        self.logger = logging.getLogger(__name__)
        self._round_robin_index = 0
        self._consistent_hash_ring = {}

    def select_worker(
        self,
        workers: List[WorkerInfo],
        context: Optional[Dict[str, Any]] = None
    ) -> WorkerInfo:
        """Select worker based on algorithm"""
        if not workers:
            raise ValueError("No workers available")

        if self.algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
            return self._round_robin(workers)
        elif self.algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
            return self._least_connections(workers)
        elif self.algorithm == LoadBalancingAlgorithm.CONSISTENT_HASH:
            return self._consistent_hash(workers, context)
        elif self.algorithm == LoadBalancingAlgorithm.RANDOM:
            return random.choice(workers)
        elif self.algorithm == LoadBalancingAlgorithm.IP_HASH:
            return self._ip_hash(workers, context)
        else:
            return workers[0]

    def _round_robin(self, workers: List[WorkerInfo]) -> WorkerInfo:
        """Round-robin selection"""
        worker = workers[self._round_robin_index % len(workers)]
        self._round_robin_index += 1
        return worker

    def _least_connections(self, workers: List[WorkerInfo]) -> WorkerInfo:
        """Select worker with least active connections"""
        return min(workers, key=lambda w: w.active_pipelines)

    def _consistent_hash(
        self,
        workers: List[WorkerInfo],
        context: Optional[Dict[str, Any]]
    ) -> WorkerInfo:
        """Consistent hashing"""
        if not context or "key" not in context:
            return workers[0]

        key = context["key"]
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)

        # Build hash ring if not exists
        if not self._consistent_hash_ring:
            self._build_hash_ring(workers)

        # Find closest worker
        sorted_hashes = sorted(self._consistent_hash_ring.keys())

        for h in sorted_hashes:
            if hash_value <= h:
                return self._consistent_hash_ring[h]

        return self._consistent_hash_ring[sorted_hashes[0]]

    def _build_hash_ring(self, workers: List[WorkerInfo], replicas: int = 100):
        """Build consistent hash ring"""
        self._consistent_hash_ring = {}

        for worker in workers:
            for i in range(replicas):
                key = f"{worker.worker_id}:{i}"
                hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
                self._consistent_hash_ring[hash_value] = worker

    def _ip_hash(
        self,
        workers: List[WorkerInfo],
        context: Optional[Dict[str, Any]]
    ) -> WorkerInfo:
        """Hash based on client IP"""
        if not context or "client_ip" not in context:
            return workers[0]

        ip = context["client_ip"]
        hash_value = int(hashlib.md5(ip.encode()).hexdigest(), 16)
        index = hash_value % len(workers)

        return workers[index]
```

### Implementation Checklist

- [ ] Create Istio VirtualService configuration
- [ ] Create Istio DestinationRule with circuit breaker
- [ ] Create Istio Gateway configuration
- [ ] Add traffic splitting for canary deployments
- [ ] Implement load balancer class
- [ ] Add consistent hashing support
- [ ] Add health-based load balancing
- [ ] Configure connection pooling
- [ ] Add outlier detection
- [ ] Create service mesh documentation
- [ ] Add monitoring for service mesh
- [ ] Test load balancing strategies

**Estimated Effort**: 1-2 weeks

---

## 8. Blue-Green Deployment Support

### Overview
Implement blue-green deployment capability for zero-downtime releases with automatic rollback.

### Requirements

#### 8.1 Blue-Green Deployment Controller

```python
# ia_modules/deployment/blue_green.py

from typing import Dict, Any, Optional
from enum import Enum
import asyncio
import logging
from dataclasses import dataclass

class DeploymentColor(Enum):
    """Deployment colors"""
    BLUE = "blue"
    GREEN = "green"

@dataclass
class DeploymentStatus:
    """Deployment status"""
    version: str
    color: DeploymentColor
    replicas: int
    ready_replicas: int
    traffic_percentage: int
    health_check_passed: bool
    error_rate: float

class BlueGreenDeployer:
    """Blue-green deployment controller"""

    def __init__(
        self,
        blue_version: str,
        green_version: str,
        traffic_split: Optional[Dict[str, int]] = None,
        rollback_threshold: float = 5.0,
        health_check_duration: int = 300
    ):
        self.blue_version = blue_version
        self.green_version = green_version
        self.traffic_split = traffic_split or {"blue": 100, "green": 0}
        self.rollback_threshold = rollback_threshold
        self.health_check_duration = health_check_duration
        self.logger = logging.getLogger(__name__)

        self.active_color = DeploymentColor.BLUE
        self.blue_status: Optional[DeploymentStatus] = None
        self.green_status: Optional[DeploymentStatus] = None

    async def deploy_new_version(self, new_version: str):
        """Deploy new version to inactive color"""
        # Determine which color is inactive
        inactive_color = (
            DeploymentColor.GREEN
            if self.active_color == DeploymentColor.BLUE
            else DeploymentColor.BLUE
        )

        self.logger.info(
            f"Deploying {new_version} to {inactive_color.value}"
        )

        # Deploy to inactive
        await self._deploy_to_color(inactive_color, new_version)

        # Wait for deployment to be ready
        ready = await self._wait_for_ready(inactive_color)

        if not ready:
            raise Exception(f"Deployment to {inactive_color.value} failed")

        # Run health checks
        healthy = await self._run_health_checks(inactive_color)

        if not healthy:
            raise Exception(f"Health checks failed for {inactive_color.value}")

        # Gradually shift traffic
        await self._shift_traffic(inactive_color)

        # Monitor error rate
        should_rollback = await self._monitor_error_rate(inactive_color)

        if should_rollback:
            self.logger.error("Error rate exceeded threshold, rolling back")
            await self.rollback()
            raise Exception("Deployment rolled back due to high error rate")

        # Switch active color
        self.active_color = inactive_color
        self.logger.info(f"Deployment successful, active: {inactive_color.value}")

    async def _deploy_to_color(self, color: DeploymentColor, version: str):
        """Deploy specific version to color"""
        from kubernetes import client

        deployment_name = f"ia-modules-runner-{color.value}"

        # Update deployment image
        apps_v1 = client.AppsV1Api()

        deployment = apps_v1.read_namespaced_deployment(
            name=deployment_name,
            namespace="ia-modules"
        )

        deployment.spec.template.spec.containers[0].image = f"ia-modules:{version}"

        apps_v1.patch_namespaced_deployment(
            name=deployment_name,
            namespace="ia-modules",
            body=deployment
        )

        self.logger.info(f"Deployed {version} to {color.value}")

    async def _wait_for_ready(
        self,
        color: DeploymentColor,
        timeout: int = 600
    ) -> bool:
        """Wait for deployment to be ready"""
        from kubernetes import client

        deployment_name = f"ia-modules-runner-{color.value}"
        apps_v1 = client.AppsV1Api()

        start_time = asyncio.get_event_loop().time()

        while True:
            deployment = apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace="ia-modules"
            )

            ready_replicas = deployment.status.ready_replicas or 0
            replicas = deployment.spec.replicas

            if ready_replicas >= replicas:
                self.logger.info(f"{color.value} deployment ready ({ready_replicas}/{replicas})")
                return True

            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                self.logger.error(f"Timeout waiting for {color.value} deployment")
                return False

            await asyncio.sleep(5)

    async def _run_health_checks(self, color: DeploymentColor) -> bool:
        """Run health checks on deployment"""
        import httpx

        # Get pod IPs
        pods = await self._get_pods_for_color(color)

        for pod in pods:
            try:
                url = f"http://{pod.status.pod_ip}:8000/health"

                async with httpx.AsyncClient() as client:
                    response = await client.get(url, timeout=10.0)

                    if response.status_code != 200:
                        self.logger.error(
                            f"Health check failed for {pod.metadata.name}: "
                            f"status={response.status_code}"
                        )
                        return False

            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                return False

        return True

    async def _shift_traffic(self, new_color: DeploymentColor):
        """Gradually shift traffic to new deployment"""
        steps = [10, 25, 50, 75, 100]

        for percentage in steps:
            self.logger.info(f"Shifting {percentage}% traffic to {new_color.value}")

            await self._update_traffic_split(new_color, percentage)

            # Wait and monitor
            await asyncio.sleep(60)

            # Check error rate
            error_rate = await self._get_error_rate(new_color)

            if error_rate > self.rollback_threshold:
                self.logger.error(
                    f"Error rate {error_rate}% exceeds threshold {self.rollback_threshold}%"
                )
                return False

        return True

    async def _update_traffic_split(self, new_color: DeploymentColor, percentage: int):
        """Update traffic split in Istio"""
        from kubernetes import client

        # Update VirtualService
        custom_api = client.CustomObjectsApi()

        old_percentage = 100 - percentage
        old_color = (
            DeploymentColor.BLUE
            if new_color == DeploymentColor.GREEN
            else DeploymentColor.GREEN
        )

        virtual_service = {
            "apiVersion": "networking.istio.io/v1beta1",
            "kind": "VirtualService",
            "metadata": {
                "name": "ia-modules-vs",
                "namespace": "ia-modules"
            },
            "spec": {
                "hosts": ["ia-modules-runner"],
                "http": [{
                    "route": [
                        {
                            "destination": {
                                "host": f"ia-modules-runner-{old_color.value}"
                            },
                            "weight": old_percentage
                        },
                        {
                            "destination": {
                                "host": f"ia-modules-runner-{new_color.value}"
                            },
                            "weight": percentage
                        }
                    ]
                }]
            }
        }

        custom_api.patch_namespaced_custom_object(
            group="networking.istio.io",
            version="v1beta1",
            namespace="ia-modules",
            plural="virtualservices",
            name="ia-modules-vs",
            body=virtual_service
        )

    async def _monitor_error_rate(self, color: DeploymentColor) -> bool:
        """Monitor error rate during deployment"""
        # Query Prometheus for error rate
        # Return True if should rollback

        import httpx

        prometheus_url = "http://prometheus:9090/api/v1/query"
        query = f'''
            rate(http_requests_total{{
                deployment=~"ia-modules-runner-{color.value}.*",
                status=~"5.."
            }}[5m])
        '''

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    prometheus_url,
                    params={"query": query}
                )

                data = response.json()

                if data['data']['result']:
                    error_rate = float(data['data']['result'][0]['value'][1])

                    if error_rate > self.rollback_threshold:
                        return True

        except Exception as e:
            self.logger.error(f"Failed to query metrics: {e}")

        return False

    async def _get_error_rate(self, color: DeploymentColor) -> float:
        """Get current error rate"""
        # Simplified version
        return 0.0

    async def _get_pods_for_color(self, color: DeploymentColor):
        """Get pods for specific color"""
        from kubernetes import client

        core_v1 = client.CoreV1Api()

        pods = core_v1.list_namespaced_pod(
            namespace="ia-modules",
            label_selector=f"app=ia-modules,deployment-color={color.value}"
        )

        return pods.items

    async def rollback(self):
        """Rollback to previous version"""
        # Shift all traffic back to old deployment
        old_color = (
            DeploymentColor.BLUE
            if self.active_color == DeploymentColor.GREEN
            else DeploymentColor.GREEN
        )

        self.logger.warning(f"Rolling back to {old_color.value}")

        await self._update_traffic_split(old_color, 100)

        self.logger.info("Rollback complete")
```

#### 8.2 Kubernetes Manifests for Blue-Green

```yaml
# ia_modules/deployment/k8s/blue-green/blue-deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: ia-modules-runner-blue
  namespace: ia-modules
  labels:
    app: ia-modules
    deployment-color: blue
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ia-modules
      deployment-color: blue
  template:
    metadata:
      labels:
        app: ia-modules
        deployment-color: blue
    spec:
      containers:
      - name: pipeline-runner
        image: ia-modules:0.0.2  # Old version
        # ... rest of container spec

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ia-modules-runner-green
  namespace: ia-modules
  labels:
    app: ia-modules
    deployment-color: green
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ia-modules
      deployment-color: green
  template:
    metadata:
      labels:
        app: ia-modules
        deployment-color: green
    spec:
      containers:
      - name: pipeline-runner
        image: ia-modules:0.0.3  # New version
        # ... rest of container spec

---
apiVersion: v1
kind: Service
metadata:
  name: ia-modules-runner-blue
  namespace: ia-modules
spec:
  selector:
    app: ia-modules
    deployment-color: blue
  ports:
  - port: 80
    targetPort: 8000

---
apiVersion: v1
kind: Service
metadata:
  name: ia-modules-runner-green
  namespace: ia-modules
spec:
  selector:
    app: ia-modules
    deployment-color: green
  ports:
  - port: 80
    targetPort: 8000
```

### Implementation Checklist

- [ ] Create `ia_modules/deployment/blue_green.py`
- [ ] Implement blue-green deployment controller
- [ ] Create Kubernetes manifests for blue/green
- [ ] Add traffic shifting logic (Istio)
- [ ] Implement health check validation
- [ ] Add error rate monitoring
- [ ] Implement automatic rollback
- [ ] Create deployment CLI commands
- [ ] Add canary deployment support
- [ ] Create deployment dashboard
- [ ] Write deployment tests
- [ ] Document deployment procedures

**Estimated Effort**: 2-3 weeks

---

## Implementation Timeline

### Phase 1: Container & Orchestration (Weeks 1-3)
- Week 1: Dockerfile, Docker Compose, health checks
- Week 2: Kubernetes manifests, HPA
- Week 3: Helm charts, documentation

### Phase 2: Distributed Execution (Weeks 4-7)
- Week 4: Shared state backend, worker registry
- Week 5: Kubernetes backend implementation
- Week 6: Load balancing strategies
- Week 7: Testing and documentation

### Phase 3: Database & Queue (Weeks 8-11)
- Week 8: Connection pooling implementation
- Week 9: Message queue interface
- Week 10: Queue backend implementations
- Week 11: Integration and testing

### Phase 4: Resilience & Security (Weeks 12-15)
- Week 12: Rate limiting implementation
- Week 13: Circuit breaker pattern
- Week 14: Additional resilience patterns
- Week 15: Testing and documentation

### Phase 5: Service Mesh & Deployment (Weeks 16-19)
- Week 16: Istio configuration
- Week 17: Load balancer implementation
- Week 18: Blue-green deployment
- Week 19: End-to-end testing and documentation

**Total Estimated Time**: 19 weeks (~5 months)

---

## Dependencies & Prerequisites

### Python Packages

```txt
# requirements-production.txt

# Container & Orchestration
kubernetes>=27.0.0           # Kubernetes Python client
docker>=6.1.0                # Docker SDK

# Distributed Execution
redis>=5.0.0                 # Redis client
ray[default]>=2.6.0          # Ray distributed computing (optional)
celery>=5.3.0                # Celery task queue (optional)

# Database
asyncpg>=0.28.0              # PostgreSQL async driver
aiomysql>=0.2.0              # MySQL async driver
pyodbc>=4.0.39               # MSSQL driver
SQLAlchemy>=2.0.20           # ORM with connection pooling

# Message Queues
aio-pika>=9.2.0              # RabbitMQ async client
aiokafka>=0.8.1              # Kafka async client
aioboto3>=11.3.0             # AWS SQS client

# Monitoring & Metrics
prometheus-client>=0.17.0    # Prometheus metrics
psutil>=5.9.5                # System monitoring

# HTTP & Networking
httpx>=0.24.1                # Async HTTP client
aiohttp>=3.8.5               # Async HTTP

# Utilities
tenacity>=8.2.3              # Retry logic
python-dotenv>=1.0.0         # Environment management
pydantic>=2.3.0              # Data validation
```

### System Requirements

- Python 3.11+
- Kubernetes 1.25+ (for orchestration)
- Docker 24.0+ (for containerization)
- Redis 7+ (for shared state and caching)
- PostgreSQL 15+ or MySQL 8+ (for database)
- 8GB+ RAM (for local development)
- 16GB+ RAM (for production)

### Infrastructure Requirements

- Kubernetes cluster (EKS, GKE, AKS, or self-hosted)
- Container registry (Docker Hub, ECR, GCR, ACR)
- Load balancer (ALB, NLB, or nginx)
- Service mesh (Istio or Linkerd) - optional
- Monitoring stack (Prometheus + Grafana)
- Message queue (RabbitMQ, Kafka, or SQS) - optional

---

## Success Criteria

### Container Orchestration
- [ ] Docker images build successfully with <500MB size
- [ ] Kubernetes deployments scale automatically based on load
- [ ] Health checks detect and restart unhealthy pods
- [ ] Zero-downtime deployments achieved

### Distributed Execution
- [ ] Pipelines execute across multiple workers
- [ ] Load balancing distributes work evenly
- [ ] Worker failures handled gracefully with failover
- [ ] Shared state synchronized correctly

### Database Performance
- [ ] Connection pool maintains stable pool size
- [ ] Query latency <10ms for simple queries
- [ ] No connection leaks under load
- [ ] Pool exhaustion handled gracefully

### Message Queue
- [ ] Messages processed reliably (no data loss)
- [ ] Failed messages retry with backoff
- [ ] Queue throughput >1000 messages/sec
- [ ] Dead letter queue captures failed messages

### Rate Limiting
- [ ] API rate limits enforced correctly
- [ ] Distributed rate limiting works across replicas
- [ ] Legitimate requests not blocked
- [ ] Abuse attempts blocked effectively

### Circuit Breaker
- [ ] Circuit opens after failures
- [ ] Circuit closes after recovery
- [ ] Cascading failures prevented
- [ ] Fallbacks execute when circuit open

### Blue-Green Deployment
- [ ] Zero-downtime deployments
- [ ] Automatic rollback on errors
- [ ] Traffic shifting works correctly
- [ ] Canary deployments supported

---

**Document Status**: Planning Phase
**Last Updated**: 2025-10-25
**Next Review**: After team review and prioritization
