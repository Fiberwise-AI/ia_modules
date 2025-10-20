"""
Data models for the Dashboard API
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum


# ============================================================================
# Pipeline Models
# ============================================================================

class Pipeline(BaseModel):
    """Pipeline model"""
    id: str
    name: str
    description: Optional[str] = None
    config: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    enabled: bool = True


class PipelineCreate(BaseModel):
    """Create pipeline request"""
    name: str
    description: Optional[str] = None
    config: Dict[str, Any]
    tags: List[str] = Field(default_factory=list)


class PipelineUpdate(BaseModel):
    """Update pipeline request"""
    name: Optional[str] = None
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    enabled: Optional[bool] = None


# ============================================================================
# Execution Models
# ============================================================================

class ExecutionStatusEnum(str, Enum):
    """Execution status enum"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionRequest(BaseModel):
    """Execute pipeline request"""
    input_data: Dict[str, Any] = Field(default_factory=dict)
    config_overrides: Optional[Dict[str, Any]] = None


class ExecutionResponse(BaseModel):
    """Execute pipeline response"""
    execution_id: str
    pipeline_id: str
    status: str
    started_at: str


class StepStatus(BaseModel):
    """Status of a pipeline step"""
    step_name: str
    status: ExecutionStatusEnum
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ExecutionStatus(BaseModel):
    """Detailed execution status"""
    execution_id: str
    pipeline_id: str
    status: ExecutionStatusEnum
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    steps: List[StepStatus] = Field(default_factory=list)
    current_step: Optional[str] = None
    progress_percent: float = 0.0
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


# ============================================================================
# Metrics Models
# ============================================================================

class MetricPoint(BaseModel):
    """A single metric data point"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = Field(default_factory=dict)


class Metric(BaseModel):
    """Metric with time series data"""
    name: str
    type: str  # counter, gauge, histogram, summary
    help_text: str
    points: List[MetricPoint] = Field(default_factory=list)


class MetricsResponse(BaseModel):
    """Metrics response"""
    metrics: List[Metric]
    time_range: str
    pipeline_id: Optional[str] = None


class BenchmarkResponse(BaseModel):
    """Benchmark result"""
    id: str
    pipeline_id: str
    pipeline_name: str
    timestamp: datetime
    iterations: int
    mean_time: float
    median_time: float
    p95_time: float
    p99_time: float
    operations_per_second: float
    items_processed: int = 0
    api_calls_count: int = 0
    estimated_cost_usd: float = 0.0
    memory_mb: Optional[float] = None
    cpu_percent: Optional[float] = None


# ============================================================================
# WebSocket Message Models
# ============================================================================

class WSMessageType(str, Enum):
    """WebSocket message types"""
    EXECUTION_STARTED = "execution_started"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    LOG_MESSAGE = "log_message"
    PROGRESS_UPDATE = "progress_update"
    EXECUTION_COMPLETED = "execution_completed"
    EXECUTION_FAILED = "execution_failed"
    METRICS_UPDATE = "metrics_update"


class WSMessage(BaseModel):
    """WebSocket message"""
    type: WSMessageType
    execution_id: str
    timestamp: datetime
    data: Dict[str, Any] = Field(default_factory=dict)


class LogLevel(str, Enum):
    """Log levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LogMessage(BaseModel):
    """Log message"""
    level: LogLevel
    message: str
    timestamp: datetime
    step_name: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


# ============================================================================
# Validation Models
# ============================================================================

class ValidationError(BaseModel):
    """Validation error"""
    field: str
    message: str
    severity: str  # error, warning, info


class ValidationResult(BaseModel):
    """Validation result"""
    valid: bool
    errors: List[ValidationError] = Field(default_factory=list)
    warnings: List[ValidationError] = Field(default_factory=list)
