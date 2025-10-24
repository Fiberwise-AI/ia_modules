"""Pydantic models for API requests and responses"""

from datetime import datetime
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field
from enum import Enum


class ExecutionStatus(str, Enum):
    """Pipeline execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PipelineCreate(BaseModel):
    """Create pipeline request"""
    name: str = Field(..., description="Pipeline name")
    description: Optional[str] = Field(None, description="Pipeline description")
    config: Dict[str, Any] = Field(..., description="Pipeline configuration (JSON)")
    tags: List[str] = Field(default_factory=list, description="Pipeline tags")


class PipelineUpdate(BaseModel):
    """Update pipeline request"""
    name: Optional[str] = None
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None


class PipelineResponse(BaseModel):
    """Pipeline response"""
    id: str
    name: str
    description: Optional[str]
    config: Dict[str, Any]
    tags: List[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ExecutionRequest(BaseModel):
    """Start pipeline execution request"""
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data")
    checkpoint_enabled: bool = Field(default=True, description="Enable checkpointing")


class ExecutionResponse(BaseModel):
    """Pipeline execution response"""
    job_id: str
    pipeline_id: str
    status: ExecutionStatus
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]]
    current_step: Optional[str]
    started_at: datetime
    completed_at: Optional[datetime]
    error: Optional[str]
    progress: float = Field(ge=0, le=1, description="Execution progress (0-1)")

    class Config:
        from_attributes = True


class StepUpdate(BaseModel):
    """Real-time step execution update"""
    job_id: str
    step_id: str
    status: str
    data: Optional[Dict[str, Any]]
    timestamp: datetime


class MetricsReport(BaseModel):
    """Reliability metrics report"""
    sr: float = Field(description="Success Rate")
    cr: float = Field(description="Compensation Rate")
    pc: float = Field(description="Pass Confidence")
    hir: float = Field(description="Human Intervention Rate")
    ma: float = Field(description="Model Accuracy")
    tcl: Optional[float] = Field(None, description="Tool Call Latency (ms)")
    wct: Optional[float] = Field(None, description="Workflow Completion Time (ms)")
    total_steps: int
    successful_steps: int
    failed_steps: int
    timestamp: datetime


class SLOCompliance(BaseModel):
    """SLO compliance status"""
    sr_compliant: bool
    sr_current: float
    sr_target: float
    cr_compliant: bool
    cr_current: float
    cr_target: float
    hir_compliant: bool
    hir_current: float
    hir_target: float
    overall_compliant: bool
    timestamp: datetime


class EventLog(BaseModel):
    """Event log entry"""
    id: str
    event_type: str
    agent_name: str
    success: bool
    timestamp: datetime
    data: Optional[Dict[str, Any]]


class WebSocketMessage(BaseModel):
    """WebSocket message format"""
    type: str  # "step_update", "execution_complete", "metrics_update", etc.
    data: Dict[str, Any]
    timestamp: datetime


# Telemetry Models

class SpanAttributes(BaseModel):
    """Span attributes"""
    job_id: Optional[str] = None
    step_name: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class SpanResponse(BaseModel):
    """Telemetry span response"""
    span_id: str
    parent_id: Optional[str]
    name: str
    start_time: Optional[str]
    end_time: Optional[str]
    duration_ms: Optional[float]
    status: str = "ok"
    attributes: Dict[str, Any] = Field(default_factory=dict)


class SpanTimelineResponse(BaseModel):
    """Span timeline response"""
    span_id: str
    parent_id: Optional[str]
    name: str
    start_time: Optional[str]
    end_time: Optional[str]
    duration_ms: float = 0
    status: str = "ok"
    depth: int = 0
    attributes: Dict[str, Any] = Field(default_factory=dict)


class TelemetryMetrics(BaseModel):
    """Aggregated telemetry metrics"""
    total_spans: int
    total_duration_ms: float
    step_count: int
    error_count: int
    avg_step_duration_ms: float = 0


# Checkpoint Models

class CheckpointResponse(BaseModel):
    """Checkpoint response"""
    id: str
    job_id: str
    step_name: str
    created_at: Optional[str]
    state_size: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CheckpointStateResponse(BaseModel):
    """Checkpoint state response"""
    checkpoint_id: str
    state: Dict[str, Any]


class CheckpointResumeResponse(BaseModel):
    """Checkpoint resume response"""
    original_job_id: str
    new_job_id: str
    resumed_from_checkpoint: str
    resumed_at_step: str


# Memory Models

class MemoryMessage(BaseModel):
    """Memory message"""
    role: str = "user"
    content: str
    timestamp: Optional[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryStats(BaseModel):
    """Memory statistics"""
    message_count: int
    total_tokens: int
    first_message: Optional[str]
    last_message: Optional[str]
    avg_message_length: int = 0


class MemorySearchRequest(BaseModel):
    """Memory search request"""
    query: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    limit: int = Field(10, ge=1, le=100)


# Replay Models

class ReplayExecutionResponse(BaseModel):
    """Replay execution response"""
    original_job_id: str
    replay_job_id: str
    original: Dict[str, Any]
    replay: Dict[str, Any]
    comparison: Dict[str, Any]
    replayed_at: str


class ReplayComparison(BaseModel):
    """Replay comparison"""
    identical: bool
    difference_count: int
    differences: List[Dict[str, Any]]
    output_match: bool
    status_match: bool


class ReplayHistoryItem(BaseModel):
    """Replay history item"""
    replay_id: Optional[str]
    original_job_id: str
    replay_job_id: str
    success: bool
    differences: List[Dict[str, Any]] = Field(default_factory=list)
    replayed_at: Optional[str]


# Step Detail Models

class StepDetailResponse(BaseModel):
    """Detailed step information"""
    step_name: str
    status: str
    started_at: Optional[str]
    completed_at: Optional[str]
    duration_ms: Optional[float]
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str]
    retry_count: int = 0
    tokens: Optional[int]
    cost: Optional[float]
    logs: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Pipeline Graph Models

class GraphNode(BaseModel):
    """Pipeline graph node"""
    id: str
    type: str = "step"
    label: str
    config: Dict[str, Any] = Field(default_factory=dict)
    position: Dict[str, float] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    """Pipeline graph edge"""
    source: str
    target: str
    condition: Optional[Dict[str, Any]]
    label: Optional[str]


class PipelineGraphResponse(BaseModel):
    """Pipeline graph structure"""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
