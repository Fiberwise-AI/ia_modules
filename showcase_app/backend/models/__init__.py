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
