"""
Agentic Patterns API Endpoints

Exposes the PatternService (reflection, planning, tool use, agentic RAG,
metacognition) over HTTP so the frontend PatternsPage can execute them.
"""

import sys
from pathlib import Path

_backend_dir = str(Path(__file__).parent.parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from services.pattern_service import PatternService

router = APIRouter(prefix="/api/patterns", tags=["patterns"])

# Shared service instance
pattern_service = PatternService()


# ==================== REQUEST MODELS ====================

class ReflectionRequest(BaseModel):
    initial_output: str = Field(..., description="Text to improve via reflection")
    criteria: Dict[str, str] = Field(..., description="Quality criteria keyed by name")
    max_iterations: int = Field(default=3, ge=1, le=10)


class PlanningRequest(BaseModel):
    goal: str = Field(..., description="High-level goal to decompose")
    constraints: Optional[Dict[str, Any]] = None


class ToolUseRequest(BaseModel):
    task: str = Field(..., description="Task to accomplish")
    available_tools: List[str] = Field(..., description="Available tool names")


class AgenticRAGRequest(BaseModel):
    query: str = Field(default="", description="Initial search query")
    initial_query: str = Field(default="", description="Alias for query")
    max_refinements: int = Field(default=3, ge=1, le=10)


class MetacognitionRequest(BaseModel):
    execution_trace: List[Dict[str, Any]] = Field(..., description="History of agent actions")
    performance_metrics: Dict[str, float] = Field(..., description="Performance measurements")


# ==================== ENDPOINTS ====================

@router.post("/reflection")
async def run_reflection(request: ReflectionRequest) -> Dict[str, Any]:
    """Run the reflection pattern: self-critique and iterative improvement"""
    try:
        result = await pattern_service.reflection_example(
            initial_output=request.initial_output,
            criteria=request.criteria,
            max_iterations=request.max_iterations
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/planning")
async def run_planning(request: PlanningRequest) -> Dict[str, Any]:
    """Run the planning pattern: goal decomposition into steps"""
    try:
        result = await pattern_service.planning_example(
            goal=request.goal,
            constraints=request.constraints
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tool-use")
async def run_tool_use(request: ToolUseRequest) -> Dict[str, Any]:
    """Run the tool use pattern: dynamic tool selection and planning"""
    try:
        result = await pattern_service.tool_use_example(
            task=request.task,
            available_tools=request.available_tools
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agentic-rag")
async def run_agentic_rag(request: AgenticRAGRequest) -> Dict[str, Any]:
    """Run the agentic RAG pattern: query refinement and retrieval"""
    try:
        query = request.query or request.initial_query
        if not query:
            raise ValueError("Either 'query' or 'initial_query' is required")
        result = await pattern_service.agentic_rag_example(
            initial_query=query,
            max_refinements=request.max_refinements
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metacognition")
async def run_metacognition(request: MetacognitionRequest) -> Dict[str, Any]:
    """Run the metacognition pattern: self-monitoring and adaptation"""
    try:
        result = await pattern_service.metacognition_example(
            execution_trace=request.execution_trace,
            performance_metrics=request.performance_metrics
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
