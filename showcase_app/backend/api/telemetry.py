"""Telemetry API endpoints for spans and metrics"""

from fastapi import APIRouter, HTTPException, Depends, Request
from typing import List
from ..models import SpanResponse, SpanTimelineResponse, TelemetryMetrics

router = APIRouter()


def get_telemetry_service(request: Request):
    """Dependency to get telemetry service"""
    return request.app.state.services.telemetry_service


@router.get("/spans/{job_id}")
async def get_execution_spans(job_id: str, service=Depends(get_telemetry_service)):
    """
    Get all telemetry spans for a specific execution
    
    Returns list of spans with timing and attribute information
    """
    try:
        spans = await service.get_execution_spans(job_id)
        span_responses = [SpanResponse(**span) for span in spans]
        return {"job_id": job_id, "spans": span_responses, "count": len(span_responses)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{job_id}", response_model=TelemetryMetrics)
async def get_execution_metrics(job_id: str, service=Depends(get_telemetry_service)):
    """
    Get aggregated metrics for a specific execution
    
    Returns summary metrics including duration, step count, errors
    """
    try:
        metrics = await service.get_execution_metrics(job_id)
        return TelemetryMetrics(**metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/timeline/{job_id}")
async def get_span_timeline(job_id: str, service=Depends(get_telemetry_service)):
    """
    Get spans formatted for timeline visualization
    
    Returns spans with depth calculation for nested visualization
    """
    try:
        timeline = await service.get_span_timeline(job_id)
        timeline_responses = [SpanTimelineResponse(**span) for span in timeline]
        return {"job_id": job_id, "timeline": timeline_responses, "count": len(timeline_responses)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
