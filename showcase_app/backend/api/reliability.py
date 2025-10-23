"""Reliability metrics API endpoints"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
from pydantic import BaseModel

router = APIRouter()


def get_reliability_service():
    """Dependency to get reliability service"""
    from main import get_reliability_service
    return get_reliability_service()


@router.get("/metrics")
async def get_reliability_metrics(
    pipeline_id: Optional[str] = None,
    service=Depends(get_reliability_service)
):
    """Get reliability metrics (SR, CR, PC, HIR, MA, TCL, WCT)"""
    try:
        metrics = await service.get_metrics(pipeline_id=pipeline_id)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/slo")
async def get_slo_status(
    pipeline_id: Optional[str] = None,
    service=Depends(get_reliability_service)
):
    """Get SLO compliance status"""
    try:
        slo_status = await service.get_slo_status(pipeline_id=pipeline_id)
        return slo_status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/anomalies")
async def get_anomalies(
    pipeline_id: Optional[str] = None,
    limit: int = 50,
    service=Depends(get_reliability_service)
):
    """Get detected anomalies"""
    try:
        anomalies = await service.get_anomalies(
            pipeline_id=pipeline_id,
            limit=limit
        )
        return {"anomalies": anomalies}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def get_alerts(
    pipeline_id: Optional[str] = None,
    active_only: bool = True,
    limit: int = 50,
    service=Depends(get_reliability_service)
):
    """Get active alerts"""
    try:
        alerts = await service.get_alerts(
            pipeline_id=pipeline_id,
            active_only=active_only,
            limit=limit
        )
        return {"alerts": alerts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/circuit-breakers")
async def get_circuit_breaker_status(
    service=Depends(get_reliability_service)
):
    """Get circuit breaker status for all pipelines"""
    try:
        status = await service.get_circuit_breaker_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cost")
async def get_cost_metrics(
    pipeline_id: Optional[str] = None,
    service=Depends(get_reliability_service)
):
    """Get cost tracking metrics"""
    try:
        cost = await service.get_cost_metrics(pipeline_id=pipeline_id)
        return cost
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trends")
async def get_trend_analysis(
    metric_name: str,
    pipeline_id: Optional[str] = None,
    window_size: int = 10,
    service=Depends(get_reliability_service)
):
    """Get trend analysis for a specific metric"""
    try:
        trends = await service.get_trend_analysis(
            metric_name=metric_name,
            pipeline_id=pipeline_id,
            window_size=window_size
        )
        return trends
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
