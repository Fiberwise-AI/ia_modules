"""Checkpoint management API endpoints"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

router = APIRouter()


class ResumeRequest(BaseModel):
    checkpoint_id: str
    input_data: Optional[Dict[str, Any]] = None


def get_pipeline_service():
    """Dependency to get pipeline service"""
    from main import get_pipeline_service
    return get_pipeline_service()


@router.get("/{pipeline_id}")
async def list_checkpoints(
    pipeline_id: str,
    thread_id: Optional[str] = None,
    service=Depends(get_pipeline_service)
):
    """List checkpoints for a pipeline"""
    try:
        checkpoints = await service.list_checkpoints(
            pipeline_id=pipeline_id,
            thread_id=thread_id
        )
        return {"checkpoints": checkpoints}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/checkpoint/{checkpoint_id}")
async def get_checkpoint(
    checkpoint_id: str,
    service=Depends(get_pipeline_service)
):
    """Get checkpoint details"""
    try:
        checkpoint = await service.get_checkpoint(checkpoint_id)
        if not checkpoint:
            raise HTTPException(status_code=404, detail="Checkpoint not found")
        return checkpoint
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/resume")
async def resume_from_checkpoint(
    request: ResumeRequest,
    service=Depends(get_pipeline_service)
):
    """Resume pipeline execution from a checkpoint"""
    try:
        job_id = await service.resume_from_checkpoint(
            checkpoint_id=request.checkpoint_id,
            input_data=request.input_data
        )
        execution = await service.get_execution(job_id)
        return execution
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{checkpoint_id}")
async def delete_checkpoint(
    checkpoint_id: str,
    service=Depends(get_pipeline_service)
):
    """Delete a checkpoint"""
    try:
        success = await service.delete_checkpoint(checkpoint_id)
        if not success:
            raise HTTPException(status_code=404, detail="Checkpoint not found")
        return {"message": "Checkpoint deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
