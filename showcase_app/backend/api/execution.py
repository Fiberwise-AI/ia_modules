"""Pipeline execution API endpoints"""

from fastapi import APIRouter, HTTPException, Depends, Request
from typing import List, Dict, Any
from pydantic import BaseModel

router = APIRouter()


class ExecutionRequest(BaseModel):
    input_data: Dict[str, Any] = {}
    checkpoint_enabled: bool = True


def get_pipeline_service(request: Request):
    """Dependency to get pipeline service"""
    return request.app.state.services.pipeline_service


@router.post("/{pipeline_id}")
async def execute_pipeline(
    pipeline_id: str,
    request: ExecutionRequest,
    service=Depends(get_pipeline_service)
):
    """Start pipeline execution"""
    try:
        job_id = await service.execute_pipeline(
            pipeline_id=pipeline_id,
            input_data=request.input_data,
            checkpoint_enabled=request.checkpoint_enabled
        )

        execution = await service.get_execution(job_id)
        return execution

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{job_id}")
async def get_execution_status(job_id: str, service=Depends(get_pipeline_service)):
    """Get execution status"""
    import logging
    logger = logging.getLogger(__name__)

    execution = await service.get_execution(job_id)
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")

    # Add pipeline_name if missing (for old executions)
    if not execution.get('pipeline_name') and execution.get('pipeline_id'):
        pipeline = await service.get_pipeline(execution['pipeline_id'])
        if pipeline:
            execution['pipeline_name'] = pipeline.get('name', 'Unknown Pipeline')

    logger.info(f"Returning execution {job_id}: pipeline_name={execution.get('pipeline_name')}")

    return execution


@router.get("/")
async def list_executions(service=Depends(get_pipeline_service)):
    """List all executions"""
    executions = await service.list_executions()

    # Add pipeline_name to all executions that don't have it
    for execution in executions:
        if not execution.get('pipeline_name') and execution.get('pipeline_id'):
            pipeline = await service.get_pipeline(execution['pipeline_id'])
            if pipeline:
                execution['pipeline_name'] = pipeline.get('name', 'Unknown Pipeline')

    return executions


@router.delete("/{job_id}")
async def cancel_execution(job_id: str, service=Depends(get_pipeline_service)):
    """Cancel execution"""
    success = await service.cancel_execution(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Execution not found or cannot be cancelled")

    return {"message": "Execution cancelled successfully"}
