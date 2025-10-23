"""Pipeline management API endpoints"""

from fastapi import APIRouter, HTTPException, Depends, Request
from typing import List
from models import PipelineCreate, PipelineUpdate, PipelineResponse

router = APIRouter()


def get_pipeline_service(request: Request):
    """Dependency to get pipeline service"""
    return request.app.state.services.pipeline_service


@router.get("", response_model=List[PipelineResponse])
async def list_pipelines(service=Depends(get_pipeline_service)):
    """List all pipelines"""
    pipelines = await service.list_pipelines()
    return pipelines


@router.get("/{pipeline_id}", response_model=PipelineResponse)
async def get_pipeline(pipeline_id: str, service=Depends(get_pipeline_service)):
    """Get pipeline by ID"""
    pipeline = await service.get_pipeline(pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    return pipeline


@router.post("/", response_model=PipelineResponse)
async def create_pipeline(data: PipelineCreate, service=Depends(get_pipeline_service)):
    """Create new pipeline"""
    pipeline_id = await service.create_pipeline(data.model_dump())
    pipeline = await service.get_pipeline(pipeline_id)
    return pipeline


@router.put("/{pipeline_id}", response_model=PipelineResponse)
async def update_pipeline(
    pipeline_id: str,
    data: PipelineUpdate,
    service=Depends(get_pipeline_service)
):
    """Update pipeline"""
    success = await service.update_pipeline(pipeline_id, data.model_dump(exclude_unset=True))
    if not success:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    pipeline = await service.get_pipeline(pipeline_id)
    return pipeline


@router.delete("/{pipeline_id}")
async def delete_pipeline(pipeline_id: str, service=Depends(get_pipeline_service)):
    """Delete pipeline"""
    success = await service.delete_pipeline(pipeline_id)
    if not success:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    return {"message": "Pipeline deleted successfully"}
