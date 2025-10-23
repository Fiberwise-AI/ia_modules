"""Benchmarking API endpoints"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

router = APIRouter()


class BenchmarkRequest(BaseModel):
    pipeline_id: str
    iterations: int = 10
    input_data: Dict[str, Any] = {}
    warmup_iterations: int = 2


class ComparisonRequest(BaseModel):
    pipeline_ids: List[str]
    iterations: int = 10
    input_data: Dict[str, Any] = {}


def get_benchmark_service():
    """Dependency to get benchmark service"""
    from main import get_benchmark_service
    return get_benchmark_service()


@router.post("/run")
async def run_benchmark(
    request: BenchmarkRequest,
    service=Depends(get_benchmark_service)
):
    """Run performance benchmark for a pipeline"""
    try:
        result = await service.run_benchmark(
            pipeline_id=request.pipeline_id,
            iterations=request.iterations,
            input_data=request.input_data,
            warmup_iterations=request.warmup_iterations
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
async def compare_pipelines(
    request: ComparisonRequest,
    service=Depends(get_benchmark_service)
):
    """Compare performance of multiple pipelines"""
    try:
        if len(request.pipeline_ids) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least 2 pipeline IDs required for comparison"
            )

        comparison = await service.compare_pipelines(
            pipeline_ids=request.pipeline_ids,
            iterations=request.iterations,
            input_data=request.input_data
        )
        return comparison
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results")
async def list_benchmark_results(
    pipeline_id: Optional[str] = None,
    limit: int = 50,
    service=Depends(get_benchmark_service)
):
    """List benchmark results"""
    try:
        results = await service.list_results(
            pipeline_id=pipeline_id,
            limit=limit
        )
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results/{result_id}")
async def get_benchmark_result(
    result_id: str,
    service=Depends(get_benchmark_service)
):
    """Get benchmark result details"""
    try:
        result = await service.get_result(result_id)
        if not result:
            raise HTTPException(status_code=404, detail="Result not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/results/{result_id}")
async def delete_benchmark_result(
    result_id: str,
    service=Depends(get_benchmark_service)
):
    """Delete a benchmark result"""
    try:
        success = await service.delete_result(result_id)
        if not success:
            raise HTTPException(status_code=404, detail="Result not found")
        return {"message": "Result deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
