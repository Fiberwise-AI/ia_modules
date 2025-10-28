"""
Prompt Optimization API endpoints.

This module provides REST API endpoints for demonstrating prompt optimization,
including genetic algorithms, A/B testing, and reinforcement learning approaches.
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import logging
import asyncio

logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response Models

class OptimizationMethod(str, Enum):
    """Methods for prompt optimization."""
    GENETIC = "genetic"  # Genetic algorithm
    AB_TEST = "ab_test"  # A/B testing
    REINFORCEMENT = "reinforcement"  # Reinforcement learning
    GRID_SEARCH = "grid_search"  # Grid search over parameters
    BAYESIAN = "bayesian"  # Bayesian optimization


class JobStatus(str, Enum):
    """Status of optimization job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EvaluationMetric(str, Enum):
    """Metrics for evaluating prompts."""
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    CREATIVITY = "creativity"
    CONCISENESS = "conciseness"
    CUSTOM = "custom"


class PromptVariant(BaseModel):
    """A variant of a prompt."""
    id: str = Field(..., description="Variant ID")
    template: str = Field(..., description="Prompt template")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Template parameters")
    score: Optional[float] = Field(None, description="Evaluation score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "variant_1",
                "template": "You are a {role}. {instruction}",
                "parameters": {"role": "helpful assistant", "instruction": "Answer concisely"},
                "score": 0.85
            }
        }


class OptimizationConfig(BaseModel):
    """Configuration for optimization."""
    method: OptimizationMethod = Field(..., description="Optimization method to use")
    population_size: int = Field(10, ge=2, le=100, description="Population size (for genetic/AB)")
    generations: int = Field(5, ge=1, le=50, description="Number of generations/iterations")
    mutation_rate: float = Field(0.1, ge=0, le=1, description="Mutation rate (for genetic)")
    crossover_rate: float = Field(0.7, ge=0, le=1, description="Crossover rate (for genetic)")
    evaluation_metric: EvaluationMetric = Field(EvaluationMetric.ACCURACY, description="Metric to optimize")
    custom_evaluator: Optional[str] = Field(None, description="Custom evaluator function")
    early_stopping: bool = Field(True, description="Stop early if convergence detected")
    target_score: float = Field(0.9, ge=0, le=1, description="Target score to achieve")


class TestCase(BaseModel):
    """Test case for evaluating prompts."""
    input: str = Field(..., description="Test input")
    expected_output: Optional[str] = Field(None, description="Expected output (for supervised)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "input": "What is 2+2?",
                "expected_output": "4",
                "metadata": {"difficulty": "easy"}
            }
        }


class OptimizeRequest(BaseModel):
    """Request to optimize a prompt."""
    base_prompt: str = Field(..., description="Base prompt template to optimize")
    task_description: str = Field(..., description="Description of what the prompt should accomplish")
    test_cases: List[TestCase] = Field(..., description="Test cases for evaluation")
    config: OptimizationConfig = Field(..., description="Optimization configuration")
    model: Optional[str] = Field(None, description="LLM model to use for testing")

    class Config:
        json_schema_extra = {
            "example": {
                "base_prompt": "Answer the question: {question}",
                "task_description": "Answer math questions accurately",
                "test_cases": [
                    {"input": "What is 5+3?", "expected_output": "8"},
                    {"input": "What is 10-4?", "expected_output": "6"}
                ],
                "config": {
                    "method": "genetic",
                    "population_size": 10,
                    "generations": 5,
                    "evaluation_metric": "accuracy"
                }
            }
        }


class OptimizeResponse(BaseModel):
    """Response from optimization request."""
    job_id: str = Field(..., description="Unique job ID for tracking")
    status: JobStatus = Field(..., description="Initial job status")
    message: str = Field(..., description="Status message")
    estimated_duration_seconds: float = Field(..., description="Estimated completion time")


class Generation(BaseModel):
    """A generation in the optimization process."""
    generation_number: int = Field(..., description="Generation number")
    best_variant: PromptVariant = Field(..., description="Best variant in this generation")
    avg_score: float = Field(..., description="Average score across population")
    max_score: float = Field(..., description="Maximum score in population")
    timestamp: float = Field(..., description="Generation timestamp")


class OptimizationResult(BaseModel):
    """Result from prompt optimization."""
    best_prompt: str = Field(..., description="Best optimized prompt")
    best_variant: PromptVariant = Field(..., description="Best variant details")
    all_variants: List[PromptVariant] = Field(..., description="All tested variants")
    generations: List[Generation] = Field(..., description="Generation history")
    improvement: float = Field(..., description="Improvement over baseline")
    converged: bool = Field(..., description="Whether optimization converged")
    total_evaluations: int = Field(..., description="Total number of evaluations")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class JobStatusResponse(BaseModel):
    """Status of an optimization job."""
    job_id: str = Field(..., description="Job ID")
    status: JobStatus = Field(..., description="Current status")
    progress: float = Field(..., ge=0, le=1, description="Progress (0-1)")
    current_generation: Optional[int] = Field(None, description="Current generation number")
    total_generations: Optional[int] = Field(None, description="Total generations planned")
    best_score_so_far: Optional[float] = Field(None, description="Best score achieved so far")
    started_at: float = Field(..., description="Start timestamp")
    estimated_completion: Optional[float] = Field(None, description="Estimated completion timestamp")
    error: Optional[str] = Field(None, description="Error message if failed")


class JobResultsResponse(BaseModel):
    """Results of a completed optimization job."""
    job_id: str = Field(..., description="Job ID")
    status: JobStatus = Field(..., description="Job status")
    result: Optional[OptimizationResult] = Field(None, description="Optimization result (if completed)")
    started_at: float = Field(..., description="Start timestamp")
    completed_at: Optional[float] = Field(None, description="Completion timestamp")
    duration_seconds: Optional[float] = Field(None, description="Total duration")


# In-memory storage for optimization jobs
optimization_jobs: Dict[str, Dict[str, Any]] = {}
job_counter = 0


# Dependency injection
def get_optimization_service(request: Request):
    """Get prompt optimization service from app state."""
    # For now, return None. In production, return request.app.state.services.optimization_service
    return None


# Helper functions

async def run_optimization(job_id: str, request_data: OptimizeRequest):
    """Run optimization in background."""
    import time

    try:
        job = optimization_jobs[job_id]
        job["status"] = JobStatus.RUNNING
        job["started_at"] = time.time()

        config = request_data.config
        generations_data = []
        all_variants = []

        # Simulate optimization process
        for gen in range(config.generations):
            job["current_generation"] = gen + 1
            job["progress"] = (gen + 1) / config.generations

            # Create variants for this generation
            gen_variants = []
            for i in range(config.population_size):
                variant_id = f"gen{gen}_var{i}"

                # Generate variant (in production, use actual mutation/crossover)
                template = request_data.base_prompt
                if config.method == OptimizationMethod.GENETIC:
                    # Simulate genetic variation
                    template = template.replace("{question}", "{input}")

                # Simulate evaluation
                score = 0.6 + (gen * 0.05) + (i * 0.01)
                score = min(score, 1.0)

                variant = PromptVariant(
                    id=variant_id,
                    template=template,
                    parameters={"generation": gen, "variant": i},
                    score=score,
                    metadata={"method": config.method.value}
                )

                gen_variants.append(variant)
                all_variants.append(variant)

            # Find best in generation
            best_variant = max(gen_variants, key=lambda v: v.score or 0)
            avg_score = sum(v.score or 0 for v in gen_variants) / len(gen_variants)
            max_score = best_variant.score or 0

            generation = Generation(
                generation_number=gen + 1,
                best_variant=best_variant,
                avg_score=avg_score,
                max_score=max_score,
                timestamp=time.time()
            )
            generations_data.append(generation)

            job["best_score_so_far"] = max_score

            # Check early stopping
            if config.early_stopping and max_score >= config.target_score:
                logger.info(f"Job {job_id} reached target score {config.target_score}")
                break

            # Simulate work
            await asyncio.sleep(0.2)

        # Create final result
        best_overall = max(all_variants, key=lambda v: v.score or 0)
        baseline_score = 0.6  # Assumed baseline
        improvement = (best_overall.score or 0) - baseline_score

        result = OptimizationResult(
            best_prompt=best_overall.template,
            best_variant=best_overall,
            all_variants=all_variants,
            generations=generations_data,
            improvement=improvement,
            converged=generations_data[-1].max_score >= config.target_score,
            total_evaluations=len(all_variants),
            metadata={
                "method": config.method.value,
                "metric": config.evaluation_metric.value,
                "final_score": best_overall.score
            }
        )

        # Update job
        job["status"] = JobStatus.COMPLETED
        job["result"] = result
        job["completed_at"] = time.time()
        job["progress"] = 1.0

        logger.info(f"Job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        job["status"] = JobStatus.FAILED
        job["error"] = str(e)
        job["completed_at"] = time.time()


# API Endpoints

@router.post("/optimize", response_model=OptimizeResponse)
async def optimize_prompt(
    request: OptimizeRequest,
    service=Depends(get_optimization_service)
) -> OptimizeResponse:
    """
    Start prompt optimization job.

    This endpoint initiates an asynchronous optimization process that will:
    1. Generate prompt variants based on the selected method
    2. Evaluate each variant against test cases
    3. Iteratively improve prompts over multiple generations
    4. Return the best optimized prompt

    Supported optimization methods:
    - GENETIC: Uses genetic algorithm with mutation and crossover
    - AB_TEST: A/B testing of different variants
    - REINFORCEMENT: Reinforcement learning approach
    - GRID_SEARCH: Systematic parameter search
    - BAYESIAN: Bayesian optimization

    The job runs asynchronously. Use the returned job_id to check status
    and retrieve results.

    Example:
        ```python
        response = await client.post("/api/prompt-optimization/optimize", json={
            "base_prompt": "Solve: {question}",
            "task_description": "Math problem solver",
            "test_cases": [
                {"input": "2+2", "expected_output": "4"}
            ],
            "config": {
                "method": "genetic",
                "population_size": 10,
                "generations": 5
            }
        })
        job_id = response.job_id
        ```
    """
    global job_counter
    import time

    try:
        # Generate job ID
        job_counter += 1
        job_id = f"opt_{job_counter}"

        # Estimate duration
        estimated_duration = (
            request.config.population_size *
            request.config.generations *
            0.5  # seconds per evaluation
        )

        # Create job
        optimization_jobs[job_id] = {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "progress": 0.0,
            "current_generation": None,
            "total_generations": request.config.generations,
            "best_score_so_far": None,
            "started_at": None,
            "completed_at": None,
            "estimated_completion": time.time() + estimated_duration,
            "request": request.model_dump(),
            "result": None,
            "error": None
        }

        # Start optimization in background
        asyncio.create_task(run_optimization(job_id, request))

        logger.info(
            f"Started optimization job {job_id} using {request.config.method.value} "
            f"with {request.config.population_size} variants, {request.config.generations} generations"
        )

        return OptimizeResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            message=f"Optimization job started with ID {job_id}",
            estimated_duration_seconds=estimated_duration
        )

    except Exception as e:
        logger.error(f"Error starting optimization: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start optimization: {str(e)}"
        )


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_optimization_status(
    job_id: str,
    service=Depends(get_optimization_service)
) -> JobStatusResponse:
    """
    Get status of an optimization job.

    Returns current status, progress, and best score achieved so far.

    Example:
        ```python
        status = await client.get("/api/prompt-optimization/status/opt_1")
        print(f"Progress: {status.progress * 100}%")
        print(f"Best score: {status.best_score_so_far}")
        ```
    """
    try:
        if job_id not in optimization_jobs:
            raise HTTPException(
                status_code=404,
                detail=f"Job '{job_id}' not found"
            )

        job = optimization_jobs[job_id]

        return JobStatusResponse(
            job_id=job_id,
            status=job["status"],
            progress=job["progress"],
            current_generation=job["current_generation"],
            total_generations=job["total_generations"],
            best_score_so_far=job["best_score_so_far"],
            started_at=job["started_at"] or 0,
            estimated_completion=job["estimated_completion"],
            error=job["error"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get job status: {str(e)}"
        )


@router.get("/results/{job_id}", response_model=JobResultsResponse)
async def get_optimization_results(
    job_id: str,
    service=Depends(get_optimization_service)
) -> JobResultsResponse:
    """
    Get results of a completed optimization job.

    Returns the complete optimization results including:
    - Best optimized prompt
    - All tested variants
    - Generation history
    - Performance metrics

    The job must be in COMPLETED status to return results.

    Example:
        ```python
        results = await client.get("/api/prompt-optimization/results/opt_1")
        if results.status == "completed":
            print(f"Best prompt: {results.result.best_prompt}")
            print(f"Score: {results.result.best_variant.score}")
            print(f"Improvement: {results.result.improvement}")
        ```
    """
    try:
        if job_id not in optimization_jobs:
            raise HTTPException(
                status_code=404,
                detail=f"Job '{job_id}' not found"
            )

        job = optimization_jobs[job_id]

        # Calculate duration if completed
        duration = None
        if job["completed_at"] and job["started_at"]:
            duration = job["completed_at"] - job["started_at"]

        return JobResultsResponse(
            job_id=job_id,
            status=job["status"],
            result=job["result"],
            started_at=job["started_at"] or 0,
            completed_at=job["completed_at"],
            duration_seconds=duration
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job results: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get job results: {str(e)}"
        )


@router.delete("/jobs/{job_id}")
async def cancel_optimization_job(
    job_id: str,
    service=Depends(get_optimization_service)
):
    """
    Cancel a running optimization job.

    Cancels a job that is currently running or pending.
    Completed or failed jobs cannot be cancelled.

    Example:
        ```python
        response = await client.delete("/api/prompt-optimization/jobs/opt_1")
        ```
    """
    try:
        if job_id not in optimization_jobs:
            raise HTTPException(
                status_code=404,
                detail=f"Job '{job_id}' not found"
            )

        job = optimization_jobs[job_id]

        if job["status"] in [JobStatus.COMPLETED, JobStatus.FAILED]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel job in {job['status'].value} status"
            )

        job["status"] = JobStatus.CANCELLED
        import time
        job["completed_at"] = time.time()

        logger.info(f"Cancelled optimization job {job_id}")

        return {
            "success": True,
            "job_id": job_id,
            "message": f"Job {job_id} cancelled successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel job: {str(e)}"
        )


@router.get("/jobs")
async def list_optimization_jobs(
    status: Optional[JobStatus] = None,
    limit: int = 50,
    service=Depends(get_optimization_service)
):
    """
    List all optimization jobs.

    Args:
        status: Filter by job status (optional)
        limit: Maximum number of jobs to return

    Returns:
        List of optimization jobs

    Example:
        ```python
        jobs = await client.get("/api/prompt-optimization/jobs?status=completed&limit=10")
        ```
    """
    try:
        jobs = list(optimization_jobs.values())

        # Filter by status if specified
        if status:
            jobs = [j for j in jobs if j["status"] == status]

        # Sort by started_at (newest first)
        jobs.sort(key=lambda j: j["started_at"] or 0, reverse=True)

        # Apply limit
        jobs = jobs[:limit]

        return {
            "jobs": jobs,
            "total": len(jobs),
            "filtered_by_status": status.value if status else None
        }

    except Exception as e:
        logger.error(f"Error listing jobs: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list jobs: {str(e)}"
        )
