"""Benchmark service using ia_modules library"""

import logging
import uuid
from typing import Dict, Any, Optional, List
from ia_modules.benchmarking import (
    BenchmarkRunner,
    BenchmarkConfig,
    BenchmarkComparator
)

logger = logging.getLogger(__name__)


class BenchmarkService:
    """Service for benchmarking using ia_modules"""

    def __init__(self, pipeline_service, db_manager):
        self.pipeline_service = pipeline_service
        self.db_manager = db_manager
        self.results = {}
        self.runner = BenchmarkRunner()
        self.comparator = BenchmarkComparator()

    async def run_benchmark(
        self,
        pipeline_id: str,
        iterations: int = 10,
        input_data: Dict[str, Any] = {},
        warmup_iterations: int = 2
    ) -> Dict[str, Any]:
        """Run performance benchmark for a pipeline"""
        try:
            # Get pipeline
            pipeline = await self.pipeline_service.get_pipeline(pipeline_id)
            if not pipeline:
                raise ValueError(f"Pipeline {pipeline_id} not found")

            # Create benchmark config
            config = BenchmarkConfig(
                iterations=iterations,
                warmup_iterations=warmup_iterations,
                measure_memory=True,
                measure_cpu=True
            )

            # Get pipeline object from service
            pipeline_obj = self.pipeline_service.pipelines.get(pipeline_id)
            if not pipeline_obj:
                raise ValueError(f"Pipeline {pipeline_id} not loaded")

            # Run benchmark using library
            result = await self.runner.run(
                pipeline=pipeline_obj,
                input_data=input_data,
                config=config
            )

            # Store result
            result_id = str(uuid.uuid4())
            self.results[result_id] = {
                "result_id": result_id,
                "pipeline_id": pipeline_id,
                "pipeline_name": pipeline.get("name", "Unknown"),
                "iterations": iterations,
                "warmup_iterations": warmup_iterations,
                "total_time": result.total_time,
                "average_time": result.average_time,
                "median_time": result.median_time,
                "min_time": result.min_time,
                "max_time": result.max_time,
                "p95_time": result.p95_time,
                "p99_time": result.p99_time,
                "std_dev": result.std_dev,
                "throughput": result.throughput,
                "memory_per_operation_mb": result.memory_per_operation_mb,
                "cpu_per_operation_percent": result.cpu_per_operation_percent,
                "items_processed": result.items_processed,
                "api_calls_count": result.api_calls_count,
                "estimated_cost_usd": result.estimated_cost_usd,
                "timestamp": result.timestamp.isoformat() if hasattr(result, 'timestamp') else None
            }

            logger.info(f"Completed benchmark {result_id} for pipeline {pipeline_id}")
            return self.results[result_id]

        except Exception as e:
            logger.error(f"Failed to run benchmark: {e}")
            raise

    async def compare_pipelines(
        self,
        pipeline_ids: List[str],
        iterations: int = 10,
        input_data: Dict[str, Any] = {}
    ) -> Dict[str, Any]:
        """Compare performance of multiple pipelines"""
        try:
            # Run benchmarks for each pipeline
            results = []
            for pipeline_id in pipeline_ids:
                result = await self.run_benchmark(
                    pipeline_id=pipeline_id,
                    iterations=iterations,
                    input_data=input_data
                )
                results.append(result)

            # Use library comparator to analyze results
            comparison = {
                "pipelines": pipeline_ids,
                "results": results,
                "winner": None,
                "metrics": {}
            }

            # Find fastest pipeline
            fastest = min(results, key=lambda r: r["average_time"])
            comparison["winner"] = fastest["pipeline_id"]

            # Calculate relative performance
            for result in results:
                relative_perf = result["average_time"] / fastest["average_time"]
                result["relative_performance"] = relative_perf

            logger.info(f"Completed comparison of {len(pipeline_ids)} pipelines")
            return comparison

        except Exception as e:
            logger.error(f"Failed to compare pipelines: {e}")
            raise

    async def list_results(
        self,
        pipeline_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """List benchmark results"""
        results = list(self.results.values())

        if pipeline_id:
            results = [r for r in results if r["pipeline_id"] == pipeline_id]

        # Sort by timestamp (newest first)
        results = sorted(
            results,
            key=lambda r: r.get("timestamp", ""),
            reverse=True
        )

        return results[:limit]

    async def get_result(self, result_id: str) -> Optional[Dict[str, Any]]:
        """Get benchmark result details"""
        return self.results.get(result_id)

    async def delete_result(self, result_id: str) -> bool:
        """Delete a benchmark result"""
        if result_id not in self.results:
            return False

        del self.results[result_id]
        logger.info(f"Deleted benchmark result {result_id}")
        return True
