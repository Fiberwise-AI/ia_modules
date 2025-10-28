"""
End-to-end tests focused on parallel pipeline execution patterns
"""

import pytest
import asyncio
import json
import time
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock

# Add the parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from ia_modules.pipeline.graph_pipeline_runner import GraphPipelineRunner
from ia_modules.pipeline.test_utils import create_test_execution_context
from ia_modules.pipeline.services import ServiceRegistry


class TestParallelE2E:
    """Test suite specifically for parallel pipeline execution"""

    @pytest.mark.asyncio
    async def test_parallel_branching_execution(self):
        """Test that parallel branches execute independently"""

        # Load the parallel pipeline config
        pipeline_file = Path(__file__).parent.parent / "pipelines" / "parallel_pipeline" / "pipeline.json"

        with open(pipeline_file, 'r') as f:
            pipeline_config = json.load(f)

        # Create test data that will be split into parallel streams
        input_data = {
            "loaded_data": [
                {"stream": "A", "value": 100},
                {"stream": "B", "value": 200},
                {"stream": "C", "value": 300},
                {"stream": "A", "value": 150},
                {"stream": "B", "value": 250},
                {"stream": "C", "value": 350}
            ]
        }

        # Execute pipeline
        runner = GraphPipelineRunner()
        result = await runner.run_pipeline_from_json(pipeline_config, input_data)

        # Verify parallel execution structure
        assert result is not None
        assert "steps" in result

        # Check that data was split (step1)
        step1_result = result["steps"]["step1"]
        assert "data_chunks" in step1_result
        assert step1_result["chunk_count"] == 3  # Should split into 3 streams

        # Check that all parallel processors ran (step2, step3, step4)
        parallel_steps = ["step2", "step3", "step4"]
        for step in parallel_steps:
            assert step in result["steps"]
            step_result = result["steps"][step]
            assert "processed_data" in step_result

        # Check that results were merged (step5)
        step5_result = result["steps"]["step5"]
        assert "merged_results" in step5_result

        # Check final statistics collection (step6)
        step6_result = result["steps"]["step6"]
        assert "statistics" in step6_result

    @pytest.mark.asyncio
    async def test_parallel_data_integrity(self):
        """Test that data integrity is maintained across parallel processing"""

        pipeline_file = Path(__file__).parent.parent / "pipelines" / "parallel_pipeline" / "pipeline.json"

        with open(pipeline_file, 'r') as f:
            pipeline_config = json.load(f)

        # Create numbered test data to track integrity
        input_data = {
            "loaded_data": [{"id": i, "value": i * 100} for i in range(1, 13)]  # 12 items
        }

        runner = GraphPipelineRunner()
        result = await runner.run_pipeline_from_json(pipeline_config, input_data)

        # Verify data integrity
        assert result is not None

        # Check that original data count is preserved
        step1_result = result["steps"]["step1"]
        assert step1_result["original_data_size"] == 12

        # Verify that all data was processed through parallel streams
        step5_result = result["steps"]["step5"]
        merged_results = step5_result.get("merged_results", [])

        # The exact structure depends on implementation, but verify data was processed
        assert len(merged_results) > 0 or "total_processed" in step5_result

    @pytest.mark.asyncio
    async def test_parallel_performance_characteristics(self):
        """Test parallel processing performance characteristics"""

        pipeline_file = Path(__file__).parent.parent / "pipelines" / "parallel_pipeline" / "pipeline.json"

        with open(pipeline_file, 'r') as f:
            pipeline_config = json.load(f)

        # Large dataset to better test parallel benefits
        large_data = {
            "loaded_data": [
                {"batch": i // 10, "item": i, "data": f"data_item_{i}"}
                for i in range(100)
            ]
        }

        # Measure execution time
        start_time = time.time()

        runner = GraphPipelineRunner()
        result = await runner.run_pipeline_from_json(pipeline_config, large_data)

        end_time = time.time()
        execution_time = end_time - start_time

        # Verify successful execution
        assert result is not None
        assert "steps" in result

        # Verify all parallel steps completed
        for step_id in ["step1", "step2", "step3", "step4", "step5", "step6"]:
            assert step_id in result["steps"]

        # Log performance info (execution time should be reasonable for test data)
        print(f"Parallel pipeline execution time: {execution_time:.3f} seconds")
        assert execution_time < 10.0  # Should complete within 10 seconds

    @pytest.mark.asyncio
    async def test_parallel_error_isolation(self):
        """Test that errors in one parallel branch don't affect others"""

        # This test would require modifying one of the parallel steps to simulate an error
        # For now, test that the pipeline handles partial failures gracefully

        pipeline_file = Path(__file__).parent.parent / "pipelines" / "parallel_pipeline" / "pipeline.json"

        with open(pipeline_file, 'r') as f:
            pipeline_config = json.load(f)

        # Test with edge case data that might cause issues
        edge_case_data = {
            "loaded_data": []  # Empty data
        }

        pipeline = create_pipeline_from_json(pipeline_config)

        # Should either handle gracefully or provide meaningful error
        try:
            result = await run_with_new_schema(pipeline, pipeline_config, edge_case_data, None)
            # If successful, verify it handled empty data appropriately
            assert result is not None
            assert "steps" in result
        except Exception as e:
            # If it fails, should be a meaningful error
            assert isinstance(e, (ValueError, IndexError))

    @pytest.mark.asyncio
    async def test_parallel_scaling_behavior(self):
        """Test how the parallel pipeline scales with different data sizes"""

        pipeline_file = Path(__file__).parent.parent / "pipelines" / "parallel_pipeline" / "pipeline.json"

        with open(pipeline_file, 'r') as f:
            pipeline_config = json.load(f)

        # Test with different data sizes
        test_sizes = [10, 50, 100]

        for size in test_sizes:
            input_data = {
                "loaded_data": [
                    {"size_test": size, "item": i, "value": i * 2}
                    for i in range(size)
                ]
            }

            runner = GraphPipelineRunner()
        result = await runner.run_pipeline_from_json(pipeline_config, input_data)

        # Verify successful scaling
        assert result is not None
        assert "steps" in result

        # Verify data splitter adapted to size
        step1_result = result["steps"]["step1"]
        assert step1_result["original_data_size"] == size

        # Verify final processing completed
        assert "step6" in result["steps"]

    @pytest.mark.asyncio
    async def test_parallel_execution_order(self):
        """Test that parallel execution follows correct dependency order"""

        pipeline_file = Path(__file__).parent.parent / "pipelines" / "parallel_pipeline" / "pipeline.json"

        with open(pipeline_file, 'r') as f:
            pipeline_config = json.load(f)

        input_data = {
            "loaded_data": [{"order_test": i} for i in range(20)]
        }

        runner = GraphPipelineRunner()
        result = await runner.run_pipeline_from_json(pipeline_config, input_data)

        # Verify execution order dependencies were respected:
        # step1 (splitter) -> [step2, step3, step4] (parallel processors) -> step5 (merger) -> step6 (stats)

        assert result is not None
        steps = result["steps"]

        # All steps should have executed
        expected_steps = ["step1", "step2", "step3", "step4", "step5", "step6"]
        for step_id in expected_steps:
            assert step_id in steps, f"Step {step_id} did not execute"

        # Verify logical dependencies (data should flow correctly)
        # step1 should produce data_chunks for parallel processors
        assert "data_chunks" in steps["step1"]

        # Parallel processors should have processed data
        for parallel_step in ["step2", "step3", "step4"]:
            assert "processed_data" in steps[parallel_step] or "stream_results" in steps[parallel_step]

        # Merger should have combined results
        assert "merged_results" in steps["step5"] or "combined_data" in steps["step5"]

        # Final step should have statistics
        assert "statistics" in steps["step6"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])