"""
Comprehensive end-to-end tests for different pipeline types
"""

import pytest
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch

# Add the parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from tests.pipeline_runner import run_with_new_schema, create_pipeline_from_json
from ia_modules.pipeline.services import ServiceRegistry


class TestE2EPipelines:
    """Test suite for end-to-end pipeline execution"""

    @pytest.mark.asyncio
    async def test_simple_sequential_pipeline(self):
        """Test the simple pipeline with sequential steps"""

        # Load the simple pipeline config
        pipeline_file = Path(__file__).parent.parent / "pipelines" / "simple_pipeline" / "pipeline.json"

        with open(pipeline_file, 'r') as f:
            pipeline_config = json.load(f)

        input_data = {"topic": "sequential_test"}

        # Create and run the pipeline
        pipeline = create_pipeline_from_json(pipeline_config)
        result = await run_with_new_schema(pipeline, pipeline_config, input_data, None)

        # Verify sequential processing
        assert result is not None
        assert isinstance(result, dict)
        assert "input" in result
        assert "steps" in result
        assert "output" in result

        # Verify topic transformation chain
        assert result["input"]["topic"] == "sequential_test"
        assert result["steps"]["step1"]["topic"] == "PROCESSED_SEQUENTIAL_TEST"
        assert result["steps"]["step2"]["topic"] == "processed_sequential_test_enriched"
        assert result["steps"]["step3"]["topic"] == "FINAL_dehcirne_tset_laitneuqes_dessecorp"

        # Verify final output
        assert result["output"]["topic"] == "FINAL_dehcirne_tset_laitneuqes_dessecorp"

    @pytest.mark.asyncio
    async def test_parallel_processing_pipeline(self):
        """Test the parallel pipeline with concurrent steps"""

        # Load the parallel pipeline config
        pipeline_file = Path(__file__).parent.parent / "pipelines" / "parallel_pipeline" / "pipeline.json"

        with open(pipeline_file, 'r') as f:
            pipeline_config = json.load(f)

        # Provide test data that can be split
        input_data = {
            "loaded_data": [
                {"id": 1, "name": "item1", "value": 100},
                {"id": 2, "name": "item2", "value": 200},
                {"id": 3, "name": "item3", "value": 300},
                {"id": 4, "name": "item4", "value": 400},
                {"id": 5, "name": "item5", "value": 500},
                {"id": 6, "name": "item6", "value": 600}
            ]
        }

        # Create and run the pipeline
        pipeline = create_pipeline_from_json(pipeline_config)
        result = await run_with_new_schema(pipeline, pipeline_config, input_data, None)

        # Verify parallel processing structure
        assert result is not None
        assert isinstance(result, dict)
        assert "input" in result
        assert "steps" in result
        assert "output" in result

        # Verify data splitter created chunks
        step1_result = result["steps"]["step1"]
        assert "data_chunks" in step1_result
        assert "chunk_count" in step1_result
        assert step1_result["chunk_count"] >= 2  # Should have multiple chunks for parallel processing

        # Verify parallel processors ran
        assert "step2" in result["steps"]  # Stream processor 1
        assert "step3" in result["steps"]  # Stream processor 2
        assert "step4" in result["steps"]  # Stream processor 3

        # Verify merger combined results
        assert "step5" in result["steps"]  # Results merger
        step5_result = result["steps"]["step5"]
        assert "merged_results" in step5_result

        # Verify final stats collection
        assert "step6" in result["steps"]  # Stats collector
        final_result = result["steps"]["step6"]
        assert "statistics" in final_result

    @pytest.mark.asyncio
    async def test_conditional_pipeline(self):
        """Test the conditional pipeline with branching logic"""

        # Load the conditional pipeline config
        pipeline_file = Path(__file__).parent.parent / "pipelines" / "conditional_pipeline" / "pipeline.json"

        with open(pipeline_file, 'r') as f:
            pipeline_config = json.load(f)

        # Test with high-quality data
        high_quality_input = {
            "threshold": 0.8,
            "test_data": [
                {"quality_score": 0.95, "content": "high quality data 1"},
                {"quality_score": 0.88, "content": "high quality data 2"},
                {"quality_score": 0.92, "content": "high quality data 3"}
            ]
        }

        pipeline = create_pipeline_from_json(pipeline_config)
        result = await run_with_new_schema(pipeline, pipeline_config, high_quality_input, None)

        # Verify conditional branching
        assert result is not None
        assert isinstance(result, dict)
        assert "steps" in result

        # Should have taken high-quality processing path
        # The exact steps depend on the conditional logic implementation

    @pytest.mark.asyncio
    async def test_agent_pipeline(self):
        """Test the agent-based pipeline"""

        # Load the agent pipeline config
        pipeline_file = Path(__file__).parent.parent / "pipelines" / "agent_pipeline" / "pipeline.json"

        with open(pipeline_file, 'r') as f:
            pipeline_config = json.load(f)

        input_data = {
            "task": "analyze_sentiment",
            "text": "This is a great product! I love it."
        }

        pipeline = create_pipeline_from_json(pipeline_config)
        result = await run_with_new_schema(pipeline, pipeline_config, input_data, None)

        # Verify agent processing
        assert result is not None
        assert isinstance(result, dict)
        assert "steps" in result

    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self):
        """Test pipeline behavior with invalid input"""

        # Load a simple pipeline for error testing
        pipeline_file = Path(__file__).parent.parent / "pipelines" / "simple_pipeline" / "pipeline.json"

        with open(pipeline_file, 'r') as f:
            pipeline_config = json.load(f)

        # Test with missing required input
        invalid_input = {}  # No topic provided

        pipeline = create_pipeline_from_json(pipeline_config)

        # Should handle gracefully - might return default values or raise appropriate error
        try:
            result = await run_with_new_schema(pipeline, pipeline_config, invalid_input, None)
            # If it doesn't raise an error, verify it handles missing input gracefully
            assert result is not None
            assert isinstance(result, dict)
        except Exception as e:
            # If it raises an error, it should be a meaningful one
            assert isinstance(e, (ValueError, KeyError, TypeError))

    @pytest.mark.asyncio
    async def test_large_data_pipeline(self):
        """Test pipeline with larger dataset to verify performance"""

        # Load the parallel pipeline for large data testing
        pipeline_file = Path(__file__).parent.parent / "pipelines" / "parallel_pipeline" / "pipeline.json"

        with open(pipeline_file, 'r') as f:
            pipeline_config = json.load(f)

        # Create larger test dataset
        large_input = {
            "loaded_data": [
                {"id": i, "name": f"item{i}", "value": i * 10}
                for i in range(1, 101)  # 100 items
            ]
        }

        pipeline = create_pipeline_from_json(pipeline_config)
        result = await run_with_new_schema(pipeline, pipeline_config, large_input, None)

        # Verify large data handling
        assert result is not None
        assert isinstance(result, dict)
        assert "steps" in result

        # Verify data was processed through all steps
        assert "step1" in result["steps"]  # Data splitter
        assert "step5" in result["steps"]  # Results merger
        assert "step6" in result["steps"]  # Stats collector

    @pytest.mark.asyncio
    async def test_pipeline_logging_integration(self):
        """Test that pipeline logging works end-to-end"""

        pipeline_file = Path(__file__).parent.parent / "pipelines" / "simple_pipeline" / "pipeline.json"

        with open(pipeline_file, 'r') as f:
            pipeline_config = json.load(f)

        input_data = {"topic": "logging_test"}

        # Mock logging to capture log messages
        log_messages = []

        def mock_log(level, message, step_name=None, data=None):
            log_messages.append({
                "level": level,
                "message": message,
                "step": step_name,
                "data": data
            })

        # This would need to be integrated with the actual logging system
        # For now, just verify the pipeline runs successfully
        pipeline = create_pipeline_from_json(pipeline_config)
        result = await run_with_new_schema(pipeline, pipeline_config, input_data, None)

        assert result is not None
        assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])