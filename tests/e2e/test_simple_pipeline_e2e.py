"""
End-to-end test using the actual simple pipeline
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add the parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from tests.pipeline_runner import run_with_new_schema, create_pipeline_from_json
import json


@pytest.mark.asyncio
async def test_simple_pipeline_e2e():
    """Test the complete simple pipeline from start to finish"""

    # Load the actual simple pipeline config
    pipeline_file = Path(__file__).parent.parent / "pipelines" / "simple_pipeline" / "pipeline.json"

    with open(pipeline_file, 'r') as f:
        pipeline_config = json.load(f)

    input_data = {"topic": "e2e_test"}

    # Create the pipeline
    pipeline = create_pipeline_from_json(pipeline_config)

    # Run the pipeline directly
    result = await run_with_new_schema(pipeline, pipeline_config, input_data, None)

    # Verify the result structure
    assert result is not None
    assert isinstance(result, dict)
    assert "input" in result
    assert "steps" in result
    assert "output" in result

    # Verify the topic transformation chain
    assert result["input"]["topic"] == "e2e_test"
    assert result["steps"]["step1"]["topic"] == "PROCESSED_E2E_TEST"
    assert result["steps"]["step2"]["topic"] == "processed_e2e_test_enriched"
    assert result["steps"]["step3"]["topic"] == "FINAL_dehcirne_tset_e2e_dessecorp"

    # Verify final output
    assert result["output"]["topic"] == "FINAL_dehcirne_tset_e2e_dessecorp"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])