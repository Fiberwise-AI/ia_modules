"""
Test that pattern steps work in actual pipeline execution

This tests the integration between pattern steps and the graph pipeline runner.
"""

import pytest
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ia_modules.pipeline.graph_pipeline_runner import GraphPipelineRunner
from ia_modules.pipeline.llm_provider_service import LLMResponse, LLMProvider
from backend.pipelines.pattern_steps import ReflectionStep, PlanningStep, ToolUseStep


class MockLLMService:
    """Mock LLM service for testing"""

    def __init__(self):
        self.providers = {"test": object()}

    async def generate_completion(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000, **kwargs):
        """Mock completion"""
        # Determine response based on prompt content
        if "critique" in prompt.lower() or "score" in prompt.lower():
            content = "Score: 8/10\nGood quality, minor improvements possible"
        elif "plan" in prompt.lower() or "step" in prompt.lower():
            content = "Step 1: First action\nExpected: Outcome 1\n\nStep 2: Second action\nExpected: Outcome 2"
        elif "valid" in prompt.lower():
            content = "VALID: The plan is comprehensive and achievable"
        elif "tool" in prompt.lower() or "calculator" in prompt.lower():
            content = "calculator, search"
        else:
            content = "Mock response for the given prompt"

        return LLMResponse(
            content=content,
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        )

    def register_provider(self, name: str, provider, api_key: str, model: str, is_default: bool = False):
        """Mock provider registration"""
        pass


@pytest.mark.asyncio
async def test_pipeline_json_structure_validation():
    """Test that pipeline JSON validates correctly"""

    pipeline_config = {
        "name": "test_pipeline",
        "version": "1.0.0",
        "steps": [
            {
                "id": "test_step",
                "name": "Test Step",
                "step_class": "ReflectionStep",
                "module": "showcase_app.backend.pipelines.pattern_steps",
                "config": {
                    "initial_output": "Test",
                    "criteria": {"quality": "High"},
                    "max_iterations": 1
                }
            }
        ],
        "flow": {
            "start_at": "test_step",
            "paths": [
                {
                    "from": "test_step",
                    "to": "end_with_success",
                    "condition": {"type": "always"}
                }
            ]
        }
    }

    GraphPipelineRunner()

    # Should validate without errors
    from ia_modules.pipeline.graph_pipeline_runner import PipelineConfig
    config = PipelineConfig(**pipeline_config)

    assert config.name == "test_pipeline"
    assert len(config.steps) == 1
    assert config.steps[0].id == "test_step"
    assert config.steps[0].step_class == "ReflectionStep"
    assert config.flow.start_at == "test_step"


@pytest.mark.asyncio
async def test_pipeline_json_missing_required_fields():
    """Test that pipeline JSON fails validation with missing fields"""

    from ia_modules.pipeline.graph_pipeline_runner import PipelineConfig

    # Missing 'id' field
    with pytest.raises(Exception):
        PipelineConfig(**{
            "name": "test",
            "steps": [
                {
                    # Missing "id"
                    "name": "Test",
                    "step_class": "TestStep",
                    "module": "test.module",
                    "config": {}
                }
            ],
            "flow": {"start_at": "test", "paths": []}
        })

    # Missing 'flow' field
    with pytest.raises(Exception):
        PipelineConfig(**{
            "name": "test",
            "steps": [
                {
                    "id": "test",
                    "name": "Test",
                    "step_class": "TestStep",
                    "module": "test.module",
                    "config": {}
                }
            ]
            # Missing "flow"
        })


@pytest.mark.asyncio
async def test_agentic_patterns_demo_json_valid():
    """Test that the demo pipeline JSON is valid"""

    # Load the actual demo pipeline
    demo_file = Path(__file__).parent.parent / "backend" / "pipelines" / "agentic_patterns_demo.json"

    if not demo_file.exists():
        pytest.skip("Demo pipeline file not found")

    with open(demo_file, 'r') as f:
        pipeline_config = json.load(f)

    # Should validate
    from ia_modules.pipeline.graph_pipeline_runner import PipelineConfig
    config = PipelineConfig(**pipeline_config)

    assert config.name == "agentic_patterns_demo"
    assert len(config.steps) == 3
    assert config.steps[0].id == "plan_research"
    assert config.steps[0].step_class == "PlanningStep"
    assert config.steps[0].module == "showcase_app.backend.pipelines.pattern_steps"


@pytest.mark.asyncio
async def test_pattern_step_execution_in_pipeline():
    """Test that pattern steps execute correctly in pipeline context"""

    # Inject mock LLM into pattern steps

    # Temporarily replace LLM initialization
    original_init = ReflectionStep._init_llm

    def mock_init(self):
        self.llm_service = MockLLMService()

    ReflectionStep._init_llm = mock_init

    try:
        # Create simple pipeline with reflection step
        pipeline_config = {
            "name": "reflection_test",
            "version": "1.0.0",
            "steps": [
                {
                    "id": "improve",
                    "name": "Improve Text",
                    "step_class": "ReflectionStep",
                    "module": "showcase_app.backend.pipelines.pattern_steps",
                    "config": {
                        "initial_output": "This is a test",
                        "criteria": {"quality": "Must be high quality"},
                        "max_iterations": 2
                    }
                }
            ],
            "flow": {
                "start_at": "improve",
                "paths": [
                    {
                        "from": "improve",
                        "to": "end_with_success",
                        "condition": {"type": "always"}
                    }
                ]
            }
        }

        runner = GraphPipelineRunner()
        result = await runner.run_pipeline_from_json(pipeline_config, {})

        # Should have reflection results
        assert "improve" in result
        assert "final_output" in result["improve"]
        assert "final_score" in result["improve"]
        assert result["improve"]["final_score"] >= 6.0

    finally:
        # Restore original
        ReflectionStep._init_llm = original_init


@pytest.mark.asyncio
async def test_multi_pattern_pipeline():
    """Test pipeline with multiple pattern types"""

    # Mock all pattern LLM services
    original_reflection_init = ReflectionStep._init_llm
    original_planning_init = PlanningStep._init_llm
    original_tool_init = ToolUseStep._init_llm

    def mock_init(self):
        self.llm_service = MockLLMService()

    ReflectionStep._init_llm = mock_init
    PlanningStep._init_llm = mock_init
    ToolUseStep._init_llm = mock_init

    try:
        pipeline_config = {
            "name": "multi_pattern_test",
            "version": "1.0.0",
            "steps": [
                {
                    "id": "plan",
                    "name": "Plan",
                    "step_class": "PlanningStep",
                    "module": "showcase_app.backend.pipelines.pattern_steps",
                    "config": {
                        "goal": "Test goal",
                        "constraints": []
                    }
                },
                {
                    "id": "reflect",
                    "name": "Reflect",
                    "step_class": "ReflectionStep",
                    "module": "showcase_app.backend.pipelines.pattern_steps",
                    "config": {
                        "initial_output": "Test output",
                        "criteria": {"quality": "High"},
                        "max_iterations": 1
                    }
                }
            ],
            "flow": {
                "start_at": "plan",
                "paths": [
                    {
                        "from": "plan",
                        "to": "reflect",
                        "condition": {"type": "always"}
                    },
                    {
                        "from": "reflect",
                        "to": "end_with_success",
                        "condition": {"type": "always"}
                    }
                ]
            }
        }

        runner = GraphPipelineRunner()
        result = await runner.run_pipeline_from_json(pipeline_config, {})

        # Should have both results
        assert "plan" in result
        assert "reflect" in result
        assert "plan" in result["plan"]
        assert "final_output" in result["reflect"]

    finally:
        ReflectionStep._init_llm = original_reflection_init
        PlanningStep._init_llm = original_planning_init
        ToolUseStep._init_llm = original_tool_init


@pytest.mark.asyncio
async def test_pipeline_json_field_names():
    """Test that correct field names are used (id, step_class, module)"""

    from ia_modules.pipeline.graph_pipeline_runner import PipelineConfig

    # Correct structure
    config = PipelineConfig(**{
        "name": "test",
        "version": "1.0.0",
        "steps": [
            {
                "id": "my_step",              # ✅ Correct: id
                "name": "My Step",             # ✅ Correct: name
                "step_class": "MyStep",        # ✅ Correct: step_class
                "module": "my.module",         # ✅ Correct: module
                "config": {}
            }
        ],
        "flow": {
            "start_at": "my_step",
            "paths": [
                {
                    "from": "my_step",         # ✅ Correct: from
                    "to": "end_with_success",   # ✅ Correct: to
                    "condition": {"type": "always"}
                }
            ]
        }
    })

    assert config.steps[0].id == "my_step"
    assert config.steps[0].step_class == "MyStep"
    assert config.steps[0].module == "my.module"
    assert config.flow.paths[0].from_step == "my_step"
    assert config.flow.paths[0].to_step == "end_with_success"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
