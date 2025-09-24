"""
Unit tests for core pipeline components
"""

import asyncio
import json
from unittest.mock import Mock, patch
from pathlib import Path

import pytest

from ia_modules.pipeline.core import Step, Pipeline, TemplateParameterResolver, StepLogger
from ia_modules.pipeline.services import ServiceRegistry


class TestStep(Step):
    """Test step implementation"""
    
    async def work(self, data: dict) -> dict:
        return {"test_result": "success", "input_data": data}


def test_template_parameter_resolver():
    """Test template parameter resolution"""
    context = {
        "pipeline_input": {"business_type": "retail"},
        "steps": {"geocoder": {"result": {"city": "New York"}}},
        "parameters": {"custom_value": "test_value"}
    }
    
    # Test string resolution
    config = {
        "url": "{pipeline_input.business_type}",
        "city": "{steps.geocoder.result.city}",
        "value": "{parameters.custom_value}"
    }
    
    resolved = TemplateParameterResolver.resolve_parameters(config, context)
    
    assert resolved["url"] == "retail"
    assert resolved["city"] == "New York"
    assert resolved["value"] == "test_value"


def test_step_logger():
    """Test step logger functionality"""
    # Mock database manager
    mock_db = Mock()
    
    logger = StepLogger("test_step", 1, "test_job_id", mock_db)
    
    # Test logging start
    asyncio.run(logger.log_step_start({"input": "data"}))
    
    # Test logging completion
    asyncio.run(logger.log_step_complete({"result": "success"}))
    
    # Test logging error
    asyncio.run(logger.log_step_error(Exception("Test error")))


def test_step():
    """Test step execution"""
    step = TestStep("test_step", {"param": "value"})
    
    # Mock services
    services = ServiceRegistry()
    step.set_services(services)
    
    assert step.name == "test_step"
    assert step.config == {"param": "value"}


def test_pipeline_creation():
    """Test pipeline creation"""
    steps = [TestStep("step1", {}), TestStep("step2", {})]
    pipeline = Pipeline(steps)
    
    assert len(pipeline.steps) == 2
    assert pipeline.steps[0].name == "step1"
    assert pipeline.steps[1].name == "step2"


def test_pipeline_with_structure():
    """Test pipeline with graph structure"""
    steps = [TestStep("step1", {})]
    structure = {
        "flow": {
            "start_at": "step1",
            "paths": [
                {
                    "from_step": "step1",
                    "to_step": "end_with_success",
                    "condition": {"type": "always"}
                }
            ]
        }
    }
    
    pipeline = Pipeline(steps, structure=structure)
    
    assert pipeline.has_flow_definition() is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
