"""
Integration tests for enhanced pipeline with advanced routing features
"""

import asyncio
from unittest.mock import Mock

import pytest

from ia_modules.pipeline.enhanced_pipeline import EnhancedPipeline
from ia_modules.pipeline.core import Step, Pipeline
from ia_modules.pipeline.services import ServiceRegistry
from ia_modules.pipeline.routing import RoutingContext


class IntegrationTestStep(Step):
    """Test step implementation"""
    
    async def work(self, data: dict) -> dict:
        return {"processed": True, "data": data, "step_name": self.name}


class TestAgentConditionEvaluator:
    """Mock agent condition evaluator for testing"""
    
    async def evaluate(self, context: RoutingContext) -> bool:
        # Simple mock that returns True for testing
        return True


@pytest.mark.asyncio
async def test_enhanced_pipeline_creation():
    """Test enhanced pipeline creation"""
    steps = [IntegrationTestStep("step1", {})]
    services = ServiceRegistry()
    
    # Test that EnhancedPipeline can be instantiated
    pipeline = EnhancedPipeline(steps, services)
    
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 1


@pytest.mark.asyncio
async def test_enhanced_pipeline_with_structure():
    """Test enhanced pipeline with graph structure"""
    steps = [IntegrationTestStep("step1", {})]
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
    
    services = ServiceRegistry()
    pipeline = EnhancedPipeline(steps, services, structure=structure)
    
    assert pipeline.has_flow_definition() is True


@pytest.mark.asyncio
async def test_enhanced_pipeline_execution():
    """Test enhanced pipeline execution with basic flow"""
    steps = [
        IntegrationTestStep("step1", {"param": "value1"}),
        IntegrationTestStep("step2", {"param": "value2"})
    ]
    
    structure = {
        "flow": {
            "start_at": "step1",
            "paths": [
                {
                    "from_step": "step1",
                    "to_step": "step2",
                    "condition": {"type": "always"}
                },
                {
                    "from_step": "step2",
                    "to_step": "end_with_success",
                    "condition": {"type": "always"}
                }
            ]
        }
    }
    
    services = ServiceRegistry()
    pipeline = EnhancedPipeline(steps, services, structure=structure)
    
    # Test execution
    result = await pipeline.run({"input": "test_data"})
    
    assert result is not None
    assert isinstance(result, dict)
    assert result["input"] == "test_data"
    assert result["processed"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
