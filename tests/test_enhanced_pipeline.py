"""
Unit tests for enhanced pipeline with advanced routing features
"""

import asyncio
from unittest.mock import Mock

import pytest

from ia_modules.pipeline.enhanced_pipeline import EnhancedPipeline
from ia_modules.pipeline.core import Step, Pipeline
from ia_modules.pipeline.services import ServiceRegistry


class TestStep(Step):
    """Test step implementation"""
    
    async def work(self, data: dict) -> dict:
        return {"test_result": "success", "input_data": data}


def test_enhanced_pipeline_creation():
    """Test enhanced pipeline creation"""
    steps = [TestStep("step1", {})]
    services = ServiceRegistry()
    
    # Test that EnhancedPipeline can be instantiated
    pipeline = EnhancedPipeline(steps, services)
    
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 1


def test_enhanced_pipeline_with_structure():
    """Test enhanced pipeline with graph structure"""
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
    
    services = ServiceRegistry()
    pipeline = EnhancedPipeline(steps, services, structure=structure)
    
    assert pipeline.has_flow_definition() is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
