"""
Unit tests for graph pipeline runner
"""

import asyncio
from unittest.mock import Mock

import pytest

from ia_modules.pipeline.graph_pipeline_runner import GraphPipelineRunner, PipelineConfig
from ia_modules.pipeline.services import ServiceRegistry


def test_graph_pipeline_runner_creation():
    """Test graph pipeline runner creation"""
    services = ServiceRegistry()
    runner = GraphPipelineRunner(services)
    
    assert runner is not None
    assert runner.services == services


def test_pipeline_config_validation():
    """Test pipeline configuration validation"""
    # Test valid config
    config_dict = {
        "name": "Test Pipeline",
        "steps": [
            {
                "id": "step1",
                "name": "Step 1",
                "step_class": "TestStep",
                "module": "test.module"
            }
        ],
        "flow": {
            "start_at": "step1",
            "paths": []
        }
    }
    
    # This should not raise an exception
    config = PipelineConfig(**config_dict)
    
    assert config.name == "Test Pipeline"
    assert len(config.steps) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
