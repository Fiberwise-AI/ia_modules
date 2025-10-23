"""
Unit tests for pipeline runner
"""

import asyncio
from unittest.mock import Mock, patch
import json

import pytest

from ia_modules.pipeline.runner import (
    load_step_class,
    create_step_from_json,
    create_pipeline_from_json
)
from ia_modules.pipeline.core import Step


class MockStep(Step):
    """Mock step implementation for testing"""

    async def run(self, data: dict) -> dict:
        return {"result": "success", "input_data": data}


def test_load_step_class():
    """Test loading step class"""
    # This would normally load from a real module
    # For testing purposes, we'll mock the import
    with patch('importlib.import_module') as mock_import:
        mock_module = Mock()
        mock_module.MockStep = MockStep
        mock_import.return_value = mock_module
        
        loaded_class = load_step_class("test.module", "MockStep")
        assert loaded_class == MockStep


def test_create_step_from_json():
    """Test creating step from JSON"""
    step_def = {
        "id": "test_step",
        "step_class": "MockStep",
        "module": "test.module",
        "config": {"param1": "value1"}
    }
    
    # Mock the import
    with patch('importlib.import_module') as mock_import:
        mock_module = Mock()
        mock_module.MockStep = MockStep
        mock_import.return_value = mock_module
        
        context = {"parameters": {}}
        step = create_step_from_json(step_def, context)
        
        assert step.name == "test_step"
        assert step.config == {"param1": "value1"}


def test_create_pipeline_from_json():
    """Test creating pipeline from JSON"""
    pipeline_config = {
        "name": "Test Pipeline",
        "steps": [
            {
                "id": "step1",
                "step_class": "MockStep",
                "module": "test.module",
                "config": {"param": "value"}
            }
        ],
        "flow": {
            "start_at": "step1",
            "paths": []
        }
    }
    
    # Mock the import
    with patch('importlib.import_module') as mock_import:
        mock_module = Mock()
        mock_module.MockStep = MockStep
        mock_import.return_value = mock_module
        
        pipeline = create_pipeline_from_json(pipeline_config)
        
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].name == "step1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
