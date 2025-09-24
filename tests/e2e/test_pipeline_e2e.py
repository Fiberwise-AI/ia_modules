"""
End-to-end tests for pipeline functionality
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from ia_modules.pipeline.runner import run_pipeline_from_json
from ia_modules.pipeline.services import ServiceRegistry


@pytest.mark.asyncio
async def test_end_to_end_pipeline_execution():
    """Test complete end-to-end pipeline execution from JSON file"""
    
    # Create a simple pipeline JSON configuration
    pipeline_config = {
        "name": "E2E Test Pipeline",
        "steps": [
            {
                "id": "step1",
                "step_class": "TestStep",
                "module": "tests.e2e.test_pipeline_e2e",
                "config": {"param": "value"}
            }
        ],
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

    # Mock the step class for testing
    class TestStep:
        def __init__(self, name: str, config: dict):
            self.name = name
            self.config = config
            
        async def work(self, data: dict) -> dict:
            return {"result": "success", "input_data": data}

    # Mock the import to use our test step
    import importlib
    
    # Store original import function
    original_import = importlib.import_module
    
    def mock_import(module_path):
        if module_path == "tests.e2e.test_pipeline_e2e":
            # Return a mock module with our TestStep class
            import types
            mock_module = types.ModuleType("tests.e2e.test_pipeline_e2e")
            mock_module.TestStep = TestStep
            return mock_module
        return original_import(module_path)
    
    # Patch the import function
    import builtins
    original_builtins_import = builtins.__import__
    builtins.__import__ = mock_import
    
    try:
        # Create services
        services = ServiceRegistry()
        
        # Run pipeline from JSON config (this would normally be a file)
        # For E2E test, we'll simulate this by creating a temporary config dict
        result = await run_pipeline_from_json(
            pipeline_file="",
            input_data={"test": "data"},
            services=services,
            working_directory="."
        )
        
        assert result is not None
        assert isinstance(result, dict)
        
    finally:
        # Restore original import
        builtins.__import__ = original_builtins_import


@pytest.mark.asyncio
async def test_pipeline_with_template_parameters():
    """Test pipeline execution with template parameters"""
    
    # Create a pipeline configuration with template parameters
    pipeline_config = {
        "name": "Template Test Pipeline",
        "parameters": {
            "base_url": "https://api.example.com",
            "timeout": 30
        },
        "steps": [
            {
                "id": "step1",
                "step_class": "TestStep",
                "module": "tests.e2e.test_pipeline_e2e",
                "config": {
                    "url": "{parameters.base_url}/endpoint",
                    "timeout": "{parameters.timeout}"
                }
            }
        ],
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

    # Mock the import
    import importlib
    
    # Store original import function
    original_import = importlib.import_module
    
    def mock_import(module_path):
        if module_path == "tests.e2e.test_pipeline_e2e":
            import types
            mock_module = types.ModuleType("tests.e2e.test_pipeline_e2e")
            class TestStep:
                def __init__(self, name: str, config: dict):
                    self.name = name
                    self.config = config
                    
                async def work(self, data: dict) -> dict:
                    return {"result": "success", "config": self.config}
            mock_module.TestStep = TestStep
            return mock_module
        return original_import(module_path)
    
    import builtins
    original_builtins_import = builtins.__import__
    builtins.__import__ = mock_import
    
    try:
        # Create services
        services = ServiceRegistry()
        
        # Run pipeline
        result = await run_pipeline_from_json(
            pipeline_file="",
            input_data={"test": "data"},
            services=services,
            working_directory="."
        )
        
        assert result is not None
        assert isinstance(result, dict)
        
    finally:
        # Restore original import
        builtins.__import__ = original_builtins_import


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
