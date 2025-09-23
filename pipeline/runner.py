"""
Pipeline Execution Engine

Production-ready pipeline runner with JSON configuration support.
Clean implementation without POC dependencies.
"""

import json
import importlib
from typing import Dict, Any, List, Optional
from pathlib import Path

from .core import Step, Pipeline, TemplateParameterResolver
from .services import ServiceRegistry


def load_step_class(module_path: str, class_name: str):
    """Dynamically import step class"""
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except ImportError as e:
        raise ImportError(f"Cannot import module '{module_path}': {e}")
    except AttributeError as e:
        raise AttributeError(f"Module '{module_path}' has no class '{class_name}': {e}")


def create_step_from_json(step_def: Dict[str, Any], context: Dict[str, Any] = None) -> Step:
    """Create step instance from JSON definition"""
    # Resolve template parameters in step configuration
    resolved_config = step_def.get('config', {})
    if context:
        resolved_config = TemplateParameterResolver.resolve_parameters(resolved_config, context)
    
    step_class = load_step_class(step_def['module'], step_def.get('step_class', step_def.get('class', '')))
    step_name = step_def.get('id', step_def.get('name', 'Unknown'))
    return step_class(step_name, resolved_config)


def create_pipeline_from_json(pipeline_config: Dict[str, Any], services: Optional[ServiceRegistry] = None) -> Pipeline:
    """Create pipeline from JSON configuration with graph-based execution support"""

    # Prepare context for template resolution
    context = {
        'parameters': pipeline_config.get('parameters', {}),
        'pipeline_input': {}
    }

    # Create steps with template resolution
    steps = []
    for step_def in pipeline_config['steps']:
        step = create_step_from_json(step_def, context)
        steps.append(step)

    # Extract structure for graph-based execution (if present)
    structure = None
    if 'flow' in pipeline_config:
        structure = {
            'steps': pipeline_config.get('steps', []),
            'flow': pipeline_config['flow'],
            'parameters': pipeline_config.get('parameters', {})
        }

    # Create pipeline with optional structure
    return Pipeline(steps, services=services, structure=structure)


async def run_pipeline_from_json(
    pipeline_file: str, 
    input_data: Dict[str, Any] = None,
    services: Optional[ServiceRegistry] = None,
    working_directory: Optional[str] = None
) -> Dict[str, Any]:
    """Main runner function - execute pipeline from JSON configuration"""
    
    # Set working directory for relative module imports
    if working_directory:
        import sys
        if working_directory not in sys.path:
            sys.path.insert(0, working_directory)
    
    # Load JSON configuration
    json_path = Path(pipeline_file)
    if not json_path.exists():
        raise FileNotFoundError(f"Pipeline configuration file not found: {pipeline_file}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        pipeline_config = json.load(f)
    
    # Ensure input data exists
    if input_data is None:
        input_data = {}
    
    # Create and run pipeline with injected services
    pipeline = create_pipeline_from_json(pipeline_config, services)
    result = await pipeline.run(input_data)
    
    return result
