"""
Generic Graph-Based Pipeline Runner

A comprehensive runner for executing graph-based pipelines with conditional flows,
parallel execution, and advanced routing capabilities.
"""

import asyncio
import json
import sys
import os
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict  # Changed from validator to field_validator

# Add parent directories to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent.parent))
sys.path.insert(0, str(current_dir))

from ia_modules.pipeline.core import Pipeline, Step
from ia_modules.pipeline.runner import create_pipeline_from_json
from ia_modules.pipeline.services import ServiceRegistry


class AgentStepWrapper(Step):
    """Wrapper to make agent-style classes compatible with Step interface"""

    def __init__(self, name: str, agent_instance: Any, config: Dict[str, Any]):
        super().__init__(name, config)
        self.agent_instance = agent_instance

    async def run(self, data: Dict[str, Any]) -> Any:
        """Execute the agent and return results in Step-compatible format"""
        # Call the agent's process method
        result = await self.agent_instance.process(data)

        if result.get('success', False):
            return result.get('data', {})
        else:
            raise Exception(f"Agent {self.name} failed: {result.get('error', 'Unknown error')}")


class PipelineStep(BaseModel):
    """Pydantic model for pipeline step configuration"""
    id: str = Field(..., description="Unique identifier for the step")
    name: str = Field(..., description="Display name for the step")
    type: str = Field(default="task", description="Step type")
    step_class: str = Field(..., description="Python class name to instantiate")
    module: str = Field(..., description="Module path containing the step class")
    config: Dict[str, Any] = Field(default_factory=dict, description="Step configuration")
    input_schema: Optional[Dict[str, Any]] = Field(None, description="Input validation schema")
    output_schema: Optional[Dict[str, Any]] = Field(None, description="Output validation schema")


class FlowCondition(BaseModel):
    """Pydantic model for flow condition"""
    type: str = Field(default="always", description="Condition type")
    config: Dict[str, Any] = Field(default_factory=dict, description="Condition configuration")


class FlowPath(BaseModel):
    """Pydantic model for flow path"""
    model_config = ConfigDict(populate_by_name=True)
    
    from_step: str = Field(..., alias="from", description="Source step ID")
    to_step: str = Field(..., alias="to", description="Target step ID")
    condition: FlowCondition = Field(default_factory=lambda: FlowCondition(), description="Path condition")


class PipelineFlow(BaseModel):
    """Pydantic model for pipeline flow definition"""
    start_at: str = Field(..., description="Starting step ID")
    paths: List[FlowPath] = Field(default_factory=list, description="Flow paths")


class PipelineConfig(BaseModel):
    """Pydantic model for complete pipeline configuration"""
    name: str = Field(..., description="Pipeline name")
    description: Optional[str] = Field(None, description="Pipeline description")
    version: Optional[str] = Field(None, description="Pipeline version")
    parameters: Union[Dict[str, Any], List[Dict[str, Any]]] = Field(default_factory=list, description="Pipeline parameter schemas")
    steps: List[PipelineStep] = Field(..., description="Pipeline steps")
    flow: PipelineFlow = Field(..., description="Pipeline flow definition")
    input_schema: Optional[Dict[str, Any]] = Field(None, description="Input validation schema")
    output_schema: Optional[Dict[str, Any]] = Field(None, description="Output validation schema")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator('parameters', mode='before')
    @classmethod
    def normalize_parameters(cls, v):
        """Normalize parameters - accept both list (schema format) and dict (value format)"""
        if isinstance(v, list):
            # Convert list of parameter schemas to empty dict
            # Actual values come from input_data at runtime
            return {}
        if isinstance(v, dict):
            # Dict format is already fine
            return v
        return {}

    @field_validator('steps')
    @classmethod
    def validate_step_ids_unique(cls, v):
        """Validate that all step IDs are unique"""
        ids = [step.id for step in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Step IDs must be unique")
        return v

    @field_validator('flow')
    @classmethod
    def validate_flow_references(cls, v, info):
        """Validate that flow references exist in steps"""
        # Get the steps from the model data using the info parameter
        if info.data and 'steps' in info.data:
            step_ids = {step.id for step in info.data['steps']}
            step_ids.update(['end', 'end_with_success', 'end_with_failure'])

            for path in v.paths:
                if path.from_step not in step_ids:
                    raise ValueError(f"Flow path references unknown step: {path.from_step}")
                if path.to_step not in step_ids:
                    raise ValueError(f"Flow path references unknown step: {path.to_step}")

        return v


class GraphPipelineRunner:
    """Generic runner for graph-based pipelines with advanced features"""

    def __init__(self, services: Optional[ServiceRegistry] = None):
        self.services = services or ServiceRegistry()
        self.execution_stats = {
            'start_time': None,
            'end_time': None,
            'steps_executed': 0,
            'parallel_executions': 0,
            'routing_decisions': 0
        }

    def _get_central_logger(self):
        """Get the central logging service"""
        if self.services and hasattr(self.services, 'get'):
            return self.services.get('central_logger')
        return None

    def _log_to_central_service(self, level: str, message: str, step_name: Optional[str] = None, data: Optional[Dict[str, Any]] = None):
        """Log a message to the central logging service"""
        logger = self._get_central_logger()
        if logger:
            logger.log(level, message, step_name, data)

    def _log_step_data(self, step_name: str, step_id: str, data_type: str, data: Any):
        """Log step data with selective information to avoid overwhelming logs"""
        if data is None:
            self._log_to_central_service("INFO", f"{step_name} {data_type}: None", step_name=step_id)
            return

        if isinstance(data, dict):
            # Log dictionary keys and types, but not full content for large data
            data_info = {}
            for key, value in data.items():
                if isinstance(value, (list, tuple)) and len(value) > 10:
                    # For large arrays, just log length and type
                    data_info[key] = f"array[{len(value)}] of {type(value[0]).__name__ if value else 'unknown'}"
                elif isinstance(value, (int, float, str, bool)):
                    data_info[key] = value
                else:
                    data_info[key] = f"{type(value).__name__}"

            self._log_to_central_service("INFO", f"{step_name} {data_type}: {data_info}", step_name=step_id, data=data_info)

        elif isinstance(data, (list, tuple)):
            if len(data) > 10:
                self._log_to_central_service("INFO", f"{step_name} {data_type}: array[{len(data)}] of {type(data[0]).__name__ if data else 'unknown'}", step_name=step_id)
            else:
                self._log_to_central_service("INFO", f"{step_name} {data_type}: {data}", step_name=step_id)

        else:
            self._log_to_central_service("INFO", f"{step_name} {data_type}: {data} ({type(data).__name__})", step_name=step_id)

    async def run_pipeline_from_json(
        self,
        pipeline_config: Dict[str, Any],
        input_data: Dict[str, Any] = None,
        use_enhanced_features: bool = True
    ) -> Dict[str, Any]:
        """
        Run a graph-based pipeline from JSON configuration dictionary

        Args:
            pipeline_config: Pipeline configuration as dictionary
            input_data: Initial input data for the pipeline
            use_enhanced_features: Whether to use enhanced pipeline features

        Returns:
            Pipeline execution results
        """
        # Validate and parse pipeline configuration with Pydantic
        try:
            config = PipelineConfig(**pipeline_config)
        except Exception as e:
            raise ValueError(f"Invalid pipeline configuration: {e}")

        # Validate pipeline structure
        self._validate_pipeline_config(config)

        # Prepare input data
        input_data = input_data or {}

        # Log execution start to central service
        execution_id = self._start_execution_logging(config)

        # Create and run pipeline
        try:
            self.execution_stats['start_time'] = datetime.now()

            if use_enhanced_features and self._has_advanced_features(config):
                result = await self._run_enhanced_pipeline(config, input_data)
            else:
                result = await self._run_standard_pipeline(config, input_data)

            self.execution_stats['end_time'] = datetime.now()

            # Log successful execution end to database
            self._log_execution_end_to_database(execution_id, success=True)

            # Write central logs to database
            await self._write_central_logs_to_database()

            # Log successful execution end
            self._log_to_central_service("SUCCESS", f"Pipeline execution completed successfully", data={"execution_stats": self.execution_stats})

            return result

        except Exception as e:
            # Log failed execution end to database
            self._log_execution_end_to_database(execution_id, success=False, error=str(e))

            # Write central logs to database even on failure
            await self._write_central_logs_to_database()

            # Log failed execution end
            self._log_to_central_service("ERROR", f"Pipeline execution failed: {str(e)}")

            raise

    def _start_execution_logging(self, config: PipelineConfig) -> str:
        """Start execution logging and return execution ID"""
        # Generate execution ID
        execution_id = str(uuid.uuid4())
        
        # Get execution tracker service if available
        if self.services and hasattr(self.services, 'get'):
            execution_tracker = self.services.get('execution_tracker')
            if execution_tracker and hasattr(execution_tracker, 'start_execution'):
                # Use execution tracker to log start
                execution_tracker.start_execution(
                    execution_id=execution_id,
                    pipeline_name=config.name,
                    pipeline_version=config.version,
                    config=config.dict()
                )

        # Set execution ID in central logger
        logger = self._get_central_logger()
        if logger and hasattr(logger, 'set_execution_id'):
            logger.set_execution_id(execution_id)

        # Log execution start
        self._log_to_central_service("INFO", f"Starting pipeline execution: {config.name}", data={
            "pipeline_name": config.name,
            "pipeline_version": config.version,
            "step_count": len(config.steps)
        })

        return execution_id

    def _log_execution_end_to_database(self, execution_id: str, success: bool, error: Optional[str] = None):
        """Log pipeline execution end to execution tracker"""
        if self.services and hasattr(self.services, 'get'):
            execution_tracker = self.services.get('execution_tracker')
            if execution_tracker and hasattr(execution_tracker, 'end_execution'):
                execution_tracker.end_execution(
                    execution_id=execution_id,
                    success=success,
                    error=error
                )

    async def _write_central_logs_to_database(self):
        """Write central logger logs to database via execution tracker"""
        if self.services and hasattr(self.services, 'get'):
            execution_tracker = self.services.get('execution_tracker')
            central_logger = self.services.get('central_logger')
            if execution_tracker and central_logger and hasattr(central_logger, 'write_to_database'):
                await central_logger.write_to_database(execution_tracker)

    def _validate_pipeline_config(self, config: PipelineConfig):
        """Validate pipeline configuration structure (already done by Pydantic)"""
        pass

    def _has_advanced_features(self, config: PipelineConfig) -> bool:
        """Check if pipeline uses advanced features requiring enhanced pipeline"""
        paths = config.flow.paths

        # Check for advanced condition types
        for path in paths:
            condition_type = path.condition.type
            if condition_type in ['agent', 'function']:
                return True

        # Check for parallel execution indicators
        for path in paths:
            if isinstance(path.to_step, list):
                return True

        return False

    async def _run_standard_pipeline(self, config: PipelineConfig, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run pipeline using standard graph execution"""
        # Convert Pydantic model back to dict for compatibility
        config_dict = config.dict()
        pipeline = create_pipeline_from_json(config_dict, self.services)

        # Log pipeline start
        self._log_to_central_service("INFO", f"Using standard graph pipeline execution with {len(config.steps)} steps")

        # Log individual step details
        for step in config.steps:
            self._log_to_central_service("INFO", f"Step configured: {step.name} ({step.id})", step_name=step.id, data={
                "step_class": step.step_class,
                "module": step.module,
                "config": step.config
            })

        # Run pipeline and track step executions
        start_time = datetime.now()
        try:
            result = await pipeline.run(input_data)
            end_time = datetime.now()

            # Log successful completion
            duration = (end_time - start_time).total_seconds()
            self._log_to_central_service("SUCCESS", f"Pipeline completed in {duration:.2f} seconds", data={
                "duration_seconds": duration,
                "steps_executed": len(config.steps)
            })

            self.execution_stats['steps_executed'] = len(config.steps)
            return result

        except Exception as e:
            # Log pipeline failure
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            self._log_to_central_service("ERROR", f"Pipeline failed after {duration:.2f} seconds: {str(e)}")
            raise

    async def _run_enhanced_pipeline(self, config: PipelineConfig, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run pipeline using enhanced features"""
        # Convert config format for enhanced pipeline
        enhanced_config = self._convert_to_enhanced_format(config)

        # Log enhanced pipeline start
        self._log_to_central_service("INFO", f"Using enhanced graph pipeline execution with {len(config.steps)} steps")

        # Log individual step details
        for step in config.steps:
            self._log_to_central_service("INFO", f"Step configured: {step.name} ({step.id})", step_name=step.id, data={
                "step_class": step.step_class,
                "module": step.module,
                "config": step.config
            })

        pipeline = EnhancedPipeline(
            steps=[],  # Will be created from config
            services=self.services,
            structure=enhanced_config
        )

        try:
            result = await pipeline.run(input_data)

            # Update execution stats
            stats = pipeline.get_execution_stats()
            self.execution_stats['parallel_executions'] = stats.get('active_parallel_tasks', 0)

            # Log successful completion
            self._log_to_central_service("SUCCESS", "Enhanced pipeline completed successfully", data={
                "parallel_executions": self.execution_stats['parallel_executions']
            })

            return result

        except Exception as e:
            # Log enhanced pipeline failure
            self._log_to_central_service("ERROR", f"Enhanced pipeline failed: {str(e)}")
            raise

    def _convert_to_enhanced_format(self, config: PipelineConfig) -> Dict[str, Any]:
        """Convert Pydantic config to enhanced pipeline format"""
        config_dict = config.dict(by_alias=True)

        # Convert step format
        if 'steps' in config_dict:
            enhanced_steps = []
            for step in config_dict['steps']:
                enhanced_step = step.copy()
                enhanced_step['id'] = step['name']  # Enhanced pipeline uses 'id'
                enhanced_steps.append(enhanced_step)
            config_dict['steps'] = enhanced_steps

        # Convert flow paths format
        if 'flow' in config_dict and 'paths' in config_dict['flow']:
            enhanced_paths = []
            for path in config_dict['flow']['paths']:
                enhanced_path = path.copy()
                enhanced_path['from_step'] = path['from']
                enhanced_path['to_step'] = path['to']
                enhanced_paths.append(enhanced_path)
            config_dict['flow']['paths'] = enhanced_paths

        return config_dict

    async def run_pipeline_with_real_classes(
        self,
        pipeline_config: Dict[str, Any],
        input_data: Dict[str, Any],
        real_classes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run pipeline with real agent class implementations

        Args:
            pipeline_config: Pipeline configuration as dictionary
            input_data: Initial input data for the pipeline
            real_classes: Dictionary mapping step class names to actual implementations

        Returns:
            Pipeline execution results with real agent processing
        """
        # Validate and parse pipeline configuration
        try:
            config = PipelineConfig(**pipeline_config)
        except Exception as e:
            raise ValueError(f"Invalid pipeline configuration: {e}")

        # Log execution start
        execution_id = self._start_execution_logging(config)

        try:
            self.execution_stats['start_time'] = datetime.now()

            # Create pipeline with real agents
            result = await self._run_pipeline_with_real_agents(config, input_data, real_classes)

            self.execution_stats['end_time'] = datetime.now()

            # Log successful execution
            self._log_execution_end_to_database(execution_id, success=True)
            await self._write_central_logs_to_database()
            self._log_to_central_service("SUCCESS", "Real agent pipeline execution completed successfully",
                                       data={"execution_stats": self.execution_stats})

            return result

        except Exception as e:
            # Log failed execution
            self._log_execution_end_to_database(execution_id, success=False, error=str(e))
            await self._write_central_logs_to_database()
            self._log_to_central_service("ERROR", f"Real agent pipeline execution failed: {str(e)}")
            raise

    async def _run_pipeline_with_real_agents(self, config: PipelineConfig, input_data: Dict[str, Any], real_classes: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pipeline with real agent implementations"""

        # Create custom pipeline runner that uses real agents
        from ia_modules.pipeline.core import Pipeline, Step

        # Convert config to format suitable for direct agent execution
        steps = []

        for step_config in config.steps:
            step_class_name = step_config.step_class

            if step_class_name in real_classes:
                # Get the real agent class
                AgentClass = real_classes[step_class_name]

                # Create agent instance based on type
                if hasattr(AgentClass, 'process'):
                    # This is an agent-style class (like DestinationAnalysisAgent)
                    agent_instance = AgentClass(step_config.config)
                    # Wrap it in a Step-compatible interface
                    step = AgentStepWrapper(step_config.id, agent_instance, step_config.config)
                else:
                    # This is a step-style class (like HotelScraperStep)
                    step = AgentClass(step_config.id, step_config.config)

                steps.append(step)

                # Log step creation
                self._log_to_central_service("INFO", f"Created real agent: {step_config.name} ({step_class_name})",
                                           step_name=step_config.id)
            else:
                raise ValueError(f"Real agent class not found: {step_class_name}")

        # Create pipeline with real agents
        # Pipeline expects: (name, steps, flow, services, ...)
        pipeline_structure = config.model_dump() if hasattr(config, 'model_dump') else config.dict()
        pipeline = Pipeline(
            name=config.name,
            steps=steps,
            flow=pipeline_structure.get('flow', {}),
            services=self.services
        )

        # Execute pipeline
        self._log_to_central_service("INFO", f"Executing pipeline with {len(steps)} real agents")

        start_time = datetime.now()
        result = await pipeline.run(input_data)
        end_time = datetime.now()

        duration = (end_time - start_time).total_seconds()
        self.execution_stats['steps_executed'] = len(steps)

        self._log_to_central_service("SUCCESS", f"Real agent pipeline completed in {duration:.2f} seconds",
                                   data={"duration_seconds": duration, "steps_executed": len(steps)})

        return result

    async def run_with_different_scenarios(
        self,
        pipeline_config: Dict[str, Any],
        scenarios: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Run the same pipeline with different input scenarios

        Args:
            pipeline_config: Pipeline configuration as dictionary
            scenarios: List of scenario dictionaries with 'name', 'input_data', and optional 'description'

        Returns:
            List of execution results for each scenario
        """
        results = []

        for scenario in scenarios:
            try:
                result = await self.run_pipeline_from_json(
                    pipeline_config=pipeline_config,
                    input_data=scenario['input_data'],
                    use_enhanced_features=False  # Keep it simple for scenarios
                )

                scenario_result = {
                    'scenario_name': scenario['name'],
                    'success': True,
                    'result': result,
                    'execution_stats': self.execution_stats.copy()
                }

            except Exception as e:
                scenario_result = {
                    'scenario_name': scenario['name'],
                    'success': False,
                    'error': str(e),
                    'execution_stats': self.execution_stats.copy()
                }

            results.append(scenario_result)

        return results


async def run_graph_pipeline(
    pipeline_config: Dict[str, Any],
    input_data: Dict[str, Any] = None,
    services: Optional[ServiceRegistry] = None
) -> Dict[str, Any]:
    """
    Convenience function to run a graph-based pipeline

    Args:
        pipeline_config: Pipeline configuration as dictionary
        input_data: Input data for the pipeline
        services: Service registry for dependency injection

    Returns:
        Pipeline execution results
    """
    runner = GraphPipelineRunner(services)
    return await runner.run_pipeline_from_json(pipeline_config, input_data)


async def run_graph_pipeline_from_file(
    pipeline_file: str,
    input_data: Dict[str, Any] = None,
    services: Optional[ServiceRegistry] = None
) -> Dict[str, Any]:
    """
    Convenience function to run a graph-based pipeline from file

    Args:
        pipeline_file: Path to JSON pipeline configuration file
        input_data: Input data for the pipeline
        services: Service registry for dependency injection

    Returns:
        Pipeline execution results
    """
    # Load pipeline configuration from file
    json_path = Path(pipeline_file)
    if not json_path.exists():
        raise FileNotFoundError(f"Pipeline configuration file not found: {pipeline_file}")

    with open(json_path, 'r', encoding='utf-8') as f:
        pipeline_config = json.load(f)

    return await run_graph_pipeline(pipeline_config, input_data, services)


# Example usage
async def example_usage():
    """Example of how to use the generic pipeline runner"""

    # Sample pipeline configuration
    pipeline_config = {
        "name": "Example Pipeline",
        "description": "A simple example pipeline",
        "steps": [
            {
                "id": "step1",
                "name": "First Step",
                "step_class": "ExampleStep",
                "module": "my_module.steps",
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

    # Basic usage with JSON config
    result = await run_graph_pipeline(
        pipeline_config=pipeline_config,
        input_data={"key": "value"}
    )

    # With custom services
    services = ServiceRegistry()
    # Add your services here...

    result = await run_graph_pipeline(
        pipeline_config=pipeline_config,
        input_data={"key": "value"},
        services=services
    )

    return result


if __name__ == "__main__":
    # CLI usage example
    if len(sys.argv) < 2:
        print("Usage: python graph_pipeline_runner.py <pipeline_file> [input_json]")
        print("Example: python graph_pipeline_runner.py pipeline.json '{\"key\": \"value\"}'")
        sys.exit(1)

    pipeline_file = sys.argv[1]
    input_data = {}

    if len(sys.argv) > 2:
        try:
            input_data = json.loads(sys.argv[2])
        except json.JSONDecodeError as e:
            print(f"Error parsing input JSON: {e}")
            sys.exit(1)

    asyncio.run(run_graph_pipeline_from_file(pipeline_file, input_data))
