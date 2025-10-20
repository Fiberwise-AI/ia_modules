"""
Core Pipeline Implementation
"""

from typing import Dict, Any, List, Optional
import asyncio
import logging

from ia_modules.pipeline.services import ServiceRegistry
from ia_modules.telemetry.integration import get_telemetry


class TemplateParameterResolver:
    """Resolves template parameters in configuration"""

    @staticmethod
    def resolve_parameters(config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve template parameters in configuration using context"""
        def resolve_value(obj):
            if isinstance(obj, dict):
                return {k: resolve_value(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [resolve_value(item) for item in obj]
            elif isinstance(obj, str):
                # Replace {{ parameters.name }} with actual values
                import re
                def replace_param(match):
                    param_path = match.group(1).strip()
                    if param_path.startswith('parameters.'):
                        param_name = param_path[11:]  # Remove 'parameters.'
                        return str(context.get('parameters', {}).get(param_name, match.group(0)))
                    return match.group(0)

                return re.sub(r'\{\{\s*([^}]+)\s*\}\}', replace_param, obj)
            else:
                return obj

        return resolve_value(config)


class InputResolver:
    """Resolves step input templates"""

    @staticmethod
    def resolve_step_inputs(inputs: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve input templates for a step"""
        resolved = {}

        for inp in inputs:
            name = inp["name"]
            source = inp["source"]
            resolved[name] = InputResolver._resolve_source(source, context)

        return resolved

    @staticmethod
    def _resolve_source(source: str, context: Dict[str, Any]) -> Any:
        """Resolve a single source template"""
        if not isinstance(source, str) or not source.startswith('{') or not source.endswith('}'):
            return source

        path = source[1:-1]  # Remove braces
        parts = path.split('.')

        if parts[0] == "parameters":
            return context.get("parameters", {}).get(parts[1])
        elif parts[0] == "pipeline_input":
            return context.get("pipeline_input", {})
        elif parts[0] == "steps":
            step_name = parts[1]
            output_name = parts[3]  # skip "output"
            return context.get("steps", {}).get(step_name, {}).get(output_name)

        return source


class Step:
    """Base class for all pipeline steps with error handling support"""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"Step.{name}")

        # Error handling configuration
        self.error_config = config.get('error_handling', {})
        self.continue_on_error = self.error_config.get('continue_on_error', False)
        self.enable_fallback = self.error_config.get('enable_fallback', False)

        # Retry configuration
        retry_config_dict = self.error_config.get('retry', {})
        if retry_config_dict:
            from .retry import RetryConfig
            self.retry_config = RetryConfig(**retry_config_dict)
        else:
            self.retry_config = None

        # Services will be injected by Pipeline
        self.services = None

    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the step logic

        Override this method in subclasses to implement step behavior.

        Args:
            data: Input data for the step

        Returns:
            Output data from the step

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(
            f"Subclasses must implement run() method. "
            f"{self.__class__.__name__} has not implemented run()."
        )

    async def execute_with_error_handling(
        self,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute step with comprehensive error handling

        This wrapper adds retry logic, fallback mechanisms, and error recovery
        around the step's run() method.

        Args:
            data: Input data for the step

        Returns:
            Output data from the step, or error state if continue_on_error=True

        Raises:
            PipelineError: If step fails and continue_on_error=False
        """
        from .errors import PipelineError, classify_exception
        from .retry import RetryStrategy

        try:
            # Execute with retry if configured
            if self.retry_config and self.retry_config.max_attempts > 1:
                strategy = RetryStrategy(self.retry_config)
                return await strategy.execute_with_retry(self.run, data)
            else:
                return await self.run(data)

        except PipelineError as e:
            # Handle known pipeline errors
            return await self._handle_pipeline_error(e, data)

        except Exception as e:
            # Classify and handle unexpected errors
            pipeline_error = classify_exception(e, step_id=self.name)
            return await self._handle_pipeline_error(pipeline_error, data)

    async def _handle_pipeline_error(
        self,
        error: 'PipelineError',
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle pipeline errors with fallback and recovery

        Args:
            error: The pipeline error that occurred
            data: Original input data

        Returns:
            Fallback result or error state

        Raises:
            PipelineError: If error cannot be handled
        """
        from .errors import ErrorSeverity

        # Log the error
        log_method = (
            self.logger.critical if error.severity == ErrorSeverity.CRITICAL
            else self.logger.error if error.severity == ErrorSeverity.ERROR
            else self.logger.warning
        )
        log_method(f"Error in step '{self.name}': {error}")

        # Try fallback if configured and error is recoverable
        if self.enable_fallback and error.recoverable:
            try:
                self.logger.info(f"Attempting fallback for step '{self.name}'")
                return await self.fallback(data, error)
            except Exception as fallback_error:
                self.logger.error(
                    f"Fallback failed for step '{self.name}': {fallback_error}"
                )
                # Continue to error handling below

        # Return error state if continue_on_error is enabled
        if self.continue_on_error:
            self.logger.warning(
                f"Step '{self.name}' failed but continuing pipeline execution"
            )
            return {
                "step_error": True,
                "error_message": str(error),
                "error_category": error.category.value,
                "error_severity": error.severity.value,
                "step_name": self.name,
                "original_data": data,
                "context": error.context
            }

        # Re-raise to stop pipeline
        raise error

    async def fallback(
        self,
        data: Dict[str, Any],
        error: 'PipelineError'
    ) -> Dict[str, Any]:
        """
        Fallback handler when step execution fails

        Override this method in subclasses to provide custom fallback behavior.
        Default implementation logs a warning and re-raises the error.

        Args:
            data: Original input data
            error: The error that triggered the fallback

        Returns:
            Fallback output data

        Raises:
            PipelineError: If no fallback is implemented

        Example:
            async def fallback(self, data, error):
                # Return cached data if API call fails
                cache_key = data.get('id')
                if cache_key in self.cache:
                    return {"data": self.cache[cache_key], "from_cache": True}
                raise error
        """
        self.logger.warning(f"No fallback implemented for step '{self.name}'")
        raise error

    def get_db(self):
        """Get database service from registry"""
        if self.services:
            return self.services.get('database')
        return None

    def get_http(self):
        """Get HTTP client service from registry"""
        if self.services:
            return self.services.get('http')
        return None


class Pipeline:
    """Main pipeline executor"""

    def __init__(self, name: str, steps: List[Step], flow: Dict[str, Any], services: ServiceRegistry, enable_telemetry: bool = True):
        self.name = name
        self.steps = steps
        self.flow = flow
        self.services = services
        self.logger = logging.getLogger(f"Pipeline.{name}")

        # Telemetry integration
        self.enable_telemetry = enable_telemetry
        self.telemetry = get_telemetry(enabled=enable_telemetry) if enable_telemetry else None

        # Inject services into all steps
        for step in self.steps:
            step.services = self.services

        # Create step mapping for easy lookup
        self.step_map = {step.name: step for step in steps}

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the pipeline with given input data"""
        self.logger.info(f"Starting pipeline execution")

        # Initialize results tracking
        results = {
            "input": input_data,
            "steps": {},
            "output": None
        }

        # Start with input data
        current_data = input_data

        # Execute with telemetry if enabled
        if self.enable_telemetry and self.telemetry:
            with self.telemetry.trace_pipeline(self.name, input_data) as pipeline_ctx:
                return await self._execute_pipeline(input_data, results, current_data, pipeline_ctx)
        else:
            return await self._execute_pipeline(input_data, results, current_data, None)

    async def _execute_pipeline(
        self,
        input_data: Dict[str, Any],
        results: Dict[str, Any],
        current_data: Dict[str, Any],
        pipeline_ctx
    ) -> Dict[str, Any]:
        """Internal pipeline execution with telemetry context"""
        # Execute steps in order based on flow
        try:
            # Get the starting step
            start_step_name = self.flow.get("start_at")
            if not start_step_name:
                raise ValueError("No start step defined in pipeline flow")

            # Build execution path from flow definition
            execution_path = self._build_execution_path()

            # Execute each step in sequence
            for step_name in execution_path:
                step = self.step_map.get(step_name)
                if not step:
                    raise ValueError(f"Step '{step_name}' not found")

                self.logger.info(f"Executing step: {step_name}")

                # Execute step with telemetry if enabled
                if self.enable_telemetry and self.telemetry:
                    parent_span = pipeline_ctx.span if pipeline_ctx else None
                    with self.telemetry.trace_step(self.name, step_name, parent_span) as step_ctx:
                        step_result = await step.execute_with_error_handling(current_data)
                        step_ctx.set_output(step_result)
                else:
                    step_result = await step.execute_with_error_handling(current_data)

                # Store result for this step
                results["steps"][step_name] = step_result

                # Update current data for next step
                current_data = step_result

            # Set final output
            results["output"] = current_data

            # Record result in telemetry
            if pipeline_ctx:
                pipeline_ctx.set_result(results["output"])

            self.logger.info("Pipeline execution completed successfully")
            return results

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            raise

    def _build_execution_path(self) -> List[str]:
        """Build the execution path based on flow definition"""
        # Simple implementation - in a real system this would be more complex
        # For now, we'll execute steps in order they appear in the flow
        paths = self.flow.get("paths", [])

        # Build a simple execution order
        step_order = []
        visited = set()

        # Start with the start_at step
        start_step = self.flow.get("start_at")
        if start_step:
            step_order.append(start_step)
            visited.add(start_step)

        # Add steps based on flow paths
        for path in paths:
            from_step = path.get("from_step")
            to_step = path.get("to_step")

            if from_step and to_step and from_step in step_order:
                # Skip termination markers like "end_with_success"
                if to_step.startswith("end_"):
                    continue
                if to_step not in visited:
                    step_order.append(to_step)
                    visited.add(to_step)

        return step_order


def run_pipeline(name: str, steps: List[Step], flow: Dict[str, Any], services: ServiceRegistry) -> Pipeline:
    """Factory function to create a pipeline instance"""
    return Pipeline(name, steps, flow, services)
