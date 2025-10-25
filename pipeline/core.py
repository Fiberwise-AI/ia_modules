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

    @staticmethod
    def resolve_string_template(template: str, context: Dict[str, Any]) -> str:
        """Resolve template placeholders in a string using context

        Supports both {{ variable }} and {variable} syntax.
        Can access nested values using dot notation: {{ context.field }}
        """
        import re

        def replace_param(match):
            param_path = match.group(1).strip()

            # Navigate through nested dict using dot notation
            parts = param_path.split('.')
            value = context

            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part)
                    if value is None:
                        return match.group(0)  # Return original if not found
                else:
                    return match.group(0)  # Return original if can't navigate

            return str(value) if value is not None else match.group(0)

        # Replace both {{ }} and { } patterns
        result = re.sub(r'\{\{\s*([^}]+)\s*\}\}', replace_param, template)
        result = re.sub(r'\{\s*([^}]+)\s*\}', replace_param, result)

        return result


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

    def __init__(self, name: str, steps: List[Step], flow: Dict[str, Any], services: ServiceRegistry, enable_telemetry: bool = True, loop_config: Optional[Dict[str, Any]] = None, checkpointer: Optional[Any] = None):
        self.name = name
        self.steps = steps
        self.flow = flow
        self.services = services
        self.logger = logging.getLogger(f"Pipeline.{name}")
        self.loop_config = loop_config or {}
        self.checkpointer = checkpointer  # Optional checkpoint storage backend

        # Telemetry integration
        self.enable_telemetry = enable_telemetry
        self.telemetry = get_telemetry(enabled=enable_telemetry) if enable_telemetry else None

        # Loop detection and execution
        self.loop_detector = None
        self.loop_executor = None

        # Initialize loop support if flow has paths
        if flow and 'paths' in flow:
            try:
                from ia_modules.pipeline.loop_detector import LoopDetector
                from ia_modules.pipeline.loop_executor import LoopAwareExecutor

                self.loop_detector = LoopDetector(flow)
                loops = self.loop_detector.detect_loops()

                # Only initialize executor if loops actually detected
                if loops:
                    self.loop_executor = LoopAwareExecutor(flow, self.loop_config)
                    self.logger.info(f"Detected {len(loops)} loop(s) in pipeline '{name}'")

                    # Validate loops for safety
                    validation_errors = self.loop_detector.validate_loops()
                    if validation_errors:
                        for error in validation_errors:
                            self.logger.warning(f"Loop validation: {error}")
            except ImportError:
                self.logger.debug("Loop detection modules not available, continuing without loop support")

        # Inject services into all steps
        for step in self.steps:
            step.services = self.services

        # Create step mapping for easy lookup
        self.step_map = {step.name: step for step in steps}

    def has_loops(self) -> bool:
        """Public method to check if pipeline has loops"""
        return self.loop_executor is not None  # Only True if loops actually detected

    def get_loops(self) -> List[Any]:
        """Get detected loops in the pipeline"""
        if self.loop_detector:
            return self.loop_detector.detect_loops()
        return []

    async def run(self, input_data: Dict[str, Any], thread_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the pipeline with given input data.

        Args:
            input_data: Input data for the pipeline
            thread_id: Thread ID for checkpointing (optional, required if checkpointer enabled)

        Returns:
            Pipeline execution results

        Example:
            >>> pipeline = Pipeline(...)
            >>> result = await pipeline.run({'data': 'value'}, thread_id='user-123')
        """
        self.logger.info(f"Starting pipeline execution")

        # Initialize results tracking
        results = {
            "input": input_data,
            "steps": [],
            "output": None
        }

        # Start with input data
        current_data = input_data

        # Execute with telemetry if enabled
        if self.enable_telemetry and self.telemetry:
            with self.telemetry.trace_pipeline(self.name, input_data) as pipeline_ctx:
                return await self._execute_pipeline(input_data, results, current_data, pipeline_ctx, thread_id)
        else:
            return await self._execute_pipeline(input_data, results, current_data, None, thread_id)

    async def resume(self, thread_id: str, checkpoint_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Resume pipeline execution from a checkpoint.

        Args:
            thread_id: Thread ID to resume
            checkpoint_id: Specific checkpoint ID (optional, defaults to latest)

        Returns:
            Pipeline execution results

        Raises:
            ValueError: If checkpointer not configured or checkpoint not found

        Example:
            >>> pipeline = Pipeline(..., checkpointer=checkpointer)
            >>> # Later, resume from where it left off
            >>> result = await pipeline.resume(thread_id='user-123')
        """
        if not self.checkpointer:
            raise ValueError("Cannot resume: checkpointer not configured")

        self.logger.info(f"Resuming pipeline execution for thread {thread_id}")

        # Load checkpoint
        checkpoint = await self.checkpointer.load_checkpoint(thread_id, checkpoint_id)

        if not checkpoint:
            raise ValueError(f"No checkpoint found for thread {thread_id}")

        self.logger.info(f"Loaded checkpoint: {checkpoint.checkpoint_id} (step: {checkpoint.step_id})")

        # Restore state from checkpoint
        state = checkpoint.state
        input_data = state.get('pipeline_input', {})
        completed_steps = state.get('steps', {})
        current_data = state.get('current_data', {})

        # Initialize results with restored state
        results = {
            "input": input_data,
            "steps": completed_steps,
            "output": None
        }

        # Determine next step to execute
        execution_path = self._build_execution_path()
        next_step_index = checkpoint.step_index + 1

        if next_step_index >= len(execution_path):
            # Pipeline was already complete
            self.logger.info("Pipeline was already complete at checkpoint")
            results["output"] = current_data
            return results

        # Continue execution from next step
        self.logger.info(f"Continuing execution from step index {next_step_index}")

        # Execute remaining steps
        for step_index in range(next_step_index, len(execution_path)):
            step_name = execution_path[step_index]
            step = self.step_map.get(step_name)

            if not step:
                raise ValueError(f"Step '{step_name}' not found")

            self.logger.info(f"Executing step: {step_name}")

            # Execute step
            step_result = await step.execute_with_error_handling(current_data)

            # Store result
            results["steps"].append({
                "step_name": step_name,
                "step_index": step_index,
                "result": step_result,
                "status": "completed"
            })
            current_data = step_result

            # Save checkpoint after each step
            try:
                checkpoint_id = await self.checkpointer.save_checkpoint(
                    thread_id=thread_id,
                    pipeline_id=self.name,
                    step_id=step_name,
                    step_index=step_index,
                    state={
                        'pipeline_input': input_data,
                        'steps': results["steps"],
                        'current_data': current_data
                    },
                    metadata={'execution_path': execution_path, 'resumed': True},
                    step_name=step_name,
                    parent_checkpoint_id=checkpoint.checkpoint_id
                )
                self.logger.debug(f"Saved checkpoint: {checkpoint_id}")
            except Exception as e:
                self.logger.warning(f"Failed to save checkpoint: {e}")

        # Set final output
        results["output"] = current_data

        self.logger.info("Pipeline execution completed successfully (resumed)")
        return results

    async def _execute_pipeline(
        self,
        input_data: Dict[str, Any],
        results: Dict[str, Any],
        current_data: Dict[str, Any],
        pipeline_ctx,
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Internal pipeline execution with telemetry context and checkpointing"""
        # Execute steps in order based on flow
        try:
            # Get the starting step
            current_step_name = self.flow.get("start_at")
            if not current_step_name:
                raise ValueError("No start step defined in pipeline flow")

            step_index = 0
            visited_steps = set()  # Track visited steps for loop detection
            max_steps = 100  # Safety limit to prevent infinite loops
            
            # Execute steps dynamically based on flow transitions
            while current_step_name and step_index < max_steps:
                # Check if we've hit a termination step
                if current_step_name.startswith("end_"):
                    self.logger.info(f"Reached termination step: {current_step_name}")
                    break
                
                step = self.step_map.get(current_step_name)
                if not step:
                    raise ValueError(f"Step '{current_step_name}' not found")

                self.logger.info(f"Executing step {step_index}: {current_step_name}")

                # Check loop safety if loop executor available
                if self.loop_executor:
                    loop_id = self.loop_executor.get_loop_id_for_step(current_step_name)
                    if loop_id:
                        is_safe, error_msg = await self.loop_executor.check_loop_safety(current_step_name, loop_id)
                        if not is_safe:
                            raise RuntimeError(f"Loop safety check failed: {error_msg}")
                        self.loop_executor.loop_context.increment_iteration(current_step_name)

                # Track step execution start if tracker available
                step_execution_id = None
                tracker = self.services.get('execution_tracker') if self.services else None
                execution_id = self.services.get('execution_id') if self.services else None
                
                if tracker and execution_id:
                    step_execution_id = await tracker.start_step_execution(
                        execution_id=execution_id,
                        step_id=current_step_name,
                        step_name=current_step_name,
                        step_type="task",
                        input_data=current_data
                    )

                # Execute step with telemetry if enabled
                step_error = None
                try:
                    if self.enable_telemetry and self.telemetry:
                        parent_span = pipeline_ctx.span if pipeline_ctx else None
                        with self.telemetry.trace_step(self.name, current_step_name, parent_span) as step_ctx:
                            step_result = await step.execute_with_error_handling(current_data)
                            step_ctx.set_output(step_result)

                            # Extract and set LLM usage if present in result
                            if isinstance(step_result, dict) and 'llm_response' in step_result:
                                llm_resp = step_result['llm_response']
                                if isinstance(llm_resp, dict) and 'usage' in llm_resp:
                                    step_ctx.set_attribute('usage', llm_resp['usage'])
                    else:
                        step_result = await step.execute_with_error_handling(current_data)
                except Exception as e:
                    step_error = e
                    step_result = None

                # Track step execution completion if tracker available
                if tracker and step_execution_id:
                    from ia_modules.pipeline.execution_tracker import StepStatus
                    await tracker.complete_step_execution(
                        step_execution_id=step_execution_id,
                        status=StepStatus.FAILED if step_error else StepStatus.COMPLETED,
                        output_data=step_result,
                        error_message=str(step_error) if step_error else None
                    )

                # Re-raise error if step failed
                if step_error:
                    raise step_error

                # Store result for this step
                results["steps"].append({
                    "step_name": current_step_name,
                    "step_index": step_index,
                    "result": step_result,
                    "status": "completed"
                })

                # Merge step result into current data (keep all previous data)
                # Store result both at top level AND under step name for condition access
                if isinstance(step_result, dict):
                    current_data.update(step_result)
                    current_data[current_step_name] = step_result
                else:
                    current_data[current_step_name] = step_result

                # Save checkpoint after each step (if checkpointer enabled)
                if self.checkpointer and thread_id:
                    try:
                        checkpoint_id = await self.checkpointer.save_checkpoint(
                            thread_id=thread_id,
                            pipeline_id=self.name,
                            step_id=current_step_name,
                            step_index=step_index,
                            state={
                                'pipeline_input': input_data,
                                'steps': results["steps"],
                                'current_data': current_data
                            },
                            metadata={'visited_steps': list(visited_steps)},
                            step_name=current_step_name
                        )
                        self.logger.debug(f"Saved checkpoint: {checkpoint_id}")
                    except Exception as e:
                        self.logger.warning(f"Failed to save checkpoint: {e}")

                # Find next steps by evaluating transitions
                next_steps = self._get_next_steps(current_step_name, current_data)
                
                if not next_steps:
                    self.logger.info(f"No more transitions from step '{current_step_name}', pipeline complete")
                    break
                
                # Handle parallel or sequential execution
                if len(next_steps) > 1:
                    self.logger.info(f"Parallel fanout: {len(next_steps)} branches from '{current_step_name}'")
                    # For parallel execution, we need to execute all branches
                    # Use a simple approach: add all to a queue and process each
                    # (Real parallel would use asyncio.gather, but this maintains order for now)
                    pending_steps = list(next_steps)
                    current_step_name = pending_steps.pop(0)
                    step_index += 1
                    
                    # Store remaining parallel steps to execute after current
                    if not hasattr(self, '_pending_parallel_steps'):
                        self._pending_parallel_steps = []
                    self._pending_parallel_steps.extend(pending_steps)
                elif hasattr(self, '_pending_parallel_steps') and self._pending_parallel_steps:
                    # Continue with pending parallel steps
                    current_step_name = self._pending_parallel_steps.pop(0)
                    step_index += 1
                else:
                    # Single next step
                    current_step_name = next_steps[0]
                    step_index += 1

            if step_index >= max_steps:
                raise RuntimeError(f"Pipeline exceeded maximum steps ({max_steps}), possible infinite loop")

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
    
    def _get_next_steps(self, current_step: str, current_data: Dict[str, Any]) -> List[str]:
        """
        Determine next steps based on current step and data.
        Returns list of all matching target steps (for parallel execution).
        """
        paths = self.flow.get("paths", [])
        candidates = [p for p in paths if p.get("from") == current_step]

        if not candidates:
            return []

        next_steps = []
        for path in candidates:
            condition = path.get("condition", {"type": "always"})
            if self._evaluate_condition(condition, current_data):
                next_step = path.get("to")
                next_steps.append(next_step)
        
        return next_steps
    
    def _evaluate_condition(self, condition: Dict[str, Any], data: Dict[str, Any]) -> bool:
        """
        Evaluate a condition against current data.
        
        Supports:
        - always: Always true
        - expression: Evaluate an expression (simple key==value checks)
        """
        condition_type = condition.get("type", "always")
        
        if condition_type == "always":
            return True
        
        if condition_type == "expression":
            config = condition.get("config", {})
            source = config.get("source", "")
            operator = config.get("operator", "equals")
            expected_value = config.get("value")
            
            # Parse source (e.g., "review_content.approved")
            parts = source.split(".")
            actual_value = data
            for part in parts:
                if isinstance(actual_value, dict):
                    actual_value = actual_value.get(part)
                else:
                    return False
            
            # Evaluate operator
            if operator == "equals":
                return actual_value == expected_value
            elif operator == "not_equals":
                return actual_value != expected_value
            elif operator == "greater_than":
                return actual_value > expected_value
            elif operator == "less_than":
                return actual_value < expected_value
        
        # Unknown condition type - default to False for safety
        self.logger.warning(f"Unknown condition type: {condition_type}")
        return False

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
            from_step = path.get("from")
            to_step = path.get("to")

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
