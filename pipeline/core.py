"""
Core Pipeline Implementation
"""

from typing import Dict, Any, List, Optional
import asyncio
import logging

from ia_modules.pipeline.services import ServiceRegistry


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
    """Base class for all pipeline steps"""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"Step.{name}")

    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the step logic"""
        raise NotImplementedError("Subclasses must implement run method")


class Pipeline:
    """Main pipeline executor"""

    def __init__(self, name: str, steps: List[Step], flow: Dict[str, Any], services: ServiceRegistry):
        self.name = name
        self.steps = steps
        self.flow = flow
        self.services = services
        self.logger = logging.getLogger(f"Pipeline.{name}")

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

                # Run the step with current data
                step_result = await step.run(current_data)

                # Store result for this step
                results["steps"][step_name] = step_result

                # Update current data for next step
                current_data = step_result

            # Set final output
            results["output"] = current_data

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
