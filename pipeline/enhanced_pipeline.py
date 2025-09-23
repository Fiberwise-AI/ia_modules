"""
Enhanced Pipeline with Phase 3 Advanced Routing

Extends the base Pipeline class with agent-based conditions, function-based conditions,
and parallel execution capabilities.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from .core import Pipeline, Step, TemplateParameterResolver
from .routing import AdvancedRouter, ParallelExecutor, RoutingContext


class EnhancedPipeline(Pipeline):
    """Enhanced pipeline with Phase 3 advanced routing capabilities"""

    def __init__(self,
                 steps: List[Step],
                 services: Optional['ServiceRegistry'] = None,
                 structure: Optional[Dict[str, Any]] = None):
        super().__init__(steps, services, structure)
        self.router = AdvancedRouter()
        self.parallel_executor = ParallelExecutor()
        self.execution_id = None

    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced run method with advanced routing"""
        self.execution_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(self)}"

        try:
            return await self._run_with_advanced_routing(data)
        finally:
            # Cleanup parallel tasks
            self.parallel_executor.cancel_all_tasks()

    async def _run_with_advanced_routing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run pipeline with advanced routing capabilities"""
        if not self.structure:
            return await super().run(data)

        steps_map = {step.id: step for step in self.steps}
        flow = self.structure.get('flow', {})
        paths = flow.get('paths', [])
        current_step_id = flow.get('start_at')

        if not current_step_id:
            raise ValueError("Pipeline flow missing 'start_at' field")

        current_data = data.copy()
        step_results = {}

        # Build initial context
        context = {
            'pipeline_input': data,
            'steps': step_results,
            'pipeline_parameters': self.structure.get('parameters', {})
        }

        while current_step_id and current_step_id not in ['end', 'end_with_success']:
            if current_step_id in ['end_with_failure']:
                raise Exception("Pipeline flow ended in a failure state.")

            # Handle parallel execution
            if isinstance(current_step_id, list):
                parallel_results = await self._execute_parallel_steps(
                    current_step_id, steps_map, current_data, context
                )

                # Merge results from parallel execution
                for step_id, result in parallel_results.items():
                    step_results[step_id] = result
                    if 'output' in result:
                        current_data.update(result['output'])

                # Find next step(s) after parallel execution
                next_steps = await self._find_next_steps_advanced(
                    current_step_id, paths, current_data, step_results
                )

            else:
                # Single step execution
                step = steps_map.get(current_step_id)
                if not step:
                    raise ValueError(f"Step '{current_step_id}' not found in pipeline definition.")

                # Resolve template parameters
                step = self._resolve_step_templates(step, context)

                # Execute step
                result_data = await step.run(current_data)
                step_results[current_step_id] = {
                    'success': True,
                    'result': result_data.get(step.name, {}),
                    'output': result_data,
                    'timestamp': datetime.now().isoformat()
                }

                # Update context
                context['steps'] = step_results
                current_data = result_data

                # Find next step using advanced routing
                next_steps = await self._find_next_steps_advanced(
                    [current_step_id], paths, current_data, step_results
                )

            # Determine next step(s)
            if not next_steps:
                break
            elif len(next_steps) == 1:
                current_step_id = next_steps[0]
            else:
                # Multiple next steps - prepare for parallel execution
                current_step_id = next_steps

        return current_data

    async def _execute_parallel_steps(self,
                                    step_ids: List[str],
                                    steps_map: Dict[str, Step],
                                    data: Dict[str, Any],
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multiple steps in parallel"""

        async def execute_single_step(step_id: str) -> Dict[str, Any]:
            step = steps_map.get(step_id)
            if not step:
                raise ValueError(f"Step '{step_id}' not found in pipeline definition.")

            # Resolve templates for this step
            resolved_step = self._resolve_step_templates(step, context)

            # Execute step
            result_data = await resolved_step.run(data.copy())

            return {
                'success': True,
                'result': result_data.get(step.name, {}),
                'output': result_data,
                'timestamp': datetime.now().isoformat()
            }

        # Create routing context for parallel execution
        routing_context = RoutingContext(
            pipeline_data=data,
            step_results=context.get('steps', {}),
            current_step_id=str(step_ids),
            execution_id=self.execution_id
        )

        return await self.parallel_executor.execute_parallel_steps(
            step_ids, execute_single_step, routing_context
        )

    async def _find_next_steps_advanced(self,
                                      current_steps: List[str],
                                      paths: List[Dict[str, Any]],
                                      data: Dict[str, Any],
                                      step_results: Dict[str, Any]) -> List[str]:
        """Find next steps using advanced routing"""

        # Create routing context
        routing_context = RoutingContext(
            pipeline_data=data,
            step_results=step_results,
            current_step_id=str(current_steps),
            execution_id=self.execution_id
        )

        all_next_steps = []

        for current_step in current_steps:
            next_steps = await self.router.find_next_steps(
                current_step, paths, routing_context
            )
            all_next_steps.extend(next_steps)

        # Remove duplicates and end states
        unique_next_steps = []
        for step in all_next_steps:
            if step not in unique_next_steps and step not in ['end', 'end_with_success', 'end_with_failure']:
                unique_next_steps.append(step)

        return unique_next_steps

    def _resolve_step_templates(self, step: Step, context: Dict[str, Any]) -> Step:
        """Resolve template parameters in step configuration"""
        if not step.config:
            return step

        resolved_config = TemplateParameterResolver.resolve_parameters(step.config, context)

        # Create new step instance with resolved config
        resolved_step = type(step)(step.id, resolved_config)
        resolved_step.name = step.name
        if hasattr(step, 'logger'):
            resolved_step.logger = step.logger

        return resolved_step

    async def validate_advanced_flow(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Validate pipeline flow with advanced routing features"""
        errors = []
        warnings = []

        flow = structure.get('flow', {})
        paths = flow.get('paths', [])
        steps = structure.get('steps', [])
        step_ids = {step['id'] for step in steps}

        # Check for advanced condition types
        for path in paths:
            condition = path.get('condition', {})
            condition_type = condition.get('type', 'always')

            if condition_type == 'agent':
                config = condition.get('config', {})
                if not config.get('model'):
                    errors.append(f"Agent condition missing 'model' configuration in path {path}")
                if not config.get('prompt_template'):
                    errors.append(f"Agent condition missing 'prompt_template' in path {path}")

            elif condition_type == 'function':
                config = condition.get('config', {})
                if not config.get('function_name'):
                    errors.append(f"Function condition missing 'function_name' in path {path}")
                if not config.get('module_path'):
                    errors.append(f"Function condition missing 'module_path' in path {path}")

        # Check for parallel execution opportunities
        parallel_opportunities = self._analyze_parallel_opportunities(paths, step_ids)
        if parallel_opportunities:
            warnings.append(f"Found {len(parallel_opportunities)} potential parallel execution opportunities")

        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'parallel_opportunities': parallel_opportunities
        }

    def _analyze_parallel_opportunities(self,
                                      paths: List[Dict[str, Any]],
                                      step_ids: set) -> List[Dict[str, Any]]:
        """Analyze flow for parallel execution opportunities"""
        opportunities = []

        # Group paths by source step
        paths_by_source = {}
        for path in paths:
            source = path.get('from_step')
            if source not in paths_by_source:
                paths_by_source[source] = []
            paths_by_source[source].append(path)

        # Find steps with multiple outgoing paths that could be parallel
        for source_step, outgoing_paths in paths_by_source.items():
            if len(outgoing_paths) > 1:
                target_steps = [path.get('to_step') for path in outgoing_paths]

                # Check if target steps are independent (no dependencies between them)
                if self._are_steps_independent(target_steps, paths):
                    opportunities.append({
                        'source_step': source_step,
                        'parallel_targets': target_steps,
                        'potential_speedup': len(target_steps)
                    })

        return opportunities

    def _are_steps_independent(self,
                              step_ids: List[str],
                              all_paths: List[Dict[str, Any]]) -> bool:
        """Check if a set of steps can be executed independently"""
        # Simple check: no step in the set depends on another step in the set
        for path in all_paths:
            source = path.get('from_step')
            target = path.get('to_step')

            if source in step_ids and target in step_ids:
                return False  # There's a dependency within the set

        return True

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get statistics about the current execution"""
        return {
            'execution_id': self.execution_id,
            'active_parallel_tasks': self.parallel_executor.get_active_step_count(),
            'router_type': type(self.router).__name__,
            'parallel_executor_type': type(self.parallel_executor).__name__
        }