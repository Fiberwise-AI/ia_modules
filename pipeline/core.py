"""
Core Pipeline Classes

Production-ready implementations of Step, Pipeline, and StepLogger
with clean architecture and Windows-compatible logging.
"""

import json
import time
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    from .services import ServiceRegistry


class TemplateParameterResolver:
    """Template parameter resolution system for dynamic pipeline configuration"""

    @staticmethod
    def resolve_parameters(config_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve template parameters in configuration using context data.

        Supports template patterns like:
        - {pipeline_input.business_type}
        - {steps.geocoder.result.city}
        - {parameters.custom_value}
        """
        if isinstance(config_data, dict):
            resolved = {}
            for key, value in config_data.items():
                resolved[key] = TemplateParameterResolver.resolve_parameters(value, context)
            return resolved
        elif isinstance(config_data, list):
            return [TemplateParameterResolver.resolve_parameters(item, context) for item in config_data]
        elif isinstance(config_data, str):
            return TemplateParameterResolver._resolve_string_template(config_data, context)
        else:
            return config_data

    @staticmethod
    def _resolve_string_template(template: str, context: Dict[str, Any]) -> Any:
        """Resolve template placeholders in a string and convert to appropriate type"""
        def replace_placeholder(match):
            placeholder = match.group(1)
            value = TemplateParameterResolver._get_nested_value(context, placeholder)
            return str(value) if value is not None else match.group(0)  # Return original if not found

        # Match patterns like {pipeline_input.business_type}
        pattern = r'\{([^}]+)\}'
        resolved = re.sub(pattern, replace_placeholder, template)
        
        # If the entire string was a template and got resolved, return the actual value
        if re.fullmatch(pattern, template) and resolved != template:
            return resolved
        
        # Try to convert to appropriate type
        return TemplateParameterResolver._convert_value_type(resolved)

    @staticmethod
    def _convert_value_type(value: str) -> Any:
        """Convert string value to appropriate type (int, float, bool)"""
        if not isinstance(value, str):
            return value
            
        # Try boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
            
        # Try integer conversion
        try:
            if '.' not in value:
                return int(value)
        except ValueError:
            pass
            
        # Try float conversion
        try:
            return float(value)
        except ValueError:
            pass
            
        # Return as string if no conversion possible
        return value

    @staticmethod
    def _get_nested_value(obj: Dict[str, Any], path: str):
        """Extract nested value using dot notation"""
        try:
            keys = path.split('.')
            current = obj
            for key in keys:
                if isinstance(current, dict):
                    current = current.get(key)
                else:
                    return None
            return current
        except:
            return None

    @staticmethod
    def extract_template_parameters(config_data: Dict[str, Any]) -> List[str]:
        """Extract all template parameter references from configuration"""
        parameters = []

        def extract_from_value(value):
            if isinstance(value, dict):
                for v in value.values():
                    extract_from_value(v)
            elif isinstance(value, list):
                for item in value:
                    extract_from_value(item)
            elif isinstance(value, str):
                pattern = r'\{([^}]+)\}'
                matches = re.findall(pattern, value)
                parameters.extend(matches)

        extract_from_value(config_data)
        return list(set(parameters))  # Remove duplicates


class StepLogger:
    """Simple console logging with Windows compatibility"""
    
    def __init__(self, step_name: str, step_number: int, job_id: Optional[str] = None, db_manager=None):
        self.step_name = step_name
        self.step_number = step_number
        self.job_id = job_id or str(uuid.uuid4())[:8]
        self.db_manager = db_manager
        self.start_time = None
    
    def _log(self, message: str):
        """Log message - disabled for central logging system"""
        # Silent logging - all logging goes through central logger
        pass
    
    async def log_step_start(self, input_data: Dict[str, Any]):
        """Log step start"""
        self.start_time = time.time()
        self._log(f"Step {self.step_number}: {self.step_name} - Starting")
        
        # Log to database if available
        if self.db_manager:
            try:
                self.db_manager.execute(
                    """INSERT INTO pipeline_logs (job_id, step_name, step_number, event_type, timestamp, data) 
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (self.job_id, self.step_name, self.step_number, 'start', 
                     datetime.now().isoformat(), json.dumps(input_data, default=str))
                )
            except Exception as e:
                self._log(f"Warning: Could not log to database: {e}")
    
    async def log_step_complete(self, result: Any, next_step: Optional[str] = None):
        """Log step completion"""
        duration = time.time() - self.start_time if self.start_time else 0
        self._log(f"Step {self.step_number}: {self.step_name} - Completed ({duration:.2f}s)")
        
        # Log to database if available
        if self.db_manager:
            try:
                self.db_manager.execute(
                    """INSERT INTO pipeline_logs (job_id, step_name, step_number, event_type, timestamp, data, duration) 
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (self.job_id, self.step_name, self.step_number, 'complete', 
                     datetime.now().isoformat(), json.dumps(result, default=str), duration)
                )
            except Exception as e:
                self._log(f"Warning: Could not log to database: {e}")
    
    async def log_step_error(self, error: Exception):
        """Log step error"""
        duration = time.time() - self.start_time if self.start_time else 0
        self._log(f"Step {self.step_number}: {self.step_name} - ERROR: {error}")
        
        # Log to database if available
        if self.db_manager:
            try:
                self.db_manager.execute(
                    """INSERT INTO pipeline_logs (job_id, step_name, step_number, event_type, timestamp, data, duration) 
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (self.job_id, self.step_name, self.step_number, 'error', 
                     datetime.now().isoformat(), str(error), duration)
                )
            except Exception as e:
                self._log(f"Warning: Could not log to database: {e}")


class Step:
    """Base step class with simple service access"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.logger = None
        self.services = None
    
    def set_logger(self, logger: StepLogger):
        """Set the step logger"""
        self.logger = logger
    
    def set_services(self, services: Optional['ServiceRegistry']):
        """Set services for dependency injection"""
        self.services = services
    
    def get_db(self):
        """Get database manager - simple access method"""
        if self.services:
            db_service = self.services.get('database')
            if db_service:
                # Handle both DatabaseManager directly and wrapped services
                if hasattr(db_service, 'db_manager'):
                    return db_service.db_manager
                elif hasattr(db_service, 'execute'):
                    return db_service
        return None
    
    def get_http(self):
        """Get HTTP client - simple access method"""
        if self.services:
            http_service = self.services.get('http')
            if http_service:
                return http_service.client
        return None
    
    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the step with logging and error handling"""
        start_time = datetime.now()
        success = False
        error_message = None
        result_data = None

        try:
            if self.logger:
                await self.logger.log_step_start(data)

            # Log input data to central logger if available
            if self.services:
                central_logger = self.services.get('central_logger')
                if central_logger and hasattr(central_logger, 'log'):
                    # Log input data characteristics (not full data for large arrays)
                    input_summary = self._summarize_data(data, "input")
                    central_logger.log("INFO", f"Step {self.name} input: {input_summary}", self.name, {"input_summary": input_summary})

            result = await self.work(data)
            result_data = result
            merged_result = {**data, **result}
            merged_result[self.name] = result

            # Log output data to central logger if available
            if self.services:
                central_logger = self.services.get('central_logger')
                if central_logger and hasattr(central_logger, 'log'):
                    # Log output data characteristics
                    output_summary = self._summarize_data(result, "output")
                    central_logger.log("INFO", f"Step {self.name} output: {output_summary}", self.name, {"output_summary": output_summary})

            if self.logger:
                await self.logger.log_step_complete(result)

            success = True
            return merged_result

        except Exception as e:
            error_message = str(e)
            if self.logger:
                await self.logger.log_step_error(e)
            raise

        finally:
            # Log step execution to database
            self._log_step_execution_to_database(success, start_time, error_message, result_data)

    def _log_step_execution_to_database(self, success: bool, start_time: datetime, error_message: Optional[str] = None, result: Optional[Any] = None):
        """Log step execution to database"""
        if not self.services:
            return

        # Get database service
        db_service = self.services.get('database')
        if not db_service:
            return

        # Get execution ID from central logger
        central_logger = self.services.get('central_logger')
        execution_id = None
        if central_logger and hasattr(central_logger, 'current_execution_id'):
            execution_id = central_logger.current_execution_id

        if not execution_id:
            return

        # Calculate duration
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Log step execution
        db_service.log_step_execution(
            execution_id=execution_id,
            step_name=self.name,
            success=success,
            duration=duration,
            error=error_message,
            result=result
        )

    def _summarize_data(self, data: Any, data_type: str) -> str:
        """Create a summary of data for logging (avoid logging huge arrays)"""
        if data is None:
            return "None"

        if isinstance(data, dict):
            summary = {}
            for key, value in data.items():
                if isinstance(value, (list, tuple)) and len(value) > 5:
                    summary[key] = f"array[{len(value)}] of {type(value[0]).__name__ if value else 'unknown'}"
                elif isinstance(value, (int, float, str, bool)):
                    summary[key] = value
                else:
                    summary[key] = f"{type(value).__name__}"
            return str(summary)

        elif isinstance(data, (list, tuple)):
            if len(data) > 5:
                return f"array[{len(data)}] of {type(data[0]).__name__ if data else 'unknown'}"
            else:
                return str(data)

        else:
            return f"{data} ({type(data).__name__})"
    
    async def work(self, data: Dict[str, Any]) -> Any:
        """Override this method in subclasses"""
        return f"result from {self.name}"


class HumanInputStep(Step):
    """Special step type for human-in-the-loop workflows"""

    async def work(self, data: Dict[str, Any]) -> Any:
        """Handle human input step - must be overridden with specific implementation"""
        ui_schema = self.config.get('ui_schema', {})

        # This is a base implementation - in practice, this would integrate
        # with your HITL system to pause execution and wait for input
        return {
            'human_input_required': True,
            'ui_schema': ui_schema,
            'message': f'Human input step {self.name} - override with specific HITL implementation'
        }


class Pipeline:
    """Pipeline orchestrator with both sequential and graph-based execution"""

    def __init__(self, steps: List[Step], job_id: Optional[str] = None, services: Optional['ServiceRegistry'] = None,
                 structure: Optional[Dict[str, Any]] = None):
        self.steps = steps
        self.job_id = job_id or str(uuid.uuid4())[:8]
        self.services = services
        self.structure = structure  # Graph-based pipeline definition

        # Get database manager from services if available
        db_manager = None
        if services and services.get('database'):
            db_service = services.get('database')
            # Use the DatabaseManager directly
            if hasattr(db_service, 'db_manager'):
                db_manager = db_service.db_manager
            elif hasattr(db_service, 'execute'):  # It's already a DatabaseManager
                db_manager = db_service

        # Set up loggers for each step
        for i, step in enumerate(self.steps):
            logger = StepLogger(step.name, i + 1, self.job_id, db_manager)
            step.set_logger(logger)

            # Inject services into step if it supports it
            if hasattr(step, 'set_services'):
                step.set_services(services)

    def has_flow_definition(self) -> bool:
        """Check if pipeline has graph-based flow definition"""
        return self.structure and 'flow' in self.structure

    async def run(self, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run pipeline with graph-based execution"""
        data = data or {}

        if not self.has_flow_definition():
            raise ValueError("Pipeline requires flow definition for graph-based execution")

        return await self._execute_graph(data)

    async def _execute_graph(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Graph-based execution with conditional flows and template parameter resolution"""
        steps_map = {step.name: step for step in self.steps}
        flow = self.structure.get('flow', {})
        paths = flow.get('paths', [])

        step_results = {}

        # Initialize context for template resolution
        context = {
            'pipeline_input': data,
            'steps': step_results,
            'parameters': self.structure.get('parameters', {})
        }

        # Graph traversal logic
        current_step_id = flow.get('start_at')
        if not current_step_id:
            raise ValueError("Pipeline flow missing 'start_at' field")

        current_data = data.copy()

        while current_step_id and current_step_id not in ['end', 'end_with_success']:
            if current_step_id in ['end_with_failure']:
                raise Exception("Pipeline flow ended in a failure state.")

            step = steps_map.get(current_step_id)
            if not step:
                raise ValueError(f"Step '{current_step_id}' not found in pipeline definition.")

            # Resolve template parameters in step configuration before execution
            step = self._resolve_step_templates(step, context)

            # Execute step
            result_data = await step.run(current_data)
            step_results[current_step_id] = {
                'success': True,
                'result': result_data.get(step.name, {}),
                'output': result_data
            }

            # Update context with latest step results
            context['steps'] = step_results

            current_data = result_data

            # Find next step based on flow conditions
            next_step_id = self._find_next_step(current_step_id, paths, step_results[current_step_id])
            current_step_id = next_step_id

        return current_data

    def _resolve_step_templates(self, step: 'Step', context: Dict[str, Any]) -> 'Step':
        """Resolve template parameters in step configuration"""
        if not step.config:
            return step

        # Create a copy of the step with resolved configuration
        resolved_config = TemplateParameterResolver.resolve_parameters(step.config, context)

        # Create new step instance with resolved config
        resolved_step = step.__class__(step.name, resolved_config)
        resolved_step.set_logger(step.logger)
        resolved_step.set_services(step.services)

        return resolved_step

    def _find_next_step(self, current_step_id: str, paths: List[Dict], step_output: Dict) -> Optional[str]:
        """Find next step based on conditional flow paths"""
        # Try both 'from' and 'from_step' keys for compatibility
        outgoing_paths = []
        for p in paths:
            from_key = p.get('from') or p.get('from_step')
            if from_key == current_step_id:
                outgoing_paths.append(p)

        for path in outgoing_paths:  # Order in manifest is the priority
            condition = path.get('condition', {'type': 'always'})

            if self._evaluate_path_condition(condition, step_output):
                to_key = path.get('to') or path.get('to_step')
                return to_key

        return None

    def _evaluate_path_condition(self, condition: Dict, step_output: Dict) -> bool:
        """Evaluate path condition to determine if path should be taken"""
        condition_type = condition.get('type', 'always')
        config = condition.get('config', {})

        if condition_type == 'always':
            return True

        elif condition_type == 'expression':
            source = config.get('source', '')
            operator = config.get('operator', '')
            value = config.get('value')

            # Extract value using dot notation - check both result and output
            actual_value = self._get_nested_value(step_output.get('result', {}), source)

            if actual_value is None:
                # Also check the full output data
                actual_value = self._get_nested_value(step_output.get('output', {}), source)

            # Support both "==" and "equals" operators
            if operator in ['==', 'equals']:
                return actual_value == value
            elif operator == 'greater_than':
                try:
                    return float(actual_value) > float(value)
                except (ValueError, TypeError):
                    return False
            elif operator == 'greater_than_or_equal':
                try:
                    return float(actual_value) >= float(value)
                except (ValueError, TypeError):
                    return False
            elif operator == 'less_than':
                try:
                    return float(actual_value) < float(value)
                except (ValueError, TypeError):
                    return False
            elif operator == 'equals_ignore_case':
                return str(actual_value).lower() == str(value).lower()

        return False

    def _get_nested_value(self, obj: Dict, path: str):
        """Extract nested value using dot notation (e.g., 'result.score')"""
        try:
            keys = path.split('.')
            current = obj
            for key in keys:
                if isinstance(current, dict):
                    current = current.get(key)
                else:
                    return None
            return current
        except:
            return None


async def run_pipeline(pipeline: Pipeline, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Run a pipeline with input data"""
    return await pipeline.run(input_data)
