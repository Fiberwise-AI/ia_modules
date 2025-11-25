"""
Service System for Pipeline Dependency Injection

Production-ready service registry for pipeline components.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import json


class LogEntry:
    """Represents a single log entry"""
    def __init__(self, level: str, message: str, step_name: Optional[str] = None, data: Optional[Dict[str, Any]] = None):
        self.timestamp = datetime.now()
        self.level = level
        self.message = message
        self.step_name = step_name
        self.data = data or {}


class CentralLoggingService:
    """Central logging service that collects logs during execution and writes to database"""

    def __init__(self):
        self.execution_logs: List[LogEntry] = []
        self.current_execution_id: Optional[str] = None

    def set_execution_id(self, execution_id: str):
        """Set the current execution ID for logging"""
        self.current_execution_id = execution_id

    def log(self, level: str, message: str, step_name: Optional[str] = None, data: Optional[Dict[str, Any]] = None):
        """Log a message to the central service"""
        entry = LogEntry(level, message, step_name, data)
        self.execution_logs.append(entry)

    def info(self, message: str, step_name: Optional[str] = None, data: Optional[Dict[str, Any]] = None):
        """Log an info message"""
        self.log("INFO", message, step_name, data)

    def error(self, message: str, step_name: Optional[str] = None, data: Optional[Dict[str, Any]] = None):
        """Log an error message"""
        self.log("ERROR", message, step_name, data)

    def warning(self, message: str, step_name: Optional[str] = None, data: Optional[Dict[str, Any]] = None):
        """Log a warning message"""
        self.log("WARNING", message, step_name, data)

    def success(self, message: str, step_name: Optional[str] = None, data: Optional[Dict[str, Any]] = None):
        """Log a success message"""
        self.log("SUCCESS", message, step_name, data)

    async def write_to_database(self, db_service):
        """Write all collected logs to the database"""
        if not self.current_execution_id or not db_service:
            return

        try:
            for log_entry in self.execution_logs:
                # Write each log entry to database
                if hasattr(db_service, 'log_execution_event'):
                    await db_service.log_execution_event(
                        execution_id=self.current_execution_id,
                        event_type=log_entry.level,
                        message=log_entry.message,
                        step_name=log_entry.step_name,
                        data=json.dumps(log_entry.data),
                        timestamp=log_entry.timestamp.isoformat()
                    )
        except Exception:
            pass  # Silently fail database logging

    def clear_logs(self):
        """Clear all collected logs"""
        self.execution_logs.clear()
        self.current_execution_id = None


class ServiceRegistry:
    """Simple service container for dependency injection"""

    def __init__(self):
        self._services: Dict[str, Any] = {}
        # Initialize central logging service
        self._services['central_logger'] = CentralLoggingService()

    def register(self, name: str, service: Any):
        """Register a service"""
        self._services[name] = service

    def get(self, name: str) -> Optional[Any]:
        """Get a service by name"""
        return self._services.get(name)

    def has(self, name: str) -> bool:
        """Check if service is registered"""
        return name in self._services

    async def cleanup_all(self):
        """Cleanup all services that support it"""
        for service in self._services.values():
            if hasattr(service, 'cleanup'):
                await service.cleanup()
