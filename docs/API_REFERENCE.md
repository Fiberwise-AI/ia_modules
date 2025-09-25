# IA Modules - API Reference

## Overview

This comprehensive API reference covers all public interfaces, classes, methods, and functions in the IA Modules framework. The library is organized into logical modules for pipeline execution, database management, authentication, and web utilities.

## Table of Contents

- [Pipeline System](#pipeline-system)
- [Database System](#database-system)
- [Authentication System](#authentication-system)
- [Service Registry](#service-registry)
- [Utility Functions](#utility-functions)
- [Exception Classes](#exception-classes)
- [Type Definitions](#type-definitions)

## Pipeline System

### Core Classes

#### Step

Base class for all pipeline steps with service injection and logging capabilities.

```python
class Step:
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        Initialize a pipeline step.

        Args:
            name: Unique identifier for the step
            config: Step-specific configuration dictionary
        """
```

**Methods:**

- `async def work(self, data: Dict[str, Any]) -> Any`
  - **Purpose**: Main step execution logic (override in subclasses)
  - **Parameters**: `data` - Input data dictionary
  - **Returns**: Step output data
  - **Raises**: `NotImplementedError` if not overridden

- `async def run(self, data: Dict[str, Any]) -> Dict[str, Any]`
  - **Purpose**: Execute step with logging and error handling
  - **Parameters**: `data` - Input data dictionary
  - **Returns**: Merged input and output data
  - **Side Effects**: Logs execution, sends WebSocket notifications

- `def get_db(self) -> Optional[DatabaseManager]`
  - **Purpose**: Access injected database service
  - **Returns**: Database manager instance or None
  - **Usage**: `db = self.get_db(); result = db.fetch_all("SELECT * FROM table")`

- `def get_http(self) -> Optional[Any]`
  - **Purpose**: Access injected HTTP client service
  - **Returns**: HTTP client instance or None
  - **Usage**: `http = self.get_http(); response = await http.get(url)`

- `def set_services(self, services: ServiceRegistry)`
  - **Purpose**: Inject service registry for dependency access
  - **Parameters**: `services` - ServiceRegistry instance

- `def set_logger(self, logger: StepLogger)`
  - **Purpose**: Set step-specific logger
  - **Parameters**: `logger` - StepLogger instance

**Example Implementation:**

```python
class DataProcessorStep(Step):
    async def work(self, data: Dict[str, Any]) -> Any:
        # Access database service
        db = self.get_db()
        if not db:
            return {"error": "Database service unavailable"}

        # Process data
        processed_data = self._process(data['input'])

        # Store results
        await db.execute_query(
            "INSERT INTO processed_data (data, timestamp) VALUES (?, ?)",
            (json.dumps(processed_data), datetime.now())
        )

        return {
            "processed_data": processed_data,
            "status": "success"
        }
```

#### HumanInputStep

Special step type for human-in-the-loop workflows.

```python
class HumanInputStep(Step):
    async def work(self, data: Dict[str, Any]) -> Any:
        """
        Handle human input requirement.

        Returns:
            Dictionary with UI schema and human input flag
        """
```

**Configuration Example:**

```json
{
  "id": "human_review",
  "step_class": "HumanInputStep",
  "config": {
    "ui_schema": {
      "title": "Review Generated Content",
      "fields": [
        {
          "name": "approval",
          "type": "radio",
          "options": ["approve", "reject", "modify"]
        },
        {
          "name": "comments",
          "type": "textarea",
          "label": "Review Comments"
        }
      ]
    }
  }
}
```

#### Pipeline

Main orchestrator for step execution with graph-based flow control.

```python
class Pipeline:
    def __init__(
        self,
        steps: List[Step],
        job_id: Optional[str] = None,
        services: Optional[ServiceRegistry] = None,
        structure: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize pipeline orchestrator.

        Args:
            steps: List of step instances
            job_id: Unique job identifier
            services: Service registry for dependency injection
            structure: Graph structure for conditional flows
        """
```

**Methods:**

- `async def run(self, data: Dict[str, Any] = None) -> Dict[str, Any]`
  - **Purpose**: Execute pipeline with graph-based flow control
  - **Parameters**: `data` - Initial input data
  - **Returns**: Final pipeline output
  - **Raises**: `ValueError` if flow definition missing

- `def has_flow_definition(self) -> bool`
  - **Purpose**: Check if pipeline has graph-based flow definition
  - **Returns**: True if flow structure present

**Flow Structure:**

```json
{
  "flow": {
    "start_at": "step_1",
    "paths": [
      {
        "from": "step_1",
        "to": "step_2",
        "condition": {
          "type": "expression",
          "config": {
            "source": "result.quality_score",
            "operator": "greater_than",
            "value": 0.8
          }
        }
      }
    ]
  }
}
```

#### TemplateParameterResolver

Static utility class for resolving template parameters in pipeline configurations.

```python
class TemplateParameterResolver:
    @staticmethod
    def resolve_parameters(
        config_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resolve template parameters in configuration.

        Args:
            config_data: Configuration with template placeholders
            context: Context data for parameter resolution

        Returns:
            Configuration with resolved parameters
        """
```

**Template Syntax:**
- `{{ parameters.param_name }}` - Pipeline parameters
- `{{ pipeline_input.field }}` - Input data fields
- `{{ steps.step_id.result.field }}` - Step output fields

**Methods:**

- `extract_template_parameters(config_data: Dict[str, Any]) -> List[str]`
  - **Purpose**: Extract all template parameter references
  - **Returns**: List of parameter paths found in configuration

#### StepLogger

Logging utility for step execution tracking.

```python
class StepLogger:
    def __init__(
        self,
        step_name: str,
        step_number: int,
        job_id: Optional[str] = None,
        db_manager=None
    ):
        """
        Initialize step logger.

        Args:
            step_name: Name of the step being logged
            step_number: Sequential step number
            job_id: Unique job identifier
            db_manager: Database manager for persistent logging
        """
```

**Methods:**

- `async def log_step_start(self, input_data: Dict[str, Any])`
  - **Purpose**: Log step initiation
  - **Side Effects**: Records start time, logs to database if available

- `async def log_step_complete(self, result: Any)`
  - **Purpose**: Log successful step completion
  - **Side Effects**: Records duration, logs results

- `async def log_step_error(self, error: Exception)`
  - **Purpose**: Log step execution error
  - **Side Effects**: Records error details and duration

### Runner Functions

#### run_pipeline_from_json

Main entry point for executing pipelines from JSON configuration files.

```python
async def run_pipeline_from_json(
    pipeline_file: str,
    input_data: Dict[str, Any] = None,
    services: Optional[ServiceRegistry] = None,
    working_directory: Optional[str] = None,
    websocket_manager=None,
    user_id: int = None,
    execution_id: str = None
) -> Dict[str, Any]:
    """
    Execute pipeline from JSON configuration file.

    Args:
        pipeline_file: Path to JSON pipeline configuration
        input_data: Initial input data dictionary
        services: Service registry for dependency injection
        working_directory: Directory for Python module imports
        websocket_manager: WebSocket manager for real-time updates
        user_id: User ID for WebSocket notifications
        execution_id: Execution ID for tracking

    Returns:
        Final pipeline execution result

    Raises:
        FileNotFoundError: If pipeline file doesn't exist
        ValueError: If pipeline_file is empty
        ImportError: If step modules cannot be imported
    """
```

#### create_pipeline_from_json

Factory function for creating pipeline instances from JSON configurations.

```python
def create_pipeline_from_json(
    pipeline_config: Dict[str, Any],
    services: Optional[ServiceRegistry] = None
) -> Pipeline:
    """
    Create pipeline instance from JSON configuration.

    Args:
        pipeline_config: Pipeline configuration dictionary
        services: Service registry for dependency injection

    Returns:
        Configured Pipeline instance
    """
```

#### create_step_from_json

Factory function for creating step instances from JSON definitions.

```python
def create_step_from_json(
    step_def: Dict[str, Any],
    context: Dict[str, Any] = None
) -> Step:
    """
    Create step instance from JSON definition.

    Args:
        step_def: Step definition dictionary
        context: Context for template parameter resolution

    Returns:
        Configured Step instance

    Raises:
        ImportError: If step module cannot be imported
        AttributeError: If step class not found in module
    """
```

## Database System

### Interface Classes

#### DatabaseInterface

Abstract base class defining database operation contracts.

```python
class DatabaseInterface(ABC):
    def __init__(self, connection_string: str, db_type: DatabaseType):
        """
        Initialize database interface.

        Args:
            connection_string: Database connection string
            db_type: Type of database (SQLite, PostgreSQL, etc.)
        """
```

**Abstract Methods:**

- `async def connect(self) -> bool`
  - **Purpose**: Establish database connection
  - **Returns**: True if connection successful

- `async def disconnect(self) -> None`
  - **Purpose**: Close database connection

- `async def execute_query(self, query: str, parameters: Optional[Tuple] = None) -> QueryResult`
  - **Purpose**: Execute SQL query with optional parameters
  - **Parameters**:
    - `query` - SQL query string
    - `parameters` - Optional parameter tuple
  - **Returns**: QueryResult with execution details

- `async def fetch_all(self, query: str, parameters: Optional[Tuple] = None) -> QueryResult`
  - **Purpose**: Execute query and fetch all results
  - **Returns**: QueryResult with all matching rows

- `async def fetch_one(self, query: str, parameters: Optional[Tuple] = None) -> QueryResult`
  - **Purpose**: Execute query and fetch single result
  - **Returns**: QueryResult with first matching row

- `async def table_exists(self, table_name: str) -> bool`
  - **Purpose**: Check if table exists in database
  - **Returns**: True if table exists

**Concrete Methods:**

- `async def health_check(self) -> bool`
  - **Purpose**: Verify database connectivity
  - **Returns**: True if database is responsive

#### DatabaseManager

Concrete database implementation with SQLite support.

```python
class DatabaseManager:
    def __init__(self, database_url: str):
        """
        Initialize database manager.

        Args:
            database_url: Database connection URL
        """
```

**Methods:**

- `async def initialize(self, apply_schema: bool = True, app_migration_paths: Optional[List[str]] = None) -> bool`
  - **Purpose**: Initialize database with optional schema application
  - **Parameters**:
    - `apply_schema` - Whether to run migrations
    - `app_migration_paths` - List of custom migration directories
  - **Returns**: True if initialization successful

- `def execute(self, query: str, params: Optional[tuple] = None) -> Any`
  - **Purpose**: Execute SQL query with parameter binding
  - **Returns**: Database cursor object

- `def fetch_one(self, query: str, params: Optional[tuple] = None) -> Optional[Dict]`
  - **Purpose**: Execute query and return single row
  - **Returns**: Dictionary representation of row or None

- `def fetch_all(self, query: str, params: Optional[tuple] = None) -> List[Dict]`
  - **Purpose**: Execute query and return all rows
  - **Returns**: List of dictionary representations

- `def create_table(self, table_name: str, schema: str)`
  - **Purpose**: Create table with specified schema
  - **Parameters**:
    - `table_name` - Name of table to create
    - `schema` - SQL column definitions

**Context Manager Support:**

```python
# Automatic connection management
with DatabaseManager("sqlite:///app.db") as db:
    users = db.fetch_all("SELECT * FROM users WHERE active = ?", (True,))
```

#### QueryResult

Standardized result container for database operations.

```python
@dataclass
class QueryResult:
    success: bool
    data: List[Dict[str, Any]]
    row_count: int
    error_message: Optional[str] = None
    execution_time_ms: Optional[float] = None
```

**Methods:**

- `def get_first_row(self) -> Optional[Dict[str, Any]]`
  - **Purpose**: Get first row from results
  - **Returns**: First row dictionary or None

- `def get_column_values(self, column_name: str) -> List[Any]`
  - **Purpose**: Extract all values for specific column
  - **Returns**: List of column values

#### ConnectionConfig

Database connection configuration container.

```python
@dataclass
class ConnectionConfig:
    database_type: DatabaseType
    database_url: str
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    database_name: Optional[str] = None
```

**Class Methods:**

- `@classmethod def from_url(cls, database_url: str) -> 'ConnectionConfig'`
  - **Purpose**: Create configuration from database URL
  - **Supports**: SQLite, PostgreSQL, DuckDB URLs
  - **Raises**: `ValueError` for unsupported URLs

#### DatabaseType

Enumeration of supported database types.

```python
class DatabaseType(Enum):
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    DUCKDB = "duckdb"
    CLOUDFLARE_D1 = "cloudflare_d1"
```

### Migration System

#### MigrationRunner

Database schema migration management system.

```python
class MigrationRunner:
    def __init__(
        self,
        database: DatabaseInterface,
        migration_path: Optional[Path] = None,
        migration_type: str = "app"
    ):
        """
        Initialize migration runner.

        Args:
            database: Database interface for migration execution
            migration_path: Directory containing migration files
            migration_type: Type of migrations ("system" or "app")
        """
```

**Methods:**

- `async def run_pending_migrations(self) -> bool`
  - **Purpose**: Execute all pending migrations
  - **Returns**: True if all migrations successful
  - **Side Effects**: Updates migration tracking table

- `async def get_pending_migrations(self) -> List[Path]`
  - **Purpose**: Get list of unapplied migration files
  - **Returns**: Sorted list of migration file paths

- `async def run_specific_migration(self, version: str) -> bool`
  - **Purpose**: Execute specific migration by version
  - **Parameters**: `version` - Migration version identifier
  - **Returns**: True if migration successful

- `async def get_applied_migrations(self) -> List[MigrationRecord]`
  - **Purpose**: Get history of applied migrations
  - **Returns**: List of MigrationRecord objects

#### MigrationRecord

Record of applied database migration.

```python
@dataclass
class MigrationRecord:
    version: str
    description: str
    applied_at: datetime
    checksum: Optional[str] = None
```

### Utility Functions

- `def create_query_result(success: bool = True, data: List[Dict[str, Any]] = None, **kwargs) -> QueryResult`
  - **Purpose**: Create QueryResult with defaults
  - **Returns**: Configured QueryResult instance

- `def create_error_result(error_message: str) -> QueryResult`
  - **Purpose**: Create error QueryResult
  - **Returns**: QueryResult with error state

## Authentication System

### User Models

#### CurrentUser

Authenticated user data container.

```python
@dataclass
class CurrentUser:
    id: int
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    role: str = "user"
    facility_id: Optional[int] = None
    active: bool = True
    is_admin: bool = False
    is_super_admin: bool = False
    is_facility_admin: bool = False
    username: Optional[str] = None
```

**Properties:**

- `@property def full_name(self) -> str`
  - **Purpose**: Get formatted full name
  - **Returns**: Combined first/last name or email fallback

**Methods:**

- `def to_dict(self) -> Dict[str, Any]`
  - **Purpose**: Serialize user to dictionary
  - **Returns**: Dictionary representation for JSON serialization

#### UserRole

User role enumeration.

```python
class UserRole(Enum):
    USER = "user"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"
    FACILITY_ADMIN = "facility_admin"
```

## Service Registry

### ServiceRegistry

Dependency injection container for pipeline services.

```python
class ServiceRegistry:
    def __init__(self):
        """Initialize service registry with central logging service."""
```

**Methods:**

- `def register(self, name: str, service: Any)`
  - **Purpose**: Register service with registry
  - **Parameters**:
    - `name` - Service identifier
    - `service` - Service instance

- `def get(self, name: str) -> Optional[Any]`
  - **Purpose**: Retrieve registered service
  - **Parameters**: `name` - Service identifier
  - **Returns**: Service instance or None

- `def has(self, name: str) -> bool`
  - **Purpose**: Check if service is registered
  - **Returns**: True if service exists

- `async def initialize_all()`
  - **Purpose**: Initialize all services that support it
  - **Side Effects**: Calls `initialize()` method on supporting services

- `async def cleanup_all()`
  - **Purpose**: Cleanup all services that support it
  - **Side Effects**: Calls `cleanup()` method on supporting services

### CentralLoggingService

Centralized logging service for pipeline execution tracking.

```python
class CentralLoggingService:
    def __init__(self):
        """Initialize central logging service."""
```

**Methods:**

- `def set_execution_id(self, execution_id: str)`
  - **Purpose**: Set current execution ID for log correlation
  - **Parameters**: `execution_id` - Unique execution identifier

- `def log(self, level: str, message: str, step_name: Optional[str] = None, data: Optional[Dict[str, Any]] = None)`
  - **Purpose**: Log message with specified level
  - **Parameters**:
    - `level` - Log level (INFO, ERROR, WARNING, SUCCESS)
    - `message` - Log message text
    - `step_name` - Optional step name for context
    - `data` - Optional structured data

**Convenience Methods:**

- `def info(self, message: str, step_name: Optional[str] = None, data: Optional[Dict[str, Any]] = None)`
- `def error(self, message: str, step_name: Optional[str] = None, data: Optional[Dict[str, Any]] = None)`
- `def warning(self, message: str, step_name: Optional[str] = None, data: Optional[Dict[str, Any]] = None)`
- `def success(self, message: str, step_name: Optional[str] = None, data: Optional[Dict[str, Any]] = None)`

**Persistence Methods:**

- `async def write_to_database(self, db_service)`
  - **Purpose**: Write collected logs to database
  - **Parameters**: `db_service` - Database service for log persistence

- `def clear_logs()`
  - **Purpose**: Clear all collected log entries

## Utility Functions

### Module Loading

- `def load_step_class(module_path: str, class_name: str)`
  - **Purpose**: Dynamically import and return step class
  - **Parameters**:
    - `module_path` - Python module path
    - `class_name` - Class name within module
  - **Returns**: Step class object
  - **Raises**: `ImportError`, `AttributeError`

### Pipeline Execution

- `async def run_pipeline(pipeline: Pipeline, input_data: Dict[str, Any] = None) -> Dict[str, Any]`
  - **Purpose**: Execute pipeline instance with input data
  - **Parameters**:
    - `pipeline` - Pipeline instance to execute
    - `input_data` - Initial data dictionary
  - **Returns**: Final execution result

## Exception Classes

### Standard Exceptions

The framework uses standard Python exceptions with specific contexts:

- `ImportError` - Module or class import failures
- `AttributeError` - Missing class or method attributes
- `ValueError` - Invalid configuration or parameter values
- `FileNotFoundError` - Missing pipeline or migration files
- `RuntimeError` - Database connection or execution errors

## Type Definitions

### Common Type Aliases

```python
from typing import Dict, Any, List, Optional, Tuple, Union

# Common data structures
ConfigDict = Dict[str, Any]
InputData = Dict[str, Any]
OutputData = Dict[str, Any]
ParameterTuple = Optional[Tuple]
ServiceMap = Dict[str, Any]

# Step and pipeline types
StepDefinition = Dict[str, Any]
PipelineConfig = Dict[str, Any]
FlowDefinition = Dict[str, Any]
```

### Example Usage Patterns

#### Basic Pipeline Execution

```python
from ia_modules.pipeline.runner import run_pipeline_from_json
from ia_modules.pipeline.services import ServiceRegistry
from ia_modules.database.manager import DatabaseManager

# Setup services
db_manager = DatabaseManager("sqlite:///app.db")
await db_manager.initialize()

services = ServiceRegistry()
services.register('database', db_manager)

# Execute pipeline
result = await run_pipeline_from_json(
    pipeline_file="./pipelines/data_processing.json",
    input_data={"source_data": data},
    services=services
)
```

#### Custom Step Implementation

```python
from ia_modules.pipeline.core import Step
from typing import Dict, Any

class CustomAnalysisStep(Step):
    async def work(self, data: Dict[str, Any]) -> Any:
        # Access configuration
        threshold = self.config.get('threshold', 0.5)

        # Access database service
        db = self.get_db()
        historical_data = db.fetch_all(
            "SELECT * FROM analysis_history WHERE type = ?",
            (data['analysis_type'],)
        )

        # Perform analysis
        result = self._analyze(data['input'], historical_data, threshold)

        # Store results
        db.execute(
            "INSERT INTO analysis_results (data, score, timestamp) VALUES (?, ?, ?)",
            (json.dumps(result), result['score'], datetime.now())
        )

        return {
            "analysis_result": result,
            "confidence": result['score'],
            "status": "completed"
        }

    def _analyze(self, input_data, historical_data, threshold):
        # Custom analysis logic
        pass
```

#### Database Migration Usage

```python
from ia_modules.database.migrations import MigrationRunner
from ia_modules.database.manager import DatabaseManager

# Setup database
db_manager = DatabaseManager("sqlite:///app.db")
await db_manager.initialize(apply_schema=False)

# Run migrations
migration_runner = MigrationRunner(
    database=db_manager,
    migration_path=Path("./migrations"),
    migration_type="app"
)

success = await migration_runner.run_pending_migrations()
if success:
    print("All migrations completed successfully")
```

This API reference provides comprehensive coverage of all public interfaces in the IA Modules framework, enabling developers to effectively utilize the pipeline system, database abstractions, authentication components, and service injection capabilities.