"""
Database interfaces and configuration classes
Consolidated from interfaces/database.py for easier imports
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum


class DatabaseType(Enum):
    """Database types"""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MSSQL = "mssql"
    DUCKDB = "duckdb"
    CLOUDFLARE_D1 = "cloudflare_d1"


@dataclass
class ConnectionConfig:
    """Database connection configuration"""
    database_type: DatabaseType
    database_url: str
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    database_name: Optional[str] = None
    
    @classmethod
    def from_url(cls, database_url: str) -> 'ConnectionConfig':
        """Create configuration from database URL"""
        if database_url.startswith('sqlite'):
            return cls(database_type=DatabaseType.SQLITE, database_url=database_url)
        elif database_url.startswith('postgresql') or database_url.startswith('postgres'):
            return cls(database_type=DatabaseType.POSTGRESQL, database_url=database_url)
        elif database_url.startswith('mysql'):
            return cls(database_type=DatabaseType.MYSQL, database_url=database_url)
        elif database_url.startswith('mssql'):
            return cls(database_type=DatabaseType.MSSQL, database_url=database_url)
        elif database_url.startswith('duckdb'):
            return cls(database_type=DatabaseType.DUCKDB, database_url=database_url)
        else:
            raise ValueError(f"Unsupported database URL: {database_url}")


@dataclass
class QueryResult:
    """Result from database query"""
    success: bool
    data: List[Dict[str, Any]]
    row_count: int
    error_message: Optional[str] = None
    execution_time_ms: Optional[float] = None
    
    def __post_init__(self):
        """Update row count based on data"""
        if self.data:
            self.row_count = len(self.data)
    
    def get_first_row(self) -> Optional[Dict[str, Any]]:
        """Get first row if available"""
        return self.data[0] if self.data else None
    
    def get_column_values(self, column_name: str) -> List[Any]:
        """Get all values for a specific column"""
        return [row.get(column_name) for row in self.data if column_name in row]


class DatabaseInterface(ABC):
    """
    Abstract database interface for framework-agnostic data access.
    Provides standardized methods for database operations.
    """
    
    def __init__(self, connection_string: str, db_type: DatabaseType):
        self.connection_string = connection_string
        self.db_type = db_type
        self.is_connected = False
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the database"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the database"""
        pass

    @abstractmethod
    async def fetch_all(self, query: str, parameters: Optional[Dict] = None) -> QueryResult:
        """Fetch all results from a query"""
        pass
    
    @abstractmethod
    async def fetch_one(self, query: str, parameters: Optional[Dict] = None) -> QueryResult:
        """Fetch one result from a query"""
        pass
    
    @abstractmethod
    async def table_exists(self, table_name: str) -> bool:
        """Check if table exists"""
        pass
    
    async def health_check(self) -> bool:
        """Check database health"""
        try:
            result = await self.execute_async("SELECT 1 as health_check")
            return result.success
        except Exception:
            return False


# Utility functions
def create_query_result(success: bool = True, data: List[Dict[str, Any]] = None, **kwargs) -> QueryResult:
    """Create a query result with common defaults"""
    return QueryResult(success=success, data=data or [], row_count=len(data) if data else 0, **kwargs)


def create_error_result(error_message: str) -> QueryResult:
    """Create an error query result"""
    return QueryResult(success=False, data=[], row_count=0, error_message=error_message)