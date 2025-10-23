"""
Test configuration and fixtures
"""

import asyncio
import os
import sys
from pathlib import Path
import pytest

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ia_modules.database import DatabaseManager, ConnectionConfig, DatabaseType

# This file can be used to define pytest fixtures that are shared across all tests

# Configure asyncio for pytest
pytest_plugins = [
    "pytest_asyncio",
]

# Set up event loop policy for Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# Database connection configurations for testing
def get_database_configs():
    """
    Get database configurations for testing.

    Environment variables:
    - TEST_POSTGRESQL_URL: PostgreSQL connection URL (e.g., postgresql://user:pass@localhost:5432/testdb)
    - TEST_MYSQL_URL: MySQL connection URL (e.g., mysql://user:pass@localhost:3306/testdb)
    - TEST_MSSQL_URL: MSSQL connection URL (e.g., mssql://user:pass@localhost:1433/testdb)

    If not set, those database types will be skipped in parameterized tests.
    """
    configs = [
        ("sqlite", ConnectionConfig(
            database_type=DatabaseType.SQLITE,
            database_url="sqlite://:memory:"
        ))
    ]

    # PostgreSQL
    postgresql_url = os.environ.get("TEST_POSTGRESQL_URL")
    if postgresql_url:
        configs.append(("postgresql", ConnectionConfig(
            database_type=DatabaseType.POSTGRESQL,
            database_url=postgresql_url
        )))

    # MySQL
    mysql_url = os.environ.get("TEST_MYSQL_URL")
    if mysql_url:
        configs.append(("mysql", ConnectionConfig(
            database_type=DatabaseType.MYSQL,
            database_url=mysql_url
        )))

    # MSSQL
    mssql_url = os.environ.get("TEST_MSSQL_URL")
    if mssql_url:
        configs.append(("mssql", ConnectionConfig(
            database_type=DatabaseType.MSSQL,
            database_url=mssql_url
        )))

    return configs


@pytest.fixture(params=get_database_configs(), ids=lambda x: x[0])
def db_config(request):
    """Parameterized fixture that provides database configs for all available databases"""
    return request.param[1]


@pytest.fixture(params=get_database_configs(), ids=lambda x: x[0])
async def db_manager(request):
    """Parameterized fixture that provides DatabaseManager instances for all available databases"""
    config = request.param[1]
    db = DatabaseManager(config)
    db.connect()

    yield db

    # Cleanup
    db.disconnect()
