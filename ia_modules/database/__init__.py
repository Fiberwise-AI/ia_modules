"""
Database abstraction layer for ia_modules.

Supports multiple database backends:
- nexusql (default): Multi-database abstraction with SQL translation
- sqlalchemy: Standard Python ORM with advanced features

Usage:
    # Using nexusql (default)
    from ia_modules.database import get_database
    db = get_database("sqlite:///app.db")

    # Using SQLAlchemy
    from ia_modules.database import get_database
    db = get_database("sqlite:///app.db", backend="sqlalchemy")

    # Using environment variable
    export IA_DATABASE_BACKEND=sqlalchemy
    db = get_database("sqlite:///app.db")
"""

from .interfaces import DatabaseInterface, DatabaseBackend
from .factory import get_database, set_default_backend
from .adapters.nexusql_adapter import NexuSQLAdapter
from .adapters.sqlalchemy_adapter import SQLAlchemyAdapter

__all__ = [
    'DatabaseInterface',
    'DatabaseBackend',
    'get_database',
    'set_default_backend',
    'NexuSQLAdapter',
    'SQLAlchemyAdapter',
]
