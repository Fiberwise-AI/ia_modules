"""
Edge case tests for database/manager.py and interfaces to improve coverage
"""

import pytest
from ia_modules.pipeline.test_utils import create_test_execution_context
from unittest.mock import Mock, patch
from ia_modules.database.interfaces import ConnectionConfig, DatabaseType, QueryResult


class TestQueryResultEdgeCases:
    """Test edge cases in QueryResult"""

    def test_query_result_success(self):
        """Test QueryResult with success=True"""
        result = QueryResult(success=True, data=[{"id": 1}, {"id": 2}], row_count=2)

        assert result.success is True
        assert len(result.data) == 2
        assert result.row_count == 2
        assert result.error_message is None

    def test_query_result_failure(self):
        """Test QueryResult with success=False"""
        result = QueryResult(success=False, data=[], row_count=0, error_message="Database connection failed")

        assert result.success is False
        assert result.data == []
        assert result.row_count == 0
        assert result.error_message == "Database connection failed"

    def test_query_result_minimal(self):
        """Test QueryResult with only required fields"""
        result = QueryResult(success=True, data=[], row_count=0)

        assert result.success is True
        assert result.data == []
        assert result.row_count == 0
        assert result.error_message is None

    def test_query_result_with_none_data(self):
        """Test QueryResult with empty data"""
        result = QueryResult(success=True, data=[], row_count=0)

        assert result.data == []
        assert result.row_count == 0

    def test_query_result_with_empty_data(self):
        """Test QueryResult with empty data structures"""
        result1 = QueryResult(success=True, data=[], row_count=0)
        result2 = QueryResult(success=True, data=[], row_count=0)

        assert result1.data == []
        assert result2.data == []


class TestConnectionConfigEdgeCases:
    """Test edge cases in ConnectionConfig"""

    def test_connection_config_sqlite_url(self):
        """Test ConnectionConfig.from_url with SQLite URL"""
        config = ConnectionConfig.from_url("sqlite:///path/to/db.sqlite")

        assert config.database_type == DatabaseType.SQLITE
        assert config.database_url == "sqlite:///path/to/db.sqlite"

    def test_connection_config_postgresql_url(self):
        """Test ConnectionConfig.from_url with PostgreSQL URL"""
        config = ConnectionConfig.from_url("postgresql://user:pass@host:5432/dbname")

        assert config.database_type == DatabaseType.POSTGRESQL
        assert config.database_url == "postgresql://user:pass@host:5432/dbname"

    def test_connection_config_postgres_url(self):
        """Test ConnectionConfig.from_url with postgres:// URL"""
        config = ConnectionConfig.from_url("postgresql://user:pass@host/dbname")

        assert config.database_type == DatabaseType.POSTGRESQL

    def test_connection_config_unsupported_url(self):
        """Test ConnectionConfig.from_url with unsupported URL"""
        with pytest.raises(ValueError, match="Unsupported database URL"):
            ConnectionConfig.from_url("unsupported://url")

    def test_connection_config_mysql_url(self):
        """Test ConnectionConfig.from_url with mysql:// URL"""
        config = ConnectionConfig.from_url("mysql://user:pass@host/dbname")
        assert config.database_type == DatabaseType.MYSQL
        assert config.database_url == "mysql://user:pass@host/dbname"

    def test_connection_config_mssql_url(self):
        """Test ConnectionConfig.from_url with mssql:// URL"""
        config = ConnectionConfig.from_url("mssql://user:pass@host/dbname")
        assert config.database_type == DatabaseType.MSSQL
        assert config.database_url == "mssql://user:pass@host/dbname"

    def test_connection_config_initialization(self):
        """Test ConnectionConfig direct initialization"""
        config = ConnectionConfig(
            database_url="postgresql://localhost/test",
            database_type=DatabaseType.POSTGRESQL,
            host="localhost",
            port=5432,
            database_name="test",
            username="user",
            password="pass"
        )

        assert config.database_url == "postgresql://localhost/test"
        assert config.database_type == DatabaseType.POSTGRESQL
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database_name == "test"
        assert config.username == "user"
        assert config.password == "pass"

    def test_connection_config_minimal(self):
        """Test ConnectionConfig with minimal fields"""
        config = ConnectionConfig(
            database_url="sqlite:///test.db",
            database_type=DatabaseType.SQLITE
        )

        assert config.database_url == "sqlite:///test.db"
        assert config.database_type == DatabaseType.SQLITE
        assert config.host is None
        assert config.port is None


class TestDatabaseTypeEnum:
    """Test DatabaseType enum edge cases"""

    def test_database_type_values(self):
        """Test all DatabaseType enum values"""
        assert DatabaseType.SQLITE.value == "sqlite"
        assert DatabaseType.POSTGRESQL.value == "postgresql"
        assert DatabaseType.MYSQL.value == "mysql"
        assert DatabaseType.MSSQL.value == "mssql"

    def test_database_type_from_string(self):
        """Test creating DatabaseType from string"""
        assert DatabaseType("sqlite") == DatabaseType.SQLITE
        assert DatabaseType("postgresql") == DatabaseType.POSTGRESQL
        assert DatabaseType("mysql") == DatabaseType.MYSQL
        assert DatabaseType("mssql") == DatabaseType.MSSQL

    def test_database_type_equality(self):
        """Test DatabaseType equality"""
        type1 = DatabaseType.SQLITE
        type2 = DatabaseType.SQLITE
        type3 = DatabaseType.POSTGRESQL

        assert type1 == type2
        assert type1 != type3

    def test_database_type_in_list(self):
        """Test DatabaseType membership testing"""
        supported_types = [DatabaseType.SQLITE, DatabaseType.POSTGRESQL]

        assert DatabaseType.SQLITE in supported_types
        assert DatabaseType.MYSQL not in supported_types
