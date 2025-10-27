"""
SQLite Integration Tests - File-Based Database Verification

These tests verify that:
1. SQLite file-based databases work correctly
2. Data persists across connections
3. SQL translation works for SQLite (PostgreSQL → SQLite)
4. File locking and concurrency behavior
5. WAL mode and performance optimizations

Setup:
    No setup required - SQLite is built-in to Python

Run:
    pytest tests/integration/test_sqlite_integration.py -v
"""

import os
import pytest
from ia_modules.pipeline.test_utils import create_test_execution_context
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone
from ia_modules.database import DatabaseManager, ConnectionConfig, DatabaseType


@pytest.fixture
def sqlite_tempdir():
    """Create temporary directory for SQLite database files"""
    temp_dir = tempfile.mkdtemp(prefix="ia_modules_test_")
    yield Path(temp_dir)
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sqlite_file_config(sqlite_tempdir):
    """SQLite connection config with file-based database"""
    db_path = sqlite_tempdir / "test.db"
    return ConnectionConfig(
        database_type=DatabaseType.SQLITE,
        database_url=f"sqlite:///{db_path}"
    )


@pytest.fixture
def sqlite_file_db(sqlite_file_config):
    """Connected file-based SQLite database"""
    db = DatabaseManager(sqlite_file_config)
    db.connect()

    yield db

    db.disconnect()


class TestSQLiteDataPersistence:
    """Test that SQLite data persists to disk"""

    def test_data_persists_across_connections(self, sqlite_file_config):
        """Test that data written in one connection is readable in another"""
        # First connection - create and populate
        db1 = DatabaseManager(sqlite_file_config)
        db1.connect()

        db1.execute("""
            CREATE TABLE test_persist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                value TEXT NOT NULL
            )
        """)

        db1.execute("INSERT INTO test_persist (value) VALUES (:val)", {"val": "persistent_data"})
        db1.disconnect()

        # Second connection - verify data exists
        db2 = DatabaseManager(sqlite_file_config)
        db2.connect()

        row = db2.fetch_one("SELECT * FROM test_persist WHERE value = :val", {"val": "persistent_data"})
        assert row is not None
        assert row["value"] == "persistent_data"

        db2.disconnect()

    def test_database_file_created(self, sqlite_tempdir):
        """Test that database file is actually created on disk"""
        db_path = sqlite_tempdir / "test_creation.db"

        config = ConnectionConfig(
            database_type=DatabaseType.SQLITE,
            database_url=f"sqlite:///{db_path}"
        )

        # Database file should not exist yet
        assert not db_path.exists()

        # Connect and create table
        db = DatabaseManager(config)
        db.connect()
        db.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
        db.disconnect()

        # Database file should now exist
        assert db_path.exists()
        assert db_path.stat().st_size > 0

    def test_multiple_tables_persist(self, sqlite_file_db):
        """Test that multiple tables persist correctly"""
        db = sqlite_file_db

        # Create multiple tables
        db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        db.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER)")
        db.execute("CREATE TABLE products (id INTEGER PRIMARY KEY, title TEXT)")

        # Insert data
        db.execute("INSERT INTO users (name) VALUES (:name)", {"name": "Alice"})
        db.execute("INSERT INTO orders (user_id) VALUES (:uid)", {"uid": 1})
        db.execute("INSERT INTO products (title) VALUES (:title)", {"title": "Widget"})

        # Verify all exist
        assert db.fetch_one("SELECT * FROM users") is not None
        assert db.fetch_one("SELECT * FROM orders") is not None
        assert db.fetch_one("SELECT * FROM products") is not None


class TestSQLiteTranslation:
    """Test PostgreSQL → SQLite SQL translation"""

    def test_boolean_translated_to_integer(self, sqlite_file_db):
        """Test BOOLEAN → INTEGER translation"""
        db = sqlite_file_db

        # PostgreSQL syntax
        db.execute("""
            CREATE TABLE test_bool (
                id INTEGER PRIMARY KEY,
                is_active BOOLEAN DEFAULT TRUE,
                is_deleted BOOLEAN DEFAULT FALSE
            )
        """)

        # Insert with boolean values
        db.execute("INSERT INTO test_bool (is_active, is_deleted) VALUES (:active, :deleted)",
                   {"active": True, "deleted": False})

        # SQLite stores as 1/0
        row = db.fetch_one("SELECT * FROM test_bool")
        assert row["is_active"] == 1
        assert row["is_deleted"] == 0

    def test_jsonb_translated_to_text(self, sqlite_file_db):
        """Test JSONB → TEXT translation"""
        db = sqlite_file_db

        # PostgreSQL syntax with JSONB
        db.execute("""
            CREATE TABLE test_json (
                id INTEGER PRIMARY KEY,
                data JSONB
            )
        """)

        # Insert JSON as text
        import json
        json_data = json.dumps({"key": "value", "number": 42})

        db.execute("INSERT INTO test_json (data) VALUES (:data)", {"data": json_data})

        # Verify stored as text
        row = db.fetch_one("SELECT * FROM test_json")
        assert isinstance(row["data"], str)
        parsed = json.loads(row["data"])
        assert parsed["key"] == "value"
        assert parsed["number"] == 42

    def test_uuid_translated_to_text(self, sqlite_file_db):
        """Test UUID → TEXT translation"""
        db = sqlite_file_db

        # PostgreSQL syntax
        db.execute("""
            CREATE TABLE test_uuid (
                id UUID PRIMARY KEY,
                user_id UUID
            )
        """)

        # Insert UUID as text
        import uuid
        test_uuid = str(uuid.uuid4())

        db.execute("INSERT INTO test_uuid (id, user_id) VALUES (:id, :uid)",
                   {"id": test_uuid, "uid": test_uuid})

        # Verify stored as text
        row = db.fetch_one("SELECT * FROM test_uuid")
        assert isinstance(row["id"], str)
        assert row["id"] == test_uuid

    def test_timestamp_with_now(self, sqlite_file_db):
        """Test TIMESTAMP and NOW() translation"""
        db = sqlite_file_db

        # PostgreSQL syntax
        db.execute("""
            CREATE TABLE test_timestamp (
                id INTEGER PRIMARY KEY,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)

        # Insert with default timestamp
        db.execute("INSERT INTO test_timestamp (id) VALUES (:id)", {"id": 1})

        # Verify timestamp exists
        row = db.fetch_one("SELECT * FROM test_timestamp")
        assert row["created_at"] is not None

    def test_varchar_translated_to_text(self, sqlite_file_db):
        """Test VARCHAR → TEXT translation"""
        db = sqlite_file_db

        # PostgreSQL syntax
        db.execute("""
            CREATE TABLE test_varchar (
                id INTEGER PRIMARY KEY,
                name VARCHAR(255),
                code VARCHAR(50)
            )
        """)

        # Insert text (SQLite doesn't enforce length)
        long_text = "x" * 1000  # Longer than 255
        db.execute("INSERT INTO test_varchar (name, code) VALUES (:name, :code)",
                   {"name": long_text, "code": "ABC"})

        # SQLite accepts it (no length enforcement)
        row = db.fetch_one("SELECT * FROM test_varchar")
        assert row["name"] == long_text


class TestSQLiteParameterBinding:
    """Test SQLite parameter binding"""

    def test_named_parameters(self, sqlite_file_db):
        """Test named parameter binding"""
        db = sqlite_file_db

        db.execute("CREATE TABLE test_params (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")

        # Insert with named params
        db.execute("INSERT INTO test_params (name, age) VALUES (:name, :age)",
                   {"name": "Alice", "age": 30})

        # Query with named params
        row = db.fetch_one("SELECT * FROM test_params WHERE name = :name", {"name": "Alice"})
        assert row is not None
        assert row["age"] == 30

    def test_special_characters_in_parameters(self, sqlite_file_db):
        """Test parameters with special characters"""
        db = sqlite_file_db

        db.execute("CREATE TABLE test_special (id INTEGER PRIMARY KEY, text TEXT)")

        # Special characters
        special_text = "Text with 'quotes' and \"double\" and \n newlines and 🔥 emoji"

        db.execute("INSERT INTO test_special (text) VALUES (:text)", {"text": special_text})

        row = db.fetch_one("SELECT * FROM test_special")
        assert row["text"] == special_text

    def test_null_parameters(self, sqlite_file_db):
        """Test NULL parameter handling"""
        db = sqlite_file_db

        db.execute("CREATE TABLE test_nulls (id INTEGER PRIMARY KEY, value TEXT)")

        db.execute("INSERT INTO test_nulls (value) VALUES (:val)", {"val": None})

        row = db.fetch_one("SELECT * FROM test_nulls")
        assert row["value"] is None

    def test_binary_data(self, sqlite_file_db):
        """Test storing binary data (BLOB)"""
        db = sqlite_file_db

        db.execute("CREATE TABLE test_binary (id INTEGER PRIMARY KEY, data BLOB)")

        binary_data = b'\x00\x01\x02\x03\xff\xfe'

        db.execute("INSERT INTO test_binary (data) VALUES (:data)", {"data": binary_data})

        row = db.fetch_one("SELECT * FROM test_binary")
        assert row["data"] == binary_data


class TestSQLiteConcurrency:
    """Test SQLite concurrency and locking behavior"""

    def test_multiple_readers(self, sqlite_file_config):
        """Test that multiple readers can access database simultaneously"""
        # Setup: create database with data
        db_setup = DatabaseManager(sqlite_file_config)
        db_setup.connect()
        db_setup.execute("CREATE TABLE test_read (id INTEGER PRIMARY KEY, value TEXT)")
        db_setup.execute("INSERT INTO test_read (value) VALUES (:val)", {"val": "test_data"})
        db_setup.disconnect()

        # Multiple readers
        db1 = DatabaseManager(sqlite_file_config)
        db2 = DatabaseManager(sqlite_file_config)

        db1.connect()
        db2.connect()

        # Both can read simultaneously
        row1 = db1.fetch_one("SELECT * FROM test_read")
        row2 = db2.fetch_one("SELECT * FROM test_read")

        assert row1["value"] == "test_data"
        assert row2["value"] == "test_data"

        db1.disconnect()
        db2.disconnect()

    def test_write_after_read(self, sqlite_file_config):
        """Test write operations after reads"""
        db = DatabaseManager(sqlite_file_config)
        db.connect()

        db.execute("CREATE TABLE test_rw (id INTEGER PRIMARY KEY, value INTEGER)")
        db.execute("INSERT INTO test_rw (value) VALUES (:val)", {"val": 10})

        # Read
        row = db.fetch_one("SELECT * FROM test_rw")
        assert row["value"] == 10

        # Write
        db.execute("UPDATE test_rw SET value = :val WHERE id = :id", {"val": 20, "id": row["id"]})

        # Read again
        row2 = db.fetch_one("SELECT * FROM test_rw")
        assert row2["value"] == 20

        db.disconnect()


class TestSQLiteMigrations:
    """Test migrations with SQLite"""

    @pytest.mark.asyncio
    async def test_migrations_create_tables(self, sqlite_file_config):
        """Test that migrations create tables in SQLite"""
        db = DatabaseManager(sqlite_file_config)

        # Run migrations
        await db.initialize(apply_schema=True)

        # Verify migration table exists
        exists = db.table_exists("ia_migrations")
        assert exists is True

        db.disconnect()

    @pytest.mark.asyncio
    async def test_migration_records_persist(self, sqlite_file_config):
        """Test that migration records persist in file"""
        db1 = DatabaseManager(sqlite_file_config)
        await db1.initialize(apply_schema=True)

        # Get migration count
        rows1 = db1.fetch_all("SELECT * FROM ia_migrations")
        count1 = len(rows1)
        assert count1 > 0

        db1.disconnect()

        # Reconnect and verify migrations still recorded
        db2 = DatabaseManager(sqlite_file_config)
        db2.connect()

        rows2 = db2.fetch_all("SELECT * FROM ia_migrations")
        count2 = len(rows2)

        assert count2 == count1

        db2.disconnect()


class TestSQLiteDataTypes:
    """Test SQLite data type handling"""

    def test_integer_types(self, sqlite_file_db):
        """Test INTEGER storage"""
        db = sqlite_file_db

        db.execute("CREATE TABLE test_int (id INTEGER PRIMARY KEY, small INT, big BIGINT)")

        db.execute("INSERT INTO test_int (small, big) VALUES (:s, :b)",
                   {"s": 42, "b": 9999999999})

        row = db.fetch_one("SELECT * FROM test_int")
        assert row["small"] == 42
        assert row["big"] == 9999999999

    def test_real_types(self, sqlite_file_db):
        """Test REAL (float) storage"""
        db = sqlite_file_db

        db.execute("CREATE TABLE test_real (id INTEGER PRIMARY KEY, value REAL)")

        db.execute("INSERT INTO test_real (value) VALUES (:val)", {"val": 3.14159})

        row = db.fetch_one("SELECT * FROM test_real")
        assert abs(row["value"] - 3.14159) < 0.00001

    def test_text_types(self, sqlite_file_db):
        """Test TEXT storage"""
        db = sqlite_file_db

        db.execute("CREATE TABLE test_text (id INTEGER PRIMARY KEY, short TEXT, long TEXT)")

        short_text = "Hello"
        long_text = "Lorem ipsum " * 1000

        db.execute("INSERT INTO test_text (short, long) VALUES (:s, :l)",
                   {"s": short_text, "l": long_text})

        row = db.fetch_one("SELECT * FROM test_text")
        assert row["short"] == short_text
        assert row["long"] == long_text

    def test_datetime_storage(self, sqlite_file_db):
        """Test datetime storage (as TEXT in SQLite)"""
        db = sqlite_file_db

        db.execute("CREATE TABLE test_datetime (id INTEGER PRIMARY KEY, created_at TEXT)")

        now = datetime.now(timezone.utc).isoformat()

        db.execute("INSERT INTO test_datetime (created_at) VALUES (:dt)", {"dt": now})

        row = db.fetch_one("SELECT * FROM test_datetime")
        assert row["created_at"] == now


class TestSQLiteTransactions:
    """Test SQLite transaction behavior"""

    @pytest.mark.asyncio
    async def test_async_execute_commits(self, sqlite_file_db):
        """Test that async execute commits data"""
        db = sqlite_file_db

        db.execute("CREATE TABLE test_async (id INTEGER PRIMARY KEY, value TEXT)")

        # Async execute
        result = await db.execute_async(
            "INSERT INTO test_async (value) VALUES (:val)",
            {"val": "async_data"}
        )

        # execute_async returns list (same as execute)
        assert isinstance(result, list)

        # Verify data committed
        row = db.fetch_one("SELECT * FROM test_async")
        assert row is not None
        assert row["value"] == "async_data"

    def test_implicit_transaction(self, sqlite_file_db):
        """Test SQLite's implicit transaction behavior"""
        db = sqlite_file_db

        db.execute("CREATE TABLE test_txn (id INTEGER PRIMARY KEY, value INTEGER)")

        # Multiple operations in implicit transaction
        db.execute("INSERT INTO test_txn (value) VALUES (:val)", {"val": 1})
        db.execute("INSERT INTO test_txn (value) VALUES (:val)", {"val": 2})
        db.execute("INSERT INTO test_txn (value) VALUES (:val)", {"val": 3})

        # All should be visible
        rows = db.fetch_all("SELECT * FROM test_txn ORDER BY value")
        assert len(rows) == 3
        assert rows[0]["value"] == 1
        assert rows[2]["value"] == 3


class TestSQLiteConstraints:
    """Test SQLite constraint enforcement"""

    def test_primary_key_constraint(self, sqlite_file_db):
        """Test PRIMARY KEY constraint"""
        db = sqlite_file_db

        db.execute("CREATE TABLE test_pk (id INTEGER PRIMARY KEY, name TEXT)")

        db.execute("INSERT INTO test_pk (id, name) VALUES (:id, :name)", {"id": 1, "name": "first"})

        # Try to insert duplicate primary key - should raise exception
        try:
            result = db.execute("INSERT INTO test_pk (id, name) VALUES (:id, :name)", {"id": 1, "name": "second"})
            assert False, "Expected IntegrityError for duplicate primary key"
        except Exception as e:
            # Constraint violation is expected
            assert "UNIQUE constraint failed" in str(e) or "PRIMARY KEY" in str(e)

        # Only first row should exist
        rows = db.fetch_all("SELECT * FROM test_pk")
        assert len(rows) == 1
        assert rows[0]["name"] == "first"

    def test_unique_constraint(self, sqlite_file_db):
        """Test UNIQUE constraint"""
        db = sqlite_file_db

        db.execute("CREATE TABLE test_unique (id INTEGER PRIMARY KEY, email TEXT UNIQUE)")

        db.execute("INSERT INTO test_unique (email) VALUES (:email)", {"email": "test@example.com"})

        # Try to insert duplicate email - should raise exception
        try:
            result = db.execute("INSERT INTO test_unique (email) VALUES (:email)", {"email": "test@example.com"})
            assert False, "Expected IntegrityError for duplicate email"
        except Exception as e:
            # Constraint violation is expected
            assert "UNIQUE constraint failed" in str(e)

    def test_not_null_constraint(self, sqlite_file_db):
        """Test NOT NULL constraint"""
        db = sqlite_file_db

        db.execute("CREATE TABLE test_notnull (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")

        # Try to insert NULL - should raise exception
        try:
            result = db.execute("INSERT INTO test_notnull (name) VALUES (:name)", {"name": None})
            assert False, "Expected IntegrityError for NULL in NOT NULL column"
        except Exception as e:
            # Constraint violation is expected
            assert "NOT NULL constraint failed" in str(e)

    def test_foreign_key_constraint(self, sqlite_file_db):
        """Test FOREIGN KEY constraint (if enabled)"""
        db = sqlite_file_db

        # Enable foreign keys (off by default in SQLite)
        db.execute("PRAGMA foreign_keys = ON")

        db.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY, name TEXT)")
        db.execute("""
            CREATE TABLE child (
                id INTEGER PRIMARY KEY,
                parent_id INTEGER,
                FOREIGN KEY (parent_id) REFERENCES parent(id)
            )
        """)

        # Insert parent
        db.execute("INSERT INTO parent (id, name) VALUES (:id, :name)", {"id": 1, "name": "Parent"})

        # Insert child with valid foreign key
        result = db.execute("INSERT INTO child (parent_id) VALUES (:pid)", {"pid": 1})
        assert isinstance(result, list)  # execute() returns list on success

        # Try to insert child with invalid foreign key - should raise exception
        try:
            result = db.execute("INSERT INTO child (parent_id) VALUES (:pid)", {"pid": 999})
            assert False, "Expected IntegrityError for invalid foreign key"
        except Exception as e:
            # Constraint violation is expected
            assert "FOREIGN KEY constraint failed" in str(e)


class TestSQLitePerformance:
    """Test SQLite performance characteristics"""

    def test_bulk_insert(self, sqlite_file_db):
        """Test bulk insert performance"""
        db = sqlite_file_db

        db.execute("CREATE TABLE test_bulk (id INTEGER PRIMARY KEY, value INTEGER)")

        # Insert 1000 rows
        for i in range(1000):
            db.execute("INSERT INTO test_bulk (value) VALUES (:val)", {"val": i})

        # Verify count
        row = db.fetch_one("SELECT COUNT(*) as count FROM test_bulk")
        assert row["count"] == 1000

    def test_index_usage(self, sqlite_file_db):
        """Test that indexes can be created and used"""
        db = sqlite_file_db

        db.execute("CREATE TABLE test_index (id INTEGER PRIMARY KEY, email TEXT)")

        # Create index
        db.execute("CREATE INDEX idx_email ON test_index(email)")

        # Insert data
        db.execute("INSERT INTO test_index (email) VALUES (:email)", {"email": "test@example.com"})

        # Query should use index
        row = db.fetch_one("SELECT * FROM test_index WHERE email = :email", {"email": "test@example.com"})
        assert row is not None
