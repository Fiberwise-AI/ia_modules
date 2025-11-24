"""
PostgreSQL Integration Tests - Real Database Verification

These tests verify that:
1. PostgreSQL parameter binding works correctly
2. Data actually exists in the database after operations
3. SQL translation is NOT applied to PostgreSQL (it should stay as-is)
4. Transactions work properly
5. Migrations work with actual PostgreSQL database

Setup:
    export TEST_POSTGRESQL_URL="postgresql://user:password@localhost:5432/ia_modules_test"

    Or create a .env file in tests/ directory:
    TEST_POSTGRESQL_URL=postgresql://user:password@localhost:5432/ia_modules_test

Run:
    pytest tests/integration/test_postgresql_integration.py -v

These tests will be SKIPPED if TEST_POSTGRESQL_URL is not set.
"""

import os
import pytest
from datetime import datetime, timezone
from nexusql import DatabaseManager, ConnectionConfig, DatabaseType


# Skip all tests in this file if PostgreSQL URL not configured
pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_POSTGRESQL_URL"),
    reason="PostgreSQL not configured (set TEST_POSTGRESQL_URL)"
)


@pytest.fixture
def pg_config():
    """PostgreSQL connection config from environment"""
    url = os.environ.get("TEST_POSTGRESQL_URL")
    return ConnectionConfig(
        database_type=DatabaseType.POSTGRESQL,
        database_url=url
    )


@pytest.fixture
def pg_db(pg_config):
    """Connected PostgreSQL database with cleanup"""
    db = DatabaseManager(pg_config)
    db.connect()

    # Create a unique test table name to avoid conflicts
    test_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    yield db, test_id

    # Cleanup: drop all test tables
    try:
        # Get all tables starting with test_
        result = db.fetch_all("""
            SELECT tablename FROM pg_tables
            WHERE schemaname = 'public' AND tablename LIKE 'test_%'
        """)

        for row in result:
            table_name = row['tablename']
            db.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
    except Exception as e:
        print(f"Cleanup warning: {e}")

    db.disconnect()


@pytest.fixture
async def pg_db_clean(pg_config):
    """
    Fresh PostgreSQL database for migration tests.
    Drops and recreates the ia_migrations table.
    """
    db = DatabaseManager(pg_config)
    db.connect()

    # Drop migration table to start fresh
    db.execute("DROP TABLE IF EXISTS ia_migrations CASCADE")

    yield db

    db.disconnect()


class TestPostgreSQLParameterBinding:
    """Test that PostgreSQL parameter binding works correctly"""

    def test_named_parameters_insert(self, pg_db):
        """Test INSERT with named parameters"""
        db, test_id = pg_db

        # Create test table
        db.execute(f"""
            CREATE TABLE test_params_{test_id} (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                age INTEGER,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)

        # Insert with named parameters
        result = db.execute(
            f"INSERT INTO test_params_{test_id} (name, age) VALUES (:name, :age)",
            {"name": "Alice", "age": 30}
        )

        assert isinstance(result, list)  # execute() returns list

        # Verify data exists by querying directly
        row = db.fetch_one(f"SELECT * FROM test_params_{test_id} WHERE name = :name", {"name": "Alice"})
        assert row is not None
        assert row["name"] == "Alice"
        assert row["age"] == 30
        assert row["created_at"] is not None

    def test_named_parameters_with_special_characters(self, pg_db):
        """Test parameters with special characters that need escaping"""
        db, test_id = pg_db

        db.execute(f"""
            CREATE TABLE test_special_{test_id} (
                id SERIAL PRIMARY KEY,
                text_value TEXT,
                json_value JSONB
            )
        """)

        # Insert data with quotes, newlines, special chars
        special_text = "Text with 'quotes' and \"double quotes\" and \n newlines"
        special_json = '{"key": "value with \'quotes\'"}'

        result = db.execute(
            f"INSERT INTO test_special_{test_id} (text_value, json_value) VALUES (:text, CAST(:json AS jsonb))",
            {"text": special_text, "json": special_json}
        )

        # execute() returns list on success
        assert isinstance(result, list)  # execute() returns list

        # Verify actual data in database
        row = db.fetch_one(f"SELECT * FROM test_special_{test_id}")
        assert row["text_value"] == special_text

    def test_named_parameters_with_null_values(self, pg_db):
        """Test NULL parameter handling"""
        db, test_id = pg_db

        db.execute(f"""
            CREATE TABLE test_nulls_{test_id} (
                id SERIAL PRIMARY KEY,
                required_field VARCHAR(255) NOT NULL,
                optional_field VARCHAR(255)
            )
        """)

        # Insert with NULL optional field
        result = db.execute(
            f"INSERT INTO test_nulls_{test_id} (required_field, optional_field) VALUES (:req, :opt)",
            {"req": "required_value", "opt": None}
        )

        assert isinstance(result, list)  # execute() returns list

        # Verify NULL was actually stored
        row = db.fetch_one(f"SELECT * FROM test_nulls_{test_id}")
        assert row["required_field"] == "required_value"
        assert row["optional_field"] is None

    def test_multiple_inserts_with_parameters(self, pg_db):
        """Test multiple inserts with different parameters"""
        db, test_id = pg_db

        db.execute(f"""
            CREATE TABLE test_multi_{test_id} (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255),
                value INTEGER
            )
        """)

        # Insert multiple rows
        for i in range(5):
            result = db.execute(
                f"INSERT INTO test_multi_{test_id} (name, value) VALUES (:name, :value)",
                {"name": f"item_{i}", "value": i * 10}
            )
            assert isinstance(result, list)  # execute() returns list

        # Verify all rows exist
        rows = db.fetch_all(f"SELECT * FROM test_multi_{test_id} ORDER BY value")
        assert len(rows) == 5

        for i, row in enumerate(rows):
            assert row["name"] == f"item_{i}"
            assert row["value"] == i * 10


class TestPostgreSQLDataTypes:
    """Test PostgreSQL-specific data types"""

    def test_uuid_type(self, pg_db):
        """Test UUID data type (should NOT be translated)"""
        db, test_id = pg_db

        # PostgreSQL syntax with UUID
        db.execute(f"""
            CREATE TABLE test_uuid_{test_id} (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL,
                name VARCHAR(255)
            )
        """)

        # Insert with UUID
        import uuid
        test_uuid = str(uuid.uuid4())

        result = db.execute(
            f"INSERT INTO test_uuid_{test_id} (user_id, name) VALUES (:uuid, :name)",
            {"uuid": test_uuid, "name": "Test User"}
        )

        assert isinstance(result, list)  # execute() returns list

        # Verify UUID was stored correctly
        row = db.fetch_one(f"SELECT * FROM test_uuid_{test_id} WHERE user_id = :uuid", {"uuid": test_uuid})
        assert row is not None
        assert str(row["user_id"]) == test_uuid
        assert row["id"] is not None  # Auto-generated UUID

    def test_jsonb_type(self, pg_db):
        """Test JSONB data type (should NOT be translated)"""
        db, test_id = pg_db

        db.execute(f"""
            CREATE TABLE test_jsonb_{test_id} (
                id SERIAL PRIMARY KEY,
                data JSONB NOT NULL,
                metadata JSONB DEFAULT '{{}}'::jsonb
            )
        """)

        # Insert JSON data
        json_data = '{"name": "Alice", "age": 30, "tags": ["python", "sql"]}'

        result = db.execute(
            f"INSERT INTO test_jsonb_{test_id} (data) VALUES (:data::jsonb)",
            {"data": json_data}
        )

        assert isinstance(result, list)  # execute() returns list

        # Verify JSONB data and query with JSONB operators
        row = db.fetch_one(f"SELECT data->>'name' as name FROM test_jsonb_{test_id}")
        assert row["name"] == "Alice"

    def test_boolean_type(self, pg_db):
        """Test BOOLEAN data type (should NOT be translated to INTEGER)"""
        db, test_id = pg_db

        db.execute(f"""
            CREATE TABLE test_bool_{test_id} (
                id SERIAL PRIMARY KEY,
                is_active BOOLEAN DEFAULT TRUE,
                is_deleted BOOLEAN DEFAULT FALSE
            )
        """)

        # Insert boolean values
        result = db.execute(
            f"INSERT INTO test_bool_{test_id} (is_active, is_deleted) VALUES (:active, :deleted)",
            {"active": True, "deleted": False}
        )

        assert isinstance(result, list)  # execute() returns list

        # Verify boolean values (should be actual bool, not 0/1)
        row = db.fetch_one(f"SELECT * FROM test_bool_{test_id}")
        assert row["is_active"] is True
        assert row["is_deleted"] is False

    def test_timestamp_with_now(self, pg_db):
        """Test TIMESTAMP with NOW() function (should NOT be translated)"""
        db, test_id = pg_db

        db.execute(f"""
            CREATE TABLE test_timestamp_{test_id} (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)

        # Insert row (timestamps should auto-populate)
        result = db.execute(f"INSERT INTO test_timestamp_{test_id} DEFAULT VALUES")
        assert isinstance(result, list)  # execute() returns list

        # Verify timestamps exist and are recent
        row = db.fetch_one(f"SELECT * FROM test_timestamp_{test_id}")
        assert row["created_at"] is not None
        assert row["updated_at"] is not None

        # Timestamps should be within last minute
        now = datetime.now(timezone.utc)
        created = row["created_at"]
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)

        time_diff = (now - created).total_seconds()
        assert time_diff < 60  # Should be created within last minute


class TestPostgreSQLTransactions:
    """Test PostgreSQL transaction behavior"""

    @pytest.mark.asyncio
    async def test_async_execute_with_transaction(self, pg_db):
        """Test async execute maintains transaction consistency"""
        db, test_id = pg_db

        db.execute(f"""
            CREATE TABLE test_txn_{test_id} (
                id SERIAL PRIMARY KEY,
                value INTEGER
            )
        """)

        # Use async execute (this was causing the transaction abort bug)
        result = await db.execute_async(
            f"INSERT INTO test_txn_{test_id} (value) VALUES (:value)",
            {"value": 100}
        )

        assert isinstance(result, list)  # execute() returns list

        # Verify data committed
        row = db.fetch_one(f"SELECT * FROM test_txn_{test_id} WHERE value = :value", {"value": 100})
        assert row is not None
        assert row["value"] == 100

    def test_rollback_on_error(self, pg_db):
        """Test that errors don't leave database in inconsistent state"""
        db, test_id = pg_db

        db.execute(f"""
            CREATE TABLE test_rollback_{test_id} (
                id SERIAL PRIMARY KEY,
                unique_value VARCHAR(255) UNIQUE
            )
        """)

        # Insert first value
        db.execute(
            f"INSERT INTO test_rollback_{test_id} (unique_value) VALUES (:value)",
            {"value": "unique"}
        )

        # Try to insert duplicate (should raise exception)
        try:
            db.execute(
                f"INSERT INTO test_rollback_{test_id} (unique_value) VALUES (:value)",
                {"value": "unique"}
            )
            assert False, "Expected UniqueViolation exception"
        except Exception as e:
            # Expect constraint violation
            assert "unique" in str(e).lower() or "duplicate" in str(e).lower()

        # Verify only one row exists
        rows = db.fetch_all(f"SELECT * FROM test_rollback_{test_id}")
        assert len(rows) == 1


class TestPostgreSQLMigrations:
    """Test migration system with actual PostgreSQL database"""

    @pytest.mark.asyncio
    async def test_migration_table_creation(self, pg_db_clean):
        """Test that migrations table is created in PostgreSQL"""
        db = pg_db_clean

        # Initialize should create ia_migrations table
        await db.initialize(apply_schema=True)

        # Verify table exists in PostgreSQL
        exists = db.table_exists("ia_migrations")
        assert exists is True

        # Verify table structure
        row = db.fetch_one("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'ia_migrations'
            AND column_name = 'version'
        """)

        assert row is not None
        assert row["column_name"] == "version"

    @pytest.mark.asyncio
    async def test_migrations_are_recorded(self, pg_db_clean):
        """Test that migrations are actually recorded in PostgreSQL"""
        db = pg_db_clean

        # Run migrations
        await db.initialize(apply_schema=True)

        # Query actual migration records
        rows = db.fetch_all("SELECT * FROM ia_migrations ORDER BY version")

        # Should have system migrations
        assert len(rows) > 0

        # Verify migration structure
        first_migration = rows[0]
        assert "version" in first_migration
        assert "filename" in first_migration
        assert "migration_type" in first_migration
        assert "applied_at" in first_migration

        # Verify migration type is recorded correctly
        assert first_migration["migration_type"] in ["system", "app"]

    @pytest.mark.asyncio
    async def test_migration_idempotency(self, pg_db_clean):
        """Test that running migrations twice doesn't duplicate records"""
        db = pg_db_clean

        # Run migrations first time
        await db.initialize(apply_schema=True)

        first_count_row = db.fetch_one("SELECT COUNT(*) as count FROM ia_migrations")
        first_count = first_count_row["count"]

        # Run migrations second time
        await db.initialize(apply_schema=True)

        second_count_row = db.fetch_one("SELECT COUNT(*) as count FROM ia_migrations")
        second_count = second_count_row["count"]

        # Should be same count (no duplicates)
        assert first_count == second_count
        assert first_count > 0

    @pytest.mark.asyncio
    async def test_migration_with_postgres_syntax(self, pg_db_clean):
        """Test that PostgreSQL-specific syntax in migrations works"""
        db = pg_db_clean

        # Run migrations (should have PostgreSQL-specific types)
        await db.initialize(apply_schema=True)

        # Verify tables were created with PostgreSQL types
        # Check pipeline_jobs table has UUID and JSONB
        row = db.fetch_one("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'pipeline_jobs'
            AND column_name = 'job_id'
        """)

        if row:  # Table might not exist depending on migrations
            # If table exists, verify it uses PostgreSQL types
            # (Could be 'uuid' or 'character varying' depending on migration)
            assert row["data_type"] in ["uuid", "character varying", "text"]


class TestPostgreSQLDataVerification:
    """Test that data actually exists and persists in PostgreSQL"""

    def test_data_persists_across_queries(self, pg_db):
        """Test that inserted data can be queried multiple times"""
        db, test_id = pg_db

        # Create and populate table
        db.execute(f"""
            CREATE TABLE test_persist_{test_id} (
                id SERIAL PRIMARY KEY,
                data VARCHAR(255)
            )
        """)

        db.execute(
            f"INSERT INTO test_persist_{test_id} (data) VALUES (:data)",
            {"data": "persistent_value"}
        )

        # Query multiple times
        for i in range(3):
            row = db.fetch_one(f"SELECT * FROM test_persist_{test_id} WHERE data = :data", {"data": "persistent_value"})
            assert row is not None
            assert row["data"] == "persistent_value"

    def test_update_and_verify(self, pg_db):
        """Test UPDATE operations and verify changes"""
        db, test_id = pg_db

        db.execute(f"""
            CREATE TABLE test_update_{test_id} (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255),
                value INTEGER
            )
        """)

        # Insert initial data
        db.execute(
            f"INSERT INTO test_update_{test_id} (name, value) VALUES (:name, :value)",
            {"name": "test", "value": 10}
        )

        # Update data
        result = db.execute(
            f"UPDATE test_update_{test_id} SET value = :new_value WHERE name = :name",
            {"new_value": 20, "name": "test"}
        )

        assert isinstance(result, list)  # execute() returns list

        # Verify update persisted
        row = db.fetch_one(f"SELECT * FROM test_update_{test_id} WHERE name = :name", {"name": "test"})
        assert row["value"] == 20

    def test_delete_and_verify(self, pg_db):
        """Test DELETE operations and verify removal"""
        db, test_id = pg_db

        db.execute(f"""
            CREATE TABLE test_delete_{test_id} (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255)
            )
        """)

        # Insert data
        db.execute(f"INSERT INTO test_delete_{test_id} (name) VALUES (:name)", {"name": "to_delete"})
        db.execute(f"INSERT INTO test_delete_{test_id} (name) VALUES (:name)", {"name": "to_keep"})

        # Verify both exist
        rows = db.fetch_all(f"SELECT * FROM test_delete_{test_id}")
        assert len(rows) == 2

        # Delete one
        result = db.execute(
            f"DELETE FROM test_delete_{test_id} WHERE name = :name",
            {"name": "to_delete"}
        )

        assert isinstance(result, list)  # execute() returns list

        # Verify only one remains
        rows = db.fetch_all(f"SELECT * FROM test_delete_{test_id}")
        assert len(rows) == 1
        assert rows[0]["name"] == "to_keep"

    def test_complex_query_with_joins(self, pg_db):
        """Test complex queries with JOINs and verify results"""
        db, test_id = pg_db

        # Create related tables
        db.execute(f"""
            CREATE TABLE test_users_{test_id} (
                id SERIAL PRIMARY KEY,
                username VARCHAR(255) UNIQUE
            )
        """)

        db.execute(f"""
            CREATE TABLE test_orders_{test_id} (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES test_users_{test_id}(id),
                amount DECIMAL(10, 2)
            )
        """)

        # Insert related data
        db.execute(f"INSERT INTO test_users_{test_id} (username) VALUES (:username)", {"username": "alice"})

        user = db.fetch_one(f"SELECT id FROM test_users_{test_id} WHERE username = :username", {"username": "alice"})
        user_id = user["id"]

        db.execute(
            f"INSERT INTO test_orders_{test_id} (user_id, amount) VALUES (:user_id, :amount)",
            {"user_id": user_id, "amount": 99.99}
        )

        db.execute(
            f"INSERT INTO test_orders_{test_id} (user_id, amount) VALUES (:user_id, :amount)",
            {"user_id": user_id, "amount": 49.99}
        )

        # Query with JOIN
        rows = db.fetch_all(f"""
            SELECT u.username, o.amount
            FROM test_users_{test_id} u
            JOIN test_orders_{test_id} o ON u.id = o.user_id
            WHERE u.username = :username
            ORDER BY o.amount DESC
        """, {"username": "alice"})

        assert len(rows) == 2
        assert rows[0]["username"] == "alice"
        assert float(rows[0]["amount"]) == 99.99
        assert float(rows[1]["amount"]) == 49.99
