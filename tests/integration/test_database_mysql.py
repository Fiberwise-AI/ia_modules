"""
Integration tests for MySQL database backend

These tests require a running MySQL instance.
Run with: pytest -m mysql
"""

import os
import pytest
from ia_modules.pipeline.test_utils import create_test_execution_context
from datetime import datetime
from nexusql import DatabaseManager, ConnectionConfig, DatabaseType


@pytest.fixture
def mysql_config():
    """Get MySQL connection configuration from environment"""
    url = os.environ.get(
        "TEST_MYSQL_URL",
        "mysql://testuser:testpass@localhost:3306/ia_modules_test"
    )
    return ConnectionConfig(
        database_type=DatabaseType.MYSQL,
        database_url=url
    )


@pytest.fixture
def mysql_db(mysql_config):
    """Create a MySQL database manager for testing"""
    db = DatabaseManager(mysql_config)
    db.connect()

    # Clean up test tables
    test_tables = [
        'test_users', 'test_products', 'test_data', 'test_table',
        'test_items', 'test_json', 'test_timestamps', 'test_transactions'
    ]
    for table in test_tables:
        try:
            db.execute(f'DROP TABLE IF EXISTS {table}')
        except:
            pass

    yield db

    # Cleanup after tests
    for table in test_tables:
        try:
            db.execute(f'DROP TABLE IF EXISTS {table}')
        except:
            pass

    db.disconnect()


@pytest.mark.mysql
@pytest.mark.integration
class TestMySQLBasicOperations:
    """Test basic CRUD operations with MySQL"""

    def test_create_table(self, mysql_db):
        """Test table creation"""
        mysql_db.execute('''
            CREATE TABLE test_users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(100) NOT NULL,
                email VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Verify table exists
        result = mysql_db.execute(
            "SELECT COUNT(*) as count FROM information_schema.tables "
            "WHERE table_schema = DATABASE() AND table_name = 'test_users'"
        )
        assert result[0]['count'] == 1

    def test_insert_and_select(self, mysql_db):
        """Test inserting and selecting data"""
        mysql_db.execute('''
            CREATE TABLE test_users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(100) NOT NULL,
                email VARCHAR(255) NOT NULL
            )
        ''')

        # Insert data
        mysql_db.execute(
            "INSERT INTO test_users (username, email) VALUES (%s, %s)",
            ("john_doe", "john@example.com")
        )

        # Select data
        result = mysql_db.execute("SELECT * FROM test_users WHERE username = %s", ("john_doe",))
        assert len(result) == 1
        assert result[0]['username'] == "john_doe"
        assert result[0]['email'] == "john@example.com"

    def test_update(self, mysql_db):
        """Test updating data"""
        mysql_db.execute('''
            CREATE TABLE test_users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(100) NOT NULL,
                email VARCHAR(255) NOT NULL
            )
        ''')

        mysql_db.execute(
            "INSERT INTO test_users (username, email) VALUES (%s, %s)",
            ("john_doe", "john@example.com")
        )

        # Update data
        mysql_db.execute(
            "UPDATE test_users SET email = %s WHERE username = %s",
            ("newemail@example.com", "john_doe")
        )

        # Verify update
        result = mysql_db.execute("SELECT email FROM test_users WHERE username = %s", ("john_doe",))
        assert result[0]['email'] == "newemail@example.com"

    def test_delete(self, mysql_db):
        """Test deleting data"""
        mysql_db.execute('''
            CREATE TABLE test_users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(100) NOT NULL
            )
        ''')

        mysql_db.execute("INSERT INTO test_users (username) VALUES (%s)", ("john_doe",))
        mysql_db.execute("DELETE FROM test_users WHERE username = %s", ("john_doe",))

        result = mysql_db.execute("SELECT * FROM test_users WHERE username = %s", ("john_doe",))
        assert len(result) == 0


@pytest.mark.mysql
@pytest.mark.integration
class TestMySQLDataTypes:
    """Test MySQL-specific data types"""

    def test_json_type(self, mysql_db):
        """Test JSON data type support"""
        mysql_db.execute('''
            CREATE TABLE test_json (
                id INT AUTO_INCREMENT PRIMARY KEY,
                data JSON
            )
        ''')

        import json
        test_data = {"name": "John", "age": 30, "tags": ["python", "mysql"]}

        mysql_db.execute(
            "INSERT INTO test_json (data) VALUES (%s)",
            (json.dumps(test_data),)
        )

        result = mysql_db.execute("SELECT data FROM test_json")
        # MySQL may return JSON as string, depending on driver
        if isinstance(result[0]['data'], str):
            retrieved_data = json.loads(result[0]['data'])
        else:
            retrieved_data = result[0]['data']

        assert retrieved_data == test_data

    def test_timestamp_handling(self, mysql_db):
        """Test timestamp data type"""
        mysql_db.execute('''
            CREATE TABLE test_timestamps (
                id INT AUTO_INCREMENT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
        ''')

        mysql_db.execute("INSERT INTO test_timestamps () VALUES ()")

        result = mysql_db.execute("SELECT created_at, updated_at FROM test_timestamps")
        assert result[0]['created_at'] is not None
        assert result[0]['updated_at'] is not None

    def test_text_types(self, mysql_db):
        """Test various text types"""
        mysql_db.execute('''
            CREATE TABLE test_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                tiny_text TINYTEXT,
                medium_text MEDIUMTEXT,
                long_text LONGTEXT
            )
        ''')

        short_text = "Short text"
        medium_text = "Medium " * 1000
        long_text = "Long " * 10000

        mysql_db.execute(
            "INSERT INTO test_data (tiny_text, medium_text, long_text) VALUES (%s, %s, %s)",
            (short_text, medium_text, long_text)
        )

        result = mysql_db.execute("SELECT * FROM test_data")
        assert result[0]['tiny_text'] == short_text
        assert result[0]['medium_text'] == medium_text
        assert result[0]['long_text'] == long_text


@pytest.mark.mysql
@pytest.mark.integration
class TestMySQLTransactions:
    """Test transaction support in MySQL"""

    def test_commit_transaction(self, mysql_db):
        """Test committing a transaction"""
        mysql_db.execute('''
            CREATE TABLE test_transactions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                value VARCHAR(100)
            )
        ''')

        # Start transaction (implicitly with first statement)
        mysql_db.execute("START TRANSACTION")
        mysql_db.execute("INSERT INTO test_transactions (value) VALUES (%s)", ("test1",))
        mysql_db.execute("INSERT INTO test_transactions (value) VALUES (%s)", ("test2",))
        mysql_db.execute("COMMIT")

        result = mysql_db.execute("SELECT COUNT(*) as count FROM test_transactions")
        assert result[0]['count'] == 2

    def test_rollback_transaction(self, mysql_db):
        """Test rolling back a transaction"""
        mysql_db.execute('''
            CREATE TABLE test_transactions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                value VARCHAR(100)
            )
        ''')

        mysql_db.execute("START TRANSACTION")
        mysql_db.execute("INSERT INTO test_transactions (value) VALUES (%s)", ("test1",))
        mysql_db.execute("ROLLBACK")

        result = mysql_db.execute("SELECT COUNT(*) as count FROM test_transactions")
        assert result[0]['count'] == 0


@pytest.mark.mysql
@pytest.mark.integration
class TestMySQLPerformance:
    """Test MySQL performance characteristics"""

    def test_bulk_insert(self, mysql_db):
        """Test bulk insert performance"""
        mysql_db.execute('''
            CREATE TABLE test_items (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100),
                value INT
            )
        ''')

        # Insert 1000 records
        for i in range(1000):
            mysql_db.execute(
                "INSERT INTO test_items (name, value) VALUES (%s, %s)",
                (f"item_{i}", i)
            )

        result = mysql_db.execute("SELECT COUNT(*) as count FROM test_items")
        assert result[0]['count'] == 1000

    def test_indexed_query(self, mysql_db):
        """Test query with index"""
        mysql_db.execute('''
            CREATE TABLE test_items (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100),
                value INT,
                INDEX idx_name (name)
            )
        ''')

        # Insert test data
        for i in range(100):
            mysql_db.execute(
                "INSERT INTO test_items (name, value) VALUES (%s, %s)",
                (f"item_{i}", i)
            )

        # Query using index
        result = mysql_db.execute(
            "SELECT * FROM test_items WHERE name = %s",
            ("item_50",)
        )

        assert len(result) == 1
        assert result[0]['value'] == 50


@pytest.mark.mysql
@pytest.mark.integration
class TestMySQLCharsetAndCollation:
    """Test MySQL charset and collation features"""

    def test_utf8mb4_support(self, mysql_db):
        """Test UTF-8 MB4 (full Unicode) support"""
        mysql_db.execute('''
            CREATE TABLE test_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                text VARCHAR(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
            )
        ''')

        # Insert emoji and special characters
        emoji_text = "Hello 👋 World 🌍 Test 🚀"
        mysql_db.execute("INSERT INTO test_data (text) VALUES (%s)", (emoji_text,))

        result = mysql_db.execute("SELECT text FROM test_data")
        assert result[0]['text'] == emoji_text

    def test_case_insensitive_search(self, mysql_db):
        """Test case-insensitive collation"""
        mysql_db.execute('''
            CREATE TABLE test_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) COLLATE utf8mb4_unicode_ci
            )
        ''')

        mysql_db.execute("INSERT INTO test_data (name) VALUES (%s)", ("TestName",))

        # Case-insensitive search should find the record
        result = mysql_db.execute("SELECT * FROM test_data WHERE name = %s", ("testname",))
        assert len(result) == 1
