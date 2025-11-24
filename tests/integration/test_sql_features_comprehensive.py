"""
Comprehensive SQL feature tests for all databases.

Tests all real-world SQL features used in the codebase to ensure
proper translation across SQLite, PostgreSQL, MySQL, and MSSQL.
"""

import pytest
from nexusql import DatabaseManager
from nexusql import DatabaseType


class TestForeignKeyConstraints:
    """Test FOREIGN KEY constraint translation and behavior"""

    @pytest.mark.asyncio
    async def test_foreign_key_basic(self, db_config):
        """Test basic FOREIGN KEY constraint"""
        db = DatabaseManager(db_config)
        db.connect()

        try:
            # Drop tables if they exist
            db.execute("DROP TABLE IF EXISTS orders")
            db.execute("DROP TABLE IF EXISTS customers")

            # Create parent table
            parent_sql = """
            CREATE TABLE IF NOT EXISTS customers (
                customer_id INTEGER PRIMARY KEY,
                name VARCHAR(100) NOT NULL
            )
            """
            db.execute(parent_sql)

            # Create child table with FK
            child_sql = """
            CREATE TABLE IF NOT EXISTS orders (
                order_id INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL,
                amount REAL,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
            )
            """
            db.execute(child_sql)

            # Insert test data
            db.execute("INSERT INTO customers (customer_id, name) VALUES (:id, :name)",
                      {"id": 1, "name": "John Doe"})
            db.execute("INSERT INTO orders (order_id, customer_id, amount) VALUES (:id, :cust, :amt)",
                      {"id": 1, "cust": 1, "amt": 100.50})

            # Verify data
            result = db.fetch_one("SELECT COUNT(*) as count FROM orders")
            assert result["count"] == 1

        finally:
            db.execute("DROP TABLE IF EXISTS orders")
            db.execute("DROP TABLE IF EXISTS customers")
            db.disconnect()


    @pytest.mark.asyncio
    async def test_foreign_key_on_delete_no_action(self, db_config):
        """Test FOREIGN KEY with ON DELETE NO ACTION"""
        db = DatabaseManager(db_config)
        db.connect()

        try:
            # Drop tables if they exist
            db.execute("DROP TABLE IF EXISTS comments")
            db.execute("DROP TABLE IF EXISTS posts")

            # Create tables
            db.execute("""
            CREATE TABLE IF NOT EXISTS posts (
                post_id INTEGER PRIMARY KEY,
                title VARCHAR(200)
            )
            """)

            db.execute("""
            CREATE TABLE IF NOT EXISTS comments (
                comment_id INTEGER PRIMARY KEY,
                post_id INTEGER NOT NULL,
                content VARCHAR(500),
                FOREIGN KEY (post_id) REFERENCES posts(post_id) ON DELETE NO ACTION
            )
            """)

            # Insert data
            db.execute("INSERT INTO posts (post_id, title) VALUES (1, 'Test Post')")
            db.execute("INSERT INTO comments (comment_id, post_id, content) VALUES (1, 1, 'Nice post')")

            # Verify data exists
            result = db.fetch_one("SELECT COUNT(*) as count FROM comments")
            assert result["count"] == 1

        finally:
            db.execute("DROP TABLE IF EXISTS comments")
            db.execute("DROP TABLE IF EXISTS posts")
            db.disconnect()


class TestCheckConstraints:
    """Test CHECK constraint translation"""

    @pytest.mark.asyncio
    async def test_check_constraint_enum(self, db_config):
        """Test CHECK constraint with IN clause (enum-like)"""
        db = DatabaseManager(db_config)
        db.connect()

        try:
            db.execute("DROP TABLE IF EXISTS test_check_enum")

            # PostgreSQL canonical syntax with CHECK constraint
            sql = """
            CREATE TABLE IF NOT EXISTS test_check_enum (
                id INTEGER PRIMARY KEY,
                status VARCHAR(20) NOT NULL CHECK (status IN ('pending', 'active', 'completed', 'failed'))
            )
            """
            db.execute(sql)

            # Test valid values
            db.execute("INSERT INTO test_check_enum (id, status) VALUES (1, 'pending')")
            db.execute("INSERT INTO test_check_enum (id, status) VALUES (2, 'active')")

            result = db.fetch_one("SELECT COUNT(*) as count FROM test_check_enum")
            assert result["count"] == 2

            # Test invalid value (should fail on MySQL 8.0+, MSSQL, PostgreSQL)
            # SQLite doesn't enforce CHECK constraints by default
            if db_config.database_type != DatabaseType.SQLITE:
                with pytest.raises(Exception):
                    db.execute("INSERT INTO test_check_enum (id, status) VALUES (3, 'invalid')")

        finally:
            db.execute("DROP TABLE IF EXISTS test_check_enum")
            db.disconnect()


    @pytest.mark.asyncio
    async def test_check_constraint_range(self, db_config):
        """Test CHECK constraint with range condition"""
        db = DatabaseManager(db_config)
        db.connect()

        try:
            db.execute("DROP TABLE IF EXISTS test_check_range")

            sql = """
            CREATE TABLE IF NOT EXISTS test_check_range (
                id INTEGER PRIMARY KEY,
                age INTEGER CHECK (age >= 0 AND age <= 150)
            )
            """
            db.execute(sql)

            # Test valid value
            db.execute("INSERT INTO test_check_range (id, age) VALUES (1, 25)")

            result = db.fetch_one("SELECT age FROM test_check_range WHERE id = 1")
            assert result["age"] == 25

            # Test invalid value
            if db_config.database_type != DatabaseType.SQLITE:
                with pytest.raises(Exception):
                    db.execute("INSERT INTO test_check_range (id, age) VALUES (2, 200)")

        finally:
            db.execute("DROP TABLE IF EXISTS test_check_range")
            db.disconnect()


class TestUniqueConstraints:
    """Test UNIQUE constraint translation"""

    @pytest.mark.asyncio
    async def test_unique_column(self, db_config):
        """Test UNIQUE constraint on single column"""
        db = DatabaseManager(db_config)
        db.connect()

        try:
            db.execute("DROP TABLE IF EXISTS test_unique")

            sql = """
            CREATE TABLE IF NOT EXISTS test_unique (
                id INTEGER PRIMARY KEY,
                email VARCHAR(255) UNIQUE
            )
            """
            db.execute(sql)

            # Insert first record
            db.execute("INSERT INTO test_unique (id, email) VALUES (1, 'test@example.com')")

            # Try to insert duplicate (should fail)
            with pytest.raises(Exception):
                db.execute("INSERT INTO test_unique (id, email) VALUES (2, 'test@example.com')")

        finally:
            db.execute("DROP TABLE IF EXISTS test_unique")
            db.disconnect()


    @pytest.mark.asyncio
    async def test_unique_constraint_named(self, db_config):
        """Test named UNIQUE constraint"""
        db = DatabaseManager(db_config)
        db.connect()

        try:
            db.execute("DROP TABLE IF EXISTS test_unique_named")

            sql = """
            CREATE TABLE IF NOT EXISTS test_unique_named (
                id INTEGER PRIMARY KEY,
                username VARCHAR(50) NOT NULL,
                email VARCHAR(255) NOT NULL,
                UNIQUE (username, email)
            )
            """
            db.execute(sql)

            # Insert first record
            db.execute("INSERT INTO test_unique_named (id, username, email) VALUES (1, 'john', 'john@example.com')")

            # Same username but different email should work
            db.execute("INSERT INTO test_unique_named (id, username, email) VALUES (2, 'john', 'john2@example.com')")

            # Try duplicate combo (should fail)
            with pytest.raises(Exception):
                db.execute("INSERT INTO test_unique_named (id, username, email) VALUES (3, 'john', 'john@example.com')")

        finally:
            db.execute("DROP TABLE IF EXISTS test_unique_named")
            db.disconnect()


class TestIndexes:
    """Test INDEX creation and translation"""

    @pytest.mark.asyncio
    async def test_simple_index(self, db_config):
        """Test simple index creation"""
        db = DatabaseManager(db_config)
        db.connect()

        try:
            db.execute("DROP TABLE IF EXISTS test_indexed")

            # Create table
            sql = """
            CREATE TABLE IF NOT EXISTS test_indexed (
                id INTEGER PRIMARY KEY,
                name VARCHAR(100),
                created_at TIMESTAMP DEFAULT NOW()
            )
            """
            db.execute(sql)

            # Create index (IF NOT EXISTS should be translated away for MySQL/MSSQL)
            index_sql = """
            CREATE INDEX IF NOT EXISTS idx_test_name ON test_indexed(name)
            """
            db.execute(index_sql)

            # Insert test data
            db.execute("INSERT INTO test_indexed (id, name) VALUES (1, 'Alice')")
            db.execute("INSERT INTO test_indexed (id, name) VALUES (2, 'Bob')")

            # Query using index
            result = db.fetch_one("SELECT * FROM test_indexed WHERE name = 'Alice'")
            assert result["name"] == "Alice"

        finally:
            db.execute("DROP TABLE IF EXISTS test_indexed")
            db.disconnect()


    @pytest.mark.asyncio
    async def test_composite_index(self, db_config):
        """Test composite (multi-column) index"""
        db = DatabaseManager(db_config)
        db.connect()

        try:
            db.execute("DROP TABLE IF EXISTS test_composite_idx")

            sql = """
            CREATE TABLE IF NOT EXISTS test_composite_idx (
                id INTEGER PRIMARY KEY,
                category VARCHAR(50),
                status VARCHAR(20),
                created_at TIMESTAMP DEFAULT NOW()
            )
            """
            db.execute(sql)

            # Create composite index
            index_sql = """
            CREATE INDEX IF NOT EXISTS idx_category_status ON test_composite_idx(category, status)
            """
            db.execute(index_sql)

            # Insert test data
            db.execute("INSERT INTO test_composite_idx (id, category, status) VALUES (1, 'tech', 'active')")
            db.execute("INSERT INTO test_composite_idx (id, category, status) VALUES (2, 'tech', 'pending')")

            result = db.fetch_all("SELECT * FROM test_composite_idx WHERE category = 'tech' AND status = 'active'")
            assert len(result) == 1

        finally:
            db.execute("DROP TABLE IF EXISTS test_composite_idx")
            db.disconnect()


class TestJoinQueries:
    """Test JOIN queries across databases"""

    @pytest.mark.asyncio
    async def test_inner_join(self, db_config):
        """Test INNER JOIN query"""
        db = DatabaseManager(db_config)
        db.connect()

        try:
            # Drop tables if they exist
            db.execute("DROP TABLE IF EXISTS order_items")
            db.execute("DROP TABLE IF EXISTS products")

            # Create tables
            db.execute("""
            CREATE TABLE IF NOT EXISTS products (
                product_id INTEGER PRIMARY KEY,
                name VARCHAR(100),
                price REAL
            )
            """)

            db.execute("""
            CREATE TABLE IF NOT EXISTS order_items (
                item_id INTEGER PRIMARY KEY,
                product_id INTEGER,
                quantity INTEGER,
                FOREIGN KEY (product_id) REFERENCES products(product_id)
            )
            """)

            # Insert test data
            db.execute("INSERT INTO products (product_id, name, price) VALUES (1, 'Widget', 9.99)")
            db.execute("INSERT INTO products (product_id, name, price) VALUES (2, 'Gadget', 19.99)")
            db.execute("INSERT INTO order_items (item_id, product_id, quantity) VALUES (1, 1, 5)")
            db.execute("INSERT INTO order_items (item_id, product_id, quantity) VALUES (2, 2, 3)")

            # Test INNER JOIN
            query = """
            SELECT p.name, p.price, o.quantity
            FROM order_items o
            INNER JOIN products p ON o.product_id = p.product_id
            WHERE o.quantity > 2
            """
            results = db.fetch_all(query)
            assert len(results) == 2

        finally:
            db.execute("DROP TABLE IF EXISTS order_items")
            db.execute("DROP TABLE IF EXISTS products")
            db.disconnect()


    @pytest.mark.asyncio
    async def test_left_join(self, db_config):
        """Test LEFT JOIN query"""
        db = DatabaseManager(db_config)
        db.connect()

        try:
            db.execute("DROP TABLE IF EXISTS user_profiles")
            db.execute("DROP TABLE IF EXISTS users")

            # Create tables
            db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                username VARCHAR(50)
            )
            """)

            db.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                profile_id INTEGER PRIMARY KEY,
                user_id INTEGER,
                bio VARCHAR(500),
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
            """)

            # Insert data - one user has profile, one doesn't
            db.execute("INSERT INTO users (user_id, username) VALUES (1, 'alice')")
            db.execute("INSERT INTO users (user_id, username) VALUES (2, 'bob')")
            db.execute("INSERT INTO user_profiles (profile_id, user_id, bio) VALUES (1, 1, 'Alice bio')")

            # Test LEFT JOIN (should return both users)
            query = """
            SELECT u.username, p.bio
            FROM users u
            LEFT JOIN user_profiles p ON u.user_id = p.user_id
            """
            results = db.fetch_all(query)
            assert len(results) == 2

            # Verify one has NULL bio
            bios = [r.get("bio") for r in results]
            assert None in bios

        finally:
            db.execute("DROP TABLE IF EXISTS user_profiles")
            db.execute("DROP TABLE IF EXISTS users")
            db.disconnect()


class TestAggregationQueries:
    """Test aggregation functions"""

    @pytest.mark.asyncio
    async def test_count_group_by(self, db_config):
        """Test COUNT with GROUP BY"""
        db = DatabaseManager(db_config)
        db.connect()

        try:
            db.execute("DROP TABLE IF EXISTS test_aggregation")

            sql = """
            CREATE TABLE IF NOT EXISTS test_aggregation (
                id INTEGER PRIMARY KEY,
                category VARCHAR(50),
                value INTEGER
            )
            """
            db.execute(sql)

            # Insert test data
            db.execute("INSERT INTO test_aggregation (id, category, value) VALUES (1, 'A', 10)")
            db.execute("INSERT INTO test_aggregation (id, category, value) VALUES (2, 'A', 20)")
            db.execute("INSERT INTO test_aggregation (id, category, value) VALUES (3, 'B', 30)")

            # Test GROUP BY with COUNT
            query = """
            SELECT category, COUNT(*) as count, SUM(value) as total
            FROM test_aggregation
            GROUP BY category
            ORDER BY category
            """
            results = db.fetch_all(query)
            assert len(results) == 2
            assert results[0]["count"] == 2
            assert results[0]["total"] == 30

        finally:
            db.execute("DROP TABLE IF EXISTS test_aggregation")
            db.disconnect()


    @pytest.mark.asyncio
    async def test_having_clause(self, db_config):
        """Test HAVING clause with aggregation"""
        db = DatabaseManager(db_config)
        db.connect()

        try:
            db.execute("DROP TABLE IF EXISTS test_having")

            sql = """
            CREATE TABLE IF NOT EXISTS test_having (
                id INTEGER PRIMARY KEY,
                group_name VARCHAR(50),
                amount REAL
            )
            """
            db.execute(sql)

            # Insert test data
            db.execute("INSERT INTO test_having (id, group_name, amount) VALUES (1, 'G1', 100)")
            db.execute("INSERT INTO test_having (id, group_name, amount) VALUES (2, 'G1', 200)")
            db.execute("INSERT INTO test_having (id, group_name, amount) VALUES (3, 'G2', 50)")

            # Test HAVING
            query = """
            SELECT group_name, SUM(amount) as total
            FROM test_having
            GROUP BY group_name
            HAVING SUM(amount) > 100
            """
            results = db.fetch_all(query)
            assert len(results) == 1
            assert results[0]["group_name"] == "G1"

        finally:
            db.execute("DROP TABLE IF EXISTS test_having")
            db.disconnect()


class TestSubqueries:
    """Test subquery support"""

    @pytest.mark.asyncio
    async def test_subquery_in_where(self, db_config):
        """Test subquery in WHERE clause"""
        db = DatabaseManager(db_config)
        db.connect()

        try:
            db.execute("DROP TABLE IF EXISTS test_subquery")

            sql = """
            CREATE TABLE IF NOT EXISTS test_subquery (
                id INTEGER PRIMARY KEY,
                value INTEGER
            )
            """
            db.execute(sql)

            # Insert test data
            for i in range(1, 11):
                db.execute(f"INSERT INTO test_subquery (id, value) VALUES ({i}, {i * 10})")

            # Test subquery
            query = """
            SELECT * FROM test_subquery
            WHERE value > (SELECT AVG(value) FROM test_subquery)
            ORDER BY id
            """
            results = db.fetch_all(query)
            assert len(results) == 5  # Values > 55 (average of 10-100)

        finally:
            db.execute("DROP TABLE IF EXISTS test_subquery")
            db.disconnect()


class TestLimitOffset:
    """Test LIMIT and OFFSET for pagination"""

    @pytest.mark.asyncio
    async def test_limit_offset(self, db_config):
        """Test LIMIT and OFFSET clauses"""
        db = DatabaseManager(db_config)
        db.connect()

        try:
            db.execute("DROP TABLE IF EXISTS test_pagination")

            sql = """
            CREATE TABLE IF NOT EXISTS test_pagination (
                id INTEGER PRIMARY KEY,
                name VARCHAR(50)
            )
            """
            db.execute(sql)

            # Insert test data
            for i in range(1, 21):
                db.execute(f"INSERT INTO test_pagination (id, name) VALUES ({i}, 'Item {i}')")

            # Test LIMIT
            query = "SELECT * FROM test_pagination ORDER BY id LIMIT 5"
            results = db.fetch_all(query)
            assert len(results) == 5
            assert results[0]["id"] == 1

            # Test LIMIT with OFFSET
            query = "SELECT * FROM test_pagination ORDER BY id LIMIT 5 OFFSET 10"
            results = db.fetch_all(query)
            assert len(results) == 5
            assert results[0]["id"] == 11

        finally:
            db.execute("DROP TABLE IF EXISTS test_pagination")
            db.disconnect()
