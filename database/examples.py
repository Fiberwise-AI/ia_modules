"""
Examples demonstrating the database abstraction layer.

Run this file to see both NexusQL and SQLAlchemy in action.
"""

from ia_modules.database import get_database, set_default_backend, DatabaseBackend


def example_basic_usage():
    """Basic usage with default backend (NexusQL)"""
    print("=" * 60)
    print("Example 1: Basic Usage (Default Backend)")
    print("=" * 60)

    db = get_database("sqlite:///:memory:")
    db.connect()

    # Create table
    db.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            name TEXT
        )
    """)

    # Insert data
    db.execute(
        "INSERT INTO users (email, name) VALUES (:email, :name)",
        {"email": "alice@example.com", "name": "Alice"}
    )

    # Fetch data
    user = db.fetch_one(
        "SELECT * FROM users WHERE email = :email",
        {"email": "alice@example.com"}
    )
    print(f"User: {user}")

    db.disconnect()
    print("✓ Basic usage complete\n")


def example_explicit_backend():
    """Explicitly choosing backend"""
    print("=" * 60)
    print("Example 2: Explicit Backend Selection")
    print("=" * 60)

    # Test with NexusQL
    print("\nTesting with NexusQL:")
    db_nexus = get_database("sqlite:///:memory:", backend="nexusql")
    db_nexus.connect()
    print(f"  Backend: NexusQL")
    print(f"  Type: {type(db_nexus).__name__}")
    db_nexus.disconnect()

    # Test with SQLAlchemy
    print("\nTesting with SQLAlchemy:")
    try:
        db_alchemy = get_database("sqlite:///:memory:", backend="sqlalchemy")
        db_alchemy.connect()
        print(f"  Backend: SQLAlchemy")
        print(f"  Type: {type(db_alchemy).__name__}")
        db_alchemy.disconnect()
    except ImportError:
        print("  ⚠️  SQLAlchemy not installed (pip install sqlalchemy)")

    print("✓ Backend selection complete\n")


def example_context_manager():
    """Using context manager"""
    print("=" * 60)
    print("Example 3: Context Manager")
    print("=" * 60)

    with get_database("sqlite:///:memory:") as db:
        db.execute("""
            CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                name TEXT,
                price REAL
            )
        """)

        db.execute(
            "INSERT INTO products (name, price) VALUES (:name, :price)",
            {"name": "Widget", "price": 19.99}
        )

        products = db.fetch_all("SELECT * FROM products")
        print(f"Products: {products}")
        # Auto-disconnect on exit

    print("✓ Context manager complete\n")


def example_comparison():
    """Compare both backends side-by-side"""
    print("=" * 60)
    print("Example 4: Backend Comparison")
    print("=" * 60)

    # Same operations with both backends
    for backend in ["nexusql", "sqlalchemy"]:
        print(f"\nUsing {backend.upper()}:")

        try:
            db = get_database("sqlite:///:memory:", backend=backend)
            db.connect()

            # Create table
            db.execute("""
                CREATE TABLE test (
                    id INTEGER PRIMARY KEY,
                    value TEXT
                )
            """)

            # Insert
            db.execute(
                "INSERT INTO test (value) VALUES (:val)",
                {"val": f"test_{backend}"}
            )

            # Fetch
            result = db.fetch_one("SELECT * FROM test WHERE id = :id", {"id": 1})
            print(f"  Result: {result}")

            db.disconnect()

        except ImportError as e:
            print(f"  ⚠️  {e}")

    print("✓ Comparison complete\n")


def example_global_default():
    """Setting global default backend"""
    print("=" * 60)
    print("Example 5: Global Default Backend")
    print("=" * 60)

    # Set global default to SQLAlchemy
    try:
        set_default_backend(DatabaseBackend.SQLALCHEMY)
        print("Set default backend to: SQLAlchemy")

        # This will use SQLAlchemy
        db = get_database("sqlite:///:memory:")
        db.connect()
        print(f"Database type: {type(db).__name__}")
        db.disconnect()

    except ImportError:
        print("⚠️  SQLAlchemy not installed")

    # Reset to NexusQL
    set_default_backend(DatabaseBackend.NEXUSQL)
    print("Reset default backend to: NexusQL")

    print("✓ Global default complete\n")


def example_advanced_nexusql():
    """Advanced NexusQL features"""
    print("=" * 60)
    print("Example 6: Advanced NexusQL Features")
    print("=" * 60)

    db = get_database("sqlite:///:memory:", backend="nexusql")
    db.connect()

    # SQL Translation in action
    print("\nSQL Translation (PostgreSQL → SQLite):")
    db.execute("""
        CREATE TABLE products (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255),
            active BOOLEAN DEFAULT TRUE,
            price NUMERIC(10,2)
        )
    """)
    print("  ✓ Created table with PostgreSQL syntax on SQLite!")

    # Access underlying NexusQL instance
    print(f"\nDatabase type: {db.database_type}")
    print(f"Connection config: {db.config.database_type}")

    db.disconnect()
    print("✓ Advanced NexusQL complete\n")


def example_advanced_sqlalchemy():
    """Advanced SQLAlchemy features"""
    print("=" * 60)
    print("Example 7: Advanced SQLAlchemy Features")
    print("=" * 60)

    try:
        db = get_database("sqlite:///:memory:", backend="sqlalchemy")
        db.connect()

        # Transaction management
        print("\nTransaction management:")
        db.execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance REAL)")

        with db.begin_transaction():
            db.execute("INSERT INTO accounts (balance) VALUES (:bal)", {"bal": 100.0})
            print("  ✓ Transaction committed")

        # Access underlying session
        print(f"\nSQLAlchemy session: {type(db.session).__name__}")
        print(f"SQLAlchemy engine: {type(db.engine).__name__}")

        db.disconnect()
        print("✓ Advanced SQLAlchemy complete\n")

    except ImportError:
        print("⚠️  SQLAlchemy not installed (pip install sqlalchemy)\n")


def run_all_examples():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("DATABASE ABSTRACTION LAYER EXAMPLES")
    print("=" * 60 + "\n")

    example_basic_usage()
    example_explicit_backend()
    example_context_manager()
    example_comparison()
    example_global_default()
    example_advanced_nexusql()
    example_advanced_sqlalchemy()

    print("=" * 60)
    print("ALL EXAMPLES COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_examples()
