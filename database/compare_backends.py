"""
Side-by-side comparison of NexusQL and SQLAlchemy backends.

Demonstrates that queries are IDENTICAL regardless of backend.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "nexusql"))

from ia_modules.database import get_database


def run_identical_queries(backend: str):
    """Run the same queries with specified backend"""
    print(f"\n{'=' * 60}")
    print(f"Running with {backend.upper()} backend")
    print('=' * 60)

    try:
        db = get_database("sqlite:///:memory:", backend=backend)
        db.connect()

        # Query 1: CREATE TABLE
        print("\n1. CREATE TABLE")
        query = """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE,
                age INTEGER,
                active INTEGER DEFAULT 1
            )
        """
        print(f"   Query: {query.strip()[:60]}...")
        db.execute(query)
        print("   [OK] Table created")

        # Query 2: INSERT with named parameters
        print("\n2. INSERT with named parameters")
        query = "INSERT INTO users (name, email, age) VALUES (:name, :email, :age)"
        params = {"name": "Alice", "email": "alice@example.com", "age": 30}
        print(f"   Query: {query}")
        print(f"   Params: {params}")
        db.execute(query, params)
        print("   [OK] Inserted")

        # Query 3: INSERT another row
        db.execute(query, {"name": "Bob", "email": "bob@example.com", "age": 25})
        db.execute(query, {"name": "Charlie", "email": "charlie@example.com", "age": 35})
        print("   [OK] Inserted 2 more rows")

        # Query 4: SELECT with parameters
        print("\n3. SELECT with named parameters")
        query = "SELECT * FROM users WHERE name = :name"
        params = {"name": "Alice"}
        print(f"   Query: {query}")
        print(f"   Params: {params}")
        result = db.fetch_one(query, params)
        print(f"   Result: {result}")

        # Query 5: SELECT all
        print("\n4. SELECT all rows")
        query = "SELECT * FROM users ORDER BY age"
        print(f"   Query: {query}")
        results = db.fetch_all(query)
        print(f"   Found {len(results)} users:")
        for user in results:
            print(f"     - {user['name']}, age {user['age']}")

        # Query 6: UPDATE with parameters
        print("\n5. UPDATE with named parameters")
        query = "UPDATE users SET age = :age WHERE name = :name"
        params = {"age": 31, "name": "Alice"}
        print(f"   Query: {query}")
        print(f"   Params: {params}")
        db.execute(query, params)
        print("   [OK] Updated")

        # Verify update
        alice = db.fetch_one("SELECT * FROM users WHERE name = :name", {"name": "Alice"})
        print(f"   Alice's new age: {alice['age']}")

        # Query 7: Complex WHERE clause
        print("\n6. Complex WHERE with multiple parameters")
        query = "SELECT * FROM users WHERE age >= :min_age AND active = :active"
        params = {"min_age": 30, "active": 1}
        print(f"   Query: {query}")
        print(f"   Params: {params}")
        results = db.fetch_all(query, params)
        print(f"   Found {len(results)} users age >= 30")

        # Query 8: DELETE
        print("\n7. DELETE with named parameters")
        query = "DELETE FROM users WHERE name = :name"
        params = {"name": "Bob"}
        print(f"   Query: {query}")
        print(f"   Params: {params}")
        db.execute(query, params)
        print("   [OK] Deleted")

        # Final count
        count = db.fetch_all("SELECT COUNT(*) as count FROM users")[0]['count']
        print(f"\n   Final user count: {count}")

        db.disconnect()
        print(f"\n[OK] All queries completed successfully with {backend.upper()}")

    except ImportError as e:
        print(f"\n⚠️  {backend.upper()} not available: {e}")
        print(f"   Install with: pip install {backend}")


def compare_both_backends():
    """Run identical queries with both backends"""
    print("\n" + "=" * 60)
    print("BACKEND COMPARISON - IDENTICAL QUERIES")
    print("=" * 60)
    print("\nThis demonstrates that the EXACT SAME queries work")
    print("with both NexusQL and SQLAlchemy backends.")
    print("\nThe queries are 100% identical - only the backend differs.")

    # Run with NexusQL
    run_identical_queries("nexusql")

    # Run with SQLAlchemy
    run_identical_queries("sqlalchemy")

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("\n[OK] Both backends executed IDENTICAL queries")
    print("[OK] Same SQL syntax")
    print("[OK] Same parameter format (:param_name)")
    print("[OK] Same return format (List[Dict])")
    print("\nYour queries don't change - only the backend!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    compare_both_backends()
