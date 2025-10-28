"""
Regression test for parameter order bug in _convert_params

This test specifically validates that UPDATE queries with multiple parameters
work correctly even when the params dict is in a different order than the query.

Bug: Previously, SQLite/MySQL/MSSQL would build the positional param tuple in dict
iteration order, not query appearance order, causing silent failures.

Fix: Use regex to extract parameter names in query order, then build tuple accordingly.
"""

import pytest
from nexusql import DatabaseManager


@pytest.fixture(params=["sqlite", "postgresql", "mysql"], ids=lambda x: x)
def db(request, db_manager):
    """Use parameterized db_manager from conftest"""
    return db_manager


def test_update_with_params_in_wrong_dict_order(db):
    """
    Test UPDATE with params dict in different order than query.

    This is a regression test for the parameter order bug where positional
    databases (SQLite, MySQL, MSSQL) would fail when dict order != query order.
    """
    # Drop table if exists (for parameterized tests)
    try:
        db.execute("DROP TABLE IF EXISTS test_param_order")
    except:
        pass

    # Create table
    db.execute("""
        CREATE TABLE test_param_order (
            id INTEGER PRIMARY KEY,
            field1 TEXT,
            field2 TEXT,
            field3 TEXT,
            field4 TEXT
        )
    """)

    # Insert test data
    db.execute(
        "INSERT INTO test_param_order (id, field1, field2, field3, field4) VALUES (:id, :f1, :f2, :f3, :f4)",
        {"id": 1, "f1": "a", "f2": "b", "f3": "c", "f4": "d"}
    )

    # UPDATE with params in DIFFERENT order than query
    # Query order: field2, field3, field4, field1, id
    # Dict order:  id, field1, field2, field3, field4
    db.execute("""
        UPDATE test_param_order
        SET field2 = :field2,
            field3 = :field3,
            field4 = :field4,
            field1 = :field1
        WHERE id = :id
    """, {
        'id': 1,  # First in dict
        'field1': 'new_a',  # Second in dict
        'field2': 'new_b',  # Third in dict
        'field3': 'new_c',  # Fourth in dict
        'field4': 'new_d'   # Fifth in dict
    })

    # Verify all fields updated correctly
    result = db.execute("SELECT * FROM test_param_order WHERE id = :id", {"id": 1})[0]
    assert result['field1'] == 'new_a', f"field1 should be 'new_a', got {result['field1']}"
    assert result['field2'] == 'new_b', f"field2 should be 'new_b', got {result['field2']}"
    assert result['field3'] == 'new_c', f"field3 should be 'new_c', got {result['field3']}"
    assert result['field4'] == 'new_d', f"field4 should be 'new_d', got {result['field4']}"


def test_update_multiple_fields_reverse_order(db):
    """
    Test UPDATE with params in completely reverse order.

    Even more extreme case - params dict in reverse order of query.
    """
    # Drop table if exists
    try:
        db.execute("DROP TABLE IF EXISTS test_reverse_order")
    except:
        pass

    # Create table
    db.execute("""
        CREATE TABLE test_reverse_order (
            id INTEGER PRIMARY KEY,
            name TEXT,
            email TEXT,
            age INTEGER
        )
    """)

    # Insert test data
    db.execute(
        "INSERT INTO test_reverse_order (id, name, email, age) VALUES (:id, :name, :email, :age)",
        {"id": 1, "name": "Alice", "email": "alice@example.com", "age": 30}
    )

    # UPDATE with params in REVERSE order
    # Query order: name, email, age, id
    # Dict order:  id, age, email, name (reverse!)
    db.execute("""
        UPDATE test_reverse_order
        SET name = :name,
            email = :email,
            age = :age
        WHERE id = :id
    """, {
        'id': 1,          # Last in query, first in dict
        'age': 31,        # Third in query, second in dict
        'email': 'alice_new@example.com',  # Second in query, third in dict
        'name': 'Alice Smith'  # First in query, last in dict
    })

    # Verify
    result = db.execute("SELECT * FROM test_reverse_order WHERE id = :id", {"id": 1})[0]
    assert result['name'] == 'Alice Smith'
    assert result['email'] == 'alice_new@example.com'
    assert result['age'] == 31


def test_insert_with_params_in_wrong_order(db):
    """Test INSERT also handles param order correctly"""
    # Drop table if exists
    try:
        db.execute("DROP TABLE IF EXISTS test_insert_order")
    except:
        pass

    # Create table
    db.execute("""
        CREATE TABLE test_insert_order (
            id INTEGER PRIMARY KEY,
            col_a TEXT,
            col_b TEXT,
            col_c TEXT
        )
    """)

    # INSERT with params in different order
    # Query order: col_c, id, col_a, col_b
    # Dict order:  id, col_a, col_b, col_c
    db.execute("""
        INSERT INTO test_insert_order (col_c, id, col_a, col_b)
        VALUES (:col_c, :id, :col_a, :col_b)
    """, {
        'id': 1,
        'col_a': 'value_a',
        'col_b': 'value_b',
        'col_c': 'value_c'
    })

    # Verify
    result = db.execute("SELECT * FROM test_insert_order WHERE id = :id", {"id": 1})[0]
    assert result['col_a'] == 'value_a'
    assert result['col_b'] == 'value_b'
    assert result['col_c'] == 'value_c'


def test_complex_update_like_hitl(db):
    """
    Reproduce the exact pattern from HITL that exposed the bug.

    This mimics the HITL respond_to_interaction UPDATE that failed.
    """
    # Drop table if exists
    try:
        db.execute("DROP TABLE IF EXISTS test_hitl_like")
    except:
        pass

    # Create table similar to hitl_interactions
    # Use VARCHAR(255) instead of TEXT for PRIMARY KEY (MySQL compatibility)
    db.execute("""
        CREATE TABLE test_hitl_like (
            interaction_id VARCHAR(255) PRIMARY KEY,
            status TEXT,
            human_input TEXT,
            responded_by TEXT,
            completed_at TEXT
        )
    """)

    # Insert pending interaction
    db.execute("""
        INSERT INTO test_hitl_like (interaction_id, status, human_input, responded_by, completed_at)
        VALUES (:id, :status, :input, :by, :at)
    """, {
        'id': 'test-123',
        'status': 'pending',
        'input': None,
        'by': None,
        'at': None
    })

    # UPDATE exactly like HITL does - params dict has interaction_id first!
    # Query order: status, human_input, responded_by, completed_at, interaction_id
    # Dict order:  interaction_id, human_input, responded_by, completed_at (status not in dict, it's literal)
    db.execute("""
        UPDATE test_hitl_like
        SET status = 'completed',
            human_input = :human_input,
            responded_by = :responded_by,
            completed_at = :completed_at
        WHERE interaction_id = :interaction_id
    """, {
        'interaction_id': 'test-123',  # Last in query, first in dict
        'human_input': '{"decision": "approve"}',
        'responded_by': 'user@example.com',
        'completed_at': '2025-01-01T00:00:00'
    })

    # Verify - THIS FAILED BEFORE THE FIX!
    result = db.execute("SELECT * FROM test_hitl_like WHERE interaction_id = :id", {"id": 'test-123'})[0]
    assert result['status'] == 'completed'
    assert result['human_input'] == '{"decision": "approve"}'
    assert result['responded_by'] == 'user@example.com'
    assert result['completed_at'] == '2025-01-01T00:00:00'
