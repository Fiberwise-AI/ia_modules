#!/usr/bin/env python3
"""Test MSSQL SQL translation"""

from database.manager import DatabaseManager
from database.interfaces import ConnectionConfig, DatabaseType

# Read the migration file
with open('database/migrations/V001__complete_schema.sql', 'r') as f:
    sql = f.read()

# Create a manager with MSSQL config
config = ConnectionConfig(
    database_type=DatabaseType.MSSQL,
    database_url="mssql://sa:TestPass123!@localhost:11433/master"
)
manager = DatabaseManager(config)

# Translate the SQL
translated = manager._translate_sql(sql)

# Print first 50 lines
lines = translated.split('\n')
for i, line in enumerate(lines[:100], 1):
    print(f"{i:3d}: {line}")
