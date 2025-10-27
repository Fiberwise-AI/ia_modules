# Potential Database Issues Without Test Coverage

Based on analysis of `database/manager.py`, here are potential issues similar to the parameter order bug that may not have test coverage:

## 1. ⚠️ **fetch_one() May Not Handle MSSQL Dict Conversion**

**Location**: `database/manager.py:652-673`

**Issue**: The `fetch_one()` method uses `dict(row)` for all databases, but we just fixed this same issue in `execute()`. MSSQL rows need special handling.

```python
def fetch_one(self, query: str, params: Optional[Dict] = None) -> Optional[Dict]:
    try:
        cursor = self._execute_raw(query, params)
        row = cursor.fetchone()
        return dict(row) if row else None  # ❌ This will fail for MSSQL!
```

**Fix Needed**: Apply same MSSQL dict conversion as in `execute()`.

**Test Gap**: No `fetch_one()` tests with MSSQL.

---

## 2. ⚠️ **fetch_all() Also Missing MSSQL Dict Conversion**

**Location**: `database/manager.py:675-697`

**Issue**: Same as `fetch_one()` - uses `dict(row)` without MSSQL handling.

```python
def fetch_all(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
    cursor = self._execute_raw(query, params)
    rows = cursor.fetchall()
    return [dict(row) for row in rows] if rows else []  # ❌ MSSQL issue
```

**Fix Needed**: Apply same MSSQL dict conversion.

**Test Gap**: No `fetch_all()` tests with MSSQL.

---

## 3. ⚠️ **Transaction Rollback May Not Work Consistently**

**Location**: `database/manager.py:651-650` (execute method)

**Issue**: When query fails, rollback is called but there's no verification it succeeded. Different databases handle failed transactions differently.

```python
except Exception as e:
    if self._connection:
        try:
            self._connection.rollback()
            logger.debug("Rolled back transaction after error")
        except Exception:
            pass  # ❌ Silently ignoring rollback failures
```

**Concern**: Connection could be left in inconsistent state if rollback fails.

**Test Gap**: No tests for failed transactions and rollback across all databases.

---

## 4. ⚠️ **execute_script() Only Works for SQLite**

**Location**: `database/manager.py:765-778`

**Issue**: `execute_script()` uses SQLite's `executescript()` but falls back to splitting on `;` for other databases. This is fragile.

```python
if self.config.database_type == DatabaseType.SQLITE:
    self._connection.executescript(translated_script)
else:
    statements = [s.strip() for s in script.split(';') if s.strip()]
    for statement in statements:
        self.execute(statement)  # ❌ Naive split on ; is dangerous
```

**Problem**: SQL with string literals containing `;` will be split incorrectly.

**Test Gap**: No tests for multi-statement scripts with `;` in strings for PostgreSQL/MySQL/MSSQL.

---

## 5. ⚠️ **No Tests for NULL/None Parameter Handling**

**Issue**: How do different databases handle `None` values in params dict?

```python
db.execute("INSERT INTO table (col1, col2) VALUES (:val1, :val2)", {
    'val1': None,  # Is this NULL or 'None' string?
    'val2': 'value'
})
```

**Test Gap**: No tests verifying NULL handling across all databases.

---

## 6. ⚠️ **No Tests for Empty String vs NULL**

**Issue**: Databases handle empty strings differently (PostgreSQL: `''` ≠ NULL, Oracle: `'' = NULL`).

**Test Gap**: No tests for empty string parameter behavior.

---

## 7. ⚠️ **Datetime Parameter Handling Not Tested**

**Issue**: Python datetime objects may be handled differently across databases.

```python
from datetime import datetime
db.execute("INSERT INTO table (created_at) VALUES (:dt)", {
    'dt': datetime.now()  # How is this converted for each DB?
})
```

**Test Gap**: No tests for datetime parameter conversion across databases.

---

## 8. ⚠️ **Boolean Parameter Conversion for MSSQL Not Tested**

**Location**: `database/manager.py:546-549`

**Issue**: SQLite converts booleans to 0/1, but what about MSSQL?

```python
# Convert booleans to integers for SQLite
if self.config.database_type == DatabaseType.SQLITE and isinstance(value, bool):
    param_list.append(1 if value else 0)
```

**Question**: Does MSSQL need similar conversion?

**Test Gap**: No tests for boolean parameters with MSSQL.

---

## 9. ⚠️ **Binary/BLOB Parameter Handling Not Tested**

**Issue**: Passing binary data (bytes) as parameters may behave differently.

```python
db.execute("INSERT INTO table (image_data) VALUES (:data)", {
    'data': b'\\x89PNG...'  # Binary data
})
```

**Test Gap**: No tests for binary parameter handling.

---

## 10. ⚠️ **Large Parameter Values (>1MB) Not Tested**

**Issue**: Databases have different limits for parameter sizes.

**Test Gap**: No tests for large string/binary parameters.

---

## 11. ⚠️ **Unicode/Special Characters in Parameters Not Tested**

**Issue**: Non-ASCII characters, emojis, NULL bytes may cause issues.

```python
db.execute("INSERT INTO table (name) VALUES (:name)", {
    'name': '你好🎉\\x00'  # Unicode + emoji + null byte
})
```

**Test Gap**: No tests for special character handling.

---

## 12. ⚠️ **Parameter Name Collision with SQL Keywords**

**Issue**: Using SQL keywords as parameter names may cause issues.

```python
db.execute("SELECT * FROM table WHERE select = :select", {
    'select': 'value'  # 'select' is SQL keyword
})
```

**Test Gap**: No tests for keyword parameter names.

---

## 13. ⚠️ **Connection Pooling/Concurrent Access Not Tested**

**Issue**: Multiple threads/async tasks using same DatabaseManager.

**Test Gap**: No tests for concurrent database access.

---

## 14. ⚠️ **Database-Specific Error Handling**

**Issue**: Different databases raise different exceptions for same errors.

**Test Gap**: No tests verifying error types across databases (IntegrityError, OperationalError, etc.).

---

## 15. ⚠️ **Auto-increment/SERIAL ID Return Values**

**Issue**: After INSERT with auto-increment, getting the ID back works differently.

**Test Gap**: No tests for retrieving last insert ID across databases.

---

## Recommended Actions

### High Priority (Fix Immediately)
1. ✅ Fix `fetch_one()` MSSQL dict conversion
2. ✅ Fix `fetch_all()` MSSQL dict conversion
3. ⚠️ Add tests for NULL parameters
4. ⚠️ Add tests for datetime parameters
5. ⚠️ Fix `execute_script()` SQL splitting

### Medium Priority
6. Add tests for boolean parameters with MSSQL
7. Add tests for empty string vs NULL
8. Add tests for Unicode/special characters
9. Add tests for binary data
10. Improve transaction rollback error handling

### Low Priority
11. Test large parameter values
12. Test parameter names that are SQL keywords
13. Test concurrent access
14. Test database-specific errors
15. Test auto-increment ID retrieval
