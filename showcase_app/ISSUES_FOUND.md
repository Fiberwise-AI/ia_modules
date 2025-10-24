# Showcase App Issues - SQL Execution Tracker Analysis

## Critical Issues Found

### 1. **SQL Syntax Incompatibility - ExecutionTracker**
**Location:** `ia_modules/web/execution_tracker.py`

**Problem:** The ExecutionTracker uses SQLite-specific syntax that won't work with PostgreSQL:

```python
# Line 102 - SQLite parameter style (?)
query = "SELECT * FROM pipeline_executions WHERE status IN (?, ?) ORDER BY started_at DESC"
rows = self.db.fetch_all(query, tuple(active_statuses))

# Line 446-448 - SQLite INSERT OR REPLACE
query = """
    INSERT OR REPLACE INTO pipeline_executions
    (execution_id, pipeline_id, pipeline_name, ...)
    VALUES (?, ?, ?, ...)
"""
```

**Impact:** 
- ExecutionTracker will fail when used with PostgreSQL (the default for showcase app)
- All execution tracking features will break
- Frontend won't receive real-time execution updates

**Fix Required:**
- Use named parameters (`:param_name`) instead of positional (`?`)
- Replace `INSERT OR REPLACE` with PostgreSQL `INSERT ... ON CONFLICT ... DO UPDATE`
- Use `self.db.execute_async()` for async operations

### 2. **Async/Sync Method Mismatch**
**Location:** `ia_modules/web/execution_tracker.py`

**Problem:** Async methods calling sync database operations:

```python
async def _save_execution(self, execution: ExecutionRecord):
    # ... prepare query ...
    self.db.execute(query, params)  # Sync call in async method!
```

**Impact:**
- Blocks the event loop
- Degrades performance
- May cause timeouts under load

**Fix Required:**
- Use `await self.db.execute_async(query, params)` throughout

### 3. **Missing async Methods in DatabaseManager**
**Location:** `ia_modules/database/manager.py`

**Problem:** DatabaseManager only has `execute_async()` but execution tracker needs:
- `fetch_all_async()` 
- `fetch_one_async()`

**Current state:**
```python
# Line 296 - Only execute_async exists
async def execute_async(self, query: str, params: Optional[Dict] = None) -> Any:
    return self.execute(query, params)  # Just wraps sync method
```

**Impact:**
- Can't properly query data asynchronously
- ExecutionTracker's query methods will fail

**Fix Required:**
Add these methods to DatabaseManager:
```python
async def fetch_all_async(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
    return self.fetch_all(query, params)

async def fetch_one_async(self, query: str, params: Optional[Dict] = None) -> Optional[Dict]:
    return self.fetch_one(query, params)
```

### 4. **Parameter Style Inconsistency**
**Location:** `showcase_app/backend/services/pipeline_service.py`

**Problem:** Pipeline service uses PostgreSQL-style parameters but ExecutionTracker uses SQLite-style:

```python
# pipeline_service.py - Line 87 (CORRECT for PostgreSQL)
query = """
    INSERT INTO pipeline_executions
    (execution_id, pipeline_id, ...)
    VALUES (:execution_id, :pipeline_id, ...)
    ON CONFLICT (execution_id) DO UPDATE SET ...
"""

# execution_tracker.py - Line 446 (WRONG for PostgreSQL)
query = """
    INSERT OR REPLACE INTO pipeline_executions
    (execution_id, pipeline_id, ...)
    VALUES (?, ?, ?, ...)
"""
```

**Impact:**
- Inconsistent behavior between components
- ExecutionTracker queries will fail with PostgreSQL

### 5. **Duplicate Execution Tracking**
**Location:** `showcase_app/backend/services/pipeline_service.py`

**Problem:** Both in-memory tracking AND database tracking:

```python
# Line 250-264 - Creates in-memory execution
self.executions[job_id] = {
    "job_id": job_id,
    "pipeline_id": pipeline_id,
    # ...
}

# Line 242-249 - ALSO uses ExecutionTracker
if self.tracker:
    job_id = await self.tracker.start_execution(
        pipeline_id=pipeline_id,
        # ...
    )
```

**Impact:**
- Data duplication
- Potential inconsistencies between memory and database
- Memory leaks (executions dict never cleaned up)

**Recommendation:**
- Remove in-memory dict
- Use ExecutionTracker exclusively
- Add method `get_execution_from_db()` to load on-demand

## Secondary Issues

### 6. **WebSocket Broadcast Not Implemented**
**Location:** `ia_modules/web/execution_tracker.py` line 556

```python
async def _broadcast_execution_update(self, execution: ExecutionRecord):
    """Broadcast execution update to connected WebSocket clients"""
    # Implementation needed - connect to actual WebSocket manager
    pass
```

**Impact:** Real-time updates won't work

### 7. **Missing Foreign Key Constraint in Migration**
**Location:** `database/migrations/V001__complete_schema.sql`

The `pipeline_executions.pipeline_id` has no foreign key to `pipelines.id`:

```sql
CREATE TABLE IF NOT EXISTS pipeline_executions (
    execution_id TEXT PRIMARY KEY,
    pipeline_id TEXT,  -- No FK constraint!
    -- ...
);
```

**Impact:** Orphaned execution records if pipelines deleted

### 8. **Health Check References Undefined Variables**
**Location:** `showcase_app/backend/main.py` line 111

```python
@app.get("/health")
async def health_check():
    return {
        # ...
        "services": {
            "metrics": metrics_service is not None,  # NameError!
            "pipelines": pipeline_service is not None,  # NameError!
        }
    }
```

**Fix:**
```python
"services": {
    "metrics": app.state.services.metrics_service is not None,
    "pipelines": app.state.services.pipeline_service is not None,
}
```

## Recommended Fixes Priority

### Priority 1 (Critical - Breaks Execution Tracking)
1. Fix ExecutionTracker SQL syntax for PostgreSQL compatibility
2. Add missing `fetch_all_async` and `fetch_one_async` to DatabaseManager
3. Fix async/sync method calls in ExecutionTracker

### Priority 2 (Important - Data Integrity)
4. Remove duplicate execution tracking in pipeline_service
5. Fix health check endpoint variable references

### Priority 3 (Enhancement)
6. Implement WebSocket broadcast functionality
7. Add foreign key constraints in migration

## Testing Checklist

After fixes:
- [ ] ExecutionTracker can start/update executions with PostgreSQL
- [ ] Execution list API returns data from database
- [ ] Execution detail API shows step execution data
- [ ] Real-time WebSocket updates work (if implemented)
- [ ] No memory leaks from in-memory execution dict
- [ ] Health check endpoint works
- [ ] Database migration creates all tables correctly

## Code Fixes Needed

### Fix 1: Update ExecutionTracker for PostgreSQL

File: `ia_modules/web/execution_tracker.py`

Replace all SQLite-style queries with PostgreSQL-compatible ones:
- Use `:param_name` instead of `?`
- Use `INSERT ... ON CONFLICT ... DO UPDATE` instead of `INSERT OR REPLACE`
- Use `await self.db.execute_async()` and `await self.db.fetch_all_async()`

### Fix 2: Add Async Query Methods to DatabaseManager

File: `ia_modules/database/manager.py`

Add:
```python
async def fetch_all_async(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
    return self.fetch_all(query, params)

async def fetch_one_async(self, query: str, params: Optional[Dict] = None) -> Optional[Dict]:
    return self.fetch_one(query, params)
```

### Fix 3: Clean Up Pipeline Service

File: `showcase_app/backend/services/pipeline_service.py`

Remove `self.executions` dict and related code, use only ExecutionTracker.

### Fix 4: Fix Health Check

File: `showcase_app/backend/main.py`

Update health check to use `app.state.services`.
