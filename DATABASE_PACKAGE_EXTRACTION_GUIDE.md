# Database Package Extraction Guide

**Goal**: Extract `ia_modules/database` into a separate pip package (`ia-database`) and make it a dependency of `ia_modules`.

**Difficulty**: ⭐⭐⚪⚪⚪ (Easy - module is already well-isolated)
None: the neam of the backage should be called : nexusql
---

## Table of Contents
1. [Current State Analysis](#current-state-analysis)
2. [Benefits of Extraction](#benefits-of-extraction)
3. [Step-by-Step Extraction Plan](#step-by-step-extraction-plan)
4. [Migration Strategy](#migration-strategy)
5. [Testing Strategy](#testing-strategy)
6. [Rollback Plan](#rollback-plan)

---

## Current State Analysis

### Module Structure
```
ia_modules/
├── database/
│   ├── __init__.py           # Public API
│   ├── interfaces.py         # 150 lines - Type definitions
│   ├── manager.py            # 800 lines - Core DatabaseManager
│   ├── migrations.py         # 250 lines - Migration system
│   └── migrations/
│       └── V001__complete_schema.sql
```

### Dependencies Analysis

#### ✅ **Database Module Has NO ia_modules Dependencies**
```python
# database/__init__.py
from .manager import DatabaseManager
from .interfaces import ConnectionConfig, DatabaseType

# database/manager.py
import sqlite3
import logging
import re
from pathlib import Path
from typing import Optional, Any, Dict, List
from .interfaces import ConnectionConfig, DatabaseType, QueryResult

# database/migrations.py
import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from .interfaces import DatabaseInterface, QueryResult
```

**Result**: ✅ Database module is **completely self-contained** with zero internal dependencies!

#### 📊 **ia_modules Dependencies ON Database** (54 files)

**Core Modules** (13 files):
- `pipeline/hitl_manager.py` - HITL database operations
- `pipeline/importer.py` - Pipeline import with validation
- `pipeline/db_step_loader.py` - Load pipeline steps from database
- `reliability/sql_metric_storage.py` - Metrics storage
- `checkpoint/sql.py` - Checkpoint storage
- `showcase_app/backend/main.py` - FastAPI app initialization
- `showcase_app/backend/services/*.py` - Service layer (3 files)
- `showcase_app/backend/worker.py` - Background worker
- `showcase_app/backend/api/step_modules.py` - Step API

**Test Files** (41 files):
- All database tests
- Integration tests using database
- End-to-end tests

---

## Benefits of Extraction

### 1. ✅ **Independent Versioning**
- Database package can have its own release cycle
- Breaking changes don't force ia_modules version bump
- Faster iteration on database features

### 2. ✅ **Reusability**
- Use in other projects without pulling in all of ia_modules
- Lightweight dependency for pure database needs
- Easier to contribute to (smaller codebase)

### 3. ✅ **Clearer Boundaries**
- Explicit public API
- Better documentation
- Forces design discipline

### 4. ✅ **Easier Testing**
- Test database package independently
- Faster CI/CD (smaller test suite)
- Pin to specific versions for stability

### 5. ✅ **Better Maintenance**
- Separate issue tracker
- Dedicated maintainers
- Clear responsibility boundaries

---

## Step-by-Step Extraction Plan

### Phase 1: Create New Package (1-2 hours)

#### 1.1 Create Package Structure
```bash
# Create new directory for ia-database package
mkdir -p ../ia-database
cd ../ia-database

# Initialize git repo
git init
git remote add origin git@github.com:yourorg/ia-database.git

# Create package structure
mkdir -p ia_database
mkdir -p tests
mkdir -p docs
```

#### 1.2 Create Package Files
```bash
# ia-database/
├── ia_database/
│   ├── __init__.py
│   ├── interfaces.py
│   ├── manager.py
│   ├── migrations.py
│   └── migrations/
│       └── V001__complete_schema.sql
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── integration/
│   │   ├── test_database_complete.py
│   │   ├── test_database_param_order.py
│   │   └── test_database_edge_cases.py
│   └── unit/
│       └── test_database_manager.py
├── docs/
│   ├── README.md
│   ├── API_REFERENCE.md
│   └── MIGRATION_GUIDE.md
├── pyproject.toml
├── setup.py
├── README.md
├── LICENSE
└── .gitignore
```

#### 1.3 Copy Files
```bash
# Copy database module files
cp -r ../ia_modules/ia_modules/database/* ia_database/

# Copy relevant tests
cp ../ia_modules/tests/integration/test_database_*.py tests/integration/
cp ../ia_modules/tests/unit/test_database_manager.py tests/unit/
cp ../ia_modules/tests/conftest.py tests/

# Copy documentation
cp ../ia_modules/docs/DATABASE_*.md docs/
```

#### 1.4 Create `pyproject.toml`
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "ia-database"
version = "0.1.0"
description = "Multi-database abstraction layer with unified API"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
keywords = ["database", "sql", "postgresql", "mysql", "sqlite", "mssql"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Database",
]

dependencies = [
    "psycopg2-binary>=2.9.0",  # PostgreSQL
    "pymysql>=1.0.0",          # MySQL
    "pyodbc>=4.0.0",           # MSSQL
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
]

[project.urls]
Homepage = "https://github.com/yourorg/ia-database"
Documentation = "https://ia-database.readthedocs.io"
Repository = "https://github.com/yourorg/ia-database"
Issues = "https://github.com/yourorg/ia-database/issues"

[tool.setuptools.packages.find]
where = ["."]
include = ["ia_database*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
addopts = "-v --tb=short"
```

#### 1.5 Create README.md
```markdown
# ia-database

Multi-database abstraction layer with unified API for PostgreSQL, MySQL, SQLite, and MSSQL.

## Features

- ✅ **Unified API** - Same code works across 4 databases
- ✅ **Named Parameters** - Safe from SQL injection
- ✅ **Auto-migrations** - Versioned schema management
- ✅ **Type Safety** - Full typing support
- ✅ **Async Ready** - Async/await compatible
- ✅ **Production Ready** - Comprehensive test suite

## Installation

```bash
pip install ia-database

# With specific database drivers
pip install ia-database[postgresql]  # PostgreSQL only
pip install ia-database[mysql]       # MySQL only
pip install ia-database[mssql]       # MSSQL only
pip install ia-database[all]         # All drivers
```

## Quick Start

```python
from ia_database import DatabaseManager

# SQLite (no server needed)
db = DatabaseManager("sqlite:///app.db")

# PostgreSQL
db = DatabaseManager("postgresql://user:pass@localhost:5432/mydb")

# MySQL
db = DatabaseManager("mysql://user:pass@localhost:3306/mydb")

# MSSQL
db = DatabaseManager("mssql+pyodbc://user:pass@localhost:1433/mydb")

# Initialize with schema migrations
await db.initialize(apply_schema=True)

# Execute queries with named parameters (safe from SQL injection)
users = db.execute(
    "SELECT * FROM users WHERE email = :email",
    {"email": "user@example.com"}
)

# Fetch single row
user = db.fetch_one(
    "SELECT * FROM users WHERE id = :id",
    {"id": 123}
)

# Transactions
async with db.transaction():
    db.execute("INSERT INTO users (email) VALUES (:email)", {"email": "new@example.com"})
    db.execute("INSERT INTO profiles (user_id) VALUES (:id)", {"id": 123})
    # Auto-commit on success, rollback on exception
```

## Documentation

- [API Reference](docs/API_REFERENCE.md)
- [Migration Guide](docs/MIGRATION_GUIDE.md)
- [Testing Guide](docs/TESTING_GUIDE.md)

## License

MIT
```

### Phase 2: Update ia_modules to Use New Package (1 hour)

#### 2.1 Update ia_modules pyproject.toml
```toml
[project]
dependencies = [
    "ia-database>=0.1.0",  # Add new dependency
    # ... other dependencies
]

[project.optional-dependencies]
dev = [
    "ia-database[dev]>=0.1.0",  # Include dev dependencies for testing
    # ... other dev dependencies
]
```

#### 2.2 Update Import Statements (Find & Replace)

**Option A: Minimal Changes (Recommended)**
```python
# Keep using ia_modules.database as import path
# Add to ia_modules/database/__init__.py:
from ia_database import DatabaseManager, ConnectionConfig, DatabaseType

__all__ = ['DatabaseManager', 'ConnectionConfig', 'DatabaseType']
```

**Result**: No code changes needed in rest of codebase!

**Option B: Update All Imports (More Work)**
```bash
# Find all imports
grep -r "from ia_modules.database" --include="*.py" ia_modules/

# Replace with
# from ia_modules.database import X
# → from ia_database import X

# Use sed or IDE find-replace
find ia_modules -name "*.py" -exec sed -i 's/from ia_modules\.database/from ia_database/g' {} +
```

#### 2.3 Remove Old Database Files (After Verification)
```bash
# Move to archive first (don't delete yet)
mkdir -p ia_modules/.archive
mv ia_modules/database ia_modules/.archive/database_old

# Create new shim module (if using Option A)
mkdir -p ia_modules/database
cat > ia_modules/database/__init__.py << 'EOF'
"""
Compatibility shim for ia-database package.
This maintains backward compatibility with existing imports.
"""
from ia_database import DatabaseManager, ConnectionConfig, DatabaseType

__all__ = ['DatabaseManager', 'ConnectionConfig', 'DatabaseType']
EOF
```

### Phase 3: Publish Package (30 minutes)

#### 3.1 Build Package
```bash
cd ia-database

# Install build tools
pip install build twine

# Build distributions
python -m build

# Result:
# dist/
#   ia_database-0.1.0-py3-none-any.whl
#   ia_database-0.1.0.tar.gz
```

#### 3.2 Publish to PyPI
```bash
# Test on TestPyPI first
python -m twine upload --repository testpypi dist/*

# Install from TestPyPI to verify
pip install --index-url https://test.pypi.org/simple/ ia-database

# If all works, publish to real PyPI
python -m twine upload dist/*
```

#### 3.3 Publish to Private PyPI (Alternative)
```bash
# If using private PyPI server or Artifactory
twine upload --repository-url https://your-pypi.example.com/simple/ dist/*

# Or add to pyproject.toml:
[tool.poetry.source]
name = "private-pypi"
url = "https://your-pypi.example.com/simple/"
```

### Phase 4: Testing & Verification (2 hours)

#### 4.1 Test ia-database Independently
```bash
cd ia-database

# Install package locally in editable mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v --cov=ia_database

# Expected: All 224 tests passing (48 param order + 176 edge cases)
```

#### 4.2 Test ia_modules with New Dependency
```bash
cd ia_modules

# Uninstall old package (if testing locally)
pip uninstall ia-database

# Install from PyPI
pip install ia-database==0.1.0

# Or install local editable version for development
pip install -e ../ia-database

# Run ia_modules tests
pytest tests/unit/ -v
pytest tests/integration/ -v

# Verify database tests still pass
pytest tests/integration/test_database_*.py -v
pytest tests/unit/test_database_manager.py -v
```

#### 4.3 Test Imports
```python
# Test that both import styles work

# Old style (backward compatible)
from ia_modules.database import DatabaseManager
db = DatabaseManager("sqlite:///:memory:")

# New style (direct)
from ia_database import DatabaseManager
db = DatabaseManager("sqlite:///:memory:")

# Both should work!
```

### Phase 5: Documentation & Communication (1 hour)

#### 5.1 Update ia_modules README
```markdown
## Dependencies

- **ia-database** - Multi-database abstraction layer
  - Supports PostgreSQL, MySQL, SQLite, MSSQL
  - Unified API with named parameters
  - See [ia-database docs](https://github.com/yourorg/ia-database)
```

#### 5.2 Create Migration Guide
```markdown
# Migrating to ia-database Package

## For Users

**No changes needed!** The database module still works the same way:

```python
from ia_modules.database import DatabaseManager  # Still works!
```

## For Contributors

The database code now lives in a separate repository:
- Repository: https://github.com/yourorg/ia-database
- Issues: https://github.com/yourorg/ia-database/issues
- Docs: https://ia-database.readthedocs.io

## For Developers Using ia-database Directly

You can now use the database package without ia_modules:

```bash
pip install ia-database
```

```python
from ia_database import DatabaseManager  # New direct import
```
```

#### 5.3 Announce Changes
```markdown
# Announcement: Database Module Extracted to ia-database

We've extracted the database module into a separate package: **ia-database**

## What Changed?
- Database code moved to https://github.com/yourorg/ia-database
- Now available as standalone pip package: `pip install ia-database`
- Can be used in other projects without ia_modules dependency

## What Stayed the Same?
- ✅ All imports still work (`from ia_modules.database import DatabaseManager`)
- ✅ All APIs unchanged
- ✅ All tests passing
- ✅ No breaking changes

## Benefits?
- Faster database feature iteration
- Reusable in other projects
- Better testing and documentation
- Independent versioning
```

---

## Migration Strategy

### Approach 1: Big Bang Migration (Recommended)
**Timeline**: 1 day
**Risk**: Low (database module is isolated)

1. ✅ Create ia-database package (2 hours)
2. ✅ Publish to PyPI (30 minutes)
3. ✅ Update ia_modules dependency (15 minutes)
4. ✅ Test everything (2 hours)
5. ✅ Deploy to production (1 hour)

**Pros**: Clean, fast, minimal coordination
**Cons**: Requires coordinated release

### Approach 2: Gradual Migration
**Timeline**: 1 week
**Risk**: Very Low (backward compatible)

**Week 1**: Create ia-database package, publish as beta
- Publish ia-database 0.1.0-beta1
- Keep ia_modules/database as-is
- Test in staging

**Week 2**: Add dependency, keep shim
- Add ia-database to dependencies
- Keep ia_modules/database/__init__.py as shim
- Deploy to production

**Week 3**: Remove old code
- Archive ia_modules/.archive/database_old
- Monitor for issues

**Pros**: Very safe, easy rollback
**Cons**: Slower, more coordination

---

## Testing Strategy

### Pre-Extraction Tests
```bash
# Baseline: Run all tests before extraction
cd ia_modules
pytest tests/ -v --cov=ia_modules --cov-report=term > baseline_coverage.txt

# Save test results
# Expected: ~1900 passing tests
```

### Post-Extraction Tests

#### Test ia-database Package
```bash
cd ia-database
pytest tests/ -v --cov=ia_database --cov-report=html

# Must have:
# - 48 param order tests passing
# - 176 edge case tests passing
# - 100% coverage on manager.py core functions
```

#### Test ia_modules with New Dependency
```bash
cd ia_modules
pytest tests/ -v --cov=ia_modules --cov-report=term > post_extraction_coverage.txt

# Compare with baseline
diff baseline_coverage.txt post_extraction_coverage.txt

# Expected: No differences in test results
```

#### Integration Tests
```bash
# Test that database operations work end-to-end
cd ia_modules
pytest tests/integration/ -v -k database

# Test HITL (heavy database user)
pytest tests/unit/test_hitl_comprehensive.py -v

# Test pipeline execution with database
pytest tests/integration/test_pipeline_execution_e2e.py -v
```

---

## Rollback Plan

### If Issues Found After Deployment

#### Option 1: Rollback to Embedded Database (Fast - 15 minutes)
```bash
# Restore old database code
cp -r ia_modules/.archive/database_old ia_modules/database

# Remove ia-database dependency from pyproject.toml
# [dependencies]
# ia-database>=0.1.0  # ← Remove this line

# Reinstall
pip install -e .

# Deploy
```

#### Option 2: Fix Forward (Preferred if issue is minor)
```bash
# Fix bug in ia-database
cd ia-database
# ... make fix ...

# Bump version
# pyproject.toml: version = "0.1.1"

# Publish hotfix
python -m build
twine upload dist/*

# Update ia_modules
# pyproject.toml: ia-database>=0.1.1

pip install --upgrade ia-database
```

---

## Potential Issues & Solutions

### Issue 1: Import Errors
**Problem**: `ModuleNotFoundError: No module named 'ia_database'`

**Solution**:
```bash
# Ensure ia-database is installed
pip install ia-database

# Or install from local repo during development
pip install -e ../ia-database
```

### Issue 2: Version Conflicts
**Problem**: `ia_modules requires ia-database>=0.2.0 but 0.1.0 is installed`

**Solution**:
```bash
# Upgrade to latest version
pip install --upgrade ia-database

# Or pin to compatible version
pip install "ia-database>=0.1.0,<0.2.0"
```

### Issue 3: Migration Files Not Found
**Problem**: `FileNotFoundError: V001__complete_schema.sql`

**Solution**:
```python
# Ensure migrations are included in package

# ia-database/pyproject.toml:
[tool.setuptools.package-data]
ia_database = ["migrations/*.sql"]

# Or use MANIFEST.in:
include ia_database/migrations/*.sql
```

### Issue 4: Test Failures
**Problem**: Tests fail after extraction

**Solution**:
```bash
# Check import paths in tests
# Old: from ia_modules.database import DatabaseManager
# New: from ia_database import DatabaseManager

# Or use shim (keep old imports working):
# ia_modules/database/__init__.py
from ia_database import *
```

---

## Future Enhancements

Once extraction is complete, these become easier:

1. **Connection Pooling** - Add without affecting ia_modules
2. **Query Builder** - Optional feature in ia-database
3. **ORM Layer** - Can be separate ia-database-orm package
4. **Performance Optimizations** - Faster iteration cycle
5. **Multi-Database Transactions** - Advanced feature

---

## Checklist

### Pre-Extraction
- [ ] Run full test suite (baseline)
- [ ] Document current test coverage
- [ ] Review all database imports
- [ ] Check for hidden dependencies

### During Extraction
- [ ] Create ia-database repo
- [ ] Copy database files
- [ ] Copy tests
- [ ] Create pyproject.toml
- [ ] Create README
- [ ] Build package
- [ ] Test package independently
- [ ] Publish to TestPyPI
- [ ] Test install from TestPyPI

### Post-Extraction
- [ ] Publish to PyPI
- [ ] Update ia_modules dependencies
- [ ] Test ia_modules with new package
- [ ] Run full test suite (compare with baseline)
- [ ] Update documentation
- [ ] Announce changes
- [ ] Monitor for issues

### Verification
- [ ] All 224 database tests pass in ia-database
- [ ] All ia_modules tests pass (no regressions)
- [ ] Imports work both ways (backward compatible)
- [ ] PyPI package installable
- [ ] Documentation up to date

---

## Conclusion

**Difficulty**: ⭐⭐⚪⚪⚪ **EASY**

The database module is already well-isolated with zero internal dependencies. Extraction is straightforward:

1. Copy 4 files + tests to new repo
2. Create pyproject.toml
3. Publish to PyPI
4. Add dependency to ia_modules
5. Done!

**Estimated Time**: 4-6 hours total (including testing)

**Risk Level**: **Low** - Database module is self-contained and heavily tested

**Recommendation**: ✅ **Proceed with Big Bang Migration** - Clean, fast, low risk
