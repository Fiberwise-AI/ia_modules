# Publishing ia_modules to PyPI

This document describes how to publish `ia_modules` to PyPI using GitHub Actions.

## Quick Summary

✅ **What's ready:**
- GitHub Actions workflow (`.github/workflows/publish.yml`)
- Build configuration (`pyproject.toml`)
- Test script for local validation (`test_publish.py`)
- Package metadata and dependencies

## Automated Publishing (Recommended)

The package automatically publishes to PyPI when you:
1. Create a GitHub release, OR
2. Push a tag starting with `v` (e.g., `v0.0.5`)

### Prerequisites

1. **Create a PyPI API token:**
   - Go to https://pypi.org/manage/account/token/
   - Create a token scoped to the `ia_modules` project (or create after first manual upload)
   - Copy the token (starts with `pypi-`)

2. **Add GitHub repository secret:**
   - Go to repository Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `PYPI_API_TOKEN`
   - Value: paste your PyPI token
   - Click "Add secret"

### Release Process

```pwsh
# 1. Update version in pyproject.toml
#    version = "0.0.5"

# 2. Commit and tag
git add pyproject.toml
git commit -m "Release v0.0.5"
git tag v0.0.5
git push origin main --tags

# 3. GitHub Actions will automatically:
#    - Run unit tests
#    - Build the package
#    - Verify with twine
#    - Upload to PyPI
```

Or create a release via GitHub UI:
- Go to Releases → Draft a new release
- Create tag (e.g., `v0.0.5`)
- Add release notes
- Click "Publish release"

## Local Testing (Before Publishing)

Test the build process locally before publishing:

```pwsh
# Quick test (skip tests, just build & check)
python test_publish.py --skip-tests

# Full test (run tests, build, check)
python test_publish.py

# Test publish to TestPyPI
$env:TEST_PYPI_TOKEN = "your-testpypi-token"
python test_publish.py --test-pypi
```

### Manual Steps for Testing

```pwsh
# 1. Clean previous builds
Remove-Item -Recurse -Force dist, build, *.egg-info -ErrorAction SilentlyContinue

# 2. Install build tools
python -m pip install build twine

# 3. Build package
python -m build

# 4. Check package
python -m twine check dist/*

# Output should show:
# Checking dist\ia_modules-0.0.4-py3-none-any.whl: PASSED
# Checking dist\ia_modules-0.0.4.tar.gz: PASSED
```

## Manual Publishing (Not Recommended)

If you need to publish manually:

```pwsh
# Build the package
python -m build

# Upload to PyPI
python -m twine upload dist/* -u __token__ -p YOUR_PYPI_TOKEN
```

## Workflow Details

The GitHub Actions workflow (`.github/workflows/publish.yml`):
1. ✅ Checks out code
2. ✅ Sets up Python 3.11
3. ✅ Installs dependencies
4. ✅ Runs unit tests (`pytest tests/unit/`)
5. ✅ Builds package (`python -m build`)
6. ✅ Verifies package (`twine check dist/*`)
7. ✅ Publishes to PyPI (if `PYPI_API_TOKEN` secret exists)

## Testing the Workflow

To test without publishing:
1. Comment out the "Publish to PyPI" step in `.github/workflows/publish.yml`
2. Push a test tag: `git tag test-v0.0.4 && git push origin test-v0.0.4`
3. Check the Actions tab to see if build succeeds
4. Delete test tag: `git push --delete origin test-v0.0.4 && git tag -d test-v0.0.4`

## TestPyPI (Optional Staging)

To test publishing to TestPyPI first:

1. Create token at https://test.pypi.org/manage/account/token/
2. Add `TEST_PYPI_TOKEN` secret to GitHub
3. Modify workflow to publish to TestPyPI:
   ```yaml
   - name: Publish to TestPyPI
     env:
       TWINE_USERNAME: __token__
       TWINE_PASSWORD: ${{ secrets.TEST_PYPI_TOKEN }}
     run: |
       twine upload --repository-url https://test.pypi.org/legacy/ dist/*
   ```

## Troubleshooting

### Build warnings about license
✅ **Fixed** - Updated `pyproject.toml` to use SPDX license expression (`license = "MIT"`) instead of deprecated TOML table format.

### Tests fail in CI
- Check that all dependencies are listed in `pyproject.toml`
- Verify tests pass locally: `pytest tests/unit/ -v`
- Check GitHub Actions logs for specific errors

### Upload fails
- Verify `PYPI_API_TOKEN` secret is set correctly
- Ensure you haven't already published this version
- Check that the token has permissions for the `ia_modules` project

### Version already exists
- Bump version in `pyproject.toml`
- PyPI doesn't allow re-uploading the same version

## Version Numbering

Follow semantic versioning:
- **Major** (1.0.0): Breaking changes
- **Minor** (0.1.0): New features, backward compatible
- **Patch** (0.0.1): Bug fixes, backward compatible

Current version: **0.0.4**

## Post-Publish Verification

After publishing:

```pwsh
# Install from PyPI
pip install ia_modules

# Verify installation
python -c "import ia_modules; print(ia_modules.__version__)"

# Check on PyPI
# Visit: https://pypi.org/project/ia-modules/
```

## Rollback

PyPI doesn't allow deleting versions. To fix a bad release:
1. Increment version (e.g., 0.0.5 → 0.0.6)
2. Publish fixed version
3. Optionally yank the bad version on PyPI (it will still be available but marked as yanked)

## Additional Resources

- [Python Packaging Guide](https://packaging.python.org/)
- [PyPI Help](https://pypi.org/help/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [GitHub Actions for Python](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python)
