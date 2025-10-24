# GitHub Actions Setup for LLM Integration Tests

## Overview

This guide shows you how to configure GitHub Secrets so that LLM integration tests run in GitHub Actions CI/CD without skipping.

## Quick Setup

### 1. Add Secrets to GitHub Repository

1. Go to your repository on GitHub
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add each of the following secrets:

| Secret Name | Description | Where to Get |
|------------|-------------|--------------|
| `OPENAI_API_KEY` | OpenAI API key | https://platform.openai.com/api-keys |
| `ANTHROPIC_API_KEY` | Anthropic/Claude API key | https://console.anthropic.com/account/keys |
| `GOOGLE_API_KEY` | Google AI/Gemini API key | https://makersuite.google.com/app/apikey |
| `OLLAMA_AVAILABLE` | Set to `"true"` if using Ollama | N/A (usually `"false"` for CI) |

### 2. Create or Update GitHub Actions Workflow

Create `.github/workflows/test.yml`:

```yaml
name: Run Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest

    # Optional: Use Docker services for database tests
    services:
      postgres:
        image: postgres:15-alpine
        env:
          POSTGRES_USER: testuser
          POSTGRES_PASSWORD: testpass
          POSTGRES_DB: ia_modules_test
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.13'
          cache: 'pip'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev,test]"

      - name: Run unit tests
        run: |
          pytest ia_modules/tests/unit/ -v --cov=ia_modules --cov-report=xml

      - name: Run integration tests (with LLM API keys)
        env:
          # Database connection strings
          TEST_POSTGRESQL_URL: postgresql://testuser:testpass@localhost:5432/ia_modules_test
          TEST_REDIS_URL: redis://localhost:6379/0

          # LLM API keys from GitHub Secrets
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
          OLLAMA_AVAILABLE: ${{ secrets.OLLAMA_AVAILABLE }}
        run: |
          pytest ia_modules/tests/integration/ -v --cov=ia_modules --cov-report=xml --cov-append

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
          fail_ci_if_error: true
```

### 3. Minimal Example (Tests Only)

If you just want to run tests without coverage:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.13'

      - name: Install dependencies
        run: |
          pip install -e ".[test]"

      - name: Run tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
        run: |
          pytest ia_modules/tests/ -v
```

## Advanced Configuration

### Cost Control

To avoid unexpected API costs, you can:

1. **Limit test runs**: Only run on specific branches or with manual approval
2. **Use separate test API keys**: Create dedicated API keys with spending limits
3. **Cache results**: Cache test results to avoid re-running expensive tests

```yaml
on:
  push:
    branches: [ main ]  # Only run on main branch
  workflow_dispatch:    # Allow manual trigger
```

### Conditional LLM Tests

Run LLM tests only when secrets are available:

```yaml
- name: Check if LLM secrets available
  id: check_secrets
  run: |
    if [ -n "${{ secrets.OPENAI_API_KEY }}" ]; then
      echo "has_llm_keys=true" >> $GITHUB_OUTPUT
    else
      echo "has_llm_keys=false" >> $GITHUB_OUTPUT
    fi

- name: Run LLM integration tests
  if: steps.check_secrets.outputs.has_llm_keys == 'true'
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
    GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
  run: |
    pytest ia_modules/tests/integration/test_llm_provider_integration.py -v
```

### Matrix Testing (Multiple Python Versions)

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12', '3.13']

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Run tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: pytest ia_modules/tests/ -v
```

## Security Best Practices

### 1. Never Commit Secrets

✅ **DO:**
- Use GitHub Secrets for all API keys
- Add `.env` to `.gitignore`
- Use environment variables in CI

❌ **DON'T:**
- Hardcode API keys in workflow files
- Commit `.env` files
- Log API keys in test output

### 2. Use Dependabot for Secrets Scanning

Add `.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
```

### 3. Limit Secret Access

In organization settings:
- Restrict which repositories can access organization secrets
- Use environment-specific secrets for production vs. testing
- Regularly rotate API keys

### 4. Monitor Usage

- Set up billing alerts in provider dashboards
- Review GitHub Actions usage regularly
- Use API key usage dashboards

## Troubleshooting

### Tests Still Skipping

**Check 1**: Verify secrets are set
```bash
# In your workflow, add this debug step:
- name: Debug secrets
  run: |
    echo "OPENAI_API_KEY: $([ -n "$OPENAI_API_KEY" ] && echo 'SET' || echo 'NOT SET')"
    echo "ANTHROPIC_API_KEY: $([ -n "$ANTHROPIC_API_KEY" ] && echo 'SET' || echo 'NOT SET')"
```

**Check 2**: Verify test markers
```bash
# Run locally to see which tests need API keys
pytest ia_modules/tests/integration/test_llm_provider_integration.py -v -m "not skip"
```

### API Rate Limits

If you hit rate limits:

1. **Reduce test frequency**:
   ```yaml
   on:
     push:
       branches: [ main ]  # Only main, not all branches
   ```

2. **Use pytest-xdist for parallel execution**:
   ```bash
   pip install pytest-xdist
   pytest -n auto  # Run tests in parallel
   ```

3. **Cache test results**:
   ```yaml
   - uses: actions/cache@v3
     with:
       path: .pytest_cache
       key: pytest-${{ hashFiles('ia_modules/**/*.py') }}
   ```

### Failed Tests Due to API Changes

If tests fail due to provider API changes:

1. Check provider status pages
2. Review provider changelog
3. Update `ia_modules/pipeline/llm_provider_service.py` if needed
4. Pin provider package versions in `requirements.txt`

## Example: Full CI/CD Pipeline

Here's a complete example with all best practices:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 30

    services:
      postgres:
        image: postgres:15-alpine
        env:
          POSTGRES_USER: testuser
          POSTGRES_PASSWORD: testpass
          POSTGRES_DB: ia_modules_test
        ports:
          - 5432:5432
        options: --health-cmd pg_isready --health-interval 10s --health-timeout 5s --health-retries 5

      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
        options: --health-cmd "redis-cli ping" --health-interval 10s --health-timeout 5s --health-retries 5

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python 3.13
        uses: actions/setup-python@v5
        with:
          python-version: '3.13'
          cache: 'pip'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev,test]"

      - name: Lint with ruff
        run: |
          pip install ruff
          ruff check ia_modules/ --output-format=github

      - name: Type check with mypy
        run: |
          pip install mypy
          mypy ia_modules/ --ignore-missing-imports

      - name: Run unit tests
        run: |
          pytest ia_modules/tests/unit/ -v --cov=ia_modules --cov-report=xml --cov-report=term

      - name: Run integration tests
        env:
          TEST_POSTGRESQL_URL: postgresql://testuser:testpass@localhost:5432/ia_modules_test
          TEST_REDIS_URL: redis://localhost:6379/0
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
        run: |
          pytest ia_modules/tests/integration/ -v --cov=ia_modules --cov-report=xml --cov-append --cov-report=term

      - name: Upload coverage reports
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
          flags: unittests
          name: codecov-umbrella
          fail_ci_if_error: false

      - name: Build package
        run: |
          pip install build
          python -m build

      - name: Test package installation
        run: |
          pip install dist/*.whl
          python -c "import ia_modules; print(ia_modules.__version__)"
```

## Summary

1. **Add secrets** to GitHub repository settings
2. **Reference secrets** in workflow with `${{ secrets.SECRET_NAME }}`
3. **Set environment variables** in test steps
4. **Monitor costs** with provider dashboards
5. **Test locally** first to verify setup

For more details, see:
- [LLM API Keys Setup Guide](./LLM_API_KEYS_SETUP.md)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [GitHub Secrets Documentation](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
