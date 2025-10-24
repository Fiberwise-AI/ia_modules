# LLM API Keys Setup for Integration Tests

## Overview

The LLM provider integration tests verify that the `LLMProviderService` works correctly with real API providers. These tests are **optional** and will skip automatically if API keys are not configured.

## Quick Start

### 1. Set Environment Variables

You have two options to configure API keys:

#### Option A: Using Environment Variables (Recommended for CI/CD)

Set these environment variables in your shell:

```bash
# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-..."
$env:ANTHROPIC_API_KEY="sk-ant-..."
$env:GOOGLE_API_KEY="AIza..."
$env:OLLAMA_AVAILABLE="true"  # if you have Ollama running locally

# Windows (Command Prompt)
set OPENAI_API_KEY=sk-...
set ANTHROPIC_API_KEY=sk-ant-...
set GOOGLE_API_KEY=AIza...
set OLLAMA_AVAILABLE=true

# Linux/Mac
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AIza..."
export OLLAMA_AVAILABLE="true"
```

#### Option B: Using .env File (Recommended for Local Development)

1. Copy the example file:
   ```bash
   cp tests/.env.example tests/.env
   ```

2. Edit `tests/.env` and uncomment the API keys you want to use:
   ```bash
   # Uncomment and add your keys:
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...
   GOOGLE_API_KEY=AIza...
   OLLAMA_AVAILABLE=true
   ```

3. Install python-dotenv if not already installed:
   ```bash
   pip install python-dotenv
   ```

4. Load environment variables before running tests:
   ```bash
   # The test suite will automatically load from .env if python-dotenv is available
   pytest tests/integration/test_llm_provider_integration.py -v
   ```

### 2. Get API Keys

#### OpenAI
1. Go to https://platform.openai.com/api-keys
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the key (starts with `sk-...`)

#### Anthropic (Claude)
1. Go to https://console.anthropic.com/account/keys
2. Sign in or create an account
3. Click "Create Key"
4. Copy the key (starts with `sk-ant-...`)

#### Google AI (Gemini)
1. Go to https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key (starts with `AIza...`)

#### Ollama (Local)
1. Install Ollama: https://ollama.ai/download
2. Start Ollama service: `ollama serve`
3. Pull a model: `ollama pull llama2`
4. Set `OLLAMA_AVAILABLE=true`

## Running Tests

### Run All LLM Tests
```bash
# Unit tests (no API keys needed)
pytest tests/unit/test_llm_provider_service.py -v

# Integration tests (requires API keys)
pytest tests/integration/test_llm_provider_integration.py -v
```

### Run Specific Provider Tests
```bash
# OpenAI only
pytest tests/integration/test_llm_provider_integration.py::TestOpenAIIntegration -v

# Anthropic only
pytest tests/integration/test_llm_provider_integration.py::TestAnthropicIntegration -v

# Google only
pytest tests/integration/test_llm_provider_integration.py::TestGoogleIntegration -v

# Ollama only
pytest tests/integration/test_llm_provider_integration.py::TestOllamaIntegration -v
```

### View Skipped Tests
```bash
# See which tests are skipped due to missing API keys
pytest tests/integration/test_llm_provider_integration.py -v -rs
```

## Test Behavior

### Without API Keys
- **Unit tests**: ✅ All 42 tests run (no API calls)
- **Integration tests**: ⏭️ 15 tests skip, 2 tests pass (consistency checks)

### With API Keys
- **Unit tests**: ✅ All 42 tests run (no API calls)
- **Integration tests**: ✅ Tests run for configured providers
  - 3 tests per provider (OpenAI, Anthropic, Google)
  - 2 tests for Ollama
  - 2 tests for structured output
  - 4 tests for multi-provider features

## Cost Considerations

The integration tests are designed to minimize API costs:

- **Low token usage**: Each test uses `max_tokens=10-50`
- **Simple prompts**: "Say 'Hello'" or "Count from 1 to 3"
- **Temperature 0**: Deterministic responses where possible
- **Estimated cost per full test run**: < $0.10 USD

### Free Tier Limits
- **OpenAI**: $5 free credit for new accounts
- **Anthropic**: Some free usage on new accounts
- **Google AI**: Generous free tier for Gemini
- **Ollama**: Completely free (runs locally)

## Troubleshooting

### Tests Skip Even With API Keys

1. **Check environment variables are set**:
   ```bash
   # Windows (PowerShell)
   echo $env:OPENAI_API_KEY

   # Linux/Mac
   echo $OPENAI_API_KEY
   ```

2. **Verify .env file is loaded**:
   ```python
   import os
   from dotenv import load_dotenv
   load_dotenv("tests/.env")
   print(os.getenv("OPENAI_API_KEY"))
   ```

3. **Run with verbose output**:
   ```bash
   pytest tests/integration/test_llm_provider_integration.py -v -s
   ```

### Authentication Errors

If you see authentication errors:

1. **Verify API key is valid**: Check the provider's dashboard
2. **Check for extra spaces**: Trim whitespace from keys
3. **Verify account has credits**: Some providers require payment setup

### Import Errors

If you see "package not available" errors:

```bash
# Install all LLM provider packages
pip install openai anthropic google-generativeai aiohttp

# Or install ia_modules with LLM extras
pip install -e ".[llm]"
```

## Security Best Practices

1. **Never commit API keys**: Add `.env` to `.gitignore`
2. **Use separate keys for testing**: Create test-specific API keys
3. **Set spending limits**: Configure billing alerts in provider dashboards
4. **Rotate keys regularly**: Regenerate keys periodically
5. **Use minimal permissions**: If available, use read-only or limited keys

## CI/CD Integration

### GitHub Actions
```yaml
- name: Run LLM Integration Tests
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
    GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
  run: |
    pytest tests/integration/test_llm_provider_integration.py -v
```

### GitLab CI
```yaml
test_llm:
  script:
    - pytest tests/integration/test_llm_provider_integration.py -v
  variables:
    OPENAI_API_KEY: $OPENAI_API_KEY
    ANTHROPIC_API_KEY: $ANTHROPIC_API_KEY
    GOOGLE_API_KEY: $GOOGLE_API_KEY
```

## Optional: Load .env Automatically

To automatically load `.env` files in tests, add to `conftest.py`:

```python
# At the top of tests/conftest.py
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass  # python-dotenv not installed, skip
```

## Summary

- ✅ **Unit tests work without any setup** (42 tests)
- ⏭️ **Integration tests skip gracefully** without API keys
- 🔑 **Add API keys** via environment variables or `.env` file
- 💰 **Low cost** per test run (< $0.10 USD)
- 🔒 **Keep keys secure** and never commit them

For questions or issues, see the main test documentation: [TESTING_README.md](./TESTING_README.md)
