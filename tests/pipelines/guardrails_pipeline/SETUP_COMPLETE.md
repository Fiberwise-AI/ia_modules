# Guardrails Pipeline Setup - Complete

## What Was Done

The existing generic pipeline runner at `tests/pipeline_runner.py` has been enhanced to support LLM service injection from .env file.

### Changes Made

1. **Enhanced pipeline_runner.py**
   - Added automatic .env file loading from `tests/.env`
   - Added `setup_llm_service()` function that:
     - Reads API keys from environment variables
     - Creates and configures LLMProviderService
     - Registers all available providers (OpenAI, Anthropic, Google, Ollama)
     - Uses model names from environment or sensible defaults
   - Updated `run_pipeline_test()` to set up and register LLM service
   - Updated help message to document LLM service setup

2. **Updated Documentation**
   - Updated [guardrails_pipeline/README.md](README.md) to use generic runner
   - Removed references to custom run_pipeline.py
   - Added clear instructions for .env setup
   - Documented how runner works automatically

3. **Added Dependencies**
   - Added `python-dotenv>=1.0.0` to dev dependencies in pyproject.toml

## How to Use

### 1. Set Up Environment

```bash
cd ia_modules/tests
cp .env.example .env
# Edit .env and add your API keys
```

In `.env`, uncomment and add keys:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...

# Optional: Override models
OPENAI_MODEL=gpt-4o
ANTHROPIC_MODEL=claude-sonnet-4-5-20250929
GEMINI_MODEL=gemini-2.5-flash
```

### 2. Run Guardrails Pipeline

```bash
cd ia_modules
python tests/pipeline_runner.py tests/pipelines/guardrails_pipeline/pipeline.json
```

With custom input:

```bash
python tests/pipeline_runner.py tests/pipelines/guardrails_pipeline/pipeline.json \
  --input '{"user_input": "What is machine learning?"}'
```

### 3. Use in Your Own Pipelines

The generic runner now works with ANY pipeline that uses LLM steps:

```bash
# Any pipeline JSON file
python tests/pipeline_runner.py path/to/your/pipeline.json

# With custom input
python tests/pipeline_runner.py path/to/your/pipeline.json \
  --input '{"your": "data"}'

# With output directory
python tests/pipeline_runner.py path/to/your/pipeline.json \
  --output ./results
```

## What the Runner Does Automatically

1. ✅ Loads `.env` file from `tests/.env`
2. ✅ Detects which LLM API keys are available
3. ✅ Creates `LLMProviderService` instance
4. ✅ Registers all available providers
5. ✅ Sets first provider as default
6. ✅ Registers service in `ServiceRegistry` as `'llm_provider'`
7. ✅ Creates `GraphPipelineRunner` with services
8. ✅ Runs your pipeline
9. ✅ Saves results to timestamped directory

## Environment Variables Supported

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `GOOGLE_API_KEY` | Google AI API key | - |
| `OLLAMA_AVAILABLE` | Set to `true` if Ollama is running | `false` |
| `OPENAI_MODEL` | OpenAI model name | `gpt-4o` |
| `ANTHROPIC_MODEL` | Anthropic model name | `claude-sonnet-4-5-20250929` |
| `GEMINI_MODEL` | Google model name | `gemini-2.5-flash` |

## How LLM Steps Access the Service

In your pipeline steps, access the LLM service from ServiceRegistry:

```python
from ia_modules.pipeline.core import Step

class MyLLMStep(Step):
    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Get LLM service from services registry
        llm_service = self.services.get('llm_provider')
        if not llm_service:
            raise RuntimeError("LLMProviderService not registered")

        # Use the service
        response = await llm_service.generate_completion(
            prompt="Your prompt here",
            temperature=0.7,
            max_tokens=500
        )

        data["response"] = response.content
        return data
```

## Files Changed

- ✅ `tests/pipeline_runner.py` - Enhanced with LLM service support
- ✅ `tests/pipelines/guardrails_pipeline/README.md` - Updated documentation
- ✅ `pyproject.toml` - Added python-dotenv dependency

## Testing the Setup

Test the runner help message:

```bash
python tests/pipeline_runner.py
```

You should see:

```
LLM Service:
  The runner automatically loads LLM API keys from tests/.env file
  Copy tests/.env.example to tests/.env and add your API keys:
    OPENAI_API_KEY=sk-...
    ANTHROPIC_API_KEY=sk-ant-...
    GOOGLE_API_KEY=AIza...
```

## No Breaking Changes

The enhancements are backward compatible:
- ✅ Existing pipelines without LLM steps still work
- ✅ If no API keys are set, runner logs info and continues
- ✅ All existing command-line options still work
- ✅ Database support unchanged

## Next Steps

1. **Install dependencies**: `pip install -e ".[dev]"`
2. **Copy .env file**: `cp tests/.env.example tests/.env`
3. **Add API keys**: Edit `tests/.env`
4. **Run the example**: See README.md in this directory

## Success!

The generic pipeline runner now fully supports LLM service injection from .env file. No per-execution API key setup needed. Just set up `.env` once and run any pipeline.
