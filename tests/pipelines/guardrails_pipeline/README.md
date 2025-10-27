# Guardrails Pipeline Example

Complete working example showing how to add safety guardrails to an LLM pipeline using the existing ia_modules infrastructure.

## What This Demonstrates

This pipeline uses **real, production-ready components** from ia_modules:

✅ **LLMProviderService** - Multi-provider LLM service (OpenAI, Anthropic, Google, Ollama)
✅ **ServiceRegistry** - Dependency injection for pipeline steps
✅ **GuardrailsEngine** - Input/output safety validation
✅ **GraphPipelineRunner** - Pipeline execution engine

**No mocks. No fake components. Real production code.**

## Pipeline Architecture

```
User Input
    ↓
[InputGuardrailStep]
 └─ Checks for jailbreaks, toxicity, PII
 └─ Redacts sensitive information
    ↓
[LLMStep]
 └─ Uses LLMProviderService
 └─ Supports OpenAI, Anthropic, Google, Ollama
    ↓
[OutputGuardrailStep]
 └─ Validates LLM response
 └─ Adds disclaimers where needed
 └─ Enforces length limits
    ↓
Safe Response
```

## Quick Start

### 1. Set Up API Keys

Copy the example .env file and add your API keys:

```bash
cd ia_modules/tests
cp .env.example .env
# Edit .env and add your API keys
```

In `tests/.env`, uncomment and add your keys:

```bash
# Choose one or more providers:
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...

# Optional: Model overrides
OPENAI_MODEL=gpt-4o
ANTHROPIC_MODEL=claude-sonnet-4-5-20250929
GEMINI_MODEL=gemini-2.5-flash

# Local Ollama (no API key needed)
# OLLAMA_AVAILABLE=true
```

### 2. Run the Pipeline

Use the generic pipeline runner from the tests directory:

```bash
cd ia_modules
python tests/pipeline_runner.py tests/pipelines/guardrails_pipeline/pipeline.json
```

Or with custom input:

```bash
python tests/pipeline_runner.py tests/pipelines/guardrails_pipeline/pipeline.json \
  --input '{"user_input": "What is machine learning?"}'
```

That's it! The runner will:
1. Automatically load API keys from tests/.env
2. Register all available LLM providers
3. Set up the ServiceRegistry with LLMProviderService
4. Run the pipeline with guardrails
5. Save results to timestamped output directory

## How It Works

### Generic Pipeline Runner

The pipeline uses the generic runner at `tests/pipeline_runner.py`. This runner:

1. **Loads .env file** - Automatically loads `tests/.env` with your API keys
2. **Sets up LLM service** - Creates `LLMProviderService` and registers all providers with keys
3. **Creates ServiceRegistry** - Registers LLM service for pipeline steps to use
4. **Runs pipeline** - Executes the pipeline with full guardrails

The runner handles all the setup automatically. You just provide the pipeline JSON.

### Manual Setup (if you need it)

If you want to integrate this into your own code:

```python
from ia_modules.pipeline.graph_pipeline_runner import GraphPipelineRunner
from ia_modules.pipeline.services import ServiceRegistry
from ia_modules.pipeline.llm_provider_service import LLMProviderService, LLMProvider
import os

# 1. Create services
services = ServiceRegistry()

# 2. Create and configure LLM service
llm_service = LLMProviderService()

# Register your provider(s)
if os.getenv("OPENAI_API_KEY"):
    llm_service.register_provider(
        name="openai",
        provider=LLMProvider.OPENAI,
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o",
        is_default=True
    )

# 3. Register services
services.register('llm_provider', llm_service)

# 4. Create runner with services
runner = GraphPipelineRunner(services=services)

# 5. Load and run pipeline
with open("pipeline.json") as f:
    config = json.load(f)

result = await runner.run_pipeline_from_json(config, {
    "user_input": "What is machine learning?"
})
```

### Pipeline Configuration (pipeline.json)

The pipeline has 3 steps:

1. **Input Guardrails** - `InputGuardrailStep`
   - Blocks jailbreak attempts
   - Blocks toxic input
   - Redacts PII (emails, SSNs, etc.)

2. **LLM Processing** - `LLMStep`
   - Uses `LLMProviderService` from ServiceRegistry
   - Calls whichever provider you configured
   - Returns LLM response

3. **Output Guardrails** - `OutputGuardrailStep`
   - Blocks toxic outputs
   - Adds disclaimers to medical/legal/financial advice
   - Enforces maximum response length

## File Structure

```
tests/
├── pipeline_runner.py                    # Generic runner (loads .env, sets up LLM)
├── .env.example                          # Environment template
├── .env                                  # Your API keys (git-ignored)
└── pipelines/
    └── guardrails_pipeline/
        ├── pipeline.json                 # Pipeline configuration
        ├── README.md                     # This file
        └── steps/
            ├── __init__.py
            └── llm_step.py              # LLM step using LLMProviderService
```

## Configuration Options

### LLM Step Configuration

In `pipeline.json`:

```json
{
    "id": "llm_processing",
    "step_class": "LLMStep",
    "module": "tests.pipelines.guardrails_pipeline.steps.llm_step",
    "config": {
        "provider_name": "openai",        // Optional: specific provider
        "temperature": 0.7,               // LLM temperature
        "max_tokens": 500,                // Max response length
        "system_prompt": "You are...",   // Optional system prompt
        "input_field": "user_input",      // Input field name
        "output_field": "llm_response"    // Output field name
    }
}
```

### Guardrails Configuration

**Input Guardrails:**
```json
{
    "id": "input_guardrails",
    "step_class": "InputGuardrailStep",
    "config": {
        "content_field": "user_input",
        "fail_on_block": true,     // Raise error if blocked
        "redact_pii": true         // Redact PII instead of blocking
    }
}
```

**Output Guardrails:**
```json
{
    "id": "output_guardrails",
    "step_class": "OutputGuardrailStep",
    "config": {
        "content_field": "llm_response",
        "fail_on_block": true,
        "max_length": 500          // Max response length
    }
}
```

## Using in Your Own Pipelines

### Option 1: Copy This Example

Copy the entire `guardrails_pipeline` directory and modify:
- Change `pipeline.json` to add your own steps
- Modify `run_pipeline.py` to use your data
- Keep the guardrails steps as-is

### Option 2: Add to Existing Pipeline

Add guardrail steps to your existing pipeline JSON:

```json
{
    "steps": [
        {
            "id": "input_guardrails",
            "step_class": "InputGuardrailStep",
            "module": "ia_modules.guardrails.pipeline_steps"
        },
        {
            "id": "your_existing_step",
            "step_class": "YourStep",
            "module": "your.module"
        },
        {
            "id": "output_guardrails",
            "step_class": "OutputGuardrailStep",
            "module": "ia_modules.guardrails.pipeline_steps"
        }
    ]
}
```

Then in your runner:

```python
# Make sure LLMProviderService is registered if you use LLMStep
services = ServiceRegistry()
llm_service = LLMProviderService()
llm_service.register_provider(...)
services.register('llm_provider', llm_service)

runner = GraphPipelineRunner(services=services)
```

## What Gets Blocked/Modified

### Input Guardrails Block:
- ❌ "Ignore all previous instructions..."
- ❌ "You are now in developer mode..."
- ❌ Toxic/harmful language
- ✏️ PII gets redacted: "My email is [EMAIL_REDACTED]"

### Output Guardrails Block:
- ❌ Toxic LLM responses
- ❌ Responses over max length

### Output Guardrails Modify:
- ✏️ Add disclaimers to medical/legal/financial advice

## Advanced Usage

See [PIPELINE_INTEGRATION.md](../../../guardrails/PIPELINE_INTEGRATION.md) for:
- Custom guardrails configuration
- RAG pipelines with retrieval guardrails
- Execution guardrails for code/tool safety
- Loading guardrails from external config files

## Troubleshooting

**"LLMProviderService not registered"**
- Make sure you create and register the LLM service in ServiceRegistry before running

**"Provider 'X' not found"**
- Check that you registered the provider with `llm_service.register_provider()`

**"No API key"**
- Set environment variable for your provider
- OR use Ollama (local, no key needed)

**Pipeline fails with guardrails error**
- This is expected! Guardrails are blocking unsafe content
- Check the error message to see what was blocked

## Production Deployment

This is production-ready code. For deployment:

1. **Store API keys securely** (env vars, secrets manager)
2. **Configure appropriate providers** for your use case
3. **Adjust guardrails thresholds** based on your risk tolerance
4. **Monitor guardrails trigger rates** to tune sensitivity
5. **Log blocked requests** for security monitoring

## No Mocks, Just Real Code

Every component in this pipeline is production infrastructure:
- `LLMProviderService` - Real LLM calls
- `ServiceRegistry` - Real dependency injection
- `GuardrailsEngine` - Real safety validation
- `GraphPipelineRunner` - Real pipeline execution

This is not a demo. This is the actual system.
