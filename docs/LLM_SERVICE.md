# LLM Provider Service

**Clean LiteLLM wrapper for ia_modules**

## Overview

The LLM Provider Service is a simple wrapper around [LiteLLM](https://docs.litellm.ai/) that:
- Manages API keys by provider name
- Passes all parameters directly to LiteLLM
- Adds automatic cost tracking
- Supports 100+ LLM providers

## Quick Start

```python
from ia_modules.pipeline.llm_provider_service import LLMProviderService
import os

# Initialize service
service = LLMProviderService()

# Register providers (do this once at startup)
service.register_provider(
    name="openai",
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    is_default=True
)

service.register_provider(
    name="anthropic",
    model="claude-sonnet-4-5-20250929",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# Use registered provider
response = await service.generate_completion(
    messages=[{"role": "user", "content": "Hello!"}],
    provider_name="openai",
    temperature=0.7,
    max_tokens=100
)

print(response["content"])
print(f"Cost: ${response['usage']['cost_usd']:.4f}")
```

## Features

### All LiteLLM Parameters Supported

Since this is a direct passthrough to LiteLLM, all parameters work:

```python
response = await service.generate_completion(
    messages=[{"role": "user", "content": "Hello"}],
    model="gpt-4o",
    temperature=0.8,
    max_tokens=500,
    top_p=0.9,
    frequency_penalty=0.5,
    presence_penalty=0.2,
    stream=True,
    tools=[...],
    response_format={"type": "json_object"},
    seed=42
)
```

### 100+ Providers Supported

Use any LiteLLM-supported provider:

**OpenAI:**
```python
model="gpt-4o"
model="gpt-4o-mini"
model="o1-preview"
```

**Anthropic:**
```python
model="claude-sonnet-4-5-20250929"
model="claude-3-5-haiku-20241022"
```

**Google:**
```python
model="gemini/gemini-2.0-flash-exp"
model="gemini/gemini-pro"
```

**Local (Ollama):**
```python
model="ollama/llama3.2"
model="ollama/qwen2.5-coder"
```

**AWS Bedrock:**
```python
model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
```

**And many more:** Cohere, Mistral, Groq, DeepSeek, Together AI, Replicate, etc.

See [LiteLLM docs](https://docs.litellm.ai/docs/providers) for complete list.

## API Reference

### `register_provider(name, model, api_key=None, base_url=None, is_default=False, **kwargs)`

Register a provider configuration.

**Parameters:**
- `name` (str): Friendly name for this provider
- `model` (str): Default LiteLLM model string
- `api_key` (str, optional): API key (reads from environment if not provided)
- `base_url` (str, optional): Custom base URL (for Ollama, Azure, etc.)
- `is_default` (bool): Set as default provider
- `**kwargs`: Additional provider-specific parameters

**Example:**
```python
service.register_provider(
    name="my_provider",
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    is_default=True
)
```

### `list_providers() -> List[Dict]`

List all registered providers.

**Returns:** List of dicts with `name`, `model`, `is_default`

**Example:**
```python
for provider in service.list_providers():
    print(f"{provider['name']}: {provider['model']}")
```

### `get_provider(name) -> Optional[Dict]`

Get provider configuration by name.

**Parameters:**
- `name` (str): Provider name

**Returns:** Provider config dict or None

### `generate_completion(messages, provider_name=None, **litellm_params) -> Dict`

Generate completion using LiteLLM.

**Parameters:**
- `messages` (List[Dict]): LiteLLM messages format
- `provider_name` (str, optional): Use registered provider
- `**litellm_params`: Any LiteLLM parameter (model, temperature, stream, etc.)

**Returns:** Dict with:
- `content` (str): Response text
- `model` (str): Model used
- `usage` (Dict): Token counts and `cost_usd`
- `metadata` (Dict): Additional info

**Examples:**

Using registered provider:
```python
response = await service.generate_completion(
    messages=[{"role": "user", "content": "Hello"}],
    provider_name="openai"
)
```

Direct model (bypass registration):
```python
response = await service.generate_completion(
    messages=[{"role": "user", "content": "Hello"}],
    model="gpt-3.5-turbo"
)
```

With all LiteLLM features:
```python
response = await service.generate_completion(
    messages=[
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"}
    ],
    model="gpt-4o",
    temperature=0.8,
    max_tokens=500,
    stream=True,
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {...}
        }
    }]
)
```

## Environment Variables

The service automatically reads API keys from environment:

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Google
GEMINI_API_KEY=...

# AWS Bedrock
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION_NAME=us-east-1

# Cohere
COHERE_API_KEY=...

# Local Ollama (no key needed)
# Just run: ollama pull llama3.2
```

## Cost Tracking

Every response includes automatic cost calculation:

```python
response = await service.generate_completion(...)

print(f"Prompt tokens: {response['usage']['prompt_tokens']}")
print(f"Completion tokens: {response['usage']['completion_tokens']}")
print(f"Total tokens: {response['usage']['total_tokens']}")
print(f"Cost: ${response['usage']['cost_usd']:.4f}")
```

## Error Handling

```python
try:
    response = await service.generate_completion(
        messages=[{"role": "user", "content": "Hello"}],
        provider_name="nonexistent"
    )
except ValueError as e:
    print(f"Provider not found: {e}")
except RuntimeError as e:
    print(f"LLM call failed: {e}")
```

## Migration from Old Service

The old service had:
- `LLMProvider` enum
- `LLMConfig` class
- `LLMResponse` class
- `prompt` parameter

The new service:
- No enums or classes
- Direct `messages` parameter (LiteLLM standard)
- Returns plain dicts
- All LiteLLM features work

**Old:**
```python
response = await service.generate_completion(
    prompt="Hello",
    provider_name="openai_gpt4"
)
content = response.content  # LLMResponse object
```

**New:**
```python
response = await service.generate_completion(
    messages=[{"role": "user", "content": "Hello"}],
    provider_name="openai"
)
content = response["content"]  # Plain dict
```

## See Also

- [LiteLLM Documentation](https://docs.litellm.ai/)
- [Supported Providers](https://docs.litellm.ai/docs/providers)
- [ia_modules Pipeline Guide](../README.md)
