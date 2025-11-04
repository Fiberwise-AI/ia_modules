"""
LLM Provider Service for Pipeline Integration

A unified service for interacting with multiple LLM providers within the pipeline system.
Based on the Fiberwise LLM Provider Service pattern.
"""

from typing import Dict, Any, Optional, List, Union
import asyncio
import logging
import json
from datetime import datetime
from enum import Enum

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE_OPENAI = "azure_openai"
    OLLAMA = "ollama"


class LLMConfig:
    """Configuration for LLM provider"""
    def __init__(
        self,
        provider: LLMProvider,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ):
        self.provider = provider
        self.api_key = api_key
        self.model = model or self._get_default_model(provider)
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_params = kwargs

    def _get_default_model(self, provider: LLMProvider) -> str:
        """Get default model for each provider"""
        defaults = {
            LLMProvider.OPENAI: "gpt-3.5-turbo",
            LLMProvider.ANTHROPIC: "claude-3-haiku-20240307",
            LLMProvider.GOOGLE: "gemini-2.5-flash",
            LLMProvider.AZURE_OPENAI: "gpt-3.5-turbo",
            LLMProvider.OLLAMA: "llama2"
        }
        return defaults.get(provider, "gpt-3.5-turbo")


class LLMResponse:
    """Standardized LLM response"""
    def __init__(
        self,
        content: str,
        provider: LLMProvider,
        model: str,
        usage: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.content = content
        self.provider = provider
        self.model = model
        self.usage = usage or {}
        self.metadata = metadata or {}
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "content": self.content,
            "provider": self.provider.value,
            "model": self.model,
            "usage": self.usage,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class LLMProviderService:
    """
    Unified LLM Provider Service for Pipeline Integration

    Provides a standardized interface for multiple LLM providers within the pipeline system.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.providers: Dict[str, LLMConfig] = {}
        self.default_provider: Optional[str] = None

    async def cleanup(self):
        """Cleanup the service"""
        self.logger.info("LLM Provider Service cleaned up")

    def register_provider(
        self,
        name: str,
        provider: LLMProvider,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        is_default: bool = False,
        **kwargs
    ) -> None:
        """
        Register an LLM provider configuration

        Args:
            name: Unique name for this provider configuration
            provider: The LLM provider type
            api_key: API key for the provider
            model: Default model to use (can be overridden at call time)
            base_url: Custom base URL (for Azure, Ollama, etc.)
            temperature: Default temperature
            max_tokens: Default max tokens
            is_default: Whether this should be the default provider
            **kwargs: Additional provider-specific parameters
        """
        config = LLMConfig(
            provider=provider,
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        self.providers[name] = config

        if is_default or not self.default_provider:
            self.default_provider = name

        self.logger.info(f"Registered LLM provider: {name} ({provider.value}, model: {config.model})")

    def register_providers_from_dict(self, providers_config: Dict[str, Dict[str, Any]]) -> None:
        """
        Register multiple providers from a structured dictionary

        Args:
            providers_config: Dictionary with provider configurations

        Example:
            llm_providers = {
                "openai_gpt4": {
                    "provider": LLMProvider.OPENAI,
                    "api_key": "...",
                    "model": "gpt-4o",
                    "is_default": True
                },
                "openai_gpt3": {
                    "provider": LLMProvider.OPENAI,
                    "api_key": "...",
                    "model": "gpt-3.5-turbo"
                },
                "anthropic_sonnet": {
                    "provider": LLMProvider.ANTHROPIC,
                    "api_key": "...",
                    "model": "claude-sonnet-4-5-20250929"
                }
            }
            llm_service.register_providers_from_dict(llm_providers)
        """
        for name, config in providers_config.items():
            self.register_provider(name=name, **config)

        self.logger.info(f"Registered {len(providers_config)} providers from config")

    def set_default_provider(self, name: str) -> None:
        """
        Set the default provider

        Args:
            name: Name of the provider to set as default

        Raises:
            ValueError: If provider not found
        """
        if name not in self.providers:
            raise ValueError(f"Provider '{name}' not found. Available: {list(self.providers.keys())}")

        self.default_provider = name
        self.logger.info(f"Set default provider to: {name}")

    def get_provider(self, name: Optional[str] = None) -> Optional[LLMConfig]:
        """Get provider configuration by name or default"""
        if name is None:
            name = self.default_provider
        return self.providers.get(name) if name else None

    async def generate_completion(
        self,
        prompt: str,
        provider_name: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a completion using the specified or default provider

        Args:
            prompt: The input prompt
            provider_name: Name of the provider to use (uses default if None)
            model: Override the model for this specific call
            temperature: Override temperature
            max_tokens: Override max tokens
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse with standardized response data

        Raises:
            ValueError: If provider not found or no default set
        """
        config = self.get_provider(provider_name)
        if not config:
            available = list(self.providers.keys())
            raise ValueError(
                f"Provider '{provider_name}' not found. "
                f"Available providers: {available}. "
                f"Default provider: {self.default_provider or 'None set'}"
            )

        # Override config parameters if provided
        effective_model = model if model is not None else config.model
        effective_temperature = temperature if temperature is not None else config.temperature
        effective_max_tokens = max_tokens if max_tokens is not None else config.max_tokens

        # Create a temporary config with overridden model if needed
        effective_config = config
        if model is not None:
            effective_config = LLMConfig(
                provider=config.provider,
                api_key=config.api_key,
                model=model,
                base_url=config.base_url,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                **config.extra_params
            )

        try:
            if effective_config.provider == LLMProvider.OPENAI:
                return await self._call_openai(effective_config, prompt, effective_temperature, effective_max_tokens, **kwargs)
            elif effective_config.provider == LLMProvider.ANTHROPIC:
                return await self._call_anthropic(effective_config, prompt, effective_temperature, effective_max_tokens, **kwargs)
            elif effective_config.provider == LLMProvider.GOOGLE:
                return await self._call_google(effective_config, prompt, effective_temperature, effective_max_tokens, **kwargs)
            elif effective_config.provider == LLMProvider.OLLAMA:
                return await self._call_ollama(effective_config, prompt, effective_temperature, effective_max_tokens, **kwargs)
            else:
                raise ValueError(f"Provider {effective_config.provider} not implemented")

        except Exception as e:
            self.logger.error(f"LLM call failed for provider {effective_config.provider} (model: {effective_model}): {str(e)}")
            raise

    async def generate_structured_output(
        self,
        prompt: str,
        schema: Dict[str, Any],
        provider_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate structured output using JSON schema

        Args:
            prompt: The input prompt
            schema: JSON schema for the expected output
            provider_name: Name of the provider to use
            **kwargs: Additional parameters

        Returns:
            Parsed structured data
        """
        # Add schema instructions to prompt
        schema_prompt = f"{prompt}\n\nPlease respond with valid JSON that conforms to this schema:\n{json.dumps(schema, indent=2)}"

        response = await self.generate_completion(schema_prompt, provider_name, **kwargs)

        try:
            # Clean the response content - remove markdown code blocks if present
            content = response.content.strip()
            
            # Remove markdown code blocks (```json...``` or ```...```)
            if content.startswith('```'):
                # Find the first newline after ```
                first_newline = content.find('\n')
                if first_newline != -1:
                    # Remove the opening ```json or ``` line
                    content = content[first_newline + 1:]
                
                # Remove the closing ```
                if content.endswith('```'):
                    content = content[:-3]
                    
            # Clean up any remaining whitespace
            content = content.strip()
            
            # Try to parse JSON response
            return json.loads(content)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            # Return a fallback structure
            return {"error": "Failed to parse structured output", "raw_content": response.content}

    async def _call_openai(
        self,
        config: LLMConfig,
        prompt: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> LLMResponse:
        """Call OpenAI API"""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not available. Install with: pip install openai")

        client = openai.AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )

        response = await client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            provider=config.provider,
            model=config.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        )

    async def _call_anthropic(
        self,
        config: LLMConfig,
        prompt: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> LLMResponse:
        """Call Anthropic API"""
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic package not available. Install with: pip install anthropic")

        client = anthropic.AsyncAnthropic(api_key=config.api_key)

        response = await client.messages.create(
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        return LLMResponse(
            content=response.content[0].text,
            provider=config.provider,
            model=config.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        )

    async def _call_google(
        self,
        config: LLMConfig,
        prompt: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> LLMResponse:
        """Call Google Gemini API"""
        if not GOOGLE_AVAILABLE:
            raise ImportError("Google Generative AI package not available. Install with: pip install google-generativeai")

        # Configure the API key
        genai.configure(api_key=config.api_key)

        # Create model instance with safety settings if provided
        safety_settings = config.extra_params.get('safety_settings', {})
        if safety_settings:
            # Convert safety settings to Google format
            formatted_safety = []
            for category, threshold in safety_settings.items():
                formatted_safety.append({
                    "category": getattr(genai.types.HarmCategory, category),
                    "threshold": getattr(genai.types.HarmBlockThreshold, threshold)
                })
            self.logger.info(f"Applying safety settings: {formatted_safety}")
            model = genai.GenerativeModel(config.model, safety_settings=formatted_safety)
        else:
            self.logger.info("No safety settings configured, using defaults")
            model = genai.GenerativeModel(config.model)

        # Configure generation settings
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            **{k: v for k, v in kwargs.items() if k != 'safety_settings'}
        )

        # Generate response
        response = await model.generate_content_async(
            prompt,
            generation_config=generation_config
        )

        # Handle safety filtering and other response issues
        try:
            content = response.text
        except Exception as e:
            # Check finish reason and provide appropriate error
            if response.candidates and len(response.candidates) > 0:
                finish_reason = response.candidates[0].finish_reason
                if finish_reason == 2:  # SAFETY
                    raise Exception("Content was filtered by safety settings. Try rephrasing your prompt.")
                elif finish_reason == 3:  # RECITATION
                    raise Exception("Content was filtered due to recitation concerns.")
                else:
                    raise Exception(f"Content generation failed with finish_reason: {finish_reason}")
            else:
                raise Exception(f"No candidates returned from API: {str(e)}")

        return LLMResponse(
            content=content,
            provider=config.provider,
            model=config.model,
            usage={
                "prompt_token_count": response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
                "candidates_token_count": response.usage_metadata.candidates_token_count if response.usage_metadata else 0,
                "total_token_count": response.usage_metadata.total_token_count if response.usage_metadata else 0,
            },
            metadata={
                "finish_reason": response.candidates[0].finish_reason if response.candidates else None,
                "safety_ratings": [rating.category.name + ":" + rating.probability.name
                                 for rating in (response.candidates[0].safety_ratings if response.candidates else [])]
            }
        )

    async def _call_ollama(
        self,
        config: LLMConfig,
        prompt: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> LLMResponse:
        """Call Ollama API"""
        import aiohttp

        url = config.base_url or "http://localhost:11434"

        async with aiohttp.ClientSession() as session:
            payload = {
                "model": config.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    **kwargs
                }
            }

            async with session.post(f"{url}/api/generate", json=payload) as response:
                result = await response.json()

                return LLMResponse(
                    content=result.get("response", ""),
                    provider=config.provider,
                    model=config.model,
                    usage={
                        "prompt_eval_count": result.get("prompt_eval_count", 0),
                        "eval_count": result.get("eval_count", 0),
                    },
                    metadata=result
                )

    def list_providers(self) -> List[Dict[str, Any]]:
        """List all registered providers"""
        return [
            {
                "name": name,
                "provider": config.provider.value,
                "model": config.model,
                "is_default": name == self.default_provider
            }
            for name, config in self.providers.items()
        ]