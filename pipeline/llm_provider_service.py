"""
LLM Provider Service - Clean LiteLLM Wrapper

Simple service that:
- Registers providers with their API keys (so app logic doesn't need to track them)
- Allows selecting provider/model at runtime
- Just calls LiteLLM directly - no complexity

Usage:
    service = LLMProviderService()

    # Register providers once (usually at startup)
    service.register_provider("openai", model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
    service.register_provider("anthropic", model="claude-sonnet-4-5-20250929", api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Use anywhere - just specify which one
    response = await service.generate_completion(
        prompt="Hello!",
        provider_name="openai"  # or "anthropic"
    )
"""

from typing import Dict, Any, Optional, List, Union
import base64
import logging
import tempfile
from litellm import acompletion, completion_cost
import litellm


class LLMProviderService:
    """
    Simple LiteLLM wrapper with provider registration.

    Register providers with their API keys once, then use them by name.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._providers: Dict[str, Dict[str, Any]] = {}
        self._default_provider: Optional[str] = None
        self.logger.info("LLM Provider Service initialized")

    def register_provider(
        self,
        name: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        is_default: bool = False,
        **kwargs
    ) -> None:
        """
        Register a provider configuration.

        Args:
            name: Friendly name (e.g., "openai", "anthropic", "my_custom")
            model: LiteLLM model string (e.g., "gpt-4o", "claude-sonnet-4-5-20250929")
            api_key: API key (optional if set in environment)
            base_url: Custom base URL (for Ollama, Azure, etc.)
            is_default: Use this provider when none specified
            **kwargs: Additional LiteLLM parameters
        """
        self._providers[name] = {
            "model": model,
            "api_key": api_key,
            "base_url": base_url,
            **kwargs
        }

        if is_default or not self._default_provider:
            self._default_provider = name

        self.logger.info(f"Registered provider '{name}' with model '{model}'")

    def list_providers(self) -> List[Dict[str, Any]]:
        """List all registered providers."""
        return [
            {
                "name": name,
                "model": config["model"],
                "is_default": name == self._default_provider
            }
            for name, config in self._providers.items()
        ]

    def get_provider(self, name: str) -> Optional[Dict[str, Any]]:
        """Get provider config by name."""
        return self._providers.get(name)

    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        provider_name: Optional[str] = None,
        **litellm_params
    ) -> Dict[str, Any]:
        """
        Generate completion - direct passthrough to LiteLLM.

        Args:
            messages: LiteLLM messages format [{"role": "user", "content": "..."}]
            provider_name: Name of registered provider (uses default if None)
            **litellm_params: Any LiteLLM parameter (model, temperature, stream, tools, etc.)

        Returns:
            Dict with: content, model, usage (with cost_usd), metadata

        Examples:
            # Use registered provider
            response = await service.generate_completion(
                messages=[{"role": "user", "content": "Hello"}],
                provider_name="openai"
            )

            # Direct model
            response = await service.generate_completion(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-4o"
            )

            # With streaming
            response = await service.generate_completion(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-4o",
                stream=True
            )
        """
        # Start with litellm_params (contains model, temperature, etc.)
        call_params = litellm_params.copy()

        # If provider_name given, merge provider config
        if provider_name or (not call_params.get("model")):
            if provider_name is None:
                provider_name = self._default_provider

            if not provider_name:
                raise ValueError("No provider specified and no default set")

            provider_config = self._providers.get(provider_name)
            if not provider_config:
                available = list(self._providers.keys())
                raise ValueError(f"Provider '{provider_name}' not found. Available: {available}")

            # Merge provider config (but allow litellm_params to override)
            for key, value in provider_config.items():
                if key not in call_params and value is not None:
                    call_params[key] = value

        try:
            # Call LiteLLM with messages + all params
            response = await acompletion(
                messages=messages,
                **call_params
            )

            # Extract data
            content = response.choices[0].message.content
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

            # Calculate cost
            try:
                usage["cost_usd"] = completion_cost(completion_response=response)
            except Exception as e:
                self.logger.warning(f"Could not calculate cost: {e}")
                usage["cost_usd"] = 0.0

            return {
                "content": content,
                "model": response.model,
                "usage": usage,
                "metadata": {
                    "finish_reason": response.choices[0].finish_reason,
                    "provider_name": provider_name,
                }
            }

        except Exception as e:
            self.logger.error(f"LiteLLM failed: {e}")
            raise RuntimeError(f"LLM completion failed: {str(e)}")

    async def generate_vision(
        self,
        image: Union[bytes, str],
        prompt: str = "Describe this image",
        provider_name: Optional[str] = None,
        **litellm_params
    ) -> str:
        """
        Process image with vision model.

        Args:
            image: Image bytes or file path
            prompt: Prompt for image analysis
            provider_name: Name of registered provider
            **litellm_params: Additional LiteLLM parameters

        Returns:
            Model's analysis of the image
        """
        # Load image
        if isinstance(image, bytes):
            image_data = image
        else:
            with open(image, 'rb') as f:
                image_data = f.read()

        base64_image = base64.b64encode(image_data).decode('utf-8')

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]

        result = await self.generate_completion(
            messages=messages,
            provider_name=provider_name,
            **litellm_params
        )
        return result["content"]

    async def transcribe(
        self,
        audio: Union[bytes, str],
        model: str = "whisper-1",
        language: Optional[str] = None
    ) -> str:
        """
        Transcribe audio to text (speech-to-text).

        Args:
            audio: Audio bytes or file path
            model: Whisper model to use
            language: Optional language code

        Returns:
            Transcribed text
        """
        # Handle bytes input
        if isinstance(audio, bytes):
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                f.write(audio)
                audio_path = f.name
        else:
            audio_path = audio

        response = await litellm.atranscription(
            model=model,
            file=open(audio_path, 'rb'),
            language=language
        )

        return response.text

    async def synthesize_speech(
        self,
        text: str,
        voice: str = "alloy",
        model: str = "tts-1",
        output_format: str = "mp3"
    ) -> bytes:
        """
        Convert text to speech (text-to-speech).

        Args:
            text: Text to synthesize
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
            model: TTS model
            output_format: Output format (mp3, opus, aac, flac)

        Returns:
            Audio bytes
        """
        response = await litellm.aspeech(
            model=model,
            voice=voice,
            input=text,
            response_format=output_format
        )

        return response.content


# Example usage
if __name__ == "__main__":
    import asyncio
    import os

    async def test():
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

        # List providers
        print("Registered providers:")
        for p in service.list_providers():
            print(f"  {p['name']}: {p['model']} (default: {p['is_default']})")

        # Use default provider
        print("\nUsing default provider:")
        response = await service.generate_completion(
            messages=[{"role": "user", "content": "Say hello in one sentence"}],
            max_tokens=50
        )
        print(f"  {response['content']}")
        print(f"  Cost: ${response['usage']['cost_usd']:.4f}")

        # Use specific provider
        print("\nUsing specific provider:")
        response = await service.generate_completion(
            messages=[{"role": "user", "content": "Say hello in one sentence"}],
            provider_name="anthropic",
            max_tokens=50
        )
        print(f"  {response['content']}")
        print(f"  Cost: ${response['usage']['cost_usd']:.4f}")

        # Direct model (bypass providers)
        print("\nDirect model:")
        response = await service.generate_completion(
            messages=[{"role": "user", "content": "Say hello in one sentence"}],
            model="gpt-3.5-turbo",
            max_tokens=50
        )
        print(f"  {response['content']}")

        # All LiteLLM features work
        print("\nWith streaming:")
        response = await service.generate_completion(
            messages=[{"role": "user", "content": "Count to 5"}],
            model="gpt-3.5-turbo",
            stream=True,
            temperature=0.8
        )
        print(f"  Stream: {response.get('stream', 'N/A')}")

    asyncio.run(test())
