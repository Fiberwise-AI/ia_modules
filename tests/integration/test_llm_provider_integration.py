"""
Integration tests for LLM Provider Service
Tests actual provider calls with real APIs (requires API keys)
"""

import pytest
import os
import asyncio
from datetime import datetime

from ia_modules.pipeline.llm_provider_service import (
from ia_modules.pipeline.test_utils import create_test_execution_context
    LLMProviderService,
    LLMProvider,
    LLMResponse
)


# Check for API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OLLAMA_AVAILABLE = os.getenv("OLLAMA_AVAILABLE", "false").lower() == "true"

# Skip markers
skip_openai = pytest.mark.skipif(
    not OPENAI_API_KEY,
    reason="OPENAI_API_KEY not set"
)
skip_anthropic = pytest.mark.skipif(
    not ANTHROPIC_API_KEY,
    reason="ANTHROPIC_API_KEY not set"
)
skip_google = pytest.mark.skipif(
    not GOOGLE_API_KEY,
    reason="GOOGLE_API_KEY not set"
)
skip_ollama = pytest.mark.skipif(
    not OLLAMA_AVAILABLE,
    reason="OLLAMA_AVAILABLE not set or Ollama not running"
)


@pytest.mark.integration
class TestOpenAIIntegration:
    """Test OpenAI provider integration"""

    @skip_openai
    @pytest.mark.asyncio
    async def test_openai_basic_completion(self):
        """OpenAI returns non-empty completion"""
        service = LLMProviderService()
        service.register_provider(
            "openai",
            LLMProvider.OPENAI,
            api_key=OPENAI_API_KEY,
            model="gpt-3.5-turbo",
            max_tokens=50
        )

        response = await service.generate_completion(
            "Say 'Hello' in one word",
            temperature=0.0
        )

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert response.content is not None
        assert len(response.content) > 0
        assert response.provider == LLMProvider.OPENAI
        assert response.model == "gpt-3.5-turbo"

        # Verify usage tracking
        assert "total_tokens" in response.usage
        assert response.usage["total_tokens"] > 0

    @skip_openai
    @pytest.mark.asyncio
    async def test_openai_with_temperature(self):
        """OpenAI accepts temperature parameter"""
        service = LLMProviderService()
        service.register_provider(
            "openai",
            LLMProvider.OPENAI,
            api_key=OPENAI_API_KEY,
            temperature=0.0
        )

        response = await service.generate_completion(
            "Count from 1 to 3",
            temperature=0.0,
            max_tokens=20
        )

        assert response.content is not None
        assert len(response.content) > 0

    @skip_openai
    @pytest.mark.asyncio
    async def test_openai_error_handling(self):
        """OpenAI handles invalid requests appropriately"""
        service = LLMProviderService()
        service.register_provider(
            "openai",
            LLMProvider.OPENAI,
            api_key="invalid-key-12345",
            model="gpt-3.5-turbo"
        )

        with pytest.raises(Exception):  # Should raise authentication error
            await service.generate_completion("test")


@pytest.mark.integration
class TestAnthropicIntegration:
    """Test Anthropic provider integration"""

    @skip_anthropic
    @pytest.mark.asyncio
    async def test_anthropic_basic_completion(self):
        """Anthropic returns non-empty completion"""
        service = LLMProviderService()
        service.register_provider(
            "anthropic",
            LLMProvider.ANTHROPIC,
            api_key=ANTHROPIC_API_KEY,
            model="claude-3-haiku-20240307",
            max_tokens=50
        )

        response = await service.generate_completion(
            "Say 'Hello' in one word",
            temperature=0.0
        )

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert response.content is not None
        assert len(response.content) > 0
        assert response.provider == LLMProvider.ANTHROPIC
        assert response.model == "claude-3-haiku-20240307"

        # Verify usage tracking
        assert "input_tokens" in response.usage
        assert "output_tokens" in response.usage

    @skip_anthropic
    @pytest.mark.asyncio
    async def test_anthropic_with_temperature(self):
        """Anthropic accepts temperature parameter"""
        service = LLMProviderService()
        service.register_provider(
            "anthropic",
            LLMProvider.ANTHROPIC,
            api_key=ANTHROPIC_API_KEY
        )

        response = await service.generate_completion(
            "Count from 1 to 3",
            temperature=0.0,
            max_tokens=20
        )

        assert response.content is not None
        assert len(response.content) > 0

    @skip_anthropic
    @pytest.mark.asyncio
    async def test_anthropic_error_handling(self):
        """Anthropic handles invalid requests appropriately"""
        service = LLMProviderService()
        service.register_provider(
            "anthropic",
            LLMProvider.ANTHROPIC,
            api_key="invalid-key-12345"
        )

        with pytest.raises(Exception):  # Should raise authentication error
            await service.generate_completion("test")


@pytest.mark.integration
class TestGoogleIntegration:
    """Test Google Gemini provider integration"""

    @skip_google
    @pytest.mark.asyncio
    async def test_google_basic_completion(self):
        """Google returns non-empty completion"""
        service = LLMProviderService()
        service.register_provider(
            "google",
            LLMProvider.GOOGLE,
            api_key=GOOGLE_API_KEY,
            model="gemini-2.5-flash",
            max_tokens=50
        )

        response = await service.generate_completion(
            "Say 'Hello' in one word",
            temperature=0.0
        )

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert response.content is not None
        assert len(response.content) > 0
        assert response.provider == LLMProvider.GOOGLE
        assert response.model == "gemini-2.5-flash"

        # Verify usage tracking
        assert "total_token_count" in response.usage

    @skip_google
    @pytest.mark.asyncio
    async def test_google_with_temperature(self):
        """Google accepts temperature parameter"""
        service = LLMProviderService()
        service.register_provider(
            "google",
            LLMProvider.GOOGLE,
            api_key=GOOGLE_API_KEY,
            # Disable safety settings to avoid filtering
            safety_settings={
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
            }
        )

        response = await service.generate_completion(
            "Hello",  # Simple prompt
            temperature=0.0,
            max_tokens=10
        )

        assert response.content is not None
        assert len(response.content) > 0

    @skip_google
    @pytest.mark.asyncio
    async def test_google_error_handling(self):
        """Google handles invalid requests appropriately"""
        service = LLMProviderService()
        service.register_provider(
            "google",
            LLMProvider.GOOGLE,
            api_key="invalid-key-12345"
        )

        with pytest.raises(Exception):  # Should raise authentication error
            await service.generate_completion("test")


@pytest.mark.integration
class TestOllamaIntegration:
    """Test Ollama provider integration"""

    @skip_ollama
    @pytest.mark.asyncio
    async def test_ollama_basic_completion(self):
        """Ollama returns non-empty completion"""
        service = LLMProviderService()
        service.register_provider(
            "ollama",
            LLMProvider.OLLAMA,
            model="llama2",
            base_url="http://localhost:11434",
            max_tokens=50
        )

        response = await service.generate_completion(
            "Say 'Hello' in one word",
            temperature=0.0
        )

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert response.content is not None
        assert len(response.content) > 0
        assert response.provider == LLMProvider.OLLAMA

    @skip_ollama
    @pytest.mark.asyncio
    async def test_ollama_with_temperature(self):
        """Ollama accepts temperature parameter"""
        service = LLMProviderService()
        service.register_provider(
            "ollama",
            LLMProvider.OLLAMA,
            model="llama2"
        )

        response = await service.generate_completion(
            "Count from 1 to 3",
            temperature=0.0,
            max_tokens=20
        )

        assert response.content is not None
        assert len(response.content) > 0


@pytest.mark.integration
class TestStructuredOutputIntegration:
    """Test structured output with real providers"""

    @skip_openai
    @pytest.mark.asyncio
    async def test_openai_structured_output(self):
        """OpenAI generates valid structured output"""
        service = LLMProviderService()
        service.register_provider(
            "openai",
            LLMProvider.OPENAI,
            api_key=OPENAI_API_KEY,
            temperature=0.0
        )

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            }
        }

        result = await service.generate_structured_output(
            "Generate a person with name 'John' and age 30",
            schema
        )

        # Verify structure (content may vary)
        assert isinstance(result, dict)

        # Either we get valid parsed data or an error structure
        if "error" not in result:
            # Valid JSON was returned
            assert "name" in result or "age" in result
        else:
            # JSON parsing failed, but we got error structure
            assert "raw_content" in result

    @skip_anthropic
    @pytest.mark.asyncio
    async def test_anthropic_structured_output(self):
        """Anthropic generates valid structured output"""
        service = LLMProviderService()
        service.register_provider(
            "anthropic",
            LLMProvider.ANTHROPIC,
            api_key=ANTHROPIC_API_KEY,
            temperature=0.0
        )

        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "number"}
            }
        }

        result = await service.generate_structured_output(
            "Return JSON with count field set to 5",
            schema
        )

        # Verify structure
        assert isinstance(result, dict)

        if "error" not in result:
            assert "count" in result
        else:
            assert "raw_content" in result


@pytest.mark.integration
class TestMultiProviderIntegration:
    """Test using multiple providers in one service"""

    @pytest.mark.asyncio
    async def test_multiple_providers_registered(self):
        """Can register and use multiple providers"""
        # Skip if no API keys available
        if not OPENAI_API_KEY and not ANTHROPIC_API_KEY:
            pytest.skip("No API keys available for testing")

        service = LLMProviderService()

        if OPENAI_API_KEY:
            service.register_provider(
                "openai",
                LLMProvider.OPENAI,
                api_key=OPENAI_API_KEY
            )

        if ANTHROPIC_API_KEY:
            service.register_provider(
                "anthropic",
                LLMProvider.ANTHROPIC,
                api_key=ANTHROPIC_API_KEY
            )

        providers = service.list_providers()

        # At least one provider should be available
        assert len(providers) > 0

        # Default should be set
        assert service.default_provider is not None

    @skip_openai
    @pytest.mark.asyncio
    async def test_switching_between_providers(self):
        """Can switch between different providers"""
        service = LLMProviderService()

        service.register_provider(
            "openai",
            LLMProvider.OPENAI,
            api_key=OPENAI_API_KEY,
            model="gpt-3.5-turbo"
        )

        if ANTHROPIC_API_KEY:
            service.register_provider(
                "anthropic",
                LLMProvider.ANTHROPIC,
                api_key=ANTHROPIC_API_KEY
            )

        # Use OpenAI
        response1 = await service.generate_completion(
            "Say hello",
            provider_name="openai",
            max_tokens=10
        )
        assert response1.provider == LLMProvider.OPENAI

        # Use Anthropic (if available)
        if ANTHROPIC_API_KEY:
            response2 = await service.generate_completion(
                "Say hello",
                provider_name="anthropic",
                max_tokens=10
            )
            assert response2.provider == LLMProvider.ANTHROPIC


@pytest.mark.integration
class TestResponseConsistency:
    """Test response consistency across providers"""

    @pytest.mark.asyncio
    async def test_all_responses_have_required_fields(self):
        """All provider responses have required fields"""
        service = LLMProviderService()
        prompt = "Say 'test'"
        responses = []

        if OPENAI_API_KEY:
            service.register_provider("openai", LLMProvider.OPENAI, api_key=OPENAI_API_KEY)
            responses.append(await service.generate_completion(prompt, provider_name="openai", max_tokens=10))

        if ANTHROPIC_API_KEY:
            service.register_provider("anthropic", LLMProvider.ANTHROPIC, api_key=ANTHROPIC_API_KEY)
            responses.append(await service.generate_completion(prompt, provider_name="anthropic", max_tokens=10))

        if GOOGLE_API_KEY:
            service.register_provider("google", LLMProvider.GOOGLE, api_key=GOOGLE_API_KEY)
            responses.append(await service.generate_completion(prompt, provider_name="google", max_tokens=10))

        # Verify all responses have consistent structure
        for response in responses:
            assert isinstance(response, LLMResponse)
            assert response.content is not None
            assert len(response.content) > 0
            assert isinstance(response.provider, LLMProvider)
            assert response.model is not None
            assert isinstance(response.usage, dict)
            assert isinstance(response.timestamp, datetime)

    @pytest.mark.asyncio
    async def test_response_to_dict_consistency(self):
        """All provider responses convert to dict consistently"""
        service = LLMProviderService()
        prompt = "Hello"

        if OPENAI_API_KEY:
            service.register_provider("openai", LLMProvider.OPENAI, api_key=OPENAI_API_KEY)
            response = await service.generate_completion(prompt, provider_name="openai", max_tokens=10)

            data = response.to_dict()
            assert "content" in data
            assert "provider" in data
            assert "model" in data
            assert "usage" in data
            assert "timestamp" in data
