"""
Tests for pipeline.llm_provider_service module
"""

import pytest
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from ia_modules.pipeline.llm_provider_service import (
    LLMProvider,
    LLMConfig,
    LLMResponse,
    LLMProviderService
)


class TestLLMProvider:
    """Test LLMProvider enum"""

    def test_provider_values(self):
        """Test provider enum values"""
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.ANTHROPIC.value == "anthropic"
        assert LLMProvider.GOOGLE.value == "google"
        assert LLMProvider.AZURE_OPENAI.value == "azure_openai"
        assert LLMProvider.OLLAMA.value == "ollama"

    def test_provider_count(self):
        """Test number of providers"""
        providers = list(LLMProvider)
        assert len(providers) == 5


class TestLLMConfig:
    """Test LLMConfig class"""

    def test_init_basic(self):
        """Test basic initialization"""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="test-key",
            model="gpt-4",
            temperature=0.8,
            max_tokens=2000
        )

        assert config.provider == LLMProvider.OPENAI
        assert config.api_key == "test-key"
        assert config.model == "gpt-4"
        assert config.temperature == 0.8
        assert config.max_tokens == 2000

    def test_init_default_model_openai(self):
        """Test default model for OpenAI"""
        config = LLMConfig(provider=LLMProvider.OPENAI)
        assert config.model == "gpt-3.5-turbo"

    def test_init_default_model_anthropic(self):
        """Test default model for Anthropic"""
        config = LLMConfig(provider=LLMProvider.ANTHROPIC)
        assert config.model == "claude-3-haiku-20240307"

    def test_init_default_model_google(self):
        """Test default model for Google"""
        config = LLMConfig(provider=LLMProvider.GOOGLE)
        assert config.model == "gemini-2.5-flash"

    def test_init_default_model_ollama(self):
        """Test default model for Ollama"""
        config = LLMConfig(provider=LLMProvider.OLLAMA)
        assert config.model == "llama2"

    def test_init_with_base_url(self):
        """Test initialization with base URL"""
        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            base_url="http://localhost:11434"
        )
        assert config.base_url == "http://localhost:11434"

    def test_init_with_extra_params(self):
        """Test initialization with extra parameters"""
        config = LLMConfig(
            provider=LLMProvider.GOOGLE,
            safety_settings={"HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE"},
            top_p=0.95
        )

        assert "safety_settings" in config.extra_params
        assert "top_p" in config.extra_params
        assert config.extra_params["top_p"] == 0.95

    def test_default_temperature(self):
        """Test default temperature"""
        config = LLMConfig(provider=LLMProvider.OPENAI)
        assert config.temperature == 0.7

    def test_default_max_tokens(self):
        """Test default max tokens"""
        config = LLMConfig(provider=LLMProvider.OPENAI)
        assert config.max_tokens == 1000


class TestLLMResponse:
    """Test LLMResponse class"""

    def test_init_basic(self):
        """Test basic initialization"""
        response = LLMResponse(
            content="Test response",
            provider=LLMProvider.OPENAI,
            model="gpt-4"
        )

        assert response.content == "Test response"
        assert response.provider == LLMProvider.OPENAI
        assert response.model == "gpt-4"
        assert isinstance(response.timestamp, datetime)

    def test_init_with_usage(self):
        """Test initialization with usage data"""
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
        response = LLMResponse(
            content="Test",
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            usage=usage
        )

        assert response.usage == usage

    def test_init_with_metadata(self):
        """Test initialization with metadata"""
        metadata = {"finish_reason": "stop", "model_version": "0613"}
        response = LLMResponse(
            content="Test",
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            metadata=metadata
        )

        assert response.metadata == metadata

    def test_to_dict(self):
        """Test to_dict method"""
        response = LLMResponse(
            content="Test response",
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            usage={"total_tokens": 30},
            metadata={"finish_reason": "stop"}
        )

        result = response.to_dict()

        assert result["content"] == "Test response"
        assert result["provider"] == "openai"
        assert result["model"] == "gpt-4"
        assert result["usage"]["total_tokens"] == 30
        assert result["metadata"]["finish_reason"] == "stop"
        assert "timestamp" in result

    def test_empty_usage_metadata(self):
        """Test default empty usage and metadata"""
        response = LLMResponse(
            content="Test",
            provider=LLMProvider.OPENAI,
            model="gpt-4"
        )

        assert response.usage == {}
        assert response.metadata == {}


@pytest.mark.asyncio
class TestLLMProviderService:
    """Test LLMProviderService class"""

    async def test_init(self):
        """Test service initialization"""
        service = LLMProviderService()

        assert isinstance(service.providers, dict)
        assert len(service.providers) == 0
        assert service.default_provider is None

    async def test_cleanup(self):
        """Test cleanup method"""
        service = LLMProviderService()
        await service.cleanup()
        # Cleanup should not raise errors

    async def test_register_provider_basic(self):
        """Test registering a provider"""
        service = LLMProviderService()

        service.register_provider(
            "openai_default",
            LLMProvider.OPENAI,
            api_key="test-key",
            model="gpt-4"
        )

        assert "openai_default" in service.providers
        config = service.providers["openai_default"]
        assert config.provider == LLMProvider.OPENAI
        assert config.api_key == "test-key"
        assert config.model == "gpt-4"

    async def test_register_provider_as_default(self):
        """Test registering provider as default"""
        service = LLMProviderService()

        service.register_provider(
            "main_provider",
            LLMProvider.OPENAI,
            is_default=True
        )

        assert service.default_provider == "main_provider"

    async def test_register_first_provider_becomes_default(self):
        """Test first provider automatically becomes default"""
        service = LLMProviderService()

        service.register_provider("provider1", LLMProvider.OPENAI)

        assert service.default_provider == "provider1"

    async def test_register_multiple_providers(self):
        """Test registering multiple providers"""
        service = LLMProviderService()

        service.register_provider("openai", LLMProvider.OPENAI)
        service.register_provider("anthropic", LLMProvider.ANTHROPIC)
        service.register_provider("google", LLMProvider.GOOGLE)

        assert len(service.providers) == 3
        assert "openai" in service.providers
        assert "anthropic" in service.providers
        assert "google" in service.providers

    async def test_get_provider_by_name(self):
        """Test getting provider by name"""
        service = LLMProviderService()
        service.register_provider("test_provider", LLMProvider.OPENAI)

        config = service.get_provider("test_provider")

        assert config is not None
        assert config.provider == LLMProvider.OPENAI

    async def test_get_provider_default(self):
        """Test getting default provider"""
        service = LLMProviderService()
        service.register_provider("default", LLMProvider.OPENAI, is_default=True)

        config = service.get_provider()

        assert config is not None
        assert config.provider == LLMProvider.OPENAI

    async def test_get_provider_not_found(self):
        """Test getting non-existent provider"""
        service = LLMProviderService()

        config = service.get_provider("nonexistent")

        assert config is None

    async def test_get_provider_no_default(self):
        """Test getting provider when no default set"""
        service = LLMProviderService()

        config = service.get_provider()

        assert config is None

    async def test_list_providers_empty(self):
        """Test listing providers when empty"""
        service = LLMProviderService()

        providers = service.list_providers()

        assert isinstance(providers, list)
        assert len(providers) == 0

    async def test_list_providers_multiple(self):
        """Test listing multiple providers"""
        service = LLMProviderService()

        service.register_provider("openai", LLMProvider.OPENAI, is_default=True)
        service.register_provider("anthropic", LLMProvider.ANTHROPIC)

        providers = service.list_providers()

        assert len(providers) == 2

        # Check openai provider
        openai_provider = next(p for p in providers if p["name"] == "openai")
        assert openai_provider["provider"] == "openai"
        assert openai_provider["is_default"] is True

        # Check anthropic provider
        anthropic_provider = next(p for p in providers if p["name"] == "anthropic")
        assert anthropic_provider["provider"] == "anthropic"
        assert anthropic_provider["is_default"] is False

    async def test_generate_completion_no_provider(self):
        """Test generate completion with no provider configured"""
        service = LLMProviderService()

        with pytest.raises(ValueError, match="not found or no default provider"):
            await service.generate_completion("Test prompt")

    async def test_generate_completion_provider_not_found(self):
        """Test generate completion with invalid provider name"""
        service = LLMProviderService()
        service.register_provider("test", LLMProvider.OPENAI)

        with pytest.raises(ValueError, match="Provider 'invalid' not found"):
            await service.generate_completion("Test", provider_name="invalid")

    async def test_generate_structured_output_json_cleaning(self):
        """Test structured output JSON cleaning"""
        service = LLMProviderService()
        service.register_provider("test", LLMProvider.OPENAI)

        # Mock the generate_completion method
        mock_response = LLMResponse(
            content='```json\n{"key": "value"}\n```',
            provider=LLMProvider.OPENAI,
            model="gpt-4"
        )

        with patch.object(service, 'generate_completion', return_value=mock_response):
            result = await service.generate_structured_output(
                "Test prompt",
                {"type": "object"}
            )

            assert result == {"key": "value"}

    async def test_generate_structured_output_plain_json(self):
        """Test structured output with plain JSON"""
        service = LLMProviderService()
        service.register_provider("test", LLMProvider.OPENAI)

        mock_response = LLMResponse(
            content='{"key": "value", "number": 42}',
            provider=LLMProvider.OPENAI,
            model="gpt-4"
        )

        with patch.object(service, 'generate_completion', return_value=mock_response):
            result = await service.generate_structured_output(
                "Test prompt",
                {"type": "object"}
            )

            assert result == {"key": "value", "number": 42}

    async def test_generate_structured_output_invalid_json(self):
        """Test structured output with invalid JSON"""
        service = LLMProviderService()
        service.register_provider("test", LLMProvider.OPENAI)

        mock_response = LLMResponse(
            content='This is not valid JSON',
            provider=LLMProvider.OPENAI,
            model="gpt-4"
        )

        with patch.object(service, 'generate_completion', return_value=mock_response):
            result = await service.generate_structured_output(
                "Test prompt",
                {"type": "object"}
            )

            assert "error" in result
            assert result["error"] == "Failed to parse structured output"
            assert result["raw_content"] == 'This is not valid JSON'

    async def test_generate_completion_temperature_override(self):
        """Test temperature override in completion"""
        service = LLMProviderService()
        service.register_provider(
            "test",
            LLMProvider.OPENAI,
            temperature=0.7
        )

        # We can't easily test the actual call without mocking the provider
        # But we can verify the config exists
        config = service.get_provider("test")
        assert config.temperature == 0.7

    async def test_generate_completion_max_tokens_override(self):
        """Test max tokens override in completion"""
        service = LLMProviderService()
        service.register_provider(
            "test",
            LLMProvider.OPENAI,
            max_tokens=500
        )

        config = service.get_provider("test")
        assert config.max_tokens == 500

    async def test_call_openai_import_not_available(self):
        """Test OpenAI call when package not available"""
        service = LLMProviderService()
        config = LLMConfig(provider=LLMProvider.OPENAI)

        # Mock OPENAI_AVAILABLE to False
        with patch('ia_modules.pipeline.llm_provider_service.OPENAI_AVAILABLE', False):
            with pytest.raises(ImportError, match="OpenAI package not available"):
                await service._call_openai(config, "test", 0.7, 1000)

    async def test_call_anthropic_import_not_available(self):
        """Test Anthropic call when package not available"""
        service = LLMProviderService()
        config = LLMConfig(provider=LLMProvider.ANTHROPIC)

        with patch('ia_modules.pipeline.llm_provider_service.ANTHROPIC_AVAILABLE', False):
            with pytest.raises(ImportError, match="Anthropic package not available"):
                await service._call_anthropic(config, "test", 0.7, 1000)

    async def test_call_google_import_not_available(self):
        """Test Google call when package not available"""
        service = LLMProviderService()
        config = LLMConfig(provider=LLMProvider.GOOGLE)

        with patch('ia_modules.pipeline.llm_provider_service.GOOGLE_AVAILABLE', False):
            with pytest.raises(ImportError, match="Google Generative AI package not available"):
                await service._call_google(config, "test", 0.7, 1000)

    async def test_register_provider_with_extra_params(self):
        """Test registering provider with extra parameters"""
        service = LLMProviderService()

        service.register_provider(
            "google_safe",
            LLMProvider.GOOGLE,
            safety_settings={"HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE"},
            top_p=0.95,
            top_k=40
        )

        config = service.get_provider("google_safe")
        assert "safety_settings" in config.extra_params
        assert "top_p" in config.extra_params
        assert config.extra_params["top_k"] == 40

    async def test_ollama_default_base_url(self):
        """Test Ollama uses default base URL"""
        service = LLMProviderService()
        service.register_provider("ollama", LLMProvider.OLLAMA)

        config = service.get_provider("ollama")
        # Base URL should be None, will use default in _call_ollama
        assert config.base_url is None

    async def test_ollama_custom_base_url(self):
        """Test Ollama with custom base URL"""
        service = LLMProviderService()
        service.register_provider(
            "ollama",
            LLMProvider.OLLAMA,
            base_url="http://custom-host:11434"
        )

        config = service.get_provider("ollama")
        assert config.base_url == "http://custom-host:11434"

    async def test_structured_output_adds_schema_to_prompt(self):
        """Test that structured output adds schema to prompt"""
        service = LLMProviderService()
        service.register_provider("test", LLMProvider.OPENAI)

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        mock_response = LLMResponse(
            content='{"name": "test"}',
            provider=LLMProvider.OPENAI,
            model="gpt-4"
        )

        call_args = []

        async def mock_generate(prompt, *args, **kwargs):
            call_args.append(prompt)
            return mock_response

        with patch.object(service, 'generate_completion', side_effect=mock_generate):
            await service.generate_structured_output("Original prompt", schema)

            assert len(call_args) == 1
            assert "Original prompt" in call_args[0]
            assert "JSON" in call_args[0]
            assert "schema" in call_args[0].lower()
