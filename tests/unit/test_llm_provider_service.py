"""
Tests for LLM Provider Service - Clean LiteLLM Wrapper
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from ia_modules.pipeline.llm_provider_service import LLMProviderService


class TestLLMProviderService:
    """Test LLMProviderService"""

    def test_init(self):
        """Test service initialization"""
        service = LLMProviderService()
        assert service._providers == {}
        assert service._default_provider is None

    def test_register_provider(self):
        """Test registering a provider"""
        service = LLMProviderService()

        service.register_provider(
            name="openai",
            model="gpt-4o",
            api_key="test-key"
        )

        assert "openai" in service._providers
        assert service._providers["openai"]["model"] == "gpt-4o"
        assert service._providers["openai"]["api_key"] == "test-key"
        assert service._default_provider == "openai"

    def test_register_multiple_providers(self):
        """Test registering multiple providers"""
        service = LLMProviderService()

        service.register_provider("openai", model="gpt-4o", api_key="key1", is_default=True)
        service.register_provider("anthropic", model="claude-sonnet-4-5-20250929", api_key="key2")

        assert len(service._providers) == 2
        assert service._default_provider == "openai"

    def test_list_providers(self):
        """Test listing providers"""
        service = LLMProviderService()

        service.register_provider("openai", model="gpt-4o", is_default=True)
        service.register_provider("anthropic", model="claude-sonnet-4-5-20250929")

        providers = service.list_providers()
        assert len(providers) == 2
        assert providers[0]["name"] == "openai"
        assert providers[0]["is_default"] is True
        assert providers[1]["name"] == "anthropic"
        assert providers[1]["is_default"] is False

    def test_get_provider(self):
        """Test getting provider config"""
        service = LLMProviderService()
        service.register_provider("openai", model="gpt-4o", api_key="test-key")

        config = service.get_provider("openai")
        assert config is not None
        assert config["model"] == "gpt-4o"
        assert config["api_key"] == "test-key"

    def test_get_provider_not_found(self):
        """Test getting non-existent provider"""
        service = LLMProviderService()

        config = service.get_provider("nonexistent")
        assert config is None

    async def test_generate_completion_with_provider(self):
        """Test generating completion using registered provider"""
        service = LLMProviderService()
        service.register_provider("openai", model="gpt-4o", api_key="test-key")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4o"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15

        with patch('ia_modules.pipeline.llm_provider_service.acompletion', new_callable=AsyncMock) as mock_acompletion:
            with patch('ia_modules.pipeline.llm_provider_service.completion_cost', return_value=0.001):
                mock_acompletion.return_value = mock_response

                response = await service.generate_completion(
                    messages=[{"role": "user", "content": "Hello"}],
                    provider_name="openai"
                )

                mock_acompletion.assert_called_once()
                assert response["content"] == "Hello!"
                assert response["usage"]["cost_usd"] == 0.001

    async def test_generate_completion_with_direct_model(self):
        """Test generating completion with direct model"""
        service = LLMProviderService()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-3.5-turbo"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15

        with patch('ia_modules.pipeline.llm_provider_service.acompletion', new_callable=AsyncMock) as mock_acompletion:
            with patch('ia_modules.pipeline.llm_provider_service.completion_cost', return_value=0.0005):
                mock_acompletion.return_value = mock_response

                response = await service.generate_completion(
                    messages=[{"role": "user", "content": "Test"}],
                    model="gpt-3.5-turbo"
                )

                assert response["content"] == "Test"

    async def test_generate_completion_passes_all_params(self):
        """Test that all LiteLLM parameters pass through"""
        service = LLMProviderService()
        service.register_provider("openai", model="gpt-4o")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4o"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15

        with patch('ia_modules.pipeline.llm_provider_service.acompletion', new_callable=AsyncMock) as mock_acompletion:
            with patch('ia_modules.pipeline.llm_provider_service.completion_cost', return_value=0.001):
                mock_acompletion.return_value = mock_response

                response = await service.generate_completion(
                    messages=[{"role": "user", "content": "Test"}],
                    provider_name="openai",
                    temperature=0.8,
                    max_tokens=100,
                    top_p=0.9
                )

                call_args = mock_acompletion.call_args
                assert call_args.kwargs["temperature"] == 0.8
                assert call_args.kwargs["max_tokens"] == 100
                assert call_args.kwargs["top_p"] == 0.9

    async def test_no_provider_error(self):
        """Test error when no provider specified"""
        service = LLMProviderService()

        with pytest.raises(ValueError, match="No provider specified"):
            await service.generate_completion(
                messages=[{"role": "user", "content": "Test"}]
            )

    async def test_provider_not_found_error(self):
        """Test error when provider not found"""
        service = LLMProviderService()
        service.register_provider("openai", model="gpt-4o")

        with pytest.raises(ValueError, match="not found"):
            await service.generate_completion(
                messages=[{"role": "user", "content": "Test"}],
                provider_name="nonexistent"
            )
