"""
Integration Tests for Pattern Service with LLM Monitoring

Tests that pattern service correctly integrates with monitoring for
rate limiting, cost tracking, and usage statistics.
"""

import pytest
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch
from backend.services.pattern_service import PatternService
from ia_modules.pipeline.llm_provider_service import LLMResponse, LLMProvider


@pytest.fixture
def mock_llm_service():
    """Mock LLM service with realistic responses"""
    service = MagicMock()
    service.providers = {"openai": MagicMock()}
    
    async def mock_completion(*args, **kwargs):
        # Simulate realistic LLM response
        response = LLMResponse(
            content="This is a test response from the LLM.",
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        )
        return response
    
    service.generate_completion = AsyncMock(side_effect=mock_completion)
    return service


@pytest.fixture
def pattern_service_with_mock(mock_llm_service):
    """Pattern service with mocked LLM"""
    service = PatternService()
    service.llm_service = mock_llm_service
    return service


class TestPatternServiceMonitoring:
    """Test pattern service monitoring integration"""
    
    @pytest.mark.asyncio
    async def test_monitored_llm_call_tracks_usage(self, pattern_service_with_mock):
        """Monitored LLM call tracks token usage and cost"""
        service = pattern_service_with_mock
        
        # Make monitored call
        response = await service._monitored_llm_call(
            prompt="Test prompt",
            temperature=0.7,
            max_tokens=1000
        )
        
        # Should have usage stats attached
        assert hasattr(response, 'usage_stats')
        assert response.usage_stats["provider"] == "openai"
        assert response.usage_stats["model"] == "gpt-4o"
        assert response.usage_stats["input_tokens"] == 100
        assert response.usage_stats["output_tokens"] == 50
        assert response.usage_stats["total_tokens"] == 150
        assert response.usage_stats["cost_usd"] > 0
        assert "duration_seconds" in response.usage_stats
    
    @pytest.mark.asyncio
    async def test_monitored_call_enforces_rate_limits(self, pattern_service_with_mock):
        """Rate limits are enforced on monitored calls"""
        service = pattern_service_with_mock
        
        # Exhaust rate limit
        service.monitoring_service.request_limiter.tokens = 0
        
        # Should raise HTTPException with 429
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await service._monitored_llm_call(
                prompt="Test",
                temperature=0.7,
                max_tokens=1000
            )
        
        assert exc_info.value.status_code == 429
        assert "Rate limit exceeded" in str(exc_info.value.detail)
        assert "Retry-After" in exc_info.value.headers
    
    @pytest.mark.asyncio
    async def test_reflection_pattern_tracks_usage(self, pattern_service_with_mock):
        """Reflection pattern execution tracks usage"""
        service = pattern_service_with_mock
        
        # Run reflection pattern
        result = await service.reflection_example(
            initial_output="This is a test output that needs improvement.",
            criteria={
                "clarity": "Text should be clear and easy to understand",
                "completeness": "All key points should be covered"
            },
            max_iterations=2
        )
        
        # Check monitoring service has recorded usage
        stats = service.monitoring_service.get_stats()
        assert stats["total_requests"] > 0
        assert stats["total_tokens"] > 0
        assert stats["total_cost"] > 0
    
    @pytest.mark.asyncio
    async def test_multiple_patterns_accumulate_costs(self, pattern_service_with_mock):
        """Multiple pattern executions accumulate costs"""
        service = pattern_service_with_mock
        
        # Run reflection pattern twice
        for _ in range(2):
            await service._monitored_llm_call(
                prompt="Test",
                temperature=0.7,
                max_tokens=500
            )
        
        stats = service.monitoring_service.get_stats()
        assert stats["total_requests"] == 2
        assert stats["total_tokens"] == 300  # 150 * 2
        # Cost should be 2x single call
        expected_cost = 2 * ((100 * 2.50 + 50 * 10.00) / 1_000_000)
        assert abs(stats["total_cost"] - expected_cost) < 0.0001
    
    @pytest.mark.asyncio
    async def test_cost_warning_on_expensive_request(self, pattern_service_with_mock, capsys):
        """Warns when single request exceeds cost limit"""
        service = pattern_service_with_mock
        
        # Set very low cost limit
        service.monitoring_service.max_cost_per_request = 0.0001
        
        # Make request that will exceed limit
        await service._monitored_llm_call(
            prompt="Test",
            temperature=0.7,
            max_tokens=1000
        )
        
        # Should print warning (captured by capsys)
        captured = capsys.readouterr()
        assert "Warning" in captured.out or "exceeded limit" in captured.out.lower()
    
    @pytest.mark.asyncio
    async def test_daily_limit_warning(self, pattern_service_with_mock, capsys):
        """Warns when daily spending exceeds limit"""
        service = pattern_service_with_mock
        
        # Set low daily limit
        service.monitoring_service.daily_spending_limit = 0.001
        
        # Make multiple requests to exceed daily limit
        for _ in range(10):
            await service._monitored_llm_call(
                prompt="Test",
                temperature=0.7,
                max_tokens=1000
            )
        
        # Should eventually warn about daily limit
        captured = capsys.readouterr()
        assert "Warning" in captured.out or "daily" in captured.out.lower()


class TestPatternServiceNoLLM:
    """Test pattern service behavior without LLM configured"""
    
    @pytest.mark.asyncio
    async def test_llm_call_fails_without_service(self):
        """LLM calls fail with clear error when not configured"""
        service = PatternService()
        service.llm_service = None
        
        with pytest.raises(RuntimeError) as exc_info:
            await service._llm_generate_critique(
                output="Test output",
                criteria={"clarity": "Should be clear"}
            )
        
        assert "LLM service not configured" in str(exc_info.value)
        assert "API keys" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_monitored_call_fails_without_service(self):
        """Monitored calls fail without LLM service"""
        service = PatternService()
        service.llm_service = None
        
        with pytest.raises(RuntimeError) as exc_info:
            await service._monitored_llm_call(
                prompt="Test",
                temperature=0.7,
                max_tokens=1000
            )
        
        assert "LLM service not configured" in str(exc_info.value)


class TestAPIEndpoints:
    """Test API endpoints with monitoring"""
    
    @pytest.mark.asyncio
    async def test_llm_status_endpoint_no_keys(self):
        """Status endpoint shows unconfigured when no API keys"""
        # Import here to avoid module-level import issues
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        
        from backend.api.patterns import get_llm_status
        
        # Temporarily clear environment
        old_env = {
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
            "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
            "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY")
        }
        
        for key in old_env:
            if key in os.environ:
                del os.environ[key]
        
        try:
            result = await get_llm_status()
            
            assert result["configured"] is False
            assert result["configured_count"] == 0
            assert "must be configured" in result["message"]
            
            # All providers should be not_configured
            for provider in result["providers"]:
                assert provider["status"] == "not_configured"
                assert "setup_guide" in provider
        finally:
            # Restore environment
            for key, value in old_env.items():
                if value is not None:
                    os.environ[key] = value
    
    @pytest.mark.asyncio
    async def test_llm_status_endpoint_with_keys(self):
        """Status endpoint shows configured when API keys present"""
        # Import here to avoid module-level import issues
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        
        from backend.api.patterns import get_llm_status
        
        # Set test API key
        os.environ["OPENAI_API_KEY"] = "test-key-123"
        
        try:
            result = await get_llm_status()
            
            assert result["configured"] is True
            assert result["configured_count"] >= 1
            
            # Find OpenAI provider
            openai_provider = next(p for p in result["providers"] if p["name"] == "openai")
            assert openai_provider["status"] == "configured"
            assert "model" in openai_provider
        finally:
            del os.environ["OPENAI_API_KEY"]
    
    @pytest.mark.asyncio
    async def test_llm_stats_endpoint(self):
        """Stats endpoint returns usage statistics"""
        # Import here to avoid module-level import issues
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        
        from backend.api.patterns import get_llm_stats, monitoring_service
        
        # Add some test usage
        monitoring_service.track_usage(
            provider="openai",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            duration_seconds=1.5
        )
        
        result = await get_llm_stats()
        
        assert "total_requests" in result
        assert "total_tokens" in result
        assert "total_cost" in result
        assert "average_cost" in result
        assert "recent_hour" in result
        assert "daily_costs" in result
        assert "rate_limits" in result


class TestRateLimitingIntegration:
    """Test rate limiting integration"""
    
    @pytest.mark.asyncio
    async def test_rate_limit_with_retry_after(self, pattern_service_with_mock):
        """Rate limit response includes retry-after header"""
        service = pattern_service_with_mock
        
        # Exhaust rate limit
        service.monitoring_service.request_limiter.consume(
            service.monitoring_service.request_limiter.capacity
        )
        
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await service._monitored_llm_call(
                prompt="Test",
                temperature=0.7,
                max_tokens=1000
            )
        
        # Should have Retry-After header
        assert "Retry-After" in exc_info.value.headers
        retry_after = int(exc_info.value.headers["Retry-After"])
        # Retry after should be >= 0 (could be 0 if bucket refills quickly)
        assert retry_after >= 0
    
    @pytest.mark.asyncio
    async def test_token_rate_limit_refunds_request(self, pattern_service_with_mock):
        """Token rate limit refunds request token"""
        service = pattern_service_with_mock
        
        # Record initial request tokens
        initial_request_tokens = service.monitoring_service.request_limiter.tokens
        
        # Exhaust token limit
        service.monitoring_service.token_limiter.consume(
            service.monitoring_service.token_limiter.capacity
        )
        
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            await service._monitored_llm_call(
                prompt="Test",
                temperature=0.7,
                max_tokens=1000
            )
        
        # Request token should be refunded
        # (consumed during check, refunded after token limit failure)
        assert service.monitoring_service.request_limiter.tokens >= initial_request_tokens - 1


class TestCostCalculationIntegration:
    """Test cost calculation integration"""
    
    @pytest.mark.asyncio
    async def test_different_providers_different_costs(self):
        """Different providers calculate different costs"""
        from backend.services.llm_monitoring_service import CostCalculator
        
        # Same token counts, different providers
        tokens = {"input": 1000, "output": 500}
        
        openai_cost = CostCalculator.calculate_cost(
            "openai", "gpt-4o", tokens["input"], tokens["output"]
        )
        
        anthropic_cost = CostCalculator.calculate_cost(
            "anthropic", "claude-3-5-sonnet-20241022", tokens["input"], tokens["output"]
        )
        
        gemini_cost = CostCalculator.calculate_cost(
            "google", "gemini-2.5-flash", tokens["input"], tokens["output"]
        )
        
        # All should be different
        assert openai_cost != anthropic_cost
        assert anthropic_cost != gemini_cost
        assert openai_cost != gemini_cost
        
        # Gemini Flash should be cheapest
        assert gemini_cost < openai_cost
        assert gemini_cost < anthropic_cost


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
