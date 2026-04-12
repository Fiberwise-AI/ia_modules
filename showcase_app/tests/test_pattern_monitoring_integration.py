"""
Integration Tests for Pattern Service with LLM Monitoring

Tests that pattern service correctly integrates with monitoring for
rate limiting, cost tracking, and usage statistics.
"""

import pytest
import os
import sys
from unittest.mock import patch, AsyncMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class MockAdapter:
    """Mock adapter matching SubprocessAgentAdapter.generate() interface."""

    async def generate(self, prompt: str, model: str = None, temperature: float = 0.7, max_tokens: int = None, **kwargs):
        """Return plain string based on prompt content.

        Uses distinctive phrases from each pattern's prompt to avoid
        ambiguity (e.g. 'step' appears in tool-use JSON templates too).
        """
        prompt_lower = prompt.lower()
        # Reflection: critique step
        if "critical evaluator" in prompt_lower:
            return "The output is acceptable. Clarity is good. The content is complete."
        # Reflection: improvement step
        elif "expert editor" in prompt_lower:
            return "This is an improved version with better clarity and completeness."
        # Tool use: analyze and select tools
        elif "available tools" in prompt_lower:
            return '{"analysis": {"task_type": "test", "required_capabilities": ["search"], "complexity": "low"}, "selected_tools": [{"tool": "search", "reasoning": "needed", "priority": 1}], "execution_plan": [{"step": 1, "tool": "search", "action": "search", "input": "query", "output": "results"}], "reasoning": "search is best"}'
        # Planning: goal decomposition
        elif "break down this goal" in prompt_lower:
            return '[{"description": "Step 1", "reasoning": "Important", "duration": 30, "dependencies": [], "success_criteria": ["Done"]}]'
        # RAG: query refinement
        elif "refining search queries" in prompt_lower:
            return "refined search query with better keywords"
        # RAG: document evaluation
        elif "evaluating document relevance" in prompt_lower:
            return '{"document_scores": [{"document_number": 1, "title": "Doc 1", "relevance_score": 0.9, "reasoning": "relevant"}], "average_relevance": 0.9, "reasoning": "good results", "refinement_suggestion": "none needed"}'
        # Metacognition: performance analysis
        elif "analyzing your own performance" in prompt_lower:
            return '{"assessment": {"overall_score": 0.8, "summary": "Good", "strengths": ["fast"], "weaknesses": ["none"]}, "patterns": ["consistent"], "issues": [], "adjustments": ["none needed"], "confidence": 0.9}'
        else:
            return "Mock response for the given prompt"


@pytest.fixture
def mock_adapter():
    return MockAdapter()


@pytest.fixture
def pattern_service_with_mock(mock_adapter):
    """Pattern service with mocked adapter."""
    with patch("backend.services.pattern_service._make_adapter", return_value=mock_adapter), \
         patch("showcase_app.backend.services.pattern_service._make_adapter", return_value=mock_adapter):
        from backend.services.pattern_service import PatternService
        service = PatternService()
        return service


class TestPatternServiceMonitoring:
    """Test pattern service monitoring integration"""

    @pytest.mark.asyncio
    async def test_monitored_llm_call_returns_string(self, pattern_service_with_mock):
        """Monitored call returns plain string"""
        service = pattern_service_with_mock

        result = await service._monitored_llm_call(
            prompt="Test prompt",
            temperature=0.7,
            max_tokens=1000
        )

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_monitored_call_enforces_rate_limits(self, pattern_service_with_mock):
        """Rate limits are enforced on monitored calls"""
        service = pattern_service_with_mock

        # Exhaust rate limit
        service.monitoring_service.request_limiter.tokens = 0

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
    async def test_reflection_pattern_tracks_requests(self, pattern_service_with_mock):
        """Reflection pattern execution tracks request count"""
        service = pattern_service_with_mock

        await service.reflection_example(
            initial_output="This is a test output that needs improvement.",
            criteria={
                "clarity": "Text should be clear and easy to understand",
                "completeness": "All key points should be covered"
            },
            max_iterations=2
        )

        stats = service.monitoring_service.get_stats()
        assert stats["total_requests"] > 0

    @pytest.mark.asyncio
    async def test_multiple_calls_accumulate_requests(self, pattern_service_with_mock):
        """Multiple calls accumulate request counts"""
        service = pattern_service_with_mock

        for _ in range(3):
            await service._monitored_llm_call(
                prompt="Test",
                temperature=0.7,
                max_tokens=500
            )

        stats = service.monitoring_service.get_stats()
        assert stats["total_requests"] == 3


class TestPatternServiceWithAdapter:
    """Test pattern service with adapter integration"""

    @pytest.mark.asyncio
    async def test_reflection_produces_result(self, pattern_service_with_mock):
        """Reflection pattern produces valid result structure"""
        service = pattern_service_with_mock

        result = await service.reflection_example(
            initial_output="Test output",
            criteria={"clarity": "Must be clear"},
            max_iterations=1
        )

        assert result["pattern"] == "reflection"
        assert "final_output" in result
        assert "iterations" in result
        assert result["total_iterations"] >= 1

    @pytest.mark.asyncio
    async def test_planning_produces_result(self, pattern_service_with_mock):
        """Planning pattern produces valid result structure"""
        service = pattern_service_with_mock

        result = await service.planning_example(
            goal="Research machine learning"
        )

        assert result["pattern"] == "planning"
        assert "plan" in result

    @pytest.mark.asyncio
    async def test_tool_use_produces_result(self, pattern_service_with_mock):
        """Tool use pattern produces valid result structure"""
        service = pattern_service_with_mock

        result = await service.tool_use_example(
            task="Search for information",
            available_tools=["search", "calculator"]
        )

        assert result["pattern"] == "tool_use"
        assert "selected_tools" in result
        assert "execution_plan" in result

    @pytest.mark.asyncio
    async def test_metacognition_produces_result(self, pattern_service_with_mock):
        """Metacognition pattern produces valid result structure"""
        service = pattern_service_with_mock

        result = await service.metacognition_example(
            execution_trace=[{"action": "search", "status": "ok"}],
            performance_metrics={"accuracy": 0.85, "speed": 0.9}
        )

        assert result["pattern"] == "metacognition"
        assert "performance_assessment" in result
        assert "strategy_adjustments" in result


class TestAPIEndpoints:
    """Test API endpoints with monitoring"""

    @pytest.mark.asyncio
    async def test_llm_status_endpoint_no_keys(self):
        """Status endpoint shows unconfigured when no API keys"""
        from backend.api.patterns import get_llm_status

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

            for provider in result["providers"]:
                assert provider["status"] == "not_configured"
                assert "setup_guide" in provider
        finally:
            for key, value in old_env.items():
                if value is not None:
                    os.environ[key] = value

    @pytest.mark.asyncio
    async def test_llm_status_endpoint_with_keys(self):
        """Status endpoint shows configured when API keys present"""
        from backend.api.patterns import get_llm_status

        os.environ["OPENAI_API_KEY"] = "test-key-123"

        try:
            result = await get_llm_status()

            assert result["configured"] is True
            assert result["configured_count"] >= 1

            openai_provider = next(p for p in result["providers"] if p["name"] == "openai")
            assert openai_provider["status"] == "configured"
            assert "model" in openai_provider
        finally:
            del os.environ["OPENAI_API_KEY"]

    @pytest.mark.asyncio
    async def test_llm_stats_endpoint(self):
        """Stats endpoint returns usage statistics"""
        from backend.api.patterns import get_llm_stats, monitoring_service

        monitoring_service.track_usage(
            provider="subprocess",
            model="cli_agent",
            input_tokens=0,
            output_tokens=0,
            duration_seconds=1.5
        )

        result = await get_llm_stats()

        assert "total_requests" in result
        assert "total_tokens" in result
        assert "total_cost" in result


class TestRateLimitingIntegration:
    """Test rate limiting integration"""

    @pytest.mark.asyncio
    async def test_rate_limit_with_retry_after(self, pattern_service_with_mock):
        """Rate limit response includes retry-after header"""
        service = pattern_service_with_mock

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

        assert "Retry-After" in exc_info.value.headers
        retry_after = int(exc_info.value.headers["Retry-After"])
        assert retry_after >= 0


class TestCostCalculationIntegration:
    """Test cost calculation integration"""

    @pytest.mark.asyncio
    async def test_different_providers_different_costs(self):
        """Different providers calculate different costs"""
        from backend.services.llm_monitoring_service import CostCalculator

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

        assert openai_cost != anthropic_cost
        assert anthropic_cost != gemini_cost
        assert openai_cost != gemini_cost

        # Gemini Flash should be cheapest
        assert gemini_cost < openai_cost
        assert gemini_cost < anthropic_cost


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
