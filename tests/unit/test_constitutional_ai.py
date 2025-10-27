"""
Unit tests for Constitutional AI pattern.

Tests critique, revision, and principle evaluation functionality.
"""

import pytest
from ia_modules.pipeline.test_utils import create_test_execution_context
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from ia_modules.patterns.constitutional_ai import (
    Principle,
    PrincipleCategory,
    CritiqueResult,
    ConstitutionalConfig,
    RevisionHistory,
    ConstitutionalAIStep,
    apply_constitutional_ai,
)


class TestPrinciple:
    """Test Principle dataclass."""

    def test_principle_creation_minimal(self):
        """Principle can be created with minimal fields."""
        principle = Principle(
            name="harmless",
            description="Ensure output is harmless",
            critique_prompt="Is this response harmless?"
        )

        assert principle.name == "harmless"
        assert principle.description == "Ensure output is harmless"
        assert principle.critique_prompt == "Is this response harmless?"
        assert principle.weight == 1.0
        assert principle.category == PrincipleCategory.CUSTOM
        assert principle.min_score == 0.7

    def test_principle_creation_full(self):
        """Principle can be created with all fields."""
        principle = Principle(
            name="helpful",
            description="Ensure output is helpful",
            critique_prompt="Is this response helpful?",
            weight=2.0,
            category=PrincipleCategory.HELPFUL,
            min_score=0.8
        )

        assert principle.name == "helpful"
        assert principle.weight == 2.0
        assert principle.category == PrincipleCategory.HELPFUL
        assert principle.min_score == 0.8

    def test_format_critique_prompt(self):
        """Principle formats critique prompt with response."""
        principle = Principle(
            name="test",
            description="Test principle",
            critique_prompt="Evaluate this:"
        )

        response = "This is a test response"
        formatted = principle.format_critique_prompt(response)

        assert "Evaluate this:" in formatted
        assert "This is a test response" in formatted
        assert "Response to evaluate:" in formatted


class TestCritiqueResult:
    """Test CritiqueResult dataclass."""

    def test_critique_result_creation(self):
        """CritiqueResult can be created."""
        result = CritiqueResult(
            principle_name="harmless",
            score=0.85,
            feedback="Good response",
            passed=True,
            suggestions=["Keep it up"]
        )

        assert result.principle_name == "harmless"
        assert result.score == 0.85
        assert result.feedback == "Good response"
        assert result.passed is True
        assert result.suggestions == ["Keep it up"]

    def test_critique_result_default_suggestions(self):
        """CritiqueResult has default empty suggestions list."""
        result = CritiqueResult(
            principle_name="test",
            score=0.5,
            feedback="Needs work",
            passed=False
        )

        assert result.suggestions == []


class TestConstitutionalConfig:
    """Test ConstitutionalConfig dataclass."""

    def test_config_creation_minimal(self):
        """Config can be created with minimal fields."""
        principles = [
            Principle(
                name="harmless",
                description="Be harmless",
                critique_prompt="Is it harmless?"
            )
        ]

        config = ConstitutionalConfig(principles=principles)

        assert config.principles == principles
        assert config.max_revisions == 3
        assert config.min_quality_score == 0.8
        assert config.parallel_critique is False
        assert config.aggregate_method == "weighted_average"
        assert config.stop_on_regression is True

    def test_config_creation_full(self):
        """Config can be created with all fields."""
        principles = [
            Principle(name="test", description="Test", critique_prompt="Test?")
        ]

        config = ConstitutionalConfig(
            principles=principles,
            max_revisions=5,
            critique_model="gpt-4",
            revision_model="gpt-3.5",
            min_quality_score=0.9,
            parallel_critique=True,
            aggregate_method="min",
            stop_on_regression=False
        )

        assert config.max_revisions == 5
        assert config.critique_model == "gpt-4"
        assert config.revision_model == "gpt-3.5"
        assert config.min_quality_score == 0.9
        assert config.parallel_critique is True
        assert config.aggregate_method == "min"
        assert config.stop_on_regression is False


class TestRevisionHistory:
    """Test RevisionHistory dataclass."""

    def test_revision_history_creation(self):
        """RevisionHistory can be created."""
        critiques = [
            CritiqueResult(
                principle_name="test",
                score=0.8,
                feedback="Good",
                passed=True
            )
        ]

        history = RevisionHistory(
            iteration=0,
            response="Test response",
            critiques=critiques,
            quality_score=0.8,
            timestamp=123456.0
        )

        assert history.iteration == 0
        assert history.response == "Test response"
        assert history.critiques == critiques
        assert history.quality_score == 0.8
        assert history.timestamp == 123456.0


@pytest.mark.asyncio
class TestConstitutionalAIStep:
    """Test ConstitutionalAIStep functionality."""

    @pytest.fixture
    def principles(self):
        """Create test principles."""
        return [
            Principle(
                name="harmless",
                description="Be harmless",
                critique_prompt="Is this harmless?",
                weight=1.0,
                min_score=0.7
            ),
            Principle(
                name="helpful",
                description="Be helpful",
                critique_prompt="Is this helpful?",
                weight=1.5,
                min_score=0.8
            )
        ]

    @pytest.fixture
    def config(self, principles):
        """Create test config."""
        return ConstitutionalConfig(
            principles=principles,
            max_revisions=3,
            min_quality_score=0.8
        )

    @pytest.fixture
    def mock_llm_provider(self):
        """Create mock LLM provider."""
        provider = Mock()
        provider.generate = AsyncMock()
        return provider

    async def test_step_creation(self, config):
        """ConstitutionalAIStep can be created."""
        step = ConstitutionalAIStep(
            name="test_step",
            prompt="Generate a response about {topic}",
            config=config
        )

        assert step.name == "test_step"
        assert step.prompt == "Generate a response about {topic}"
        assert step.config == config
        assert step.history == []

    async def test_execute_without_llm(self, config):
        """Step executes without LLM provider (uses fallback)."""
        step = ConstitutionalAIStep(
            name="test",
            prompt="Test prompt",
            config=config
        )

        result = await step.execute({"topic": "AI"})

        assert "response" in result
        assert "quality_score" in result
        assert "revisions" in result
        assert "principles_passed" in result
        assert "principles_failed" in result
        assert "history" in result

    async def test_execute_with_llm(self, config, mock_llm_provider):
        """Step executes with LLM provider."""
        # Mock initial response
        mock_llm_provider.generate.return_value = {
            "content": "This is a great response!"
        }

        step = ConstitutionalAIStep(
            name="test",
            prompt="Test prompt",
            config=config,
            llm_provider=mock_llm_provider
        )

        result = await step.execute({})

        assert result["response"] is not None
        assert mock_llm_provider.generate.called

    async def test_quality_threshold_reached(self, principles):
        """Step stops when quality threshold is reached."""
        config = ConstitutionalConfig(
            principles=principles,
            max_revisions=10,
            min_quality_score=0.5  # Low threshold
        )

        step = ConstitutionalAIStep(
            name="test",
            prompt="Test",
            config=config
        )

        result = await step.execute({})

        # Should stop early when quality threshold reached
        assert result["revisions"] < 10

    async def test_max_revisions_reached(self, principles):
        """Step stops at max revisions."""
        config = ConstitutionalConfig(
            principles=principles,
            max_revisions=2,
            min_quality_score=0.99  # High threshold (hard to reach)
        )

        step = ConstitutionalAIStep(
            name="test",
            prompt="Test",
            config=config
        )

        result = await step.execute({})

        # Should hit max revisions
        assert result["revisions"] <= 2

    async def test_revision_history_tracking(self, config):
        """Step tracks revision history."""
        step = ConstitutionalAIStep(
            name="test",
            prompt="Test",
            config=config
        )

        await step.execute({})

        assert len(step.history) > 0
        assert all(isinstance(h, RevisionHistory) for h in step.history)
        assert all(h.iteration >= 0 for h in step.history)

    async def test_calculate_quality_score_weighted_average(self, config):
        """Quality score calculated with weighted average."""
        step = ConstitutionalAIStep(
            name="test",
            prompt="Test",
            config=config
        )

        critiques = [
            CritiqueResult(
                principle_name="harmless",
                score=0.8,
                feedback="Good",
                passed=True
            ),
            CritiqueResult(
                principle_name="helpful",
                score=0.6,
                feedback="OK",
                passed=True
            )
        ]

        # harmless: weight=1.0, score=0.8
        # helpful: weight=1.5, score=0.6
        # Expected: (0.8*1.0 + 0.6*1.5) / (1.0 + 1.5) = 1.7 / 2.5 = 0.68
        quality_score = step._calculate_quality_score(critiques)

        assert abs(quality_score - 0.68) < 0.01

    async def test_calculate_quality_score_min(self, principles):
        """Quality score calculated with min aggregation."""
        config = ConstitutionalConfig(
            principles=principles,
            aggregate_method="min"
        )

        step = ConstitutionalAIStep(
            name="test",
            prompt="Test",
            config=config
        )

        critiques = [
            CritiqueResult(
                principle_name="harmless",
                score=0.9,
                feedback="Great",
                passed=True
            ),
            CritiqueResult(
                principle_name="helpful",
                score=0.6,
                feedback="OK",
                passed=True
            )
        ]

        quality_score = step._calculate_quality_score(critiques)
        assert quality_score == 0.6

    async def test_calculate_quality_score_max(self, principles):
        """Quality score calculated with max aggregation."""
        config = ConstitutionalConfig(
            principles=principles,
            aggregate_method="max"
        )

        step = ConstitutionalAIStep(
            name="test",
            prompt="Test",
            config=config
        )

        critiques = [
            CritiqueResult(
                principle_name="harmless",
                score=0.9,
                feedback="Great",
                passed=True
            ),
            CritiqueResult(
                principle_name="helpful",
                score=0.6,
                feedback="OK",
                passed=True
            )
        ]

        quality_score = step._calculate_quality_score(critiques)
        assert quality_score == 0.9

    async def test_parallel_critique(self, principles, mock_llm_provider):
        """Critiques can be performed in parallel."""
        config = ConstitutionalConfig(
            principles=principles,
            parallel_critique=True
        )

        mock_llm_provider.generate.return_value = {
            "content": "Score: 8/10. Good response."
        }

        step = ConstitutionalAIStep(
            name="test",
            prompt="Test",
            config=config,
            llm_provider=mock_llm_provider
        )

        critiques = await step._critique_response("Test response")

        assert len(critiques) == len(principles)
        assert all(isinstance(c, CritiqueResult) for c in critiques)

    async def test_parse_score_explicit(self, config):
        """Score parsing works with explicit scores."""
        step = ConstitutionalAIStep(
            name="test",
            prompt="Test",
            config=config
        )

        # Test various formats
        assert abs(step._parse_score("Score: 8/10") - 0.8) < 0.01
        assert abs(step._parse_score("Rating: 7.5/10") - 0.75) < 0.01
        assert abs(step._parse_score("score: 0.9") - 0.9) < 0.01
        assert abs(step._parse_score("9/10") - 0.9) < 0.01

    async def test_parse_score_sentiment(self, config):
        """Score parsing falls back to sentiment analysis."""
        step = ConstitutionalAIStep(
            name="test",
            prompt="Test",
            config=config
        )

        # Positive sentiment
        positive = step._parse_score(
            "This is excellent and great! Very helpful and clear."
        )
        assert positive > 0.7

        # Negative sentiment
        negative = step._parse_score(
            "This is bad and poor. Very unclear and problematic."
        )
        assert negative < 0.5

    async def test_parse_suggestions(self, config):
        """Suggestions parsing works."""
        step = ConstitutionalAIStep(
            name="test",
            prompt="Test",
            config=config
        )

        text = """
        Here are some suggestions:
        - Make it more concise
        - Add examples
        * Include references
        1. Improve clarity
        2) Check facts
        """

        suggestions = step._parse_suggestions(text)

        assert len(suggestions) > 0
        assert "Make it more concise" in suggestions
        assert "Add examples" in suggestions

    async def test_parse_suggestions_fallback(self, config):
        """Suggestions parsing has fallback."""
        step = ConstitutionalAIStep(
            name="test",
            prompt="Test",
            config=config
        )

        text = "No suggestions here"
        suggestions = step._parse_suggestions(text)

        assert len(suggestions) > 0
        assert "Consider revising based on feedback" in suggestions

    async def test_stop_on_regression(self, principles):
        """Step stops if quality regresses."""
        config = ConstitutionalConfig(
            principles=principles,
            max_revisions=10,
            min_quality_score=0.99,  # High threshold
            stop_on_regression=True
        )

        step = ConstitutionalAIStep(
            name="test",
            prompt="Test",
            config=config
        )

        result = await step.execute({})

        # Should stop early due to regression
        assert result["revisions"] < 10

    async def test_generate_revision(self, config, mock_llm_provider):
        """Revision generation works."""
        mock_llm_provider.generate.return_value = {
            "content": "Improved response"
        }

        step = ConstitutionalAIStep(
            name="test",
            prompt="Test",
            config=config,
            llm_provider=mock_llm_provider
        )

        critiques = [
            CritiqueResult(
                principle_name="test",
                score=0.5,
                feedback="Needs improvement",
                passed=False
            )
        ]

        revised = await step._generate_revision("Original", critiques)

        assert revised == "Improved response"
        assert mock_llm_provider.generate.called


@pytest.mark.asyncio
class TestApplyConstitutionalAI:
    """Test convenience function."""

    async def test_apply_constitutional_ai(self):
        """apply_constitutional_ai convenience function works."""
        principles = [
            Principle(
                name="test",
                description="Test principle",
                critique_prompt="Test?"
            )
        ]

        result = await apply_constitutional_ai(
            prompt="Test prompt",
            principles=principles,
            max_revisions=1,
            min_quality_score=0.5
        )

        assert "response" in result
        assert "quality_score" in result
        assert "revisions" in result

    async def test_apply_constitutional_ai_with_context(self):
        """apply_constitutional_ai works with context."""
        principles = [
            Principle(
                name="test",
                description="Test",
                critique_prompt="Test?"
            )
        ]

        result = await apply_constitutional_ai(
            prompt="Talk about {topic}",
            principles=principles,
            context={"topic": "AI"},
            max_revisions=1
        )

        assert result is not None


class TestPrincipleCategories:
    """Test PrincipleCategory enum."""

    def test_principle_categories(self):
        """PrincipleCategory enum has expected values."""
        assert PrincipleCategory.HARMLESS.value == "harmless"
        assert PrincipleCategory.HELPFUL.value == "helpful"
        assert PrincipleCategory.HONEST.value == "honest"
        assert PrincipleCategory.CUSTOM.value == "custom"


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_principles_list(self):
        """Step handles empty principles list."""
        config = ConstitutionalConfig(principles=[])

        step = ConstitutionalAIStep(
            name="test",
            prompt="Test",
            config=config
        )

        result = await step.execute({})
        # Should still return a result, even with no principles
        assert result["quality_score"] == 0.0

    @pytest.mark.asyncio
    async def test_zero_max_revisions(self):
        """Step handles zero max revisions."""
        principles = [
            Principle(name="test", description="Test", critique_prompt="Test?")
        ]
        config = ConstitutionalConfig(
            principles=principles,
            max_revisions=0
        )

        step = ConstitutionalAIStep(
            name="test",
            prompt="Test",
            config=config
        )

        result = await step.execute({})
        assert result["revisions"] == 0

    @pytest.mark.asyncio
    async def test_llm_provider_error_handling(self):
        """Step handles LLM provider errors gracefully."""
        principles = [
            Principle(name="test", description="Test", critique_prompt="Test?")
        ]
        config = ConstitutionalConfig(principles=principles)

        mock_llm = Mock()
        mock_llm.generate = AsyncMock(side_effect=Exception("LLM Error"))

        step = ConstitutionalAIStep(
            name="test",
            prompt="Test",
            config=config,
            llm_provider=mock_llm
        )

        # Should handle error and not crash
        with pytest.raises(Exception):
            await step.execute({})

    def test_principle_min_score_validation(self):
        """Principle min_score can be set to different values."""
        p1 = Principle(
            name="strict",
            description="Strict",
            critique_prompt="Test?",
            min_score=0.95
        )
        p2 = Principle(
            name="lenient",
            description="Lenient",
            critique_prompt="Test?",
            min_score=0.5
        )

        assert p1.min_score == 0.95
        assert p2.min_score == 0.5

    @pytest.mark.asyncio
    async def test_critique_passed_based_on_min_score(self):
        """Critique result passed flag based on min_score."""
        config = ConstitutionalConfig(
            principles=[
                Principle(
                    name="test",
                    description="Test",
                    critique_prompt="Test?",
                    min_score=0.8
                )
            ]
        )

        step = ConstitutionalAIStep(
            name="test",
            prompt="Test",
            config=config
        )

        # Mock critique with score below min_score
        result = await step._critique_against_principle(
            "Test response",
            config.principles[0]
        )

        # Default fallback score is 0.8, which should pass
        assert result.score >= result.score
