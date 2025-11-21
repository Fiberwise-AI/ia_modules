"""
Unit tests for prompt optimization strategies.

Tests genetic algorithms, A/B testing, and reinforcement learning.
"""

import pytest
from ia_modules.pipeline.test_utils import create_test_execution_context
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import random

from ia_modules.prompt_optimization.genetic import (
    GeneticOptimizer,
    GeneticConfig
)
from ia_modules.prompt_optimization.ab_testing import (
    ABTester,
    ABTestConfig,
    PromptVariant
)
from ia_modules.prompt_optimization.optimizer import (
    OptimizationStrategy,
    OptimizationResult
)
from ia_modules.prompt_optimization.evaluators import PromptEvaluator


# Mock evaluator for testing
class MockEvaluator(PromptEvaluator):
    """Mock evaluator that returns deterministic scores."""

    def __init__(self, score=0.8):
        self.score = score
        self.eval_count = 0

    async def evaluate(self, prompt: str, test_cases: list = None) -> float:
        """Return mock score."""
        self.eval_count += 1
        await asyncio.sleep(0.001)  # Simulate evaluation time
        # Simple scoring: longer prompts get slightly higher scores
        base_score = self.score
        length_bonus = min(len(prompt) / 1000, 0.1)
        return min(1.0, base_score + length_bonus)


class TestGeneticConfig:
    """Test GeneticConfig dataclass."""

    def test_creation_defaults(self):
        """GeneticConfig has proper defaults."""
        config = GeneticConfig()

        assert config.population_size == 20
        assert config.elite_size == 4
        assert config.mutation_rate == 0.2
        assert config.crossover_rate == 0.7
        assert config.tournament_size == 3
        assert config.max_generations == 50
        assert config.convergence_threshold == 0.001

    def test_creation_custom(self):
        """GeneticConfig can be customized."""
        config = GeneticConfig(
            population_size=50,
            elite_size=10,
            mutation_rate=0.3,
            max_generations=100
        )

        assert config.population_size == 50
        assert config.elite_size == 10
        assert config.mutation_rate == 0.3
        assert config.max_generations == 100

    def test_validate_success(self):
        """Valid config passes validation."""
        config = GeneticConfig()
        config.validate()  # Should not raise

    def test_validate_population_too_small(self):
        """Validation fails with population too small."""
        config = GeneticConfig(population_size=1)

        with pytest.raises(ValueError, match="population_size"):
            config.validate()

    def test_validate_elite_too_large(self):
        """Validation fails with elite size too large."""
        config = GeneticConfig(population_size=10, elite_size=10)

        with pytest.raises(ValueError, match="elite_size"):
            config.validate()

    def test_validate_mutation_rate_invalid(self):
        """Validation fails with invalid mutation rate."""
        config = GeneticConfig(mutation_rate=1.5)

        with pytest.raises(ValueError, match="mutation_rate"):
            config.validate()

    def test_validate_crossover_rate_invalid(self):
        """Validation fails with invalid crossover rate."""
        config = GeneticConfig(crossover_rate=-0.1)

        with pytest.raises(ValueError, match="crossover_rate"):
            config.validate()


class TestGeneticOptimizer:
    """Test GeneticOptimizer functionality."""

    @pytest.fixture
    def evaluator(self):
        """Create mock evaluator."""
        return MockEvaluator(score=0.7)

    @pytest.fixture
    def optimizer(self, evaluator):
        """Create genetic optimizer."""
        config = GeneticConfig(
            population_size=10,
            max_generations=5,
            elite_size=2
        )
        return GeneticOptimizer(
            evaluator=evaluator,
            config=config,
            seed_prompts=["Initial prompt 1", "Initial prompt 2"]
        )

    def test_creation(self, optimizer):
        """GeneticOptimizer can be created."""
        assert optimizer.config.population_size == 10
        assert len(optimizer.seed_prompts) == 2

    def test_get_strategy(self, optimizer):
        """Returns correct optimization strategy."""
        assert optimizer.get_strategy() == OptimizationStrategy.GENETIC

    @pytest.mark.asyncio
    async def test_optimize(self, optimizer):
        """Can run genetic optimization."""
        result = await optimizer.optimize("Base prompt")

        assert isinstance(result, OptimizationResult)
        assert result.best_prompt is not None
        assert result.best_score > 0
        assert result.iterations > 0

    @pytest.mark.asyncio
    async def test_default_mutation(self, optimizer):
        """Default mutation changes prompt."""
        original = "The quick brown fox jumps over the lazy dog"

        # Run multiple times to see variation
        mutations = set()
        for _ in range(10):
            mutated = optimizer._default_mutation(original)
            mutations.add(mutated)

        # Should have some variation
        assert len(mutations) > 1

    @pytest.mark.asyncio
    async def test_default_crossover(self, optimizer):
        """Default crossover creates offspring."""
        parent1 = "This is the first parent prompt"
        parent2 = "This is the second parent prompt"

        child1, child2 = optimizer._default_crossover(parent1, parent2)

        assert isinstance(child1, str)
        assert isinstance(child2, str)
        # Children should be different from parents (usually)
        assert child1 != parent1 or child2 != parent2

    @pytest.mark.asyncio
    async def test_custom_mutation_function(self, evaluator):
        """Can use custom mutation function."""
        def custom_mutation(prompt):
            return prompt + " [mutated]"

        config = GeneticConfig(population_size=5, max_generations=2)
        optimizer = GeneticOptimizer(
            evaluator=evaluator,
            config=config,
            mutation_fn=custom_mutation
        )

        mutated = optimizer.mutation_fn("test")
        assert "[mutated]" in mutated

    @pytest.mark.asyncio
    async def test_custom_crossover_function(self, evaluator):
        """Can use custom crossover function."""
        def custom_crossover(p1, p2):
            return (f"{p1}+{p2}", f"{p2}+{p1}")

        config = GeneticConfig(population_size=5, max_generations=2)
        optimizer = GeneticOptimizer(
            evaluator=evaluator,
            config=config,
            crossover_fn=custom_crossover
        )

        c1, c2 = optimizer.crossover_fn("A", "B")
        assert c1 == "A+B"
        assert c2 == "B+A"

    @pytest.mark.asyncio
    async def test_convergence(self, evaluator):
        """Optimizer converges when improvement plateaus."""
        config = GeneticConfig(
            population_size=5,
            max_generations=100,
            convergence_threshold=0.001
        )

        optimizer = GeneticOptimizer(evaluator=evaluator, config=config)

        result = await optimizer.optimize("Test prompt")

        # Should converge before max generations
        assert result.iterations < 100


class TestABTestConfig:
    """Test ABTestConfig dataclass."""

    def test_creation_defaults(self):
        """ABTestConfig has proper defaults."""
        config = ABTestConfig()

        assert config.min_samples == 30
        assert config.confidence_level == 0.95
        assert config.max_iterations == 1000
        assert config.convergence_threshold == 0.001
        assert config.explore_probability == 0.1
        assert config.ucb_exploration_factor == 2.0

    def test_validate_success(self):
        """Valid config passes validation."""
        config = ABTestConfig()
        config.validate()  # Should not raise

    def test_validate_min_samples_invalid(self):
        """Validation fails with invalid min_samples."""
        config = ABTestConfig(min_samples=1)

        with pytest.raises(ValueError, match="min_samples"):
            config.validate()

    def test_validate_confidence_level_invalid(self):
        """Validation fails with invalid confidence level."""
        config = ABTestConfig(confidence_level=1.5)

        with pytest.raises(ValueError, match="confidence_level"):
            config.validate()


class TestPromptVariant:
    """Test PromptVariant dataclass."""

    def test_creation(self):
        """PromptVariant can be created."""
        variant = PromptVariant(
            name="variant_a",
            prompt="Test prompt A"
        )

        assert variant.name == "variant_a"
        assert variant.prompt == "Test prompt A"
        assert len(variant.samples) == 0
        assert variant.total_reward == 0.0
        assert variant.num_selections == 0

    def test_mean_reward(self):
        """Can calculate mean reward."""
        variant = PromptVariant(name="test", prompt="test")
        variant.samples = [0.8, 0.9, 0.7]

        mean = variant.mean_reward

        assert abs(mean - 0.8) < 0.01

    def test_mean_reward_empty(self):
        """Mean reward is 0 for empty samples."""
        variant = PromptVariant(name="test", prompt="test")

        assert variant.mean_reward == 0.0

    def test_variance(self):
        """Can calculate variance."""
        variant = PromptVariant(name="test", prompt="test")
        variant.samples = [1.0, 2.0, 3.0]

        variance = variant.variance

        assert variance > 0

    def test_variance_insufficient_samples(self):
        """Variance is 0 with insufficient samples."""
        variant = PromptVariant(name="test", prompt="test")
        variant.samples = [1.0]

        assert variant.variance == 0.0

    def test_std_dev(self):
        """Can calculate standard deviation."""
        variant = PromptVariant(name="test", prompt="test")
        variant.samples = [1.0, 2.0, 3.0]

        std_dev = variant.std_dev

        assert std_dev > 0

    def test_add_sample(self):
        """Can add sample to variant."""
        variant = PromptVariant(name="test", prompt="test")

        variant.add_sample(0.8)

        assert len(variant.samples) == 1
        assert variant.samples[0] == 0.8
        assert variant.total_reward == 0.8
        assert variant.num_selections == 1

    def test_to_dict(self):
        """Can convert variant to dictionary."""
        variant = PromptVariant(name="test", prompt="test prompt")
        variant.add_sample(0.8)
        variant.add_sample(0.9)

        data = variant.to_dict()

        assert data["name"] == "test"
        assert data["prompt"] == "test prompt"
        assert data["num_samples"] == 2
        assert "mean_reward" in data
        assert "std_dev" in data


class TestABTester:
    """Test ABTester functionality."""

    @pytest.fixture
    def evaluator(self):
        """Create mock evaluator."""
        return MockEvaluator(score=0.7)

    @pytest.fixture
    def variants(self):
        """Create test variants."""
        return [
            ("variant_a", "Prompt variant A"),
            ("variant_b", "Prompt variant B"),
            ("variant_c", "Prompt variant C")
        ]

    @pytest.fixture
    def ab_tester(self, evaluator, variants):
        """Create AB tester."""
        config = ABTestConfig(
            min_samples=5,
            max_iterations=20
        )
        return ABTester(
            evaluator=evaluator,
            variants=variants,
            config=config,
            selection_strategy="ucb"
        )

    def test_creation(self, ab_tester):
        """ABTester can be created."""
        assert len(ab_tester.variants) == 3
        assert ab_tester.selection_strategy == "ucb"

    def test_creation_insufficient_variants(self, evaluator):
        """Creating with < 2 variants raises error."""
        with pytest.raises(ValueError, match="At least 2 variants"):
            ABTester(
                evaluator=evaluator,
                variants=[("only_one", "prompt")],
                config=ABTestConfig()
            )

    def test_creation_invalid_strategy(self, evaluator, variants):
        """Creating with invalid strategy raises error."""
        with pytest.raises(ValueError, match="selection_strategy"):
            ABTester(
                evaluator=evaluator,
                variants=variants,
                selection_strategy="invalid"
            )

    def test_get_strategy(self, ab_tester):
        """Returns correct optimization strategy."""
        assert ab_tester.get_strategy() == OptimizationStrategy.AB_TESTING

    @pytest.mark.asyncio
    async def test_optimize(self, ab_tester):
        """Can run A/B test optimization."""
        result = await ab_tester.optimize("Base prompt")

        assert isinstance(result, OptimizationResult)
        assert result.best_prompt is not None
        assert result.best_score > 0
        assert result.iterations > 0

    @pytest.mark.asyncio
    async def test_all_variants_tested(self, ab_tester):
        """All variants get tested."""
        await ab_tester.optimize("Base")

        # All variants should have at least one sample
        for variant in ab_tester.variants:
            assert len(variant.samples) > 0

    @pytest.mark.asyncio
    async def test_epsilon_greedy_strategy(self, evaluator, variants):
        """Epsilon-greedy strategy works."""
        config = ABTestConfig(min_samples=3, max_iterations=10)
        tester = ABTester(
            evaluator=evaluator,
            variants=variants,
            config=config,
            selection_strategy="epsilon_greedy"
        )

        result = await tester.optimize("Base")

        assert result is not None

    @pytest.mark.asyncio
    async def test_thompson_sampling_strategy(self, evaluator, variants):
        """Thompson sampling strategy works."""
        config = ABTestConfig(min_samples=3, max_iterations=10)
        tester = ABTester(
            evaluator=evaluator,
            variants=variants,
            config=config,
            selection_strategy="thompson"
        )

        result = await tester.optimize("Base")

        assert result is not None


class TestOptimizationResult:
    """Test OptimizationResult dataclass."""

    def test_creation(self):
        """OptimizationResult can be created."""
        result = OptimizationResult(
            best_prompt="Optimized prompt",
            best_score=0.95,
            iterations=10,
            strategy=OptimizationStrategy.GENETIC,
            history=[0.7, 0.8, 0.9, 0.95]
        )

        assert result.best_prompt == "Optimized prompt"
        assert result.best_score == 0.95
        assert result.iterations == 10
        assert result.strategy == OptimizationStrategy.GENETIC
        assert len(result.history) == 4


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_genetic_empty_prompt(self):
        """Genetic optimizer handles empty prompt."""
        evaluator = MockEvaluator()
        config = GeneticConfig(population_size=5, max_generations=2)
        optimizer = GeneticOptimizer(evaluator=evaluator, config=config)

        result = await optimizer.optimize("")

        assert result is not None

    @pytest.mark.asyncio
    async def test_genetic_very_long_prompt(self):
        """Genetic optimizer handles very long prompt."""
        evaluator = MockEvaluator()
        config = GeneticConfig(population_size=5, max_generations=2)
        optimizer = GeneticOptimizer(evaluator=evaluator, config=config)

        long_prompt = "word " * 1000

        result = await optimizer.optimize(long_prompt)

        assert result is not None

    def test_prompt_variant_edge_cases(self):
        """PromptVariant handles edge cases."""
        variant = PromptVariant(name="test", prompt="test")

        # Empty samples
        assert variant.mean_reward == 0.0
        assert variant.variance == 0.0
        assert variant.std_dev == 0.0

        # Single sample
        variant.add_sample(1.0)
        assert variant.mean_reward == 1.0
        assert variant.variance == 0.0


class TestIntegration:
    """Integration tests for prompt optimization."""

    @pytest.mark.asyncio
    async def test_compare_strategies(self):
        """Compare different optimization strategies."""
        evaluator = MockEvaluator(score=0.7)
        base_prompt = "Solve this problem"

        # Genetic
        genetic_config = GeneticConfig(
            population_size=10,
            max_generations=10
        )
        genetic_opt = GeneticOptimizer(
            evaluator=evaluator,
            config=genetic_config
        )
        genetic_result = await genetic_opt.optimize(base_prompt)

        # A/B Testing
        ab_config = ABTestConfig(min_samples=5, max_iterations=20)
        ab_tester = ABTester(
            evaluator=evaluator,
            variants=[
                ("a", "Solve this problem carefully"),
                ("b", "Solve this problem step by step"),
                ("c", "Solve this problem thoroughly")
            ],
            config=ab_config
        )
        ab_result = await ab_tester.optimize(base_prompt)

        # Both should produce results
        assert genetic_result.best_score > 0
        assert ab_result.best_score > 0

    @pytest.mark.asyncio
    async def test_optimization_with_variance(self):
        """Test optimization with varying evaluator scores."""
        class VariableEvaluator(PromptEvaluator):
            async def evaluate(self, prompt, test_cases=None):
                # Simulate variable scores
                await asyncio.sleep(0.001)
                base = 0.7
                variance = random.uniform(-0.1, 0.1)
                return max(0, min(1.0, base + variance))

        evaluator = VariableEvaluator()
        config = GeneticConfig(population_size=5, max_generations=5)
        optimizer = GeneticOptimizer(evaluator=evaluator, config=config)

        result = await optimizer.optimize("Test prompt")

        assert result is not None
        assert 0 <= result.best_score <= 1.0
