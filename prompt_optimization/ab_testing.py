"""
A/B testing optimizer with multi-armed bandit algorithms.
"""

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import math

from .optimizer import OptimizationStrategy, PromptOptimizer, OptimizationResult
from .evaluators import PromptEvaluator


@dataclass
class ABTestConfig:
    """Configuration for A/B testing."""

    min_samples: int = 30
    confidence_level: float = 0.95
    max_iterations: int = 1000
    convergence_threshold: float = 0.001
    explore_probability: float = 0.1
    ucb_exploration_factor: float = 2.0

    def validate(self):
        """Validate configuration parameters."""
        if self.min_samples < 2:
            raise ValueError("min_samples must be at least 2")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")
        if not 0 <= self.explore_probability <= 1:
            raise ValueError("explore_probability must be between 0 and 1")


@dataclass
class PromptVariant:
    """A prompt variant for A/B testing."""

    name: str
    prompt: str
    samples: List[float] = field(default_factory=list)
    total_reward: float = 0.0
    num_selections: int = 0

    @property
    def mean_reward(self) -> float:
        """Get mean reward for this variant."""
        if not self.samples:
            return 0.0
        return sum(self.samples) / len(self.samples)

    @property
    def variance(self) -> float:
        """Get variance of rewards."""
        if len(self.samples) < 2:
            return 0.0
        mean = self.mean_reward
        return sum((x - mean) ** 2 for x in self.samples) / len(self.samples)

    @property
    def std_dev(self) -> float:
        """Get standard deviation of rewards."""
        return math.sqrt(self.variance)

    def add_sample(self, reward: float):
        """Add a reward sample."""
        self.samples.append(reward)
        self.total_reward += reward
        self.num_selections += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "prompt": self.prompt,
            "num_samples": len(self.samples),
            "mean_reward": self.mean_reward,
            "std_dev": self.std_dev,
            "num_selections": self.num_selections,
        }


class ABTester(PromptOptimizer):
    """
    A/B testing optimizer with multi-armed bandit algorithms.

    Supports multiple selection strategies:
    - Epsilon-greedy: Explore with fixed probability
    - UCB (Upper Confidence Bound): Optimistic exploration
    - Thompson Sampling: Bayesian approach
    """

    def __init__(
        self,
        evaluator: PromptEvaluator,
        variants: List[Tuple[str, str]],  # List of (name, prompt) tuples
        config: Optional[ABTestConfig] = None,
        selection_strategy: str = "ucb",
        verbose: bool = False,
    ):
        """
        Initialize A/B tester.

        Args:
            evaluator: Evaluator to score prompts
            variants: List of (name, prompt) tuples to test
            config: A/B testing configuration
            selection_strategy: Strategy for variant selection (epsilon_greedy, ucb, thompson)
            verbose: Print progress
        """
        self.config = config or ABTestConfig()
        self.config.validate()

        super().__init__(
            evaluator,
            max_iterations=self.config.max_iterations,
            convergence_threshold=self.config.convergence_threshold,
            verbose=verbose,
        )

        if len(variants) < 2:
            raise ValueError("At least 2 variants required for A/B testing")

        self.variants = [
            PromptVariant(name=name, prompt=prompt)
            for name, prompt in variants
        ]

        valid_strategies = ["epsilon_greedy", "ucb", "thompson"]
        if selection_strategy not in valid_strategies:
            raise ValueError(f"selection_strategy must be one of {valid_strategies}")
        self.selection_strategy = selection_strategy

    def get_strategy(self) -> OptimizationStrategy:
        """Get the optimization strategy."""
        return OptimizationStrategy.AB_TESTING

    def _select_variant_epsilon_greedy(self) -> PromptVariant:
        """
        Select variant using epsilon-greedy strategy.

        Returns:
            Selected variant
        """
        if random.random() < self.config.explore_probability:
            # Explore: random variant
            return random.choice(self.variants)
        else:
            # Exploit: best variant
            return max(self.variants, key=lambda v: v.mean_reward)

    def _select_variant_ucb(self, total_selections: int) -> PromptVariant:
        """
        Select variant using Upper Confidence Bound.

        Args:
            total_selections: Total number of selections across all variants

        Returns:
            Selected variant
        """
        if total_selections == 0:
            return random.choice(self.variants)

        ucb_scores = []
        for variant in self.variants:
            if variant.num_selections == 0:
                # Ensure each variant is tried at least once
                ucb_scores.append((variant, float('inf')))
            else:
                # UCB1 formula
                exploitation = variant.mean_reward
                exploration = math.sqrt(
                    self.config.ucb_exploration_factor * math.log(total_selections) / variant.num_selections
                )
                ucb_scores.append((variant, exploitation + exploration))

        return max(ucb_scores, key=lambda x: x[1])[0]

    def _select_variant_thompson(self) -> PromptVariant:
        """
        Select variant using Thompson Sampling.

        Assumes Beta distribution for rewards (scaled to 0-1).

        Returns:
            Selected variant
        """
        samples = []

        for variant in self.variants:
            if not variant.samples:
                # Prior: uniform distribution
                alpha, beta = 1, 1
            else:
                # Posterior based on observed rewards
                # Assume rewards are between 0 and 1
                successes = sum(1 for r in variant.samples if r > 0.5)
                failures = len(variant.samples) - successes
                alpha = 1 + successes
                beta = 1 + failures

            # Sample from Beta distribution
            sample = random.betavariate(alpha, beta)
            samples.append((variant, sample))

        return max(samples, key=lambda x: x[1])[0]

    def _select_variant(self, total_selections: int) -> PromptVariant:
        """
        Select variant using configured strategy.

        Args:
            total_selections: Total selections so far

        Returns:
            Selected variant
        """
        if self.selection_strategy == "epsilon_greedy":
            return self._select_variant_epsilon_greedy()
        elif self.selection_strategy == "ucb":
            return self._select_variant_ucb(total_selections)
        elif self.selection_strategy == "thompson":
            return self._select_variant_thompson()
        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")

    def _calculate_statistical_significance(
        self,
        variant_a: PromptVariant,
        variant_b: PromptVariant,
    ) -> Tuple[float, bool]:
        """
        Calculate statistical significance between two variants.

        Uses Welch's t-test for unequal variances.

        Args:
            variant_a: First variant
            variant_b: Second variant

        Returns:
            Tuple of (p_value, is_significant)
        """
        if len(variant_a.samples) < 2 or len(variant_b.samples) < 2:
            return 1.0, False

        mean_a = variant_a.mean_reward
        mean_b = variant_b.mean_reward
        var_a = variant_a.variance
        var_b = variant_b.variance
        n_a = len(variant_a.samples)
        n_b = len(variant_b.samples)

        # Welch's t-statistic
        t_stat = (mean_a - mean_b) / math.sqrt(var_a / n_a + var_b / n_b)

        # Degrees of freedom (Welch-Satterthwaite)
        df = (var_a / n_a + var_b / n_b) ** 2 / (
            (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
        )

        # Simple p-value approximation (for demonstration)
        # In production, use scipy.stats.t.sf
        p_value = 2 * (1 - self._t_cdf(abs(t_stat), df))

        is_significant = p_value < (1 - self.config.confidence_level)

        return p_value, is_significant

    def _t_cdf(self, t: float, df: float) -> float:
        """
        Approximate CDF of Student's t-distribution.

        Simple approximation - in production use scipy.stats.

        Args:
            t: t-statistic
            df: Degrees of freedom

        Returns:
            Cumulative probability
        """
        # Very rough approximation
        x = df / (df + t ** 2)
        # Using normal approximation for large df
        if df > 30:
            return 0.5 * (1 + math.erf(t / math.sqrt(2)))
        return 0.5  # Placeholder

    async def optimize(
        self,
        initial_prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        """
        Run A/B test to find best prompt variant.

        Args:
            initial_prompt: Ignored (using provided variants)
            context: Additional context

        Returns:
            OptimizationResult
        """
        self.reset()

        if self.verbose:
            print(f"Starting A/B test with {len(self.variants)} variants...")
            print(f"Selection strategy: {self.selection_strategy}")

        total_selections = 0

        for iteration in range(self.config.max_iterations):
            self.current_iteration = iteration + 1

            # Select variant
            variant = self._select_variant(total_selections)
            total_selections += 1

            # Evaluate
            score = await self.evaluate_prompt(variant.prompt, context)
            variant.add_sample(score)

            # Track
            self.track_candidate(
                variant.prompt,
                score,
                {
                    "variant": variant.name,
                    "selection_strategy": self.selection_strategy,
                }
            )

            # Check if we have enough samples
            min_samples_met = all(
                len(v.samples) >= self.config.min_samples
                for v in self.variants
            )

            if min_samples_met:
                # Check for convergence
                if self.has_converged(window_size=20):
                    if self.verbose:
                        print(f"Converged at iteration {self.current_iteration}")
                    break

        # Find best variant
        best_variant = max(self.variants, key=lambda v: v.mean_reward)

        if self.verbose:
            print(f"\nBest variant: {best_variant.name}")
            print(f"Mean reward: {best_variant.mean_reward:.4f}")
            print(f"Samples: {len(best_variant.samples)}")

        result = self.get_optimization_result()
        result.metadata["variants"] = [v.to_dict() for v in self.variants]
        result.metadata["selection_strategy"] = self.selection_strategy
        result.metadata["total_selections"] = total_selections

        # Add statistical significance results
        if len(self.variants) == 2:
            p_value, is_significant = self._calculate_statistical_significance(
                self.variants[0], self.variants[1]
            )
            result.metadata["statistical_test"] = {
                "p_value": p_value,
                "is_significant": is_significant,
                "confidence_level": self.config.confidence_level,
            }

        return result

    def get_variant_statistics(self) -> List[Dict[str, Any]]:
        """
        Get detailed statistics for all variants.

        Returns:
            List of variant statistics dictionaries
        """
        return [v.to_dict() for v in self.variants]

    def get_winner(self, min_confidence: float = 0.95) -> Optional[PromptVariant]:
        """
        Get the winning variant if statistically significant.

        Args:
            min_confidence: Minimum confidence level

        Returns:
            Winning variant or None if no clear winner
        """
        if len(self.variants) != 2:
            # For multiple variants, return best performing
            return max(self.variants, key=lambda v: v.mean_reward)

        # For two variants, check statistical significance
        p_value, is_significant = self._calculate_statistical_significance(
            self.variants[0], self.variants[1]
        )

        if is_significant:
            return max(self.variants, key=lambda v: v.mean_reward)

        return None
