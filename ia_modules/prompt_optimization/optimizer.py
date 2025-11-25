"""
Base prompt optimizer with optimization strategies.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .evaluators import PromptEvaluator


class OptimizationStrategy(Enum):
    """Available optimization strategies."""

    GENETIC = "genetic"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    AB_TESTING = "ab_testing"
    RANDOM_SEARCH = "random_search"
    GRID_SEARCH = "grid_search"


@dataclass
class OptimizationResult:
    """Result of a prompt optimization run."""

    best_prompt: str
    best_score: float
    history: List[Dict[str, Any]]
    iterations: int
    strategy: OptimizationStrategy
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "best_prompt": self.best_prompt,
            "best_score": self.best_score,
            "history": self.history,
            "iterations": self.iterations,
            "strategy": self.strategy.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class PromptCandidate:
    """A candidate prompt with its evaluation score."""

    prompt: str
    score: float
    iteration: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert candidate to dictionary."""
        return {
            "prompt": self.prompt,
            "score": self.score,
            "iteration": self.iteration,
            "metadata": self.metadata,
        }


class PromptOptimizer(ABC):
    """
    Base class for prompt optimizers.

    Provides common functionality for tracking optimization history,
    managing evaluators, and coordinating optimization runs.
    """

    def __init__(
        self,
        evaluator: PromptEvaluator,
        max_iterations: int = 100,
        convergence_threshold: float = 0.001,
        verbose: bool = False,
    ):
        """
        Initialize the prompt optimizer.

        Args:
            evaluator: Evaluator to score prompt candidates
            max_iterations: Maximum number of optimization iterations
            convergence_threshold: Score improvement threshold for convergence
            verbose: Whether to print optimization progress
        """
        self.evaluator = evaluator
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.verbose = verbose

        self.history: List[PromptCandidate] = []
        self.best_candidate: Optional[PromptCandidate] = None
        self.current_iteration: int = 0

    @abstractmethod
    async def optimize(
        self,
        initial_prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        """
        Optimize the prompt using the implemented strategy.

        Args:
            initial_prompt: Starting prompt to optimize
            context: Additional context for optimization

        Returns:
            OptimizationResult with best prompt and optimization history
        """
        pass

    @abstractmethod
    def get_strategy(self) -> OptimizationStrategy:
        """Get the optimization strategy used by this optimizer."""
        pass

    async def evaluate_prompt(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Evaluate a prompt using the configured evaluator.

        Args:
            prompt: Prompt to evaluate
            context: Additional context for evaluation

        Returns:
            Evaluation score (higher is better)
        """
        return await self.evaluator.evaluate(prompt, context or {})

    async def evaluate_batch(
        self,
        prompts: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[float]:
        """
        Evaluate multiple prompts concurrently.

        Args:
            prompts: List of prompts to evaluate
            context: Additional context for evaluation

        Returns:
            List of evaluation scores
        """
        tasks = [
            self.evaluate_prompt(prompt, context)
            for prompt in prompts
        ]
        return await asyncio.gather(*tasks)

    def track_candidate(
        self,
        prompt: str,
        score: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PromptCandidate:
        """
        Track a prompt candidate in optimization history.

        Args:
            prompt: Candidate prompt
            score: Evaluation score
            metadata: Additional metadata

        Returns:
            PromptCandidate object
        """
        candidate = PromptCandidate(
            prompt=prompt,
            score=score,
            iteration=self.current_iteration,
            metadata=metadata or {},
        )

        self.history.append(candidate)

        if self.best_candidate is None or score > self.best_candidate.score:
            self.best_candidate = candidate
            if self.verbose:
                print(f"New best score: {score:.4f} at iteration {self.current_iteration}")

        return candidate

    def has_converged(self, window_size: int = 10) -> bool:
        """
        Check if optimization has converged.

        Args:
            window_size: Number of recent iterations to check

        Returns:
            True if optimization has converged
        """
        if len(self.history) < window_size:
            return False

        recent_scores = [c.score for c in self.history[-window_size:]]
        score_range = max(recent_scores) - min(recent_scores)

        return score_range < self.convergence_threshold

    def get_optimization_result(self) -> OptimizationResult:
        """
        Create an OptimizationResult from current state.

        Returns:
            OptimizationResult object
        """
        if self.best_candidate is None:
            raise ValueError("No candidates evaluated yet")

        return OptimizationResult(
            best_prompt=self.best_candidate.prompt,
            best_score=self.best_candidate.score,
            history=[c.to_dict() for c in self.history],
            iterations=self.current_iteration,
            strategy=self.get_strategy(),
        )

    def reset(self):
        """Reset optimizer state for a new optimization run."""
        self.history = []
        self.best_candidate = None
        self.current_iteration = 0


class RandomSearchOptimizer(PromptOptimizer):
    """
    Simple random search optimizer.

    Generates random variations of the prompt and selects the best one.
    """

    def __init__(
        self,
        evaluator: PromptEvaluator,
        variation_fn: Callable[[str], str],
        max_iterations: int = 100,
        convergence_threshold: float = 0.001,
        verbose: bool = False,
    ):
        """
        Initialize random search optimizer.

        Args:
            evaluator: Evaluator to score prompts
            variation_fn: Function to generate prompt variations
            max_iterations: Maximum iterations
            convergence_threshold: Convergence threshold
            verbose: Print progress
        """
        super().__init__(evaluator, max_iterations, convergence_threshold, verbose)
        self.variation_fn = variation_fn

    def get_strategy(self) -> OptimizationStrategy:
        """Get the optimization strategy."""
        return OptimizationStrategy.RANDOM_SEARCH

    async def optimize(
        self,
        initial_prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        """
        Optimize using random search.

        Args:
            initial_prompt: Starting prompt
            context: Additional context

        Returns:
            OptimizationResult
        """
        self.reset()

        # Evaluate initial prompt
        initial_score = await self.evaluate_prompt(initial_prompt, context)
        self.track_candidate(initial_prompt, initial_score, {"type": "initial"})

        for iteration in range(self.max_iterations):
            self.current_iteration = iteration + 1

            # Generate variation
            variation = self.variation_fn(initial_prompt)
            score = await self.evaluate_prompt(variation, context)
            self.track_candidate(variation, score, {"type": "variation"})

            # Check convergence
            if self.has_converged():
                if self.verbose:
                    print(f"Converged at iteration {self.current_iteration}")
                break

        return self.get_optimization_result()


class GridSearchOptimizer(PromptOptimizer):
    """
    Grid search optimizer for systematic exploration.

    Evaluates prompts on a predefined grid of parameter values.
    """

    def __init__(
        self,
        evaluator: PromptEvaluator,
        parameter_grid: Dict[str, List[Any]],
        template_fn: Callable[[Dict[str, Any]], str],
        verbose: bool = False,
    ):
        """
        Initialize grid search optimizer.

        Args:
            evaluator: Evaluator to score prompts
            parameter_grid: Dictionary mapping parameter names to value lists
            template_fn: Function to generate prompt from parameters
            verbose: Print progress
        """
        super().__init__(evaluator, max_iterations=1, verbose=verbose)
        self.parameter_grid = parameter_grid
        self.template_fn = template_fn

    def get_strategy(self) -> OptimizationStrategy:
        """Get the optimization strategy."""
        return OptimizationStrategy.GRID_SEARCH

    def _generate_grid_points(self) -> List[Dict[str, Any]]:
        """Generate all combinations of parameter values."""
        import itertools

        keys = list(self.parameter_grid.keys())
        values = list(self.parameter_grid.values())

        combinations = list(itertools.product(*values))

        return [
            dict(zip(keys, combo))
            for combo in combinations
        ]

    async def optimize(
        self,
        initial_prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        """
        Optimize using grid search.

        Args:
            initial_prompt: Ignored (using parameter grid instead)
            context: Additional context

        Returns:
            OptimizationResult
        """
        self.reset()

        grid_points = self._generate_grid_points()
        self.max_iterations = len(grid_points)

        if self.verbose:
            print(f"Evaluating {len(grid_points)} grid points...")

        for i, params in enumerate(grid_points):
            self.current_iteration = i + 1

            # Generate prompt from parameters
            prompt = self.template_fn(params)
            score = await self.evaluate_prompt(prompt, context)
            self.track_candidate(prompt, score, {"parameters": params})

        return self.get_optimization_result()
