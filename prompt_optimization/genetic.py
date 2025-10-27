"""
Genetic algorithm optimizer for prompt optimization.
"""

import asyncio
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .optimizer import OptimizationStrategy, PromptOptimizer, OptimizationResult
from .evaluators import PromptEvaluator


@dataclass
class GeneticConfig:
    """Configuration for genetic algorithm optimization."""

    population_size: int = 20
    elite_size: int = 4
    mutation_rate: float = 0.2
    crossover_rate: float = 0.7
    tournament_size: int = 3
    max_generations: int = 50
    convergence_threshold: float = 0.001

    def validate(self):
        """Validate configuration parameters."""
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if self.elite_size >= self.population_size:
            raise ValueError("elite_size must be less than population_size")
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError("mutation_rate must be between 0 and 1")
        if not 0 <= self.crossover_rate <= 1:
            raise ValueError("crossover_rate must be between 0 and 1")
        if self.tournament_size < 1:
            raise ValueError("tournament_size must be at least 1")


class GeneticOptimizer(PromptOptimizer):
    """
    Genetic algorithm optimizer for prompt optimization.

    Uses evolutionary strategies including:
    - Tournament selection
    - Crossover (single-point and uniform)
    - Mutation with configurable rate
    - Elitism to preserve best solutions
    """

    def __init__(
        self,
        evaluator: PromptEvaluator,
        config: Optional[GeneticConfig] = None,
        mutation_fn: Optional[Callable[[str], str]] = None,
        crossover_fn: Optional[Callable[[str, str], Tuple[str, str]]] = None,
        seed_prompts: Optional[List[str]] = None,
        verbose: bool = False,
    ):
        """
        Initialize genetic optimizer.

        Args:
            evaluator: Evaluator to score prompts
            config: Genetic algorithm configuration
            mutation_fn: Custom mutation function
            crossover_fn: Custom crossover function
            seed_prompts: Initial population of prompts
            verbose: Print progress
        """
        self.config = config or GeneticConfig()
        self.config.validate()

        super().__init__(
            evaluator,
            max_iterations=self.config.max_generations,
            convergence_threshold=self.config.convergence_threshold,
            verbose=verbose,
        )

        self.mutation_fn = mutation_fn or self._default_mutation
        self.crossover_fn = crossover_fn or self._default_crossover
        self.seed_prompts = seed_prompts or []

        self.population: List[Tuple[str, float]] = []

    def get_strategy(self) -> OptimizationStrategy:
        """Get the optimization strategy."""
        return OptimizationStrategy.GENETIC

    def _default_mutation(self, prompt: str) -> str:
        """
        Default mutation strategy.

        Applies simple text mutations like word replacement, insertion, deletion.

        Args:
            prompt: Prompt to mutate

        Returns:
            Mutated prompt
        """
        words = prompt.split()
        if not words:
            return prompt

        mutation_type = random.choice(["replace", "insert", "delete", "swap"])

        if mutation_type == "replace" and words:
            # Replace a random word with a synonym or variation
            idx = random.randint(0, len(words) - 1)
            replacements = {
                "the": "a",
                "is": "was",
                "and": "or",
                "but": "however",
                "very": "extremely",
            }
            words[idx] = replacements.get(words[idx].lower(), words[idx])

        elif mutation_type == "insert":
            # Insert a random word
            insertions = ["please", "carefully", "thoroughly", "precisely"]
            idx = random.randint(0, len(words))
            words.insert(idx, random.choice(insertions))

        elif mutation_type == "delete" and len(words) > 1:
            # Delete a random word
            idx = random.randint(0, len(words) - 1)
            words.pop(idx)

        elif mutation_type == "swap" and len(words) > 1:
            # Swap two adjacent words
            idx = random.randint(0, len(words) - 2)
            words[idx], words[idx + 1] = words[idx + 1], words[idx]

        return " ".join(words)

    def _default_crossover(self, parent1: str, parent2: str) -> Tuple[str, str]:
        """
        Default crossover strategy.

        Performs single-point crossover on word level.

        Args:
            parent1: First parent prompt
            parent2: Second parent prompt

        Returns:
            Tuple of two offspring prompts
        """
        words1 = parent1.split()
        words2 = parent2.split()

        if not words1 or not words2:
            return parent1, parent2

        # Single-point crossover
        point1 = random.randint(0, len(words1))
        point2 = random.randint(0, len(words2))

        offspring1 = words1[:point1] + words2[point2:]
        offspring2 = words2[:point2] + words1[point1:]

        return " ".join(offspring1), " ".join(offspring2)

    def _tournament_selection(self, population: List[Tuple[str, float]]) -> str:
        """
        Select a prompt using tournament selection.

        Args:
            population: List of (prompt, score) tuples

        Returns:
            Selected prompt
        """
        tournament = random.sample(population, min(self.config.tournament_size, len(population)))
        winner = max(tournament, key=lambda x: x[1])
        return winner[0]

    async def _initialize_population(
        self,
        initial_prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Initialize the population with seed prompts and mutations.

        Args:
            initial_prompt: Starting prompt
            context: Evaluation context

        Returns:
            List of (prompt, score) tuples
        """
        prompts = [initial_prompt]

        # Add seed prompts
        prompts.extend(self.seed_prompts[:self.config.population_size - 1])

        # Fill remaining with mutations
        while len(prompts) < self.config.population_size:
            base = random.choice([initial_prompt] + self.seed_prompts)
            prompts.append(self.mutation_fn(base))

        # Evaluate all prompts
        scores = await self.evaluate_batch(prompts, context)

        population = list(zip(prompts, scores))

        # Track initial population
        for prompt, score in population:
            self.track_candidate(prompt, score, {"type": "initial"})

        return population

    async def _evolve_generation(
        self,
        population: List[Tuple[str, float]],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Evolve one generation of the population.

        Args:
            population: Current population
            context: Evaluation context

        Returns:
            New population
        """
        # Sort by fitness (score)
        population = sorted(population, key=lambda x: x[1], reverse=True)

        # Elitism: keep top performers
        new_population = population[:self.config.elite_size]
        new_prompts = []

        # Generate offspring
        while len(new_population) + len(new_prompts) < self.config.population_size:
            # Selection
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)

            # Crossover
            if random.random() < self.config.crossover_rate:
                child1, child2 = self.crossover_fn(parent1, parent2)
            else:
                child1, child2 = parent1, parent2

            # Mutation
            if random.random() < self.config.mutation_rate:
                child1 = self.mutation_fn(child1)
            if random.random() < self.config.mutation_rate:
                child2 = self.mutation_fn(child2)

            new_prompts.extend([child1, child2])

        # Trim to population size
        new_prompts = new_prompts[:self.config.population_size - len(new_population)]

        # Evaluate new prompts
        if new_prompts:
            scores = await self.evaluate_batch(new_prompts, context)
            new_population.extend(zip(new_prompts, scores))

            # Track new candidates
            for prompt, score in zip(new_prompts, scores):
                self.track_candidate(prompt, score, {"type": "offspring"})

        return new_population

    async def optimize(
        self,
        initial_prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        """
        Optimize prompt using genetic algorithm.

        Args:
            initial_prompt: Starting prompt
            context: Additional context

        Returns:
            OptimizationResult
        """
        self.reset()

        if self.verbose:
            print(f"Initializing population of {self.config.population_size}...")

        # Initialize population
        self.population = await self._initialize_population(initial_prompt, context)

        # Evolution loop
        for generation in range(self.config.max_generations):
            self.current_iteration = generation + 1

            if self.verbose:
                best_score = max(p[1] for p in self.population)
                print(f"Generation {generation + 1}: best score = {best_score:.4f}")

            # Evolve
            self.population = await self._evolve_generation(self.population, context)

            # Check convergence
            if self.has_converged(window_size=5):
                if self.verbose:
                    print(f"Converged at generation {self.current_iteration}")
                break

        result = self.get_optimization_result()
        result.metadata["final_population_size"] = len(self.population)
        result.metadata["config"] = {
            "population_size": self.config.population_size,
            "elite_size": self.config.elite_size,
            "mutation_rate": self.config.mutation_rate,
            "crossover_rate": self.config.crossover_rate,
        }

        return result

    def get_population_diversity(self) -> float:
        """
        Calculate population diversity metric.

        Returns:
            Diversity score (0-1, higher is more diverse)
        """
        if len(self.population) < 2:
            return 0.0

        unique_prompts = len(set(p[0] for p in self.population))
        return unique_prompts / len(self.population)

    def get_fitness_statistics(self) -> Dict[str, float]:
        """
        Get statistics about population fitness.

        Returns:
            Dictionary with mean, min, max, std fitness
        """
        if not self.population:
            return {}

        scores = [p[1] for p in self.population]

        import statistics

        return {
            "mean": statistics.mean(scores),
            "min": min(scores),
            "max": max(scores),
            "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        }
