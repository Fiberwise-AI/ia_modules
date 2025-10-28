"""
Prompt evaluation metrics and evaluators.
"""

import asyncio
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    score: float
    metrics: Dict[str, float]
    feedback: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "metrics": self.metrics,
            "feedback": self.feedback,
        }


class PromptEvaluator(ABC):
    """
    Base class for prompt evaluators.

    Evaluators score prompts based on various criteria such as
    clarity, specificity, coherence, and task-specific metrics.
    """

    def __init__(self, weight: float = 1.0):
        """
        Initialize evaluator.

        Args:
            weight: Weight for this evaluator in composite scoring
        """
        self.weight = weight

    @abstractmethod
    async def evaluate(
        self,
        prompt: str,
        context: Dict[str, Any],
    ) -> float:
        """
        Evaluate a prompt.

        Args:
            prompt: Prompt to evaluate
            context: Additional context for evaluation

        Returns:
            Score between 0 and 1 (higher is better)
        """
        pass

    async def evaluate_detailed(
        self,
        prompt: str,
        context: Dict[str, Any],
    ) -> EvaluationMetrics:
        """
        Evaluate with detailed metrics.

        Args:
            prompt: Prompt to evaluate
            context: Additional context

        Returns:
            EvaluationMetrics with detailed breakdown
        """
        score = await self.evaluate(prompt, context)
        return EvaluationMetrics(
            score=score,
            metrics={"overall": score},
        )


class AccuracyEvaluator(PromptEvaluator):
    """
    Evaluates prompt based on expected outputs.

    Compares actual LLM responses to expected outputs.
    """

    def __init__(
        self,
        llm_fn: Callable[[str], Any],
        test_cases: List[Dict[str, Any]],
        weight: float = 1.0,
    ):
        """
        Initialize accuracy evaluator.

        Args:
            llm_fn: Function to call LLM with prompt
            test_cases: List of test cases with expected outputs
            weight: Weight for composite scoring
        """
        super().__init__(weight)
        self.llm_fn = llm_fn
        self.test_cases = test_cases

    async def evaluate(
        self,
        prompt: str,
        context: Dict[str, Any],
    ) -> float:
        """
        Evaluate prompt accuracy.

        Args:
            prompt: Prompt to evaluate
            context: Additional context

        Returns:
            Accuracy score (0-1)
        """
        if not self.test_cases:
            return 0.0

        correct = 0

        for test_case in self.test_cases:
            # Format prompt with test input
            test_input = test_case.get("input", "")
            formatted_prompt = prompt.replace("{input}", test_input)

            # Get LLM response
            if asyncio.iscoroutinefunction(self.llm_fn):
                response = await self.llm_fn(formatted_prompt)
            else:
                response = self.llm_fn(formatted_prompt)

            # Check against expected output
            expected = test_case.get("expected", "")
            if self._check_match(response, expected):
                correct += 1

        return correct / len(self.test_cases)

    def _check_match(self, response: str, expected: str) -> bool:
        """
        Check if response matches expected output.

        Args:
            response: LLM response
            expected: Expected output

        Returns:
            True if match
        """
        # Simple substring match (can be enhanced)
        return expected.lower() in response.lower()


class CoherenceEvaluator(PromptEvaluator):
    """
    Evaluates prompt coherence and clarity.

    Uses heuristics and linguistic features.
    """

    async def evaluate(
        self,
        prompt: str,
        context: Dict[str, Any],
    ) -> float:
        """
        Evaluate prompt coherence.

        Args:
            prompt: Prompt to evaluate
            context: Additional context

        Returns:
            Coherence score (0-1)
        """
        score = 0.0
        max_score = 5.0

        # Check length (not too short, not too long)
        word_count = len(prompt.split())
        if 10 <= word_count <= 100:
            score += 1.0
        elif 5 <= word_count <= 200:
            score += 0.5

        # Check for complete sentences
        sentences = [s.strip() for s in re.split(r'[.!?]+', prompt) if s.strip()]
        if sentences:
            complete = sum(
                1 for s in sentences
                if s[0].isupper() and len(s.split()) >= 3
            )
            score += min(1.0, complete / len(sentences))

        # Check for specific instructions
        instruction_words = ["provide", "explain", "describe", "list", "analyze", "compare"]
        if any(word in prompt.lower() for word in instruction_words):
            score += 1.0

        # Check for clarity indicators
        clarity_words = ["clear", "specific", "detailed", "precise", "exact"]
        if any(word in prompt.lower() for word in clarity_words):
            score += 0.5

        # Check for proper grammar markers
        if prompt.strip().endswith((".", "?", "!")):
            score += 0.5

        # Avoid repetition
        words = prompt.lower().split()
        unique_ratio = len(set(words)) / len(words) if words else 0
        score += unique_ratio * 1.0

        return min(1.0, score / max_score)


class RelevanceEvaluator(PromptEvaluator):
    """
    Evaluates prompt relevance to a specific task.

    Uses keyword matching and semantic similarity.
    """

    def __init__(
        self,
        task_keywords: List[str],
        required_keywords: Optional[List[str]] = None,
        weight: float = 1.0,
    ):
        """
        Initialize relevance evaluator.

        Args:
            task_keywords: Keywords relevant to the task
            required_keywords: Keywords that must be present
            weight: Weight for composite scoring
        """
        super().__init__(weight)
        self.task_keywords = [k.lower() for k in task_keywords]
        self.required_keywords = [k.lower() for k in (required_keywords or [])]

    async def evaluate(
        self,
        prompt: str,
        context: Dict[str, Any],
    ) -> float:
        """
        Evaluate prompt relevance.

        Args:
            prompt: Prompt to evaluate
            context: Additional context

        Returns:
            Relevance score (0-1)
        """
        prompt_lower = prompt.lower()

        # Check required keywords
        if self.required_keywords:
            required_present = sum(
                1 for k in self.required_keywords
                if k in prompt_lower
            )
            if required_present < len(self.required_keywords):
                # Penalize missing required keywords
                return required_present / len(self.required_keywords) * 0.5

        # Check task keywords
        if self.task_keywords:
            keywords_present = sum(
                1 for k in self.task_keywords
                if k in prompt_lower
            )
            keyword_score = keywords_present / len(self.task_keywords)
        else:
            keyword_score = 1.0

        return keyword_score


class LengthEvaluator(PromptEvaluator):
    """
    Evaluates prompt length.

    Encourages prompts within a target length range.
    """

    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 200,
        optimal_length: int = 50,
        weight: float = 1.0,
    ):
        """
        Initialize length evaluator.

        Args:
            min_length: Minimum word count
            max_length: Maximum word count
            optimal_length: Optimal word count
            weight: Weight for composite scoring
        """
        super().__init__(weight)
        self.min_length = min_length
        self.max_length = max_length
        self.optimal_length = optimal_length

    async def evaluate(
        self,
        prompt: str,
        context: Dict[str, Any],
    ) -> float:
        """
        Evaluate prompt length.

        Args:
            prompt: Prompt to evaluate
            context: Additional context

        Returns:
            Length score (0-1)
        """
        word_count = len(prompt.split())

        if word_count < self.min_length:
            return word_count / self.min_length * 0.5
        elif word_count > self.max_length:
            return max(0.0, 1.0 - (word_count - self.max_length) / self.max_length)
        else:
            # Gaussian-like curve around optimal length
            distance = abs(word_count - self.optimal_length)
            range_size = self.max_length - self.min_length
            return 1.0 - (distance / range_size) * 0.5


class SpecificityEvaluator(PromptEvaluator):
    """
    Evaluates prompt specificity.

    Rewards concrete, specific language over vague terms.
    """

    def __init__(self, weight: float = 1.0):
        """
        Initialize specificity evaluator.

        Args:
            weight: Weight for composite scoring
        """
        super().__init__(weight)
        self.vague_words = {
            "thing", "stuff", "something", "anything", "everything",
            "some", "any", "might", "maybe", "perhaps", "possibly",
        }
        self.specific_indicators = {
            "specifically", "exactly", "precisely", "particular",
            "detailed", "explicit", "concrete", "definite",
        }

    async def evaluate(
        self,
        prompt: str,
        context: Dict[str, Any],
    ) -> float:
        """
        Evaluate prompt specificity.

        Args:
            prompt: Prompt to evaluate
            context: Additional context

        Returns:
            Specificity score (0-1)
        """
        words = prompt.lower().split()
        if not words:
            return 0.0

        score = 1.0

        # Penalize vague words
        vague_count = sum(1 for w in words if w in self.vague_words)
        vague_penalty = (vague_count / len(words)) * 0.5
        score -= vague_penalty

        # Reward specific indicators
        specific_count = sum(1 for w in words if w in self.specific_indicators)
        specific_bonus = min(0.2, (specific_count / len(words)) * 2)
        score += specific_bonus

        # Reward numbers (specific quantities)
        number_count = sum(1 for w in words if w.isdigit())
        number_bonus = min(0.1, (number_count / len(words)) * 2)
        score += number_bonus

        return max(0.0, min(1.0, score))


class CompositeEvaluator(PromptEvaluator):
    """
    Combines multiple evaluators with weighted averaging.
    """

    def __init__(self, evaluators: List[PromptEvaluator]):
        """
        Initialize composite evaluator.

        Args:
            evaluators: List of evaluators to combine
        """
        if not evaluators:
            raise ValueError("At least one evaluator required")

        total_weight = sum(e.weight for e in evaluators)
        super().__init__(weight=total_weight)
        self.evaluators = evaluators

    async def evaluate(
        self,
        prompt: str,
        context: Dict[str, Any],
    ) -> float:
        """
        Evaluate using all evaluators.

        Args:
            prompt: Prompt to evaluate
            context: Additional context

        Returns:
            Weighted average score (0-1)
        """
        # Evaluate with all evaluators concurrently
        tasks = [
            evaluator.evaluate(prompt, context)
            for evaluator in self.evaluators
        ]
        scores = await asyncio.gather(*tasks)

        # Weighted average
        weighted_sum = sum(
            score * evaluator.weight
            for score, evaluator in zip(scores, self.evaluators)
        )

        total_weight = sum(e.weight for e in self.evaluators)

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    async def evaluate_detailed(
        self,
        prompt: str,
        context: Dict[str, Any],
    ) -> EvaluationMetrics:
        """
        Evaluate with detailed metrics from all evaluators.

        Args:
            prompt: Prompt to evaluate
            context: Additional context

        Returns:
            EvaluationMetrics with breakdown by evaluator
        """
        # Evaluate with all evaluators
        tasks = [
            evaluator.evaluate(prompt, context)
            for evaluator in self.evaluators
        ]
        scores = await asyncio.gather(*tasks)

        # Build metrics dict
        metrics = {}
        for evaluator, score in zip(self.evaluators, scores):
            evaluator_name = evaluator.__class__.__name__
            metrics[evaluator_name] = score

        # Calculate overall score
        overall_score = await self.evaluate(prompt, context)
        metrics["overall"] = overall_score

        return EvaluationMetrics(
            score=overall_score,
            metrics=metrics,
        )
