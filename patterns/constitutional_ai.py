"""
Constitutional AI - Self-critique pattern implementation.

This module implements a system where AI critiques and improves its own outputs
based on predefined principles/constitution.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)


class PrincipleCategory(Enum):
    """Categories for organizing principles."""
    HARMLESS = "harmless"
    HELPFUL = "helpful"
    HONEST = "honest"
    CUSTOM = "custom"


@dataclass
class Principle:
    """A constitutional principle for evaluating responses."""
    name: str
    description: str
    critique_prompt: str
    weight: float = 1.0
    category: PrincipleCategory = PrincipleCategory.CUSTOM
    min_score: float = 0.7

    def format_critique_prompt(self, response: str) -> str:
        """Format the critique prompt with the response."""
        return f"{self.critique_prompt}\n\nResponse to evaluate:\n{response}"


@dataclass
class CritiqueResult:
    """Result of critiquing a response against a principle."""
    principle_name: str
    score: float  # 0.0 to 1.0
    feedback: str
    passed: bool
    suggestions: List[str] = field(default_factory=list)


@dataclass
class ConstitutionalConfig:
    """Configuration for Constitutional AI."""
    principles: List[Principle]
    max_revisions: int = 3
    critique_model: Optional[str] = None
    revision_model: Optional[str] = None
    min_quality_score: float = 0.8
    parallel_critique: bool = False
    aggregate_method: str = "weighted_average"  # or "min", "max"
    stop_on_regression: bool = True
    revision_prompt_template: str = (
        "Based on the following critique feedback, please revise your response:\n\n"
        "{feedback}\n\n"
        "Original response:\n{response}\n\n"
        "Revised response:"
    )


@dataclass
class RevisionHistory:
    """Track the history of revisions."""
    iteration: int
    response: str
    critiques: List[CritiqueResult]
    quality_score: float
    timestamp: float


class ConstitutionalAIStep:
    """
    A step that uses Constitutional AI to generate and refine responses.

    This implements the self-critique pattern where:
    1. Generate initial response
    2. Critique against constitutional principles
    3. Revise based on critique
    4. Iterate until quality threshold or max iterations
    """

    def __init__(
        self,
        name: str,
        prompt: str,
        config: ConstitutionalConfig,
        llm_provider: Optional[Any] = None,
    ):
        """
        Initialize Constitutional AI step.

        Args:
            name: Name of this step
            prompt: Initial prompt to generate response
            config: Constitutional configuration
            llm_provider: LLM provider for generation and critique
        """
        self.name = name
        self.prompt = prompt
        self.config = config
        self.llm_provider = llm_provider
        self.history: List[RevisionHistory] = []

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the Constitutional AI process.

        Args:
            context: Execution context

        Returns:
            Dict containing final response and revision history
        """
        logger.info(f"Starting Constitutional AI step: {self.name}")

        # Generate initial response
        current_response = await self._generate_initial_response(context)
        best_response = current_response
        best_score = 0.0

        for iteration in range(self.config.max_revisions):
            logger.info(f"Revision iteration {iteration + 1}/{self.config.max_revisions}")

            # Critique the response
            critiques = await self._critique_response(current_response)

            # Calculate quality score
            quality_score = self._calculate_quality_score(critiques)

            # Track history
            import time
            self.history.append(RevisionHistory(
                iteration=iteration,
                response=current_response,
                critiques=critiques,
                quality_score=quality_score,
                timestamp=time.time()
            ))

            logger.info(f"Quality score: {quality_score:.2f}")

            # Check if we've achieved the quality threshold
            if quality_score >= self.config.min_quality_score:
                logger.info(f"Quality threshold reached: {quality_score:.2f}")
                best_response = current_response
                best_score = quality_score
                break

            # Track best response
            if quality_score > best_score:
                best_response = current_response
                best_score = quality_score
            elif self.config.stop_on_regression:
                logger.info("Quality regressed, stopping revisions")
                break

            # Generate revision based on critiques
            if iteration < self.config.max_revisions - 1:
                current_response = await self._generate_revision(
                    current_response, critiques
                )

        # Get final critiques for best response
        final_critiques = await self._critique_response(best_response)

        return {
            "response": best_response,
            "quality_score": best_score,
            "revisions": len(self.history),
            "principles_passed": [
                c.principle_name for c in final_critiques if c.passed
            ],
            "principles_failed": [
                c.principle_name for c in final_critiques if not c.passed
            ],
            "history": self.history,
            "final_critiques": final_critiques
        }

    async def _generate_initial_response(self, context: Dict[str, Any]) -> str:
        """Generate the initial response."""
        formatted_prompt = self.prompt.format(**context)

        if self.llm_provider:
            response = await self.llm_provider.generate(
                prompt=formatted_prompt,
                model=self.config.revision_model
            )
            return response.get("content", response.get("text", str(response)))

        # Fallback for testing
        return f"Generated response for: {formatted_prompt}"

    async def _critique_response(self, response: str) -> List[CritiqueResult]:
        """
        Critique a response against all principles.

        Args:
            response: The response to critique

        Returns:
            List of critique results
        """
        if self.config.parallel_critique:
            tasks = [
                self._critique_against_principle(response, principle)
                for principle in self.config.principles
            ]
            return await asyncio.gather(*tasks)
        else:
            critiques = []
            for principle in self.config.principles:
                critique = await self._critique_against_principle(response, principle)
                critiques.append(critique)
            return critiques

    async def _critique_against_principle(
        self, response: str, principle: Principle
    ) -> CritiqueResult:
        """
        Critique a response against a single principle.

        Args:
            response: The response to critique
            principle: The principle to evaluate against

        Returns:
            Critique result
        """
        critique_prompt = principle.format_critique_prompt(response)

        if self.llm_provider:
            critique_response = await self.llm_provider.generate(
                prompt=critique_prompt,
                model=self.config.critique_model
            )
            critique_text = critique_response.get(
                "content", critique_response.get("text", str(critique_response))
            )

            # Parse score and feedback (simple implementation)
            score = self._parse_score(critique_text)
            feedback = critique_text
            suggestions = self._parse_suggestions(critique_text)
        else:
            # Fallback for testing
            score = 0.8
            feedback = f"Critique against {principle.name}"
            suggestions = ["Suggestion 1", "Suggestion 2"]

        passed = score >= principle.min_score

        return CritiqueResult(
            principle_name=principle.name,
            score=score,
            feedback=feedback,
            passed=passed,
            suggestions=suggestions
        )

    def _calculate_quality_score(self, critiques: List[CritiqueResult]) -> float:
        """
        Calculate overall quality score from critiques.

        Args:
            critiques: List of critique results

        Returns:
            Overall quality score (0.0 to 1.0)
        """
        if not critiques:
            return 0.0

        if self.config.aggregate_method == "weighted_average":
            total_weight = sum(
                p.weight for p in self.config.principles
            )
            weighted_sum = sum(
                c.score * p.weight
                for c, p in zip(critiques, self.config.principles)
            )
            return weighted_sum / total_weight if total_weight > 0 else 0.0

        elif self.config.aggregate_method == "min":
            return min(c.score for c in critiques)

        elif self.config.aggregate_method == "max":
            return max(c.score for c in critiques)

        else:
            # Default to average
            return sum(c.score for c in critiques) / len(critiques)

    async def _generate_revision(
        self, response: str, critiques: List[CritiqueResult]
    ) -> str:
        """
        Generate a revised response based on critiques.

        Args:
            response: Current response
            critiques: Critique results

        Returns:
            Revised response
        """
        # Format feedback
        feedback_text = "\n\n".join([
            f"**{c.principle_name}** (score: {c.score:.2f}):\n{c.feedback}"
            for c in critiques
            if not c.passed
        ])

        revision_prompt = self.config.revision_prompt_template.format(
            feedback=feedback_text,
            response=response
        )

        if self.llm_provider:
            revision_response = await self.llm_provider.generate(
                prompt=revision_prompt,
                model=self.config.revision_model
            )
            return revision_response.get(
                "content", revision_response.get("text", str(revision_response))
            )

        # Fallback for testing
        return f"Revised: {response}"

    def _parse_score(self, critique_text: str) -> float:
        """
        Parse a score from critique text.

        Simple heuristic: look for numbers between 0-10 or 0.0-1.0
        """
        import re

        # Look for explicit score patterns
        patterns = [
            r'score[:\s]+(\d+\.?\d*)/10',
            r'score[:\s]+(\d+\.?\d*)',
            r'rating[:\s]+(\d+\.?\d*)/10',
            r'rating[:\s]+(\d+\.?\d*)',
            r'(\d+\.?\d*)/10',
        ]

        for pattern in patterns:
            match = re.search(pattern, critique_text.lower())
            if match:
                score = float(match.group(1))
                # Normalize to 0-1
                if score > 1.0:
                    score = score / 10.0
                return min(1.0, max(0.0, score))

        # Default: use sentiment heuristics
        positive_words = ['good', 'excellent', 'great', 'well', 'clear', 'helpful']
        negative_words = ['bad', 'poor', 'unclear', 'problematic', 'harmful']

        text_lower = critique_text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count + negative_count == 0:
            return 0.7  # Neutral default

        return positive_count / (positive_count + negative_count)

    def _parse_suggestions(self, critique_text: str) -> List[str]:
        """Parse suggestions from critique text."""
        import re

        # Look for bullet points or numbered lists
        suggestions = []

        # Look for lines starting with bullets or numbers
        lines = critique_text.split('\n')
        for line in lines:
            line = line.strip()
            if re.match(r'^[-*•]\s+', line) or re.match(r'^\d+[\.)]\s+', line):
                suggestion = re.sub(r'^[-*•\d\.)]+\s+', '', line)
                if suggestion:
                    suggestions.append(suggestion)

        return suggestions if suggestions else ["Consider revising based on feedback"]


# Helper function for easy usage
async def apply_constitutional_ai(
    prompt: str,
    principles: List[Principle],
    llm_provider: Optional[Any] = None,
    max_revisions: int = 3,
    min_quality_score: float = 0.8,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to apply Constitutional AI to a prompt.

    Args:
        prompt: The prompt to process
        principles: List of principles to evaluate against
        llm_provider: LLM provider
        max_revisions: Maximum number of revisions
        min_quality_score: Minimum quality score to achieve
        context: Optional context dict

    Returns:
        Result dict with response and metadata
    """
    config = ConstitutionalConfig(
        principles=principles,
        max_revisions=max_revisions,
        min_quality_score=min_quality_score
    )

    step = ConstitutionalAIStep(
        name="constitutional_ai",
        prompt=prompt,
        config=config,
        llm_provider=llm_provider
    )

    return await step.execute(context or {})
