"""
Example: Constitutional AI (Self-Critique Pattern)

This example demonstrates how to use Constitutional AI to generate
and refine responses based on constitutional principles.
"""

import asyncio
from ia_modules.patterns import ConstitutionalAIStep, ConstitutionalConfig, Principle
from ia_modules.patterns.constitutions import harmless_principles, helpful_principles


async def basic_example():
    """Basic Constitutional AI example."""
    print("=== Basic Constitutional AI Example ===\n")

    # Create a simple configuration with helpful principles
    config = ConstitutionalConfig(
        principles=helpful_principles,
        max_revisions=3,
        min_quality_score=0.8
    )

    # Create step
    step = ConstitutionalAIStep(
        name="helpful_assistant",
        prompt="Explain {topic} in simple terms",
        config=config
    )

    # Execute
    context = {"topic": "quantum computing"}
    result = await step.execute(context)

    print(f"Final Response:\n{result['response']}\n")
    print(f"Quality Score: {result['quality_score']:.2f}")
    print(f"Revisions: {result['revisions']}")
    print(f"Principles Passed: {result['principles_passed']}")


async def custom_principles_example():
    """Example with custom principles."""
    print("\n=== Custom Principles Example ===\n")

    # Define custom principles for technical writing
    custom_principles = [
        Principle(
            name="technical_accuracy",
            description="Response should be technically accurate",
            critique_prompt=(
                "Evaluate the technical accuracy of this response. "
                "Are there any errors or misleading statements? "
                "Rate 0-10 where 10 is completely accurate."
            ),
            weight=2.0,
            min_score=0.9
        ),
        Principle(
            name="beginner_friendly",
            description="Response should be accessible to beginners",
            critique_prompt=(
                "Evaluate if this response is accessible to beginners. "
                "Is jargon explained? Are concepts introduced gradually? "
                "Rate 0-10 where 10 is perfectly beginner-friendly."
            ),
            weight=1.5,
            min_score=0.8
        ),
    ]

    config = ConstitutionalConfig(
        principles=custom_principles,
        max_revisions=2,
        min_quality_score=0.85,
        aggregate_method="weighted_average"
    )

    step = ConstitutionalAIStep(
        name="technical_writer",
        prompt="Explain {concept} to a beginner",
        config=config
    )

    result = await step.execute({"concept": "neural networks"})

    print(f"Final Response:\n{result['response']}\n")
    print(f"Quality Score: {result['quality_score']:.2f}")
    print(f"Principles Passed: {', '.join(result['principles_passed'])}")


async def multi_constitution_example():
    """Example using multiple constitutions."""
    print("\n=== Multi-Constitution Example ===\n")

    # Combine harmless and helpful principles
    combined_principles = harmless_principles + helpful_principles

    config = ConstitutionalConfig(
        principles=combined_principles,
        max_revisions=3,
        min_quality_score=0.85,
        parallel_critique=True  # Critique principles in parallel
    )

    step = ConstitutionalAIStep(
        name="safe_helpful_assistant",
        prompt="Provide advice on {topic}",
        config=config
    )

    result = await step.execute({"topic": "dealing with stress"})

    print(f"Final Response:\n{result['response']}\n")
    print(f"Quality Score: {result['quality_score']:.2f}")
    print(f"Total Revisions: {result['revisions']}")

    # Show revision history
    print("\nRevision History:")
    for i, revision in enumerate(result['history']):
        print(f"  Revision {i}: Score {revision.quality_score:.2f}")


if __name__ == "__main__":
    asyncio.run(basic_example())
    asyncio.run(custom_principles_example())
    asyncio.run(multi_constitution_example())
