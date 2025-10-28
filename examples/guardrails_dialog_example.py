"""
Dialog Rails Example

Demonstrates conversation flow control with dialog rails.
"""
import asyncio
from ia_modules.guardrails import GuardrailConfig, RailType, RailAction
from ia_modules.guardrails.dialog_rails import (
    ContextLengthRail,
    TopicAdherenceRail,
    ConversationFlowRail
)


async def test_context_length():
    """Test context length rail."""
    print("\n=== Context Length Rail ===")

    config = GuardrailConfig(
        name="context_length",
        type=RailType.DIALOG
    )

    rail = ContextLengthRail(config, max_turns=5, max_tokens=100)

    # Test 1: Short conversation (should pass)
    context1 = {
        "conversation_history": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
    }

    result1 = await rail.execute("I'm doing great!", context=context1)
    print(f"\nTest 1 - Short conversation:")
    print(f"  Action: {result1.action.value}")
    print(f"  Turn count: {result1.metadata.get('turn_count', 0)}")
    print(f"  Estimated tokens: {result1.metadata.get('estimated_tokens', 0)}")

    # Test 2: Long conversation (should warn)
    context2 = {
        "conversation_history": [
            {"role": "user", "content": f"Message {i}"} for i in range(10)
        ]
    }

    result2 = await rail.execute("Another message", context=context2)
    print(f"\nTest 2 - Long conversation:")
    print(f"  Action: {result2.action.value}")
    print(f"  Triggered: {result2.triggered}")
    print(f"  Reason: {result2.reason}")

    print(f"\nStatistics: {rail.get_stats()}")


async def test_topic_adherence():
    """Test topic adherence rail."""
    print("\n=== Topic Adherence Rail ===")

    config = GuardrailConfig(
        name="topic_adherence",
        type=RailType.DIALOG
    )

    rail = TopicAdherenceRail(
        config,
        allowed_topics=["python", "programming", "coding", "software"],
        strict_mode=False
    )

    # Test 1: On-topic message
    result1 = await rail.execute("How do I write a Python function?")
    print(f"\nTest 1 - On-topic (Python):")
    print(f"  Action: {result1.action.value}")
    print(f"  Matched topics: {result1.metadata.get('matched_topics', [])}")

    # Test 2: Off-topic message
    result2 = await rail.execute("What's your favorite recipe?")
    print(f"\nTest 2 - Off-topic (cooking):")
    print(f"  Action: {result2.action.value}")
    print(f"  Triggered: {result2.triggered}")
    print(f"  Reason: {result2.reason}")

    # Test 3: Strict mode (blocks off-topic)
    strict_rail = TopicAdherenceRail(
        GuardrailConfig(name="strict_topic", type=RailType.DIALOG),
        allowed_topics=["python"],
        strict_mode=True
    )

    result3 = await strict_rail.execute("Tell me about JavaScript")
    print(f"\nTest 3 - Off-topic with strict mode:")
    print(f"  Action: {result3.action.value}")
    print(f"  Triggered: {result3.triggered}")

    print(f"\nStatistics: {rail.get_stats()}")


async def test_conversation_flow():
    """Test conversation flow rail."""
    print("\n=== Conversation Flow Rail ===")

    config = GuardrailConfig(
        name="conversation_flow",
        type=RailType.DIALOG
    )

    rail = ConversationFlowRail(config, max_repetitions=2)

    # Test 1: Normal conversation (no repetition)
    context1 = {
        "conversation_history": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
    }

    result1 = await rail.execute("What's the weather?", context=context1)
    print(f"\nTest 1 - Normal conversation:")
    print(f"  Action: {result1.action.value}")
    print(f"  Similar count: {result1.metadata.get('similar_count', 0)}")

    # Test 2: Repetitive conversation (loop detected)
    context2 = {
        "conversation_history": [
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "I don't have weather info"},
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "I can't check weather"},
            {"role": "user", "content": "What's the weather?"}
        ]
    }

    result2 = await rail.execute("What's the weather?", context=context2)
    print(f"\nTest 2 - Repetitive conversation:")
    print(f"  Action: {result2.action.value}")
    print(f"  Triggered: {result2.triggered}")
    print(f"  Reason: {result2.reason}")
    print(f"  Recommendation: {result2.metadata.get('recommendation', '')}")

    print(f"\nStatistics: {rail.get_stats()}")


async def main():
    """Run all dialog rail examples."""
    print("Dialog Rails Example")
    print("=" * 50)

    await test_context_length()
    await test_topic_adherence()
    await test_conversation_flow()

    print("\n" + "=" * 50)
    print("All dialog rail tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
