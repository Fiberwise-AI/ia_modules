"""
Example: Advanced Memory Strategies

This example demonstrates semantic, episodic, and working memory
with automatic compression.
"""

import asyncio
import time
from ia_modules.memory import (
    MemoryManager,
    MemoryConfig,
    MemoryType
)


async def basic_memory_example():
    """Basic memory management example."""
    print("=== Basic Memory Example ===\n")

    # Create memory manager
    config = MemoryConfig(
        semantic_enabled=True,
        episodic_enabled=True,
        working_memory_size=5,
        compression_threshold=10,
        enable_embeddings=False  # Disable for demo
    )

    memory = MemoryManager(config)

    # Add various memories
    await memory.add(
        "User prefers Python over JavaScript",
        metadata={"type": "preference", "importance": 0.9}
    )

    await memory.add(
        "Meeting scheduled for tomorrow at 3pm",
        metadata={"type": "event", "importance": 0.7, "tags": ["meeting"]}
    )

    await memory.add(
        "User asked about machine learning",
        metadata={"importance": 0.6}
    )

    # Retrieve relevant memories
    results = await memory.retrieve("What programming language?", k=3)

    print("Retrieved memories:")
    for mem in results:
        print(f"  - {mem.content} (importance: {mem.importance})")

    # Get stats
    stats = await memory.get_stats()
    print("\nMemory Stats:")
    print(f"  Total: {stats['total_memories']}")
    print(f"  By type: {stats['by_type']}")


async def conversation_memory_example():
    """Example of maintaining conversation context."""
    print("\n=== Conversation Memory Example ===\n")

    config = MemoryConfig(
        working_memory_size=3,
        enable_embeddings=False
    )

    memory = MemoryManager(config)

    # Simulate a conversation
    conversation = [
        "What is machine learning?",
        "How does it differ from deep learning?",
        "Can you give an example?",
        "What about neural networks?",
        "How do I get started learning ML?"
    ]

    for i, message in enumerate(conversation):
        await memory.add(
            f"User: {message}",
            metadata={"importance": 0.5, "turn": i}
        )
        time.sleep(0.1)  # Simulate time passing

    # Get context window
    context = await memory.get_context_window(
        query="machine learning basics",
        max_tokens=1000
    )

    print("Context Window:")
    print(context)


async def long_term_memory_example():
    """Example with semantic and episodic memory."""
    print("\n=== Long-Term Memory Example ===\n")

    config = MemoryConfig(
        semantic_enabled=True,
        episodic_enabled=True,
        enable_embeddings=False
    )

    memory = MemoryManager(config)

    # Add facts (semantic memory)
    facts = [
        "Python was created by Guido van Rossum",
        "Python 3 was released in 2008",
        "Python is dynamically typed",
        "Python uses significant whitespace"
    ]

    for fact in facts:
        await memory.add(
            fact,
            metadata={"memory_type": "semantic", "importance": 0.8}
        )

    # Add events (episodic memory)
    events = [
        "User started learning Python on Monday",
        "User completed first project on Tuesday",
        "User asked about decorators on Wednesday"
    ]

    for i, event in enumerate(events):
        await memory.add(
            event,
            metadata={
                "memory_type": "episodic",
                "importance": 0.6,
                "tags": ["learning"],
                "day": i + 1
            }
        )

    # Query semantic memory
    print("Python facts:")
    facts_results = await memory.retrieve(
        "Python programming language",
        memory_types=[MemoryType.SEMANTIC],
        k=3
    )
    for mem in facts_results:
        print(f"  - {mem.content}")

    # Query episodic memory
    print("\nLearning history:")
    events_results = await memory.retrieve(
        "learning progress",
        memory_types=[MemoryType.EPISODIC],
        k=3
    )
    for mem in events_results:
        print(f"  - {mem.content}")


async def memory_compression_example():
    """Example of automatic memory compression."""
    print("\n=== Memory Compression Example ===\n")

    config = MemoryConfig(
        compression_threshold=5,
        compression_enabled=True,
        enable_embeddings=False
    )

    memory = MemoryManager(config)

    # Add many memories to trigger compression
    for i in range(10):
        await memory.add(
            f"Memory item {i} with some content",
            metadata={"importance": 0.3 + (i * 0.05)}
        )

    print(f"Memories before compression: {len(memory.memories)}")

    # Manually trigger compression
    compressed_count = await memory.compress()

    print(f"Compressed {compressed_count} memories")
    print(f"Memories after compression: {len(memory.memories)}")


if __name__ == "__main__":
    asyncio.run(basic_memory_example())
    asyncio.run(conversation_memory_example())
    asyncio.run(long_term_memory_example())
    asyncio.run(memory_compression_example())
