"""
Retrieval Rails Example

Demonstrates RAG safety with retrieval rails.
"""
import asyncio
from ia_modules.guardrails import GuardrailConfig, RailType, RailAction
from ia_modules.guardrails.retrieval_rails import (
    SourceValidationRail,
    RelevanceFilterRail,
    RetrievedContentFilterRail
)


async def test_source_validation():
    """Test source validation rail."""
    print("\n=== Source Validation Rail ===")

    config = GuardrailConfig(
        name="source_validation",
        type=RailType.RETRIEVAL
    )

    rail = SourceValidationRail(
        config,
        allowed_sources=["wikipedia.org", "*.gov", "scholarly.org"],
        require_metadata=["author", "date"]
    )

    # Test 1: Valid source with metadata
    doc1 = "Machine learning is a subset of artificial intelligence..."
    context1 = {
        "metadata": {
            "source": "wikipedia.org",
            "author": "Wikipedia Contributors",
            "date": "2024-01-15"
        }
    }

    result1 = await rail.execute(doc1, context=context1)
    print(f"\nTest 1 - Valid source (Wikipedia):")
    print(f"  Action: {result1.action.value}")
    print(f"  Source: {result1.metadata.get('source', 'N/A')}")

    # Test 2: Invalid source
    doc2 = "Some random content from an unknown source..."
    context2 = {
        "metadata": {
            "source": "random-blog.com",
            "author": "Anonymous"
        }
    }

    result2 = await rail.execute(doc2, context=context2)
    print(f"\nTest 2 - Invalid source:")
    print(f"  Action: {result2.action.value}")
    print(f"  Triggered: {result2.triggered}")
    print(f"  Reason: {result2.reason}")

    # Test 3: Missing metadata
    doc3 = "Content from a government website..."
    context3 = {
        "metadata": {
            "source": "cdc.gov"  # Missing author and date
        }
    }

    result3 = await rail.execute(doc3, context=context3)
    print(f"\nTest 3 - Missing metadata:")
    print(f"  Action: {result3.action.value}")
    print(f"  Triggered: {result3.triggered}")
    print(f"  Missing fields: {result3.metadata.get('missing_fields', [])}")

    print(f"\nStatistics: {rail.get_stats()}")


async def test_relevance_filter():
    """Test relevance filter rail."""
    print("\n=== Relevance Filter Rail ===")

    config = GuardrailConfig(
        name="relevance_filter",
        type=RailType.RETRIEVAL
    )

    rail = RelevanceFilterRail(config, min_score=0.6, max_documents=3)

    # Test 1: Single document with good score
    doc1 = {"content": "Python is a programming language", "score": 0.95}
    result1 = await rail.execute(doc1)
    print(f"\nTest 1 - High relevance document:")
    print(f"  Action: {result1.action.value}")
    print(f"  Document count: {result1.metadata.get('document_count', 0)}")

    # Test 2: Multiple documents with mixed scores
    docs2 = [
        {"content": "Python tutorial", "score": 0.92},
        {"content": "Python examples", "score": 0.85},
        {"content": "Random text", "score": 0.3},  # Low score
        {"content": "Python guide", "score": 0.78},
        {"content": "Unrelated", "score": 0.1}  # Low score
    ]

    result2 = await rail.execute(docs2)
    print(f"\nTest 2 - Mixed relevance documents:")
    print(f"  Action: {result2.action.value}")
    print(f"  Triggered: {result2.triggered}")
    print(f"  Original count: {result2.metadata.get('original_count', 0)}")
    print(f"  Filtered count: {result2.metadata.get('filtered_count', 0)}")
    print(f"  Reason: {result2.reason}")

    # Test 3: All low-relevance documents
    docs3 = [
        {"content": "Irrelevant 1", "score": 0.2},
        {"content": "Irrelevant 2", "score": 0.3}
    ]

    result3 = await rail.execute(docs3)
    print(f"\nTest 3 - All low-relevance:")
    print(f"  Action: {result3.action.value}")
    print(f"  Triggered: {result3.triggered}")
    print(f"  Reason: {result3.reason}")

    print(f"\nStatistics: {rail.get_stats()}")


async def test_content_filter():
    """Test retrieved content filter rail."""
    print("\n=== Retrieved Content Filter Rail ===")

    config = GuardrailConfig(
        name="content_filter",
        type=RailType.RETRIEVAL
    )

    rail = RetrievedContentFilterRail(config, block_harmful=True)

    # Test 1: Safe content
    doc1 = "Python is a high-level programming language known for its simplicity."
    result1 = await rail.execute(doc1)
    print(f"\nTest 1 - Safe content:")
    print(f"  Action: {result1.action.value}")
    print(f"  Triggered: {result1.triggered}")

    # Test 2: Harmful content
    doc2 = "This article discusses violent crime statistics and attack patterns."
    result2 = await rail.execute(doc2)
    print(f"\nTest 2 - Harmful content:")
    print(f"  Action: {result2.action.value}")
    print(f"  Triggered: {result2.triggered}")
    print(f"  Reason: {result2.reason}")
    print(f"  Match count: {result2.metadata.get('match_count', 0)}")

    # Test 3: Document object
    doc3 = {
        "content": "Research on illegal activities and criminal behavior patterns.",
        "source": "research.org"
    }
    result3 = await rail.execute(doc3)
    print(f"\nTest 3 - Document object with harmful content:")
    print(f"  Action: {result3.action.value}")
    print(f"  Triggered: {result3.triggered}")

    # Test 4: Warning mode (not blocking)
    warn_rail = RetrievedContentFilterRail(
        GuardrailConfig(name="warn_filter", type=RailType.RETRIEVAL),
        block_harmful=False
    )

    result4 = await warn_rail.execute("Content about violent protests.")
    print(f"\nTest 4 - Warning mode (not blocking):")
    print(f"  Action: {result4.action.value}")
    print(f"  Triggered: {result4.triggered}")

    print(f"\nStatistics: {rail.get_stats()}")


async def test_rag_pipeline():
    """Test complete RAG pipeline with all retrieval rails."""
    print("\n=== Complete RAG Pipeline ===")

    # Create all retrieval rails
    source_rail = SourceValidationRail(
        GuardrailConfig(name="source", type=RailType.RETRIEVAL),
        allowed_sources=["wikipedia.org", "*.edu"],
        require_metadata=["source"]
    )

    relevance_rail = RelevanceFilterRail(
        GuardrailConfig(name="relevance", type=RailType.RETRIEVAL),
        min_score=0.7,
        max_documents=5
    )

    content_rail = RetrievedContentFilterRail(
        GuardrailConfig(name="content", type=RailType.RETRIEVAL),
        block_harmful=True
    )

    # Simulate retrieved documents
    retrieved_docs = [
        {
            "content": "Machine learning is transforming technology.",
            "score": 0.92,
            "metadata": {"source": "mit.edu", "author": "MIT"}
        },
        {
            "content": "Python programming fundamentals.",
            "score": 0.88,
            "metadata": {"source": "wikipedia.org", "author": "Contributors"}
        },
        {
            "content": "Random low-relevance text.",
            "score": 0.45,
            "metadata": {"source": "random.com", "author": "Unknown"}
        }
    ]

    print("\nProcessing RAG pipeline with 3 retrieved documents...")

    # Process each document through the pipeline
    safe_docs = []
    for i, doc in enumerate(retrieved_docs):
        print(f"\n  Document {i+1}:")

        # Check source
        source_result = await source_rail.execute(
            doc["content"],
            context={"metadata": doc["metadata"]}
        )
        print(f"    Source validation: {source_result.action.value}")

        if source_result.action == RailAction.BLOCK:
            continue

        # Check relevance
        relevance_result = await relevance_rail.execute(doc)
        print(f"    Relevance filter: {relevance_result.action.value} (score: {doc['score']})")

        if relevance_result.action == RailAction.BLOCK:
            continue

        # Check content
        content_result = await content_rail.execute(doc["content"])
        print(f"    Content filter: {content_result.action.value}")

        if content_result.action != RailAction.BLOCK:
            safe_docs.append(doc)

    print(f"\n  Pipeline result: {len(safe_docs)}/{len(retrieved_docs)} documents passed all rails")

    # Print statistics
    print("\n  Rail Statistics:")
    print(f"    Source: {source_rail.get_stats()['trigger_rate']:.1%} trigger rate")
    print(f"    Relevance: {relevance_rail.get_stats()['trigger_rate']:.1%} trigger rate")
    print(f"    Content: {content_rail.get_stats()['trigger_rate']:.1%} trigger rate")


async def main():
    """Run all retrieval rail examples."""
    print("Retrieval Rails Example")
    print("=" * 50)

    await test_source_validation()
    await test_relevance_filter()
    await test_content_filter()
    await test_rag_pipeline()

    print("\n" + "=" * 50)
    print("All retrieval rail tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
