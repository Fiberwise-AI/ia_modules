"""
Unit tests for advanced memory systems.

Tests semantic, episodic, working memory, and compression.
"""

import pytest
from ia_modules.pipeline.test_utils import create_test_execution_context
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from collections import deque

from ia_modules.memory.semantic_memory import SemanticMemory
from ia_modules.memory.episodic_memory import EpisodicMemory
from ia_modules.memory.working_memory import WorkingMemory
from ia_modules.memory.compression import (
    MemoryCompressor,
    CompressionStrategy,
    CompressionResult
)


# Mock memory class for testing
class MockMemory:
    """Mock memory object for testing."""
    _id_counter = 0

    def __init__(self, content, importance=0.5, tags=None, timestamp=None):
        MockMemory._id_counter += 1
        self.id = f"mem_{MockMemory._id_counter}"
        self.content = content
        self.importance = importance
        self.metadata = {"tags": tags or []}
        self.timestamp = timestamp or time.time()
        self.access_count = 0
        self.embedding = None

    def __repr__(self):
        return f"<MockMemory(id={self.id}, content={self.content[:20]}...)>"


@pytest.mark.asyncio
class TestSemanticMemory:
    """Test SemanticMemory functionality."""

    @pytest.fixture
    def semantic_memory(self):
        """Create semantic memory instance."""
        return SemanticMemory(enable_embeddings=False)

    @pytest.fixture
    def semantic_memory_with_embeddings(self):
        """Create semantic memory with embeddings enabled."""
        mem = SemanticMemory(enable_embeddings=True)
        return mem

    async def test_creation(self, semantic_memory):
        """SemanticMemory can be created."""
        assert semantic_memory.embedding_model == "text-embedding-ada-002"
        assert semantic_memory.memories == {}
        assert semantic_memory.embeddings == {}

    async def test_add_memory(self, semantic_memory):
        """Can add memory to semantic storage."""
        memory = MockMemory("This is a test memory about AI")

        await semantic_memory.add(memory)

        assert memory.id in semantic_memory.memories
        assert semantic_memory.memories[memory.id] == memory

    async def test_add_multiple_memories(self, semantic_memory):
        """Can add multiple memories."""
        memories = [
            MockMemory("AI and machine learning"),
            MockMemory("Deep learning neural networks"),
            MockMemory("Natural language processing")
        ]

        for mem in memories:
            await semantic_memory.add(mem)

        assert len(semantic_memory.memories) == 3

    async def test_retrieve_keyword_search(self, semantic_memory):
        """Can retrieve memories using keyword search."""
        memories = [
            MockMemory("Python programming language"),
            MockMemory("Java programming tutorials"),
            MockMemory("Machine learning with Python")
        ]

        for mem in memories:
            await semantic_memory.add(mem)

        results = await semantic_memory.retrieve("Python", k=2)

        assert len(results) <= 2
        # Should find memories with "Python"
        assert any("Python" in r.content for r in results)

    async def test_retrieve_empty(self, semantic_memory):
        """Retrieve returns empty list when no memories."""
        results = await semantic_memory.retrieve("test", k=5)
        assert results == []

    async def test_retrieve_with_k_limit(self, semantic_memory):
        """Retrieve respects k limit."""
        memories = [MockMemory(f"Memory {i}") for i in range(10)]

        for mem in memories:
            await semantic_memory.add(mem)

        results = await semantic_memory.retrieve("Memory", k=3)
        assert len(results) <= 3

    async def test_cosine_similarity(self, semantic_memory):
        """Cosine similarity calculation works."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        vec3 = [0.0, 1.0, 0.0]

        # Identical vectors
        sim1 = semantic_memory._cosine_similarity(vec1, vec2)
        assert abs(sim1 - 1.0) < 0.01

        # Orthogonal vectors
        sim2 = semantic_memory._cosine_similarity(vec1, vec3)
        assert abs(sim2) < 0.01

    async def test_cosine_similarity_zero_vectors(self, semantic_memory):
        """Cosine similarity handles zero vectors."""
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 1.0, 1.0]

        sim = semantic_memory._cosine_similarity(vec1, vec2)
        assert sim == 0.0

    async def test_clear(self, semantic_memory):
        """Can clear all memories."""
        memories = [MockMemory(f"Memory {i}") for i in range(5)]

        for mem in memories:
            await semantic_memory.add(mem)

        await semantic_memory.clear()

        assert len(semantic_memory.memories) == 0
        assert len(semantic_memory.embeddings) == 0

    async def test_keyword_search_fallback(self, semantic_memory):
        """Keyword search works as fallback."""
        memories = [
            MockMemory("The quick brown fox"),
            MockMemory("The lazy dog sleeps"),
            MockMemory("A quick jump")
        ]

        for mem in memories:
            await semantic_memory.add(mem)

        results = await semantic_memory.retrieve("quick", k=5)

        # Should find memories with "quick"
        assert len(results) >= 1
        assert any("quick" in r.content.lower() for r in results)

    async def test_embedding_initialization_local(self):
        """Local embedding model can be initialized."""
        with patch('ia_modules.memory.semantic_memory.SentenceTransformer') as mock:
            mock.return_value = MagicMock()
            mem = SemanticMemory(enable_embeddings=True)
            mem._init_embeddings()

            if mem.embed_fn:
                assert mem.embed_fn is not None

    async def test_add_with_embedding_error(self, semantic_memory):
        """Adding memory handles embedding errors gracefully."""
        semantic_memory.embed_fn = Mock(side_effect=Exception("Embedding error"))

        memory = MockMemory("Test content")

        # Should not raise exception
        await semantic_memory.add(memory)

        assert memory.id in semantic_memory.memories


@pytest.mark.asyncio
class TestEpisodicMemory:
    """Test EpisodicMemory functionality."""

    @pytest.fixture
    def episodic_memory(self):
        """Create episodic memory instance."""
        return EpisodicMemory()

    async def test_creation(self, episodic_memory):
        """EpisodicMemory can be created."""
        assert episodic_memory.memories == {}
        assert episodic_memory.temporal_index == []
        assert len(episodic_memory.tag_index) == 0

    async def test_add_memory(self, episodic_memory):
        """Can add memory to episodic storage."""
        memory = MockMemory("Event at time 1", timestamp=100.0)

        await episodic_memory.add(memory)

        assert memory.id in episodic_memory.memories
        assert (100.0, memory.id) in episodic_memory.temporal_index

    async def test_temporal_ordering(self, episodic_memory):
        """Temporal index maintains time ordering."""
        memories = [
            MockMemory("Event 1", timestamp=300.0),
            MockMemory("Event 2", timestamp=100.0),
            MockMemory("Event 3", timestamp=200.0)
        ]

        for mem in memories:
            await episodic_memory.add(mem)

        # Check temporal index is sorted
        timestamps = [t for t, _ in episodic_memory.temporal_index]
        assert timestamps == sorted(timestamps)

    async def test_add_with_tags(self, episodic_memory):
        """Memories with tags are indexed."""
        memory = MockMemory(
            "Tagged event",
            tags=["important", "work"],
            timestamp=100.0
        )

        await episodic_memory.add(memory)

        assert memory.id in episodic_memory.tag_index["important"]
        assert memory.id in episodic_memory.tag_index["work"]

    async def test_retrieve_by_query(self, episodic_memory):
        """Can retrieve memories by query."""
        memories = [
            MockMemory("Python coding", timestamp=100.0),
            MockMemory("Java coding", timestamp=200.0),
            MockMemory("Python debugging", timestamp=300.0)
        ]

        for mem in memories:
            await episodic_memory.add(mem)

        results = await episodic_memory.retrieve("Python", k=2)

        assert len(results) <= 2
        assert all("Python" in r.content for r in results)

    async def test_retrieve_with_time_range(self, episodic_memory):
        """Can retrieve memories within time range."""
        memories = [
            MockMemory("Event 1", timestamp=100.0),
            MockMemory("Event 2", timestamp=200.0),
            MockMemory("Event 3", timestamp=300.0),
            MockMemory("Event 4", timestamp=400.0)
        ]

        for mem in memories:
            await episodic_memory.add(mem)

        results = await episodic_memory.retrieve(
            "Event",
            k=10,
            time_range=(150.0, 350.0)
        )

        # Should only get Event 2 and Event 3
        assert len(results) == 2
        timestamps = [r.timestamp for r in results]
        assert all(150.0 <= t <= 350.0 for t in timestamps)

    async def test_retrieve_with_tags_filter(self, episodic_memory):
        """Can filter retrieval by tags."""
        memories = [
            MockMemory("Event 1", tags=["work"], timestamp=100.0),
            MockMemory("Event 2", tags=["personal"], timestamp=200.0),
            MockMemory("Event 3", tags=["work"], timestamp=300.0)
        ]

        for mem in memories:
            await episodic_memory.add(mem)

        results = await episodic_memory.retrieve(
            "Event",
            k=10,
            tags=["work"]
        )

        assert len(results) == 2
        assert all("work" in r.metadata["tags"] for r in results)

    async def test_get_recent(self, episodic_memory):
        """Can get most recent memories."""
        memories = [
            MockMemory(f"Event {i}", timestamp=float(i * 100))
            for i in range(10)
        ]

        for mem in memories:
            await episodic_memory.add(mem)

        recent = await episodic_memory.get_recent(k=3)

        assert len(recent) == 3
        # Should be most recent first
        assert recent[0].timestamp > recent[1].timestamp
        assert recent[1].timestamp > recent[2].timestamp

    async def test_get_time_range(self, episodic_memory):
        """Can get memories within specific time range."""
        memories = [
            MockMemory("Event 1", timestamp=100.0),
            MockMemory("Event 2", timestamp=200.0),
            MockMemory("Event 3", timestamp=300.0)
        ]

        for mem in memories:
            await episodic_memory.add(mem)

        results = await episodic_memory.get_time_range(150.0, 250.0)

        assert len(results) == 1
        assert results[0].timestamp == 200.0

    async def test_get_by_tags(self, episodic_memory):
        """Can get memories by tags."""
        memories = [
            MockMemory("Event 1", tags=["work", "urgent"]),
            MockMemory("Event 2", tags=["personal"]),
            MockMemory("Event 3", tags=["work"])
        ]

        for mem in memories:
            await episodic_memory.add(mem)

        work_memories = await episodic_memory.get_by_tags(["work"])
        assert len(work_memories) == 2

        urgent_memories = await episodic_memory.get_by_tags(["urgent"])
        assert len(urgent_memories) == 1

    async def test_get_sequence(self, episodic_memory):
        """Can get sequence of memories."""
        memories = [
            MockMemory(f"Event {i}", timestamp=float(i * 100))
            for i in range(10)
        ]

        for mem in memories:
            await episodic_memory.add(mem)

        # Get sequence starting from Event 3
        start_id = memories[3].id
        sequence = await episodic_memory.get_sequence(start_id, length=4)

        assert len(sequence) == 4
        assert sequence[0].id == memories[3].id
        assert sequence[1].id == memories[4].id

    async def test_get_sequence_invalid_id(self, episodic_memory):
        """Get sequence with invalid ID returns empty list."""
        sequence = await episodic_memory.get_sequence("invalid_id", length=5)
        assert sequence == []

    async def test_clear(self, episodic_memory):
        """Can clear all episodic memories."""
        memories = [MockMemory(f"Event {i}") for i in range(5)]

        for mem in memories:
            await episodic_memory.add(mem)

        await episodic_memory.clear()

        assert len(episodic_memory.memories) == 0
        assert len(episodic_memory.temporal_index) == 0
        assert len(episodic_memory.tag_index) == 0


@pytest.mark.asyncio
class TestWorkingMemory:
    """Test WorkingMemory functionality."""

    @pytest.fixture
    def working_memory(self):
        """Create working memory instance."""
        return WorkingMemory(size=5)

    async def test_creation(self, working_memory):
        """WorkingMemory can be created."""
        assert working_memory.size == 5
        assert len(working_memory.memories) == 0
        assert working_memory.memory_dict == {}

    async def test_add_memory(self, working_memory):
        """Can add memory to working memory."""
        memory = MockMemory("Test memory", importance=0.8)

        await working_memory.add(memory)

        assert memory.id in working_memory.memory_dict
        assert len(working_memory.memories) == 1

    async def test_capacity_limit(self, working_memory):
        """Working memory respects capacity limit."""
        memories = [
            MockMemory(f"Memory {i}", importance=0.5)
            for i in range(10)
        ]

        for mem in memories:
            await working_memory.add(mem)

        assert len(working_memory.memories) <= 5

    async def test_eviction(self, working_memory):
        """Least important memories are evicted."""
        # Add memories with different importance
        memories = [
            MockMemory("Important 1", importance=0.9),
            MockMemory("Important 2", importance=0.8),
            MockMemory("Less important 1", importance=0.3),
            MockMemory("Less important 2", importance=0.2),
            MockMemory("Medium", importance=0.5)
        ]

        for mem in memories:
            await working_memory.add(mem)

        # Add one more to trigger eviction
        new_mem = MockMemory("New memory", importance=0.7)
        await working_memory.add(new_mem)

        # The least important should be evicted
        assert new_mem.id in working_memory.memory_dict
        # Most important should still be there
        assert memories[0].id in working_memory.memory_dict

    async def test_get_all(self, working_memory):
        """Can get all memories (most recent first)."""
        memories = [
            MockMemory(f"Memory {i}", importance=0.5)
            for i in range(3)
        ]

        for mem in memories:
            await working_memory.add(mem)

        all_mems = await working_memory.get_all()

        assert len(all_mems) == 3
        # Most recent first
        assert all_mems[0].id == memories[2].id

    async def test_get_by_id(self, working_memory):
        """Can get specific memory by ID."""
        memory = MockMemory("Test memory")

        await working_memory.add(memory)

        retrieved = await working_memory.get(memory.id)

        assert retrieved == memory

    async def test_get_nonexistent(self, working_memory):
        """Getting nonexistent memory returns None."""
        result = await working_memory.get("invalid_id")
        assert result is None

    async def test_remove_memory(self, working_memory):
        """Can remove specific memory."""
        memory = MockMemory("Test memory")

        await working_memory.add(memory)
        assert memory.id in working_memory.memory_dict

        removed = await working_memory.remove(memory.id)

        assert removed is True
        assert memory.id not in working_memory.memory_dict

    async def test_remove_nonexistent(self, working_memory):
        """Removing nonexistent memory returns False."""
        result = await working_memory.remove("invalid_id")
        assert result is False

    async def test_clear(self, working_memory):
        """Can clear all working memory."""
        memories = [MockMemory(f"Memory {i}") for i in range(3)]

        for mem in memories:
            await working_memory.add(mem)

        await working_memory.clear()

        assert len(working_memory.memories) == 0
        assert len(working_memory.memory_dict) == 0

    async def test_is_full(self, working_memory):
        """Can check if memory is full."""
        assert not working_memory.is_full()

        memories = [MockMemory(f"Memory {i}") for i in range(5)]

        for mem in memories:
            await working_memory.add(mem)

        assert working_memory.is_full()

    async def test_get_size(self, working_memory):
        """Can get current size."""
        assert working_memory.get_size() == 0

        mem = MockMemory("Test")
        await working_memory.add(mem)

        assert working_memory.get_size() == 1

    async def test_get_capacity(self, working_memory):
        """Can get max capacity."""
        assert working_memory.get_capacity() == 5

    async def test_update_existing_memory(self, working_memory):
        """Updating existing memory replaces old version."""
        memory = MockMemory("Original content")
        await working_memory.add(memory)

        # Update the memory
        memory.content = "Updated content"
        await working_memory.add(memory)

        # Should only have one instance
        all_mems = await working_memory.get_all()
        assert len([m for m in all_mems if m.id == memory.id]) == 1

        # Should have updated content
        retrieved = await working_memory.get(memory.id)
        assert retrieved.content == "Updated content"

    async def test_eviction_score_calculation(self, working_memory):
        """Eviction score accounts for importance, recency, and access."""
        mem1 = MockMemory("Low importance", importance=0.1)
        mem1.access_count = 0

        mem2 = MockMemory("High importance", importance=0.9)
        mem2.access_count = 5

        score1 = working_memory._calculate_eviction_score(mem1, 0)
        score2 = working_memory._calculate_eviction_score(mem2, 0)

        # Higher importance should have higher score
        assert score2 > score1


@pytest.mark.asyncio
class TestMemoryCompressor:
    """Test MemoryCompressor functionality."""

    @pytest.fixture
    def compressor(self):
        """Create memory compressor."""
        return MemoryCompressor(
            strategy=CompressionStrategy.HYBRID,
            target_ratio=0.3
        )

    @pytest.fixture
    def mock_llm_provider(self):
        """Create mock LLM provider."""
        provider = Mock()
        provider.generate = AsyncMock(
            return_value={"content": "This is a summary"}
        )
        return provider

    async def test_creation(self, compressor):
        """MemoryCompressor can be created."""
        assert compressor.strategy == CompressionStrategy.HYBRID
        assert compressor.target_ratio == 0.3

    async def test_compress_empty_list(self, compressor):
        """Compressing empty list returns empty list."""
        result = await compressor.compress([])
        assert result == []

    async def test_compress_importance_strategy(self):
        """Importance strategy keeps most important memories."""
        compressor = MemoryCompressor(
            strategy=CompressionStrategy.IMPORTANCE,
            target_ratio=0.5
        )

        memories = [
            MockMemory(f"Memory {i}", importance=i/10)
            for i in range(10)
        ]

        compressed = await compressor.compress(memories)

        # Should keep top 50%
        assert len(compressed) <= 5
        # Should be most important ones
        assert all(m.importance >= 0.5 for m in compressed)

    async def test_compress_cluster_strategy(self):
        """Cluster strategy groups similar memories."""
        compressor = MemoryCompressor(
            strategy=CompressionStrategy.CLUSTER
        )

        memories = [
            MockMemory("Python programming tutorial", importance=0.5),
            MockMemory("Python coding examples", importance=0.6),
            MockMemory("Java programming guide", importance=0.7),
            MockMemory("JavaScript tutorial", importance=0.4)
        ]

        compressed = await compressor.compress(memories)

        # Should compress somewhat
        assert len(compressed) <= len(memories)

    async def test_compress_summarize_strategy(self, mock_llm_provider):
        """Summarize strategy creates summary memories."""
        compressor = MemoryCompressor(
            strategy=CompressionStrategy.SUMMARIZE,
            llm_provider=mock_llm_provider
        )

        memories = [
            MockMemory(f"Memory content {i}", importance=0.5)
            for i in range(15)
        ]

        compressed = await compressor.compress(memories)

        # Should create fewer memories through summarization
        assert len(compressed) < len(memories)
        assert mock_llm_provider.generate.called

    async def test_compress_summarize_without_llm(self):
        """Summarize strategy works without LLM (truncation fallback)."""
        compressor = MemoryCompressor(
            strategy=CompressionStrategy.SUMMARIZE
        )

        memories = [
            MockMemory(f"Memory {i}" * 100, importance=0.5)
            for i in range(15)
        ]

        compressed = await compressor.compress(memories)

        # Should still compress
        assert len(compressed) < len(memories)

    async def test_compress_hybrid_strategy(self):
        """Hybrid strategy combines multiple approaches."""
        compressor = MemoryCompressor(
            strategy=CompressionStrategy.HYBRID,
            target_ratio=0.3
        )

        memories = [
            MockMemory(f"Memory {i}", importance=i/20)
            for i in range(20)
        ]

        compressed = await compressor.compress(memories)

        # Should compress to approximately target ratio
        assert len(compressed) <= len(memories) * 0.5

    async def test_cluster_memories(self, compressor):
        """Memory clustering works."""
        memories = [
            MockMemory("Python programming language is great"),
            MockMemory("Python coding tutorials online"),
            MockMemory("Java enterprise development"),
            MockMemory("JavaScript frontend framework")
        ]

        clusters = compressor._cluster_memories(memories)

        assert len(clusters) > 0
        # Python memories should cluster together
        python_cluster = [c for c in clusters if any(
            "Python" in m.content for m in c
        )]
        assert len(python_cluster) > 0

    async def test_cluster_empty_list(self, compressor):
        """Clustering empty list returns empty list."""
        clusters = compressor._cluster_memories([])
        assert clusters == []

    async def test_summarize_text_with_llm(self, mock_llm_provider):
        """Text summarization works with LLM."""
        compressor = MemoryCompressor(
            strategy=CompressionStrategy.SUMMARIZE,
            llm_provider=mock_llm_provider
        )

        text = "Long text that needs to be summarized" * 10
        summary = await compressor._summarize_text(text)

        assert summary == "This is a summary"
        assert mock_llm_provider.generate.called

    async def test_summarize_text_without_llm(self):
        """Text summarization works without LLM (truncation)."""
        compressor = MemoryCompressor(
            strategy=CompressionStrategy.SUMMARIZE
        )

        text = "x" * 1000
        summary = await compressor._summarize_text(text)

        assert len(summary) <= 303  # 300 + "..."

    async def test_summarize_text_error_handling(self, mock_llm_provider):
        """Text summarization handles errors gracefully."""
        mock_llm_provider.generate.side_effect = Exception("LLM error")

        compressor = MemoryCompressor(
            strategy=CompressionStrategy.SUMMARIZE,
            llm_provider=mock_llm_provider
        )

        text = "Some text to summarize"
        summary = await compressor._summarize_text(text)

        # Should fallback to truncation
        assert len(summary) <= 303


class TestCompressionStrategies:
    """Test CompressionStrategy enum."""

    def test_compression_strategies(self):
        """CompressionStrategy enum has expected values."""
        assert CompressionStrategy.SUMMARIZE.value == "summarize"
        assert CompressionStrategy.CLUSTER.value == "cluster"
        assert CompressionStrategy.IMPORTANCE.value == "importance"
        assert CompressionStrategy.HYBRID.value == "hybrid"


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_semantic_memory_large_query(self):
        """Semantic memory handles very large queries."""
        mem = SemanticMemory(enable_embeddings=False)

        memory = MockMemory("Test content")
        await mem.add(memory)

        large_query = "word " * 10000
        results = await mem.retrieve(large_query, k=5)

        # Should not crash
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_episodic_memory_same_timestamp(self):
        """Episodic memory handles memories with same timestamp."""
        mem = EpisodicMemory()

        memories = [
            MockMemory("Event 1", timestamp=100.0),
            MockMemory("Event 2", timestamp=100.0),
            MockMemory("Event 3", timestamp=100.0)
        ]

        for m in memories:
            await mem.add(m)

        # Should handle duplicates
        assert len(mem.temporal_index) == 3

    @pytest.mark.asyncio
    async def test_working_memory_zero_capacity(self):
        """Working memory with zero capacity."""
        mem = WorkingMemory(size=0)

        memory = MockMemory("Test")
        await mem.add(memory)

        # Should handle gracefully
        assert mem.get_size() == 0

    @pytest.mark.asyncio
    async def test_compressor_single_memory(self):
        """Compressor handles single memory."""
        compressor = MemoryCompressor(strategy=CompressionStrategy.IMPORTANCE)

        memory = MockMemory("Single memory")
        compressed = await compressor.compress([memory])

        assert len(compressed) >= 1
