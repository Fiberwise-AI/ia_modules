"""
Edge case tests for rag/core.py to improve coverage
"""

import pytest
from ia_modules.pipeline.test_utils import create_test_execution_context
from ia_modules.rag.core import Document, MemoryVectorStore


class TestDocumentEdgeCases:
    """Test edge cases in Document class"""

    def test_document_with_all_fields(self):
        """Test Document with all fields populated"""
        doc = Document(
            id="doc-001",
            content="This is a test document about machine learning.",
            metadata={"author": "test", "tags": ["ml", "ai"]},
            score=0.95,
            embedding=[0.1, 0.2, 0.3]
        )

        assert doc.id == "doc-001"
        assert doc.content == "This is a test document about machine learning."
        assert doc.metadata["author"] == "test"
        assert doc.score == 0.95
        assert doc.embedding == [0.1, 0.2, 0.3]
        assert "timestamp" in doc.metadata  # Added by __post_init__

    def test_document_with_minimal_fields(self):
        """Test Document with only required fields"""
        doc = Document(
            id="doc-002",
            content="Minimal document"
        )

        assert doc.id == "doc-002"
        assert doc.content == "Minimal document"
        assert doc.metadata == {"timestamp": doc.metadata["timestamp"]}
        assert doc.score is None
        assert doc.embedding is None

    def test_document_auto_adds_timestamp(self):
        """Test that timestamp is automatically added to metadata"""
        doc = Document(id="doc-003", content="test")

        assert "timestamp" in doc.metadata
        assert isinstance(doc.metadata["timestamp"], str)

    def test_document_preserves_existing_timestamp(self):
        """Test that existing timestamp in metadata is not overwritten"""
        custom_timestamp = "2025-01-01T00:00:00"
        doc = Document(
            id="doc-004",
            content="test",
            metadata={"timestamp": custom_timestamp}
        )

        assert doc.metadata["timestamp"] == custom_timestamp

    def test_document_with_empty_content(self):
        """Test Document with empty content string"""
        doc = Document(id="doc-005", content="")

        assert doc.content == ""
        assert "timestamp" in doc.metadata

    def test_document_with_zero_score(self):
        """Test Document with score of 0.0 (valid but low relevance)"""
        doc = Document(
            id="doc-006",
            content="test",
            score=0.0
        )

        assert doc.score == 0.0

    def test_document_with_negative_score(self):
        """Test Document with negative score (edge case)"""
        doc = Document(
            id="doc-007",
            content="test",
            score=-0.5
        )

        assert doc.score == -0.5

    def test_document_with_empty_embedding(self):
        """Test Document with empty embedding list"""
        doc = Document(
            id="doc-008",
            content="test",
            embedding=[]
        )

        assert doc.embedding == []

    def test_document_with_large_embedding(self):
        """Test Document with realistic embedding (1536 dimensions like OpenAI)"""
        embedding = [0.1] * 1536
        doc = Document(
            id="doc-009",
            content="test",
            embedding=embedding
        )

        assert len(doc.embedding) == 1536


class TestMemoryVectorStoreEdgeCases:
    """Test edge cases in MemoryVectorStore"""

    @pytest.mark.asyncio
    async def test_add_documents_to_new_collection(self):
        """Test adding documents creates collection automatically"""
        store = MemoryVectorStore()
        docs = [
            Document(id="1", content="First document"),
            Document(id="2", content="Second document")
        ]

        await store.add_documents(docs, "test_collection")

        collections = await store.list_collections()
        assert "test_collection" in collections
        assert len(store.collections["test_collection"]) == 2

    @pytest.mark.asyncio
    async def test_add_documents_to_existing_collection(self):
        """Test adding documents to existing collection appends them"""
        store = MemoryVectorStore()

        # Add first batch
        docs1 = [Document(id="1", content="First")]
        await store.add_documents(docs1, "test")

        # Add second batch
        docs2 = [Document(id="2", content="Second")]
        await store.add_documents(docs2, "test")

        assert len(store.collections["test"]) == 2

    @pytest.mark.asyncio
    async def test_search_nonexistent_collection(self):
        """Test searching non-existent collection returns empty list"""
        store = MemoryVectorStore()

        results = await store.search("query", "nonexistent")

        assert results == []

    @pytest.mark.asyncio
    async def test_search_empty_collection(self):
        """Test searching empty collection returns empty list"""
        store = MemoryVectorStore()
        store.collections["empty"] = []

        results = await store.search("query", "empty")

        assert results == []

    @pytest.mark.asyncio
    async def test_search_no_matches(self):
        """Test search with no matching documents"""
        store = MemoryVectorStore()
        docs = [
            Document(id="1", content="Dogs and cats"),
            Document(id="2", content="Birds and fish")
        ]
        await store.add_documents(docs, "animals")

        results = await store.search("programming", "animals")

        assert results == []

    @pytest.mark.asyncio
    async def test_search_single_match(self):
        """Test search with single matching document"""
        store = MemoryVectorStore()
        docs = [
            Document(id="1", content="Python programming"),
            Document(id="2", content="Java development")
        ]
        await store.add_documents(docs, "code")

        results = await store.search("python", "code")

        assert len(results) == 1
        assert results[0].id == "1"
        assert results[0].score is not None
        assert results[0].score > 0

    @pytest.mark.asyncio
    async def test_search_multiple_matches_sorted_by_score(self):
        """Test search returns results sorted by relevance score"""
        store = MemoryVectorStore()
        docs = [
            Document(id="1", content="python"),  # Lower score (1 occurrence, 1 word)
            Document(id="2", content="python python python"),  # Higher score (3 occurrences, 3 words)
            Document(id="3", content="python programming")  # Medium score (1 occurrence, 2 words)
        ]
        await store.add_documents(docs, "test")

        results = await store.search("python", "test")

        # Should be sorted by score descending
        assert len(results) == 3
        assert results[0].id == "1"  # python (1/1 = 1.0)
        assert results[1].id == "2"  # python python python (3/3 = 1.0, but might differ by rounding)
        # Scores should be descending or equal
        assert results[0].score >= results[1].score >= results[2].score

    @pytest.mark.asyncio
    async def test_search_respects_limit(self):
        """Test search limit parameter"""
        store = MemoryVectorStore()
        docs = [Document(id=str(i), content="test document") for i in range(10)]
        await store.add_documents(docs, "test")

        results = await store.search("test", "test", limit=3)

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_search_case_insensitive(self):
        """Test search is case-insensitive"""
        store = MemoryVectorStore()
        docs = [Document(id="1", content="Python Programming")]
        await store.add_documents(docs, "test")

        # Search with different case
        results1 = await store.search("PYTHON", "test")
        results2 = await store.search("python", "test")
        results3 = await store.search("PyThOn", "test")

        assert len(results1) == 1
        assert len(results2) == 1
        assert len(results3) == 1

    @pytest.mark.asyncio
    async def test_search_partial_word_match(self):
        """Test search matches partial words (substring match)"""
        store = MemoryVectorStore()
        docs = [Document(id="1", content="programming")]
        await store.add_documents(docs, "test")

        results = await store.search("program", "test")

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_delete_existing_collection(self):
        """Test deleting existing collection"""
        store = MemoryVectorStore()
        docs = [Document(id="1", content="test")]
        await store.add_documents(docs, "to_delete")

        await store.delete_collection("to_delete")

        collections = await store.list_collections()
        assert "to_delete" not in collections

    @pytest.mark.asyncio
    async def test_delete_nonexistent_collection(self):
        """Test deleting non-existent collection doesn't error"""
        store = MemoryVectorStore()

        # Should not raise error
        await store.delete_collection("nonexistent")

        collections = await store.list_collections()
        assert "nonexistent" not in collections

    @pytest.mark.asyncio
    async def test_list_collections_empty(self):
        """Test listing collections when none exist"""
        store = MemoryVectorStore()

        collections = await store.list_collections()

        assert collections == []

    @pytest.mark.asyncio
    async def test_list_collections_multiple(self):
        """Test listing multiple collections"""
        store = MemoryVectorStore()
        await store.add_documents([Document(id="1", content="test")], "col1")
        await store.add_documents([Document(id="2", content="test")], "col2")
        await store.add_documents([Document(id="3", content="test")], "col3")

        collections = await store.list_collections()

        assert len(collections) == 3
        assert "col1" in collections
        assert "col2" in collections
        assert "col3" in collections

    def test_repr(self):
        """Test string representation of MemoryVectorStore"""
        store = MemoryVectorStore()

        repr_str = repr(store)

        assert "MemoryVectorStore" in repr_str
        assert "collections=0" in repr_str
        assert "documents=0" in repr_str

    @pytest.mark.asyncio
    async def test_repr_with_data(self):
        """Test string representation with data"""
        store = MemoryVectorStore()
        await store.add_documents([Document(id="1", content="test")], "col1")
        await store.add_documents([Document(id="2", content="test"), Document(id="3", content="test")], "col2")

        repr_str = repr(store)

        assert "collections=2" in repr_str
        assert "documents=3" in repr_str

    @pytest.mark.asyncio
    async def test_search_with_none_score_handling(self):
        """Test that documents without scores are handled correctly in sorting"""
        store = MemoryVectorStore()
        docs = [Document(id="1", content="python test")]
        await store.add_documents(docs, "test")

        results = await store.search("python", "test")

        # Even if original doc has no score, result should have score
        assert len(results) == 1
        assert results[0].score is not None
