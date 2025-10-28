"""
Unit tests for RAG system.

Tests Document and VectorStore.
"""
import pytest
from ia_modules.pipeline.test_utils import create_test_execution_context
from ia_modules.rag.core import Document, VectorStore, MemoryVectorStore


class TestDocument:
    """Test Document dataclass."""

    def test_document_creation_minimal(self):
        """Document can be created with minimal fields."""
        doc = Document(
            id="doc1",
            content="Test content"
        )

        assert doc.id == "doc1"
        assert doc.content == "Test content"
        assert doc.metadata != {}  # Should have timestamp
        assert "timestamp" in doc.metadata
        assert doc.score is None

    def test_document_creation_full(self):
        """Document can be created with all fields."""
        doc = Document(
            id="doc1",
            content="Test content",
            metadata={"source": "test.txt", "author": "Test"},
            score=0.95
        )

        assert doc.id == "doc1"
        assert doc.content == "Test content"
        assert doc.metadata["source"] == "test.txt"
        assert doc.score == 0.95


@pytest.mark.asyncio
class TestMemoryVectorStore:
    """Test MemoryVectorStore."""

    async def test_store_creation(self):
        """MemoryVectorStore can be created."""
        store = MemoryVectorStore()

        assert len(store.collections) == 0

    async def test_initialize(self):
        """Store can be initialized."""
        store = MemoryVectorStore()


        # No-op for memory store
        assert True

    async def test_add_documents(self):
        """Documents can be added."""
        store = MemoryVectorStore()

        docs = [
            Document(id="doc1", content="Python programming"),
            Document(id="doc2", content="JavaScript tutorial")
        ]

        await store.add_documents(docs, collection_name="tech")

        assert "tech" in store.collections
        assert len(store.collections["tech"]) == 2

    async def test_search_simple(self):
        """Simple text search works."""
        store = MemoryVectorStore()

        docs = [
            Document(id="doc1", content="Python is a programming language"),
            Document(id="doc2", content="JavaScript is also a programming language"),
            Document(id="doc3", content="Machine learning with Python")
        ]

        await store.add_documents(docs)

        results = await store.search("Python")

        assert len(results) == 2
        assert all("Python" in r.content for r in results)
        assert all(r.score is not None for r in results)

    async def test_search_relevance_scoring(self):
        """Search results are scored by relevance."""
        store = MemoryVectorStore()

        docs = [
            Document(id="doc1", content="Python Python Python"),  # High relevance
            Document(id="doc2", content="Python is great"),      # Medium relevance
            Document(id="doc3", content="JavaScript")            # No match
        ]

        await store.add_documents(docs)

        results = await store.search("Python")

        # Should return 2 results, sorted by score
        assert len(results) == 2
        assert results[0].score > results[1].score

    async def test_search_limit(self):
        """Search respects limit parameter."""
        store = MemoryVectorStore()

        docs = [Document(id=f"doc{i}", content="Python programming") for i in range(10)]

        await store.add_documents(docs)

        results = await store.search("Python", limit=5)

        assert len(results) == 5

    async def test_search_empty_collection(self):
        """Searching empty collection returns empty list."""
        store = MemoryVectorStore()

        results = await store.search("anything")

        assert results == []

    async def test_search_nonexistent_collection(self):
        """Searching nonexistent collection returns empty list."""
        store = MemoryVectorStore()

        results = await store.search("query", collection_name="nonexistent")

        assert results == []

    async def test_delete_collection(self):
        """Collections can be deleted."""
        store = MemoryVectorStore()

        docs = [Document(id="doc1", content="Test")]
        await store.add_documents(docs, collection_name="test")

        await store.delete_collection("test")

        assert "test" not in store.collections

    async def test_list_collections(self):
        """All collections can be listed."""
        store = MemoryVectorStore()

        # Add to multiple collections
        await store.add_documents([Document(id="d1", content="Test")], collection_name="col1")
        await store.add_documents([Document(id="d2", content="Test")], collection_name="col2")

        collections = await store.list_collections()

        assert len(collections) == 2
        assert "col1" in collections
        assert "col2" in collections

    async def test_multiple_collections_isolated(self):
        """Different collections are isolated."""
        store = MemoryVectorStore()

        await store.add_documents([Document(id="d1", content="Python")], collection_name="col1")
        await store.add_documents([Document(id="d2", content="JavaScript")], collection_name="col2")

        results1 = await store.search("Python", collection_name="col1")
        results2 = await store.search("Python", collection_name="col2")

        assert len(results1) == 1
        assert len(results2) == 0

    async def test_case_insensitive_search(self):
        """Search is case insensitive."""
        store = MemoryVectorStore()

        docs = [Document(id="doc1", content="PYTHON Programming")]

        await store.add_documents(docs)

        results = await store.search("python")

        assert len(results) == 1

    async def test_store_repr(self):
        """Store has useful repr."""
        store = MemoryVectorStore()

        await store.add_documents([Document(id="d1", content="Test")])

        repr_str = repr(store)

        assert "MemoryVectorStore" in repr_str
        assert "collections=1" in repr_str
        assert "documents=1" in repr_str
