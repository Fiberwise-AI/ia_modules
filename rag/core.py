"""
Core RAG components.

Provides Document class and VectorStore interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime


@dataclass
class Document:
    """
    Represents a document in RAG system.

    Attributes:
        id: Unique document identifier
        content: Document text content
        metadata: Additional document metadata
        score: Similarity score (for search results)
        embedding: Vector embedding (optional)
    """
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None
    embedding: Optional[List[float]] = None

    def __post_init__(self):
        """Add timestamp if not in metadata."""
        if "timestamp" not in self.metadata:
            self.metadata["timestamp"] = datetime.now().isoformat()


class VectorStore(ABC):
    """
    Abstract vector database interface.

    Implementations:
    - MemoryVectorStore: In-memory storage (testing)
    - ChromaDBStore: ChromaDB backend (production)
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize vector store."""
        pass

    @abstractmethod
    async def add_documents(self, documents: List[Document],
                           collection_name: str = "default") -> None:
        """
        Add documents to vector store.

        Args:
            documents: List of documents to add
            collection_name: Collection to add to
        """
        pass

    @abstractmethod
    async def search(self, query: str, collection_name: str = "default",
                    limit: int = 5) -> List[Document]:
        """
        Search for similar documents.

        Args:
            query: Search query
            collection_name: Collection to search
            limit: Maximum results

        Returns:
            List of documents with similarity scores
        """
        pass

    @abstractmethod
    async def delete_collection(self, collection_name: str) -> None:
        """
        Delete a collection.

        Args:
            collection_name: Collection to delete
        """
        pass

    @abstractmethod
    async def list_collections(self) -> List[str]:
        """
        List all collections.

        Returns:
            List of collection names
        """
        pass


class MemoryVectorStore(VectorStore):
    """
    In-memory vector store for testing.

    Uses simple text matching (no real embeddings).
    """

    def __init__(self):
        """Initialize memory store."""
        self.collections: Dict[str, List[Document]] = {}

    async def initialize(self) -> None:
        """Initialize store (no-op for memory)."""
        pass

    async def add_documents(self, documents: List[Document],
                           collection_name: str = "default") -> None:
        """Add documents to collection."""
        if collection_name not in self.collections:
            self.collections[collection_name] = []

        self.collections[collection_name].extend(documents)

    async def search(self, query: str, collection_name: str = "default",
                    limit: int = 5) -> List[Document]:
        """
        Search using simple text matching.

        Args:
            query: Search query
            collection_name: Collection to search
            limit: Maximum results

        Returns:
            Matching documents with scores
        """
        if collection_name not in self.collections:
            return []

        query_lower = query.lower()
        results = []

        for doc in self.collections[collection_name]:
            # Simple relevance scoring
            content_lower = doc.content.lower()

            if query_lower in content_lower:
                # Count occurrences as score
                score = content_lower.count(query_lower) / len(content_lower.split())

                # Create result document with score
                result = Document(
                    id=doc.id,
                    content=doc.content,
                    metadata=doc.metadata,
                    score=score
                )
                results.append(result)

        # Sort by score descending
        results.sort(key=lambda d: d.score or 0, reverse=True)

        return results[:limit]

    async def delete_collection(self, collection_name: str) -> None:
        """Delete collection."""
        if collection_name in self.collections:
            del self.collections[collection_name]

    async def list_collections(self) -> List[str]:
        """List all collections."""
        return list(self.collections.keys())

    def __repr__(self) -> str:
        num_docs = sum(len(docs) for docs in self.collections.values())
        return f"<MemoryVectorStore(collections={len(self.collections)}, documents={num_docs})>"
