"""Vector database backend for semantic search."""

from typing import Optional, List, Any
import logging

logger = logging.getLogger(__name__)


class VectorBackend:
    """
    Vector database backend using ChromaDB.

    Optimized for semantic similarity search.
    """

    def __init__(self, collection_name: str = "memories", persist_directory: str = "./chroma_db"):
        """
        Initialize vector backend.

        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist database
        """
        try:
            import chromadb
            self.client = chromadb.PersistentClient(path=persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self.available = True
            logger.info("ChromaDB initialized successfully")
        except ImportError:
            logger.warning("ChromaDB not available, falling back to in-memory")
            self.available = False
            self.fallback_storage = {}

    async def store(self, memory: Any) -> None:
        """
        Store a memory with embedding.

        Args:
            memory: Memory object to store
        """
        if not self.available:
            self.fallback_storage[memory.id] = memory
            return

        try:
            # Prepare metadata (must be flat for ChromaDB)
            metadata = {
                "memory_type": memory.memory_type.value,
                "timestamp": memory.timestamp,
                "importance": memory.importance,
                "access_count": memory.access_count,
            }

            # Add embedding if available
            if memory.embedding:
                self.collection.upsert(
                    ids=[memory.id],
                    documents=[memory.content],
                    embeddings=[memory.embedding],
                    metadatas=[metadata]
                )
            else:
                # Let ChromaDB generate embedding
                self.collection.upsert(
                    ids=[memory.id],
                    documents=[memory.content],
                    metadatas=[metadata]
                )
        except Exception as e:
            logger.error(f"Failed to store in vector DB: {e}")

    async def retrieve(self, memory_id: str) -> Optional[Any]:
        """
        Retrieve a memory by ID.

        Args:
            memory_id: Memory ID

        Returns:
            Memory object or None
        """
        if not self.available:
            return self.fallback_storage.get(memory_id)

        try:
            results = self.collection.get(ids=[memory_id])

            if results['ids']:
                return self._result_to_memory(
                    results['ids'][0],
                    results['documents'][0],
                    results['metadatas'][0],
                    results.get('embeddings', [None])[0]
                )
        except Exception as e:
            logger.error(f"Failed to retrieve from vector DB: {e}")

        return None

    async def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter_metadata: Optional[dict] = None
    ) -> List[Any]:
        """
        Search for similar memories by embedding.

        Args:
            query_embedding: Query embedding vector
            k: Number of results
            filter_metadata: Optional metadata filter

        Returns:
            List of similar memories
        """
        if not self.available:
            return list(self.fallback_storage.values())[:k]

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter_metadata
            )

            memories = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    memory = self._result_to_memory(
                        results['ids'][0][i],
                        results['documents'][0][i],
                        results['metadatas'][0][i],
                        results.get('embeddings', [[None]])[0][i]
                    )
                    memories.append(memory)

            return memories
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def delete(self, memory_id: str) -> bool:
        """
        Delete a memory.

        Args:
            memory_id: Memory ID to delete

        Returns:
            True if deleted
        """
        if not self.available:
            if memory_id in self.fallback_storage:
                del self.fallback_storage[memory_id]
                return True
            return False

        try:
            self.collection.delete(ids=[memory_id])
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

    async def list_all(self) -> List[Any]:
        """
        List all memories.

        Returns:
            List of all memories
        """
        if not self.available:
            return list(self.fallback_storage.values())

        try:
            # Get all by querying with large limit
            results = self.collection.get()

            memories = []
            if results['ids']:
                for i in range(len(results['ids'])):
                    memory = self._result_to_memory(
                        results['ids'][i],
                        results['documents'][i],
                        results['metadatas'][i],
                        results.get('embeddings', [None])[i]
                    )
                    memories.append(memory)

            return memories
        except Exception as e:
            logger.error(f"List all failed: {e}")
            return []

    async def clear(self) -> None:
        """Clear all stored memories."""
        if not self.available:
            self.fallback_storage.clear()
            return

        try:
            # Delete collection and recreate
            self.client.delete_collection(name=self.collection.name)
            self.collection = self.client.create_collection(
                name=self.collection.name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            logger.error(f"Clear failed: {e}")

    async def count(self) -> int:
        """
        Count stored memories.

        Returns:
            Number of memories
        """
        if not self.available:
            return len(self.fallback_storage)

        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Count failed: {e}")
            return 0

    def _result_to_memory(
        self,
        memory_id: str,
        document: str,
        metadata: dict,
        embedding: Optional[List[float]]
    ) -> Any:
        """Convert ChromaDB result to Memory object."""
        from ..memory_manager import Memory, MemoryType

        return Memory(
            id=memory_id,
            content=document,
            memory_type=MemoryType(metadata.get("memory_type", "working")),
            timestamp=metadata.get("timestamp", 0.0),
            importance=metadata.get("importance", 0.5),
            access_count=metadata.get("access_count", 0),
            last_accessed=metadata.get("last_accessed"),
            metadata=metadata,
            embedding=embedding
        )
