"""
Semantic Memory - Long-term knowledge storage with vector search.
"""

from typing import List, Optional, Any, Dict
import logging

logger = logging.getLogger(__name__)


class SemanticMemory:
    """
    Semantic memory for storing long-term knowledge.

    Uses vector embeddings for semantic similarity search.
    """

    def __init__(
        self,
        embedding_model: Optional[str] = None,
        enable_embeddings: bool = True
    ):
        """
        Initialize semantic memory.

        Args:
            embedding_model: Model to use for embeddings (required if enable_embeddings=True)
            enable_embeddings: Whether to generate embeddings
        """
        self.embedding_model = embedding_model
        self.enable_embeddings = enable_embeddings
        self.memories: Dict[str, Any] = {}
        self.embeddings: Dict[str, List[float]] = {}

        # Initialize embedding function
        self.embed_fn = None
        self.embed_model = None
        if enable_embeddings:
            self._init_embeddings()

    def _init_embeddings(self) -> None:
        """
        Initialize embedding function.

        Raises:
            ImportError: If no embedding library is available
            ValueError: If OpenAI is used but no model specified
        """
        # First, try sentence-transformers (local)
        try:
            from sentence_transformers import SentenceTransformer
            model_name = "all-MiniLM-L6-v2"  # Fast, efficient model
            self.embed_model = SentenceTransformer(model_name)
            self.embed_fn = self._embed_local
            logger.info(f"Using local embeddings: {model_name}")
            return
        except ImportError:
            logger.debug("sentence-transformers not available, trying OpenAI")

        # Try OpenAI
        try:
            import openai
            if not self.embedding_model:
                raise ValueError(
                    "embedding_model must be specified when using OpenAI embeddings. "
                    "Example: 'text-embedding-ada-002' or 'text-embedding-3-small'"
                )
            self.embed_fn = self._embed_openai
            logger.info(f"Using OpenAI embeddings: {self.embedding_model}")
            return
        except ImportError as e:
            raise ImportError(
                "No embedding library available. Install one of:\n"
                "  - sentence-transformers (local): pip install sentence-transformers\n"
                "  - openai (API-based): pip install openai"
            ) from e

    def _embed_local(self, text: str) -> List[float]:
        """Generate embeddings using local model."""
        embedding = self.embed_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    async def _embed_openai(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI."""
        import openai
        response = await openai.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding

    async def add(self, memory: Any) -> None:
        """
        Add memory to semantic storage.

        Args:
            memory: Memory object to store

        Raises:
            Exception: If embedding generation fails and embeddings are enabled
        """
        self.memories[memory.id] = memory

        # Generate embedding
        if self.embed_fn:
            if asyncio.iscoroutinefunction(self.embed_fn):
                embedding = await self.embed_fn(memory.content)
            else:
                embedding = self.embed_fn(memory.content)
            self.embeddings[memory.id] = embedding
            memory.embedding = embedding

    async def retrieve(self, query: str, k: int = 5) -> List[Any]:
        """
        Retrieve semantically similar memories.

        Args:
            query: Query string
            k: Number of results to return

        Returns:
            List of similar memories

        Note:
            Falls back to keyword search if embeddings are not available or fail.
        """
        if not self.memories:
            return []

        if self.embed_fn and self.embeddings:
            # Use vector similarity
            try:
                if asyncio.iscoroutinefunction(self.embed_fn):
                    query_embedding = await self.embed_fn(query)
                else:
                    query_embedding = self.embed_fn(query)

                similarities = []
                for mem_id, mem_embedding in self.embeddings.items():
                    similarity = self._cosine_similarity(
                        query_embedding, mem_embedding
                    )
                    similarities.append((mem_id, similarity))

                # Sort by similarity
                similarities.sort(key=lambda x: x[1], reverse=True)

                # Return top k memories
                return [self.memories[mem_id] for mem_id, _ in similarities[:k]]

            except Exception as e:
                # This is a legitimate fallback case - vector search failed but keyword search can work
                logger.warning(f"Vector search failed: {e}, falling back to keyword search")

        # Fallback to keyword search (legitimate degraded mode)
        return self._keyword_search(query, k)

    def _keyword_search(self, query: str, k: int) -> List[Any]:
        """Simple keyword-based search fallback."""
        query_words = set(query.lower().split())
        scores = []

        for memory in self.memories.values():
            content_words = set(memory.content.lower().split())
            overlap = len(query_words & content_words)
            if overlap > 0:
                scores.append((memory, overlap))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [mem for mem, _ in scores[:k]]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    async def clear(self) -> None:
        """Clear all semantic memories."""
        self.memories.clear()
        self.embeddings.clear()


import asyncio
