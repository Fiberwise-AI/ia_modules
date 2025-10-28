# Embedding Management Implementation Plan

## Overview

This document provides a comprehensive implementation plan for embedding generation, caching, and management within the ia_modules pipeline system. Efficient embedding management is critical for RAG applications, semantic search, and AI-powered features.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Embedding Provider Interface](#embedding-provider-interface)
3. [OpenAI Embeddings](#openai-embeddings)
4. [HuggingFace Embeddings](#huggingface-embeddings)
5. [Cohere Embeddings](#cohere-embeddings)
6. [Embedding Cache Layer](#embedding-cache-layer)
7. [Batch Embedding Generation](#batch-embedding-generation)
8. [Embedding Monitoring & Cost Tracking](#embedding-monitoring--cost-tracking)
9. [Pipeline Integration](#pipeline-integration)
10. [Testing Strategy](#testing-strategy)

---

## 1. Architecture Overview

### 1.1 Design Principles

- **Provider Agnostic**: Abstract interface for multiple embedding providers
- **Intelligent Caching**: Multi-tier cache (memory, Redis, database) to minimize API calls
- **Cost Optimization**: Track usage, implement quotas, batch processing
- **Performance**: Async operations, connection pooling, parallel processing
- **Flexibility**: Support for different models and dimensions per use case

### 1.2 Component Architecture

```
ia_modules/
├── embeddings/
│   ├── __init__.py
│   ├── base.py              # Abstract interface
│   ├── models.py            # Data models
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── openai.py        # OpenAI implementation
│   │   ├── huggingface.py   # HuggingFace implementation
│   │   └── cohere.py        # Cohere implementation
│   ├── cache.py             # Multi-tier cache
│   ├── batch.py             # Batch processing
│   ├── monitoring.py        # Usage tracking
│   └── factory.py           # Provider factory
├── pipeline/
│   └── steps/
│       └── embed_text.py    # Pipeline step
└── tests/
    └── unit/
        └── test_embeddings.py
```

---

## 2. Embedding Provider Interface

### 2.1 Data Models

**File**: `ia_modules/embeddings/models.py`

```python
"""Data models for embedding operations."""
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"


class EmbeddingModel(str, Enum):
    """Pre-defined embedding models."""
    # OpenAI
    OPENAI_SMALL = "text-embedding-3-small"
    OPENAI_LARGE = "text-embedding-3-large"
    OPENAI_ADA = "text-embedding-ada-002"

    # HuggingFace
    SENTENCE_TRANSFORMER_MINI = "sentence-transformers/all-MiniLM-L6-v2"
    SENTENCE_TRANSFORMER_MPNet = "sentence-transformers/all-mpnet-base-v2"
    INSTRUCTOR = "hkunlp/instructor-large"

    # Cohere
    COHERE_ENGLISH = "embed-english-v3.0"
    COHERE_MULTILINGUAL = "embed-multilingual-v3.0"


class EmbeddingRequest(BaseModel):
    """Request for generating embeddings."""
    texts: List[str] = Field(..., description="Texts to embed", min_items=1)
    model: str = Field(..., description="Model identifier")
    task_type: Optional[str] = Field(None, description="Task type hint (search_document, search_query, classification)")
    normalize: bool = Field(True, description="Normalize embeddings to unit length")
    dimensions: Optional[int] = Field(None, description="Target dimension (if supported)")

    class Config:
        json_schema_extra = {
            "example": {
                "texts": ["Hello world", "Embedding test"],
                "model": "text-embedding-3-small",
                "normalize": True
            }
        }


class EmbeddingResponse(BaseModel):
    """Response from embedding generation."""
    embeddings: List[List[float]] = Field(..., description="Generated embeddings")
    model: str = Field(..., description="Model used")
    dimensions: int = Field(..., description="Embedding dimension")
    usage: Dict[str, int] = Field(default_factory=dict, description="Token/request usage")
    cached: List[bool] = Field(default_factory=list, description="Which embeddings were cached")
    generation_time_ms: float = Field(..., description="Time taken to generate")

    class Config:
        json_schema_extra = {
            "example": {
                "embeddings": [[0.1, 0.2, 0.3]],
                "model": "text-embedding-3-small",
                "dimensions": 1536,
                "usage": {"tokens": 5},
                "cached": [False],
                "generation_time_ms": 123.45
            }
        }


class EmbeddingConfig(BaseModel):
    """Configuration for embedding provider."""
    provider: EmbeddingProvider = Field(..., description="Provider name")
    model: str = Field(..., description="Model identifier")
    api_key: Optional[str] = Field(None, description="API key")
    api_base: Optional[str] = Field(None, description="Custom API base URL")
    dimensions: Optional[int] = Field(None, description="Embedding dimension")

    # Performance
    batch_size: int = Field(100, description="Max texts per batch", gt=0)
    max_concurrent: int = Field(5, description="Max concurrent requests", gt=0)
    timeout: float = Field(30.0, description="Request timeout in seconds")

    # Caching
    enable_cache: bool = Field(True, description="Enable embedding cache")
    cache_ttl: int = Field(86400, description="Cache TTL in seconds (24 hours)")

    # Cost control
    max_tokens_per_day: Optional[int] = Field(None, description="Daily token quota")
    cost_per_1k_tokens: Optional[float] = Field(None, description="Cost per 1000 tokens")

    class Config:
        json_schema_extra = {
            "example": {
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key": "sk-...",
                "dimensions": 1536,
                "batch_size": 100,
                "enable_cache": True
            }
        }


class EmbeddingUsageStats(BaseModel):
    """Usage statistics for embeddings."""
    provider: str
    model: str
    total_requests: int = 0
    total_tokens: int = 0
    total_embeddings: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_cost: float = 0.0
    period_start: datetime
    period_end: datetime

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
```

### 2.2 Abstract Base Class

**File**: `ia_modules/embeddings/base.py`

```python
"""Abstract base class for embedding providers."""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import time
import hashlib
from .models import EmbeddingRequest, EmbeddingResponse, EmbeddingConfig


class EmbeddingProviderBase(ABC):
    """Abstract base class for embedding providers."""

    def __init__(self, config: EmbeddingConfig):
        """
        Initialize embedding provider.

        Args:
            config: Provider configuration
        """
        self.config = config
        self._usage_stats = {
            "requests": 0,
            "tokens": 0,
            "embeddings": 0,
            "cost": 0.0
        }

    @abstractmethod
    async def generate_embeddings(
        self,
        texts: List[str],
        **kwargs
    ) -> EmbeddingResponse:
        """
        Generate embeddings for texts.

        Args:
            texts: List of texts to embed
            **kwargs: Provider-specific parameters

        Returns:
            Embedding response with vectors
        """
        pass

    @abstractmethod
    async def get_model_dimension(self) -> int:
        """
        Get embedding dimension for configured model.

        Returns:
            Embedding dimension
        """
        pass

    @abstractmethod
    async def validate_config(self) -> bool:
        """
        Validate provider configuration.

        Returns:
            True if valid, raises exception otherwise
        """
        pass

    @staticmethod
    def hash_text(text: str, model: str) -> str:
        """
        Generate cache key for text and model.

        Args:
            text: Input text
            model: Model identifier

        Returns:
            Cache key hash
        """
        key = f"{model}:{text}"
        return hashlib.sha256(key.encode()).hexdigest()

    @staticmethod
    def normalize_embedding(embedding: List[float]) -> List[float]:
        """
        Normalize embedding to unit length.

        Args:
            embedding: Input vector

        Returns:
            Normalized vector
        """
        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude == 0:
            return embedding
        return [x / magnitude for x in embedding]

    def track_usage(
        self,
        tokens: int,
        embeddings: int,
        cost: float = 0.0
    ) -> None:
        """
        Track usage statistics.

        Args:
            tokens: Number of tokens processed
            embeddings: Number of embeddings generated
            cost: Estimated cost
        """
        self._usage_stats["requests"] += 1
        self._usage_stats["tokens"] += tokens
        self._usage_stats["embeddings"] += embeddings
        self._usage_stats["cost"] += cost

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        return self._usage_stats.copy()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.validate_config()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        return False


class EmbeddingError(Exception):
    """Base exception for embedding operations."""
    pass


class ModelNotSupportedError(EmbeddingError):
    """Model not supported by provider."""
    pass


class QuotaExceededError(EmbeddingError):
    """API quota exceeded."""
    pass


class InvalidTextError(EmbeddingError):
    """Invalid text input."""
    pass
```

---

## 3. OpenAI Embeddings

### 3.1 OpenAI Implementation

**File**: `ia_modules/embeddings/providers/openai.py`

```python
"""OpenAI embedding provider implementation."""
from typing import List, Optional, Dict, Any
import asyncio
from openai import AsyncOpenAI
from ..base import EmbeddingProviderBase, ModelNotSupportedError, QuotaExceededError
from ..models import EmbeddingResponse, EmbeddingConfig, EmbeddingModel


class OpenAIEmbeddingProvider(EmbeddingProviderBase):
    """OpenAI embedding provider."""

    # Model dimensions
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536
    }

    # Costs per 1M tokens (as of 2024)
    MODEL_COSTS = {
        "text-embedding-3-small": 0.02,
        "text-embedding-3-large": 0.13,
        "text-embedding-ada-002": 0.10
    }

    def __init__(self, config: EmbeddingConfig):
        """Initialize OpenAI provider."""
        super().__init__(config)
        self._client: Optional[AsyncOpenAI] = None

    def _get_client(self) -> AsyncOpenAI:
        """Get or create OpenAI client."""
        if not self._client:
            self._client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base,
                timeout=self.config.timeout
            )
        return self._client

    async def generate_embeddings(
        self,
        texts: List[str],
        **kwargs
    ) -> EmbeddingResponse:
        """Generate embeddings using OpenAI API."""
        if not texts:
            raise InvalidTextError("No texts provided")

        start_time = time.time()
        client = self._get_client()

        try:
            # Call OpenAI API
            response = await client.embeddings.create(
                model=self.config.model,
                input=texts,
                dimensions=self.config.dimensions,
                encoding_format="float"
            )

            # Extract embeddings
            embeddings = [item.embedding for item in response.data]

            # Normalize if requested
            if kwargs.get("normalize", True):
                embeddings = [
                    self.normalize_embedding(emb) for emb in embeddings
                ]

            # Calculate cost
            tokens = response.usage.total_tokens
            cost_per_1m = self.MODEL_COSTS.get(self.config.model, 0.0)
            cost = (tokens / 1_000_000) * cost_per_1m

            # Track usage
            self.track_usage(tokens=tokens, embeddings=len(embeddings), cost=cost)

            # Build response
            return EmbeddingResponse(
                embeddings=embeddings,
                model=self.config.model,
                dimensions=len(embeddings[0]) if embeddings else 0,
                usage={"tokens": tokens},
                cached=[False] * len(embeddings),
                generation_time_ms=(time.time() - start_time) * 1000
            )

        except Exception as e:
            if "quota" in str(e).lower() or "rate_limit" in str(e).lower():
                raise QuotaExceededError(f"OpenAI quota exceeded: {e}")
            raise EmbeddingError(f"OpenAI embedding generation failed: {e}")

    async def get_model_dimension(self) -> int:
        """Get dimension for configured model."""
        if self.config.dimensions:
            return self.config.dimensions

        base_dim = self.MODEL_DIMENSIONS.get(self.config.model)
        if not base_dim:
            raise ModelNotSupportedError(
                f"Unknown model: {self.config.model}"
            )
        return base_dim

    async def validate_config(self) -> bool:
        """Validate OpenAI configuration."""
        if not self.config.api_key:
            raise ValueError("OpenAI API key required")

        if self.config.model not in self.MODEL_DIMENSIONS:
            raise ModelNotSupportedError(
                f"Model {self.config.model} not supported. "
                f"Supported: {list(self.MODEL_DIMENSIONS.keys())}"
            )

        # Validate custom dimensions
        if self.config.dimensions:
            base_dim = self.MODEL_DIMENSIONS[self.config.model]
            if self.config.dimensions > base_dim:
                raise ValueError(
                    f"Dimensions {self.config.dimensions} exceed "
                    f"model maximum {base_dim}"
                )

        return True

    async def close(self) -> None:
        """Close OpenAI client."""
        if self._client:
            await self._client.close()
            self._client = None
```

### 3.2 OpenAI Usage Example

```python
"""Example: Using OpenAI embeddings."""
import asyncio
from ia_modules.embeddings.providers.openai import OpenAIEmbeddingProvider
from ia_modules.embeddings.models import EmbeddingConfig, EmbeddingProvider


async def main():
    # Configure OpenAI provider
    config = EmbeddingConfig(
        provider=EmbeddingProvider.OPENAI,
        model="text-embedding-3-small",
        api_key="sk-...",
        dimensions=1536,
        batch_size=100
    )

    async with OpenAIEmbeddingProvider(config) as provider:
        # Generate embeddings
        texts = [
            "What is machine learning?",
            "How does neural networks work?",
            "Explain deep learning"
        ]

        response = await provider.generate_embeddings(
            texts=texts,
            normalize=True
        )

        print(f"Generated {len(response.embeddings)} embeddings")
        print(f"Dimensions: {response.dimensions}")
        print(f"Tokens used: {response.usage['tokens']}")
        print(f"Time: {response.generation_time_ms:.2f}ms")

        # Check usage stats
        stats = provider.get_usage_stats()
        print(f"Total cost: ${stats['cost']:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 4. HuggingFace Embeddings

### 4.1 HuggingFace Implementation

**File**: `ia_modules/embeddings/providers/huggingface.py`

```python
"""HuggingFace embedding provider implementation."""
from typing import List, Optional, Dict, Any
import asyncio
import torch
from sentence_transformers import SentenceTransformer
from ..base import EmbeddingProviderBase, ModelNotSupportedError
from ..models import EmbeddingResponse, EmbeddingConfig


class HuggingFaceEmbeddingProvider(EmbeddingProviderBase):
    """HuggingFace embedding provider using sentence-transformers."""

    def __init__(self, config: EmbeddingConfig):
        """Initialize HuggingFace provider."""
        super().__init__(config)
        self._model: Optional[SentenceTransformer] = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self) -> SentenceTransformer:
        """Load sentence transformer model."""
        if not self._model:
            self._model = SentenceTransformer(
                self.config.model,
                device=self._device
            )
        return self._model

    async def generate_embeddings(
        self,
        texts: List[str],
        **kwargs
    ) -> EmbeddingResponse:
        """Generate embeddings using HuggingFace model."""
        if not texts:
            raise InvalidTextError("No texts provided")

        start_time = time.time()
        model = self._load_model()

        try:
            # Generate embeddings (run in thread pool to not block)
            embeddings = await asyncio.to_thread(
                model.encode,
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=kwargs.get("normalize", True)
            )

            # Convert to list of lists
            embeddings_list = [emb.tolist() for emb in embeddings]

            # Estimate tokens (rough approximation)
            total_chars = sum(len(text) for text in texts)
            tokens = total_chars // 4  # Rough estimate

            # Track usage (HuggingFace is free, so cost = 0)
            self.track_usage(tokens=tokens, embeddings=len(embeddings_list), cost=0.0)

            return EmbeddingResponse(
                embeddings=embeddings_list,
                model=self.config.model,
                dimensions=len(embeddings_list[0]) if embeddings_list else 0,
                usage={"tokens": tokens},
                cached=[False] * len(embeddings_list),
                generation_time_ms=(time.time() - start_time) * 1000
            )

        except Exception as e:
            raise EmbeddingError(f"HuggingFace embedding generation failed: {e}")

    async def get_model_dimension(self) -> int:
        """Get embedding dimension."""
        model = self._load_model()
        return model.get_sentence_embedding_dimension()

    async def validate_config(self) -> bool:
        """Validate HuggingFace configuration."""
        try:
            # Try to load model
            self._load_model()
            return True
        except Exception as e:
            raise ModelNotSupportedError(
                f"Failed to load model {self.config.model}: {e}"
            )

    def unload_model(self) -> None:
        """Unload model from memory."""
        if self._model:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
```

### 4.2 HuggingFace Usage Example

```python
"""Example: Using HuggingFace embeddings."""
import asyncio
from ia_modules.embeddings.providers.huggingface import HuggingFaceEmbeddingProvider
from ia_modules.embeddings.models import EmbeddingConfig, EmbeddingProvider


async def main():
    # Configure HuggingFace provider
    config = EmbeddingConfig(
        provider=EmbeddingProvider.HUGGINGFACE,
        model="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=32
    )

    async with HuggingFaceEmbeddingProvider(config) as provider:
        # Generate embeddings
        texts = [
            "The cat sat on the mat",
            "A feline rested on a rug",
            "Machine learning is fascinating"
        ]

        response = await provider.generate_embeddings(
            texts=texts,
            normalize=True
        )

        print(f"Generated {len(response.embeddings)} embeddings")
        print(f"Dimensions: {response.dimensions}")
        print(f"Time: {response.generation_time_ms:.2f}ms")
        print(f"Device: {provider._device}")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 5. Cohere Embeddings

### 5.1 Cohere Implementation

**File**: `ia_modules/embeddings/providers/cohere.py`

```python
"""Cohere embedding provider implementation."""
from typing import List, Optional, Dict, Any
import asyncio
import cohere
from cohere import AsyncClient
from ..base import EmbeddingProviderBase, QuotaExceededError
from ..models import EmbeddingResponse, EmbeddingConfig


class CohereEmbeddingProvider(EmbeddingProviderBase):
    """Cohere embedding provider."""

    MODEL_DIMENSIONS = {
        "embed-english-v3.0": 1024,
        "embed-multilingual-v3.0": 1024,
        "embed-english-light-v3.0": 384,
        "embed-multilingual-light-v3.0": 384
    }

    # Costs per 1M tokens
    MODEL_COSTS = {
        "embed-english-v3.0": 0.10,
        "embed-multilingual-v3.0": 0.10,
        "embed-english-light-v3.0": 0.10,
        "embed-multilingual-light-v3.0": 0.10
    }

    def __init__(self, config: EmbeddingConfig):
        """Initialize Cohere provider."""
        super().__init__(config)
        self._client: Optional[AsyncClient] = None

    def _get_client(self) -> AsyncClient:
        """Get or create Cohere client."""
        if not self._client:
            self._client = cohere.AsyncClient(
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )
        return self._client

    async def generate_embeddings(
        self,
        texts: List[str],
        **kwargs
    ) -> EmbeddingResponse:
        """Generate embeddings using Cohere API."""
        if not texts:
            raise InvalidTextError("No texts provided")

        start_time = time.time()
        client = self._get_client()

        # Determine input type
        input_type = kwargs.get("task_type", "search_document")
        if input_type == "search_query":
            input_type = "search_query"
        else:
            input_type = "search_document"

        try:
            # Call Cohere API
            response = await client.embed(
                texts=texts,
                model=self.config.model,
                input_type=input_type,
                embedding_types=["float"]
            )

            # Extract embeddings
            embeddings = response.embeddings.float

            # Normalize if requested
            if kwargs.get("normalize", True):
                embeddings = [
                    self.normalize_embedding(emb) for emb in embeddings
                ]

            # Calculate cost
            # Cohere charges per text, approximate tokens
            total_chars = sum(len(text) for text in texts)
            tokens = total_chars // 4
            cost_per_1m = self.MODEL_COSTS.get(self.config.model, 0.0)
            cost = (tokens / 1_000_000) * cost_per_1m

            # Track usage
            self.track_usage(tokens=tokens, embeddings=len(embeddings), cost=cost)

            return EmbeddingResponse(
                embeddings=embeddings,
                model=self.config.model,
                dimensions=len(embeddings[0]) if embeddings else 0,
                usage={"tokens": tokens},
                cached=[False] * len(embeddings),
                generation_time_ms=(time.time() - start_time) * 1000
            )

        except Exception as e:
            if "quota" in str(e).lower() or "rate" in str(e).lower():
                raise QuotaExceededError(f"Cohere quota exceeded: {e}")
            raise EmbeddingError(f"Cohere embedding generation failed: {e}")

    async def get_model_dimension(self) -> int:
        """Get dimension for configured model."""
        dim = self.MODEL_DIMENSIONS.get(self.config.model)
        if not dim:
            raise ModelNotSupportedError(
                f"Unknown model: {self.config.model}"
            )
        return dim

    async def validate_config(self) -> bool:
        """Validate Cohere configuration."""
        if not self.config.api_key:
            raise ValueError("Cohere API key required")

        if self.config.model not in self.MODEL_DIMENSIONS:
            raise ModelNotSupportedError(
                f"Model {self.config.model} not supported. "
                f"Supported: {list(self.MODEL_DIMENSIONS.keys())}"
            )

        return True

    async def close(self) -> None:
        """Close Cohere client."""
        if self._client:
            await self._client.close()
            self._client = None
```

---

## 6. Embedding Cache Layer

### 6.1 Multi-Tier Cache Implementation

**File**: `ia_modules/embeddings/cache.py`

```python
"""Multi-tier caching for embeddings."""
from typing import List, Optional, Dict, Any, Tuple
import asyncio
import json
import hashlib
from datetime import datetime, timedelta
import redis.asyncio as redis
from ..database.interfaces import DatabaseInterface


class EmbeddingCache:
    """Multi-tier cache for embeddings (Memory -> Redis -> Database)."""

    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        database: Optional[DatabaseInterface] = None,
        memory_size: int = 1000,
        ttl_seconds: int = 86400
    ):
        """
        Initialize embedding cache.

        Args:
            redis_client: Optional Redis client for L2 cache
            database: Optional database for L3 cache
            memory_size: Max items in memory cache
            ttl_seconds: Time to live for cached items
        """
        self._redis = redis_client
        self._db = database
        self._memory_cache: Dict[str, Tuple[List[float], datetime]] = {}
        self._memory_size = memory_size
        self._ttl = timedelta(seconds=ttl_seconds)

        # Stats
        self._hits = {"memory": 0, "redis": 0, "db": 0}
        self._misses = 0

    @staticmethod
    def _cache_key(text: str, model: str) -> str:
        """Generate cache key."""
        key = f"emb:{model}:{text}"
        return hashlib.sha256(key.encode()).hexdigest()

    async def get(
        self,
        text: str,
        model: str
    ) -> Optional[List[float]]:
        """
        Get embedding from cache.

        Args:
            text: Input text
            model: Model identifier

        Returns:
            Cached embedding or None
        """
        cache_key = self._cache_key(text, model)

        # L1: Memory cache
        if cache_key in self._memory_cache:
            embedding, cached_at = self._memory_cache[cache_key]
            if datetime.now() - cached_at < self._ttl:
                self._hits["memory"] += 1
                return embedding
            else:
                # Expired
                del self._memory_cache[cache_key]

        # L2: Redis cache
        if self._redis:
            try:
                data = await self._redis.get(f"embedding:{cache_key}")
                if data:
                    embedding = json.loads(data)
                    self._hits["redis"] += 1
                    # Promote to memory cache
                    self._put_memory(cache_key, embedding)
                    return embedding
            except Exception:
                pass  # Fall through to L3

        # L3: Database cache
        if self._db:
            try:
                result = await self._db.fetch_one(
                    """
                    SELECT embedding
                    FROM embedding_cache
                    WHERE cache_key = :key
                    AND cached_at > :min_time
                    """,
                    {
                        "key": cache_key,
                        "min_time": datetime.now() - self._ttl
                    }
                )
                if result:
                    embedding = json.loads(result["embedding"])
                    self._hits["db"] += 1
                    # Promote to Redis and memory
                    await self._put_redis(cache_key, embedding)
                    self._put_memory(cache_key, embedding)
                    return embedding
            except Exception:
                pass

        self._misses += 1
        return None

    async def put(
        self,
        text: str,
        model: str,
        embedding: List[float]
    ) -> None:
        """
        Store embedding in cache.

        Args:
            text: Input text
            model: Model identifier
            embedding: Embedding vector
        """
        cache_key = self._cache_key(text, model)

        # Store in all tiers
        self._put_memory(cache_key, embedding)
        await self._put_redis(cache_key, embedding)
        await self._put_db(cache_key, text, model, embedding)

    def _put_memory(self, cache_key: str, embedding: List[float]) -> None:
        """Store in memory cache with LRU eviction."""
        if len(self._memory_cache) >= self._memory_size:
            # Simple LRU: remove oldest
            oldest_key = min(
                self._memory_cache.keys(),
                key=lambda k: self._memory_cache[k][1]
            )
            del self._memory_cache[oldest_key]

        self._memory_cache[cache_key] = (embedding, datetime.now())

    async def _put_redis(self, cache_key: str, embedding: List[float]) -> None:
        """Store in Redis cache."""
        if not self._redis:
            return

        try:
            await self._redis.setex(
                f"embedding:{cache_key}",
                self._ttl.total_seconds(),
                json.dumps(embedding)
            )
        except Exception:
            pass  # Non-critical

    async def _put_db(
        self,
        cache_key: str,
        text: str,
        model: str,
        embedding: List[float]
    ) -> None:
        """Store in database cache."""
        if not self._db:
            return

        try:
            await self._db.execute(
                """
                INSERT INTO embedding_cache (cache_key, text, model, embedding, cached_at)
                VALUES (:key, :text, :model, :embedding, :cached_at)
                ON CONFLICT (cache_key) DO UPDATE SET
                    embedding = :embedding,
                    cached_at = :cached_at
                """,
                {
                    "key": cache_key,
                    "text": text,
                    "model": model,
                    "embedding": json.dumps(embedding),
                    "cached_at": datetime.now()
                }
            )
        except Exception:
            pass  # Non-critical

    async def get_batch(
        self,
        texts: List[str],
        model: str
    ) -> Tuple[List[Optional[List[float]]], List[int]]:
        """
        Get embeddings for batch of texts.

        Args:
            texts: List of texts
            model: Model identifier

        Returns:
            Tuple of (embeddings, missing_indices)
            - embeddings: List with cached embeddings or None
            - missing_indices: Indices of texts not in cache
        """
        results: List[Optional[List[float]]] = [None] * len(texts)
        missing_indices: List[int] = []

        for i, text in enumerate(texts):
            embedding = await self.get(text, model)
            if embedding:
                results[i] = embedding
            else:
                missing_indices.append(i)

        return results, missing_indices

    async def put_batch(
        self,
        texts: List[str],
        model: str,
        embeddings: List[List[float]]
    ) -> None:
        """
        Store batch of embeddings.

        Args:
            texts: List of texts
            model: Model identifier
            embeddings: List of embeddings
        """
        tasks = [
            self.put(text, model, embedding)
            for text, embedding in zip(texts, embeddings)
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_hits = sum(self._hits.values())
        total_requests = total_hits + self._misses

        return {
            "hits": self._hits.copy(),
            "misses": self._misses,
            "hit_rate": total_hits / total_requests if total_requests > 0 else 0.0,
            "memory_size": len(self._memory_cache),
            "memory_max": self._memory_size
        }

    async def clear(self) -> None:
        """Clear all cache tiers."""
        self._memory_cache.clear()

        if self._redis:
            # Clear Redis keys matching pattern
            cursor = 0
            while True:
                cursor, keys = await self._redis.scan(
                    cursor=cursor,
                    match="embedding:*",
                    count=100
                )
                if keys:
                    await self._redis.delete(*keys)
                if cursor == 0:
                    break

        if self._db:
            await self._db.execute("DELETE FROM embedding_cache")
```

### 6.2 Cache Schema Migration

**File**: `ia_modules/database/migrations/V004__embedding_cache.sql`

```sql
-- Embedding cache table
CREATE TABLE IF NOT EXISTS embedding_cache (
    cache_key VARCHAR(64) PRIMARY KEY,
    text TEXT NOT NULL,
    model VARCHAR(100) NOT NULL,
    embedding TEXT NOT NULL,  -- JSON array
    cached_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_cached_at (cached_at),
    INDEX idx_model (model)
);

-- Cleanup old entries (can be run periodically)
-- DELETE FROM embedding_cache WHERE cached_at < DATE_SUB(NOW(), INTERVAL 7 DAY);
```

---

## 7. Batch Embedding Generation

### 7.1 Batch Processor with Caching

**File**: `ia_modules/embeddings/batch.py`

```python
"""Batch processing for embedding generation."""
from typing import List, Optional, Callable, Awaitable
import asyncio
from .base import EmbeddingProviderBase
from .cache import EmbeddingCache
from .models import EmbeddingResponse


class BatchEmbeddingProcessor:
    """Batch processor for efficient embedding generation with caching."""

    def __init__(
        self,
        provider: EmbeddingProviderBase,
        cache: Optional[EmbeddingCache] = None,
        batch_size: int = 100,
        max_concurrent: int = 5
    ):
        """
        Initialize batch processor.

        Args:
            provider: Embedding provider
            cache: Optional cache layer
            batch_size: Texts per batch
            max_concurrent: Max concurrent batches
        """
        self.provider = provider
        self.cache = cache
        self.batch_size = batch_size
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def generate_batch(
        self,
        texts: List[str],
        progress_callback: Optional[Callable[[int, int], Awaitable[None]]] = None,
        **kwargs
    ) -> EmbeddingResponse:
        """
        Generate embeddings for batch with caching.

        Args:
            texts: List of texts
            progress_callback: Async callback for progress
            **kwargs: Provider-specific parameters

        Returns:
            Complete embedding response
        """
        total = len(texts)
        all_embeddings: List[Optional[List[float]]] = [None] * total
        total_tokens = 0
        cached_flags = [False] * total

        # Check cache first
        if self.cache:
            cached_embeddings, missing_indices = await self.cache.get_batch(
                texts=texts,
                model=self.provider.config.model
            )

            for i, emb in enumerate(cached_embeddings):
                if emb:
                    all_embeddings[i] = emb
                    cached_flags[i] = True

            texts_to_generate = [texts[i] for i in missing_indices]
        else:
            missing_indices = list(range(total))
            texts_to_generate = texts

        # Generate missing embeddings in batches
        if texts_to_generate:
            async def process_batch(batch_texts: List[str], batch_indices: List[int]):
                """Process single batch."""
                async with self._semaphore:
                    response = await self.provider.generate_embeddings(
                        texts=batch_texts,
                        **kwargs
                    )

                    # Store in cache
                    if self.cache:
                        await self.cache.put_batch(
                            texts=batch_texts,
                            model=self.provider.config.model,
                            embeddings=response.embeddings
                        )

                    # Update results
                    for batch_idx, global_idx in enumerate(batch_indices):
                        all_embeddings[global_idx] = response.embeddings[batch_idx]

                    # Progress callback
                    if progress_callback:
                        completed = sum(1 for e in all_embeddings if e is not None)
                        await progress_callback(completed, total)

                    return response.usage.get("tokens", 0)

            # Create batches
            tasks = []
            for i in range(0, len(texts_to_generate), self.batch_size):
                batch_texts = texts_to_generate[i:i + self.batch_size]
                batch_indices = missing_indices[i:i + self.batch_size]
                tasks.append(process_batch(batch_texts, batch_indices))

            # Process all batches
            token_counts = await asyncio.gather(*tasks)
            total_tokens = sum(token_counts)

        # Build response
        dimension = len(all_embeddings[0]) if all_embeddings[0] else 0

        return EmbeddingResponse(
            embeddings=all_embeddings,
            model=self.provider.config.model,
            dimensions=dimension,
            usage={"tokens": total_tokens},
            cached=cached_flags,
            generation_time_ms=0  # Not tracked for batch
        )
```

### 7.2 Batch Processing Example

```python
"""Example: Batch embedding generation with caching."""
import asyncio
from ia_modules.embeddings.providers.openai import OpenAIEmbeddingProvider
from ia_modules.embeddings.batch import BatchEmbeddingProcessor
from ia_modules.embeddings.cache import EmbeddingCache
from ia_modules.embeddings.models import EmbeddingConfig, EmbeddingProvider


async def main():
    # Setup
    config = EmbeddingConfig(
        provider=EmbeddingProvider.OPENAI,
        model="text-embedding-3-small",
        api_key="sk-...",
        batch_size=100
    )

    provider = OpenAIEmbeddingProvider(config)
    cache = EmbeddingCache(memory_size=5000, ttl_seconds=86400)
    processor = BatchEmbeddingProcessor(
        provider=provider,
        cache=cache,
        batch_size=100,
        max_concurrent=5
    )

    # Large dataset
    texts = [f"Document {i} about topic {i % 10}" for i in range(1000)]

    # Progress callback
    async def on_progress(completed: int, total: int):
        print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")

    # Generate embeddings
    response = await processor.generate_batch(
        texts=texts,
        progress_callback=on_progress,
        normalize=True
    )

    # Stats
    cache_hits = sum(response.cached)
    print(f"\nGenerated {len(response.embeddings)} embeddings")
    print(f"Cache hits: {cache_hits}/{len(texts)} ({cache_hits/len(texts)*100:.1f}%)")
    print(f"Tokens used: {response.usage['tokens']}")

    # Cache stats
    cache_stats = cache.get_stats()
    print(f"Cache hit rate: {cache_stats['hit_rate']*100:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 8. Embedding Monitoring & Cost Tracking

### 8.1 Usage Monitor

**File**: `ia_modules/embeddings/monitoring.py`

```python
"""Monitoring and cost tracking for embeddings."""
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import asyncio
from ..database.interfaces import DatabaseInterface


@dataclass
class UsageRecord:
    """Record of embedding usage."""
    timestamp: datetime
    provider: str
    model: str
    tokens: int
    embeddings: int
    cost: float
    user_id: Optional[str] = None
    cached: bool = False


class EmbeddingMonitor:
    """Monitor and track embedding usage and costs."""

    def __init__(
        self,
        database: Optional[DatabaseInterface] = None,
        daily_token_limit: Optional[int] = None,
        daily_cost_limit: Optional[float] = None
    ):
        """
        Initialize embedding monitor.

        Args:
            database: Database for persistent storage
            daily_token_limit: Max tokens per day
            daily_cost_limit: Max cost per day (USD)
        """
        self._db = database
        self._daily_token_limit = daily_token_limit
        self._daily_cost_limit = daily_cost_limit
        self._records: List[UsageRecord] = []
        self._lock = asyncio.Lock()

    async def record_usage(
        self,
        provider: str,
        model: str,
        tokens: int,
        embeddings: int,
        cost: float,
        user_id: Optional[str] = None,
        cached: bool = False
    ) -> None:
        """
        Record embedding usage.

        Args:
            provider: Provider name
            model: Model identifier
            tokens: Tokens used
            embeddings: Number of embeddings
            cost: Cost in USD
            user_id: Optional user identifier
            cached: Whether embeddings were cached
        """
        async with self._lock:
            record = UsageRecord(
                timestamp=datetime.now(),
                provider=provider,
                model=model,
                tokens=tokens,
                embeddings=embeddings,
                cost=cost,
                user_id=user_id,
                cached=cached
            )
            self._records.append(record)

            # Persist to database
            if self._db:
                await self._db.execute(
                    """
                    INSERT INTO embedding_usage
                    (timestamp, provider, model, tokens, embeddings, cost, user_id, cached)
                    VALUES (:timestamp, :provider, :model, :tokens, :embeddings, :cost, :user_id, :cached)
                    """,
                    {
                        "timestamp": record.timestamp,
                        "provider": record.provider,
                        "model": record.model,
                        "tokens": record.tokens,
                        "embeddings": record.embeddings,
                        "cost": record.cost,
                        "user_id": record.user_id,
                        "cached": record.cached
                    }
                )

    async def check_quotas(self, provider: str, model: str) -> Dict[str, any]:
        """
        Check if quotas are exceeded.

        Args:
            provider: Provider name
            model: Model identifier

        Returns:
            Quota status dictionary
        """
        today = datetime.now().date()
        today_start = datetime.combine(today, datetime.min.time())

        # Calculate today's usage
        daily_tokens = sum(
            r.tokens for r in self._records
            if r.timestamp >= today_start
            and r.provider == provider
            and r.model == model
            and not r.cached
        )

        daily_cost = sum(
            r.cost for r in self._records
            if r.timestamp >= today_start
            and r.provider == provider
            and r.model == model
            and not r.cached
        )

        # Check limits
        token_exceeded = (
            self._daily_token_limit
            and daily_tokens >= self._daily_token_limit
        )
        cost_exceeded = (
            self._daily_cost_limit
            and daily_cost >= self._daily_cost_limit
        )

        return {
            "daily_tokens": daily_tokens,
            "daily_cost": daily_cost,
            "token_limit": self._daily_token_limit,
            "cost_limit": self._daily_cost_limit,
            "token_exceeded": token_exceeded,
            "cost_exceeded": cost_exceeded,
            "quota_ok": not (token_exceeded or cost_exceeded)
        }

    async def get_stats(
        self,
        period_hours: int = 24,
        provider: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Get usage statistics.

        Args:
            period_hours: Time period for stats
            provider: Filter by provider
            user_id: Filter by user

        Returns:
            Statistics dictionary
        """
        cutoff = datetime.now() - timedelta(hours=period_hours)

        # Filter records
        filtered = [
            r for r in self._records
            if r.timestamp >= cutoff
            and (provider is None or r.provider == provider)
            and (user_id is None or r.user_id == user_id)
        ]

        if not filtered:
            return {
                "total_requests": 0,
                "total_tokens": 0,
                "total_embeddings": 0,
                "total_cost": 0.0,
                "cache_hits": 0,
                "cache_misses": 0,
                "cache_hit_rate": 0.0
            }

        cache_hits = sum(1 for r in filtered if r.cached)
        cache_misses = len(filtered) - cache_hits

        return {
            "total_requests": len(filtered),
            "total_tokens": sum(r.tokens for r in filtered if not r.cached),
            "total_embeddings": sum(r.embeddings for r in filtered),
            "total_cost": sum(r.cost for r in filtered if not r.cached),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_hit_rate": cache_hits / len(filtered) if filtered else 0.0
        }
```

### 8.2 Monitoring Schema

**File**: `ia_modules/database/migrations/V005__embedding_monitoring.sql`

```sql
-- Embedding usage tracking
CREATE TABLE IF NOT EXISTS embedding_usage (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    provider VARCHAR(50) NOT NULL,
    model VARCHAR(100) NOT NULL,
    tokens INTEGER NOT NULL,
    embeddings INTEGER NOT NULL,
    cost DECIMAL(10, 6) NOT NULL,
    user_id VARCHAR(100),
    cached BOOLEAN DEFAULT FALSE,
    INDEX idx_timestamp (timestamp),
    INDEX idx_provider_model (provider, model),
    INDEX idx_user_id (user_id)
);

-- Daily usage summary view
CREATE OR REPLACE VIEW embedding_daily_usage AS
SELECT
    DATE(timestamp) as date,
    provider,
    model,
    SUM(tokens) as total_tokens,
    SUM(embeddings) as total_embeddings,
    SUM(cost) as total_cost,
    SUM(CASE WHEN cached THEN 1 ELSE 0 END) as cache_hits,
    COUNT(*) - SUM(CASE WHEN cached THEN 1 ELSE 0 END) as cache_misses
FROM embedding_usage
GROUP BY DATE(timestamp), provider, model;
```

---

## 9. Pipeline Integration

### 9.1 Embedding Pipeline Step

**File**: `ia_modules/pipeline/steps/embed_text.py`

```python
"""Pipeline step for text embedding generation."""
from typing import Dict, Any, List
from ...embeddings.factory import create_embedding_provider
from ...embeddings.batch import BatchEmbeddingProcessor
from ...embeddings.cache import EmbeddingCache
from ...embeddings.monitoring import EmbeddingMonitor
from ...embeddings.models import EmbeddingConfig, EmbeddingProvider


async def embed_text_step(
    context: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Pipeline step to generate embeddings for text.

    Config:
        provider: Embedding provider (openai, huggingface, cohere)
        model: Model identifier
        api_key: API key (for cloud providers)
        text_field: Field containing text to embed
        output_field: Field to store embeddings
        batch_size: Batch size for processing
        enable_cache: Enable caching
        normalize: Normalize embeddings

    Context Input:
        <text_field>: Text or list of texts

    Context Output:
        <output_field>: Embedding or list of embeddings

    Example:
        {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "text_field": "content",
            "output_field": "embedding",
            "batch_size": 100,
            "enable_cache": true
        }
    """
    # Get configuration
    provider_name = config.get("provider", "openai")
    model = config["model"]
    api_key = config.get("api_key")
    text_field = config.get("text_field", "text")
    output_field = config.get("output_field", "embedding")
    batch_size = config.get("batch_size", 100)
    enable_cache = config.get("enable_cache", True)
    normalize = config.get("normalize", True)

    # Get text(s) from context
    text_data = context.get(text_field)
    if not text_data:
        raise ValueError(f"Field '{text_field}' not found in context")

    # Handle single text or list
    is_list = isinstance(text_data, list)
    texts = text_data if is_list else [text_data]

    # Create embedding config
    emb_config = EmbeddingConfig(
        provider=EmbeddingProvider(provider_name),
        model=model,
        api_key=api_key,
        batch_size=batch_size,
        enable_cache=enable_cache
    )

    # Create provider and processor
    provider = create_embedding_provider(emb_config)
    cache = EmbeddingCache() if enable_cache else None
    processor = BatchEmbeddingProcessor(
        provider=provider,
        cache=cache,
        batch_size=batch_size
    )

    # Generate embeddings
    response = await processor.generate_batch(
        texts=texts,
        normalize=normalize
    )

    # Store in context
    if is_list:
        context[output_field] = response.embeddings
    else:
        context[output_field] = response.embeddings[0]

    # Store metadata
    context[f"{output_field}_metadata"] = {
        "model": response.model,
        "dimensions": response.dimensions,
        "tokens": response.usage.get("tokens", 0),
        "cached_count": sum(response.cached),
        "generation_time_ms": response.generation_time_ms
    }

    return context
```

### 9.2 Usage in Pipeline

```json
{
  "name": "Document Embedding Pipeline",
  "steps": [
    {
      "id": "load_documents",
      "type": "data_loader",
      "config": {
        "source": "database",
        "query": "SELECT id, content FROM documents WHERE embedded = false LIMIT 100"
      }
    },
    {
      "id": "generate_embeddings",
      "type": "python",
      "module_path": "ia_modules.pipeline.steps.embed_text",
      "function_name": "embed_text_step",
      "config": {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "api_key": "${OPENAI_API_KEY}",
        "text_field": "content",
        "output_field": "embedding",
        "batch_size": 50,
        "enable_cache": true,
        "normalize": true
      }
    },
    {
      "id": "store_embeddings",
      "type": "vector_store",
      "config": {
        "provider": "pinecone",
        "index_name": "documents",
        "embedding_field": "embedding",
        "metadata_fields": ["id"]
      }
    }
  ]
}
```

---

## 10. Testing Strategy

### 10.1 Unit Tests

**File**: `ia_modules/tests/unit/test_embeddings.py`

```python
"""Unit tests for embedding providers."""
import pytest
from ia_modules.embeddings.providers.openai import OpenAIEmbeddingProvider
from ia_modules.embeddings.providers.huggingface import HuggingFaceEmbeddingProvider
from ia_modules.embeddings.models import EmbeddingConfig, EmbeddingProvider


@pytest.fixture
def openai_config():
    """OpenAI configuration fixture."""
    return EmbeddingConfig(
        provider=EmbeddingProvider.OPENAI,
        model="text-embedding-3-small",
        api_key="sk-test",
        dimensions=1536
    )


@pytest.fixture
def hf_config():
    """HuggingFace configuration fixture."""
    return EmbeddingConfig(
        provider=EmbeddingProvider.HUGGINGFACE,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )


@pytest.mark.asyncio
async def test_openai_dimension():
    """Test OpenAI dimension detection."""
    config = EmbeddingConfig(
        provider=EmbeddingProvider.OPENAI,
        model="text-embedding-3-small",
        api_key="sk-test"
    )
    provider = OpenAIEmbeddingProvider(config)
    dim = await provider.get_model_dimension()
    assert dim == 1536


@pytest.mark.asyncio
async def test_cache_key_generation():
    """Test cache key generation."""
    from ia_modules.embeddings.base import EmbeddingProviderBase

    key1 = EmbeddingProviderBase.hash_text("hello", "model-a")
    key2 = EmbeddingProviderBase.hash_text("hello", "model-b")
    key3 = EmbeddingProviderBase.hash_text("hello", "model-a")

    assert key1 != key2  # Different models
    assert key1 == key3  # Same text and model


@pytest.mark.asyncio
async def test_normalize_embedding():
    """Test embedding normalization."""
    from ia_modules.embeddings.base import EmbeddingProviderBase

    embedding = [3.0, 4.0]  # Magnitude = 5
    normalized = EmbeddingProviderBase.normalize_embedding(embedding)

    assert abs(normalized[0] - 0.6) < 0.001
    assert abs(normalized[1] - 0.8) < 0.001

    # Check unit length
    magnitude = sum(x * x for x in normalized) ** 0.5
    assert abs(magnitude - 1.0) < 0.001
```

---

## Summary

This implementation plan provides:

✅ **Provider abstraction** for OpenAI, HuggingFace, and Cohere
✅ **Multi-tier caching** (Memory → Redis → Database) for cost savings
✅ **Batch processing** with parallel execution and progress tracking
✅ **Cost monitoring** with quotas and usage analytics
✅ **Pipeline integration** as reusable step
✅ **Type safety** with Pydantic models
✅ **Async/await** throughout for performance
✅ **Testing strategy** with unit tests

Next: [HYBRID_SEARCH.md](HYBRID_SEARCH.md) for combining vector and keyword search.
