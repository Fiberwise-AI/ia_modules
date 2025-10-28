# Vector Database Integration Implementation Plan

## Overview

This document provides a comprehensive implementation plan for integrating vector databases into the ia_modules pipeline system. Vector databases enable semantic search, similarity matching, and efficient retrieval of high-dimensional embeddings for RAG (Retrieval-Augmented Generation) applications.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Abstract Vector Store Interface](#abstract-vector-store-interface)
3. [Pinecone Integration](#pinecone-integration)
4. [Weaviate Integration](#weaviate-integration)
5. [Qdrant Integration](#qdrant-integration)
6. [ChromaDB Integration](#chromadb-integration)
7. [Connection Pooling & Resource Management](#connection-pooling--resource-management)
8. [Batch Operations](#batch-operations)
9. [Advanced Search Features](#advanced-search-features)
10. [Testing Strategy](#testing-strategy)
11. [Migration Path](#migration-path)

---

## 1. Architecture Overview

### 1.1 Design Principles

- **Provider Agnostic**: Abstract interface allows switching between vector databases
- **Async First**: All operations use async/await for non-blocking I/O
- **Type Safe**: Full type hints with Pydantic models for validation
- **Resource Efficient**: Connection pooling and batch operations
- **Extensible**: Easy to add new vector database providers

### 1.2 Component Architecture

```
ia_modules/
├── vector_stores/
│   ├── __init__.py
│   ├── base.py              # Abstract interface
│   ├── models.py            # Shared data models
│   ├── pooling.py           # Connection pooling
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── pinecone.py      # Pinecone implementation
│   │   ├── weaviate.py      # Weaviate implementation
│   │   ├── qdrant.py        # Qdrant implementation
│   │   └── chroma.py        # ChromaDB implementation
│   └── factory.py           # Provider factory
├── pipeline/
│   └── steps/
│       └── vector_search.py # Pipeline step for vector search
└── tests/
    └── integration/
        └── test_vector_stores.py
```

---

## 2. Abstract Vector Store Interface

### 2.1 Base Models

**File**: `ia_modules/vector_stores/models.py`

```python
"""Shared data models for vector store operations."""
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class DistanceMetric(str, Enum):
    """Supported distance metrics for similarity search."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


class VectorDocument(BaseModel):
    """Document with vector embedding and metadata."""
    id: str = Field(..., description="Unique document identifier")
    vector: List[float] = Field(..., description="Dense vector embedding")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    text: Optional[str] = Field(None, description="Original text content")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "doc_123",
                "vector": [0.1, 0.2, 0.3],
                "metadata": {"source": "wikipedia", "title": "Python"},
                "text": "Python is a programming language"
            }
        }


class SearchFilter(BaseModel):
    """Metadata filter for vector search."""
    field: str = Field(..., description="Metadata field name")
    operator: str = Field(..., description="Filter operator: eq, ne, gt, lt, in, nin")
    value: Any = Field(..., description="Filter value")

    def to_native(self, provider: str) -> Dict[str, Any]:
        """Convert to provider-specific filter format."""
        if provider == "pinecone":
            return {self.field: {f"${self.operator}": self.value}}
        elif provider == "weaviate":
            op_map = {
                "eq": "Equal",
                "ne": "NotEqual",
                "gt": "GreaterThan",
                "lt": "LessThan",
                "in": "ContainsAny"
            }
            return {
                "path": [self.field],
                "operator": op_map.get(self.operator, "Equal"),
                "valueText": str(self.value)
            }
        elif provider == "qdrant":
            op_map = {
                "eq": "must",
                "ne": "must_not",
                "gt": "range",
                "lt": "range"
            }
            if self.operator in ["gt", "lt"]:
                return {
                    "key": self.field,
                    self.operator: self.value
                }
            return {
                "key": self.field,
                "match": {"value": self.value}
            }
        elif provider == "chroma":
            return {self.field: self.value}
        raise ValueError(f"Unsupported provider: {provider}")


class SearchResult(BaseModel):
    """Single search result with score."""
    id: str = Field(..., description="Document ID")
    score: float = Field(..., description="Similarity score")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    text: Optional[str] = None
    vector: Optional[List[float]] = None


class SearchResponse(BaseModel):
    """Response from vector search operation."""
    results: List[SearchResult] = Field(default_factory=list)
    total: int = Field(..., description="Total results found")
    query_time_ms: float = Field(..., description="Query execution time")


class IndexStats(BaseModel):
    """Statistics about a vector index."""
    total_vectors: int = 0
    dimension: int = 0
    index_fullness: float = 0.0  # Percentage 0-1
    namespace_count: Optional[int] = None


class VectorStoreConfig(BaseModel):
    """Configuration for vector store connection."""
    provider: str = Field(..., description="Provider name: pinecone, weaviate, qdrant, chroma")
    api_key: Optional[str] = Field(None, description="API key for authentication")
    host: Optional[str] = Field(None, description="Host URL for self-hosted instances")
    port: Optional[int] = Field(None, description="Port number")
    index_name: str = Field(..., description="Index/collection name")
    namespace: Optional[str] = Field(None, description="Namespace for data isolation")
    dimension: int = Field(..., description="Vector dimension", gt=0)
    metric: DistanceMetric = Field(DistanceMetric.COSINE, description="Distance metric")

    # Connection pooling
    pool_size: int = Field(10, description="Max connections in pool", gt=0)
    pool_timeout: float = Field(30.0, description="Pool acquisition timeout in seconds")

    # Retry configuration
    max_retries: int = Field(3, description="Max retry attempts", ge=0)
    retry_delay: float = Field(1.0, description="Delay between retries in seconds")

    class Config:
        json_schema_extra = {
            "example": {
                "provider": "pinecone",
                "api_key": "pc-xxxxx",
                "index_name": "my-index",
                "dimension": 1536,
                "metric": "cosine"
            }
        }
```

### 2.2 Abstract Base Class

**File**: `ia_modules/vector_stores/base.py`

```python
"""Abstract base class for vector store implementations."""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, AsyncIterator
from contextlib import asynccontextmanager
import time
import asyncio
from .models import (
    VectorDocument,
    SearchFilter,
    SearchResponse,
    IndexStats,
    VectorStoreConfig,
    DistanceMetric
)


class VectorStoreBase(ABC):
    """Abstract base class for vector database implementations."""

    def __init__(self, config: VectorStoreConfig):
        """
        Initialize vector store.

        Args:
            config: Vector store configuration
        """
        self.config = config
        self._connected = False

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to vector database."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to vector database."""
        pass

    @abstractmethod
    async def create_index(
        self,
        dimension: int,
        metric: DistanceMetric = DistanceMetric.COSINE,
        **kwargs
    ) -> None:
        """
        Create a new vector index.

        Args:
            dimension: Vector dimension
            metric: Distance metric for similarity
            **kwargs: Provider-specific parameters
        """
        pass

    @abstractmethod
    async def delete_index(self) -> None:
        """Delete the vector index."""
        pass

    @abstractmethod
    async def index_exists(self) -> bool:
        """Check if index exists."""
        pass

    @abstractmethod
    async def upsert(
        self,
        documents: List[VectorDocument],
        namespace: Optional[str] = None,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Insert or update vectors in the index.

        Args:
            documents: List of documents to upsert
            namespace: Optional namespace for data isolation
            batch_size: Number of vectors per batch

        Returns:
            Operation statistics
        """
        pass

    @abstractmethod
    async def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[List[SearchFilter]] = None,
        namespace: Optional[str] = None,
        include_metadata: bool = True,
        include_vectors: bool = False
    ) -> SearchResponse:
        """
        Search for similar vectors.

        Args:
            query_vector: Query vector
            top_k: Number of results to return
            filters: Metadata filters
            namespace: Optional namespace
            include_metadata: Include metadata in results
            include_vectors: Include vectors in results

        Returns:
            Search results with scores
        """
        pass

    @abstractmethod
    async def fetch(
        self,
        ids: List[str],
        namespace: Optional[str] = None
    ) -> List[VectorDocument]:
        """
        Fetch vectors by IDs.

        Args:
            ids: List of document IDs
            namespace: Optional namespace

        Returns:
            List of documents
        """
        pass

    @abstractmethod
    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filters: Optional[List[SearchFilter]] = None,
        namespace: Optional[str] = None,
        delete_all: bool = False
    ) -> Dict[str, Any]:
        """
        Delete vectors from index.

        Args:
            ids: Specific IDs to delete
            filters: Delete by metadata filters
            namespace: Optional namespace
            delete_all: Delete all vectors in namespace

        Returns:
            Operation statistics
        """
        pass

    @abstractmethod
    async def get_stats(self) -> IndexStats:
        """
        Get index statistics.

        Returns:
            Index statistics
        """
        pass

    @abstractmethod
    async def list_namespaces(self) -> List[str]:
        """
        List all namespaces in index.

        Returns:
            List of namespace names
        """
        pass

    # Helper methods (implemented in base class)

    async def _retry_operation(
        self,
        operation,
        *args,
        **kwargs
    ) -> Any:
        """
        Retry an operation with exponential backoff.

        Args:
            operation: Async function to retry
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Operation result
        """
        last_error = None
        for attempt in range(self.config.max_retries + 1):
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    raise last_error

    @asynccontextmanager
    async def _timed_operation(self, operation_name: str) -> AsyncIterator[Dict[str, Any]]:
        """
        Context manager for timing operations.

        Args:
            operation_name: Name of operation for logging

        Yields:
            Context dictionary
        """
        start_time = time.time()
        context = {"operation": operation_name, "start_time": start_time}
        try:
            yield context
        finally:
            context["duration_ms"] = (time.time() - start_time) * 1000

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
        return False


class VectorStoreError(Exception):
    """Base exception for vector store operations."""
    pass


class ConnectionError(VectorStoreError):
    """Error connecting to vector database."""
    pass


class IndexNotFoundError(VectorStoreError):
    """Index does not exist."""
    pass


class InvalidDimensionError(VectorStoreError):
    """Vector dimension mismatch."""
    pass
```

---

## 3. Pinecone Integration

### 3.1 Pinecone Implementation

**File**: `ia_modules/vector_stores/providers/pinecone.py`

```python
"""Pinecone vector database implementation."""
from typing import List, Optional, Dict, Any
import asyncio
from pinecone import Pinecone, ServerlessSpec
from pinecone.grpc import PineconeGRPC
from ..base import VectorStoreBase, IndexNotFoundError, InvalidDimensionError
from ..models import (
    VectorDocument,
    SearchFilter,
    SearchResponse,
    SearchResult,
    IndexStats,
    VectorStoreConfig,
    DistanceMetric
)


class PineconeVectorStore(VectorStoreBase):
    """Pinecone vector database implementation."""

    def __init__(self, config: VectorStoreConfig):
        """Initialize Pinecone client."""
        super().__init__(config)
        self._client: Optional[Pinecone] = None
        self._index = None

    async def connect(self) -> None:
        """Establish connection to Pinecone."""
        if self._connected:
            return

        try:
            # Initialize Pinecone client
            self._client = PineconeGRPC(api_key=self.config.api_key)

            # Check if index exists
            if await self.index_exists():
                self._index = self._client.Index(self.config.index_name)

            self._connected = True
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Pinecone: {e}")

    async def disconnect(self) -> None:
        """Close Pinecone connection."""
        self._index = None
        self._client = None
        self._connected = False

    async def create_index(
        self,
        dimension: int,
        metric: DistanceMetric = DistanceMetric.COSINE,
        **kwargs
    ) -> None:
        """
        Create Pinecone index.

        Args:
            dimension: Vector dimension
            metric: Distance metric
            **kwargs: cloud, region, pods, pod_type, etc.
        """
        if not self._client:
            await self.connect()

        # Map metric to Pinecone format
        metric_map = {
            DistanceMetric.COSINE: "cosine",
            DistanceMetric.EUCLIDEAN: "euclidean",
            DistanceMetric.DOT_PRODUCT: "dotproduct"
        }

        # Default to serverless
        spec = ServerlessSpec(
            cloud=kwargs.get("cloud", "aws"),
            region=kwargs.get("region", "us-east-1")
        )

        await asyncio.to_thread(
            self._client.create_index,
            name=self.config.index_name,
            dimension=dimension,
            metric=metric_map.get(metric, "cosine"),
            spec=spec
        )

        self._index = self._client.Index(self.config.index_name)

    async def delete_index(self) -> None:
        """Delete Pinecone index."""
        if not self._client:
            await self.connect()

        await asyncio.to_thread(
            self._client.delete_index,
            self.config.index_name
        )
        self._index = None

    async def index_exists(self) -> bool:
        """Check if Pinecone index exists."""
        if not self._client:
            await self.connect()

        indexes = await asyncio.to_thread(self._client.list_indexes)
        return self.config.index_name in [idx.name for idx in indexes]

    async def upsert(
        self,
        documents: List[VectorDocument],
        namespace: Optional[str] = None,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """Upsert vectors to Pinecone."""
        if not self._index:
            raise IndexNotFoundError(f"Index {self.config.index_name} not initialized")

        namespace = namespace or self.config.namespace or ""
        total_upserted = 0

        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            # Convert to Pinecone format
            vectors = []
            for doc in batch:
                vector_data = {
                    "id": doc.id,
                    "values": doc.vector,
                    "metadata": {**doc.metadata}
                }
                if doc.text:
                    vector_data["metadata"]["text"] = doc.text
                vectors.append(vector_data)

            # Upsert batch
            result = await asyncio.to_thread(
                self._index.upsert,
                vectors=vectors,
                namespace=namespace
            )
            total_upserted += result.upserted_count

        return {
            "upserted": total_upserted,
            "namespace": namespace
        }

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[List[SearchFilter]] = None,
        namespace: Optional[str] = None,
        include_metadata: bool = True,
        include_vectors: bool = False
    ) -> SearchResponse:
        """Search Pinecone index."""
        if not self._index:
            raise IndexNotFoundError(f"Index {self.config.index_name} not initialized")

        namespace = namespace or self.config.namespace or ""

        async with self._timed_operation("pinecone_search") as ctx:
            # Build filter
            filter_dict = None
            if filters:
                filter_dict = {}
                for f in filters:
                    filter_dict.update(f.to_native("pinecone"))

            # Execute search
            results = await asyncio.to_thread(
                self._index.query,
                vector=query_vector,
                top_k=top_k,
                namespace=namespace,
                filter=filter_dict,
                include_metadata=include_metadata,
                include_values=include_vectors
            )

            # Convert results
            search_results = []
            for match in results.matches:
                result = SearchResult(
                    id=match.id,
                    score=match.score,
                    metadata=match.metadata if include_metadata else {},
                    vector=match.values if include_vectors else None
                )
                if include_metadata and "text" in match.metadata:
                    result.text = match.metadata["text"]
                search_results.append(result)

            return SearchResponse(
                results=search_results,
                total=len(search_results),
                query_time_ms=ctx["duration_ms"]
            )

    async def fetch(
        self,
        ids: List[str],
        namespace: Optional[str] = None
    ) -> List[VectorDocument]:
        """Fetch vectors by IDs from Pinecone."""
        if not self._index:
            raise IndexNotFoundError(f"Index {self.config.index_name} not initialized")

        namespace = namespace or self.config.namespace or ""

        result = await asyncio.to_thread(
            self._index.fetch,
            ids=ids,
            namespace=namespace
        )

        documents = []
        for id, vector_data in result.vectors.items():
            doc = VectorDocument(
                id=id,
                vector=vector_data.values,
                metadata=vector_data.metadata,
                text=vector_data.metadata.get("text")
            )
            documents.append(doc)

        return documents

    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filters: Optional[List[SearchFilter]] = None,
        namespace: Optional[str] = None,
        delete_all: bool = False
    ) -> Dict[str, Any]:
        """Delete vectors from Pinecone."""
        if not self._index:
            raise IndexNotFoundError(f"Index {self.config.index_name} not initialized")

        namespace = namespace or self.config.namespace or ""

        if delete_all:
            await asyncio.to_thread(
                self._index.delete,
                delete_all=True,
                namespace=namespace
            )
            return {"deleted": "all", "namespace": namespace}

        if ids:
            await asyncio.to_thread(
                self._index.delete,
                ids=ids,
                namespace=namespace
            )
            return {"deleted": len(ids), "namespace": namespace}

        if filters:
            filter_dict = {}
            for f in filters:
                filter_dict.update(f.to_native("pinecone"))

            await asyncio.to_thread(
                self._index.delete,
                filter=filter_dict,
                namespace=namespace
            )
            return {"deleted_by_filter": filter_dict, "namespace": namespace}

        raise ValueError("Must specify ids, filters, or delete_all=True")

    async def get_stats(self) -> IndexStats:
        """Get Pinecone index statistics."""
        if not self._index:
            raise IndexNotFoundError(f"Index {self.config.index_name} not initialized")

        stats = await asyncio.to_thread(self._index.describe_index_stats)

        return IndexStats(
            total_vectors=stats.total_vector_count,
            dimension=stats.dimension,
            index_fullness=stats.index_fullness or 0.0,
            namespace_count=len(stats.namespaces) if stats.namespaces else 0
        )

    async def list_namespaces(self) -> List[str]:
        """List all namespaces in Pinecone index."""
        if not self._index:
            raise IndexNotFoundError(f"Index {self.config.index_name} not initialized")

        stats = await asyncio.to_thread(self._index.describe_index_stats)
        return list(stats.namespaces.keys()) if stats.namespaces else []
```

### 3.2 Pinecone Usage Example

```python
"""Example: Using Pinecone vector store."""
import asyncio
from ia_modules.vector_stores.providers.pinecone import PineconeVectorStore
from ia_modules.vector_stores.models import (
    VectorStoreConfig,
    VectorDocument,
    SearchFilter,
    DistanceMetric
)


async def main():
    # Configure Pinecone
    config = VectorStoreConfig(
        provider="pinecone",
        api_key="pc-xxxxx",
        index_name="my-embeddings",
        dimension=1536,
        metric=DistanceMetric.COSINE
    )

    # Use async context manager
    async with PineconeVectorStore(config) as store:
        # Create index if needed
        if not await store.index_exists():
            await store.create_index(
                dimension=1536,
                metric=DistanceMetric.COSINE,
                cloud="aws",
                region="us-east-1"
            )

        # Upsert documents
        documents = [
            VectorDocument(
                id="doc1",
                vector=[0.1] * 1536,
                metadata={"source": "wiki", "title": "Python"},
                text="Python is a programming language"
            ),
            VectorDocument(
                id="doc2",
                vector=[0.2] * 1536,
                metadata={"source": "wiki", "title": "JavaScript"},
                text="JavaScript is a scripting language"
            )
        ]

        result = await store.upsert(documents, namespace="articles")
        print(f"Upserted {result['upserted']} vectors")

        # Search with filters
        query_vector = [0.15] * 1536
        filters = [
            SearchFilter(field="source", operator="eq", value="wiki")
        ]

        response = await store.search(
            query_vector=query_vector,
            top_k=5,
            filters=filters,
            namespace="articles",
            include_metadata=True
        )

        print(f"Found {response.total} results in {response.query_time_ms:.2f}ms")
        for result in response.results:
            print(f"  {result.id}: {result.score:.4f} - {result.metadata.get('title')}")

        # Get stats
        stats = await store.get_stats()
        print(f"Index has {stats.total_vectors} vectors, {stats.index_fullness*100:.1f}% full")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 4. Weaviate Integration

### 4.1 Weaviate Implementation

**File**: `ia_modules/vector_stores/providers/weaviate.py`

```python
"""Weaviate vector database implementation."""
from typing import List, Optional, Dict, Any
import asyncio
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter, MetadataQuery
from ..base import VectorStoreBase, IndexNotFoundError
from ..models import (
    VectorDocument,
    SearchFilter,
    SearchResponse,
    SearchResult,
    IndexStats,
    VectorStoreConfig,
    DistanceMetric
)


class WeaviateVectorStore(VectorStoreBase):
    """Weaviate vector database implementation."""

    def __init__(self, config: VectorStoreConfig):
        """Initialize Weaviate client."""
        super().__init__(config)
        self._client: Optional[weaviate.WeaviateClient] = None
        self._collection = None

    async def connect(self) -> None:
        """Establish connection to Weaviate."""
        if self._connected:
            return

        try:
            # Connect to Weaviate
            if self.config.api_key:
                # Weaviate Cloud
                self._client = weaviate.connect_to_wcs(
                    cluster_url=self.config.host,
                    auth_credentials=weaviate.auth.AuthApiKey(self.config.api_key)
                )
            else:
                # Self-hosted
                self._client = weaviate.connect_to_local(
                    host=self.config.host or "localhost",
                    port=self.config.port or 8080
                )

            # Get collection if exists
            if await self.index_exists():
                self._collection = self._client.collections.get(self.config.index_name)

            self._connected = True
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Weaviate: {e}")

    async def disconnect(self) -> None:
        """Close Weaviate connection."""
        if self._client:
            await asyncio.to_thread(self._client.close)
        self._client = None
        self._collection = None
        self._connected = False

    async def create_index(
        self,
        dimension: int,
        metric: DistanceMetric = DistanceMetric.COSINE,
        **kwargs
    ) -> None:
        """
        Create Weaviate collection (class).

        Args:
            dimension: Vector dimension
            metric: Distance metric
            **kwargs: vectorizer, properties, etc.
        """
        if not self._client:
            await self.connect()

        # Map metric to Weaviate format
        metric_map = {
            DistanceMetric.COSINE: "cosine",
            DistanceMetric.DOT_PRODUCT: "dot",
            DistanceMetric.EUCLIDEAN: "l2-squared",
            DistanceMetric.MANHATTAN: "manhattan"
        }

        # Define properties
        properties = [
            Property(name="text", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="title", data_type=DataType.TEXT),
        ]

        # Add custom properties from kwargs
        for prop in kwargs.get("properties", []):
            properties.append(Property(
                name=prop["name"],
                data_type=DataType[prop["type"].upper()]
            ))

        # Create collection
        await asyncio.to_thread(
            self._client.collections.create,
            name=self.config.index_name,
            vectorizer_config=Configure.Vectorizer.none(),
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=metric_map.get(metric, "cosine")
            ),
            properties=properties
        )

        self._collection = self._client.collections.get(self.config.index_name)

    async def delete_index(self) -> None:
        """Delete Weaviate collection."""
        if not self._client:
            await self.connect()

        await asyncio.to_thread(
            self._client.collections.delete,
            self.config.index_name
        )
        self._collection = None

    async def index_exists(self) -> bool:
        """Check if Weaviate collection exists."""
        if not self._client:
            await self.connect()

        return await asyncio.to_thread(
            self._client.collections.exists,
            self.config.index_name
        )

    async def upsert(
        self,
        documents: List[VectorDocument],
        namespace: Optional[str] = None,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """Upsert vectors to Weaviate."""
        if not self._collection:
            raise IndexNotFoundError(f"Collection {self.config.index_name} not initialized")

        total_upserted = 0

        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            # Use batch insert
            with self._collection.batch.dynamic() as batch_ctx:
                for doc in batch:
                    # Prepare properties (metadata)
                    properties = {**doc.metadata}
                    if doc.text:
                        properties["text"] = doc.text
                    if namespace:
                        properties["namespace"] = namespace

                    # Add object
                    await asyncio.to_thread(
                        batch_ctx.add_object,
                        properties=properties,
                        uuid=doc.id,
                        vector=doc.vector
                    )
                    total_upserted += 1

        return {
            "upserted": total_upserted,
            "namespace": namespace
        }

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[List[SearchFilter]] = None,
        namespace: Optional[str] = None,
        include_metadata: bool = True,
        include_vectors: bool = False
    ) -> SearchResponse:
        """Search Weaviate collection."""
        if not self._collection:
            raise IndexNotFoundError(f"Collection {self.config.index_name} not initialized")

        async with self._timed_operation("weaviate_search") as ctx:
            # Build filter
            where_filter = None
            if filters:
                # Combine filters with AND
                filter_conditions = []
                for f in filters:
                    filter_conditions.append(
                        Filter.by_property(f.field).equal(f.value)
                    )
                if len(filter_conditions) == 1:
                    where_filter = filter_conditions[0]
                else:
                    where_filter = Filter.all_of(filter_conditions)

            # Add namespace filter
            if namespace:
                ns_filter = Filter.by_property("namespace").equal(namespace)
                where_filter = (
                    Filter.all_of([where_filter, ns_filter])
                    if where_filter
                    else ns_filter
                )

            # Execute search
            response = await asyncio.to_thread(
                self._collection.query.near_vector,
                near_vector=query_vector,
                limit=top_k,
                return_metadata=MetadataQuery(distance=True, certainty=True),
                return_properties=["*"] if include_metadata else [],
                include_vector=include_vectors,
                where=where_filter
            )

            # Convert results
            search_results = []
            for obj in response.objects:
                result = SearchResult(
                    id=str(obj.uuid),
                    score=1 - obj.metadata.distance,  # Convert distance to similarity
                    metadata=dict(obj.properties) if include_metadata else {},
                    vector=obj.vector.get("default") if include_vectors else None
                )
                if include_metadata and "text" in obj.properties:
                    result.text = obj.properties["text"]
                search_results.append(result)

            return SearchResponse(
                results=search_results,
                total=len(search_results),
                query_time_ms=ctx["duration_ms"]
            )

    async def fetch(
        self,
        ids: List[str],
        namespace: Optional[str] = None
    ) -> List[VectorDocument]:
        """Fetch vectors by IDs from Weaviate."""
        if not self._collection:
            raise IndexNotFoundError(f"Collection {self.config.index_name} not initialized")

        documents = []
        for id in ids:
            obj = await asyncio.to_thread(
                self._collection.query.fetch_object_by_id,
                uuid=id,
                include_vector=True
            )

            if obj:
                doc = VectorDocument(
                    id=str(obj.uuid),
                    vector=obj.vector.get("default", []),
                    metadata=dict(obj.properties),
                    text=obj.properties.get("text")
                )
                documents.append(doc)

        return documents

    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filters: Optional[List[SearchFilter]] = None,
        namespace: Optional[str] = None,
        delete_all: bool = False
    ) -> Dict[str, Any]:
        """Delete vectors from Weaviate."""
        if not self._collection:
            raise IndexNotFoundError(f"Collection {self.config.index_name} not initialized")

        if delete_all:
            # Delete all objects
            await asyncio.to_thread(
                self._collection.data.delete_many,
                where=Filter.by_property("namespace").equal(namespace) if namespace else None
            )
            return {"deleted": "all", "namespace": namespace}

        if ids:
            # Delete by IDs
            for id in ids:
                await asyncio.to_thread(
                    self._collection.data.delete_by_id,
                    uuid=id
                )
            return {"deleted": len(ids)}

        if filters:
            # Build filter
            filter_conditions = []
            for f in filters:
                filter_conditions.append(
                    Filter.by_property(f.field).equal(f.value)
                )
            where_filter = (
                filter_conditions[0]
                if len(filter_conditions) == 1
                else Filter.all_of(filter_conditions)
            )

            if namespace:
                where_filter = Filter.all_of([
                    where_filter,
                    Filter.by_property("namespace").equal(namespace)
                ])

            await asyncio.to_thread(
                self._collection.data.delete_many,
                where=where_filter
            )
            return {"deleted_by_filter": str(where_filter)}

        raise ValueError("Must specify ids, filters, or delete_all=True")

    async def get_stats(self) -> IndexStats:
        """Get Weaviate collection statistics."""
        if not self._collection:
            raise IndexNotFoundError(f"Collection {self.config.index_name} not initialized")

        # Get collection config
        config = await asyncio.to_thread(self._collection.config.get)

        # Aggregate count
        result = await asyncio.to_thread(
            self._collection.aggregate.over_all,
            total_count=True
        )

        return IndexStats(
            total_vectors=result.total_count,
            dimension=config.vector_config.get("default", {}).get("vector_index_config", {}).get("dimensions", 0),
            index_fullness=0.0  # Weaviate doesn't have this concept
        )

    async def list_namespaces(self) -> List[str]:
        """List all namespaces in Weaviate collection."""
        if not self._collection:
            raise IndexNotFoundError(f"Collection {self.config.index_name} not initialized")

        # Query unique namespaces
        result = await asyncio.to_thread(
            self._collection.aggregate.over_all,
            group_by="namespace"
        )

        return [group.grouped_by.value for group in result.groups]
```

---

## 5. Qdrant Integration

### 5.1 Qdrant Implementation

**File**: `ia_modules/vector_stores/providers/qdrant.py`

```python
"""Qdrant vector database implementation."""
from typing import List, Optional, Dict, Any
import asyncio
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter as QdrantFilter,
    FieldCondition,
    MatchValue,
    Range
)
from ..base import VectorStoreBase, IndexNotFoundError
from ..models import (
    VectorDocument,
    SearchFilter,
    SearchResponse,
    SearchResult,
    IndexStats,
    VectorStoreConfig,
    DistanceMetric
)


class QdrantVectorStore(VectorStoreBase):
    """Qdrant vector database implementation."""

    def __init__(self, config: VectorStoreConfig):
        """Initialize Qdrant client."""
        super().__init__(config)
        self._client: Optional[AsyncQdrantClient] = None

    async def connect(self) -> None:
        """Establish connection to Qdrant."""
        if self._connected:
            return

        try:
            if self.config.host:
                # Remote Qdrant
                self._client = AsyncQdrantClient(
                    host=self.config.host,
                    port=self.config.port or 6333,
                    api_key=self.config.api_key
                )
            else:
                # In-memory Qdrant
                self._client = AsyncQdrantClient(":memory:")

            self._connected = True
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Qdrant: {e}")

    async def disconnect(self) -> None:
        """Close Qdrant connection."""
        if self._client:
            await self._client.close()
        self._client = None
        self._connected = False

    async def create_index(
        self,
        dimension: int,
        metric: DistanceMetric = DistanceMetric.COSINE,
        **kwargs
    ) -> None:
        """
        Create Qdrant collection.

        Args:
            dimension: Vector dimension
            metric: Distance metric
            **kwargs: on_disk, hnsw_config, etc.
        """
        if not self._client:
            await self.connect()

        # Map metric to Qdrant format
        metric_map = {
            DistanceMetric.COSINE: Distance.COSINE,
            DistanceMetric.EUCLIDEAN: Distance.EUCLID,
            DistanceMetric.DOT_PRODUCT: Distance.DOT,
            DistanceMetric.MANHATTAN: Distance.MANHATTAN
        }

        await self._client.create_collection(
            collection_name=self.config.index_name,
            vectors_config=VectorParams(
                size=dimension,
                distance=metric_map.get(metric, Distance.COSINE)
            ),
            on_disk_payload=kwargs.get("on_disk", False)
        )

    async def delete_index(self) -> None:
        """Delete Qdrant collection."""
        if not self._client:
            await self.connect()

        await self._client.delete_collection(
            collection_name=self.config.index_name
        )

    async def index_exists(self) -> bool:
        """Check if Qdrant collection exists."""
        if not self._client:
            await self.connect()

        try:
            await self._client.get_collection(self.config.index_name)
            return True
        except Exception:
            return False

    async def upsert(
        self,
        documents: List[VectorDocument],
        namespace: Optional[str] = None,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """Upsert vectors to Qdrant."""
        if not self._client:
            raise IndexNotFoundError(f"Collection {self.config.index_name} not initialized")

        total_upserted = 0

        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            # Convert to Qdrant points
            points = []
            for doc in batch:
                payload = {**doc.metadata}
                if doc.text:
                    payload["text"] = doc.text
                if namespace:
                    payload["namespace"] = namespace

                point = PointStruct(
                    id=doc.id,
                    vector=doc.vector,
                    payload=payload
                )
                points.append(point)

            # Upsert batch
            await self._client.upsert(
                collection_name=self.config.index_name,
                points=points
            )
            total_upserted += len(points)

        return {
            "upserted": total_upserted,
            "namespace": namespace
        }

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[List[SearchFilter]] = None,
        namespace: Optional[str] = None,
        include_metadata: bool = True,
        include_vectors: bool = False
    ) -> SearchResponse:
        """Search Qdrant collection."""
        if not self._client:
            raise IndexNotFoundError(f"Collection {self.config.index_name} not initialized")

        async with self._timed_operation("qdrant_search") as ctx:
            # Build filter
            query_filter = None
            if filters or namespace:
                conditions = []

                # Add user filters
                if filters:
                    for f in filters:
                        if f.operator == "eq":
                            conditions.append(
                                FieldCondition(
                                    key=f.field,
                                    match=MatchValue(value=f.value)
                                )
                            )
                        elif f.operator in ["gt", "lt"]:
                            range_params = {}
                            if f.operator == "gt":
                                range_params["gt"] = f.value
                            else:
                                range_params["lt"] = f.value
                            conditions.append(
                                FieldCondition(
                                    key=f.field,
                                    range=Range(**range_params)
                                )
                            )

                # Add namespace filter
                if namespace:
                    conditions.append(
                        FieldCondition(
                            key="namespace",
                            match=MatchValue(value=namespace)
                        )
                    )

                query_filter = QdrantFilter(must=conditions)

            # Execute search
            results = await self._client.search(
                collection_name=self.config.index_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=query_filter,
                with_payload=include_metadata,
                with_vectors=include_vectors
            )

            # Convert results
            search_results = []
            for hit in results:
                result = SearchResult(
                    id=str(hit.id),
                    score=hit.score,
                    metadata=hit.payload if include_metadata else {},
                    vector=hit.vector if include_vectors else None
                )
                if include_metadata and hit.payload and "text" in hit.payload:
                    result.text = hit.payload["text"]
                search_results.append(result)

            return SearchResponse(
                results=search_results,
                total=len(search_results),
                query_time_ms=ctx["duration_ms"]
            )

    async def fetch(
        self,
        ids: List[str],
        namespace: Optional[str] = None
    ) -> List[VectorDocument]:
        """Fetch vectors by IDs from Qdrant."""
        if not self._client:
            raise IndexNotFoundError(f"Collection {self.config.index_name} not initialized")

        results = await self._client.retrieve(
            collection_name=self.config.index_name,
            ids=ids,
            with_payload=True,
            with_vectors=True
        )

        documents = []
        for point in results:
            doc = VectorDocument(
                id=str(point.id),
                vector=point.vector,
                metadata=point.payload,
                text=point.payload.get("text")
            )
            documents.append(doc)

        return documents

    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filters: Optional[List[SearchFilter]] = None,
        namespace: Optional[str] = None,
        delete_all: bool = False
    ) -> Dict[str, Any]:
        """Delete vectors from Qdrant."""
        if not self._client:
            raise IndexNotFoundError(f"Collection {self.config.index_name} not initialized")

        if delete_all:
            # Delete all or by namespace
            if namespace:
                query_filter = QdrantFilter(
                    must=[
                        FieldCondition(
                            key="namespace",
                            match=MatchValue(value=namespace)
                        )
                    ]
                )
                await self._client.delete(
                    collection_name=self.config.index_name,
                    points_selector=query_filter
                )
            else:
                # Recreate collection
                await self.delete_index()
                await self.create_index(
                    dimension=self.config.dimension,
                    metric=self.config.metric
                )
            return {"deleted": "all", "namespace": namespace}

        if ids:
            await self._client.delete(
                collection_name=self.config.index_name,
                points_selector=ids
            )
            return {"deleted": len(ids)}

        if filters:
            # Build filter
            conditions = []
            for f in filters:
                if f.operator == "eq":
                    conditions.append(
                        FieldCondition(
                            key=f.field,
                            match=MatchValue(value=f.value)
                        )
                    )

            if namespace:
                conditions.append(
                    FieldCondition(
                        key="namespace",
                        match=MatchValue(value=namespace)
                    )
                )

            query_filter = QdrantFilter(must=conditions)
            await self._client.delete(
                collection_name=self.config.index_name,
                points_selector=query_filter
            )
            return {"deleted_by_filter": str(query_filter)}

        raise ValueError("Must specify ids, filters, or delete_all=True")

    async def get_stats(self) -> IndexStats:
        """Get Qdrant collection statistics."""
        if not self._client:
            raise IndexNotFoundError(f"Collection {self.config.index_name} not initialized")

        info = await self._client.get_collection(self.config.index_name)

        return IndexStats(
            total_vectors=info.points_count,
            dimension=info.config.params.vectors.size,
            index_fullness=0.0  # Qdrant doesn't have this concept
        )

    async def list_namespaces(self) -> List[str]:
        """List all namespaces in Qdrant collection."""
        if not self._client:
            raise IndexNotFoundError(f"Collection {self.config.index_name} not initialized")

        # Scroll through all points and collect unique namespaces
        namespaces = set()
        offset = None

        while True:
            results, offset = await self._client.scroll(
                collection_name=self.config.index_name,
                limit=100,
                offset=offset,
                with_payload=["namespace"],
                with_vectors=False
            )

            for point in results:
                if point.payload and "namespace" in point.payload:
                    namespaces.add(point.payload["namespace"])

            if offset is None:
                break

        return list(namespaces)
```

---

## 6. ChromaDB Integration

### 6.1 ChromaDB Implementation

**File**: `ia_modules/vector_stores/providers/chroma.py`

```python
"""ChromaDB vector database implementation."""
from typing import List, Optional, Dict, Any
import asyncio
import chromadb
from chromadb.config import Settings
from ..base import VectorStoreBase, IndexNotFoundError
from ..models import (
    VectorDocument,
    SearchFilter,
    SearchResponse,
    SearchResult,
    IndexStats,
    VectorStoreConfig,
    DistanceMetric
)


class ChromaVectorStore(VectorStoreBase):
    """ChromaDB vector database implementation."""

    def __init__(self, config: VectorStoreConfig):
        """Initialize ChromaDB client."""
        super().__init__(config)
        self._client: Optional[chromadb.ClientAPI] = None
        self._collection = None

    async def connect(self) -> None:
        """Establish connection to ChromaDB."""
        if self._connected:
            return

        try:
            if self.config.host:
                # Remote ChromaDB
                self._client = await asyncio.to_thread(
                    chromadb.HttpClient,
                    host=self.config.host,
                    port=self.config.port or 8000,
                    settings=Settings(
                        chroma_api_impl="chromadb.api.fastapi.FastAPI"
                    )
                )
            else:
                # Persistent local ChromaDB
                self._client = await asyncio.to_thread(
                    chromadb.PersistentClient,
                    path="./chroma_db"
                )

            # Get collection if exists
            if await self.index_exists():
                self._collection = await asyncio.to_thread(
                    self._client.get_collection,
                    name=self.config.index_name
                )

            self._connected = True
        except Exception as e:
            raise ConnectionError(f"Failed to connect to ChromaDB: {e}")

    async def disconnect(self) -> None:
        """Close ChromaDB connection."""
        self._collection = None
        self._client = None
        self._connected = False

    async def create_index(
        self,
        dimension: int,
        metric: DistanceMetric = DistanceMetric.COSINE,
        **kwargs
    ) -> None:
        """
        Create ChromaDB collection.

        Args:
            dimension: Vector dimension (informational, ChromaDB infers this)
            metric: Distance metric
            **kwargs: embedding_function, metadata, etc.
        """
        if not self._client:
            await self.connect()

        # Map metric to ChromaDB format
        metric_map = {
            DistanceMetric.COSINE: "cosine",
            DistanceMetric.EUCLIDEAN: "l2",
            DistanceMetric.DOT_PRODUCT: "ip"  # inner product
        }

        metadata = {
            "hnsw:space": metric_map.get(metric, "cosine"),
            "dimension": dimension
        }
        metadata.update(kwargs.get("metadata", {}))

        self._collection = await asyncio.to_thread(
            self._client.create_collection,
            name=self.config.index_name,
            metadata=metadata,
            embedding_function=kwargs.get("embedding_function")
        )

    async def delete_index(self) -> None:
        """Delete ChromaDB collection."""
        if not self._client:
            await self.connect()

        await asyncio.to_thread(
            self._client.delete_collection,
            name=self.config.index_name
        )
        self._collection = None

    async def index_exists(self) -> bool:
        """Check if ChromaDB collection exists."""
        if not self._client:
            await self.connect()

        try:
            await asyncio.to_thread(
                self._client.get_collection,
                name=self.config.index_name
            )
            return True
        except Exception:
            return False

    async def upsert(
        self,
        documents: List[VectorDocument],
        namespace: Optional[str] = None,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """Upsert vectors to ChromaDB."""
        if not self._collection:
            raise IndexNotFoundError(f"Collection {self.config.index_name} not initialized")

        total_upserted = 0

        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            # Prepare batch data
            ids = []
            embeddings = []
            metadatas = []
            documents_text = []

            for doc in batch:
                ids.append(doc.id)
                embeddings.append(doc.vector)

                metadata = {**doc.metadata}
                if namespace:
                    metadata["namespace"] = namespace
                metadatas.append(metadata)

                documents_text.append(doc.text or "")

            # Upsert batch
            await asyncio.to_thread(
                self._collection.upsert,
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents_text
            )
            total_upserted += len(ids)

        return {
            "upserted": total_upserted,
            "namespace": namespace
        }

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[List[SearchFilter]] = None,
        namespace: Optional[str] = None,
        include_metadata: bool = True,
        include_vectors: bool = False
    ) -> SearchResponse:
        """Search ChromaDB collection."""
        if not self._collection:
            raise IndexNotFoundError(f"Collection {self.config.index_name} not initialized")

        async with self._timed_operation("chroma_search") as ctx:
            # Build where filter
            where = {}
            if filters:
                for f in filters:
                    if f.operator == "eq":
                        where[f.field] = f.value
                    elif f.operator == "ne":
                        where[f.field] = {"$ne": f.value}
                    elif f.operator == "gt":
                        where[f.field] = {"$gt": f.value}
                    elif f.operator == "lt":
                        where[f.field] = {"$lt": f.value}
                    elif f.operator == "in":
                        where[f.field] = {"$in": f.value}

            # Add namespace filter
            if namespace:
                where["namespace"] = namespace

            # Execute search
            results = await asyncio.to_thread(
                self._collection.query,
                query_embeddings=[query_vector],
                n_results=top_k,
                where=where if where else None,
                include=["metadatas", "documents", "distances"] + (["embeddings"] if include_vectors else [])
            )

            # Convert results
            search_results = []
            for i in range(len(results["ids"][0])):
                # ChromaDB returns distances, convert to similarity score
                distance = results["distances"][0][i]
                score = 1.0 / (1.0 + distance)  # Simple conversion

                result = SearchResult(
                    id=results["ids"][0][i],
                    score=score,
                    metadata=results["metadatas"][0][i] if include_metadata else {},
                    text=results["documents"][0][i] if results["documents"] else None,
                    vector=results["embeddings"][0][i] if include_vectors and "embeddings" in results else None
                )
                search_results.append(result)

            return SearchResponse(
                results=search_results,
                total=len(search_results),
                query_time_ms=ctx["duration_ms"]
            )

    async def fetch(
        self,
        ids: List[str],
        namespace: Optional[str] = None
    ) -> List[VectorDocument]:
        """Fetch vectors by IDs from ChromaDB."""
        if not self._collection:
            raise IndexNotFoundError(f"Collection {self.config.index_name} not initialized")

        results = await asyncio.to_thread(
            self._collection.get,
            ids=ids,
            include=["metadatas", "documents", "embeddings"]
        )

        documents = []
        for i in range(len(results["ids"])):
            doc = VectorDocument(
                id=results["ids"][i],
                vector=results["embeddings"][i] if results["embeddings"] else [],
                metadata=results["metadatas"][i] if results["metadatas"] else {},
                text=results["documents"][i] if results["documents"] else None
            )
            documents.append(doc)

        return documents

    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filters: Optional[List[SearchFilter]] = None,
        namespace: Optional[str] = None,
        delete_all: bool = False
    ) -> Dict[str, Any]:
        """Delete vectors from ChromaDB."""
        if not self._collection:
            raise IndexNotFoundError(f"Collection {self.config.index_name} not initialized")

        if delete_all:
            # Delete all or by namespace
            where = {"namespace": namespace} if namespace else None

            if where:
                await asyncio.to_thread(
                    self._collection.delete,
                    where=where
                )
            else:
                # Delete and recreate collection
                await self.delete_index()
                await self.create_index(
                    dimension=self.config.dimension,
                    metric=self.config.metric
                )
            return {"deleted": "all", "namespace": namespace}

        if ids:
            await asyncio.to_thread(
                self._collection.delete,
                ids=ids
            )
            return {"deleted": len(ids)}

        if filters:
            # Build where filter
            where = {}
            for f in filters:
                if f.operator == "eq":
                    where[f.field] = f.value

            if namespace:
                where["namespace"] = namespace

            await asyncio.to_thread(
                self._collection.delete,
                where=where
            )
            return {"deleted_by_filter": where}

        raise ValueError("Must specify ids, filters, or delete_all=True")

    async def get_stats(self) -> IndexStats:
        """Get ChromaDB collection statistics."""
        if not self._collection:
            raise IndexNotFoundError(f"Collection {self.config.index_name} not initialized")

        count = await asyncio.to_thread(self._collection.count)
        metadata = self._collection.metadata or {}

        return IndexStats(
            total_vectors=count,
            dimension=metadata.get("dimension", 0),
            index_fullness=0.0  # ChromaDB doesn't have this concept
        )

    async def list_namespaces(self) -> List[str]:
        """List all namespaces in ChromaDB collection."""
        if not self._collection:
            raise IndexNotFoundError(f"Collection {self.config.index_name} not initialized")

        # Get all unique namespaces
        # Note: This can be inefficient for large collections
        results = await asyncio.to_thread(
            self._collection.get,
            include=["metadatas"]
        )

        namespaces = set()
        if results["metadatas"]:
            for metadata in results["metadatas"]:
                if "namespace" in metadata:
                    namespaces.add(metadata["namespace"])

        return list(namespaces)
```

---

## 7. Connection Pooling & Resource Management

### 7.1 Connection Pool Implementation

**File**: `ia_modules/vector_stores/pooling.py`

```python
"""Connection pooling for vector stores."""
from typing import Dict, Optional, AsyncIterator
from contextlib import asynccontextmanager
import asyncio
from .base import VectorStoreBase
from .models import VectorStoreConfig


class VectorStorePool:
    """Connection pool for vector stores."""

    def __init__(
        self,
        store_class: type[VectorStoreBase],
        config: VectorStoreConfig,
        max_size: int = 10,
        min_size: int = 2
    ):
        """
        Initialize connection pool.

        Args:
            store_class: Vector store class to instantiate
            config: Vector store configuration
            max_size: Maximum pool size
            min_size: Minimum pool size
        """
        self.store_class = store_class
        self.config = config
        self.max_size = max_size
        self.min_size = min_size

        self._pool: asyncio.Queue[VectorStoreBase] = asyncio.Queue(maxsize=max_size)
        self._active_connections: int = 0
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize pool with minimum connections."""
        if self._initialized:
            return

        for _ in range(self.min_size):
            store = self.store_class(self.config)
            await store.connect()
            await self._pool.put(store)
            self._active_connections += 1

        self._initialized = True

    async def acquire(self, timeout: float = 30.0) -> VectorStoreBase:
        """
        Acquire connection from pool.

        Args:
            timeout: Acquisition timeout in seconds

        Returns:
            Vector store instance
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Try to get existing connection
            store = await asyncio.wait_for(
                self._pool.get(),
                timeout=timeout
            )
            return store
        except asyncio.TimeoutError:
            # Pool exhausted, create new connection if under max_size
            async with self._lock:
                if self._active_connections < self.max_size:
                    store = self.store_class(self.config)
                    await store.connect()
                    self._active_connections += 1
                    return store

            # Still couldn't get connection
            raise TimeoutError(f"Could not acquire connection within {timeout}s")

    async def release(self, store: VectorStoreBase) -> None:
        """
        Release connection back to pool.

        Args:
            store: Vector store instance to release
        """
        await self._pool.put(store)

    @asynccontextmanager
    async def connection(self, timeout: float = 30.0) -> AsyncIterator[VectorStoreBase]:
        """
        Context manager for acquiring/releasing connections.

        Args:
            timeout: Acquisition timeout

        Yields:
            Vector store instance
        """
        store = await self.acquire(timeout)
        try:
            yield store
        finally:
            await self.release(store)

    async def close(self) -> None:
        """Close all connections in pool."""
        while not self._pool.empty():
            store = await self._pool.get()
            await store.disconnect()
            self._active_connections -= 1

        self._initialized = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        return False


class VectorStorePoolManager:
    """Manager for multiple vector store pools."""

    def __init__(self):
        """Initialize pool manager."""
        self._pools: Dict[str, VectorStorePool] = {}
        self._lock = asyncio.Lock()

    async def get_pool(
        self,
        name: str,
        store_class: type[VectorStoreBase],
        config: VectorStoreConfig,
        max_size: int = 10
    ) -> VectorStorePool:
        """
        Get or create pool by name.

        Args:
            name: Pool identifier
            store_class: Vector store class
            config: Configuration
            max_size: Maximum pool size

        Returns:
            Vector store pool
        """
        async with self._lock:
            if name not in self._pools:
                pool = VectorStorePool(
                    store_class=store_class,
                    config=config,
                    max_size=max_size
                )
                await pool.initialize()
                self._pools[name] = pool

            return self._pools[name]

    async def close_all(self) -> None:
        """Close all pools."""
        for pool in self._pools.values():
            await pool.close()
        self._pools.clear()
```

### 7.2 Pool Usage Example

```python
"""Example: Using connection pool."""
import asyncio
from ia_modules.vector_stores.providers.pinecone import PineconeVectorStore
from ia_modules.vector_stores.pooling import VectorStorePool, VectorStorePoolManager
from ia_modules.vector_stores.models import VectorStoreConfig, DistanceMetric


async def main():
    # Create pool
    config = VectorStoreConfig(
        provider="pinecone",
        api_key="pc-xxxxx",
        index_name="my-index",
        dimension=1536,
        metric=DistanceMetric.COSINE
    )

    async with VectorStorePool(
        store_class=PineconeVectorStore,
        config=config,
        max_size=10,
        min_size=2
    ) as pool:
        # Use pool for concurrent operations
        async def search_task(query_id: int):
            async with pool.connection() as store:
                response = await store.search(
                    query_vector=[0.1] * 1536,
                    top_k=5
                )
                print(f"Query {query_id}: {response.total} results")

        # Run 20 concurrent searches with pool of 10
        tasks = [search_task(i) for i in range(20)]
        await asyncio.gather(*tasks)


async def multi_pool_example():
    """Example: Managing multiple pools."""
    manager = VectorStorePoolManager()

    # Pinecone pool
    pinecone_config = VectorStoreConfig(
        provider="pinecone",
        api_key="pc-xxxxx",
        index_name="pinecone-index",
        dimension=1536
    )

    pinecone_pool = await manager.get_pool(
        name="pinecone",
        store_class=PineconeVectorStore,
        config=pinecone_config
    )

    # Use pools
    async with pinecone_pool.connection() as store:
        await store.search(query_vector=[0.1] * 1536, top_k=10)

    # Cleanup
    await manager.close_all()


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 8. Batch Operations

### 8.1 Batch Processor

**File**: `ia_modules/vector_stores/batch.py`

```python
"""Batch operations for vector stores."""
from typing import List, Dict, Any, Callable, Awaitable
import asyncio
from .base import VectorStoreBase
from .models import VectorDocument


class BatchProcessor:
    """Batch processor for efficient vector operations."""

    def __init__(
        self,
        store: VectorStoreBase,
        batch_size: int = 100,
        max_concurrent: int = 5
    ):
        """
        Initialize batch processor.

        Args:
            store: Vector store instance
            batch_size: Documents per batch
            max_concurrent: Max concurrent batches
        """
        self.store = store
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def upsert_large_dataset(
        self,
        documents: List[VectorDocument],
        namespace: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Upsert large dataset with concurrent batching.

        Args:
            documents: Documents to upsert
            namespace: Optional namespace
            progress_callback: Async callback for progress updates

        Returns:
            Operation statistics
        """
        total = len(documents)
        completed = 0
        errors = []

        async def process_batch(batch: List[VectorDocument]) -> int:
            """Process single batch."""
            nonlocal completed

            async with self._semaphore:
                try:
                    result = await self.store.upsert(
                        documents=batch,
                        namespace=namespace,
                        batch_size=len(batch)
                    )

                    completed += result["upserted"]

                    if progress_callback:
                        await progress_callback(completed, total)

                    return result["upserted"]
                except Exception as e:
                    errors.append(str(e))
                    return 0

        # Create batches
        batches = [
            documents[i:i + self.batch_size]
            for i in range(0, len(documents), self.batch_size)
        ]

        # Process batches concurrently
        tasks = [process_batch(batch) for batch in batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            "total": total,
            "upserted": completed,
            "batches": len(batches),
            "errors": errors
        }

    async def parallel_search(
        self,
        query_vectors: List[List[float]],
        top_k: int = 10,
        **kwargs
    ) -> List[SearchResponse]:
        """
        Execute multiple searches in parallel.

        Args:
            query_vectors: List of query vectors
            top_k: Results per query
            **kwargs: Additional search parameters

        Returns:
            List of search responses
        """
        async def search_one(vector: List[float]):
            """Search for single vector."""
            async with self._semaphore:
                return await self.store.search(
                    query_vector=vector,
                    top_k=top_k,
                    **kwargs
                )

        tasks = [search_one(vec) for vec in query_vectors]
        return await asyncio.gather(*tasks)
```

---

## 9. Advanced Search Features

### 9.1 Hybrid Search (Preview)

**File**: `ia_modules/vector_stores/hybrid.py`

```python
"""Hybrid search combining vector and metadata search."""
from typing import List, Optional, Dict, Any
from .base import VectorStoreBase
from .models import SearchFilter, SearchResponse, SearchResult


class HybridSearchMixin:
    """Mixin for hybrid search capabilities."""

    async def hybrid_search(
        self: VectorStoreBase,
        query_vector: List[float],
        query_text: Optional[str] = None,
        top_k: int = 10,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
        filters: Optional[List[SearchFilter]] = None,
        **kwargs
    ) -> SearchResponse:
        """
        Hybrid search combining vector similarity and text matching.

        Args:
            query_vector: Query vector for semantic search
            query_text: Query text for keyword search
            top_k: Number of results
            vector_weight: Weight for vector similarity (0-1)
            text_weight: Weight for text matching (0-1)
            filters: Metadata filters
            **kwargs: Provider-specific parameters

        Returns:
            Combined search results
        """
        # Vector search
        vector_response = await self.search(
            query_vector=query_vector,
            top_k=top_k * 2,  # Get more candidates
            filters=filters,
            include_metadata=True,
            **kwargs
        )

        # Text search (if supported and query_text provided)
        text_scores: Dict[str, float] = {}
        if query_text:
            # Simple text matching (can be enhanced with BM25, etc.)
            for result in vector_response.results:
                if result.text:
                    text_match = self._text_similarity(query_text, result.text)
                    text_scores[result.id] = text_match

        # Combine scores
        combined_results = []
        for result in vector_response.results:
            vector_score = result.score
            text_score = text_scores.get(result.id, 0.0)

            # Weighted combination
            combined_score = (
                vector_weight * vector_score +
                text_weight * text_score
            )

            combined_results.append(SearchResult(
                id=result.id,
                score=combined_score,
                metadata=result.metadata,
                text=result.text,
                vector=result.vector
            ))

        # Sort by combined score
        combined_results.sort(key=lambda x: x.score, reverse=True)
        combined_results = combined_results[:top_k]

        return SearchResponse(
            results=combined_results,
            total=len(combined_results),
            query_time_ms=vector_response.query_time_ms
        )

    @staticmethod
    def _text_similarity(query: str, text: str) -> float:
        """Simple text similarity (Jaccard)."""
        query_tokens = set(query.lower().split())
        text_tokens = set(text.lower().split())

        if not query_tokens or not text_tokens:
            return 0.0

        intersection = query_tokens & text_tokens
        union = query_tokens | text_tokens

        return len(intersection) / len(union)
```

---

## 10. Testing Strategy

### 10.1 Integration Tests

**File**: `ia_modules/tests/integration/test_vector_stores.py`

```python
"""Integration tests for vector stores."""
import pytest
import asyncio
from ia_modules.vector_stores.providers.pinecone import PineconeVectorStore
from ia_modules.vector_stores.providers.qdrant import QdrantVectorStore
from ia_modules.vector_stores.providers.chroma import ChromaVectorStore
from ia_modules.vector_stores.models import (
    VectorStoreConfig,
    VectorDocument,
    SearchFilter,
    DistanceMetric
)


@pytest.fixture
async def vector_store_config():
    """Fixture for vector store configuration."""
    return VectorStoreConfig(
        provider="chroma",  # Use ChromaDB for testing (in-memory)
        index_name="test_index",
        dimension=384,
        metric=DistanceMetric.COSINE
    )


@pytest.fixture
async def chroma_store(vector_store_config):
    """Fixture for ChromaDB store."""
    store = ChromaVectorStore(vector_store_config)
    await store.connect()

    # Create index if needed
    if not await store.index_exists():
        await store.create_index(
            dimension=384,
            metric=DistanceMetric.COSINE
        )

    yield store

    # Cleanup
    try:
        await store.delete_index()
    except:
        pass
    await store.disconnect()


@pytest.mark.asyncio
async def test_upsert_and_search(chroma_store):
    """Test upserting and searching vectors."""
    # Create test documents
    documents = [
        VectorDocument(
            id="doc1",
            vector=[0.1] * 384,
            metadata={"category": "tech", "title": "Python"},
            text="Python programming language"
        ),
        VectorDocument(
            id="doc2",
            vector=[0.2] * 384,
            metadata={"category": "tech", "title": "JavaScript"},
            text="JavaScript programming language"
        ),
        VectorDocument(
            id="doc3",
            vector=[0.9] * 384,
            metadata={"category": "science", "title": "Physics"},
            text="Physics and quantum mechanics"
        )
    ]

    # Upsert
    result = await chroma_store.upsert(documents)
    assert result["upserted"] == 3

    # Search
    response = await chroma_store.search(
        query_vector=[0.15] * 384,
        top_k=2,
        include_metadata=True
    )

    assert response.total == 2
    assert all(r.metadata for r in response.results)


@pytest.mark.asyncio
async def test_filtered_search(chroma_store):
    """Test search with metadata filters."""
    # Upsert documents
    documents = [
        VectorDocument(
            id=f"doc{i}",
            vector=[0.1 * i] * 384,
            metadata={"category": "tech" if i < 3 else "science"}
        )
        for i in range(5)
    ]
    await chroma_store.upsert(documents)

    # Search with filter
    filters = [SearchFilter(field="category", operator="eq", value="tech")]
    response = await chroma_store.search(
        query_vector=[0.1] * 384,
        top_k=10,
        filters=filters
    )

    assert response.total == 3
    assert all(r.metadata.get("category") == "tech" for r in response.results)


@pytest.mark.asyncio
async def test_fetch_by_ids(chroma_store):
    """Test fetching documents by IDs."""
    # Upsert
    documents = [
        VectorDocument(
            id=f"doc{i}",
            vector=[0.1 * i] * 384,
            metadata={"index": i}
        )
        for i in range(5)
    ]
    await chroma_store.upsert(documents)

    # Fetch
    fetched = await chroma_store.fetch(ids=["doc1", "doc3"])
    assert len(fetched) == 2
    assert {d.id for d in fetched} == {"doc1", "doc3"}


@pytest.mark.asyncio
async def test_delete_operations(chroma_store):
    """Test delete operations."""
    # Upsert
    documents = [
        VectorDocument(id=f"doc{i}", vector=[0.1 * i] * 384)
        for i in range(5)
    ]
    await chroma_store.upsert(documents)

    # Delete by IDs
    result = await chroma_store.delete(ids=["doc1", "doc2"])
    assert result["deleted"] == 2

    # Verify deletion
    stats = await chroma_store.get_stats()
    assert stats.total_vectors == 3
```

---

## 11. Migration Path

### 11.1 Migration Guide

For existing applications, follow this migration path:

1. **Phase 1: Add Vector Store Dependency**
   ```bash
   pip install pinecone-client weaviate-client qdrant-client chromadb
   ```

2. **Phase 2: Configure Vector Store**
   ```python
   from ia_modules.vector_stores import create_vector_store

   store = await create_vector_store(
       provider="pinecone",
       api_key="pc-xxxxx",
       index_name="embeddings",
       dimension=1536
   )
   ```

3. **Phase 3: Integrate with Pipelines**
   ```python
   # In pipeline step
   from ia_modules.pipeline.steps.vector_search import VectorSearchStep

   step = VectorSearchStep(
       vector_store=store,
       embedding_model="text-embedding-3-small"
   )
   ```

4. **Phase 4: Migrate Existing Data**
   ```python
   from ia_modules.vector_stores.migration import migrate_embeddings

   await migrate_embeddings(
       source_db="sqlite:///embeddings.db",
       target_store=store,
       batch_size=1000
   )
   ```

---

## Summary

This implementation plan provides:

✅ **Abstract interface** for provider-agnostic vector operations
✅ **Complete implementations** for Pinecone, Weaviate, Qdrant, and ChromaDB
✅ **Connection pooling** for efficient resource management
✅ **Batch operations** for high-throughput scenarios
✅ **Advanced search** with filtering and hybrid search
✅ **Type safety** with Pydantic models
✅ **Async/await** throughout for non-blocking I/O
✅ **Testing strategy** with integration tests
✅ **Migration path** for existing applications

Next steps: [EMBEDDING_MANAGEMENT.md](EMBEDDING_MANAGEMENT.md) for embedding generation and caching.
