# Hybrid Search Implementation Plan

## Overview

This document provides a comprehensive implementation plan for hybrid search capabilities that combine vector similarity search with traditional keyword/full-text search. Hybrid search delivers superior results by leveraging both semantic understanding and exact keyword matching.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Search Strategy Interface](#search-strategy-interface)
3. [Full-Text Search Integration](#full-text-search-integration)
4. [Reciprocal Rank Fusion (RRF)](#reciprocal-rank-fusion-rrf)
5. [Cross-Encoder Reranking](#cross-encoder-reranking)
6. [Query Understanding & Expansion](#query-understanding--expansion)
7. [Performance Optimization](#performance-optimization)
8. [Pipeline Integration](#pipeline-integration)
9. [Testing Strategy](#testing-strategy)

---

## 1. Architecture Overview

### 1.1 Design Principles

- **Multi-Strategy**: Support vector, keyword, and hybrid search modes
- **Configurable Fusion**: Multiple result combination strategies (RRF, weighted, learned)
- **Smart Reranking**: Cross-encoder models for final result refinement
- **Query Intelligence**: Automatic query expansion and reformulation
- **Performance**: Parallel execution, caching, early termination

### 1.2 Component Architecture

```
ia_modules/
├── search/
│   ├── __init__.py
│   ├── models.py              # Data models
│   ├── base.py                # Abstract interfaces
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── vector_search.py   # Vector similarity search
│   │   ├── keyword_search.py  # Full-text/BM25 search
│   │   └── hybrid_search.py   # Combined search
│   ├── fusion/
│   │   ├── __init__.py
│   │   ├── rrf.py             # Reciprocal Rank Fusion
│   │   ├── weighted.py        # Weighted score fusion
│   │   └── learned.py         # ML-based fusion
│   ├── reranking/
│   │   ├── __init__.py
│   │   ├── cross_encoder.py   # Cross-encoder reranking
│   │   └── llm_reranker.py    # LLM-based reranking
│   ├── query/
│   │   ├── __init__.py
│   │   ├── expander.py        # Query expansion
│   │   └── analyzer.py        # Query analysis
│   └── cache.py               # Search result caching
└── tests/
    └── integration/
        └── test_hybrid_search.py
```

---

## 2. Search Strategy Interface

### 2.1 Data Models

**File**: `ia_modules/search/models.py`

```python
"""Data models for search operations."""
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class SearchMode(str, Enum):
    """Search execution mode."""
    VECTOR = "vector"           # Vector similarity only
    KEYWORD = "keyword"         # Keyword/full-text only
    HYBRID = "hybrid"           # Combined vector + keyword
    AUTO = "auto"               # Automatically choose best mode


class FusionStrategy(str, Enum):
    """Result fusion strategy."""
    RRF = "rrf"                 # Reciprocal Rank Fusion
    WEIGHTED = "weighted"       # Weighted score combination
    LEARNED = "learned"         # ML-based fusion
    MAX_SCORE = "max_score"     # Take maximum score


class SearchFilter(BaseModel):
    """Metadata filter for search."""
    field: str
    operator: str  # eq, ne, gt, lt, in, nin, contains
    value: Any


class SearchQuery(BaseModel):
    """Search query with all parameters."""
    query_text: str = Field(..., description="Search query text")
    mode: SearchMode = Field(SearchMode.HYBRID, description="Search mode")
    top_k: int = Field(10, description="Number of results", gt=0, le=1000)

    # Filters
    filters: List[SearchFilter] = Field(default_factory=list)
    namespace: Optional[str] = None

    # Hybrid search weights
    vector_weight: float = Field(0.7, description="Weight for vector search", ge=0, le=1)
    keyword_weight: float = Field(0.3, description="Weight for keyword search", ge=0, le=1)

    # Fusion
    fusion_strategy: FusionStrategy = Field(FusionStrategy.RRF)

    # Reranking
    enable_reranking: bool = Field(True, description="Enable cross-encoder reranking")
    rerank_top_n: int = Field(50, description="Rerank top N results", gt=0)

    # Query expansion
    expand_query: bool = Field(False, description="Enable query expansion")

    class Config:
        json_schema_extra = {
            "example": {
                "query_text": "machine learning algorithms",
                "mode": "hybrid",
                "top_k": 10,
                "vector_weight": 0.7,
                "keyword_weight": 0.3,
                "enable_reranking": True
            }
        }


class SearchResult(BaseModel):
    """Single search result."""
    id: str = Field(..., description="Document ID")
    score: float = Field(..., description="Relevance score")
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Score breakdown
    vector_score: Optional[float] = None
    keyword_score: Optional[float] = None
    rerank_score: Optional[float] = None

    # Highlights
    highlights: List[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    """Complete search response."""
    results: List[SearchResult] = Field(default_factory=list)
    total: int = Field(..., description="Total results found")
    query: SearchQuery
    search_time_ms: float = Field(..., description="Total search time")

    # Breakdown
    vector_time_ms: Optional[float] = None
    keyword_time_ms: Optional[float] = None
    fusion_time_ms: Optional[float] = None
    rerank_time_ms: Optional[float] = None

    # Diagnostics
    vector_candidates: int = 0
    keyword_candidates: int = 0
    reranked_count: int = 0
```

### 2.2 Abstract Base Classes

**File**: `ia_modules/search/base.py`

```python
"""Abstract base classes for search strategies."""
from abc import ABC, abstractmethod
from typing import List, Optional
import time
from .models import SearchQuery, SearchResult, SearchResponse


class SearchStrategy(ABC):
    """Abstract base class for search strategies."""

    @abstractmethod
    async def search(
        self,
        query: SearchQuery,
        **kwargs
    ) -> List[SearchResult]:
        """
        Execute search.

        Args:
            query: Search query
            **kwargs: Strategy-specific parameters

        Returns:
            List of search results
        """
        pass

    @abstractmethod
    async def explain(
        self,
        query: SearchQuery,
        result_id: str
    ) -> Dict[str, Any]:
        """
        Explain why a result was returned.

        Args:
            query: Search query
            result_id: Result to explain

        Returns:
            Explanation dictionary
        """
        pass


class ResultFusion(ABC):
    """Abstract base class for result fusion strategies."""

    @abstractmethod
    async def fuse(
        self,
        vector_results: List[SearchResult],
        keyword_results: List[SearchResult],
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        **kwargs
    ) -> List[SearchResult]:
        """
        Fuse results from multiple search strategies.

        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword search
            vector_weight: Weight for vector scores
            keyword_weight: Weight for keyword scores
            **kwargs: Fusion-specific parameters

        Returns:
            Fused and sorted results
        """
        pass


class Reranker(ABC):
    """Abstract base class for reranking strategies."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_n: int = 10
    ) -> List[SearchResult]:
        """
        Rerank search results.

        Args:
            query: Search query
            results: Initial results to rerank
            top_n: Number of results to return

        Returns:
            Reranked results
        """
        pass
```

---

## 3. Full-Text Search Integration

### 3.1 Elasticsearch/OpenSearch Integration

**File**: `ia_modules/search/strategies/keyword_search.py`

```python
"""Keyword/full-text search using Elasticsearch."""
from typing import List, Optional, Dict, Any
import asyncio
from elasticsearch import AsyncElasticsearch
from ..base import SearchStrategy
from ..models import SearchQuery, SearchResult


class KeywordSearchStrategy(SearchStrategy):
    """Keyword search using Elasticsearch/OpenSearch."""

    def __init__(
        self,
        hosts: List[str],
        index_name: str,
        api_key: Optional[str] = None,
        content_field: str = "content",
        boost_fields: Optional[Dict[str, float]] = None
    ):
        """
        Initialize keyword search.

        Args:
            hosts: Elasticsearch hosts
            index_name: Index name
            api_key: API key for authentication
            content_field: Field containing main content
            boost_fields: Fields with boost values for scoring
        """
        self._client = AsyncElasticsearch(
            hosts=hosts,
            api_key=api_key
        )
        self.index_name = index_name
        self.content_field = content_field
        self.boost_fields = boost_fields or {
            "title": 2.0,
            "content": 1.0,
            "tags": 1.5
        }

    async def search(
        self,
        query: SearchQuery,
        **kwargs
    ) -> List[SearchResult]:
        """Execute keyword search using BM25."""
        # Build Elasticsearch query
        es_query = {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query.query_text,
                            "fields": [
                                f"{field}^{boost}"
                                for field, boost in self.boost_fields.items()
                            ],
                            "type": "best_fields",
                            "operator": "or",
                            "fuzziness": "AUTO"
                        }
                    }
                ],
                "filter": []
            }
        }

        # Add filters
        for filter_item in query.filters:
            if filter_item.operator == "eq":
                es_query["bool"]["filter"].append({
                    "term": {filter_item.field: filter_item.value}
                })
            elif filter_item.operator == "in":
                es_query["bool"]["filter"].append({
                    "terms": {filter_item.field: filter_item.value}
                })
            elif filter_item.operator == "gt":
                es_query["bool"]["filter"].append({
                    "range": {filter_item.field: {"gt": filter_item.value}}
                })

        # Add namespace filter
        if query.namespace:
            es_query["bool"]["filter"].append({
                "term": {"namespace": query.namespace}
            })

        # Execute search
        response = await self._client.search(
            index=self.index_name,
            query=es_query,
            size=query.top_k,
            highlight={
                "fields": {
                    self.content_field: {
                        "fragment_size": 150,
                        "number_of_fragments": 3
                    }
                },
                "pre_tags": ["<mark>"],
                "post_tags": ["</mark>"]
            }
        )

        # Convert to SearchResult
        results = []
        for hit in response["hits"]["hits"]:
            highlights = []
            if "highlight" in hit:
                highlights = hit["highlight"].get(self.content_field, [])

            result = SearchResult(
                id=hit["_id"],
                score=hit["_score"],
                content=hit["_source"].get(self.content_field, ""),
                metadata=hit["_source"],
                keyword_score=hit["_score"],
                highlights=highlights
            )
            results.append(result)

        return results

    async def explain(
        self,
        query: SearchQuery,
        result_id: str
    ) -> Dict[str, Any]:
        """Explain keyword search score."""
        es_query = {
            "multi_match": {
                "query": query.query_text,
                "fields": list(self.boost_fields.keys())
            }
        }

        explanation = await self._client.explain(
            index=self.index_name,
            id=result_id,
            query=es_query
        )

        return {
            "matched": explanation["matched"],
            "score": explanation["explanation"]["value"],
            "details": explanation["explanation"]["description"]
        }

    async def close(self):
        """Close Elasticsearch connection."""
        await self._client.close()


class SimpleBM25Search(SearchStrategy):
    """Simple in-memory BM25 search (for testing/small datasets)."""

    def __init__(self, documents: List[Dict[str, Any]]):
        """Initialize with document corpus."""
        import rank_bm25
        self.documents = documents
        self.tokenized_docs = [
            doc["content"].lower().split() for doc in documents
        ]
        self.bm25 = rank_bm25.BM25Okapi(self.tokenized_docs)

    async def search(
        self,
        query: SearchQuery,
        **kwargs
    ) -> List[SearchResult]:
        """Execute BM25 search."""
        # Tokenize query
        query_tokens = query.query_text.lower().split()

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Sort by score
        scored_docs = [
            (score, doc)
            for score, doc in zip(scores, self.documents)
        ]
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Convert to SearchResult
        results = []
        for score, doc in scored_docs[:query.top_k]:
            result = SearchResult(
                id=doc["id"],
                score=float(score),
                content=doc["content"],
                metadata=doc.get("metadata", {}),
                keyword_score=float(score)
            )
            results.append(result)

        return results

    async def explain(
        self,
        query: SearchQuery,
        result_id: str
    ) -> Dict[str, Any]:
        """Explain BM25 score."""
        return {
            "algorithm": "BM25",
            "description": "Score based on term frequency and document length normalization"
        }
```

---

## 4. Reciprocal Rank Fusion (RRF)

### 4.1 RRF Implementation

**File**: `ia_modules/search/fusion/rrf.py`

```python
"""Reciprocal Rank Fusion for combining search results."""
from typing import List, Dict
from collections import defaultdict
from ..base import ResultFusion
from ..models import SearchResult


class ReciprocalRankFusion(ResultFusion):
    """
    Reciprocal Rank Fusion (RRF) algorithm.

    RRF Score = sum(1 / (k + rank_i))
    where k is a constant (default 60) and rank_i is the rank in each result list.

    RRF is effective because:
    - No score normalization needed
    - Handles different score scales
    - Gives more weight to top results
    - Robust to outliers
    """

    def __init__(self, k: int = 60):
        """
        Initialize RRF.

        Args:
            k: Constant for rank normalization (default 60)
        """
        self.k = k

    async def fuse(
        self,
        vector_results: List[SearchResult],
        keyword_results: List[SearchResult],
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        **kwargs
    ) -> List[SearchResult]:
        """Fuse results using RRF."""
        # Build rank maps
        vector_ranks = {
            result.id: rank
            for rank, result in enumerate(vector_results, start=1)
        }
        keyword_ranks = {
            result.id: rank
            for rank, result in enumerate(keyword_results, start=1)
        }

        # Calculate RRF scores
        rrf_scores: Dict[str, float] = defaultdict(float)
        result_map: Dict[str, SearchResult] = {}

        # Process vector results
        for result in vector_results:
            rank = vector_ranks[result.id]
            rrf_score = vector_weight * (1.0 / (self.k + rank))
            rrf_scores[result.id] += rrf_score
            result_map[result.id] = result

        # Process keyword results
        for result in keyword_results:
            rank = keyword_ranks[result.id]
            rrf_score = keyword_weight * (1.0 / (self.k + rank))
            rrf_scores[result.id] += rrf_score

            if result.id not in result_map:
                result_map[result.id] = result

        # Create fused results
        fused_results = []
        for doc_id, rrf_score in rrf_scores.items():
            result = result_map[doc_id]
            # Update score and metadata
            fused_result = SearchResult(
                id=result.id,
                score=rrf_score,
                content=result.content,
                metadata=result.metadata,
                vector_score=result.vector_score,
                keyword_score=result.keyword_score,
                highlights=result.highlights
            )
            fused_results.append(fused_result)

        # Sort by RRF score
        fused_results.sort(key=lambda x: x.score, reverse=True)

        return fused_results


class WeightedScoreFusion(ResultFusion):
    """Weighted score combination with normalization."""

    def __init__(self, normalize: bool = True):
        """
        Initialize weighted fusion.

        Args:
            normalize: Normalize scores before combining
        """
        self.normalize = normalize

    async def fuse(
        self,
        vector_results: List[SearchResult],
        keyword_results: List[SearchResult],
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        **kwargs
    ) -> List[SearchResult]:
        """Fuse using weighted score combination."""
        # Normalize scores if requested
        if self.normalize:
            vector_results = self._normalize_scores(vector_results)
            keyword_results = self._normalize_scores(keyword_results)

        # Build score maps
        vector_scores = {r.id: r.score for r in vector_results}
        keyword_scores = {r.id: r.score for r in keyword_results}
        result_map = {r.id: r for r in vector_results + keyword_results}

        # Calculate weighted scores
        all_ids = set(vector_scores.keys()) | set(keyword_scores.keys())
        fused_results = []

        for doc_id in all_ids:
            v_score = vector_scores.get(doc_id, 0.0)
            k_score = keyword_scores.get(doc_id, 0.0)
            combined_score = (
                vector_weight * v_score +
                keyword_weight * k_score
            )

            result = result_map[doc_id]
            fused_result = SearchResult(
                id=result.id,
                score=combined_score,
                content=result.content,
                metadata=result.metadata,
                vector_score=v_score,
                keyword_score=k_score,
                highlights=result.highlights
            )
            fused_results.append(fused_result)

        # Sort by combined score
        fused_results.sort(key=lambda x: x.score, reverse=True)

        return fused_results

    @staticmethod
    def _normalize_scores(results: List[SearchResult]) -> List[SearchResult]:
        """Min-max normalization of scores."""
        if not results:
            return results

        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score

        if score_range == 0:
            return results

        normalized = []
        for result in results:
            normalized_score = (result.score - min_score) / score_range
            normalized_result = SearchResult(
                id=result.id,
                score=normalized_score,
                content=result.content,
                metadata=result.metadata,
                vector_score=result.vector_score,
                keyword_score=result.keyword_score,
                highlights=result.highlights
            )
            normalized.append(normalized_result)

        return normalized
```

---

## 5. Cross-Encoder Reranking

### 5.1 Cross-Encoder Implementation

**File**: `ia_modules/search/reranking/cross_encoder.py`

```python
"""Cross-encoder reranking for search results."""
from typing import List, Tuple
import asyncio
import torch
from sentence_transformers import CrossEncoder
from ..base import Reranker
from ..models import SearchResult


class CrossEncoderReranker(Reranker):
    """
    Rerank search results using cross-encoder model.

    Cross-encoders jointly encode query and document for more accurate
    relevance scoring than bi-encoders (separate embeddings).
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
        device: Optional[str] = None
    ):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace model identifier
            batch_size: Batch size for inference
            device: Device (cuda/cpu), auto-detected if None
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Optional[CrossEncoder] = None

    def _load_model(self) -> CrossEncoder:
        """Load cross-encoder model."""
        if not self._model:
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=512
            )
        return self._model

    async def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_n: int = 10
    ) -> List[SearchResult]:
        """
        Rerank results using cross-encoder.

        Args:
            query: Search query
            results: Initial results
            top_n: Number of results to return

        Returns:
            Reranked results
        """
        if not results:
            return results

        model = self._load_model()

        # Prepare query-document pairs
        pairs = [(query, result.content) for result in results]

        # Run cross-encoder prediction (in thread to not block)
        scores = await asyncio.to_thread(
            model.predict,
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False
        )

        # Update results with rerank scores
        reranked = []
        for result, score in zip(results, scores):
            reranked_result = SearchResult(
                id=result.id,
                score=float(score),
                content=result.content,
                metadata=result.metadata,
                vector_score=result.vector_score,
                keyword_score=result.keyword_score,
                rerank_score=float(score),
                highlights=result.highlights
            )
            reranked.append(reranked_result)

        # Sort by rerank score
        reranked.sort(key=lambda x: x.score, reverse=True)

        return reranked[:top_n]

    def unload_model(self):
        """Unload model from memory."""
        if self._model:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class LLMReranker(Reranker):
    """Rerank using LLM for relevance judgment."""

    def __init__(self, llm_client: Any, model: str = "gpt-4o-mini"):
        """
        Initialize LLM reranker.

        Args:
            llm_client: LLM client (OpenAI, etc.)
            model: Model identifier
        """
        self.llm_client = llm_client
        self.model = model

    async def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_n: int = 10
    ) -> List[SearchResult]:
        """Rerank using LLM relevance scoring."""
        if not results:
            return results

        # Build prompt
        prompt = self._build_rerank_prompt(query, results)

        # Get LLM judgment
        response = await self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a search relevance expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )

        # Parse relevance scores
        scores = self._parse_scores(response.choices[0].message.content, len(results))

        # Update and sort
        reranked = []
        for result, score in zip(results, scores):
            reranked_result = SearchResult(
                id=result.id,
                score=score,
                content=result.content,
                metadata=result.metadata,
                vector_score=result.vector_score,
                keyword_score=result.keyword_score,
                rerank_score=score,
                highlights=result.highlights
            )
            reranked.append(reranked_result)

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_n]

    @staticmethod
    def _build_rerank_prompt(query: str, results: List[SearchResult]) -> str:
        """Build prompt for LLM reranking."""
        docs_text = "\n\n".join([
            f"Document {i+1}:\n{result.content[:500]}"
            for i, result in enumerate(results)
        ])

        return f"""Query: {query}

{docs_text}

Rate the relevance of each document to the query on a scale of 0-10.
Return only the scores as a JSON array: [score1, score2, ...]"""

    @staticmethod
    def _parse_scores(response: str, expected_count: int) -> List[float]:
        """Parse LLM response to extract scores."""
        import json
        try:
            scores = json.loads(response)
            if len(scores) != expected_count:
                # Fallback to uniform scores
                return [5.0] * expected_count
            return [float(s) / 10.0 for s in scores]  # Normalize to 0-1
        except:
            return [5.0] * expected_count
```

---

## 6. Query Understanding & Expansion

### 6.1 Query Expansion

**File**: `ia_modules/search/query/expander.py`

```python
"""Query expansion for improved recall."""
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI


class QueryExpander:
    """Expand search queries with synonyms and related terms."""

    def __init__(
        self,
        llm_client: Optional[AsyncOpenAI] = None,
        model: str = "gpt-4o-mini"
    ):
        """
        Initialize query expander.

        Args:
            llm_client: LLM client for semantic expansion
            model: LLM model identifier
        """
        self.llm_client = llm_client
        self.model = model

    async def expand(
        self,
        query: str,
        max_expansions: int = 5,
        strategy: str = "semantic"
    ) -> List[str]:
        """
        Expand query with related terms.

        Args:
            query: Original query
            max_expansions: Max number of expansion terms
            strategy: Expansion strategy (semantic, synonym, both)

        Returns:
            List of expanded query terms
        """
        if strategy == "semantic" and self.llm_client:
            return await self._semantic_expansion(query, max_expansions)
        elif strategy == "synonym":
            return self._synonym_expansion(query, max_expansions)
        else:
            # Combine both
            semantic = await self._semantic_expansion(query, max_expansions // 2)
            synonyms = self._synonym_expansion(query, max_expansions // 2)
            return list(set(semantic + synonyms))[:max_expansions]

    async def _semantic_expansion(
        self,
        query: str,
        max_expansions: int
    ) -> List[str]:
        """Semantic expansion using LLM."""
        if not self.llm_client:
            return []

        prompt = f"""Generate {max_expansions} semantically related search terms for: "{query}"

Return only the terms, one per line, without numbering or explanation."""

        response = await self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=100
        )

        terms = response.choices[0].message.content.strip().split("\n")
        return [t.strip() for t in terms if t.strip()][:max_expansions]

    @staticmethod
    def _synonym_expansion(query: str, max_expansions: int) -> List[str]:
        """Simple synonym expansion using WordNet."""
        try:
            from nltk.corpus import wordnet
            import nltk
            nltk.download("wordnet", quiet=True)
        except ImportError:
            return []

        synonyms = set()
        for word in query.split():
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    if lemma.name() != word:
                        synonyms.add(lemma.name().replace("_", " "))

        return list(synonyms)[:max_expansions]


class QueryAnalyzer:
    """Analyze query intent and characteristics."""

    async def analyze(self, query: str) -> Dict[str, Any]:
        """
        Analyze query.

        Args:
            query: Search query

        Returns:
            Analysis results
        """
        return {
            "length": len(query.split()),
            "is_question": self._is_question(query),
            "has_boolean": self._has_boolean_operators(query),
            "suggested_mode": self._suggest_search_mode(query),
            "language": self._detect_language(query)
        }

    @staticmethod
    def _is_question(query: str) -> bool:
        """Check if query is a question."""
        question_words = ["what", "when", "where", "who", "why", "how", "is", "are", "can", "does"]
        return any(query.lower().startswith(word) for word in question_words) or "?" in query

    @staticmethod
    def _has_boolean_operators(query: str) -> bool:
        """Check for boolean operators."""
        operators = ["AND", "OR", "NOT", "+", "-"]
        return any(op in query.upper() for op in operators)

    @staticmethod
    def _suggest_search_mode(query: str) -> str:
        """Suggest best search mode for query."""
        # Heuristics
        if len(query.split()) > 10:
            return "vector"  # Long queries: semantic search
        elif '"' in query or QueryAnalyzer._has_boolean_operators(query):
            return "keyword"  # Exact match needed
        else:
            return "hybrid"  # Default to hybrid

    @staticmethod
    def _detect_language(query: str) -> str:
        """Simple language detection."""
        # Placeholder - use langdetect or similar in production
        return "en"
```

---

## 7. Performance Optimization

### 7.1 Search Cache

**File**: `ia_modules/search/cache.py`

```python
"""Caching layer for search results."""
from typing import Optional, Dict, Any
import hashlib
import json
from datetime import datetime, timedelta
import redis.asyncio as redis
from .models import SearchQuery, SearchResponse


class SearchCache:
    """Cache for search results."""

    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        ttl_seconds: int = 3600
    ):
        """
        Initialize search cache.

        Args:
            redis_client: Redis client
            ttl_seconds: TTL for cached results
        """
        self._redis = redis_client
        self._ttl = ttl_seconds
        self._memory_cache: Dict[str, tuple] = {}

    @staticmethod
    def _cache_key(query: SearchQuery) -> str:
        """Generate cache key from query."""
        # Create deterministic key
        key_data = {
            "query_text": query.query_text,
            "mode": query.mode,
            "top_k": query.top_k,
            "filters": sorted([f.dict() for f in query.filters], key=str),
            "namespace": query.namespace,
            "vector_weight": query.vector_weight,
            "keyword_weight": query.keyword_weight
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    async def get(self, query: SearchQuery) -> Optional[SearchResponse]:
        """Get cached search response."""
        cache_key = self._cache_key(query)

        # Try memory cache first
        if cache_key in self._memory_cache:
            response, cached_at = self._memory_cache[cache_key]
            if datetime.now() - cached_at < timedelta(seconds=self._ttl):
                return response

        # Try Redis
        if self._redis:
            data = await self._redis.get(f"search:{cache_key}")
            if data:
                response = SearchResponse.parse_raw(data)
                # Promote to memory
                self._memory_cache[cache_key] = (response, datetime.now())
                return response

        return None

    async def put(self, query: SearchQuery, response: SearchResponse) -> None:
        """Cache search response."""
        cache_key = self._cache_key(query)

        # Store in memory
        self._memory_cache[cache_key] = (response, datetime.now())

        # Store in Redis
        if self._redis:
            await self._redis.setex(
                f"search:{cache_key}",
                self._ttl,
                response.json()
            )

    async def clear(self) -> None:
        """Clear cache."""
        self._memory_cache.clear()
        if self._redis:
            # Clear search keys
            cursor = 0
            while True:
                cursor, keys = await self._redis.scan(
                    cursor=cursor,
                    match="search:*",
                    count=100
                )
                if keys:
                    await self._redis.delete(*keys)
                if cursor == 0:
                    break
```

---

## 8. Pipeline Integration

### 8.1 Hybrid Search Pipeline Step

**File**: `ia_modules/pipeline/steps/hybrid_search.py`

```python
"""Pipeline step for hybrid search."""
from typing import Dict, Any
from ...search.strategies.hybrid_search import HybridSearchEngine
from ...search.models import SearchQuery, SearchMode


async def hybrid_search_step(
    context: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute hybrid search.

    Config:
        query_field: Field containing search query
        top_k: Number of results
        mode: Search mode (vector, keyword, hybrid, auto)
        vector_weight: Weight for vector search (0-1)
        keyword_weight: Weight for keyword search (0-1)
        enable_reranking: Enable cross-encoder reranking
        expand_query: Enable query expansion

    Example:
        {
            "query_field": "user_query",
            "top_k": 10,
            "mode": "hybrid",
            "vector_weight": 0.7,
            "keyword_weight": 0.3,
            "enable_reranking": true
        }
    """
    # Get configuration
    query_text = context.get(config.get("query_field", "query"))
    if not query_text:
        raise ValueError("Query not found in context")

    # Build search query
    search_query = SearchQuery(
        query_text=query_text,
        mode=SearchMode(config.get("mode", "hybrid")),
        top_k=config.get("top_k", 10),
        vector_weight=config.get("vector_weight", 0.7),
        keyword_weight=config.get("keyword_weight", 0.3),
        enable_reranking=config.get("enable_reranking", True),
        expand_query=config.get("expand_query", False)
    )

    # Execute search (engine would be initialized elsewhere)
    # search_engine = context.get("search_engine")
    # response = await search_engine.search(search_query)

    # Store results in context
    # context["search_results"] = response.results
    # context["search_metadata"] = {
    #     "total": response.total,
    #     "search_time_ms": response.search_time_ms
    # }

    return context
```

---

## 9. Testing Strategy

### 9.1 Integration Tests

**File**: `ia_modules/tests/integration/test_hybrid_search.py`

```python
"""Integration tests for hybrid search."""
import pytest
from ia_modules.search.fusion.rrf import ReciprocalRankFusion
from ia_modules.search.models import SearchResult


@pytest.mark.asyncio
async def test_rrf_fusion():
    """Test Reciprocal Rank Fusion."""
    # Vector results
    vector_results = [
        SearchResult(id="doc1", score=0.9, content="content1"),
        SearchResult(id="doc2", score=0.8, content="content2"),
        SearchResult(id="doc3", score=0.7, content="content3"),
    ]

    # Keyword results (different order)
    keyword_results = [
        SearchResult(id="doc2", score=15.0, content="content2"),
        SearchResult(id="doc1", score=12.0, content="content1"),
        SearchResult(id="doc4", score=10.0, content="content4"),
    ]

    # Fuse
    rrf = ReciprocalRankFusion(k=60)
    fused = await rrf.fuse(
        vector_results=vector_results,
        keyword_results=keyword_results,
        vector_weight=0.7,
        keyword_weight=0.3
    )

    # Check results
    assert len(fused) == 4  # doc1, doc2, doc3, doc4
    assert fused[0].id in ["doc1", "doc2"]  # Top results
    assert all(r.score > 0 for r in fused)


@pytest.mark.asyncio
async def test_cross_encoder_reranking():
    """Test cross-encoder reranking."""
    from ia_modules.search.reranking.cross_encoder import CrossEncoderReranker

    results = [
        SearchResult(id="doc1", score=0.8, content="Machine learning algorithms"),
        SearchResult(id="doc2", score=0.7, content="Deep learning neural networks"),
        SearchResult(id="doc3", score=0.6, content="Unrelated content"),
    ]

    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    reranked = await reranker.rerank(
        query="machine learning",
        results=results,
        top_n=2
    )

    assert len(reranked) == 2
    assert reranked[0].rerank_score is not None
```

---

## Summary

This implementation plan provides:

✅ **Multi-strategy search** (vector, keyword, hybrid)
✅ **Result fusion** with RRF and weighted combination
✅ **Cross-encoder reranking** for improved relevance
✅ **Query expansion** for better recall
✅ **Performance optimization** with caching
✅ **Elasticsearch integration** for full-text search
✅ **Pipeline integration** as reusable step
✅ **Type safety** with Pydantic models

Next: [KNOWLEDGE_GRAPH_INTEGRATION.md](KNOWLEDGE_GRAPH_INTEGRATION.md) for graph-based knowledge representation.
