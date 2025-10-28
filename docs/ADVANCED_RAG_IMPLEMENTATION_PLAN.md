# Advanced RAG Implementation Plan for ia_modules

## Overview

Comprehensive implementation plan for cutting-edge Retrieval-Augmented Generation (RAG) techniques in ia_modules, based on 2024-2025 research including RAPTOR, Self-RAG, HyDE, multi-hop reasoning, and advanced query transformations.

## Table of Contents

1. [RAG Architecture Overview](#1-rag-architecture-overview)
2. [RAPTOR - Hierarchical Retrieval](#2-raptor---hierarchical-retrieval)
3. [Self-RAG - Adaptive Retrieval](#3-self-rag---adaptive-retrieval)
4. [HyDE - Hypothetical Document Embeddings](#4-hyde---hypothetical-document-embeddings)
5. [Query Transformations](#5-query-transformations)
6. [Multi-Hop Reasoning](#6-multi-hop-reasoning)
7. [Context Compression & Reranking](#7-context-compression--reranking)
8. [RAG Evaluation & Metrics](#8-rag-evaluation--metrics)
9. [Multi-Agent RAG](#9-multi-agent-rag)
10. [RAG with Guardrails](#10-rag-with-guardrails)

---

## 1. RAG Architecture Overview

### 1.1 Core RAG Models

**File**: `ia_modules/rag/models.py`

```python
"""Core RAG models and data structures."""
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum
import uuid


class DocumentChunk(BaseModel):
    """Document chunk for retrieval."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Source information
    source_document_id: str
    chunk_index: int

    # Embeddings
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None

    # Hierarchical information (for RAPTOR)
    level: int = 0  # 0 = leaf (original chunks), 1+ = summaries
    parent_id: Optional[str] = None
    children_ids: List[str] = Field(default_factory=list)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)


class RetrievalResult(BaseModel):
    """Single retrieval result."""
    chunk: DocumentChunk
    score: float
    rank: int

    # Retrieval metadata
    retrieval_method: str  # "vector", "keyword", "hybrid", "raptor"
    query: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalContext(BaseModel):
    """Complete retrieval context for generation."""
    query: str
    original_query: Optional[str] = None  # Before rewriting

    # Retrieved chunks
    chunks: List[RetrievalResult] = Field(default_factory=list)

    # Reranked results (if applicable)
    reranked_chunks: Optional[List[RetrievalResult]] = None

    # Compressed context (if applicable)
    compressed_context: Optional[str] = None

    # Metadata
    retrieval_time_ms: float = 0.0
    total_tokens: int = 0
    retrieval_config: Dict[str, Any] = Field(default_factory=dict)


class RetrievalStrategy(str, Enum):
    """Retrieval strategy type."""
    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    RAPTOR = "raptor"
    SELF_RAG = "self_rag"
    MULTI_HOP = "multi_hop"


class RAGConfig(BaseModel):
    """Configuration for RAG system."""
    # Retrieval settings
    strategy: RetrievalStrategy = RetrievalStrategy.VECTOR
    top_k: int = 5
    similarity_threshold: float = 0.7

    # Embedding settings
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536

    # Query transformation
    use_query_rewriting: bool = False
    use_hyde: bool = False
    use_step_back: bool = False

    # Context processing
    use_reranking: bool = False
    reranking_model: Optional[str] = None
    use_compression: bool = False
    max_context_tokens: int = 4000

    # Multi-hop settings
    max_hops: int = 3
    enable_iterative_retrieval: bool = False

    # Self-RAG settings
    use_reflection_tokens: bool = False
    adaptive_retrieval: bool = False

    # RAPTOR settings
    use_hierarchical_retrieval: bool = False
    max_tree_depth: int = 3
    clustering_algorithm: str = "kmeans"


class ReflectionToken(BaseModel):
    """Self-RAG reflection token."""
    token_type: Literal["retrieval", "relevance", "support", "critique"]
    value: str
    confidence: float = 1.0
    position: int  # Position in generation sequence
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

---

## 2. RAPTOR - Hierarchical Retrieval

### 2.1 Recursive Abstractive Processing

**File**: `ia_modules/rag/raptor/tree_builder.py`

```python
"""RAPTOR tree builder for hierarchical retrieval."""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans
from openai import AsyncOpenAI
from ..models import DocumentChunk


class RAPTORTreeBuilder:
    """
    Build hierarchical summary tree for retrieval.

    Process:
    1. Start with leaf nodes (document chunks)
    2. Cluster chunks at current level
    3. Generate summaries for each cluster
    4. Recursively build higher levels
    5. Create tree structure with parent-child relationships
    """

    def __init__(
        self,
        llm_client: AsyncOpenAI,
        embedding_model: str = "text-embedding-3-small",
        max_depth: int = 3,
        cluster_size: int = 10,
        summarization_model: str = "gpt-4o-mini"
    ):
        """
        Initialize RAPTOR tree builder.

        Args:
            llm_client: OpenAI client for embeddings and summarization
            embedding_model: Embedding model name
            max_depth: Maximum tree depth
            cluster_size: Target cluster size
            summarization_model: Model for generating summaries
        """
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self.max_depth = max_depth
        self.cluster_size = cluster_size
        self.summarization_model = summarization_model

        self.tree: Dict[int, List[DocumentChunk]] = {}  # level -> chunks

    async def build_tree(
        self,
        leaf_chunks: List[DocumentChunk]
    ) -> Dict[int, List[DocumentChunk]]:
        """
        Build RAPTOR tree from leaf chunks.

        Args:
            leaf_chunks: Original document chunks (level 0)

        Returns:
            Tree structure with all levels
        """
        # Initialize tree with leaf nodes
        self.tree[0] = leaf_chunks

        # Recursively build higher levels
        current_level = 0

        while current_level < self.max_depth:
            current_chunks = self.tree[current_level]

            # Stop if too few chunks to cluster
            if len(current_chunks) <= self.cluster_size:
                break

            # Build next level
            parent_chunks = await self._build_level(current_chunks, current_level + 1)

            if not parent_chunks:
                break

            self.tree[current_level + 1] = parent_chunks
            current_level += 1

        return self.tree

    async def _build_level(
        self,
        chunks: List[DocumentChunk],
        level: int
    ) -> List[DocumentChunk]:
        """Build a single level of the tree."""
        # Get embeddings for all chunks
        embeddings = await self._get_embeddings([c.content for c in chunks])

        # Cluster chunks
        clusters = self._cluster_chunks(embeddings, chunks)

        # Generate summary for each cluster
        parent_chunks = []

        for cluster_idx, cluster_chunks in clusters.items():
            # Generate summary
            summary = await self._generate_summary(cluster_chunks)

            # Create parent chunk
            parent_chunk = DocumentChunk(
                content=summary,
                source_document_id=cluster_chunks[0].source_document_id,
                chunk_index=-1,  # Summary chunks don't have chunk index
                level=level,
                children_ids=[c.id for c in cluster_chunks],
                metadata={
                    "cluster_id": cluster_idx,
                    "num_children": len(cluster_chunks),
                    "summary_type": "raptor"
                }
            )

            # Update children with parent reference
            for child in cluster_chunks:
                child.parent_id = parent_chunk.id

            parent_chunks.append(parent_chunk)

        return parent_chunks

    async def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for texts."""
        response = await self.llm_client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )

        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)

    def _cluster_chunks(
        self,
        embeddings: np.ndarray,
        chunks: List[DocumentChunk]
    ) -> Dict[int, List[DocumentChunk]]:
        """Cluster chunks using KMeans."""
        # Determine number of clusters
        n_clusters = max(1, len(chunks) // self.cluster_size)

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Group chunks by cluster
        clusters: Dict[int, List[DocumentChunk]] = {}

        for chunk, label in zip(chunks, cluster_labels):
            label = int(label)
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(chunk)

        return clusters

    async def _generate_summary(self, chunks: List[DocumentChunk]) -> str:
        """Generate summary for cluster of chunks."""
        # Combine chunk contents
        combined_text = "\n\n".join([
            f"[Chunk {i+1}]\n{chunk.content}"
            for i, chunk in enumerate(chunks)
        ])

        prompt = f"""Summarize the following text chunks into a concise, coherent summary that captures the main themes and key information:

{combined_text}

Provide a comprehensive summary:"""

        response = await self.llm_client.chat.completions.create(
            model=self.summarization_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )

        summary = response.choices[0].message.content.strip()
        return summary


class RAPTORRetriever:
    """Retriever for RAPTOR tree."""

    def __init__(
        self,
        tree: Dict[int, List[DocumentChunk]],
        llm_client: AsyncOpenAI,
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize RAPTOR retriever.

        Args:
            tree: RAPTOR tree structure
            llm_client: OpenAI client
            embedding_model: Embedding model
        """
        self.tree = tree
        self.llm_client = llm_client
        self.embedding_model = embedding_model

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        search_levels: Optional[List[int]] = None
    ) -> List[DocumentChunk]:
        """
        Retrieve from RAPTOR tree.

        Args:
            query: Search query
            top_k: Number of chunks to retrieve
            search_levels: Specific levels to search (None = all levels)

        Returns:
            Top-k most relevant chunks across all levels
        """
        # Get query embedding
        query_embedding = await self._get_embedding(query)

        # Determine which levels to search
        if search_levels is None:
            search_levels = list(self.tree.keys())

        # Retrieve from each level
        all_results = []

        for level in search_levels:
            if level not in self.tree:
                continue

            level_chunks = self.tree[level]

            # Get embeddings for level chunks (cache these in production)
            chunk_embeddings = await self._get_embeddings([
                c.content for c in level_chunks
            ])

            # Calculate similarities
            similarities = self._cosine_similarity(
                query_embedding,
                chunk_embeddings
            )

            # Pair chunks with scores
            for chunk, score in zip(level_chunks, similarities):
                all_results.append((chunk, score, level))

        # Sort by score and return top-k
        all_results.sort(key=lambda x: x[1], reverse=True)

        top_chunks = []
        for chunk, score, level in all_results[:top_k]:
            # Add level info to metadata
            chunk.metadata["raptor_level"] = level
            chunk.metadata["similarity_score"] = float(score)
            top_chunks.append(chunk)

        return top_chunks

    async def retrieve_with_expansion(
        self,
        query: str,
        top_k: int = 5
    ) -> List[DocumentChunk]:
        """
        Retrieve and expand context using tree structure.

        Strategy:
        1. Retrieve top-k chunks from all levels
        2. For higher-level chunks, include their children
        3. For leaf chunks, include their parents for broader context
        """
        # Initial retrieval
        initial_chunks = await self.retrieve(query, top_k)

        expanded_chunks = []
        seen_ids = set()

        for chunk in initial_chunks:
            # Add the chunk itself
            if chunk.id not in seen_ids:
                expanded_chunks.append(chunk)
                seen_ids.add(chunk.id)

            # If high-level chunk, add children for details
            if chunk.level > 0 and chunk.children_ids:
                for child_id in chunk.children_ids[:3]:  # Limit children
                    child_chunk = self._find_chunk_by_id(child_id)
                    if child_chunk and child_chunk.id not in seen_ids:
                        expanded_chunks.append(child_chunk)
                        seen_ids.add(child_chunk.id)

            # If leaf chunk, add parent for broader context
            if chunk.level == 0 and chunk.parent_id:
                parent_chunk = self._find_chunk_by_id(chunk.parent_id)
                if parent_chunk and parent_chunk.id not in seen_ids:
                    expanded_chunks.append(parent_chunk)
                    seen_ids.add(parent_chunk.id)

        return expanded_chunks

    def _find_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Find chunk by ID in tree."""
        for level_chunks in self.tree.values():
            for chunk in level_chunks:
                if chunk.id == chunk_id:
                    return chunk
        return None

    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for single text."""
        response = await self.llm_client.embeddings.create(
            model=self.embedding_model,
            input=[text]
        )
        return np.array(response.data[0].embedding)

    async def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for multiple texts."""
        response = await self.llm_client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)

    def _cosine_similarity(
        self,
        query_embedding: np.ndarray,
        chunk_embeddings: np.ndarray
    ) -> np.ndarray:
        """Calculate cosine similarity between query and chunks."""
        # Normalize
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        chunks_norm = chunk_embeddings / np.linalg.norm(
            chunk_embeddings,
            axis=1,
            keepdims=True
        )

        # Dot product
        similarities = np.dot(chunks_norm, query_norm)
        return similarities
```

---

## 3. Self-RAG - Adaptive Retrieval

### 3.1 Reflection Tokens Implementation

**File**: `ia_modules/rag/self_rag/adaptive_retrieval.py`

```python
"""Self-RAG adaptive retrieval with reflection tokens."""
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from openai import AsyncOpenAI
from ..models import ReflectionToken, DocumentChunk, RetrievalResult


class RetrievalDecision(str, Enum):
    """Whether to retrieve or not."""
    RETRIEVE = "retrieve"
    NO_RETRIEVE = "no_retrieve"
    CONTINUE = "continue"


class RelevanceScore(str, Enum):
    """Relevance of retrieved passages."""
    RELEVANT = "relevant"
    PARTIALLY_RELEVANT = "partially_relevant"
    IRRELEVANT = "irrelevant"


class SupportScore(str, Enum):
    """Whether generation is supported by retrieval."""
    FULLY_SUPPORTED = "fully_supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    NOT_SUPPORTED = "not_supported"


class CritiqueScore(str, Enum):
    """Quality critique of generation."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"


class SelfRAGGenerator:
    """
    Self-RAG generator with adaptive retrieval.

    Uses special reflection tokens to:
    1. Decide when to retrieve
    2. Assess relevance of retrieved docs
    3. Verify support from retrieved docs
    4. Critique generation quality
    """

    def __init__(
        self,
        llm_client: AsyncOpenAI,
        retriever: Any,  # Retrieval function/class
        model: str = "gpt-4o"
    ):
        """
        Initialize Self-RAG generator.

        Args:
            llm_client: OpenAI client
            retriever: Retrieval system
            model: LLM model to use
        """
        self.llm_client = llm_client
        self.retriever = retriever
        self.model = model

        self.reflection_tokens: List[ReflectionToken] = []

    async def generate(
        self,
        query: str,
        max_retrieval_cycles: int = 3
    ) -> Dict[str, Any]:
        """
        Generate response with adaptive retrieval.

        Args:
            query: User query
            max_retrieval_cycles: Max retrieval attempts

        Returns:
            Generated response with reflection tokens
        """
        response_parts = []
        retrieved_docs: List[DocumentChunk] = []

        current_query = query
        cycle = 0

        while cycle < max_retrieval_cycles:
            # Step 1: Decide whether to retrieve
            should_retrieve = await self._should_retrieve(
                current_query,
                response_parts
            )

            if should_retrieve == RetrievalDecision.NO_RETRIEVE:
                # Generate without retrieval
                response = await self._generate_without_retrieval(current_query)
                response_parts.append(response)
                break

            if should_retrieve == RetrievalDecision.CONTINUE:
                # Continue generation without new retrieval
                break

            # Step 2: Retrieve documents
            docs = await self.retriever.retrieve(current_query, top_k=5)
            retrieved_docs.extend(docs)

            # Step 3: Assess relevance
            relevance_scores = await self._assess_relevance(current_query, docs)

            # Filter to relevant docs
            relevant_docs = [
                doc for doc, score in zip(docs, relevance_scores)
                if score in [RelevanceScore.RELEVANT, RelevanceScore.PARTIALLY_RELEVANT]
            ]

            if not relevant_docs:
                # No relevant docs, generate without them
                response = await self._generate_without_retrieval(current_query)
                response_parts.append(response)
                break

            # Step 4: Generate with retrieved context
            response = await self._generate_with_context(
                current_query,
                relevant_docs
            )

            # Step 5: Verify support from retrieved docs
            support = await self._verify_support(response, relevant_docs)

            # Step 6: Critique generation
            critique = await self._critique_generation(response, current_query)

            response_parts.append(response)

            # Decide if we need more retrieval
            if support == SupportScore.FULLY_SUPPORTED and critique in [CritiqueScore.EXCELLENT, CritiqueScore.GOOD]:
                break

            cycle += 1

        # Combine response parts
        final_response = " ".join(response_parts)

        return {
            "response": final_response,
            "retrieved_docs": retrieved_docs,
            "reflection_tokens": self.reflection_tokens,
            "retrieval_cycles": cycle + 1
        }

    async def _should_retrieve(
        self,
        query: str,
        previous_responses: List[str]
    ) -> RetrievalDecision:
        """Decide whether to retrieve documents."""
        context = "\n".join(previous_responses) if previous_responses else "No previous context"

        prompt = f"""Given the query and context, decide if we need to retrieve additional information.

Query: {query}

Context so far: {context}

Should we retrieve documents? Choose one:
- RETRIEVE: Need external information
- NO_RETRIEVE: Can answer without retrieval
- CONTINUE: Continue with existing context

Decision:"""

        response = await self.llm_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20
        )

        decision_text = response.choices[0].message.content.strip().upper()

        # Parse decision
        if "RETRIEVE" in decision_text and "NO_RETRIEVE" not in decision_text:
            decision = RetrievalDecision.RETRIEVE
        elif "NO_RETRIEVE" in decision_text or "NO RETRIEVE" in decision_text:
            decision = RetrievalDecision.NO_RETRIEVE
        else:
            decision = RetrievalDecision.CONTINUE

        # Log reflection token
        self.reflection_tokens.append(ReflectionToken(
            token_type="retrieval",
            value=decision.value,
            position=len(self.reflection_tokens)
        ))

        return decision

    async def _assess_relevance(
        self,
        query: str,
        docs: List[DocumentChunk]
    ) -> List[RelevanceScore]:
        """Assess relevance of retrieved documents."""
        relevance_scores = []

        for doc in docs:
            prompt = f"""Assess if this document is relevant to the query.

Query: {query}

Document: {doc.content[:500]}...

Relevance (RELEVANT/PARTIALLY_RELEVANT/IRRELEVANT):"""

            response = await self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=20
            )

            relevance_text = response.choices[0].message.content.strip().upper()

            if "IRRELEVANT" in relevance_text:
                score = RelevanceScore.IRRELEVANT
            elif "PARTIALLY" in relevance_text:
                score = RelevanceScore.PARTIALLY_RELEVANT
            else:
                score = RelevanceScore.RELEVANT

            relevance_scores.append(score)

            # Log reflection token
            self.reflection_tokens.append(ReflectionToken(
                token_type="relevance",
                value=score.value,
                position=len(self.reflection_tokens),
                metadata={"doc_id": doc.id}
            ))

        return relevance_scores

    async def _generate_with_context(
        self,
        query: str,
        docs: List[DocumentChunk]
    ) -> str:
        """Generate response using retrieved context."""
        context = "\n\n".join([
            f"[Document {i+1}]\n{doc.content}"
            for i, doc in enumerate(docs)
        ])

        prompt = f"""Answer the query using the provided documents.

Documents:
{context}

Query: {query}

Answer:"""

        response = await self.llm_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )

        return response.choices[0].message.content.strip()

    async def _generate_without_retrieval(self, query: str) -> str:
        """Generate response without retrieval."""
        response = await self.llm_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": query}],
            temperature=0.7,
            max_tokens=500
        )

        return response.choices[0].message.content.strip()

    async def _verify_support(
        self,
        response: str,
        docs: List[DocumentChunk]
    ) -> SupportScore:
        """Verify if response is supported by retrieved documents."""
        context = "\n\n".join([doc.content for doc in docs])

        prompt = f"""Verify if the response is supported by the documents.

Documents:
{context}

Response: {response}

Support level (FULLY_SUPPORTED/PARTIALLY_SUPPORTED/NOT_SUPPORTED):"""

        llm_response = await self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20
        )

        support_text = llm_response.choices[0].message.content.strip().upper()

        if "NOT_SUPPORTED" in support_text or "NOT SUPPORTED" in support_text:
            score = SupportScore.NOT_SUPPORTED
        elif "PARTIALLY" in support_text:
            score = SupportScore.PARTIALLY_SUPPORTED
        else:
            score = SupportScore.FULLY_SUPPORTED

        # Log reflection token
        self.reflection_tokens.append(ReflectionToken(
            token_type="support",
            value=score.value,
            position=len(self.reflection_tokens)
        ))

        return score

    async def _critique_generation(
        self,
        response: str,
        query: str
    ) -> CritiqueScore:
        """Critique the quality of generation."""
        prompt = f"""Critique the quality of this response to the query.

Query: {query}

Response: {response}

Quality (EXCELLENT/GOOD/ACCEPTABLE/POOR):"""

        llm_response = await self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20
        )

        critique_text = llm_response.choices[0].message.content.strip().upper()

        if "EXCELLENT" in critique_text:
            score = CritiqueScore.EXCELLENT
        elif "GOOD" in critique_text:
            score = CritiqueScore.GOOD
        elif "POOR" in critique_text:
            score = CritiqueScore.POOR
        else:
            score = CritiqueScore.ACCEPTABLE

        # Log reflection token
        self.reflection_tokens.append(ReflectionToken(
            token_type="critique",
            value=score.value,
            position=len(self.reflection_tokens)
        ))

        return score
```

Continue to Part 2...
