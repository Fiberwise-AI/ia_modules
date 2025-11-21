"""
Memory Compression - Automatic summarization of old memories.
"""

from dataclasses import dataclass
from typing import List, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CompressionStrategy(Enum):
    """Compression strategies."""
    SUMMARIZE = "summarize"  # LLM-based summarization
    CLUSTER = "cluster"      # Group similar memories
    IMPORTANCE = "importance"  # Keep only important ones
    HYBRID = "hybrid"        # Combination of strategies


@dataclass
class CompressionResult:
    """Result of compression operation."""
    original_count: int
    compressed_count: int
    compression_ratio: float
    strategy: CompressionStrategy
    metadata: dict


class MemoryCompressor:
    """
    Compressor for reducing memory footprint.

    Uses various strategies to compress old memories while
    preserving important information.
    """

    def __init__(
        self,
        strategy: CompressionStrategy = CompressionStrategy.HYBRID,
        llm_provider: Optional[Any] = None,
        target_ratio: float = 0.3  # Target 30% of original size
    ):
        """
        Initialize memory compressor.

        Args:
            strategy: Compression strategy to use
            llm_provider: Optional LLM provider for summarization
            target_ratio: Target compression ratio
        """
        self.strategy = strategy
        self.llm_provider = llm_provider
        self.target_ratio = target_ratio

    async def compress(self, memories: List[Any]) -> List[Any]:
        """
        Compress a list of memories.

        Args:
            memories: List of memories to compress

        Returns:
            List of compressed memories
        """
        if not memories:
            return []

        logger.info(f"Compressing {len(memories)} memories using {self.strategy.value}")

        if self.strategy == CompressionStrategy.SUMMARIZE:
            result = await self._compress_summarize(memories)
        elif self.strategy == CompressionStrategy.CLUSTER:
            result = await self._compress_cluster(memories)
        elif self.strategy == CompressionStrategy.IMPORTANCE:
            result = await self._compress_importance(memories)
        elif self.strategy == CompressionStrategy.HYBRID:
            result = await self._compress_hybrid(memories)
        else:
            result = memories

        logger.info(f"Compressed to {len(result)} memories "
                   f"({len(result)/len(memories):.1%} of original)")

        return result

    async def _compress_summarize(self, memories: List[Any]) -> List[Any]:
        """
        Compress by summarizing groups of memories.

        Note:
            Falls back to truncation if LLM provider is not available.
            This is a legitimate degraded mode.
        """
        # Group into chunks
        chunk_size = 10
        chunks = [
            memories[i:i + chunk_size]
            for i in range(0, len(memories), chunk_size)
        ]

        compressed = []

        for chunk in chunks:
            # Combine content
            combined = "\n".join([m.content for m in chunk])

            # Summarize if LLM available
            if self.llm_provider:
                summary = await self._summarize_text(combined)
            else:
                # Simple truncation fallback (legitimate degraded mode)
                logger.warning(
                    "No LLM provider available for summarization, using truncation. "
                    "Provide llm_provider for better compression quality."
                )
                summary = combined[:500] + "..." if len(combined) > 500 else combined

            # Create compressed memory
            from .memory_manager import Memory, MemoryType
            import time

            compressed_memory = Memory(
                content=summary,
                memory_type=MemoryType.COMPRESSED,
                timestamp=time.time(),
                metadata={
                    "compressed": True,
                    "original_count": len(chunk),
                    "strategy": "summarize"
                },
                importance=max(m.importance for m in chunk)
            )

            compressed.append(compressed_memory)

        return compressed

    async def _compress_cluster(self, memories: List[Any]) -> List[Any]:
        """Compress by clustering similar memories."""
        # Simple clustering by keyword similarity
        clusters = self._cluster_memories(memories)

        compressed = []

        for cluster in clusters:
            # Take most important from each cluster
            cluster.sort(key=lambda m: m.importance, reverse=True)
            representative = cluster[0]

            # Add metadata about cluster
            representative.metadata["cluster_size"] = len(cluster)
            representative.metadata["compressed"] = True
            representative.metadata["strategy"] = "cluster"

            compressed.append(representative)

        return compressed

    async def _compress_importance(self, memories: List[Any]) -> List[Any]:
        """Compress by keeping only important memories."""
        # Sort by importance
        sorted_memories = sorted(
            memories,
            key=lambda m: m.importance,
            reverse=True
        )

        # Keep top percentage
        target_count = max(1, int(len(memories) * self.target_ratio))
        return sorted_memories[:target_count]

    async def _compress_hybrid(self, memories: List[Any]) -> List[Any]:
        """Compress using hybrid strategy."""
        # First, filter by importance (keep top 50%)
        sorted_by_importance = sorted(
            memories,
            key=lambda m: m.importance,
            reverse=True
        )
        important = sorted_by_importance[:len(memories) // 2]

        # Then cluster the rest
        less_important = sorted_by_importance[len(memories) // 2:]

        if less_important:
            clustered = await self._compress_cluster(less_important)
        else:
            clustered = []

        # Combine
        result = important + clustered

        # If still too large, summarize
        target_count = max(1, int(len(memories) * self.target_ratio))
        if len(result) > target_count:
            result = await self._compress_summarize(result)

        return result

    def _cluster_memories(self, memories: List[Any]) -> List[List[Any]]:
        """
        Cluster memories by similarity.

        Simple keyword-based clustering.
        """
        if not memories:
            return []

        # Extract keywords
        def get_keywords(memory):
            words = memory.content.lower().split()
            # Simple: use words longer than 3 chars
            return set(w for w in words if len(w) > 3)

        # Build similarity matrix
        clusters = []
        remaining = list(memories)
        threshold = 0.2  # 20% keyword overlap

        while remaining:
            # Start new cluster with first remaining
            current = remaining.pop(0)
            cluster = [current]
            current_keywords = get_keywords(current)

            # Find similar memories
            i = 0
            while i < len(remaining):
                memory = remaining[i]
                keywords = get_keywords(memory)

                # Calculate similarity
                if current_keywords and keywords:
                    overlap = len(current_keywords & keywords)
                    similarity = overlap / len(current_keywords | keywords)

                    if similarity >= threshold:
                        cluster.append(memory)
                        remaining.pop(i)
                        continue

                i += 1

            clusters.append(cluster)

        return clusters

    async def _summarize_text(self, text: str) -> str:
        """
        Summarize text using LLM or truncation.

        Args:
            text: Text to summarize

        Returns:
            Summarized text

        Note:
            Falls back to truncation if llm_provider not configured or on errors.
            This is a legitimate degraded mode for compression.
        """
        if not self.llm_provider:
            # No LLM available, use truncation
            logger.debug("No LLM provider, using truncation for summarization")
            return text[:300] + "..."

        prompt = f"Summarize the following in 2-3 sentences:\n\n{text}"

        try:
            response = await self.llm_provider.generate(
                prompt=prompt,
                max_tokens=150
            )
            return response.get("content", response.get("text", text[:300]))
        except Exception as e:
            # LLM failed but we can still proceed with truncation (degraded mode)
            logger.warning(f"Summarization failed: {e}, using truncation")
            return text[:300] + "..."
