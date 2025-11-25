"""Cross-modal fusion for combining information from multiple modalities."""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ModalityFusion:
    """Fuse information from multiple modalities."""

    def __init__(self, llm_provider: Optional[Any] = None):
        """Initialize modality fusion."""
        self.llm_provider = llm_provider

    async def fuse(
        self,
        modality_results: Dict[Any, str],
        prompt: Optional[str] = None
    ) -> str:
        """
        Fuse results from multiple modalities.

        Args:
            modality_results: Dict of modality -> result
            prompt: Optional context prompt

        Returns:
            Fused result

        Note:
            Falls back to simple concatenation if LLM provider is not available
            or if LLM fusion fails. This is a legitimate degraded mode.
        """
        # Format individual results
        formatted_results = []
        for modality, result in modality_results.items():
            modality_name = modality.value if hasattr(modality, 'value') else str(modality)
            formatted_results.append(f"**{modality_name.title()}**:\n{result}")

        combined = "\n\n".join(formatted_results)

        # If LLM available, use it to synthesize
        if self.llm_provider:
            fusion_prompt = (
                "You are analyzing information from multiple sources. "
                "Synthesize the following information into a coherent response.\n\n"
            )
            if prompt:
                fusion_prompt += f"Context: {prompt}\n\n"

            fusion_prompt += f"Information:\n{combined}\n\n"
            fusion_prompt += "Synthesized response:"

            try:
                response = await self.llm_provider.generate(prompt=fusion_prompt)
                return response.get("content", response.get("text", combined))
            except Exception as e:
                # LLM fusion failed but we can still return concatenated results
                logger.warning(f"LLM fusion failed: {e}, using simple concatenation")

        # Fallback: simple concatenation (legitimate degraded mode)
        return combined

    def calculate_modality_weights(
        self,
        modality_results: Dict[Any, str]
    ) -> Dict[Any, float]:
        """
        Calculate weights for each modality based on information content.

        Returns:
            Dict of modality -> weight
        """
        weights = {}
        total_length = sum(len(result) for result in modality_results.values())

        for modality, result in modality_results.items():
            # Simple heuristic: longer results get more weight
            if total_length > 0:
                weights[modality] = len(result) / total_length
            else:
                weights[modality] = 1.0 / len(modality_results)

        # Normalize
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        return weights

    async def cross_modal_attention(
        self,
        query_modality: Any,
        query_result: str,
        context_results: Dict[Any, str]
    ) -> str:
        """
        Apply cross-modal attention to enhance query result with context.

        Args:
            query_modality: The primary modality
            query_result: Result from query modality
            context_results: Results from other modalities

        Returns:
            Enhanced result

        Note:
            Returns original query_result if LLM provider is not available
            or if enhancement fails. This is a legitimate degraded mode.
        """
        if not context_results:
            return query_result

        # Build context
        context = "\n".join([
            f"{modality.value if hasattr(modality, 'value') else modality}: {result}"
            for modality, result in context_results.items()
        ])

        # If LLM available, use it to enhance
        if self.llm_provider:
            prompt = (
                f"Enhance the following information using additional context.\n\n"
                f"Primary information:\n{query_result}\n\n"
                f"Additional context:\n{context}\n\n"
                f"Enhanced response:"
            )

            try:
                response = await self.llm_provider.generate(prompt=prompt)
                return response.get("content", response.get("text", query_result))
            except Exception as e:
                # Enhancement failed but we can still return original result
                logger.warning(f"Cross-modal attention failed: {e}, using original result")

        # Return original result (legitimate degraded mode)
        return query_result
