"""RAG LLM step that uses context."""
from typing import Dict, Any
from ia_modules.pipeline.core import Step


class RAGLLMStep(Step):
    """LLM step that uses retrieved context to answer questions."""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.query_field = config.get("query_field", "query")
        self.context_field = config.get("context_field", "context")
        self.output_field = config.get("output_field", "answer")
        self.provider_name = config.get("provider_name")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 500)

    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate answer using LLM with retrieved context."""
        # Get LLM service
        llm_service = self.services.get('llm_provider')
        if not llm_service:
            raise RuntimeError("LLMProviderService not registered in services")

        query = data.get(self.query_field, "")
        context = data.get(self.context_field, "")

        if not query:
            self.logger.warning("No query provided")
            data[self.output_field] = "No question provided"
            return data

        # Build RAG prompt
        if context:
            prompt = f"""Answer the following question using only the information from the provided context documents.

Context:
{context}

Question: {query}

Answer:"""
        else:
            prompt = f"""Answer the following question. Note: No specific context documents were retrieved for this question.

Question: {query}

Answer:"""

        self.logger.info(f"Calling LLM with {len(prompt)} char prompt")

        # Call LLM
        response = await llm_service.generate_completion(
            prompt=prompt,
            provider_name=self.provider_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        # Store answer
        data[self.output_field] = response.content

        # Store metadata
        data["llm_metadata"] = {
            "provider": response.provider.value,
            "model": response.model,
            "usage": response.usage,
            "timestamp": response.timestamp.isoformat()
        }

        self.logger.info(f"Generated answer ({len(response.content)} chars)")

        return data
