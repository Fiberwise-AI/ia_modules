"""Answer Generator Agent - Generates final answer using LLM."""
from typing import Dict, Any
from ia_modules.agents.base_agent import BaseAgent


class AnswerGeneratorAgent(BaseAgent):
    """Generates answer using LLM with retrieved context."""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        self.max_context_length = config.get("max_context_length", 3000)
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 500)

    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate answer using LLM."""
        llm_service = self.services.get('llm_provider')
        if not llm_service:
            raise RuntimeError("LLM service not available")

        query = data.get("query", "")
        retrieved_docs = data.get("retrieved_docs", [])

        # Build context
        context = self._build_context(retrieved_docs)

        # Build prompt
        if context:
            prompt = f"""Answer the following question using ONLY the information from the provided context documents.

Context:
{context}

Question: {query}

Answer:"""
        else:
            prompt = f"""Answer the following question. Note: No specific context documents were retrieved.

Question: {query}

Answer:"""

        self.logger.info(f"Generating answer with {len(context)} char context")

        # Generate answer
        response = await llm_service.generate_completion(
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return {
            **data,
            "answer": response.content,
            "context_used": len(context),
            "llm_metadata": {
                "provider": response.provider.value,
                "model": response.model,
                "usage": response.usage,
                "timestamp": response.timestamp.isoformat()
            }
        }

    def _build_context(self, docs: list) -> str:
        """Build context from retrieved documents."""
        context_parts = []
        total_length = 0

        for i, doc in enumerate(docs, 1):
            section = f"[Document {i}: {doc.get('filename', 'unknown')}]\n{doc['content']}\n"

            if total_length + len(section) > self.max_context_length:
                remaining = self.max_context_length - total_length
                if remaining > 100:
                    section = section[:remaining] + "...\n"
                    context_parts.append(section)
                break

            context_parts.append(section)
            total_length += len(section)

        return "\n".join(context_parts)
