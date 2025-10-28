"""Query Analyzer Agent - Analyzes user queries for better retrieval."""
from typing import Dict, Any
from ia_modules.agents.base_agent import BaseAgent


class QueryAnalyzerAgent(BaseAgent):
    """Analyzes user queries to extract intent and keywords."""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        self.use_llm = config.get("use_llm", True)

    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze query to extract search terms and intent."""
        query = data.get("query", "")

        if not query:
            return {
                "keywords": [],
                "intent": "unknown",
                "original_query": query,
                "analysis_type": "empty"
            }

        if self.use_llm:
            # Use LLM to analyze query
            analysis = await self._analyze_with_llm(query)
        else:
            # Simple keyword extraction
            analysis = self._analyze_simple(query)

        self.logger.info(f"Query analysis: {analysis['intent']} - {len(analysis['keywords'])} keywords")

        return {
            **data,
            "query_analysis": analysis,
            "keywords": analysis["keywords"],
            "intent": analysis["intent"]
        }

    async def _analyze_with_llm(self, query: str) -> Dict[str, Any]:
        """Use LLM to analyze query."""
        llm_service = self.services.get('llm_provider')
        if not llm_service:
            self.logger.warning("LLM service not available, falling back to simple analysis")
            return self._analyze_simple(query)

        prompt = f"""Analyze this search query and extract:
1. Search intent (question, lookup, comparison, etc.)
2. Key search terms (important words/phrases)

Query: {query}

Respond in this format:
Intent: <intent>
Keywords: <keyword1>, <keyword2>, <keyword3>
"""

        response = await llm_service.generate_completion(
            prompt=prompt,
            temperature=0.3,
            max_tokens=150
        )

        # Parse LLM response
        lines = response.content.strip().split('\n')
        intent = "question"
        keywords = []

        for line in lines:
            if line.startswith("Intent:"):
                intent = line.split(":", 1)[1].strip().lower()
            elif line.startswith("Keywords:"):
                kw_str = line.split(":", 1)[1].strip()
                keywords = [k.strip() for k in kw_str.split(",")]

        return {
            "intent": intent,
            "keywords": keywords,
            "original_query": query,
            "analysis_type": "llm"
        }

    def _analyze_simple(self, query: str) -> Dict[str, Any]:
        """Simple keyword extraction."""
        # Remove common stop words
        stop_words = {"what", "is", "the", "a", "an", "how", "why", "when", "where", "who"}

        words = query.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        # Detect intent from question words
        intent = "question" if any(w in query.lower() for w in ["what", "how", "why"]) else "lookup"

        return {
            "intent": intent,
            "keywords": keywords,
            "original_query": query,
            "analysis_type": "simple"
        }
