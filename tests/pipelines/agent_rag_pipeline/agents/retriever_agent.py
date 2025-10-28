"""Retriever Agent - Retrieves relevant documents."""
from typing import Dict, Any, List
from pathlib import Path
from ia_modules.agents.base_agent import BaseAgent


class RetrieverAgent(BaseAgent):
    """Retrieves documents based on query analysis."""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        self.docs_dir = config.get("docs_dir", "../rag_with_guardrails_pipeline/sample_docs")
        self.top_k = config.get("top_k", 3)

    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant documents."""
        keywords = data.get("keywords", [])
        query = data.get("query", "")

        # Load documents
        docs = await self._load_documents()

        # Score and rank documents
        scored_docs = []
        for doc in docs:
            score = self._calculate_relevance(doc, keywords, query)
            if score > 0:
                scored_docs.append({
                    "doc": doc,
                    "score": score
                })

        # Sort by score and take top k
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        retrieved = [item["doc"] for item in scored_docs[:self.top_k]]

        self.logger.info(f"Retrieved {len(retrieved)} documents from {len(docs)} total")

        return {
            **data,
            "retrieved_docs": retrieved,
            "num_retrieved": len(retrieved),
            "total_docs": len(docs)
        }

    async def _load_documents(self) -> List[Dict[str, Any]]:
        """Load documents from filesystem."""
        pipeline_dir = Path(__file__).parent.parent
        docs_path = pipeline_dir / self.docs_dir

        if not docs_path.exists():
            self.logger.warning(f"Docs directory not found: {docs_path}")
            return []

        documents = []
        for file_path in docs_path.glob("*"):
            if file_path.suffix in [".txt", ".md", ".json"]:
                try:
                    content = file_path.read_text(encoding="utf-8")
                    documents.append({
                        "filename": file_path.name,
                        "path": str(file_path),
                        "content": content,
                        "size": len(content)
                    })
                except Exception as e:
                    self.logger.warning(f"Failed to load {file_path.name}: {e}")

        return documents

    def _calculate_relevance(self, doc: Dict[str, Any], keywords: List[str], query: str) -> float:
        """Calculate document relevance score."""
        content = doc["content"].lower()
        score = 0.0

        # Keyword matching
        for keyword in keywords:
            if keyword.lower() in content:
                score += content.count(keyword.lower())

        # Exact query match bonus
        if query.lower() in content:
            score += 10

        return score
