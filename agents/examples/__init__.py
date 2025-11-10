"""
Agent collaboration pattern examples.

Demonstrates practical use cases for the four collaboration patterns:
- Hierarchical: Leader-worker delegation
- Debate: Adversarial argumentation
- Consensus: Agreement-based decision making
- Peer-to-Peer: Equal collaboration

Each example can be run standalone or used as a template for your own implementations.
"""

from .hierarchical_research_team import run_research_team_example
from .debate_code_review import run_code_review_debate_example
from .consensus_product_decision import run_product_decision_consensus_example
from .peer_knowledge_sharing import run_knowledge_sharing_example

__all__ = [
    "run_research_team_example",
    "run_code_review_debate_example",
    "run_product_decision_consensus_example",
    "run_knowledge_sharing_example",
]
