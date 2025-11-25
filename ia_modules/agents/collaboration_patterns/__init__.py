"""
Agent collaboration patterns for multi-agent workflows.

Provides ready-to-use collaboration patterns:
- Hierarchical: Leader-worker pattern with task delegation
- Peer-to-Peer: Equal collaboration and knowledge sharing
- Debate: Adversarial argumentation for critical analysis
- Consensus: Agreement-based decision making
"""

from .hierarchical import (
    HierarchicalCollaboration,
    LeaderAgent,
    WorkerAgent
)

from .peer_to_peer import (
    PeerToPeerCollaboration,
    PeerAgent
)

from .debate import (
    DebateCollaboration,
    DebateAgent,
    ModeratorAgent,
    DebateRole
)

from .consensus import (
    ConsensusCollaboration,
    ConsensusAgent,
    ConsensusStrategy,
    VoteType
)


__all__ = [
    # Hierarchical pattern
    "HierarchicalCollaboration",
    "LeaderAgent",
    "WorkerAgent",

    # Peer-to-peer pattern
    "PeerToPeerCollaboration",
    "PeerAgent",

    # Debate pattern
    "DebateCollaboration",
    "DebateAgent",
    "ModeratorAgent",
    "DebateRole",

    # Consensus pattern
    "ConsensusCollaboration",
    "ConsensusAgent",
    "ConsensusStrategy",
    "VoteType",
]
