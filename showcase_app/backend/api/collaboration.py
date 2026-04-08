"""
Collaboration Patterns API Endpoints

Demonstrates the four agent collaboration patterns from ia_modules:
- Consensus, Debate, Hierarchical, Peer-to-Peer
"""

import sys
from pathlib import Path

# Ensure backend directory is in sys.path
_backend_dir = str(Path(__file__).parent.parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from services.collaboration_service import CollaborationService

router = APIRouter()

collaboration_service = CollaborationService()


# ==================== REQUEST MODELS ====================

class ConsensusRequest(BaseModel):
    """Request to run consensus collaboration"""
    topic: str = Field(..., description="Proposal or topic to reach consensus on")
    agents: List[str] = Field(
        default=["Analyst", "Engineer", "Designer", "PM", "QA"],
        description="Names for participating agents"
    )
    strategy: str = Field(
        default="majority",
        description="Consensus strategy: unanimous, majority, supermajority, weighted"
    )
    max_iterations: int = Field(default=3, ge=1, le=10, description="Maximum refinement iterations")


class DebateRequest(BaseModel):
    """Request to run debate collaboration"""
    topic: str = Field(..., description="Debate topic")
    proponents: List[str] = Field(
        default=["Advocate-1"],
        description="Names for proponent agents"
    )
    opponents: List[str] = Field(
        default=["Skeptic-1"],
        description="Names for opponent agents"
    )
    moderator: str = Field(default="Moderator", description="Moderator agent name")
    rounds: int = Field(default=2, ge=1, le=5, description="Number of debate rounds")


class HierarchicalRequest(BaseModel):
    """Request to run hierarchical collaboration"""
    task: str = Field(..., description="High-level task to execute")
    leader: str = Field(default="Leader", description="Leader agent name")
    workers: List[str] = Field(
        default=["Worker-Alpha", "Worker-Beta", "Worker-Gamma"],
        description="Names for worker agents"
    )


class PeerToPeerRequest(BaseModel):
    """Request to run peer-to-peer collaboration"""
    task: str = Field(..., description="Task for peers to collaborate on")
    peers: List[str] = Field(
        default=["Peer-Alice", "Peer-Bob", "Peer-Carol", "Peer-Dave"],
        description="Names for peer agents"
    )
    rounds: int = Field(default=2, ge=1, le=5, description="Number of collaboration rounds")


# ==================== ENDPOINTS ====================

@router.get("/patterns")
async def list_patterns() -> Dict[str, Any]:
    """
    List available collaboration patterns

    Returns all four collaboration patterns with descriptions,
    use cases, and configuration options.
    """
    try:
        patterns = collaboration_service.list_patterns()
        return {
            "patterns": patterns,
            "total": len(patterns),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/consensus")
async def run_consensus(request: ConsensusRequest) -> Dict[str, Any]:
    """
    Run consensus collaboration

    Agents collaborate to reach agreement through proposal generation,
    discussion, voting, and iterative refinement.
    """
    try:
        result = await collaboration_service.run_consensus(
            topic=request.topic,
            agent_names=request.agents,
            strategy=request.strategy,
            max_iterations=request.max_iterations,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/debate")
async def run_debate(request: DebateRequest) -> Dict[str, Any]:
    """
    Run debate collaboration

    Proponents and opponents argue different perspectives through
    structured rounds, moderated by a facilitator.
    """
    try:
        result = await collaboration_service.run_debate(
            topic=request.topic,
            proponent_names=request.proponents,
            opponent_names=request.opponents,
            moderator_name=request.moderator,
            rounds=request.rounds,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hierarchical")
async def run_hierarchical(request: HierarchicalRequest) -> Dict[str, Any]:
    """
    Run hierarchical collaboration

    A leader decomposes the task, delegates subtasks to workers,
    and synthesizes their results.
    """
    try:
        result = await collaboration_service.run_hierarchical(
            task=request.task,
            leader_name=request.leader,
            worker_names=request.workers,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/peer-to-peer")
async def run_peer_to_peer(request: PeerToPeerRequest) -> Dict[str, Any]:
    """
    Run peer-to-peer collaboration

    Equal agents contribute independently, share and review each
    other's work, and iteratively refine across rounds.
    """
    try:
        result = await collaboration_service.run_peer_to_peer(
            task=request.task,
            peer_names=request.peers,
            rounds=request.rounds,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
