"""
Collaboration Patterns Service

Demonstrates the four agent collaboration patterns from ia_modules:
- Consensus: Agreement-based decision making
- Debate: Adversarial argumentation
- Hierarchical: Leader-worker delegation
- Peer-to-Peer: Equal collaboration
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, UTC
import asyncio
import logging

from ia_modules.agents.collaboration_patterns import (
    ConsensusCollaboration,
    ConsensusAgent,
    ConsensusStrategy,
    VoteType,
    DebateCollaboration,
    DebateAgent,
    ModeratorAgent,
    DebateRole,
    HierarchicalCollaboration,
    LeaderAgent,
    WorkerAgent,
    PeerToPeerCollaboration,
    PeerAgent,
)
from ia_modules.agents.core import AgentRole
from ia_modules.agents.communication import MessageBus
from ia_modules.agents.state import StateManager

logger = logging.getLogger("CollaborationService")


class CollaborationService:
    """Service for running agent collaboration pattern demos."""

    PATTERN_DESCRIPTIONS = {
        "consensus": {
            "name": "Consensus",
            "description": "Agreement-based decision making where agents collaborate through proposal generation, discussion, voting, and iterative refinement until consensus is reached.",
            "use_cases": [
                "Democratic decision making",
                "Quality assurance through agreement",
                "Collaborative problem solving",
                "Risk mitigation through diverse input",
            ],
            "strategies": [s.value for s in ConsensusStrategy],
            "vote_types": [v.value for v in VoteType],
        },
        "debate": {
            "name": "Debate",
            "description": "Adversarial argumentation where proponent and opponent agents argue different perspectives through structured rounds, moderated by a facilitator.",
            "use_cases": [
                "Exploring multiple perspectives",
                "Stress-testing ideas",
                "Critical analysis",
                "Decision making under uncertainty",
            ],
            "roles": [r.value for r in DebateRole],
        },
        "hierarchical": {
            "name": "Hierarchical",
            "description": "Leader-worker pattern where a leader agent decomposes tasks, delegates subtasks to workers, and synthesizes their results into a final output.",
            "use_cases": [
                "Complex task decomposition",
                "Parallel subtask execution",
                "Coordinated multi-step workflows",
                "Result synthesis from multiple sources",
            ],
        },
        "peer_to_peer": {
            "name": "Peer-to-Peer",
            "description": "Equal collaboration where peer agents contribute independently, share and review each other's work, and iteratively refine contributions across rounds.",
            "use_cases": [
                "Brainstorming sessions",
                "Collaborative problem solving",
                "Knowledge sharing",
                "Creative ideation",
            ],
        },
    }

    def list_patterns(self) -> List[Dict[str, Any]]:
        """List all available collaboration patterns with descriptions."""
        return [
            {"id": pattern_id, **info}
            for pattern_id, info in self.PATTERN_DESCRIPTIONS.items()
        ]

    async def run_consensus(
        self,
        topic: str,
        agent_names: List[str],
        strategy: str = "majority",
        max_iterations: int = 3,
    ) -> Dict[str, Any]:
        """
        Run a consensus collaboration demo.

        Args:
            topic: The proposal/topic to reach consensus on
            agent_names: Names for the participating agents
            strategy: Consensus strategy (unanimous, majority, supermajority, weighted)
            max_iterations: Maximum refinement iterations

        Returns:
            Consensus result with step-by-step history
        """
        started_at = datetime.now(UTC)
        history = []

        # Parse strategy
        try:
            consensus_strategy = ConsensusStrategy(strategy)
        except ValueError:
            consensus_strategy = ConsensusStrategy.MAJORITY

        # Setup infrastructure
        state = StateManager(thread_id=f"consensus_{int(started_at.timestamp())}")
        bus = MessageBus()

        # Create agents
        agents = []
        for name in agent_names:
            agent = ConsensusAgent(
                role=AgentRole(name=name, description=f"Consensus participant: {name}"),
                state_manager=state,
                message_bus=bus,
            )
            agents.append(agent)

        history.append({
            "phase": "setup",
            "message": f"Created {len(agents)} consensus agents with {consensus_strategy.value} strategy",
            "agents": agent_names,
            "timestamp": datetime.now(UTC).isoformat(),
        })

        # Create collaboration
        collaboration = ConsensusCollaboration(
            agents=agents,
            message_bus=bus,
            state_manager=state,
            strategy=consensus_strategy,
            max_iterations=max_iterations,
        )

        # Initialize
        await collaboration.initialize()
        history.append({
            "phase": "initialization",
            "message": "All agents initialized and subscribed to message bus",
            "timestamp": datetime.now(UTC).isoformat(),
        })

        # Execute
        try:
            result = await collaboration.execute({
                "proposal": topic,
                "context": {"strategy": strategy},
            })

            # Build history from result
            for vote_record in result.get("votes_history", []):
                iteration = vote_record.get("iteration", 0)
                votes = vote_record.get("votes", {})
                history.append({
                    "phase": f"voting_round_{iteration}",
                    "message": f"Voting round {iteration}",
                    "votes": {
                        agent_id: {
                            "vote": v.get("vote", "abstain"),
                            "confidence": v.get("confidence", 0),
                            "reasoning": v.get("reasoning", ""),
                        }
                        for agent_id, v in votes.items()
                    },
                    "timestamp": datetime.now(UTC).isoformat(),
                })

            for proposal in result.get("proposal_history", []):
                history.append({
                    "phase": f"proposal_v{proposal.get('version', 1)}",
                    "message": f"Proposal version {proposal.get('version', 1)} ({proposal.get('source', 'unknown')})",
                    "content": proposal.get("content", ""),
                    "timestamp": datetime.now(UTC).isoformat(),
                })

            history.append({
                "phase": "result",
                "message": "Consensus reached" if result.get("consensus_reached") else "No consensus reached",
                "consensus_reached": result.get("consensus_reached", False),
                "agreement_level": result.get("agreement_level", 0),
                "iterations": result.get("iterations", 0),
                "timestamp": datetime.now(UTC).isoformat(),
            })

        except Exception as e:
            logger.error(f"Consensus execution failed: {e}")
            result = {"status": "error", "error": str(e)}
            history.append({
                "phase": "error",
                "message": f"Execution failed: {e}",
                "timestamp": datetime.now(UTC).isoformat(),
            })
        finally:
            await collaboration.shutdown()

        return {
            "pattern": "consensus",
            "topic": topic,
            "strategy": consensus_strategy.value,
            "agents": agent_names,
            "result": result,
            "history": history,
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now(UTC).isoformat(),
        }

    async def run_debate(
        self,
        topic: str,
        proponent_names: List[str],
        opponent_names: List[str],
        moderator_name: str = "Moderator",
        rounds: int = 2,
    ) -> Dict[str, Any]:
        """
        Run a debate collaboration demo.

        Args:
            topic: The debate topic
            proponent_names: Names for proponent agents
            opponent_names: Names for opponent agents
            moderator_name: Name for the moderator agent
            rounds: Number of debate rounds

        Returns:
            Debate result with step-by-step history
        """
        started_at = datetime.now(UTC)
        history = []

        # Setup
        state = StateManager(thread_id=f"debate_{int(started_at.timestamp())}")
        bus = MessageBus()

        # Create proponents
        proponents = []
        for name in proponent_names:
            agent = DebateAgent(
                role=AgentRole(name=name, description=f"Proponent: {name}"),
                state_manager=state,
                message_bus=bus,
                debate_role=DebateRole.PROPONENT,
            )
            proponents.append(agent)

        # Create opponents
        opponents = []
        for name in opponent_names:
            agent = DebateAgent(
                role=AgentRole(name=name, description=f"Opponent: {name}"),
                state_manager=state,
                message_bus=bus,
                debate_role=DebateRole.OPPONENT,
            )
            opponents.append(agent)

        # Create moderator
        moderator = ModeratorAgent(
            role=AgentRole(name=moderator_name, description="Debate moderator"),
            state_manager=state,
            message_bus=bus,
        )

        history.append({
            "phase": "setup",
            "message": f"Debate configured: {len(proponents)} proponents vs {len(opponents)} opponents, {rounds} rounds",
            "proponents": proponent_names,
            "opponents": opponent_names,
            "moderator": moderator_name,
            "timestamp": datetime.now(UTC).isoformat(),
        })

        # Create collaboration
        debate = DebateCollaboration(
            proponents=proponents,
            opponents=opponents,
            moderator=moderator,
            message_bus=bus,
            state_manager=state,
        )

        # Initialize
        await debate.initialize()
        history.append({
            "phase": "initialization",
            "message": "All debate participants initialized",
            "timestamp": datetime.now(UTC).isoformat(),
        })

        # Execute
        try:
            result = await debate.execute({
                "topic": topic,
                "rounds": rounds,
            })

            # Build history from result
            opening = result.get("opening_statements", {})
            for side in ["proponents", "opponents"]:
                for stmt in opening.get(side, []):
                    history.append({
                        "phase": "opening_statement",
                        "side": side,
                        "agent": stmt.get("agent", "unknown"),
                        "statement": stmt.get("statement", {}),
                        "timestamp": datetime.now(UTC).isoformat(),
                    })

            for round_data in result.get("debate_rounds", []):
                round_num = round_data.get("round", 0)
                for arg in round_data.get("proponent_arguments", []):
                    history.append({
                        "phase": f"round_{round_num}_argument",
                        "side": "proponent",
                        "agent": arg.get("agent", "unknown"),
                        "argument": arg.get("argument", {}),
                        "timestamp": datetime.now(UTC).isoformat(),
                    })
                for arg in round_data.get("opponent_arguments", []):
                    history.append({
                        "phase": f"round_{round_num}_argument",
                        "side": "opponent",
                        "agent": arg.get("agent", "unknown"),
                        "argument": arg.get("argument", {}),
                        "timestamp": datetime.now(UTC).isoformat(),
                    })

            closing = result.get("closing_statements", {})
            for side in ["proponents", "opponents"]:
                for stmt in closing.get(side, []):
                    history.append({
                        "phase": "closing_statement",
                        "side": side,
                        "agent": stmt.get("agent", "unknown"),
                        "statement": stmt.get("statement", {}),
                        "timestamp": datetime.now(UTC).isoformat(),
                    })

            history.append({
                "phase": "result",
                "message": "Debate completed",
                "total_rounds": result.get("total_rounds", 0),
                "total_arguments": result.get("total_arguments", 0),
                "proponent_key_points": result.get("proponent_key_points", []),
                "opponent_key_points": result.get("opponent_key_points", []),
                "timestamp": datetime.now(UTC).isoformat(),
            })

        except Exception as e:
            logger.error(f"Debate execution failed: {e}")
            result = {"status": "error", "error": str(e)}
            history.append({
                "phase": "error",
                "message": f"Execution failed: {e}",
                "timestamp": datetime.now(UTC).isoformat(),
            })
        finally:
            await debate.shutdown()

        return {
            "pattern": "debate",
            "topic": topic,
            "proponents": proponent_names,
            "opponents": opponent_names,
            "moderator": moderator_name,
            "rounds": rounds,
            "result": result,
            "history": history,
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now(UTC).isoformat(),
        }

    async def run_hierarchical(
        self,
        task: str,
        leader_name: str = "Leader",
        worker_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run a hierarchical collaboration demo.

        Args:
            task: The high-level task to execute
            leader_name: Name for the leader agent
            worker_names: Names for worker agents

        Returns:
            Hierarchical result with step-by-step history
        """
        started_at = datetime.now(UTC)
        history = []

        if worker_names is None:
            worker_names = ["Worker-Alpha", "Worker-Beta", "Worker-Gamma"]

        # Setup
        state = StateManager(thread_id=f"hierarchical_{int(started_at.timestamp())}")
        bus = MessageBus()

        # Create leader
        leader = LeaderAgent(
            role=AgentRole(name=leader_name, description="Coordinates workers"),
            state_manager=state,
            message_bus=bus,
        )

        # Create workers
        workers = []
        for name in worker_names:
            worker = WorkerAgent(
                role=AgentRole(name=name, description=f"Worker: {name}"),
                state_manager=state,
                message_bus=bus,
            )
            workers.append(worker)

        history.append({
            "phase": "setup",
            "message": f"Hierarchical team: 1 leader + {len(workers)} workers",
            "leader": leader_name,
            "workers": worker_names,
            "timestamp": datetime.now(UTC).isoformat(),
        })

        # Create collaboration
        collaboration = HierarchicalCollaboration(
            leader=leader,
            workers=workers,
            message_bus=bus,
        )

        # Initialize
        await collaboration.initialize()
        history.append({
            "phase": "initialization",
            "message": "Leader and all workers initialized",
            "timestamp": datetime.now(UTC).isoformat(),
        })

        # Execute
        try:
            result = await collaboration.execute({"task": task})

            history.append({
                "phase": "task_decomposition",
                "message": f"Task decomposed by leader",
                "task": task,
                "timestamp": datetime.now(UTC).isoformat(),
            })

            # Worker results
            for output in result.get("worker_outputs", []):
                if isinstance(output, dict):
                    history.append({
                        "phase": "worker_result",
                        "worker": output.get("worker", "unknown"),
                        "task_id": output.get("task_id", "unknown"),
                        "status": output.get("status", "unknown"),
                        "result": output.get("result", ""),
                        "timestamp": datetime.now(UTC).isoformat(),
                    })

            history.append({
                "phase": "result",
                "message": "Hierarchical execution completed",
                "total_workers": result.get("total_workers", 0),
                "successful_workers": result.get("successful_workers", 0),
                "failed_workers": result.get("failed_workers", 0),
                "status": result.get("status", "unknown"),
                "summary": result.get("summary", ""),
                "timestamp": datetime.now(UTC).isoformat(),
            })

        except Exception as e:
            logger.error(f"Hierarchical execution failed: {e}")
            result = {"status": "error", "error": str(e)}
            history.append({
                "phase": "error",
                "message": f"Execution failed: {e}",
                "timestamp": datetime.now(UTC).isoformat(),
            })
        finally:
            await collaboration.shutdown()

        return {
            "pattern": "hierarchical",
            "task": task,
            "leader": leader_name,
            "workers": worker_names,
            "result": result,
            "history": history,
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now(UTC).isoformat(),
        }

    async def run_peer_to_peer(
        self,
        task: str,
        peer_names: Optional[List[str]] = None,
        rounds: int = 2,
    ) -> Dict[str, Any]:
        """
        Run a peer-to-peer collaboration demo.

        Args:
            task: The task for peers to collaborate on
            peer_names: Names for peer agents
            rounds: Number of collaboration rounds

        Returns:
            Peer-to-peer result with step-by-step history
        """
        started_at = datetime.now(UTC)
        history = []

        if peer_names is None:
            peer_names = ["Peer-Alice", "Peer-Bob", "Peer-Carol", "Peer-Dave"]

        # Setup
        state = StateManager(thread_id=f"p2p_{int(started_at.timestamp())}")
        bus = MessageBus()

        # Create peers
        peers = []
        for name in peer_names:
            peer = PeerAgent(
                role=AgentRole(name=name, description=f"Peer collaborator: {name}"),
                state_manager=state,
                message_bus=bus,
            )
            peers.append(peer)

        history.append({
            "phase": "setup",
            "message": f"Peer-to-peer session: {len(peers)} peers, {rounds} rounds",
            "peers": peer_names,
            "rounds": rounds,
            "timestamp": datetime.now(UTC).isoformat(),
        })

        # Create collaboration
        collaboration = PeerToPeerCollaboration(
            peers=peers,
            message_bus=bus,
            state_manager=state,
        )

        # Initialize
        await collaboration.initialize()
        history.append({
            "phase": "initialization",
            "message": "All peers initialized",
            "timestamp": datetime.now(UTC).isoformat(),
        })

        # Execute
        try:
            result = await collaboration.execute(
                task={"task": task},
                rounds=rounds,
            )

            # Build history from contributions
            contributions_by_round = result.get("contributions_by_round", {})
            for round_num in sorted(contributions_by_round.keys(), key=lambda x: int(x)):
                round_contributions = contributions_by_round[round_num]
                for contrib in round_contributions:
                    history.append({
                        "phase": f"round_{round_num}_contribution",
                        "peer": contrib.get("peer_id", "unknown"),
                        "content": contrib.get("content", {}),
                        "refined": contrib.get("refined", False),
                        "timestamp": datetime.now(UTC).isoformat(),
                    })

            history.append({
                "phase": "result",
                "message": "Peer-to-peer collaboration completed",
                "total_peers": result.get("total_peers", 0),
                "total_contributions": result.get("total_contributions", 0),
                "rounds_completed": result.get("rounds_completed", 0),
                "key_insights": result.get("key_insights", []),
                "status": result.get("status", "unknown"),
                "timestamp": datetime.now(UTC).isoformat(),
            })

        except Exception as e:
            logger.error(f"P2P execution failed: {e}")
            result = {"status": "error", "error": str(e)}
            history.append({
                "phase": "error",
                "message": f"Execution failed: {e}",
                "timestamp": datetime.now(UTC).isoformat(),
            })
        finally:
            await collaboration.shutdown()

        return {
            "pattern": "peer_to_peer",
            "task": task,
            "peers": peer_names,
            "rounds": rounds,
            "result": result,
            "history": history,
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now(UTC).isoformat(),
        }
