"""
Peer-to-peer collaboration pattern (equal collaboration).

Implements a peer-to-peer structure where agents collaborate as equals,
sharing information and building on each other's work.
"""

import asyncio
from typing import Dict, Any, List, Optional, Set
import logging

from ..base_agent import BaseCollaborativeAgent
from ..communication import MessageBus, MessageType, AgentMessage
from ..state import StateManager


class PeerToPeerCollaboration:
    """
    Peer-to-peer collaboration pattern with equal agents.

    All agents in this pattern:
    - Have equal status and authority
    - Share information freely
    - Build on each other's contributions
    - Collectively solve problems
    - No hierarchical structure

    Example:
        >>> # Setup
        >>> state = StateManager(thread_id="p2p_001")
        >>> bus = MessageBus()
        >>>
        >>> # Create peer agents
        >>> peers = [
        ...     PeerAgent(
        ...         role=AgentRole(name=f"peer_{i}", description="Collaborates equally"),
        ...         state_manager=state,
        ...         message_bus=bus
        ...     )
        ...     for i in range(4)
        ... ]
        >>>
        >>> # Setup collaboration
        >>> collaboration = PeerToPeerCollaboration(peers, bus, state)
        >>> await collaboration.initialize()
        >>>
        >>> # Execute collaborative task
        >>> result = await collaboration.execute({
        ...     "task": "Brainstorm innovative product ideas"
        ... })
    """

    def __init__(self, peers: List[BaseCollaborativeAgent],
                 message_bus: MessageBus,
                 state_manager: StateManager):
        """
        Initialize peer-to-peer collaboration.

        Args:
            peers: List of peer agents
            message_bus: Message bus for communication
            state_manager: Shared state manager
        """
        self.peers = peers
        self.message_bus = message_bus
        self.state = state_manager
        self.logger = logging.getLogger("PeerToPeerCollaboration")

        # Track contributions from each peer
        self.contributions: Dict[str, List[Dict[str, Any]]] = {}

    async def initialize(self) -> None:
        """Initialize all peer agents."""
        for peer in self.peers:
            await peer.initialize()
            self.contributions[peer.agent_id] = []

        self.logger.info(f"P2P collaboration initialized with {len(self.peers)} peers")

    async def shutdown(self) -> None:
        """Shutdown all peer agents."""
        for peer in self.peers:
            await peer.shutdown()

    async def execute(self, task: Dict[str, Any],
                     rounds: int = 3) -> Dict[str, Any]:
        """
        Execute task through peer collaboration.

        Agents collaborate in rounds:
        1. Each peer contributes independently
        2. Peers share and review contributions
        3. Peers build on each other's ideas
        4. Process repeats for multiple rounds
        5. Final synthesis of all contributions

        Args:
            task: Task for peers to collaborate on
            rounds: Number of collaboration rounds

        Returns:
            Synthesized results from all peers
        """
        self.logger.info(f"Starting P2P collaboration: {rounds} rounds")

        # Initialize shared context
        await self.state.set("collaboration_task", task)
        await self.state.set("current_round", 0)

        all_contributions = []

        # Execute collaboration rounds
        for round_num in range(1, rounds + 1):
            self.logger.info(f"Round {round_num}/{rounds}")

            await self.state.set("current_round", round_num)

            # Phase 1: Each peer contributes
            round_contributions = await self._gather_contributions(task, round_num)

            # Phase 2: Share contributions with all peers
            await self._share_contributions(round_contributions)

            # Phase 3: Peers review and refine
            refined_contributions = await self._refine_contributions(
                round_contributions, round_num
            )

            all_contributions.extend(refined_contributions)

            # Update shared context for next round
            await self.state.set(f"round_{round_num}_contributions", refined_contributions)

        # Final synthesis
        final_result = await self._synthesize_all_contributions(
            all_contributions, task
        )

        self.logger.info("P2P collaboration complete")

        return final_result

    async def _gather_contributions(self, task: Dict[str, Any],
                                   round_num: int) -> List[Dict[str, Any]]:
        """
        Gather contributions from all peers in parallel.

        Args:
            task: Task to work on
            round_num: Current round number

        Returns:
            List of contributions from all peers
        """
        # Get previous round's contributions for context
        previous_contributions = []
        if round_num > 1:
            previous_contributions = await self.state.get(
                f"round_{round_num - 1}_contributions", []
            )

        # Request contributions from all peers in parallel
        contribution_tasks = []

        for peer in self.peers:
            contribution_task = peer.send_task_request(
                recipient=peer.agent_id,
                task_data={
                    "action": "contribute",
                    "task": task,
                    "round": round_num,
                    "previous_contributions": previous_contributions
                },
                wait_for_response=True,
                timeout=30.0
            )
            contribution_tasks.append((peer.agent_id, contribution_task))

        # Wait for all contributions
        contributions = []

        for peer_id, task_future in contribution_tasks:
            try:
                response = await task_future
                if response:
                    contribution = response.content
                    contribution["peer_id"] = peer_id
                    contribution["round"] = round_num
                    contributions.append(contribution)

                    self.logger.debug(f"Received contribution from {peer_id}")

            except Exception as e:
                self.logger.error(f"Failed to get contribution from {peer_id}: {e}")

        return contributions

    async def _share_contributions(self, contributions: List[Dict[str, Any]]) -> None:
        """
        Share all contributions with all peers.

        Args:
            contributions: Contributions to share
        """
        # Broadcast contributions to all peers
        for peer in self.peers:
            # Filter out peer's own contribution
            others_contributions = [
                c for c in contributions
                if c.get("peer_id") != peer.agent_id
            ]

            # Send to peer
            try:
                await peer.send_message(
                    recipient=peer.agent_id,
                    message_type=MessageType.BROADCAST,
                    content={
                        "action": "review_contributions",
                        "contributions": others_contributions
                    }
                )
            except Exception as e:
                self.logger.error(f"Failed to share with {peer.agent_id}: {e}")

    async def _refine_contributions(self, contributions: List[Dict[str, Any]],
                                   round_num: int) -> List[Dict[str, Any]]:
        """
        Peers refine contributions based on feedback.

        Args:
            contributions: Original contributions
            round_num: Current round number

        Returns:
            Refined contributions
        """
        # Give peers time to review
        await asyncio.sleep(0.5)

        refined = []

        for contribution in contributions:
            peer_id = contribution.get("peer_id")

            # Ask peer to refine based on others' contributions
            try:
                peer = next((p for p in self.peers if p.agent_id == peer_id), None)
                if not peer:
                    continue

                response = await peer.send_query(
                    recipient=peer_id,
                    query={
                        "action": "refine",
                        "original_contribution": contribution,
                        "round": round_num
                    },
                    timeout=15.0
                )

                refined_contribution = response.content
                refined_contribution["refined"] = True
                refined.append(refined_contribution)

            except Exception as e:
                self.logger.warning(f"Failed to refine contribution from {peer_id}: {e}")
                # Use original if refinement fails
                refined.append(contribution)

        return refined

    async def _synthesize_all_contributions(self,
                                          all_contributions: List[Dict[str, Any]],
                                          original_task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize all contributions into final result.

        Args:
            all_contributions: All contributions from all rounds
            original_task: Original task

        Returns:
            Synthesized final result
        """
        # Group contributions by peer
        contributions_by_peer = {}
        for contribution in all_contributions:
            peer_id = contribution.get("peer_id", "unknown")
            if peer_id not in contributions_by_peer:
                contributions_by_peer[peer_id] = []
            contributions_by_peer[peer_id].append(contribution)

        # Group contributions by round
        contributions_by_round = {}
        for contribution in all_contributions:
            round_num = contribution.get("round", 0)
            if round_num not in contributions_by_round:
                contributions_by_round[round_num] = []
            contributions_by_round[round_num].append(contribution)

        # Create synthesis
        synthesis = {
            "task": original_task,
            "total_peers": len(self.peers),
            "total_contributions": len(all_contributions),
            "rounds_completed": len(contributions_by_round),
            "contributions_by_peer": contributions_by_peer,
            "contributions_by_round": contributions_by_round,
            "key_insights": self._extract_key_insights(all_contributions),
            "collaborative_output": self._merge_contributions(all_contributions),
            "status": "success"
        }

        return synthesis

    def _extract_key_insights(self, contributions: List[Dict[str, Any]]) -> List[str]:
        """Extract key insights from all contributions."""
        insights = []

        for contribution in contributions:
            # Extract insights from contribution
            if "insights" in contribution:
                insights.extend(contribution["insights"])
            elif "content" in contribution:
                content = contribution["content"]
                if isinstance(content, dict) and "key_points" in content:
                    insights.extend(content["key_points"])

        # Remove duplicates while preserving order
        seen = set()
        unique_insights = []
        for insight in insights:
            if insight not in seen:
                seen.add(insight)
                unique_insights.append(insight)

        return unique_insights[:10]  # Return top 10

    def _merge_contributions(self, contributions: List[Dict[str, Any]]) -> str:
        """Merge all contributions into coherent output."""
        merged_parts = ["Collaborative Output from Peer-to-Peer Session:\n"]

        # Group by round
        by_round = {}
        for contribution in contributions:
            round_num = contribution.get("round", 0)
            if round_num not in by_round:
                by_round[round_num] = []
            by_round[round_num].append(contribution)

        # Format by round
        for round_num in sorted(by_round.keys()):
            merged_parts.append(f"\nRound {round_num}:")
            for i, contribution in enumerate(by_round[round_num], 1):
                peer_id = contribution.get("peer_id", "unknown")
                content = contribution.get("content", "")
                if isinstance(content, dict):
                    content = content.get("summary", str(content))
                merged_parts.append(f"  {i}. [{peer_id}] {content}")

        return "\n".join(merged_parts)


class PeerAgent(BaseCollaborativeAgent):
    """
    Peer agent that collaborates equally with other peers.

    Extends BaseCollaborativeAgent with peer collaboration capabilities:
    - Contributing ideas and solutions
    - Reviewing peer contributions
    - Building on others' work
    - Collaborative refinement
    """

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute as peer in collaborative task.

        Args:
            input_data: Task or action to perform

        Returns:
            Contribution or response
        """
        action = input_data.get("action", "contribute")

        if action == "contribute":
            return await self._contribute(input_data)
        elif action == "review_contributions":
            return await self._review_contributions(input_data)
        elif action == "refine":
            return await self._refine_contribution(input_data)
        else:
            return {"status": "unknown_action", "action": action}

    async def _contribute(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a contribution to the collaborative task.

        Args:
            task_data: Task information and context

        Returns:
            Contribution
        """
        task = task_data.get("task", {})
        round_num = task_data.get("round", 1)
        previous = task_data.get("previous_contributions", [])

        self.logger.info(f"Peer {self.agent_id} contributing in round {round_num}")

        # Generate contribution (simplified - would use actual processing)
        contribution = await self._generate_contribution(task, previous)

        # Store in state
        await self.write_state(f"{self.agent_id}_last_contribution", contribution)

        return {
            "status": "contributed",
            "content": contribution,
            "peer_id": self.agent_id,
            "round": round_num
        }

    async def _generate_contribution(self, task: Dict[str, Any],
                                    previous: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate contribution based on task and previous contributions.

        Override in specialized peer implementations.

        Args:
            task: Task to contribute to
            previous: Previous round contributions

        Returns:
            New contribution
        """
        # Analyze previous contributions
        previous_insights = []
        for prev in previous:
            if isinstance(prev, dict) and "insights" in prev:
                previous_insights.extend(prev["insights"])

        # Generate new insights building on previous work
        new_insights = [
            f"Building on previous work: New insight from {self.agent_id}",
            f"Alternative perspective from {self.agent_id}",
            f"Synthesis of ideas by {self.agent_id}"
        ]

        return {
            "summary": f"Contribution from {self.agent_id}",
            "insights": new_insights,
            "builds_on": len(previous_insights),
            "key_points": new_insights[:2]
        }

    async def _review_contributions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Review contributions from other peers.

        Args:
            data: Contains contributions to review

        Returns:
            Review results
        """
        contributions = data.get("contributions", [])

        self.logger.info(f"Peer {self.agent_id} reviewing {len(contributions)} contributions")

        # Analyze contributions
        reviews = []
        for contribution in contributions:
            peer_id = contribution.get("peer_id", "unknown")
            content = contribution.get("content", {})

            review = {
                "reviewed_peer": peer_id,
                "feedback": f"Interesting perspective from {peer_id}",
                "rating": 0.8,  # Would use actual evaluation
                "suggestions": ["Consider expanding on key points"]
            }
            reviews.append(review)

        # Store reviews in state
        await self.write_state(f"{self.agent_id}_reviews", reviews)

        return {
            "status": "reviewed",
            "reviews": reviews
        }

    async def _refine_contribution(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refine previous contribution based on peer feedback.

        Args:
            data: Contains original contribution and refinement context

        Returns:
            Refined contribution
        """
        original = data.get("original_contribution", {})
        round_num = data.get("round", 0)

        self.logger.info(f"Peer {self.agent_id} refining contribution")

        # Get reviews from state
        reviews = await self.read_state(f"{self.agent_id}_reviews", [])

        # Refine based on feedback
        original_content = original.get("content", {})
        original_insights = original_content.get("insights", [])

        refined_insights = original_insights.copy()

        # Add refinements based on reviews
        if reviews:
            refined_insights.append(
                f"Refined based on peer feedback in round {round_num}"
            )

        refined = {
            "summary": f"Refined contribution from {self.agent_id}",
            "insights": refined_insights,
            "original_round": original.get("round", 0),
            "refined_in_round": round_num,
            "incorporated_feedback": len(reviews),
            "key_points": refined_insights[:3]
        }

        return {
            "status": "refined",
            "content": refined,
            "peer_id": self.agent_id
        }

    async def share_knowledge(self, knowledge: Dict[str, Any]) -> None:
        """
        Share knowledge with all peers.

        Args:
            knowledge: Knowledge to share
        """
        await self.broadcast_message(
            message_type=MessageType.BROADCAST,
            content={
                "action": "knowledge_share",
                "knowledge": knowledge,
                "from_peer": self.agent_id
            }
        )

        self.logger.info(f"Peer {self.agent_id} shared knowledge with all peers")

    async def request_help(self, problem: str) -> List[AgentMessage]:
        """
        Request help from peer agents.

        Args:
            problem: Problem description

        Returns:
            List of responses from peers
        """
        await self.broadcast_message(
            message_type=MessageType.QUERY,
            content={
                "action": "help_request",
                "problem": problem,
                "from_peer": self.agent_id
            }
        )

        self.logger.info(f"Peer {self.agent_id} requested help from peers")

        # Wait for responses
        await asyncio.sleep(2.0)

        # Get responses from queue
        responses = await self.get_pending_messages()

        return [msg for msg in responses if msg.message_type == MessageType.RESPONSE]
