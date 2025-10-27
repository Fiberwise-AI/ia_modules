"""
Consensus collaboration pattern (agreement-based).

Implements a consensus-building structure where agents work together to
reach agreement through discussion, voting, and iterative refinement.
"""

import asyncio
from typing import Dict, Any, List, Optional, Set
from enum import Enum
import logging

from ..base_agent import BaseCollaborativeAgent
from ..communication import MessageBus, MessageType, AgentMessage
from ..state import StateManager


class ConsensusStrategy(Enum):
    """Strategies for reaching consensus."""
    UNANIMOUS = "unanimous"  # All agents must agree
    MAJORITY = "majority"  # More than 50% must agree
    SUPERMAJORITY = "supermajority"  # 2/3 or more must agree
    WEIGHTED = "weighted"  # Votes weighted by expertise/confidence


class VoteType(Enum):
    """Types of votes agents can cast."""
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"
    CONDITIONAL = "conditional"  # Approve with conditions


class ConsensusCollaboration:
    """
    Consensus collaboration pattern for agreement-based decision making.

    Agents collaborate to reach consensus through:
    - Proposal generation
    - Discussion and refinement
    - Voting mechanisms
    - Iterative improvement until consensus
    - Conflict resolution

    This pattern is useful for:
    - Democratic decision making
    - Quality assurance through agreement
    - Collaborative problem solving
    - Risk mitigation through diverse input
    - Building shared understanding

    Example:
        >>> # Setup
        >>> state = StateManager(thread_id="consensus_001")
        >>> bus = MessageBus()
        >>>
        >>> # Create consensus agents
        >>> agents = [
        ...     ConsensusAgent(
        ...         role=AgentRole(name=f"agent_{i}", description="Participates in consensus"),
        ...         state_manager=state,
        ...         message_bus=bus
        ...     )
        ...     for i in range(5)
        ... ]
        >>>
        >>> # Setup consensus
        >>> consensus = ConsensusCollaboration(
        ...     agents=agents,
        ...     message_bus=bus,
        ...     state_manager=state,
        ...     strategy=ConsensusStrategy.MAJORITY
        ... )
        >>> await consensus.initialize()
        >>>
        >>> # Reach consensus on a decision
        >>> result = await consensus.execute({
        ...     "proposal": "Implement feature X in the system",
        ...     "context": {"urgency": "high", "risk": "medium"}
        ... })
    """

    def __init__(self,
                 agents: List[BaseCollaborativeAgent],
                 message_bus: MessageBus,
                 state_manager: StateManager,
                 strategy: ConsensusStrategy = ConsensusStrategy.MAJORITY,
                 max_iterations: int = 5):
        """
        Initialize consensus collaboration.

        Args:
            agents: List of participating agents
            message_bus: Message bus for communication
            state_manager: Shared state manager
            strategy: Consensus strategy to use
            max_iterations: Maximum refinement iterations
        """
        self.agents = agents
        self.message_bus = message_bus
        self.state = state_manager
        self.strategy = strategy
        self.max_iterations = max_iterations
        self.logger = logging.getLogger("ConsensusCollaboration")

        # Track consensus process
        self.proposals: List[Dict[str, Any]] = []
        self.votes_history: List[Dict[str, Any]] = []

    async def initialize(self) -> None:
        """Initialize all participating agents."""
        for agent in self.agents:
            await agent.initialize()

        self.logger.info(
            f"Consensus collaboration initialized: {len(self.agents)} agents, "
            f"strategy={self.strategy.value}"
        )

    async def shutdown(self) -> None:
        """Shutdown all agents."""
        for agent in self.agents:
            await agent.shutdown()

    async def execute(self, decision_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute consensus-building process.

        Args:
            decision_context: Context including:
                - proposal: Initial proposal (optional)
                - context: Decision context
                - requirements: Success requirements

        Returns:
            Consensus result with final agreed proposal
        """
        self.logger.info("Starting consensus-building process")

        # Store decision context
        await self.state.set("decision_context", decision_context)

        # Phase 1: Generate or receive initial proposal
        initial_proposal = await self._get_initial_proposal(decision_context)

        # Phase 2: Iterative consensus building
        consensus_reached = False
        iteration = 0
        current_proposal = initial_proposal

        while not consensus_reached and iteration < self.max_iterations:
            iteration += 1
            self.logger.info(f"Consensus iteration {iteration}/{self.max_iterations}")

            # Discuss proposal
            discussion = await self._discuss_proposal(current_proposal, iteration)

            # Vote on proposal
            votes = await self._collect_votes(current_proposal, iteration)

            # Check if consensus reached
            consensus_reached, agreement_level = self._check_consensus(votes)

            if consensus_reached:
                self.logger.info(f"Consensus reached! Agreement: {agreement_level:.1%}")
                break

            # Refine proposal based on feedback
            current_proposal = await self._refine_proposal(
                current_proposal, discussion, votes, iteration
            )

        # Phase 3: Finalize consensus
        final_result = await self._finalize_consensus(
            current_proposal, consensus_reached, iteration
        )

        self.logger.info(
            f"Consensus process complete: "
            f"{'Success' if consensus_reached else 'No consensus'}"
        )

        return final_result

    async def _get_initial_proposal(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get or generate initial proposal.

        Args:
            context: Decision context

        Returns:
            Initial proposal
        """
        # Check if proposal provided
        if "proposal" in context:
            proposal = {
                "content": context["proposal"],
                "source": "provided",
                "version": 1
            }
        else:
            # Generate collaborative proposal
            proposal = await self._generate_collaborative_proposal(context)

        self.proposals.append(proposal)
        await self.state.set("current_proposal", proposal)

        return proposal

    async def _generate_collaborative_proposal(self,
                                              context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate proposal collaboratively from all agents.

        Args:
            context: Decision context

        Returns:
            Generated proposal
        """
        self.logger.info("Generating collaborative proposal")

        # Ask each agent for proposal ideas
        proposals = []

        for agent in self.agents:
            try:
                response = await agent.send_task_request(
                    recipient=agent.agent_id,
                    task_data={
                        "action": "propose",
                        "context": context
                    },
                    wait_for_response=True,
                    timeout=20.0
                )

                if response:
                    proposals.append({
                        "agent": agent.agent_id,
                        "proposal": response.content
                    })

            except Exception as e:
                self.logger.error(f"Failed to get proposal from {agent.agent_id}: {e}")

        # Synthesize proposals
        synthesized = self._synthesize_proposals(proposals)

        return {
            "content": synthesized,
            "source": "collaborative",
            "contributors": [p["agent"] for p in proposals],
            "version": 1
        }

    def _synthesize_proposals(self, proposals: List[Dict[str, Any]]) -> str:
        """Synthesize multiple proposals into one."""
        if not proposals:
            return "No proposals generated"

        # Collect key elements from all proposals
        elements = []
        for proposal in proposals:
            content = proposal.get("proposal", {})
            if isinstance(content, dict):
                if "key_points" in content:
                    elements.extend(content["key_points"])
                elif "content" in content:
                    elements.append(content["content"])
            else:
                elements.append(str(content))

        # Create synthesized proposal
        return f"Synthesized proposal incorporating {len(elements)} key elements"

    async def _discuss_proposal(self, proposal: Dict[str, Any],
                               iteration: int) -> List[Dict[str, Any]]:
        """
        Agents discuss the proposal.

        Args:
            proposal: Proposal to discuss
            iteration: Current iteration

        Returns:
            List of discussion points from agents
        """
        self.logger.info("Agents discussing proposal")

        discussion = []

        for agent in self.agents:
            try:
                response = await agent.send_task_request(
                    recipient=agent.agent_id,
                    task_data={
                        "action": "discuss",
                        "proposal": proposal,
                        "iteration": iteration
                    },
                    wait_for_response=True,
                    timeout=20.0
                )

                if response:
                    discussion.append({
                        "agent": agent.agent_id,
                        "feedback": response.content
                    })

            except Exception as e:
                self.logger.error(f"Failed to get discussion from {agent.agent_id}: {e}")

        return discussion

    async def _collect_votes(self, proposal: Dict[str, Any],
                            iteration: int) -> Dict[str, Dict[str, Any]]:
        """
        Collect votes from all agents.

        Args:
            proposal: Proposal to vote on
            iteration: Current iteration

        Returns:
            Dictionary mapping agent_id to vote
        """
        self.logger.info("Collecting votes")

        votes = {}

        for agent in self.agents:
            try:
                response = await agent.send_task_request(
                    recipient=agent.agent_id,
                    task_data={
                        "action": "vote",
                        "proposal": proposal,
                        "iteration": iteration
                    },
                    wait_for_response=True,
                    timeout=15.0
                )

                if response:
                    vote_content = response.content
                    votes[agent.agent_id] = {
                        "vote": vote_content.get("vote", VoteType.ABSTAIN.value),
                        "confidence": vote_content.get("confidence", 0.5),
                        "reasoning": vote_content.get("reasoning", ""),
                        "conditions": vote_content.get("conditions", [])
                    }

            except Exception as e:
                self.logger.error(f"Failed to get vote from {agent.agent_id}: {e}")
                votes[agent.agent_id] = {
                    "vote": VoteType.ABSTAIN.value,
                    "confidence": 0.0,
                    "reasoning": "Failed to vote"
                }

        # Store votes
        self.votes_history.append({
            "iteration": iteration,
            "votes": votes
        })

        await self.state.set(f"iteration_{iteration}_votes", votes)

        return votes

    def _check_consensus(self, votes: Dict[str, Dict[str, Any]]) -> tuple[bool, float]:
        """
        Check if consensus has been reached.

        Args:
            votes: Votes from all agents

        Returns:
            Tuple of (consensus_reached, agreement_level)
        """
        if not votes:
            return False, 0.0

        # Count votes
        approve_count = sum(
            1 for v in votes.values()
            if v.get("vote") == VoteType.APPROVE.value
        )

        total_votes = len([v for v in votes.values() if v.get("vote") != VoteType.ABSTAIN.value])

        if total_votes == 0:
            return False, 0.0

        agreement_level = approve_count / total_votes

        # Check strategy
        if self.strategy == ConsensusStrategy.UNANIMOUS:
            consensus = approve_count == len(votes)
        elif self.strategy == ConsensusStrategy.MAJORITY:
            consensus = agreement_level > 0.5
        elif self.strategy == ConsensusStrategy.SUPERMAJORITY:
            consensus = agreement_level >= 0.67
        elif self.strategy == ConsensusStrategy.WEIGHTED:
            # Weighted by confidence
            weighted_approve = sum(
                v.get("confidence", 0.5)
                for v in votes.values()
                if v.get("vote") == VoteType.APPROVE.value
            )
            total_weight = sum(v.get("confidence", 0.5) for v in votes.values())
            consensus = (weighted_approve / total_weight) > 0.5 if total_weight > 0 else False
        else:
            consensus = False

        return consensus, agreement_level

    async def _refine_proposal(self, proposal: Dict[str, Any],
                              discussion: List[Dict[str, Any]],
                              votes: Dict[str, Dict[str, Any]],
                              iteration: int) -> Dict[str, Any]:
        """
        Refine proposal based on feedback.

        Args:
            proposal: Current proposal
            discussion: Discussion feedback
            votes: Voting results
            iteration: Current iteration

        Returns:
            Refined proposal
        """
        self.logger.info("Refining proposal based on feedback")

        # Collect concerns and suggestions
        concerns = []
        suggestions = []

        for agent_id, vote_data in votes.items():
            if vote_data.get("vote") in [VoteType.REJECT.value, VoteType.CONDITIONAL.value]:
                reasoning = vote_data.get("reasoning", "")
                if reasoning:
                    concerns.append(reasoning)

                conditions = vote_data.get("conditions", [])
                suggestions.extend(conditions)

        # Find agent with highest confidence to lead refinement
        refiner = max(
            self.agents,
            key=lambda a: votes.get(a.agent_id, {}).get("confidence", 0)
        )

        # Ask refiner to propose improvements
        try:
            response = await refiner.send_task_request(
                recipient=refiner.agent_id,
                task_data={
                    "action": "refine_proposal",
                    "proposal": proposal,
                    "concerns": concerns,
                    "suggestions": suggestions,
                    "iteration": iteration
                },
                wait_for_response=True,
                timeout=30.0
            )

            if response:
                refined = response.content
                refined_proposal = {
                    "content": refined.get("refined_proposal", proposal["content"]),
                    "source": "refined",
                    "version": proposal.get("version", 1) + 1,
                    "refinements": refined.get("changes", []),
                    "addressed_concerns": concerns
                }

                self.proposals.append(refined_proposal)
                return refined_proposal

        except Exception as e:
            self.logger.error(f"Refinement failed: {e}")

        # If refinement fails, return original with note
        proposal["refinement_attempted"] = True
        return proposal

    async def _finalize_consensus(self, final_proposal: Dict[str, Any],
                                 consensus_reached: bool,
                                 iterations: int) -> Dict[str, Any]:
        """
        Finalize consensus result.

        Args:
            final_proposal: Final proposal
            consensus_reached: Whether consensus was reached
            iterations: Number of iterations taken

        Returns:
            Final consensus result
        """
        # Get final votes
        final_votes = self.votes_history[-1] if self.votes_history else {}

        # Calculate agreement metrics
        votes_data = final_votes.get("votes", {})
        approve_count = sum(
            1 for v in votes_data.values()
            if v.get("vote") == VoteType.APPROVE.value
        )
        reject_count = sum(
            1 for v in votes_data.values()
            if v.get("vote") == VoteType.REJECT.value
        )

        result = {
            "consensus_reached": consensus_reached,
            "final_proposal": final_proposal,
            "iterations": iterations,
            "strategy": self.strategy.value,
            "total_agents": len(self.agents),
            "approve_count": approve_count,
            "reject_count": reject_count,
            "agreement_level": approve_count / len(votes_data) if votes_data else 0,
            "all_votes": votes_data,
            "proposal_history": self.proposals,
            "votes_history": self.votes_history,
            "status": "success" if consensus_reached else "no_consensus"
        }

        # Store final result
        await self.state.set("consensus_result", result)

        return result


class ConsensusAgent(BaseCollaborativeAgent):
    """
    Agent specialized for consensus-based collaboration.

    Participates in proposal generation, discussion, voting, and refinement.
    """

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute consensus action.

        Args:
            input_data: Action and context

        Returns:
            Response based on action
        """
        action = input_data.get("action", "")

        if action == "propose":
            return await self._make_proposal(input_data)
        elif action == "discuss":
            return await self._discuss_proposal(input_data)
        elif action == "vote":
            return await self._cast_vote(input_data)
        elif action == "refine_proposal":
            return await self._refine_proposal(input_data)
        else:
            return {"status": "unknown_action", "action": action}

    async def _make_proposal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a proposal."""
        context = data.get("context", {})

        # Generate proposal based on context
        proposal = {
            "content": f"Proposal from {self.agent_id}",
            "key_points": [
                f"Point 1 from {self.agent_id}",
                f"Point 2 from {self.agent_id}"
            ],
            "rationale": "Based on analysis of context",
            "confidence": 0.8
        }

        return proposal

    async def _discuss_proposal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Discuss a proposal."""
        proposal = data.get("proposal", {})

        # Analyze proposal and provide feedback
        feedback = {
            "strengths": [
                "Clear and actionable",
                "Addresses key concerns"
            ],
            "concerns": [
                "May need more detail on implementation"
            ],
            "questions": [
                "What is the timeline?",
                "What are the resource requirements?"
            ],
            "overall_sentiment": "positive"
        }

        return feedback

    async def _cast_vote(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Cast vote on proposal."""
        proposal = data.get("proposal", {})
        iteration = data.get("iteration", 1)

        # Evaluate proposal (simplified - would use actual analysis)
        proposal_content = proposal.get("content", "")

        # Decide vote based on analysis
        vote = VoteType.APPROVE.value
        confidence = 0.8
        reasoning = f"Proposal meets requirements assessed by {self.agent_id}"

        # In later iterations, might have conditions
        conditions = []
        if iteration > 2:
            conditions = ["Add more detail on timeline"]

        vote_response = {
            "vote": vote,
            "confidence": confidence,
            "reasoning": reasoning,
            "conditions": conditions
        }

        return vote_response

    async def _refine_proposal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Refine proposal based on feedback."""
        proposal = data.get("proposal", {})
        concerns = data.get("concerns", [])
        suggestions = data.get("suggestions", [])

        # Incorporate feedback into refined proposal
        original_content = proposal.get("content", "")

        refined = {
            "refined_proposal": f"{original_content} (refined)",
            "changes": [
                f"Addressed concern: {c[:50]}..." for c in concerns[:3]
            ],
            "incorporated_suggestions": suggestions[:3],
            "confidence": 0.85
        }

        return refined
