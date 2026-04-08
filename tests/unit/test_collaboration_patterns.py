"""
Comprehensive unit tests for agent collaboration patterns.

Tests ConsensusCollaboration, DebateCollaboration, HierarchicalCollaboration,
and PeerToPeerCollaboration along with their associated agent classes.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from contextlib import contextmanager

from ia_modules.agents.core import AgentRole
from ia_modules.agents.state import StateManager
from ia_modules.agents.communication import MessageBus, MessageType, AgentMessage
from ia_modules.agents.base_agent import BaseCollaborativeAgent

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


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_role(name: str, description: str = "test agent") -> AgentRole:
    return AgentRole(name=name, description=description)


@contextmanager
def _noop_telemetry():
    """Context manager that does nothing, used as mock for trace_collaboration."""
    yield


def _mock_telemetry():
    """Return a mock telemetry object whose trace_collaboration is a no-op CM."""
    tel = MagicMock()
    tel.trace_collaboration = MagicMock(side_effect=lambda **kw: _noop_telemetry())
    return tel


@pytest.fixture
def state():
    return StateManager(thread_id="test-collab")


@pytest.fixture
def bus():
    return MessageBus()


@pytest.fixture
def telemetry_patch():
    """Patch get_agent_telemetry globally so no real OTEL is needed."""
    tel = _mock_telemetry()
    with patch(
        "ia_modules.agents.collaboration_patterns.consensus.get_agent_telemetry",
        return_value=tel,
    ), patch(
        "ia_modules.agents.collaboration_patterns.debate.get_agent_telemetry",
        return_value=tel,
    ), patch(
        "ia_modules.agents.collaboration_patterns.hierarchical.get_agent_telemetry",
        return_value=tel,
    ), patch(
        "ia_modules.agents.collaboration_patterns.peer_to_peer.get_agent_telemetry",
        return_value=tel,
    ), patch(
        "ia_modules.agents.base_agent.get_agent_telemetry",
        return_value=tel,
    ), patch(
        "ia_modules.agents.core.get_agent_telemetry",
        return_value=tel,
    ):
        yield tel


# ---------------------------------------------------------------------------
# Consensus enums
# ---------------------------------------------------------------------------

class TestConsensusEnums:
    def test_consensus_strategy_values(self):
        assert ConsensusStrategy.UNANIMOUS.value == "unanimous"
        assert ConsensusStrategy.MAJORITY.value == "majority"
        assert ConsensusStrategy.SUPERMAJORITY.value == "supermajority"
        assert ConsensusStrategy.WEIGHTED.value == "weighted"

    def test_vote_type_values(self):
        assert VoteType.APPROVE.value == "approve"
        assert VoteType.REJECT.value == "reject"
        assert VoteType.ABSTAIN.value == "abstain"
        assert VoteType.CONDITIONAL.value == "conditional"


# ---------------------------------------------------------------------------
# ConsensusAgent
# ---------------------------------------------------------------------------

class TestConsensusAgent:
    async def test_execute_propose(self, state, bus, telemetry_patch):
        agent = ConsensusAgent(
            role=_make_role("ca1"), state_manager=state, message_bus=bus
        )
        result = await agent.execute({"action": "propose", "context": {}})
        assert "content" in result
        assert "key_points" in result

    async def test_execute_discuss(self, state, bus, telemetry_patch):
        agent = ConsensusAgent(
            role=_make_role("ca2"), state_manager=state, message_bus=bus
        )
        result = await agent.execute(
            {"action": "discuss", "proposal": {"content": "test"}, "iteration": 1}
        )
        assert "strengths" in result
        assert "concerns" in result

    async def test_execute_vote(self, state, bus, telemetry_patch):
        agent = ConsensusAgent(
            role=_make_role("ca3"), state_manager=state, message_bus=bus
        )
        result = await agent.execute(
            {"action": "vote", "proposal": {"content": "test"}, "iteration": 1}
        )
        assert result["vote"] == VoteType.APPROVE.value
        assert "confidence" in result

    async def test_execute_vote_later_iteration_has_conditions(self, state, bus, telemetry_patch):
        agent = ConsensusAgent(
            role=_make_role("ca3b"), state_manager=state, message_bus=bus
        )
        result = await agent.execute(
            {"action": "vote", "proposal": {"content": "test"}, "iteration": 3}
        )
        assert len(result["conditions"]) > 0

    async def test_execute_refine_proposal(self, state, bus, telemetry_patch):
        agent = ConsensusAgent(
            role=_make_role("ca4"), state_manager=state, message_bus=bus
        )
        result = await agent.execute(
            {
                "action": "refine_proposal",
                "proposal": {"content": "original"},
                "concerns": ["concern1"],
                "suggestions": ["suggestion1"],
            }
        )
        assert "refined_proposal" in result
        assert "original" in result["refined_proposal"]

    async def test_execute_unknown_action(self, state, bus, telemetry_patch):
        agent = ConsensusAgent(
            role=_make_role("ca5"), state_manager=state, message_bus=bus
        )
        result = await agent.execute({"action": "unknown_xyz"})
        assert result["status"] == "unknown_action"
        assert result["action"] == "unknown_xyz"


# ---------------------------------------------------------------------------
# ConsensusCollaboration
# ---------------------------------------------------------------------------

class TestConsensusCollaboration:
    def _make_collab(self, state, bus, strategy=ConsensusStrategy.MAJORITY,
                     num_agents=3, max_iterations=3):
        agents = [
            ConsensusAgent(
                role=_make_role(f"cons_{i}"), state_manager=state, message_bus=bus
            )
            for i in range(num_agents)
        ]
        return ConsensusCollaboration(
            agents=agents,
            message_bus=bus,
            state_manager=state,
            strategy=strategy,
            max_iterations=max_iterations,
        )

    async def test_initialize_and_shutdown(self, state, bus, telemetry_patch):
        collab = self._make_collab(state, bus)
        await collab.initialize()
        # Agents should be subscribed
        assert len(bus.get_active_agents()) == 3
        await collab.shutdown()

    def _mock_send_task(self, collab):
        """Mock send_task_request on all agents to return proper responses."""
        def _make_mock_send(agent):
            async def _send(recipient, task_data, **kwargs):
                # Run the agent's own execute to get a real response
                result = await agent.execute(task_data)
                msg = MagicMock()
                msg.content = result
                return msg
            return _send

        for agent in collab.agents:
            agent.send_task_request = _make_mock_send(agent)

    async def test_execute_with_provided_proposal_reaches_consensus(
        self, state, bus, telemetry_patch
    ):
        collab = self._make_collab(state, bus, strategy=ConsensusStrategy.MAJORITY)
        await collab.initialize()
        self._mock_send_task(collab)

        result = await collab.execute({
            "proposal": "Use microservices architecture",
            "context": {"urgency": "high"},
        })

        assert result["consensus_reached"] is True
        assert result["strategy"] == "majority"
        assert result["total_agents"] == 3
        assert result["status"] == "success"
        assert result["iterations"] >= 1
        await collab.shutdown()

    async def test_execute_without_proposal_generates_collaborative(
        self, state, bus, telemetry_patch
    ):
        collab = self._make_collab(state, bus)
        await collab.initialize()
        self._mock_send_task(collab)

        result = await collab.execute({"context": {"topic": "architecture"}})

        assert result["consensus_reached"] is True
        # The initial proposal should have been generated collaboratively
        assert len(collab.proposals) >= 1
        await collab.shutdown()

    async def test_unanimous_strategy_with_all_approve(self, state, bus, telemetry_patch):
        collab = self._make_collab(state, bus, strategy=ConsensusStrategy.UNANIMOUS)
        await collab.initialize()
        self._mock_send_task(collab)

        result = await collab.execute({"proposal": "Unanimous test"})
        # ConsensusAgent always votes APPROVE, so unanimous should pass
        assert result["consensus_reached"] is True
        await collab.shutdown()

    async def test_supermajority_strategy(self, state, bus, telemetry_patch):
        collab = self._make_collab(
            state, bus, strategy=ConsensusStrategy.SUPERMAJORITY, num_agents=3
        )
        await collab.initialize()
        self._mock_send_task(collab)

        result = await collab.execute({"proposal": "Supermajority test"})
        assert result["consensus_reached"] is True
        await collab.shutdown()

    async def test_weighted_strategy(self, state, bus, telemetry_patch):
        collab = self._make_collab(
            state, bus, strategy=ConsensusStrategy.WEIGHTED, num_agents=3
        )
        await collab.initialize()
        self._mock_send_task(collab)

        result = await collab.execute({"proposal": "Weighted test"})
        assert result["consensus_reached"] is True
        await collab.shutdown()

    # --- _check_consensus unit tests ---

    def test_check_consensus_empty_votes(self, state, bus, telemetry_patch):
        collab = self._make_collab(state, bus)
        reached, level = collab._check_consensus({})
        assert reached is False
        assert level == 0.0

    def test_check_consensus_all_abstain(self, state, bus, telemetry_patch):
        collab = self._make_collab(state, bus)
        votes = {
            "a1": {"vote": VoteType.ABSTAIN.value, "confidence": 0.5},
            "a2": {"vote": VoteType.ABSTAIN.value, "confidence": 0.5},
        }
        reached, level = collab._check_consensus(votes)
        assert reached is False
        assert level == 0.0

    def test_check_consensus_majority_pass(self, state, bus, telemetry_patch):
        collab = self._make_collab(state, bus, strategy=ConsensusStrategy.MAJORITY)
        votes = {
            "a1": {"vote": VoteType.APPROVE.value, "confidence": 0.8},
            "a2": {"vote": VoteType.APPROVE.value, "confidence": 0.7},
            "a3": {"vote": VoteType.REJECT.value, "confidence": 0.6},
        }
        reached, level = collab._check_consensus(votes)
        assert reached is True
        assert level == pytest.approx(2 / 3, abs=0.01)

    def test_check_consensus_majority_fail(self, state, bus, telemetry_patch):
        collab = self._make_collab(state, bus, strategy=ConsensusStrategy.MAJORITY)
        votes = {
            "a1": {"vote": VoteType.APPROVE.value, "confidence": 0.8},
            "a2": {"vote": VoteType.REJECT.value, "confidence": 0.7},
            "a3": {"vote": VoteType.REJECT.value, "confidence": 0.6},
        }
        reached, level = collab._check_consensus(votes)
        assert reached is False

    def test_check_consensus_unanimous_fail_with_one_reject(self, state, bus, telemetry_patch):
        collab = self._make_collab(state, bus, strategy=ConsensusStrategy.UNANIMOUS)
        votes = {
            "a1": {"vote": VoteType.APPROVE.value, "confidence": 0.9},
            "a2": {"vote": VoteType.APPROVE.value, "confidence": 0.9},
            "a3": {"vote": VoteType.REJECT.value, "confidence": 0.5},
        }
        reached, _ = collab._check_consensus(votes)
        assert reached is False

    def test_check_consensus_supermajority_boundary(self, state, bus, telemetry_patch):
        collab = self._make_collab(state, bus, strategy=ConsensusStrategy.SUPERMAJORITY)
        # 2/3 = 0.666... which is < 0.67 threshold
        votes = {
            "a1": {"vote": VoteType.APPROVE.value, "confidence": 0.8},
            "a2": {"vote": VoteType.APPROVE.value, "confidence": 0.8},
            "a3": {"vote": VoteType.REJECT.value, "confidence": 0.5},
        }
        reached, level = collab._check_consensus(votes)
        # 2/3 ~= 0.6667 < 0.67 so this should fail
        assert reached is False

    def test_check_consensus_weighted_pass(self, state, bus, telemetry_patch):
        collab = self._make_collab(state, bus, strategy=ConsensusStrategy.WEIGHTED)
        votes = {
            "a1": {"vote": VoteType.APPROVE.value, "confidence": 0.9},
            "a2": {"vote": VoteType.REJECT.value, "confidence": 0.1},
        }
        reached, _ = collab._check_consensus(votes)
        assert reached is True

    def test_check_consensus_weighted_fail(self, state, bus, telemetry_patch):
        collab = self._make_collab(state, bus, strategy=ConsensusStrategy.WEIGHTED)
        votes = {
            "a1": {"vote": VoteType.APPROVE.value, "confidence": 0.1},
            "a2": {"vote": VoteType.REJECT.value, "confidence": 0.9},
        }
        reached, _ = collab._check_consensus(votes)
        assert reached is False

    def test_check_consensus_weighted_zero_total_weight(self, state, bus, telemetry_patch):
        collab = self._make_collab(state, bus, strategy=ConsensusStrategy.WEIGHTED)
        votes = {
            "a1": {"vote": VoteType.REJECT.value, "confidence": 0.0},
            "a2": {"vote": VoteType.REJECT.value, "confidence": 0.0},
        }
        reached, _ = collab._check_consensus(votes)
        assert reached is False

    # --- _synthesize_proposals ---

    def test_synthesize_proposals_empty(self, state, bus, telemetry_patch):
        collab = self._make_collab(state, bus)
        result = collab._synthesize_proposals([])
        assert result == "No proposals generated"

    def test_synthesize_proposals_with_key_points(self, state, bus, telemetry_patch):
        collab = self._make_collab(state, bus)
        proposals = [
            {"agent": "a1", "proposal": {"key_points": ["point1", "point2"]}},
            {"agent": "a2", "proposal": {"content": "some content"}},
        ]
        result = collab._synthesize_proposals(proposals)
        assert "3 key elements" in result

    def test_synthesize_proposals_with_string_content(self, state, bus, telemetry_patch):
        collab = self._make_collab(state, bus)
        proposals = [
            {"agent": "a1", "proposal": "plain string proposal"},
        ]
        result = collab._synthesize_proposals(proposals)
        assert "1 key elements" in result

    # --- _finalize_consensus ---

    async def test_finalize_consensus_with_empty_history(self, state, bus, telemetry_patch):
        collab = self._make_collab(state, bus)
        collab.votes_history = []
        result = await collab._finalize_consensus(
            {"content": "final"}, True, 1
        )
        assert result["consensus_reached"] is True
        assert result["approve_count"] == 0

    async def test_finalize_consensus_stores_result_in_state(self, state, bus, telemetry_patch):
        collab = self._make_collab(state, bus)
        collab.votes_history = [
            {
                "iteration": 1,
                "votes": {
                    "a1": {"vote": VoteType.APPROVE.value},
                    "a2": {"vote": VoteType.REJECT.value},
                },
            }
        ]
        result = await collab._finalize_consensus(
            {"content": "final"}, False, 2
        )
        assert result["consensus_reached"] is False
        assert result["approve_count"] == 1
        assert result["reject_count"] == 1

        stored = await state.get("consensus_result")
        assert stored is not None

    # --- refine_proposal error path ---

    async def test_refine_proposal_when_send_task_fails(self, state, bus, telemetry_patch):
        collab = self._make_collab(state, bus)
        await collab.initialize()

        # Monkey-patch the refiner's send_task_request to raise
        for agent in collab.agents:
            agent.send_task_request = AsyncMock(side_effect=Exception("boom"))

        votes = {
            collab.agents[0].agent_id: {
                "vote": VoteType.REJECT.value,
                "confidence": 0.9,
                "reasoning": "Bad proposal",
                "conditions": ["fix it"],
            },
        }
        proposal = {"content": "original", "version": 1}
        result = await collab._refine_proposal(proposal, [], votes, 1)
        # Should return the original proposal with refinement_attempted flag
        assert result.get("refinement_attempted") is True
        await collab.shutdown()

    # --- collect_votes error path ---

    async def test_collect_votes_agent_failure(self, state, bus, telemetry_patch):
        collab = self._make_collab(state, bus, num_agents=2)
        await collab.initialize()

        # Make one agent fail
        collab.agents[0].send_task_request = AsyncMock(side_effect=Exception("vote fail"))

        proposal = {"content": "test", "version": 1}
        votes = await collab._collect_votes(proposal, 1)

        # Failed agent should get abstain vote
        failed_id = collab.agents[0].agent_id
        assert votes[failed_id]["vote"] == VoteType.ABSTAIN.value
        assert votes[failed_id]["confidence"] == 0.0
        await collab.shutdown()


# ---------------------------------------------------------------------------
# DebateRole enum
# ---------------------------------------------------------------------------

class TestDebateRole:
    def test_debate_role_values(self):
        assert DebateRole.PROPONENT.value == "proponent"
        assert DebateRole.OPPONENT.value == "opponent"
        assert DebateRole.MODERATOR.value == "moderator"
        assert DebateRole.JUDGE.value == "judge"


# ---------------------------------------------------------------------------
# DebateAgent
# ---------------------------------------------------------------------------

class TestDebateAgent:
    async def test_opening_statement_proponent(self, state, bus, telemetry_patch):
        agent = DebateAgent(
            role=_make_role("pro1"),
            state_manager=state,
            message_bus=bus,
            debate_role=DebateRole.PROPONENT,
        )
        result = await agent.execute(
            {"action": "opening_statement", "topic": "test topic"}
        )
        assert result["position"] == "for"
        assert "test topic" in result["statement"]

    async def test_opening_statement_opponent(self, state, bus, telemetry_patch):
        agent = DebateAgent(
            role=_make_role("opp1"),
            state_manager=state,
            message_bus=bus,
            debate_role=DebateRole.OPPONENT,
        )
        result = await agent.execute(
            {"action": "opening_statement", "topic": "test topic"}
        )
        assert result["position"] == "against"

    async def test_argue(self, state, bus, telemetry_patch):
        agent = DebateAgent(
            role=_make_role("pro2"),
            state_manager=state,
            message_bus=bus,
            debate_role=DebateRole.PROPONENT,
        )
        context = [{"side": "opponent", "argument": "counter point"}]
        result = await agent.execute(
            {"action": "argue", "topic": "AI", "round": 2, "context": context}
        )
        assert result["position"] == "proponent"
        assert result["round"] == 2
        assert len(result["counter_arguments"]) == 1  # min(2, len(opponent_points))
        assert len(result["key_points"]) == 3

    async def test_argue_no_context(self, state, bus, telemetry_patch):
        agent = DebateAgent(
            role=_make_role("pro3"),
            state_manager=state,
            message_bus=bus,
            debate_role=DebateRole.PROPONENT,
        )
        result = await agent.execute(
            {"action": "argue", "topic": "AI", "round": 1}
        )
        assert result["counter_arguments"] == []

    async def test_closing_statement(self, state, bus, telemetry_patch):
        agent = DebateAgent(
            role=_make_role("pro4"),
            state_manager=state,
            message_bus=bus,
            debate_role=DebateRole.PROPONENT,
        )
        result = await agent.execute(
            {"action": "closing_statement", "topic": "topic"}
        )
        assert result["position"] == "proponent"
        assert "strongest_arguments" in result

    async def test_unknown_action(self, state, bus, telemetry_patch):
        agent = DebateAgent(
            role=_make_role("pro5"),
            state_manager=state,
            message_bus=bus,
            debate_role=DebateRole.PROPONENT,
        )
        result = await agent.execute({"action": "dance"})
        assert result["status"] == "unknown_action"


# ---------------------------------------------------------------------------
# ModeratorAgent
# ---------------------------------------------------------------------------

class TestModeratorAgent:
    async def test_moderate(self, state, bus, telemetry_patch):
        agent = ModeratorAgent(
            role=_make_role("mod1"), state_manager=state, message_bus=bus
        )
        result = await agent.execute({"action": "moderate"})
        assert result["status"] == "round_moderated"

    async def test_unknown_action(self, state, bus, telemetry_patch):
        agent = ModeratorAgent(
            role=_make_role("mod2"), state_manager=state, message_bus=bus
        )
        result = await agent.execute({"action": "sing"})
        assert result["status"] == "unknown_action"


# ---------------------------------------------------------------------------
# DebateCollaboration
# ---------------------------------------------------------------------------

class TestDebateCollaboration:
    def _make_debate(self, state, bus, num_proponents=1, num_opponents=1,
                     with_judge=False):
        proponents = [
            DebateAgent(
                role=_make_role(f"pro_{i}"),
                state_manager=state,
                message_bus=bus,
                debate_role=DebateRole.PROPONENT,
            )
            for i in range(num_proponents)
        ]
        opponents = [
            DebateAgent(
                role=_make_role(f"opp_{i}"),
                state_manager=state,
                message_bus=bus,
                debate_role=DebateRole.OPPONENT,
            )
            for i in range(num_opponents)
        ]
        moderator = ModeratorAgent(
            role=_make_role("moderator"), state_manager=state, message_bus=bus
        )
        judge = None
        if with_judge:
            judge = DebateAgent(
                role=_make_role("judge"),
                state_manager=state,
                message_bus=bus,
                debate_role=DebateRole.JUDGE,
            )
        return DebateCollaboration(
            proponents=proponents,
            opponents=opponents,
            moderator=moderator,
            message_bus=bus,
            state_manager=state,
            judge=judge,
        )

    async def test_initialize_and_shutdown_no_judge(self, state, bus, telemetry_patch):
        debate = self._make_debate(state, bus)
        await debate.initialize()
        # moderator + 1 proponent + 1 opponent = 3
        assert len(bus.get_active_agents()) == 3
        await debate.shutdown()

    async def test_initialize_and_shutdown_with_judge(self, state, bus, telemetry_patch):
        debate = self._make_debate(state, bus, with_judge=True)
        await debate.initialize()
        # moderator + 1 proponent + 1 opponent + 1 judge = 4
        assert len(bus.get_active_agents()) == 4
        await debate.shutdown()

    def _mock_debate_agents(self, debate):
        """Mock send_task_request on all debate agents to return proper responses."""
        def _make_mock_send(agent):
            async def _send(recipient, task_data, **kwargs):
                result = await agent.execute(task_data)
                msg = MagicMock()
                msg.content = result
                return msg
            return _send

        for agent in debate.proponents + debate.opponents + [debate.moderator]:
            agent.send_task_request = _make_mock_send(agent)
        if debate.judge:
            debate.judge.send_task_request = _make_mock_send(debate.judge)

    async def test_execute_no_judge(self, state, bus, telemetry_patch):
        debate = self._make_debate(state, bus)
        await debate.initialize()
        self._mock_debate_agents(debate)

        result = await debate.execute({"topic": "Open source AI", "rounds": 2})

        assert result["topic"] == "Open source AI"
        assert result["total_rounds"] == 2
        assert result["status"] == "completed"
        assert result["evaluation"] is None
        assert len(result["debate_rounds"]) == 2
        await debate.shutdown()

    async def test_execute_with_judge(self, state, bus, telemetry_patch):
        debate = self._make_debate(state, bus, with_judge=True)
        await debate.initialize()
        self._mock_debate_agents(debate)

        result = await debate.execute({"topic": "Regulation", "rounds": 1})

        assert result["evaluation"] is not None
        assert result["status"] == "completed"
        await debate.shutdown()

    async def test_execute_multiple_proponents_opponents(self, state, bus, telemetry_patch):
        debate = self._make_debate(
            state, bus, num_proponents=2, num_opponents=2
        )
        await debate.initialize()
        self._mock_debate_agents(debate)

        result = await debate.execute({"topic": "Testing", "rounds": 1})

        assert result["total_arguments"] >= 4  # 2 proponents + 2 opponents
        await debate.shutdown()

    # --- _extract_key_points ---

    def test_extract_key_points_with_dict_key_points(self, state, bus, telemetry_patch):
        debate = self._make_debate(state, bus)
        rounds = [
            {
                "proponent_arguments": [
                    {"argument": {"key_points": ["p1", "p2"]}},
                    {"argument": {"summary": "summary1"}},
                ],
                "opponent_arguments": [],
            }
        ]
        points = debate._extract_key_points(rounds, "proponent")
        assert "p1" in points
        assert "p2" in points
        assert "summary1" in points

    def test_extract_key_points_empty(self, state, bus, telemetry_patch):
        debate = self._make_debate(state, bus)
        points = debate._extract_key_points([], "proponent")
        assert points == []

    def test_extract_key_points_string_argument(self, state, bus, telemetry_patch):
        debate = self._make_debate(state, bus)
        rounds = [
            {
                "opponent_arguments": [
                    {"argument": "plain string argument"},
                ],
                "proponent_arguments": [],
            }
        ]
        points = debate._extract_key_points(rounds, "opponent")
        assert points == []  # string arguments don't get extracted

    def test_extract_key_points_max_10(self, state, bus, telemetry_patch):
        debate = self._make_debate(state, bus)
        rounds = [
            {
                "proponent_arguments": [
                    {"argument": {"key_points": [f"p{i}" for i in range(15)]}},
                ],
                "opponent_arguments": [],
            }
        ]
        points = debate._extract_key_points(rounds, "proponent")
        assert len(points) == 10

    # --- error handling in opening/closing/evaluation ---

    async def test_opening_statements_agent_failure(self, state, bus, telemetry_patch):
        debate = self._make_debate(state, bus)
        await debate.initialize()
        self._mock_debate_agents(debate)

        # Now override proponent to fail
        debate.proponents[0].send_task_request = AsyncMock(
            side_effect=Exception("fail")
        )
        statements = await debate._opening_statements("topic")
        assert statements["proponents"] == []
        assert len(statements["opponents"]) == 1
        await debate.shutdown()

    async def test_closing_statements_agent_failure(self, state, bus, telemetry_patch):
        debate = self._make_debate(state, bus)
        await debate.initialize()

        debate.opponents[0].send_task_request = AsyncMock(
            side_effect=Exception("fail")
        )
        statements = await debate._closing_statements("topic")
        assert statements["opponents"] == []
        await debate.shutdown()

    async def test_evaluate_debate_no_judge(self, state, bus, telemetry_patch):
        debate = self._make_debate(state, bus, with_judge=False)
        result = await debate._evaluate_debate({}, [], {})
        assert result == {}

    async def test_evaluate_debate_judge_failure(self, state, bus, telemetry_patch):
        debate = self._make_debate(state, bus, with_judge=True)
        await debate.initialize()

        debate.judge.send_task_request = AsyncMock(side_effect=Exception("judge fail"))
        result = await debate._evaluate_debate({}, [], {})
        assert result["evaluation"] == "evaluation_failed"
        await debate.shutdown()

    async def test_collect_arguments_agent_failure(self, state, bus, telemetry_patch):
        debate = self._make_debate(state, bus)
        await debate.initialize()

        debate.proponents[0].send_task_request = AsyncMock(
            side_effect=Exception("arg fail")
        )
        args = await debate._collect_arguments(
            debate.proponents, "proponent", 1, "topic", []
        )
        assert args == []
        await debate.shutdown()


# ---------------------------------------------------------------------------
# LeaderAgent
# ---------------------------------------------------------------------------

class TestLeaderAgent:
    async def test_execute(self, state, bus, telemetry_patch):
        leader = LeaderAgent(
            role=_make_role("leader"), state_manager=state, message_bus=bus
        )
        await leader.initialize()

        result = await leader.execute({"task": "coordinate stuff"})
        assert result["status"] == "coordinating"
        assert result["task"] == "coordinate stuff"

        leader_status = await state.get("leader_status")
        assert leader_status == "coordinating"
        await leader.shutdown()


# ---------------------------------------------------------------------------
# WorkerAgent
# ---------------------------------------------------------------------------

class TestWorkerAgent:
    async def test_execute(self, state, bus, telemetry_patch):
        worker = WorkerAgent(
            role=_make_role("worker1"), state_manager=state, message_bus=bus
        )
        await worker.initialize()

        result = await worker.execute(
            {"task_id": "t1", "description": "do something"}
        )
        assert result["status"] == "success"
        assert result["task_id"] == "t1"

        worker_status = await state.get("worker_worker1_status")
        assert worker_status == "completed"
        await worker.shutdown()

    async def test_collaborate_with_peer(self, state, bus, telemetry_patch):
        worker1 = WorkerAgent(
            role=_make_role("w1"), state_manager=state, message_bus=bus
        )
        worker2 = WorkerAgent(
            role=_make_role("w2"), state_manager=state, message_bus=bus
        )
        await worker1.initialize()
        await worker2.initialize()

        # collaborate_with_peer calls send_query which waits for a reply
        # We need to mock it to avoid timeout
        worker1.send_query = AsyncMock(
            return_value=AgentMessage(
                sender="w2",
                message_type=MessageType.RESPONSE,
                content={"status": "helped"},
            )
        )
        result = await worker1.collaborate_with_peer(
            "w2", "help", {"data": "some data"}
        )
        assert result["status"] == "helped"
        await worker1.shutdown()
        await worker2.shutdown()


# ---------------------------------------------------------------------------
# HierarchicalCollaboration
# ---------------------------------------------------------------------------

class TestHierarchicalCollaboration:
    def _make_hierarchy(self, state, bus, num_workers=2):
        leader = LeaderAgent(
            role=_make_role("leader"), state_manager=state, message_bus=bus
        )
        workers = [
            WorkerAgent(
                role=_make_role(f"worker_{i}"),
                state_manager=state,
                message_bus=bus,
            )
            for i in range(num_workers)
        ]
        return HierarchicalCollaboration(leader, workers, bus)

    async def test_initialize_and_shutdown(self, state, bus, telemetry_patch):
        hier = self._make_hierarchy(state, bus)
        await hier.initialize()
        # leader + 2 workers
        assert len(bus.get_active_agents()) == 3
        assert len(hier.available_workers) == 2
        await hier.shutdown()

    def _mock_hier_agents(self, hier):
        """Mock send_task_request on leader to simulate worker responses."""
        async def _mock_send(recipient, task_data, **kwargs):
            # Find the worker and run its execute
            for w in hier.workers:
                if w.agent_id == recipient:
                    result = await w.execute(task_data)
                    msg = MagicMock()
                    msg.content = result
                    return msg
            # Fallback
            msg = MagicMock()
            msg.content = {"status": "success", "task_id": task_data.get("task_id")}
            return msg

        hier.leader.send_task_request = _mock_send

    async def test_execute(self, state, bus, telemetry_patch):
        hier = self._make_hierarchy(state, bus, num_workers=3)
        await hier.initialize()
        self._mock_hier_agents(hier)

        result = await hier.execute({"task": "Analyze data"})

        assert result["task"] == "Analyze data"
        assert result["total_workers"] > 0
        assert result["status"] in ("success", "partial_success")
        assert "summary" in result
        await hier.shutdown()

    async def test_execute_with_string_task(self, state, bus, telemetry_patch):
        hier = self._make_hierarchy(state, bus)
        await hier.initialize()
        self._mock_hier_agents(hier)

        result = await hier.execute({"task": "Simple task"})
        assert result["task"] == "Simple task"
        await hier.shutdown()

    # --- _synthesize_results ---

    async def test_synthesize_all_success(self, state, bus, telemetry_patch):
        hier = self._make_hierarchy(state, bus)
        worker_results = {
            "w1": {"task_id": "t1", "status": "completed", "output": {"status": "ok"}},
            "w2": {"task_id": "t2", "status": "completed", "output": "plain"},
        }
        result = await hier._synthesize_results(worker_results, {"task": "test"})
        assert result["status"] == "success"
        assert result["successful_workers"] == 2
        assert result["failed_workers"] == 0

    async def test_synthesize_with_failures(self, state, bus, telemetry_patch):
        hier = self._make_hierarchy(state, bus)
        worker_results = {
            "w1": {"task_id": "t1", "status": "completed", "output": None},
            "w2": {"task_id": "t2", "status": "failed", "error": "timeout"},
        }
        result = await hier._synthesize_results(worker_results, {"task": "test"})
        assert result["status"] == "partial_success"
        assert result["failed_workers"] == 1
        assert len(result["failures"]) == 1

    # --- _create_synthesis_summary ---

    def test_create_synthesis_summary_empty(self, state, bus, telemetry_patch):
        hier = self._make_hierarchy(state, bus)
        summary = hier._create_synthesis_summary([])
        assert summary == "No results to synthesize."

    def test_create_synthesis_summary_with_dict_output(self, state, bus, telemetry_patch):
        hier = self._make_hierarchy(state, bus)
        results = [
            {"task_id": "t1", "output": {"status": "done"}},
        ]
        summary = hier._create_synthesis_summary(results)
        assert "t1" in summary
        assert "done" in summary

    def test_create_synthesis_summary_with_non_dict_output(self, state, bus, telemetry_patch):
        hier = self._make_hierarchy(state, bus)
        results = [
            {"task_id": "t1", "output": "just a string"},
        ]
        summary = hier._create_synthesis_summary(results)
        assert "t1" in summary
        assert "completed" in summary

    # --- _execute_assigned_tasks error path ---

    async def test_execute_assigned_tasks_worker_failure(self, state, bus, telemetry_patch):
        hier = self._make_hierarchy(state, bus)
        await hier.initialize()

        # Mock leader's send_task_request to fail
        hier.leader.send_task_request = AsyncMock(side_effect=Exception("worker down"))

        from ia_modules.agents.task_decomposition import Task
        assignments = {
            "worker_0": Task(task_id="t1", description="test task"),
        }
        results = await hier._execute_assigned_tasks(assignments)
        assert results["worker_0"]["status"] == "failed"
        assert "worker down" in results["worker_0"]["error"]
        await hier.shutdown()


# ---------------------------------------------------------------------------
# PeerAgent
# ---------------------------------------------------------------------------

class TestPeerAgent:
    async def test_contribute(self, state, bus, telemetry_patch):
        agent = PeerAgent(
            role=_make_role("peer1"), state_manager=state, message_bus=bus
        )
        await agent.initialize()
        result = await agent.execute(
            {"action": "contribute", "task": {"goal": "brainstorm"}, "round": 1}
        )
        assert result["status"] == "contributed"
        assert result["peer_id"] == "peer1"
        assert "content" in result

        stored = await state.get("peer1_last_contribution")
        assert stored is not None
        await agent.shutdown()

    async def test_contribute_with_previous(self, state, bus, telemetry_patch):
        agent = PeerAgent(
            role=_make_role("peer2"), state_manager=state, message_bus=bus
        )
        result = await agent.execute(
            {
                "action": "contribute",
                "task": {},
                "round": 2,
                "previous_contributions": [
                    {"insights": ["prev insight 1"]},
                    {"insights": ["prev insight 2"]},
                ],
            }
        )
        content = result["content"]
        assert content["builds_on"] == 2

    async def test_review_contributions(self, state, bus, telemetry_patch):
        agent = PeerAgent(
            role=_make_role("peer3"), state_manager=state, message_bus=bus
        )
        await agent.initialize()
        result = await agent.execute(
            {
                "action": "review_contributions",
                "contributions": [
                    {"peer_id": "other1", "content": {}},
                    {"peer_id": "other2", "content": {}},
                ],
            }
        )
        assert result["status"] == "reviewed"
        assert len(result["reviews"]) == 2
        await agent.shutdown()

    async def test_refine_contribution(self, state, bus, telemetry_patch):
        agent = PeerAgent(
            role=_make_role("peer4"), state_manager=state, message_bus=bus
        )
        await agent.initialize()

        # Store some reviews to be picked up
        await state.set("peer4_reviews", [{"reviewed_peer": "x", "feedback": "good"}])

        result = await agent.execute(
            {
                "action": "refine",
                "original_contribution": {
                    "content": {"insights": ["original insight"]},
                    "round": 1,
                },
                "round": 2,
            }
        )
        assert result["status"] == "refined"
        assert result["peer_id"] == "peer4"
        content = result["content"]
        assert "original insight" in content["insights"]
        assert content["incorporated_feedback"] == 1
        await agent.shutdown()

    async def test_refine_contribution_no_reviews(self, state, bus, telemetry_patch):
        agent = PeerAgent(
            role=_make_role("peer5"), state_manager=state, message_bus=bus
        )
        result = await agent.execute(
            {
                "action": "refine",
                "original_contribution": {"content": {"insights": []}, "round": 1},
                "round": 1,
            }
        )
        assert result["content"]["incorporated_feedback"] == 0

    async def test_unknown_action(self, state, bus, telemetry_patch):
        agent = PeerAgent(
            role=_make_role("peer6"), state_manager=state, message_bus=bus
        )
        result = await agent.execute({"action": "unknown"})
        assert result["status"] == "unknown_action"

    async def test_share_knowledge(self, state, bus, telemetry_patch):
        agent = PeerAgent(
            role=_make_role("sharer"), state_manager=state, message_bus=bus
        )
        await agent.initialize()
        # Just ensure no exception
        await agent.share_knowledge({"fact": "interesting"})
        await agent.shutdown()

    async def test_request_help(self, state, bus, telemetry_patch):
        agent = PeerAgent(
            role=_make_role("helper"), state_manager=state, message_bus=bus
        )
        await agent.initialize()

        # Patch sleep to speed up test
        with patch("ia_modules.agents.collaboration_patterns.peer_to_peer.asyncio.sleep",
                    new_callable=AsyncMock):
            responses = await agent.request_help("I need help")
        # No peers to respond, so empty
        assert isinstance(responses, list)
        await agent.shutdown()


# ---------------------------------------------------------------------------
# PeerToPeerCollaboration
# ---------------------------------------------------------------------------

class TestPeerToPeerCollaboration:
    def _make_p2p(self, state, bus, num_peers=3):
        peers = [
            PeerAgent(
                role=_make_role(f"p2p_{i}"), state_manager=state, message_bus=bus
            )
            for i in range(num_peers)
        ]
        return PeerToPeerCollaboration(peers, bus, state)

    async def test_initialize_and_shutdown(self, state, bus, telemetry_patch):
        p2p = self._make_p2p(state, bus)
        await p2p.initialize()
        assert len(bus.get_active_agents()) == 3
        assert len(p2p.contributions) == 3
        await p2p.shutdown()

    def _mock_p2p_agents(self, p2p):
        """Mock send_task_request, send_message, send_query on all peers."""
        def _make_mock_send(agent):
            async def _send(recipient, task_data, **kwargs):
                result = await agent.execute(task_data)
                msg = MagicMock()
                msg.content = result
                return msg
            return _send

        for peer in p2p.peers:
            peer.send_task_request = _make_mock_send(peer)
            peer.send_message = AsyncMock()
            # send_query returns an AgentMessage with content
            async def _mock_query(recipient, query_data, **kwargs):
                msg = MagicMock()
                msg.content = {"refined": "refined contribution", "improvements": ["improved"]}
                return msg
            peer.send_query = _mock_query

    async def test_execute(self, state, bus, telemetry_patch):
        p2p = self._make_p2p(state, bus, num_peers=2)
        await p2p.initialize()
        self._mock_p2p_agents(p2p)

        # Patch out the asyncio.sleep in _refine_contributions
        with patch(
            "ia_modules.agents.collaboration_patterns.peer_to_peer.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            result = await p2p.execute(
                {"task": "brainstorm ideas"}, rounds=2
            )

        assert result["status"] == "success"
        assert result["total_peers"] == 2
        assert result["rounds_completed"] >= 1
        assert result["total_contributions"] >= 2
        assert "key_insights" in result
        assert "collaborative_output" in result
        await p2p.shutdown()

    async def test_execute_single_round(self, state, bus, telemetry_patch):
        p2p = self._make_p2p(state, bus, num_peers=2)
        await p2p.initialize()
        self._mock_p2p_agents(p2p)

        with patch(
            "ia_modules.agents.collaboration_patterns.peer_to_peer.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            result = await p2p.execute({"task": "ideas"}, rounds=1)

        assert result["status"] == "success"
        await p2p.shutdown()

    # --- _gather_contributions error ---

    async def test_gather_contributions_agent_failure(self, state, bus, telemetry_patch):
        p2p = self._make_p2p(state, bus, num_peers=2)
        await p2p.initialize()
        self._mock_p2p_agents(p2p)

        # Now override one peer to fail
        p2p.peers[0].send_task_request = AsyncMock(side_effect=Exception("fail"))

        contributions = await p2p._gather_contributions({"task": "test"}, 1)
        # Only the second peer should have contributed
        assert len(contributions) == 1
        await p2p.shutdown()

    # --- _share_contributions error ---

    async def test_share_contributions_error(self, state, bus, telemetry_patch):
        p2p = self._make_p2p(state, bus, num_peers=2)
        await p2p.initialize()

        # Fail sending
        p2p.peers[0].send_message = AsyncMock(side_effect=Exception("send fail"))

        # Should not raise
        contributions = [{"peer_id": "p2p_1", "content": "stuff"}]
        await p2p._share_contributions(contributions)
        await p2p.shutdown()

    # --- _refine_contributions error ---

    async def test_refine_contributions_agent_failure(self, state, bus, telemetry_patch):
        p2p = self._make_p2p(state, bus, num_peers=2)
        await p2p.initialize()

        # Fail send_query for first peer
        p2p.peers[0].send_query = AsyncMock(side_effect=Exception("refine fail"))

        with patch(
            "ia_modules.agents.collaboration_patterns.peer_to_peer.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            contributions = [
                {"peer_id": "p2p_0", "content": "a"},
                {"peer_id": "p2p_1", "content": "b"},
            ]
            refined = await p2p._refine_contributions(contributions, 1)

        # Failed one falls back to original, other should succeed
        assert len(refined) == 2
        await p2p.shutdown()

    # --- _refine_contributions unknown peer ---

    async def test_refine_contributions_unknown_peer(self, state, bus, telemetry_patch):
        p2p = self._make_p2p(state, bus, num_peers=1)
        await p2p.initialize()

        with patch(
            "ia_modules.agents.collaboration_patterns.peer_to_peer.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            contributions = [{"peer_id": "nonexistent", "content": "x"}]
            refined = await p2p._refine_contributions(contributions, 1)

        # Unknown peer is skipped
        assert len(refined) == 0
        await p2p.shutdown()

    # --- _extract_key_insights ---

    def test_extract_key_insights_with_insights(self, state, bus, telemetry_patch):
        p2p = self._make_p2p(state, bus)
        contributions = [
            {"insights": ["a", "b"]},
            {"insights": ["b", "c"]},
        ]
        insights = p2p._extract_key_insights(contributions)
        assert "a" in insights
        assert "b" in insights
        assert "c" in insights
        # No duplicates
        assert len([x for x in insights if x == "b"]) == 1

    def test_extract_key_insights_with_content_key_points(self, state, bus, telemetry_patch):
        p2p = self._make_p2p(state, bus)
        contributions = [
            {"content": {"key_points": ["kp1", "kp2"]}},
        ]
        insights = p2p._extract_key_insights(contributions)
        assert "kp1" in insights

    def test_extract_key_insights_max_10(self, state, bus, telemetry_patch):
        p2p = self._make_p2p(state, bus)
        contributions = [{"insights": [f"i{i}" for i in range(15)]}]
        insights = p2p._extract_key_insights(contributions)
        assert len(insights) == 10

    def test_extract_key_insights_empty(self, state, bus, telemetry_patch):
        p2p = self._make_p2p(state, bus)
        insights = p2p._extract_key_insights([])
        assert insights == []

    # --- _merge_contributions ---

    def test_merge_contributions(self, state, bus, telemetry_patch):
        p2p = self._make_p2p(state, bus)
        contributions = [
            {"peer_id": "p1", "round": 1, "content": "idea 1"},
            {"peer_id": "p2", "round": 1, "content": {"summary": "idea 2"}},
            {"peer_id": "p1", "round": 2, "content": "idea 3"},
        ]
        merged = p2p._merge_contributions(contributions)
        assert "Round 1" in merged
        assert "Round 2" in merged
        assert "[p1]" in merged
        assert "[p2]" in merged
        assert "idea 2" in merged

    def test_merge_contributions_empty(self, state, bus, telemetry_patch):
        p2p = self._make_p2p(state, bus)
        merged = p2p._merge_contributions([])
        assert "Collaborative Output" in merged


# ---------------------------------------------------------------------------
# __init__ exports
# ---------------------------------------------------------------------------

class TestModuleExports:
    def test_all_exports(self):
        from ia_modules.agents.collaboration_patterns import __all__

        expected = [
            "HierarchicalCollaboration", "LeaderAgent", "WorkerAgent",
            "PeerToPeerCollaboration", "PeerAgent",
            "DebateCollaboration", "DebateAgent", "ModeratorAgent", "DebateRole",
            "ConsensusCollaboration", "ConsensusAgent", "ConsensusStrategy", "VoteType",
        ]
        for name in expected:
            assert name in __all__
