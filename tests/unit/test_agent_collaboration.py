"""
Unit tests for agent collaboration and communication.

Tests agent messaging, orchestration, and collaboration patterns.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock

from ia_modules.agents.communication import (
    AgentMessage,
    MessageType,
    MessageBus
)
from ia_modules.agents.orchestrator import (
    AgentOrchestrator,
    Edge
)
from ia_modules.agents.state import StateManager
from ia_modules.agents.core import BaseAgent, AgentRole


# Mock agent for testing
class MockAgent(BaseAgent):
    """Mock agent for testing."""

    def __init__(self, role, state, result="success"):
        super().__init__(role, state)
        self.result = result
        self.execute_count = 0

    async def execute(self, input_data):
        """Execute mock task."""
        self.execute_count += 1
        await asyncio.sleep(0.01)  # Simulate work
        return {"status": self.result, "input": input_data}


class TestMessageType:
    """Test MessageType enum."""

    def test_message_types(self):
        """MessageType enum has expected values."""
        assert MessageType.TASK_REQUEST.value == "task_request"
        assert MessageType.TASK_RESPONSE.value == "task_response"
        assert MessageType.QUERY.value == "query"
        assert MessageType.RESPONSE.value == "response"
        assert MessageType.BROADCAST.value == "broadcast"
        assert MessageType.STATUS_UPDATE.value == "status_update"
        assert MessageType.ERROR.value == "error"
        assert MessageType.VOTE.value == "vote"
        assert MessageType.PROPOSAL.value == "proposal"
        assert MessageType.CRITIQUE.value == "critique"
        assert MessageType.AGREEMENT.value == "agreement"
        assert MessageType.DISAGREEMENT.value == "disagreement"


class TestAgentMessage:
    """Test AgentMessage dataclass."""

    def test_creation_minimal(self):
        """AgentMessage can be created with minimal fields."""
        msg = AgentMessage(
            sender="agent1",
            message_type=MessageType.QUERY,
            content="What is the status?"
        )

        assert msg.sender == "agent1"
        assert msg.message_type == MessageType.QUERY
        assert msg.content == "What is the status?"
        assert msg.recipient is None
        assert msg.metadata == {}
        assert msg.reply_to is None
        assert msg.priority == 0
        assert msg.message_id is not None

    def test_creation_full(self):
        """AgentMessage can be created with all fields."""
        msg = AgentMessage(
            sender="agent1",
            recipient="agent2",
            message_type=MessageType.TASK_REQUEST,
            content={"task": "analyze"},
            metadata={"urgent": True},
            reply_to="msg-123",
            priority=5
        )

        assert msg.sender == "agent1"
        assert msg.recipient == "agent2"
        assert msg.content == {"task": "analyze"}
        assert msg.metadata == {"urgent": True}
        assert msg.reply_to == "msg-123"
        assert msg.priority == 5

    def test_is_broadcast(self):
        """Can check if message is broadcast."""
        broadcast = AgentMessage(
            sender="agent1",
            message_type=MessageType.BROADCAST,
            content="Update"
        )

        direct = AgentMessage(
            sender="agent1",
            recipient="agent2",
            message_type=MessageType.QUERY,
            content="Question"
        )

        assert broadcast.is_broadcast() is True
        assert direct.is_broadcast() is False

    def test_is_reply(self):
        """Can check if message is a reply."""
        reply = AgentMessage(
            sender="agent1",
            message_type=MessageType.RESPONSE,
            content="Answer",
            reply_to="msg-123"
        )

        not_reply = AgentMessage(
            sender="agent1",
            message_type=MessageType.QUERY,
            content="Question"
        )

        assert reply.is_reply() is True
        assert not_reply.is_reply() is False

    def test_create_reply(self):
        """Can create reply to message."""
        original = AgentMessage(
            sender="agent1",
            recipient="agent2",
            message_type=MessageType.QUERY,
            content="Question",
            priority=3
        )

        reply = original.create_reply(
            sender="agent2",
            content="Answer"
        )

        assert reply.sender == "agent2"
        assert reply.recipient == "agent1"
        assert reply.message_type == MessageType.RESPONSE
        assert reply.content == "Answer"
        assert reply.reply_to == original.message_id
        assert reply.priority == original.priority

    def test_create_reply_custom_type(self):
        """Can create reply with custom message type."""
        original = AgentMessage(
            sender="agent1",
            message_type=MessageType.PROPOSAL,
            content="Suggestion"
        )

        agreement = original.create_reply(
            sender="agent2",
            content="I agree",
            message_type=MessageType.AGREEMENT
        )

        assert agreement.message_type == MessageType.AGREEMENT

    def test_repr(self):
        """Message has readable repr."""
        msg = AgentMessage(
            sender="agent1",
            recipient="agent2",
            message_type=MessageType.QUERY,
            content="Test"
        )

        repr_str = repr(msg)

        assert "AgentMessage" in repr_str
        assert "agent1" in repr_str
        assert "agent2" in repr_str


class TestMessageBus:
    """Test MessageBus functionality."""

    @pytest.fixture
    def message_bus(self):
        """Create message bus instance."""
        return MessageBus()

    @pytest.mark.asyncio
    async def test_creation(self, message_bus):
        """MessageBus can be created."""
        assert len(message_bus._subscribers) == 0
        assert len(message_bus._queues) == 0
        assert len(message_bus._active_agents) == 0

    @pytest.mark.asyncio
    async def test_subscribe(self, message_bus):
        """Can subscribe agent to bus."""
        handler = AsyncMock()

        await message_bus.subscribe("agent1", handler)

        assert "agent1" in message_bus._subscribers
        assert "agent1" in message_bus._queues
        assert "agent1" in message_bus._active_agents

    @pytest.mark.asyncio
    async def test_unsubscribe(self, message_bus):
        """Can unsubscribe agent from bus."""
        handler = AsyncMock()
        await message_bus.subscribe("agent1", handler)

        await message_bus.unsubscribe("agent1")

        assert "agent1" not in message_bus._subscribers
        assert "agent1" not in message_bus._queues
        assert "agent1" not in message_bus._active_agents

    @pytest.mark.asyncio
    async def test_send_direct_message(self, message_bus):
        """Can send direct message to agent."""
        handler = AsyncMock()
        await message_bus.subscribe("agent1", handler)

        msg = AgentMessage(
            sender="agent2",
            recipient="agent1",
            message_type=MessageType.QUERY,
            content="Question"
        )

        delivered = await message_bus.send(msg)

        assert delivered is True
        await asyncio.sleep(0.1)  # Wait for handler
        assert handler.called

    @pytest.mark.asyncio
    async def test_send_to_nonexistent_agent(self, message_bus):
        """Sending to nonexistent agent returns False."""
        msg = AgentMessage(
            sender="agent1",
            recipient="nonexistent",
            message_type=MessageType.QUERY,
            content="Test"
        )

        delivered = await message_bus.send(msg)

        assert delivered is False

    @pytest.mark.asyncio
    async def test_broadcast_message(self, message_bus):
        """Can broadcast message to all agents."""
        handler1 = AsyncMock()
        handler2 = AsyncMock()

        await message_bus.subscribe("agent1", handler1)
        await message_bus.subscribe("agent2", handler2)

        msg = AgentMessage(
            sender="coordinator",
            message_type=MessageType.BROADCAST,
            content="Update"
        )

        await message_bus.send(msg)
        await asyncio.sleep(0.1)  # Wait for handlers

        assert handler1.called
        assert handler2.called

    @pytest.mark.asyncio
    async def test_broadcast_excludes_sender(self, message_bus):
        """Broadcast doesn't send to sender."""
        handler1 = AsyncMock()
        handler2 = AsyncMock()

        await message_bus.subscribe("sender", handler1)
        await message_bus.subscribe("other", handler2)

        msg = AgentMessage(
            sender="sender",
            message_type=MessageType.BROADCAST,
            content="Update"
        )

        await message_bus.send(msg)
        await asyncio.sleep(0.1)

        assert not handler1.called
        assert handler2.called

    @pytest.mark.asyncio
    async def test_broadcast_convenience_method(self, message_bus):
        """Broadcast convenience method works."""
        handler = AsyncMock()
        await message_bus.subscribe("agent1", handler)

        await message_bus.broadcast(
            "sender",
            MessageType.STATUS_UPDATE,
            {"status": "complete"}
        )

        await asyncio.sleep(0.1)
        assert handler.called

    @pytest.mark.asyncio
    async def test_receive_message(self, message_bus):
        """Can receive message from queue."""
        await message_bus.subscribe("agent1", AsyncMock())

        msg = AgentMessage(
            sender="agent2",
            recipient="agent1",
            message_type=MessageType.QUERY,
            content="Test"
        )

        await message_bus.send(msg)

        received = await message_bus.receive("agent1", timeout=1.0)

        assert received is not None
        assert received.sender == "agent2"
        assert received.content == "Test"

    @pytest.mark.asyncio
    async def test_receive_timeout(self, message_bus):
        """Receive returns None on timeout."""
        await message_bus.subscribe("agent1", AsyncMock())

        received = await message_bus.receive("agent1", timeout=0.1)

        assert received is None

    @pytest.mark.asyncio
    async def test_receive_all_messages(self, message_bus):
        """Can receive all pending messages."""
        await message_bus.subscribe("agent1", AsyncMock())

        for i in range(3):
            msg = AgentMessage(
                sender="agent2",
                recipient="agent1",
                message_type=MessageType.QUERY,
                content=f"Message {i}"
            )
            await message_bus.send(msg)

        messages = await message_bus.receive_all("agent1")

        assert len(messages) == 3

    @pytest.mark.asyncio
    async def test_handler_error_sends_error_message(self, message_bus):
        """Handler error sends error message back."""
        # Handler that raises error
        async def error_handler(msg):
            raise ValueError("Handler error")

        await message_bus.subscribe("agent1", error_handler)
        await message_bus.subscribe("agent2", AsyncMock())

        msg = AgentMessage(
            sender="agent2",
            recipient="agent1",
            message_type=MessageType.QUERY,
            content="Test"
        )

        await message_bus.send(msg)
        await asyncio.sleep(0.1)

        # Check error message was sent back
        error_msg = await message_bus.receive("agent2", timeout=0.5)

        assert error_msg is not None
        assert error_msg.message_type == MessageType.ERROR

    @pytest.mark.asyncio
    async def test_get_message_history(self, message_bus):
        """Can get message history."""
        msg = AgentMessage(
            sender="agent1",
            recipient="agent2",
            message_type=MessageType.QUERY,
            content="Test"
        )

        await message_bus.send(msg)

        history = message_bus.get_message_history()

        assert len(history) > 0
        assert history[0].content == "Test"

    @pytest.mark.asyncio
    async def test_get_message_history_filtered(self, message_bus):
        """Can filter message history by agent."""
        await message_bus.subscribe("agent1", AsyncMock())
        await message_bus.subscribe("agent2", AsyncMock())

        msg1 = AgentMessage(
            sender="agent1",
            recipient="agent2",
            message_type=MessageType.QUERY,
            content="Test1"
        )
        msg2 = AgentMessage(
            sender="agent2",
            recipient="agent3",
            message_type=MessageType.QUERY,
            content="Test2"
        )

        await message_bus.send(msg1)
        await message_bus.send(msg2)

        history = message_bus.get_message_history(agent_id="agent1")

        # Should include messages from/to agent1
        assert len(history) > 0

    @pytest.mark.asyncio
    async def test_clear_history(self, message_bus):
        """Can clear message history."""
        msg = AgentMessage(
            sender="agent1",
            message_type=MessageType.BROADCAST,
            content="Test"
        )

        await message_bus.send(msg)
        assert len(message_bus._message_history) > 0

        message_bus.clear_history()

        assert len(message_bus._message_history) == 0

    @pytest.mark.asyncio
    async def test_get_active_agents(self, message_bus):
        """Can get set of active agents."""
        await message_bus.subscribe("agent1", AsyncMock())
        await message_bus.subscribe("agent2", AsyncMock())

        active = message_bus.get_active_agents()

        assert len(active) == 2
        assert "agent1" in active
        assert "agent2" in active

    @pytest.mark.asyncio
    async def test_wait_for_replies(self, message_bus):
        """Can wait for replies to a message."""
        await message_bus.subscribe("agent1", AsyncMock())
        await message_bus.subscribe("agent2", AsyncMock())
        await message_bus.subscribe("agent3", AsyncMock())

        # Send original message
        original = AgentMessage(
            sender="coordinator",
            message_type=MessageType.QUERY,
            content="Question"
        )
        await message_bus.send(original)

        # Send replies
        async def send_replies():
            await asyncio.sleep(0.1)
            for i in range(2):
                reply = AgentMessage(
                    sender=f"agent{i+1}",
                    message_type=MessageType.RESPONSE,
                    content=f"Answer {i}",
                    reply_to=original.message_id
                )
                await message_bus.send(reply)

        # Start sending replies in background
        asyncio.create_task(send_replies())

        # Wait for replies
        replies = await message_bus.wait_for_replies(
            original.message_id,
            expected_count=2,
            timeout=2.0
        )

        assert len(replies) == 2

    @pytest.mark.asyncio
    async def test_wait_for_replies_timeout(self, message_bus):
        """Wait for replies times out correctly."""
        with pytest.raises(asyncio.TimeoutError):
            await message_bus.wait_for_replies(
                "nonexistent",
                expected_count=5,
                timeout=0.1
            )


class TestEdge:
    """Test Edge dataclass."""

    def test_creation_minimal(self):
        """Edge can be created with minimal fields."""
        edge = Edge(to="agent2")

        assert edge.to == "agent2"
        assert edge.condition is None
        assert edge.metadata == {}

    def test_creation_full(self):
        """Edge can be created with all fields."""
        def condition(state):
            return True

        edge = Edge(
            to="agent2",
            condition=condition,
            metadata={"weight": 1.0}
        )

        assert edge.to == "agent2"
        assert edge.condition == condition
        assert edge.metadata == {"weight": 1.0}


class TestAgentOrchestrator:
    """Test AgentOrchestrator functionality."""

    @pytest.fixture
    def state_manager(self):
        """Create state manager."""
        return StateManager(thread_id="test-thread")

    @pytest.fixture
    def orchestrator(self, state_manager):
        """Create orchestrator instance."""
        return AgentOrchestrator(state_manager)

    @pytest.fixture
    def mock_agent(self, state_manager):
        """Create mock agent."""
        role = AgentRole(name="test", description="Test agent")
        return MockAgent(role, state_manager)

    @pytest.mark.asyncio
    async def test_creation(self, orchestrator, state_manager):
        """AgentOrchestrator can be created."""
        assert orchestrator.state == state_manager
        assert len(orchestrator.agents) == 0
        assert len(orchestrator.graph) == 0

    @pytest.mark.asyncio
    async def test_add_agent(self, orchestrator, mock_agent):
        """Can add agent to orchestrator."""
        orchestrator.add_agent("agent1", mock_agent)

        assert "agent1" in orchestrator.agents
        assert orchestrator.agents["agent1"] == mock_agent
        assert "agent1" in orchestrator.graph

    @pytest.mark.asyncio
    async def test_add_multiple_agents(self, orchestrator, state_manager):
        """Can add multiple agents."""
        role1 = AgentRole(name="agent1", description="First")
        role2 = AgentRole(name="agent2", description="Second")

        agent1 = MockAgent(role1, state_manager)
        agent2 = MockAgent(role2, state_manager)

        orchestrator.add_agent("agent1", agent1)
        orchestrator.add_agent("agent2", agent2)

        assert len(orchestrator.agents) == 2

    @pytest.mark.asyncio
    async def test_add_edge(self, orchestrator):
        """Can add edge between agents."""
        orchestrator.add_edge("agent1", "agent2")

        assert len(orchestrator.graph["agent1"]) == 1
        assert orchestrator.graph["agent1"][0].to == "agent2"

    @pytest.mark.asyncio
    async def test_add_edge_with_condition(self, orchestrator):
        """Can add conditional edge."""
        async def condition(state):
            return True

        orchestrator.add_edge("agent1", "agent2", condition=condition)

        edge = orchestrator.graph["agent1"][0]
        assert edge.condition == condition

    @pytest.mark.asyncio
    async def test_add_edge_with_metadata(self, orchestrator):
        """Can add edge with metadata."""
        orchestrator.add_edge(
            "agent1",
            "agent2",
            metadata={"priority": "high"}
        )

        edge = orchestrator.graph["agent1"][0]
        assert edge.metadata == {"priority": "high"}

    @pytest.mark.asyncio
    async def test_add_hook(self, orchestrator):
        """Can add execution hooks."""
        hook = AsyncMock()

        orchestrator.add_hook("agent_start", hook)

        assert hook in orchestrator.on_agent_start

    @pytest.mark.asyncio
    async def test_add_hook_invalid_event(self, orchestrator):
        """Adding invalid hook raises error."""
        with pytest.raises(ValueError):
            orchestrator.add_hook("invalid_event", AsyncMock())

    @pytest.mark.asyncio
    async def test_run_single_agent(self, orchestrator, mock_agent):
        """Can run single agent."""
        orchestrator.add_agent("agent1", mock_agent)

        result = await orchestrator.run("agent1", {"task": "test"})

        assert result is not None
        assert mock_agent.execute_count == 1

    @pytest.mark.asyncio
    async def test_run_sequential_agents(self, orchestrator, state_manager):
        """Can run agents in sequence."""
        role1 = AgentRole(name="agent1", description="First")
        role2 = AgentRole(name="agent2", description="Second")

        agent1 = MockAgent(role1, state_manager, result="step1")
        agent2 = MockAgent(role2, state_manager, result="step2")

        orchestrator.add_agent("agent1", agent1)
        orchestrator.add_agent("agent2", agent2)
        orchestrator.add_edge("agent1", "agent2")

        await orchestrator.run("agent1", {"task": "test"})

        assert agent1.execute_count == 1
        assert agent2.execute_count == 1

    @pytest.mark.asyncio
    async def test_run_conditional_branch(self, orchestrator, state_manager):
        """Can run conditional branch."""
        role1 = AgentRole(name="agent1", description="First")
        role2 = AgentRole(name="agent2", description="Option A")
        role3 = AgentRole(name="agent3", description="Option B")

        agent1 = MockAgent(role1, state_manager)
        agent2 = MockAgent(role2, state_manager)
        agent3 = MockAgent(role3, state_manager)

        orchestrator.add_agent("agent1", agent1)
        orchestrator.add_agent("agent2", agent2)
        orchestrator.add_agent("agent3", agent3)

        # Conditional branching
        async def go_to_agent2(state):
            return await state.get("choice") == "A"

        async def go_to_agent3(state):
            return await state.get("choice") == "B"

        orchestrator.add_edge("agent1", "agent2", condition=go_to_agent2)
        orchestrator.add_edge("agent1", "agent3", condition=go_to_agent3)

        # Set choice in state
        await state_manager.set("choice", "A")

        await orchestrator.run("agent1", {})

        assert agent1.execute_count == 1
        assert agent2.execute_count == 1
        assert agent3.execute_count == 0  # Not executed

    @pytest.mark.asyncio
    async def test_hooks_called(self, orchestrator, mock_agent):
        """Execution hooks are called."""
        start_hook = AsyncMock()
        complete_hook = AsyncMock()

        orchestrator.add_hook("agent_start", start_hook)
        orchestrator.add_hook("agent_complete", complete_hook)

        orchestrator.add_agent("agent1", mock_agent)

        await orchestrator.run("agent1", {"task": "test"})

        assert start_hook.called
        assert complete_hook.called


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_message_bus_repr(self):
        """MessageBus has readable repr."""
        bus = MessageBus()
        await bus.subscribe("agent1", AsyncMock())

        repr_str = repr(bus)

        assert "MessageBus" in repr_str
        assert "agents=1" in repr_str

    @pytest.mark.asyncio
    async def test_receive_from_nonexistent_agent(self):
        """Receiving from nonexistent agent returns None."""
        bus = MessageBus()

        result = await bus.receive("nonexistent", timeout=0.1)

        assert result is None

    @pytest.mark.asyncio
    async def test_receive_all_from_nonexistent_agent(self):
        """Receive all from nonexistent agent returns empty list."""
        bus = MessageBus()

        messages = await bus.receive_all("nonexistent")

        assert messages == []

    @pytest.mark.asyncio
    async def test_orchestrator_nonexistent_start_agent(self):
        """Running nonexistent start agent raises error."""
        state = StateManager(thread_id="test")
        orchestrator = AgentOrchestrator(state)

        with pytest.raises(Exception):
            await orchestrator.run("nonexistent", {})


class TestIntegration:
    """Integration tests for agent collaboration."""

    @pytest.mark.asyncio
    async def test_multi_agent_voting_pattern(self):
        """Test voting/consensus pattern."""
        bus = MessageBus()
        StateManager(thread_id="voting")

        # Create voting agents
        agent_handlers = {}
        for i in range(3):
            agent_id = f"voter{i}"

            async def vote_handler(msg, vote_value=i % 2):
                """Agent votes yes or no."""
                if msg.message_type == MessageType.PROPOSAL:
                    reply = msg.create_reply(
                        sender=agent_id,
                        content={"vote": vote_value},
                        message_type=MessageType.VOTE
                    )
                    await bus.send(reply)

            agent_handlers[agent_id] = vote_handler
            await bus.subscribe(agent_id, vote_handler)

        # Send proposal
        proposal = AgentMessage(
            sender="coordinator",
            message_type=MessageType.PROPOSAL,
            content={"proposal": "Deploy to production"}
        )
        await bus.send(proposal)

        # Wait for votes
        await asyncio.sleep(0.2)

        # Should have 3 votes in history
        votes = [
            msg for msg in bus.get_message_history()
            if msg.message_type == MessageType.VOTE
        ]

        assert len(votes) == 3

    @pytest.mark.asyncio
    async def test_critic_feedback_workflow(self):
        """Test critic feedback collaboration."""
        state = StateManager(thread_id="feedback")
        orchestrator = AgentOrchestrator(state)

        # Worker agent
        role1 = AgentRole(name="worker", description="Does work")
        worker = MockAgent(role1, state, result="work_done")

        # Critic agent
        role2 = AgentRole(name="critic", description="Reviews work")
        critic = MockAgent(role2, state, result="reviewed")

        orchestrator.add_agent("worker", worker)
        orchestrator.add_agent("critic", critic)

        # Add feedback loop
        orchestrator.add_feedback_loop("worker", "critic", max_iterations=2)

        # Initially not approved
        await state.set("approved", False)

        # This would normally run the loop
        # For now just verify the graph structure
        assert len(orchestrator.graph["worker"]) > 0
        assert orchestrator.graph["worker"][0].to == "critic"
