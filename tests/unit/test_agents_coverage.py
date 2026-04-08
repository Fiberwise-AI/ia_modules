"""
Comprehensive unit tests for ia_modules agents submodules with low coverage.

Covers:
- specialist_agents.py (ResearchAgent, AnalysisAgent, SynthesisAgent, CriticAgent)
- subprocess_executor.py (SubprocessExecutor)
- base_agent.py (BaseCollaborativeAgent)
- task_decomposition.py (Task, TaskDecomposer, DependencyGraph, TaskStatus, DecompositionStrategy)
- a2a_executor.py (A2AExecutor)
"""

import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from ia_modules.agents.core import AgentRole, BaseAgent
from ia_modules.agents.state import StateManager
from ia_modules.agents.communication import MessageBus, MessageType, AgentMessage
from ia_modules.agents.base_agent import BaseCollaborativeAgent
from ia_modules.agents.specialist_agents import (
    ResearchAgent, AnalysisAgent, SynthesisAgent, CriticAgent,
)
from ia_modules.agents.task_decomposition import (
    Task, TaskStatus, TaskDecomposer, DependencyGraph,
    DecompositionStrategy,
)
from ia_modules.agents.executor import (
    AgentConfig, AgentEvent, AgentMode, CLIType, EventType,
)
from ia_modules.agents.subprocess_executor import SubprocessExecutor
from ia_modules.agents.a2a_executor import A2AExecutor


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_role(name="test_agent", **kw):
    return AgentRole(
        name=name,
        description=f"{name} agent",
        **kw,
    )


def _make_state(thread_id="test-thread"):
    return StateManager(thread_id=thread_id)


def _make_bus():
    return MessageBus()


def _make_config(**overrides):
    defaults = dict(
        task="do something",
        cwd="/tmp",
        job_id="job-1234-5678-abcd",
    )
    defaults.update(overrides)
    return AgentConfig(**defaults)


# ---------------------------------------------------------------------------
# Task / TaskStatus
# ---------------------------------------------------------------------------

class TestTask:
    def test_is_ready_no_deps(self):
        t = Task(task_id="t1", description="d")
        assert t.is_ready(set()) is True

    def test_is_ready_deps_satisfied(self):
        t = Task(task_id="t2", description="d", dependencies={"t1"})
        assert t.is_ready({"t1"}) is True

    def test_is_ready_deps_not_satisfied(self):
        t = Task(task_id="t2", description="d", dependencies={"t1"})
        assert t.is_ready(set()) is False

    def test_is_ready_non_pending(self):
        t = Task(task_id="t1", description="d", status=TaskStatus.COMPLETED)
        assert t.is_ready(set()) is False

    def test_is_blocked_by_failure(self):
        t = Task(task_id="t2", description="d", dependencies={"t1"})
        assert t.is_blocked({"t1"}) is True

    def test_not_blocked(self):
        t = Task(task_id="t2", description="d", dependencies={"t1"})
        assert t.is_blocked({"t3"}) is False

    def test_mark_completed(self):
        t = Task(task_id="t1", description="d")
        t.mark_completed({"result": "ok"})
        assert t.status == TaskStatus.COMPLETED
        assert t.output_data == {"result": "ok"}

    def test_mark_failed(self):
        t = Task(task_id="t1", description="d")
        t.mark_failed("boom")
        assert t.status == TaskStatus.FAILED
        assert t.error == "boom"

    def test_repr(self):
        t = Task(task_id="t1", description="d", assigned_to="agent-a")
        r = repr(t)
        assert "t1" in r
        assert "pending" in r


class TestTaskStatus:
    def test_enum_values(self):
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.READY.value == "ready"
        assert TaskStatus.IN_PROGRESS.value == "in_progress"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.BLOCKED.value == "blocked"


class TestDecompositionStrategy:
    def test_enum_values(self):
        assert DecompositionStrategy.SEQUENTIAL.value == "sequential"
        assert DecompositionStrategy.PARALLEL.value == "parallel"
        assert DecompositionStrategy.HIERARCHICAL.value == "hierarchical"
        assert DecompositionStrategy.PIPELINE.value == "pipeline"
        assert DecompositionStrategy.DYNAMIC.value == "dynamic"


# ---------------------------------------------------------------------------
# TaskDecomposer
# ---------------------------------------------------------------------------

class TestTaskDecomposer:
    async def test_decompose_sequential(self):
        td = TaskDecomposer()
        tasks = await td.decompose("test task", DecompositionStrategy.SEQUENTIAL)
        assert len(tasks) == 3
        # second depends on first
        assert "task_1" in tasks[1].dependencies
        assert "task_2" in tasks[2].dependencies

    async def test_decompose_parallel(self):
        td = TaskDecomposer()
        tasks = await td.decompose("test task", DecompositionStrategy.PARALLEL)
        assert len(tasks) == 4
        synthesis = [t for t in tasks if t.task_id == "task_synthesis"][0]
        assert len(synthesis.dependencies) == 3

    async def test_decompose_hierarchical(self):
        td = TaskDecomposer()
        tasks = await td.decompose("test task", DecompositionStrategy.HIERARCHICAL)
        assert len(tasks) == 6
        main = [t for t in tasks if t.task_id == "main_task"][0]
        assert "subtask_1" in main.dependencies
        assert "subtask_2" in main.dependencies

    async def test_decompose_pipeline(self):
        td = TaskDecomposer()
        tasks = await td.decompose("test task", DecompositionStrategy.PIPELINE)
        assert len(tasks) == 4
        assert "stage_1_input" in tasks[1].dependencies

    async def test_decompose_dynamic(self):
        td = TaskDecomposer()
        tasks = await td.decompose("test task", DecompositionStrategy.DYNAMIC)
        assert len(tasks) == 2
        assert "analyze_requirements" in tasks[1].dependencies

    async def test_get_execution_order_sequential(self):
        td = TaskDecomposer()
        tasks = await td.decompose("t", DecompositionStrategy.SEQUENTIAL)
        levels = td.get_execution_order(tasks)
        assert len(levels) == 3
        assert levels[0][0].task_id == "task_1"

    async def test_get_execution_order_parallel(self):
        td = TaskDecomposer()
        tasks = await td.decompose("t", DecompositionStrategy.PARALLEL)
        levels = td.get_execution_order(tasks)
        # First level has the 3 parallel tasks
        assert len(levels[0]) == 3
        # Second level has synthesis
        assert levels[1][0].task_id == "task_synthesis"

    async def test_get_execution_order_circular_dependency(self):
        td = TaskDecomposer()
        t1 = Task(task_id="a", description="a", dependencies={"b"})
        t2 = Task(task_id="b", description="b", dependencies={"a"})
        with pytest.raises(ValueError, match="[Cc]ircular|[Dd]eadlock"):
            td.get_execution_order([t1, t2])

    async def test_validate_dependencies_valid(self):
        td = TaskDecomposer()
        tasks = await td.decompose("t", DecompositionStrategy.SEQUENTIAL)
        errors = td.validate_dependencies(tasks)
        assert errors == []

    async def test_validate_dependencies_missing(self):
        td = TaskDecomposer()
        t = Task(task_id="t1", description="d", dependencies={"nonexistent"})
        errors = td.validate_dependencies([t])
        assert any("missing" in e.lower() for e in errors)

    async def test_validate_dependencies_self_dep(self):
        td = TaskDecomposer()
        t = Task(task_id="t1", description="d", dependencies={"t1"})
        errors = td.validate_dependencies([t])
        assert any("itself" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# DependencyGraph
# ---------------------------------------------------------------------------

class TestDependencyGraph:
    def test_add_task_and_get_ready(self):
        g = DependencyGraph()
        t1 = Task(task_id="t1", description="d1", priority=2)
        t2 = Task(task_id="t2", description="d2", dependencies={"t1"}, priority=1)
        g.add_tasks([t1, t2])
        ready = g.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].task_id == "t1"

    def test_mark_in_progress(self):
        g = DependencyGraph()
        t = Task(task_id="t1", description="d")
        g.add_task(t)
        g.mark_in_progress("t1")
        assert t.status == TaskStatus.IN_PROGRESS
        assert "t1" in g.in_progress
        # Should not appear in ready tasks
        assert g.get_ready_tasks() == []

    def test_mark_completed(self):
        g = DependencyGraph()
        t1 = Task(task_id="t1", description="d1")
        t2 = Task(task_id="t2", description="d2", dependencies={"t1"})
        g.add_tasks([t1, t2])
        g.mark_in_progress("t1")
        g.mark_completed("t1", {"result": "ok"})
        assert t1.status == TaskStatus.COMPLETED
        assert "t1" in g.completed
        assert "t1" not in g.in_progress
        # t2 now ready
        assert len(g.get_ready_tasks()) == 1

    def test_mark_failed_blocks_dependents(self):
        g = DependencyGraph()
        t1 = Task(task_id="t1", description="d1")
        t2 = Task(task_id="t2", description="d2", dependencies={"t1"})
        t1.dependents.add("t2")
        g.add_tasks([t1, t2])
        g.mark_failed("t1", "error")
        assert t1.status == TaskStatus.FAILED
        assert t2.status == TaskStatus.BLOCKED
        assert g.has_failures()

    def test_status_summary(self):
        g = DependencyGraph()
        g.add_tasks([
            Task(task_id="t1", description="d1"),
            Task(task_id="t2", description="d2"),
        ])
        g.mark_completed("t1", {})
        summary = g.get_status_summary()
        assert summary["total_tasks"] == 2
        assert summary["completed"] == 1
        assert summary["progress_percent"] == 50.0

    def test_is_complete(self):
        g = DependencyGraph()
        g.add_task(Task(task_id="t1", description="d"))
        assert g.is_complete() is False
        g.mark_completed("t1", {})
        assert g.is_complete() is True

    def test_is_complete_with_failures(self):
        g = DependencyGraph()
        g.add_task(Task(task_id="t1", description="d"))
        g.mark_failed("t1", "err")
        assert g.is_complete() is True
        assert g.has_failures() is True

    def test_repr(self):
        g = DependencyGraph()
        g.add_task(Task(task_id="t1", description="d"))
        assert "DependencyGraph" in repr(g)

    def test_mark_nonexistent_is_noop(self):
        g = DependencyGraph()
        # Should not raise
        g.mark_in_progress("no_such")
        g.mark_completed("no_such", {})
        g.mark_failed("no_such", "e")

    def test_get_ready_sorted_by_priority(self):
        g = DependencyGraph()
        g.add_tasks([
            Task(task_id="low", description="d", priority=1),
            Task(task_id="high", description="d", priority=10),
            Task(task_id="mid", description="d", priority=5),
        ])
        ready = g.get_ready_tasks()
        assert ready[0].task_id == "high"
        assert ready[-1].task_id == "low"

    def test_empty_graph(self):
        g = DependencyGraph()
        assert g.get_status_summary()["total_tasks"] == 0
        assert g.get_status_summary()["progress_percent"] == 0
        assert g.is_complete() is True
        assert g.has_failures() is False


# ---------------------------------------------------------------------------
# BaseCollaborativeAgent
# ---------------------------------------------------------------------------

class ConcreteCollaborativeAgent(BaseCollaborativeAgent):
    """Concrete implementation for testing."""
    async def execute(self, input_data):
        return {"status": "success", "data": input_data}


class TestBaseCollaborativeAgent:
    async def test_init(self):
        role = _make_role("collab")
        state = _make_state()
        bus = _make_bus()
        agent = ConcreteCollaborativeAgent(role, state, bus, enable_telemetry=False)
        assert agent.agent_id == "collab"
        assert agent.message_bus is bus

    async def test_initialize_and_shutdown(self):
        role = _make_role("collab")
        state = _make_state()
        bus = _make_bus()
        agent = ConcreteCollaborativeAgent(role, state, bus, enable_telemetry=False)
        await agent.initialize()
        assert "collab" in bus.get_active_agents()
        await agent.shutdown()
        assert "collab" not in bus.get_active_agents()

    async def test_send_message(self):
        role = _make_role("sender")
        state = _make_state()
        bus = _make_bus()
        agent = ConcreteCollaborativeAgent(role, state, bus, enable_telemetry=False)
        await agent.initialize()

        # Register a receiver
        received = []
        await bus.subscribe("receiver", lambda m: received.append(m))

        msg = await agent.send_message(
            recipient="receiver",
            message_type=MessageType.QUERY,
            content={"q": "hello"},
        )
        assert msg.sender == "sender"
        # Allow async delivery
        await asyncio.sleep(0.05)
        assert len(received) >= 1

    async def test_send_error(self):
        role = _make_role("sender")
        state = _make_state()
        bus = _make_bus()
        agent = ConcreteCollaborativeAgent(role, state, bus, enable_telemetry=False)
        await agent.initialize()

        received = []
        await bus.subscribe("receiver", lambda m: received.append(m))

        await agent.send_error("receiver", "something failed", reply_to="orig-123")
        await asyncio.sleep(0.05)
        assert len(received) >= 1
        assert received[0].message_type == MessageType.ERROR

    async def test_broadcast_message(self):
        role = _make_role("broadcaster")
        state = _make_state()
        bus = _make_bus()
        agent = ConcreteCollaborativeAgent(role, state, bus, enable_telemetry=False)
        await agent.initialize()

        received_a = []
        received_b = []
        await bus.subscribe("a", lambda m: received_a.append(m))
        await bus.subscribe("b", lambda m: received_b.append(m))

        await agent.broadcast_message(MessageType.STATUS_UPDATE, {"status": "done"})
        await asyncio.sleep(0.05)
        assert len(received_a) >= 1
        assert len(received_b) >= 1

    async def test_send_task_request_no_wait(self):
        role = _make_role("requester")
        state = _make_state()
        bus = _make_bus()
        agent = ConcreteCollaborativeAgent(role, state, bus, enable_telemetry=False)
        await agent.initialize()
        await bus.subscribe("worker", AsyncMock())

        result = await agent.send_task_request(
            recipient="worker",
            task_data={"task": "do_work"},
            wait_for_response=False,
        )
        assert result is None

    async def test_handle_message_routes_to_handler(self):
        role = _make_role("handler_test")
        state = _make_state()
        bus = _make_bus()
        agent = ConcreteCollaborativeAgent(role, state, bus, enable_telemetry=False)
        await agent.initialize()

        handler_called = []
        agent.register_message_handler(
            MessageType.BROADCAST,
            lambda m: handler_called.append(m),
        )

        msg = AgentMessage(
            sender="other",
            recipient="handler_test",
            message_type=MessageType.BROADCAST,
            content={"x": 1},
        )
        await agent._handle_message(msg)
        assert len(handler_called) == 1

    async def test_handle_message_no_handler(self):
        """Messages with no registered handler produce a warning but don't crash."""
        role = _make_role("agent_no_handler")
        state = _make_state()
        bus = _make_bus()
        agent = ConcreteCollaborativeAgent(role, state, bus, enable_telemetry=False)

        msg = AgentMessage(
            sender="other",
            recipient="agent_no_handler",
            message_type=MessageType.VOTE,  # no handler
            content={},
        )
        # Should not raise
        await agent._handle_message(msg)

    async def test_handle_response_resolves_pending(self):
        role = _make_role("waiter")
        state = _make_state()
        bus = _make_bus()
        agent = ConcreteCollaborativeAgent(role, state, bus, enable_telemetry=False)
        await agent.initialize()

        # Create a pending future
        future = asyncio.get_event_loop().create_future()
        agent._pending_replies["msg-123"] = future

        response = AgentMessage(
            sender="responder",
            recipient="waiter",
            message_type=MessageType.RESPONSE,
            content={"answer": 42},
            reply_to="msg-123",
        )
        await agent._handle_response(response)
        assert future.done()
        assert future.result().content == {"answer": 42}

    async def test_handle_error_sets_exception(self):
        role = _make_role("waiter")
        state = _make_state()
        bus = _make_bus()
        agent = ConcreteCollaborativeAgent(role, state, bus, enable_telemetry=False)

        future = asyncio.get_event_loop().create_future()
        agent._pending_replies["msg-456"] = future

        err_msg = AgentMessage(
            sender="responder",
            recipient="waiter",
            message_type=MessageType.ERROR,
            content={"error": "bad request"},
            reply_to="msg-456",
        )
        await agent._handle_error(err_msg)
        assert future.done()
        with pytest.raises(Exception, match="bad request"):
            future.result()

    async def test_handle_error_string_content(self):
        role = _make_role("waiter")
        state = _make_state()
        bus = _make_bus()
        agent = ConcreteCollaborativeAgent(role, state, bus, enable_telemetry=False)

        future = asyncio.get_event_loop().create_future()
        agent._pending_replies["msg-789"] = future

        err_msg = AgentMessage(
            sender="responder",
            recipient="waiter",
            message_type=MessageType.ERROR,
            content="plain error string",
            reply_to="msg-789",
        )
        await agent._handle_error(err_msg)
        assert future.done()
        with pytest.raises(Exception, match="plain error string"):
            future.result()

    async def test_wait_for_reply_timeout(self):
        role = _make_role("waiter")
        state = _make_state()
        bus = _make_bus()
        agent = ConcreteCollaborativeAgent(role, state, bus, enable_telemetry=False)

        with pytest.raises(asyncio.TimeoutError):
            await agent.wait_for_reply("nonexistent-msg", timeout=0.1)

    async def test_wait_for_message_timeout(self):
        role = _make_role("waiter")
        state = _make_state()
        bus = _make_bus()
        agent = ConcreteCollaborativeAgent(role, state, bus, enable_telemetry=False)

        result = await agent.wait_for_message(timeout=0.1)
        assert result is None

    async def test_wait_for_message_filter_sender(self):
        role = _make_role("waiter")
        state = _make_state()
        bus = _make_bus()
        agent = ConcreteCollaborativeAgent(role, state, bus, enable_telemetry=False)

        # Enqueue a message from wrong sender and then the right one
        wrong = AgentMessage(
            sender="wrong",
            recipient="waiter",
            message_type=MessageType.RESPONSE,
            content="nope",
        )
        right = AgentMessage(
            sender="right",
            recipient="waiter",
            message_type=MessageType.RESPONSE,
            content="yes",
        )
        await agent._message_queue.put(wrong)
        await agent._message_queue.put(right)

        msg = await agent.wait_for_message(from_agent="right", timeout=1.0)
        assert msg is not None
        assert msg.sender == "right"

    async def test_wait_for_message_filter_type(self):
        role = _make_role("waiter")
        state = _make_state()
        bus = _make_bus()
        agent = ConcreteCollaborativeAgent(role, state, bus, enable_telemetry=False)

        wrong_type = AgentMessage(
            sender="s",
            recipient="waiter",
            message_type=MessageType.QUERY,
            content="q",
        )
        right_type = AgentMessage(
            sender="s",
            recipient="waiter",
            message_type=MessageType.RESPONSE,
            content="r",
        )
        await agent._message_queue.put(wrong_type)
        await agent._message_queue.put(right_type)

        msg = await agent.wait_for_message(message_type=MessageType.RESPONSE, timeout=1.0)
        assert msg is not None
        assert msg.message_type == MessageType.RESPONSE

    async def test_get_pending_messages(self):
        role = _make_role("agent_pending")
        state = _make_state()
        bus = _make_bus()
        agent = ConcreteCollaborativeAgent(role, state, bus, enable_telemetry=False)

        m1 = AgentMessage(sender="a", message_type=MessageType.QUERY, content="1")
        m2 = AgentMessage(sender="b", message_type=MessageType.QUERY, content="2")
        await agent._message_queue.put(m1)
        await agent._message_queue.put(m2)

        msgs = await agent.get_pending_messages()
        assert len(msgs) == 2

    async def test_has_pending_messages(self):
        role = _make_role("agent_has")
        state = _make_state()
        bus = _make_bus()
        agent = ConcreteCollaborativeAgent(role, state, bus, enable_telemetry=False)

        assert agent.has_pending_messages() is False
        await agent._message_queue.put(
            AgentMessage(sender="x", message_type=MessageType.QUERY, content="y")
        )
        assert agent.has_pending_messages() is True

    async def test_collaborate_with(self):
        state = _make_state()
        bus = _make_bus()

        worker = ConcreteCollaborativeAgent(
            _make_role("worker"), state, bus, enable_telemetry=False,
        )
        helper = ConcreteCollaborativeAgent(
            _make_role("helper"), state, bus, enable_telemetry=False,
        )
        await worker.initialize()
        await helper.initialize()

        # collaborate_with sends task request and waits; we simulate a
        # reply arriving within timeout
        async def _reply_soon():
            await asyncio.sleep(0.05)
            # Get the task request that was sent
            msgs = bus.get_message_history("worker")
            for m in msgs:
                if m.message_type == MessageType.TASK_REQUEST:
                    reply = AgentMessage(
                        sender="helper",
                        recipient="worker",
                        message_type=MessageType.TASK_RESPONSE,
                        content={"done": True},
                        reply_to=m.message_id,
                    )
                    # Deliver via the agent's response handler
                    await worker._handle_response(reply)

        asyncio.create_task(_reply_soon())
        result = await worker.collaborate_with("helper", {"task": "help me"})
        assert result == {"done": True}

    async def test_repr(self):
        role = _make_role("repr_test")
        state = _make_state()
        bus = _make_bus()
        agent = ConcreteCollaborativeAgent(role, state, bus, enable_telemetry=False)
        r = repr(agent)
        assert "ConcreteCollaborativeAgent" in r
        assert "repr_test" in r

    async def test_handle_task_request(self):
        role = _make_role("task_handler")
        state = _make_state()
        bus = _make_bus()
        agent = ConcreteCollaborativeAgent(role, state, bus, enable_telemetry=False)
        await agent.initialize()

        received_responses = []
        await bus.subscribe("requester", lambda m: received_responses.append(m))

        msg = AgentMessage(
            sender="requester",
            recipient="task_handler",
            message_type=MessageType.TASK_REQUEST,
            content={"task": "test"},
        )
        await agent._handle_task_request(msg)
        await asyncio.sleep(0.05)
        assert len(received_responses) >= 1

    async def test_handle_task_request_error(self):
        """If execute raises, error is sent back."""
        role = _make_role("failing_agent")
        state = _make_state()
        bus = _make_bus()

        class FailingAgent(BaseCollaborativeAgent):
            async def execute(self, input_data):
                raise RuntimeError("boom")

        agent = FailingAgent(role, state, bus, enable_telemetry=False)
        await agent.initialize()

        received = []
        await bus.subscribe("requester", lambda m: received.append(m))

        msg = AgentMessage(
            sender="requester",
            recipient="failing_agent",
            message_type=MessageType.TASK_REQUEST,
            content={},
        )
        await agent._handle_task_request(msg)
        await asyncio.sleep(0.05)
        assert any(m.message_type == MessageType.ERROR for m in received)

    async def test_handle_query_default(self):
        role = _make_role("query_handler")
        state = _make_state()
        bus = _make_bus()
        agent = ConcreteCollaborativeAgent(role, state, bus, enable_telemetry=False)
        await agent.initialize()

        received = []
        await bus.subscribe("asker", lambda m: received.append(m))

        msg = AgentMessage(
            sender="asker",
            recipient="query_handler",
            message_type=MessageType.QUERY,
            content={"q": "what"},
        )
        await agent._handle_query(msg)
        await asyncio.sleep(0.05)
        assert len(received) >= 1

    async def test_handle_message_handler_exception_sends_error(self):
        """If a handler raises, an error message is sent to the sender."""
        role = _make_role("err_agent")
        state = _make_state()
        bus = _make_bus()
        agent = ConcreteCollaborativeAgent(role, state, bus, enable_telemetry=False)
        await agent.initialize()

        received = []
        await bus.subscribe("sender_agent", lambda m: received.append(m))

        # Register a handler that raises
        async def bad_handler(m):
            raise ValueError("handler crashed")

        agent.register_message_handler(MessageType.QUERY, bad_handler)

        msg = AgentMessage(
            sender="sender_agent",
            recipient="err_agent",
            message_type=MessageType.QUERY,
            content={},
        )
        await agent._handle_message(msg)
        await asyncio.sleep(0.1)
        assert any(m.message_type == MessageType.ERROR for m in received)

    async def test_default_message_bus_created(self):
        role = _make_role("no_bus")
        state = _make_state()
        agent = ConcreteCollaborativeAgent(role, state, enable_telemetry=False)
        assert agent.message_bus is not None


# ---------------------------------------------------------------------------
# ResearchAgent
# ---------------------------------------------------------------------------

class TestResearchAgent:
    def _make_agent(self):
        role = _make_role("researcher")
        state = _make_state()
        bus = _make_bus()
        return ResearchAgent(role, state, bus, enable_telemetry=False), state, bus

    async def test_execute_with_topic(self):
        agent, state, bus = self._make_agent()
        result = await agent.execute({"topic": "AI safety"})
        assert result["status"] == "success"
        assert result["findings_count"] == 3
        assert 0 < result["confidence"] <= 1.0

        # Check state was updated
        findings = await state.get("research_findings")
        assert findings["topic"] == "AI safety"

    async def test_execute_topic_from_state(self):
        agent, state, bus = self._make_agent()
        await state.set("research_topic", "quantum computing")
        result = await agent.execute({})
        assert result["status"] == "success"

    async def test_execute_no_topic(self):
        agent, state, bus = self._make_agent()
        result = await agent.execute({})
        assert result["status"] == "error"

    async def test_verify_with_peers_no_peers(self):
        agent, state, bus = self._make_agent()
        findings = {"facts": ["fact1", "fact2"]}
        result = await agent._verify_with_peers(findings)
        assert len(result) == 2
        assert all(not f["verified"] for f in result)

    async def test_verify_with_peers_has_peers(self):
        agent, state, bus = self._make_agent()
        await agent.initialize()

        # Register a peer researcher
        peer_received = []

        async def peer_handler(m):
            peer_received.append(m)
            # Send response back
            reply = AgentMessage(
                sender="research_peer",
                recipient="researcher",
                message_type=MessageType.RESPONSE,
                content={"confidence": 0.9},
                reply_to=m.message_id,
            )
            await agent._handle_response(reply)

        await bus.subscribe("research_peer", peer_handler)

        findings = {"facts": ["fact1"]}
        result = await agent._verify_with_peers(findings)
        assert len(result) == 1
        assert result[0]["verified"] is True
        assert result[0]["confidence"] == 0.9

    async def test_verify_with_peers_error(self):
        agent, state, bus = self._make_agent()
        await agent.initialize()

        # Register a peer that will cause a timeout
        await bus.subscribe("research_peer", AsyncMock())

        # Monkeypatch send_query to raise
        agent.send_query = AsyncMock(side_effect=Exception("timeout"))

        findings = {"facts": ["fact1"]}
        result = await agent._verify_with_peers(findings)
        assert len(result) == 1
        assert result[0]["verified"] is False

    async def test_calculate_confidence_empty(self):
        agent, _, _ = self._make_agent()
        assert agent._calculate_confidence([]) == 0.0

    async def test_calculate_confidence(self):
        agent, _, _ = self._make_agent()
        findings = [{"confidence": 0.8}, {"confidence": 0.6}]
        assert agent._calculate_confidence(findings) == pytest.approx(0.7)

    async def test_create_summary_empty(self):
        agent, _, _ = self._make_agent()
        assert "No findings" in agent._create_summary([])

    async def test_create_summary(self):
        agent, _, _ = self._make_agent()
        findings = [{"fact": "a"}, {"fact": "b"}]
        summary = agent._create_summary(findings)
        assert "- a" in summary
        assert "- b" in summary


# ---------------------------------------------------------------------------
# AnalysisAgent
# ---------------------------------------------------------------------------

class TestAnalysisAgent:
    def _make_agent(self):
        role = _make_role("analyzer")
        state = _make_state()
        bus = _make_bus()
        return AnalysisAgent(role, state, bus, enable_telemetry=False), state, bus

    async def test_execute_with_numeric_data(self):
        agent, state, bus = self._make_agent()
        result = await agent.execute({"data": [1, 2, 3, 4, 5]})
        assert result["status"] == "success"
        assert result["patterns_found"] >= 1

        analysis = await state.get("analysis_results")
        assert analysis is not None

    async def test_execute_data_from_state(self):
        agent, state, bus = self._make_agent()
        await state.set("data_to_analyze", [10, 20, 30])
        result = await agent.execute({})
        assert result["status"] == "success"

    async def test_execute_no_data(self):
        agent, state, bus = self._make_agent()
        result = await agent.execute({})
        assert result["status"] == "error"

    async def test_perform_analysis_numeric(self):
        agent, _, _ = self._make_agent()
        result = await agent._perform_analysis([10, 20, 30])
        assert len(result["patterns"]) >= 1
        assert result["data_type"] == "list"
        assert result["data_size"] == 3
        # Check trend
        trends = [p for p in result["patterns"] if p["type"] == "trend"]
        assert len(trends) == 1
        assert trends[0]["value"] == "increasing"

    async def test_perform_analysis_decreasing(self):
        agent, _, _ = self._make_agent()
        result = await agent._perform_analysis([30, 20, 10])
        trends = [p for p in result["patterns"] if p["type"] == "trend"]
        assert trends[0]["value"] == "decreasing"

    async def test_perform_analysis_single_element(self):
        agent, _, _ = self._make_agent()
        result = await agent._perform_analysis([42])
        # Only average, no trend for single element
        assert len(result["patterns"]) == 1
        assert result["patterns"][0]["type"] == "average"

    async def test_perform_analysis_non_numeric(self):
        agent, _, _ = self._make_agent()
        result = await agent._perform_analysis(["a", "b", "c"])
        assert result["patterns"] == []

    async def test_get_peer_insights_no_peers(self):
        agent, _, _ = self._make_agent()
        insights = await agent._get_peer_insights([1, 2], {})
        assert insights == []

    async def test_get_peer_insights_with_peers(self):
        agent, state, bus = self._make_agent()
        await agent.initialize()

        async def peer_handler(m):
            reply = AgentMessage(
                sender="analysis_peer",
                recipient="analyzer",
                message_type=MessageType.RESPONSE,
                content={"insight": "looks good", "agreement": 0.9},
                reply_to=m.message_id,
            )
            await agent._handle_response(reply)

        await bus.subscribe("analysis_peer", peer_handler)

        insights = await agent._get_peer_insights([], {"patterns": []})
        assert len(insights) == 1
        assert insights[0]["agreement"] == 0.9

    async def test_get_peer_insights_error(self):
        agent, state, bus = self._make_agent()
        await agent.initialize()
        await bus.subscribe("analysis_peer", AsyncMock())
        agent.send_query = AsyncMock(side_effect=Exception("fail"))

        insights = await agent._get_peer_insights([], {})
        assert insights == []

    async def test_calculate_analysis_confidence_no_peers(self):
        agent, _, _ = self._make_agent()
        assert agent._calculate_analysis_confidence({}, []) == 0.75

    async def test_calculate_analysis_confidence_with_peers(self):
        agent, _, _ = self._make_agent()
        insights = [{"agreement": 0.9}, {"agreement": 0.8}]
        conf = agent._calculate_analysis_confidence({}, insights)
        assert conf > 0.75
        assert conf <= 0.95

    async def test_generate_recommendations_no_patterns(self):
        agent, _, _ = self._make_agent()
        recs = agent._generate_recommendations({"patterns": []})
        assert "Gather more data" in recs[0]

    async def test_generate_recommendations_with_trend(self):
        agent, _, _ = self._make_agent()
        analysis = {"patterns": [{"type": "trend", "value": "increasing"}]}
        recs = agent._generate_recommendations(analysis)
        assert any("Monitor" in r for r in recs)


# ---------------------------------------------------------------------------
# SynthesisAgent
# ---------------------------------------------------------------------------

class TestSynthesisAgent:
    def _make_agent(self):
        role = _make_role("synthesizer")
        state = _make_state()
        bus = _make_bus()
        return SynthesisAgent(role, state, bus, enable_telemetry=False), state, bus

    async def test_execute_empty(self):
        agent, state, bus = self._make_agent()
        result = await agent.execute({})
        assert result["status"] == "success"

    async def test_execute_with_state_data(self):
        agent, state, bus = self._make_agent()
        await state.set("research_findings", {
            "findings": [{"fact": "fact1"}, {"fact": "fact2"}],
        })
        await state.set("analysis_results", {
            "patterns": [{"description": "pattern1"}],
        })
        result = await agent.execute({"additional_data": {"extra": True}})
        assert result["status"] == "success"
        assert result["sources_used"] >= 2

        output = await state.get("synthesis_output")
        assert len(output["key_points"]) == 3

    async def test_synthesize_sources_with_research(self):
        agent, _, _ = self._make_agent()
        sources = [
            {"type": "research", "data": {"findings": [{"fact": "f1"}, {"fact": "f2"}]}},
            {"type": "analysis", "data": {"patterns": [{"description": "p1"}]}},
            {"type": "additional", "data": {}},
        ]
        result = await agent._synthesize_sources(sources)
        assert len(result["key_points"]) == 3
        assert result["sources_count"] == 2

    async def test_synthesize_sources_non_dict_findings(self):
        agent, _, _ = self._make_agent()
        sources = [
            {"type": "research", "data": {"findings": ["simple string finding"]}},
        ]
        result = await agent._synthesize_sources(sources)
        assert "simple string finding" in result["key_points"]

    async def test_create_synthesis_summary_empty(self):
        agent, _, _ = self._make_agent()
        assert "No information" in agent._create_synthesis_summary([])

    async def test_create_synthesis_summary(self):
        agent, _, _ = self._make_agent()
        points = [f"point_{i}" for i in range(12)]
        summary = agent._create_synthesis_summary(points)
        assert "1. point_0" in summary
        assert "10. point_9" in summary
        assert "2 more points" in summary

    async def test_refine_with_feedback_no_critics(self):
        agent, _, _ = self._make_agent()
        synthesis = {"summary": "test", "key_points": []}
        result = await agent._refine_with_feedback(synthesis)
        assert result is synthesis  # unchanged

    async def test_refine_with_feedback_critic_available(self):
        agent, state, bus = self._make_agent()
        await agent.initialize()

        async def critic_handler(m):
            reply = AgentMessage(
                sender="critic_agent",
                recipient="synthesizer",
                message_type=MessageType.RESPONSE,
                content={"feedback": {"suggestions": ["add more detail"]}},
                reply_to=m.message_id,
            )
            await agent._handle_response(reply)

        await bus.subscribe("critic_agent", critic_handler)

        synthesis = {"summary": "test"}
        result = await agent._refine_with_feedback(synthesis)
        assert result.get("refined") is True
        assert "add more detail" in result["refinements"]

    async def test_refine_with_feedback_critic_error(self):
        agent, state, bus = self._make_agent()
        await agent.initialize()
        await bus.subscribe("critic_agent", AsyncMock())
        agent.send_query = AsyncMock(side_effect=Exception("fail"))

        synthesis = {"summary": "test"}
        result = await agent._refine_with_feedback(synthesis)
        # Should return original synthesis without crash
        assert result["summary"] == "test"


# ---------------------------------------------------------------------------
# CriticAgent
# ---------------------------------------------------------------------------

class TestCriticAgent:
    def _make_agent(self, criteria=None):
        role = _make_role("critic", metadata={"criteria": ["accuracy", "completeness"]})
        state = _make_state()
        bus = _make_bus()
        return CriticAgent(role, state, bus, criteria=criteria), state, bus

    async def test_init_criteria_from_role(self):
        agent, _, _ = self._make_agent()
        assert agent.criteria == ["accuracy", "completeness"]

    async def test_init_criteria_override(self):
        agent, _, _ = self._make_agent(criteria=["clarity"])
        assert agent.criteria == ["clarity"]

    async def test_init_default_criteria(self):
        role = AgentRole(name="critic", description="d")
        state = _make_state()
        bus = _make_bus()
        agent = CriticAgent(role, state, bus, enable_telemetry=False)
        assert "accuracy" in agent.criteria

    async def test_execute_approved(self):
        agent, state, bus = self._make_agent(criteria=["clarity"])
        result = await agent.execute({"artifact": {"key_points": ["p1"], "confidence": 0.9}})
        assert result["status"] == "success"
        assert result["approved"] is True
        assert result["issues_found"] == 0

    async def test_execute_completeness_issue(self):
        agent, state, _ = self._make_agent()
        result = await agent.execute({"artifact": {"confidence": 0.9}})
        assert result["approved"] is False
        assert result["issues_found"] >= 1

    async def test_execute_accuracy_issue(self):
        agent, state, _ = self._make_agent(criteria=["accuracy"])
        result = await agent.execute({"artifact": {"confidence": 0.5}})
        assert result["approved"] is False

    async def test_execute_no_artifact(self):
        agent, state, _ = self._make_agent()
        result = await agent.execute({})
        assert result["status"] == "error"

    async def test_execute_artifact_from_state(self):
        agent, state, _ = self._make_agent(criteria=["clarity"])
        await state.set("synthesis_output", {"text": "some output"})
        result = await agent.execute({})
        assert result["status"] == "success"

    async def test_execute_custom_artifact_key(self):
        agent, state, _ = self._make_agent(criteria=["clarity"])
        await state.set("custom_key", {"text": "output"})
        result = await agent.execute({"artifact_key": "custom_key"})
        assert result["status"] == "success"

    async def test_check_criterion_non_dict(self):
        agent, _, _ = self._make_agent()
        result = agent._check_criterion("just a string", "completeness")
        assert result is None

    async def test_check_criterion_unknown(self):
        agent, _, _ = self._make_agent()
        result = agent._check_criterion({"data": "x"}, "unknown_criterion")
        assert result is None

    async def test_deduplicate_issues(self):
        agent, _, _ = self._make_agent()
        issues = [
            {"criterion": "a", "issue": "x"},
            {"criterion": "a", "issue": "x"},  # dup
            {"criterion": "b", "issue": "y"},
        ]
        result = agent._deduplicate_issues(issues)
        assert len(result) == 2

    async def test_generate_feedback_empty(self):
        agent, _, _ = self._make_agent()
        feedback = agent._generate_feedback([])
        assert "Approved" in feedback

    async def test_generate_feedback_with_issues(self):
        agent, _, _ = self._make_agent()
        issues = [{"severity": "high", "criterion": "accuracy", "issue": "low conf"}]
        feedback = agent._generate_feedback(issues)
        assert "HIGH" in feedback
        assert "accuracy" in feedback

    async def test_build_consensus_no_other_critics(self):
        agent, state, bus = self._make_agent()
        issues = [{"criterion": "a", "issue": "x"}]
        result = await agent._build_consensus({"data": "x"}, issues)
        assert len(result["issues"]) == 1
        assert result["reviewers_count"] == 1

    async def test_build_consensus_with_other_critics(self):
        agent, state, bus = self._make_agent()
        await agent.initialize()

        async def other_critic_handler(m):
            reply = AgentMessage(
                sender="critic_peer",
                recipient="critic",
                message_type=MessageType.RESPONSE,
                content={"issues": [{"criterion": "b", "issue": "y"}]},
                reply_to=m.message_id,
            )
            await agent._handle_response(reply)

        await bus.subscribe("critic_peer", other_critic_handler)

        result = await agent._build_consensus({"data": "x"}, [{"criterion": "a", "issue": "x"}])
        assert len(result["issues"]) == 2

    async def test_handle_query_verify(self):
        agent, state, bus = self._make_agent()
        await agent.initialize()

        received = []
        await bus.subscribe("asker", lambda m: received.append(m))

        msg = AgentMessage(
            sender="asker",
            recipient="critic",
            message_type=MessageType.QUERY,
            content={"action": "verify", "fact": "Earth is round"},
        )
        await agent._handle_query(msg)
        await asyncio.sleep(0.05)
        assert any(m.content.get("verified") for m in received)

    async def test_handle_query_review_analysis(self):
        agent, state, bus = self._make_agent()
        await agent.initialize()

        received = []
        await bus.subscribe("asker", lambda m: received.append(m))

        msg = AgentMessage(
            sender="asker",
            recipient="critic",
            message_type=MessageType.QUERY,
            content={"action": "review_analysis", "analysis": {}},
        )
        await agent._handle_query(msg)
        await asyncio.sleep(0.05)
        assert any(m.content.get("agreement") for m in received)

    async def test_handle_query_review(self):
        agent, state, bus = self._make_agent()
        await agent.initialize()

        received = []
        await bus.subscribe("asker", lambda m: received.append(m))

        msg = AgentMessage(
            sender="asker",
            recipient="critic",
            message_type=MessageType.QUERY,
            content={"action": "review", "artifact": {"confidence": 0.5}},
        )
        await agent._handle_query(msg)
        await asyncio.sleep(0.05)
        assert len(received) >= 1

    async def test_handle_query_review_content_key(self):
        agent, state, bus = self._make_agent()
        await agent.initialize()

        received = []
        await bus.subscribe("asker", lambda m: received.append(m))

        msg = AgentMessage(
            sender="asker",
            recipient="critic",
            message_type=MessageType.QUERY,
            content={"action": "review", "content": {"text": "stuff"}},
        )
        await agent._handle_query(msg)
        await asyncio.sleep(0.05)
        assert len(received) >= 1

    async def test_handle_query_unknown_action(self):
        agent, state, bus = self._make_agent()
        await agent.initialize()

        received = []
        await bus.subscribe("asker", lambda m: received.append(m))

        msg = AgentMessage(
            sender="asker",
            recipient="critic",
            message_type=MessageType.QUERY,
            content={"action": "unknown_action"},
        )
        await agent._handle_query(msg)
        await asyncio.sleep(0.05)
        # Should fall through to parent handler
        assert len(received) >= 1


# ---------------------------------------------------------------------------
# SubprocessExecutor
# ---------------------------------------------------------------------------

class TestSubprocessExecutor:
    def test_init_no_bridge(self):
        exe = SubprocessExecutor()
        assert exe.bridge_dir is None
        assert exe.max_concurrent == 4

    def test_init_with_bridge(self):
        exe = SubprocessExecutor(bridge_dir="/tmp/bridge", node_path="/usr/bin/node")
        assert exe.bridge_dir is not None
        from pathlib import Path
        assert exe.bridge_dir == Path("/tmp/bridge")
        assert exe.node == "/usr/bin/node"

    def test_build_prompt_no_history(self):
        exe = SubprocessExecutor()
        config = _make_config(task="hello")
        assert exe._build_prompt(config) == "hello"

    def test_build_prompt_with_history(self):
        exe = SubprocessExecutor()
        config = _make_config(
            task="current question",
            chat_history=[
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "response"},
            ],
        )
        prompt = exe._build_prompt(config)
        assert "Previous conversation:" in prompt
        assert "User: first" in prompt
        assert "Assistant: response" in prompt
        assert "Current request:" in prompt
        assert "current question" in prompt

    def test_build_prompt_empty_history(self):
        exe = SubprocessExecutor()
        config = _make_config(task="task", chat_history=[])
        assert exe._build_prompt(config) == "task"

    async def test_cancel_no_process(self):
        exe = SubprocessExecutor()
        result = await exe.cancel("nonexistent")
        assert result is False

    async def test_cancel_running_process(self):
        exe = SubprocessExecutor()
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.pid = 12345
        mock_proc.kill = MagicMock()
        exe._running["job-1"] = mock_proc
        result = await exe.cancel("job-1")
        assert result is True
        mock_proc.kill.assert_called_once()

    async def test_cancel_finished_process(self):
        exe = SubprocessExecutor()
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        exe._running["job-1"] = mock_proc
        result = await exe.cancel("job-1")
        assert result is False

    async def test_execute_inner_routes_to_bridge(self):
        exe = SubprocessExecutor(bridge_dir="/tmp/bridge", node_path="/usr/bin/node")
        config = _make_config()
        exe._run_via_bridge = AsyncMock(return_value=iter([]))

        # Make it an async generator
        async def mock_bridge(c):
            return
            yield  # make it an async generator

        exe._run_via_bridge = mock_bridge

        events = []
        async for e in exe._execute_inner(config):
            events.append(e)
        # No events from empty generator
        assert events == []

    async def test_execute_inner_routes_to_direct(self):
        exe = SubprocessExecutor()  # no bridge_dir
        config = _make_config()

        async def mock_direct(c):
            return
            yield

        exe._run_direct = mock_direct

        events = []
        async for e in exe._execute_inner(config):
            events.append(e)
        assert events == []

    async def test_execute_yields_stream_end(self):
        """Full execute should always yield stream_end at the end."""
        exe = SubprocessExecutor()
        config = _make_config()

        async def mock_inner(c):
            yield AgentEvent(type=EventType.TEXT, text="hello")

        exe._execute_inner = mock_inner

        events = []
        async for e in exe.execute(config):
            events.append(e)

        assert len(events) == 2
        assert events[0].type == EventType.TEXT
        assert events[0].seq == 1
        assert events[0].job_id == config.job_id
        assert events[1].type == EventType.SYSTEM
        assert events[1].subtype == "stream_end"

    async def test_execute_captures_result_text(self):
        exe = SubprocessExecutor()
        config = _make_config()

        async def mock_inner(c):
            yield AgentEvent(type=EventType.TEXT, text="answer")
            yield AgentEvent(type=EventType.RESULT, result="final result")

        exe._execute_inner = mock_inner

        events = []
        async for e in exe.execute(config):
            events.append(e)

        stream_end = events[-1]
        assert stream_end.subtype == "stream_end"
        # TEXT takes priority over RESULT
        assert stream_end.result == "answer"

    async def test_execute_result_fallback_when_no_text(self):
        exe = SubprocessExecutor()
        config = _make_config()

        async def mock_inner(c):
            yield AgentEvent(type=EventType.RESULT, result="only result")

        exe._execute_inner = mock_inner

        events = []
        async for e in exe.execute(config):
            events.append(e)

        stream_end = events[-1]
        assert stream_end.result == "only result"

    async def test_execute_stops_on_fatal(self):
        exe = SubprocessExecutor()
        config = _make_config()

        async def mock_inner(c):
            yield AgentEvent(type=EventType.RESULT, subtype="error_agent_exit", error="died")
            yield AgentEvent(type=EventType.TEXT, text="should not appear")

        exe._execute_inner = mock_inner

        events = []
        async for e in exe.execute(config):
            events.append(e)

        # Fatal event + stream_end, but not the second text event
        non_end = [e for e in events if e.subtype != "stream_end"]
        assert len(non_end) == 1
        assert non_end[0].subtype == "error_agent_exit"

    async def test_execute_timeout(self):
        exe = SubprocessExecutor()
        config = _make_config(timeout_seconds=0.1)

        async def mock_inner(c):
            await asyncio.sleep(10)
            yield AgentEvent(type=EventType.TEXT, text="too late")

        exe._execute_inner = mock_inner

        events = []
        async for e in exe.execute(config):
            events.append(e)

        # Should get timeout error + stream_end
        errors = [e for e in events if e.error and "timed out" in e.error]
        assert len(errors) == 1
        assert events[-1].subtype == "stream_end"

    async def test_run_via_bridge_missing_script(self):
        exe = SubprocessExecutor(bridge_dir="/nonexistent/path", node_path="/usr/bin/node")
        config = _make_config()
        with pytest.raises(FileNotFoundError, match="Bridge script not found"):
            async for _ in exe._run_via_bridge(config):
                pass

    async def test_run_via_bridge_opencode_script(self):
        exe = SubprocessExecutor(bridge_dir="/nonexistent/path", node_path="/usr/bin/node")
        config = _make_config(cli_type=CLIType.OPENCODE)
        with pytest.raises(FileNotFoundError, match="run_agent_opencode"):
            async for _ in exe._run_via_bridge(config):
                pass

    @patch("ia_modules.agents.subprocess_executor._find_executable", return_value=None)
    async def test_run_direct_claude_not_found(self, mock_find):
        exe = SubprocessExecutor()
        config = _make_config(cli_type=CLIType.CLAUDE_CODE)
        with pytest.raises(FileNotFoundError, match="claude CLI not found"):
            async for _ in exe._run_direct(config):
                pass

    @patch("ia_modules.agents.subprocess_executor._find_executable", return_value=None)
    async def test_run_direct_opencode_not_found(self, mock_find):
        exe = SubprocessExecutor()
        config = _make_config(cli_type=CLIType.OPENCODE)
        with pytest.raises(FileNotFoundError, match="opencode CLI not found"):
            async for _ in exe._run_direct(config):
                pass

    @patch("ia_modules.agents.subprocess_executor._find_executable")
    async def test_run_direct_claude_builds_cmd(self, mock_find):
        """Verify command building for direct Claude CLI mode."""
        mock_find.return_value = "/usr/bin/claude"

        exe = SubprocessExecutor()
        config = _make_config(
            cli_type=CLIType.CLAUDE_CODE,
            model="claude-sonnet-4-20250514",
            system_prompt="You are helpful",
            tools=["Read", "Write"],
        )

        # Patch _run_subprocess to capture the cmd
        captured_cmd = []

        async def capture_cmd(cmd, stdin_config, cfg):
            captured_cmd.extend(cmd)
            return
            yield

        exe._run_subprocess = capture_cmd

        async for _ in exe._run_direct(config):
            pass

        assert "/usr/bin/claude" in captured_cmd
        assert "--model" in captured_cmd
        assert "claude-sonnet-4-20250514" in captured_cmd
        assert "--system-prompt" in captured_cmd
        assert "--allowedTools" in captured_cmd

    @patch("ia_modules.agents.subprocess_executor._find_executable")
    async def test_run_direct_opencode_builds_cmd(self, mock_find):
        mock_find.return_value = "/usr/bin/opencode"

        exe = SubprocessExecutor()
        config = _make_config(
            cli_type=CLIType.OPENCODE,
            model="gpt-4",
            provider="openai",
        )

        captured_cmd = []

        async def capture_cmd(cmd, stdin_config, cfg):
            captured_cmd.extend(cmd)
            return
            yield

        exe._run_subprocess = capture_cmd

        async for _ in exe._run_direct(config):
            pass

        assert "/usr/bin/opencode" in captured_cmd
        assert "-m" in captured_cmd
        # Provider prefix
        assert "openai/gpt-4" in captured_cmd

    @patch("ia_modules.agents.subprocess_executor._find_executable")
    async def test_run_direct_opencode_model_already_prefixed(self, mock_find):
        mock_find.return_value = "/usr/bin/opencode"

        exe = SubprocessExecutor()
        config = _make_config(
            cli_type=CLIType.OPENCODE,
            model="openai/gpt-4",
            provider="openai",
        )

        captured_cmd = []

        async def capture_cmd(cmd, stdin_config, cfg):
            captured_cmd.extend(cmd)
            return
            yield

        exe._run_subprocess = capture_cmd

        async for _ in exe._run_direct(config):
            pass

        # Should not double-prefix
        assert "openai/gpt-4" in captured_cmd
        assert "openai/openai/gpt-4" not in captured_cmd

    async def test_run_via_bridge_stdin_config(self):
        """Verify stdin config building for bridge mode."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake bridge script
            script_path = os.path.join(tmpdir, "run_agent.mjs")
            with open(script_path, "w") as f:
                f.write("// fake script")

            exe = SubprocessExecutor(bridge_dir=tmpdir, node_path="/usr/bin/node")
            config = _make_config(
                system_prompt="Be helpful",
                tools=["Read", "Grep"],
                model="claude-sonnet-4-20250514",
                provider="anthropic",
                api_key="sk-test",
                business_id="biz-1",
                agent_id="agent-1",
                docs_dir="/docs",
                task_id="task-1",
            )

            captured_stdin = []

            async def capture_subprocess(cmd, stdin_config, cfg):
                captured_stdin.append(stdin_config)
                return
                yield

            exe._run_subprocess = capture_subprocess

            async for _ in exe._run_via_bridge(config):
                pass

            assert len(captured_stdin) == 1
            sc = captured_stdin[0]
            assert sc["systemPrompt"] == "Be helpful"
            assert sc["tools"] == "Read,Grep"
            assert sc["model"] == "claude-sonnet-4-20250514"
            assert sc["provider"] == "anthropic"
            assert sc["apiKey"] == "sk-test"
            assert sc["businessId"] == "biz-1"
            assert sc["agentId"] == "agent-1"
            assert sc["docsDir"] == "/docs"
            assert sc["taskId"] == "task-1"

    async def test_run_via_bridge_opencode_provider_config(self):
        """OpenCode bridge includes providerConfig."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "run_agent_opencode.mjs")
            with open(script_path, "w") as f:
                f.write("// fake")

            exe = SubprocessExecutor(bridge_dir=tmpdir, node_path="/usr/bin/node")
            config = _make_config(
                cli_type=CLIType.OPENCODE,
                model="gpt-4",
                provider="openai",
                api_key="sk-test",
            )

            captured_stdin = []

            async def capture_subprocess(cmd, stdin_config, cfg):
                captured_stdin.append(stdin_config)
                return
                yield

            exe._run_subprocess = capture_subprocess

            async for _ in exe._run_via_bridge(config):
                pass

            sc = captured_stdin[0]
            assert sc["model"] == "openai/gpt-4"
            assert sc["providerConfig"]["provider"] == "openai"
            assert sc["providerConfig"]["apiKey"] == "sk-test"


# ---------------------------------------------------------------------------
# A2AExecutor
# ---------------------------------------------------------------------------

class TestA2AExecutor:
    def test_init_defaults(self):
        exe = A2AExecutor()
        assert "localhost:3008" in exe.a2a_url
        assert exe.callback_url is None

    def test_init_custom(self):
        exe = A2AExecutor(a2a_url="http://myserver:9000", callback_url="http://cb:8080")
        assert exe.a2a_url == "http://myserver:9000"
        assert exe.callback_url == "http://cb:8080"

    @patch.dict(os.environ, {"A2A_SERVER_URL": "http://env:1234"})
    def test_init_from_env(self):
        exe = A2AExecutor()
        assert exe.a2a_url == "http://env:1234"

    def test_build_payload(self):
        exe = A2AExecutor(callback_url="http://cb:8080")
        config = _make_config(
            task="do stuff",
            agent_id="agent-1",
            system_prompt="be helpful",
            tools=["Read"],
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            api_key="sk-test",
            business_id="biz-1",
            task_id="task-1",
            metadata={"context_id": "ctx-1"},
        )
        payload = exe._build_payload(config)
        assert payload["jsonrpc"] == "2.0"
        assert payload["method"] == "message/send"
        meta = payload["params"]["message"]["metadata"]
        assert meta["agent_id"] == "agent-1"
        assert meta["callback_url"] == "http://cb:8080"
        assert meta["model"] == "claude-sonnet-4-20250514"

    def test_build_payload_with_chat_history(self):
        exe = A2AExecutor()
        config = _make_config(
            task="follow up",
            chat_history=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ],
        )
        payload = exe._build_payload(config)
        text = payload["params"]["message"]["parts"][0]["text"]
        assert "Previous conversation:" in text
        assert "User: hello" in text
        assert "Assistant: hi there" in text
        assert "follow up" in text

    def test_build_payload_empty_chat_history(self):
        exe = A2AExecutor()
        config = _make_config(task="just task", chat_history=[])
        payload = exe._build_payload(config)
        text = payload["params"]["message"]["parts"][0]["text"]
        assert text == "just task"

    @patch("httpx.AsyncClient")
    async def test_execute_success(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        exe = A2AExecutor(a2a_url="http://test:3008")
        config = _make_config()

        events = []
        async for e in exe.execute(config):
            events.append(e)

        assert len(events) == 1
        assert events[0].type == EventType.SYSTEM
        assert events[0].subtype == "submitted"
        assert events[0].metadata["a2a_url"] == "http://test:3008"

    @patch("httpx.AsyncClient")
    async def test_execute_error(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        exe = A2AExecutor()
        config = _make_config()

        events = []
        async for e in exe.execute(config):
            events.append(e)

        assert len(events) == 2
        assert events[0].type == EventType.RESULT
        assert "error" in events[0].error.lower()
        assert events[1].type == EventType.SYSTEM
        assert events[1].subtype == "stream_end"

    @patch("httpx.AsyncClient")
    async def test_cancel_success(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.is_success = True

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        exe = A2AExecutor()
        result = await exe.cancel("job-123")
        assert result is True

    @patch("httpx.AsyncClient")
    async def test_cancel_failure(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("fail"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        exe = A2AExecutor()
        result = await exe.cancel("job-123")
        assert result is False


# ---------------------------------------------------------------------------
# SubprocessExecutor._run_subprocess (integration-ish with real process)
# ---------------------------------------------------------------------------

class TestSubprocessExecutorRunSubprocess:
    async def test_run_subprocess_echo_json(self):
        """Subprocess that outputs JSON yields text events."""
        exe = SubprocessExecutor()
        config = _make_config(cwd=".")

        json_line = json.dumps({"type": "text", "part": {"text": "hello from subprocess"}})

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.pid = 100
        mock_proc.stdout = AsyncMock()
        mock_proc.stdout.readline = AsyncMock(
            side_effect=[json_line.encode() + b"\n", b""]
        )
        mock_proc.stderr = AsyncMock()
        mock_proc.stderr.readline = AsyncMock(return_value=b"")
        mock_proc.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            events = []
            async for e in exe._run_subprocess(
                [sys.executable, "-c", "pass"], None, config
            ):
                events.append(e)

        assert len(events) >= 1
        assert events[0].type == EventType.TEXT

    async def test_run_subprocess_non_json_output(self):
        """Non-JSON lines are silently skipped."""
        exe = SubprocessExecutor()
        config = _make_config(cwd=".")

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.pid = 101
        mock_proc.stdout = AsyncMock()
        mock_proc.stdout.readline = AsyncMock(
            side_effect=[b"not json\n", b""]
        )
        mock_proc.stderr = AsyncMock()
        mock_proc.stderr.readline = AsyncMock(return_value=b"")
        mock_proc.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            events = []
            async for e in exe._run_subprocess(
                [sys.executable, "-c", "pass"], None, config
            ):
                events.append(e)

        # No events from non-JSON output
        assert len(events) == 0

    async def test_run_subprocess_error_exit(self):
        """Process that exits with error code yields error event."""
        exe = SubprocessExecutor()
        config = _make_config(cwd=".")

        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.pid = 102
        mock_proc.stdout = AsyncMock()
        mock_proc.stdout.readline = AsyncMock(return_value=b"")
        mock_proc.stderr = AsyncMock()
        mock_proc.stderr.readline = AsyncMock(
            side_effect=[b"err msg\n", b""]
        )
        mock_proc.wait = AsyncMock(return_value=1)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            events = []
            async for e in exe._run_subprocess(
                [sys.executable, "-c", "pass"], None, config
            ):
                events.append(e)

        error_events = [e for e in events if e.subtype == "error_agent_exit"]
        assert len(error_events) == 1
        assert "code 1" in error_events[0].error

    async def test_run_subprocess_with_stdin(self):
        """Subprocess receives stdin JSON config."""
        exe = SubprocessExecutor()
        config = _make_config(cwd=".")

        json_line = json.dumps({"type": "text", "part": {"text": "hello from stdin"}})

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.pid = 103
        mock_proc.stdin = AsyncMock()
        mock_proc.stdin.write = MagicMock()
        mock_proc.stdin.close = MagicMock()
        mock_proc.stdout = AsyncMock()
        mock_proc.stdout.readline = AsyncMock(
            side_effect=[json_line.encode() + b"\n", b""]
        )
        mock_proc.stderr = AsyncMock()
        mock_proc.stderr.readline = AsyncMock(return_value=b"")
        mock_proc.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            stdin_config = {"task": "hello from stdin"}
            events = []
            async for e in exe._run_subprocess(
                [sys.executable, "-c", "pass"], stdin_config, config
            ):
                events.append(e)

        assert any(e.text == "hello from stdin" for e in events)

    async def test_run_subprocess_interrupt_code(self):
        """Process with returncode -2 reports 'interrupted'."""
        exe = SubprocessExecutor()
        config = _make_config(cwd=".")

        # Simulate by patching -- we test the error message logic directly
        # by using a process that we can control the return code of
        # On Windows, we can't easily get -2 return code, so test via mock
        mock_proc = AsyncMock()
        mock_proc.returncode = -2
        mock_proc.pid = 999
        mock_proc.stdout = AsyncMock()
        mock_proc.stdout.readline = AsyncMock(return_value=b"")
        mock_proc.stderr = AsyncMock()
        mock_proc.stderr.readline = AsyncMock(return_value=b"")
        mock_proc.wait = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            events = []
            async for e in exe._run_subprocess(["fake"], None, config):
                events.append(e)

        error_events = [e for e in events if e.subtype == "error_agent_exit"]
        assert len(error_events) == 1
        assert "interrupted" in error_events[0].error.lower()

    async def test_run_subprocess_killed_code(self):
        """Process with returncode -9 / 137 reports 'canceled'."""
        exe = SubprocessExecutor()
        config = _make_config(cwd=".")

        mock_proc = AsyncMock()
        mock_proc.returncode = -9
        mock_proc.pid = 999
        mock_proc.stdout = AsyncMock()
        mock_proc.stdout.readline = AsyncMock(return_value=b"")
        mock_proc.stderr = AsyncMock()
        mock_proc.stderr.readline = AsyncMock(return_value=b"")
        mock_proc.wait = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            events = []
            async for e in exe._run_subprocess(["fake"], None, config):
                events.append(e)

        error_events = [e for e in events if e.subtype == "error_agent_exit"]
        assert len(error_events) == 1
        assert "canceled" in error_events[0].error.lower()
