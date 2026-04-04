"""Integration tests for agent telemetry"""

import pytest
import asyncio
from ia_modules.agents.core import BaseAgent, AgentRole
from ia_modules.agents.state import StateManager
from ia_modules.agents.communication import MessageBus
from ia_modules.agents.base_agent import BaseCollaborativeAgent
from ia_modules.telemetry.metrics import MetricsCollector
from ia_modules.telemetry.tracing import SimpleTracer
from ia_modules.telemetry.agent_telemetry import AgentTelemetry
from ia_modules.telemetry.integration import configure_agent_telemetry


class SimpleAgent(BaseAgent):
    async def execute(self, input_data):
        value = input_data.get("value", 0)
        await self.write_state("result", value * 2)
        return {"result": value * 2}


class CollabAgent(BaseCollaborativeAgent):
    async def execute(self, input_data):
        return {"status": "done", "input": input_data}


@pytest.fixture
def agent_telemetry():
    collector = MetricsCollector()
    tracer = SimpleTracer()
    return configure_agent_telemetry(collector=collector, tracer=tracer)


class TestAgentTelemetry:

    @pytest.mark.asyncio
    async def test_agent_execution_creates_spans(self, agent_telemetry):
        state = StateManager(thread_id="test")
        role = AgentRole(name="test_agent", description="Test")
        agent = SimpleAgent(role, state)

        with agent_telemetry.trace_agent_execution("test_agent", "test") as ctx:
            result = await agent.execute({"value": 5})
            ctx.set_result(result)

        spans = agent_telemetry.get_spans()
        assert len(spans) >= 1
        agent_spans = [s for s in spans if "agent.test_agent" in s.name]
        assert len(agent_spans) == 1
        assert agent_spans[0].status == "ok"

    @pytest.mark.asyncio
    async def test_agent_execution_records_metrics(self, agent_telemetry):
        state = StateManager(thread_id="test")
        role = AgentRole(name="test_agent", description="Test")
        agent = SimpleAgent(role, state)

        with agent_telemetry.trace_agent_execution("test_agent", "test") as ctx:
            result = await agent.execute({"value": 5})

        metrics = agent_telemetry.get_metrics()
        exec_metrics = [m for m in metrics if "agent_executions" in m.name]
        assert len(exec_metrics) > 0

        duration_metrics = [m for m in metrics if "agent_execution_duration" in m.name]
        assert len(duration_metrics) > 0

    @pytest.mark.asyncio
    async def test_message_telemetry(self, agent_telemetry):
        state = StateManager(thread_id="test")
        bus = MessageBus()

        sender_role = AgentRole(name="sender", description="Sender")
        receiver_role = AgentRole(name="receiver", description="Receiver")

        sender = CollabAgent(sender_role, state, bus)
        receiver = CollabAgent(receiver_role, state, bus)

        await sender.initialize()
        await receiver.initialize()

        with agent_telemetry.trace_message_send("sender", "receiver", "task_request"):
            pass  # Message send logic

        metrics = agent_telemetry.get_metrics()
        msg_metrics = [m for m in metrics if "messages_sent" in m.name]
        assert len(msg_metrics) > 0

        await sender.shutdown()
        await receiver.shutdown()

    @pytest.mark.asyncio
    async def test_state_operation_tracking(self, agent_telemetry):
        agent_telemetry.record_state_operation("agent1", "read")
        agent_telemetry.record_state_operation("agent1", "write")
        agent_telemetry.record_state_operation("agent1", "read")

        metrics = agent_telemetry.get_metrics()
        state_metrics = [m for m in metrics if "state_operations" in m.name]
        assert len(state_metrics) > 0

    def test_disabled_telemetry_noop(self):
        collector = MetricsCollector()
        tracer = SimpleTracer()
        disabled = AgentTelemetry(collector=collector, tracer=tracer, enabled=False)

        with disabled.trace_agent_execution("agent", "role") as ctx:
            ctx.set_result({"result": 1})
            ctx.set_attribute("key", "value")
            ctx.record_iteration()

        assert len(disabled.get_spans()) == 0

    def test_collaboration_trace(self, agent_telemetry):
        with agent_telemetry.trace_collaboration(
            pattern="hierarchical",
            participants=["leader", "worker1", "worker2"]
        ):
            pass

        spans = agent_telemetry.get_spans()
        collab_spans = [s for s in spans if "collaboration.hierarchical" in s.name]
        assert len(collab_spans) == 1
        assert collab_spans[0].status == "ok"

        metrics = agent_telemetry.get_metrics()
        collab_metrics = [m for m in metrics if "collaboration_executions" in m.name]
        assert len(collab_metrics) > 0
