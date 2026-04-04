# IA Modules OpenTelemetry - Complete Implementation Plan

> All file paths relative to `C:\Users\David\Notes\projects\ia_modules\`

---

## Phase 1: Agent Telemetry Instrumentation

### 1.1 Create `ia_modules/telemetry/agent_telemetry.py` (NEW)

Agent-level telemetry class mirroring the existing `PipelineTelemetry` pattern.

```python
"""
Agent Telemetry - Automatic instrumentation for agent execution.

Provides metrics collection and distributed tracing for agents,
message passing, and collaboration patterns.
"""

import time
import logging
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

from .metrics import MetricsCollector
from .tracing import Tracer, SimpleTracer

logger = logging.getLogger(__name__)


class AgentTelemetry:
    """
    Automatic telemetry for agent execution.

    Mirrors PipelineTelemetry but focused on agent-level metrics:
    - Agent execution count, duration, errors
    - Message bus traffic (sent, received, latency)
    - Iteration tracking for feedback loops
    - State operation counts
    """

    def __init__(
        self,
        collector: Optional[MetricsCollector] = None,
        tracer: Optional[Tracer] = None,
        enabled: bool = True
    ):
        self.enabled = enabled
        self.collector = collector or MetricsCollector()
        self.tracer = tracer or SimpleTracer()

        if self.enabled:
            self._setup_metrics()

    def _setup_metrics(self):
        """Set up agent-specific metrics"""

        # --- Agent execution metrics ---
        self.agent_executions = self.collector.counter(
            "agent_executions_total",
            help_text="Total agent executions",
            labels=["agent_name", "agent_role", "status"]
        )

        self.agent_duration = self.collector.histogram(
            "agent_execution_duration_seconds",
            help_text="Agent execution duration in seconds",
            labels=["agent_name", "agent_role"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0]
        )

        self.active_agents = self.collector.gauge(
            "active_agents",
            help_text="Currently executing agents",
            labels=["agent_role"]
        )

        self.agent_errors = self.collector.counter(
            "agent_errors_total",
            help_text="Total agent errors",
            labels=["agent_name", "agent_role", "error_type"]
        )

        self.agent_iterations = self.collector.counter(
            "agent_iterations_total",
            help_text="Total agent iteration loops",
            labels=["agent_name"]
        )

        # --- Message bus metrics ---
        self.messages_sent = self.collector.counter(
            "agent_messages_sent_total",
            help_text="Total messages sent between agents",
            labels=["sender", "recipient", "message_type"]
        )

        self.messages_received = self.collector.counter(
            "agent_messages_received_total",
            help_text="Total messages received by agents",
            labels=["recipient", "message_type"]
        )

        self.message_latency = self.collector.histogram(
            "agent_message_latency_seconds",
            help_text="Message delivery latency",
            labels=["message_type"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        )

        self.pending_messages = self.collector.gauge(
            "agent_pending_messages",
            help_text="Messages pending in agent queues",
            labels=["agent_name"]
        )

        self.message_errors = self.collector.counter(
            "agent_message_errors_total",
            help_text="Message delivery failures",
            labels=["sender", "recipient", "error_type"]
        )

        # --- State operation metrics ---
        self.state_operations = self.collector.counter(
            "agent_state_operations_total",
            help_text="State read/write operations",
            labels=["agent_name", "operation"]  # operation: read, write, snapshot
        )

        # --- Collaboration metrics ---
        self.collaboration_executions = self.collector.counter(
            "agent_collaboration_executions_total",
            help_text="Collaboration pattern executions",
            labels=["pattern", "status"]  # pattern: hierarchical, consensus, debate, peer_to_peer
        )

        self.collaboration_duration = self.collector.histogram(
            "agent_collaboration_duration_seconds",
            help_text="Collaboration pattern duration",
            labels=["pattern"],
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0]
        )

    # ─── Context Managers ────────────────────────────────────────────

    @contextmanager
    def trace_agent_execution(
        self,
        agent_name: str,
        agent_role: str,
        input_data: Optional[Dict[str, Any]] = None,
        parent_span=None
    ):
        """
        Trace an agent execution.

        Usage:
            with agent_telemetry.trace_agent_execution("researcher", "research") as ctx:
                result = await agent.execute(input_data)
                ctx.set_result(result)
        """
        if not self.enabled:
            yield _NoOpAgentContext()
            return

        span = self.tracer.start_span(
            f"agent.{agent_name}.execute",
            attributes={
                "agent.name": agent_name,
                "agent.role": agent_role,
                "agent.type": "execution"
            },
            parent=parent_span
        )

        if input_data:
            span.set_attribute("agent.input_size", len(str(input_data)))

        start_time = time.time()
        self.active_agents.inc(agent_role=agent_role)

        ctx = _AgentExecutionContext(
            agent_name=agent_name,
            agent_role=agent_role,
            span=span,
            telemetry=self
        )

        try:
            yield ctx
            span.set_status("ok")
            self.agent_executions.inc(
                agent_name=agent_name,
                agent_role=agent_role,
                status="success"
            )
        except Exception as e:
            error_type = type(e).__name__
            span.set_status("error", str(e))
            span.set_attribute("error.type", error_type)
            span.set_attribute("error.message", str(e))
            self.agent_executions.inc(
                agent_name=agent_name,
                agent_role=agent_role,
                status="error"
            )
            self.agent_errors.inc(
                agent_name=agent_name,
                agent_role=agent_role,
                error_type=error_type
            )
            raise
        finally:
            duration = time.time() - start_time
            self.agent_duration.observe(
                duration,
                agent_name=agent_name,
                agent_role=agent_role
            )
            span.set_attribute("agent.duration_seconds", duration)
            self.active_agents.dec(agent_role=agent_role)
            self.tracer.end_span(span)

    @contextmanager
    def trace_message_send(
        self,
        sender: str,
        recipient: str,
        message_type: str,
        parent_span=None
    ):
        """Trace a message send operation."""
        if not self.enabled:
            yield _NoOpAgentContext()
            return

        span = self.tracer.start_span(
            f"agent.{sender}.message.send",
            attributes={
                "agent.message.sender": sender,
                "agent.message.recipient": recipient,
                "agent.message.type": message_type
            },
            parent=parent_span
        )

        start_time = time.time()

        try:
            yield span
            span.set_status("ok")
            self.messages_sent.inc(
                sender=sender,
                recipient=recipient,
                message_type=message_type
            )
        except Exception as e:
            span.set_status("error", str(e))
            self.message_errors.inc(
                sender=sender,
                recipient=recipient,
                error_type=type(e).__name__
            )
            raise
        finally:
            latency = time.time() - start_time
            self.message_latency.observe(latency, message_type=message_type)
            self.tracer.end_span(span)

    @contextmanager
    def trace_collaboration(
        self,
        pattern: str,
        participants: Optional[List[str]] = None,
        parent_span=None
    ):
        """Trace a collaboration pattern execution."""
        if not self.enabled:
            yield _NoOpAgentContext()
            return

        span = self.tracer.start_span(
            f"collaboration.{pattern}",
            attributes={
                "collaboration.pattern": pattern,
                "collaboration.participants": ",".join(participants or []),
                "collaboration.participant_count": len(participants or [])
            },
            parent=parent_span
        )

        start_time = time.time()

        try:
            yield span
            span.set_status("ok")
            self.collaboration_executions.inc(pattern=pattern, status="success")
        except Exception as e:
            span.set_status("error", str(e))
            self.collaboration_executions.inc(pattern=pattern, status="error")
            raise
        finally:
            duration = time.time() - start_time
            self.collaboration_duration.observe(duration, pattern=pattern)
            self.tracer.end_span(span)

    # ─── Direct recording methods ────────────────────────────────────

    def record_message_received(self, recipient: str, message_type: str):
        """Record a message received event."""
        if self.enabled:
            self.messages_received.inc(
                recipient=recipient,
                message_type=message_type
            )

    def record_state_operation(self, agent_name: str, operation: str):
        """Record a state read/write/snapshot operation."""
        if self.enabled:
            self.state_operations.inc(
                agent_name=agent_name,
                operation=operation
            )

    def record_iteration(self, agent_name: str):
        """Record an agent iteration."""
        if self.enabled:
            self.agent_iterations.inc(agent_name=agent_name)

    def update_pending_messages(self, agent_name: str, count: int):
        """Update the pending messages gauge for an agent."""
        if self.enabled:
            self.pending_messages.set(float(count), agent_name=agent_name)

    def get_metrics(self):
        """Get all collected metrics."""
        return self.collector.collect_all()

    def get_spans(self, trace_id: Optional[str] = None):
        """Get all spans or spans for a specific trace."""
        if trace_id:
            return self.tracer.get_spans(trace_id)
        return self.tracer.get_spans()


class _AgentExecutionContext:
    """Context object yielded from trace_agent_execution."""

    def __init__(self, agent_name: str, agent_role: str, span, telemetry: AgentTelemetry):
        self.agent_name = agent_name
        self.agent_role = agent_role
        self.span = span
        self.telemetry = telemetry

    def set_result(self, result: Any):
        if result:
            self.span.set_attribute("agent.result_size", len(str(result)))

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        self.span.add_event(name, attributes)

    def set_attribute(self, key: str, value: Any):
        self.span.set_attribute(key, value)

    def record_iteration(self):
        self.telemetry.record_iteration(self.agent_name)


class _NoOpAgentContext:
    """No-op context when telemetry is disabled."""

    def set_result(self, result: Any): pass
    def add_event(self, name: str, attributes=None): pass
    def set_attribute(self, key: str, value: Any): pass
    def record_iteration(self): pass
```

---

### 1.2 Create `ia_modules/telemetry/llm_telemetry.py` (NEW)

LLM call telemetry following OpenTelemetry gen_ai semantic conventions.

```python
"""
LLM Telemetry - OpenTelemetry gen_ai semantic conventions.

Tracks token usage, cost, latency, and model information for all LLM calls.
Follows: https://opentelemetry.io/docs/specs/semconv/gen-ai/
"""

import time
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager

from .metrics import MetricsCollector
from .tracing import Tracer, SimpleTracer

logger = logging.getLogger(__name__)


class LLMTelemetry:
    """
    Telemetry for LLM API calls using OpenTelemetry gen_ai conventions.

    Metrics follow the gen_ai semantic convention naming:
    - gen_ai.client.token.usage
    - gen_ai.client.operation.duration
    """

    def __init__(
        self,
        collector: Optional[MetricsCollector] = None,
        tracer: Optional[Tracer] = None,
        enabled: bool = True
    ):
        self.enabled = enabled
        self.collector = collector or MetricsCollector()
        self.tracer = tracer or SimpleTracer()

        if self.enabled:
            self._setup_metrics()

    def _setup_metrics(self):
        """Set up gen_ai semantic convention metrics."""

        # Token usage histogram
        self.token_usage = self.collector.histogram(
            "gen_ai_client_token_usage",
            help_text="Token usage per LLM request",
            labels=["gen_ai_operation_name", "gen_ai_system", "gen_ai_response_model", "token_type"],
            buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 50000]
        )

        # Operation duration
        self.operation_duration = self.collector.histogram(
            "gen_ai_client_operation_duration_seconds",
            help_text="LLM operation duration in seconds",
            labels=["gen_ai_operation_name", "gen_ai_system", "gen_ai_response_model"],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
        )

        # Request counter
        self.requests = self.collector.counter(
            "gen_ai_client_requests_total",
            help_text="Total LLM API requests",
            labels=["gen_ai_system", "gen_ai_response_model", "status"]
        )

        # Cost tracking
        self.cost = self.collector.counter(
            "gen_ai_client_cost_usd_total",
            help_text="Total LLM cost in USD",
            labels=["gen_ai_system", "gen_ai_response_model"]
        )

        # Tokens per second (throughput)
        self.tokens_per_second = self.collector.histogram(
            "gen_ai_client_tokens_per_second",
            help_text="Token generation throughput",
            labels=["gen_ai_system", "gen_ai_response_model"],
            buckets=[1, 5, 10, 25, 50, 100, 200, 500]
        )

    @contextmanager
    def trace_llm_call(
        self,
        operation: str = "chat",
        system: str = "unknown",
        model: str = "unknown",
        parent_span=None
    ):
        """
        Trace an LLM API call.

        Usage:
            with llm_telemetry.trace_llm_call("chat", "openai", "gpt-4") as ctx:
                response = await litellm.acompletion(...)
                ctx.record_usage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    cost_usd=response._hidden_params.get("response_cost", 0)
                )

        Args:
            operation: gen_ai operation name (chat, embedding, completion)
            system: gen_ai system (openai, anthropic, google, ollama)
            model: model identifier
            parent_span: parent span for nesting
        """
        if not self.enabled:
            yield _NoOpLLMContext()
            return

        span = self.tracer.start_span(
            f"gen_ai.{operation}",
            attributes={
                "gen_ai.operation.name": operation,
                "gen_ai.system": system,
                "gen_ai.request.model": model,
            },
            parent=parent_span
        )

        start_time = time.time()

        ctx = _LLMCallContext(
            operation=operation,
            system=system,
            model=model,
            span=span,
            telemetry=self,
            start_time=start_time
        )

        try:
            yield ctx
            span.set_status("ok")
            self.requests.inc(
                gen_ai_system=system,
                gen_ai_response_model=ctx.response_model or model,
                status="success"
            )
        except Exception as e:
            span.set_status("error", str(e))
            span.set_attribute("error.type", type(e).__name__)
            self.requests.inc(
                gen_ai_system=system,
                gen_ai_response_model=ctx.response_model or model,
                status="error"
            )
            raise
        finally:
            duration = time.time() - start_time
            response_model = ctx.response_model or model
            self.operation_duration.observe(
                duration,
                gen_ai_operation_name=operation,
                gen_ai_system=system,
                gen_ai_response_model=response_model
            )
            span.set_attribute("gen_ai.operation.duration_seconds", duration)
            self.tracer.end_span(span)

    def record_usage_direct(
        self,
        operation: str,
        system: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float = 0.0,
        duration_seconds: float = 0.0
    ):
        """Record LLM usage without span context (for after-the-fact recording)."""
        if not self.enabled:
            return

        total_tokens = prompt_tokens + completion_tokens

        # Token usage
        if prompt_tokens > 0:
            self.token_usage.observe(
                prompt_tokens,
                gen_ai_operation_name=operation,
                gen_ai_system=system,
                gen_ai_response_model=model,
                token_type="prompt"
            )
        if completion_tokens > 0:
            self.token_usage.observe(
                completion_tokens,
                gen_ai_operation_name=operation,
                gen_ai_system=system,
                gen_ai_response_model=model,
                token_type="completion"
            )

        # Cost
        if cost_usd > 0:
            self.cost.inc(
                amount=cost_usd,
                gen_ai_system=system,
                gen_ai_response_model=model
            )

        # Throughput
        if duration_seconds > 0 and completion_tokens > 0:
            tps = completion_tokens / duration_seconds
            self.tokens_per_second.observe(
                tps,
                gen_ai_system=system,
                gen_ai_response_model=model
            )

        # Request count
        self.requests.inc(
            gen_ai_system=system,
            gen_ai_response_model=model,
            status="success"
        )

    def get_metrics(self):
        return self.collector.collect_all()

    def get_spans(self, trace_id=None):
        if trace_id:
            return self.tracer.get_spans(trace_id)
        return self.tracer.get_spans()


class _LLMCallContext:
    """Context yielded from trace_llm_call."""

    def __init__(self, operation, system, model, span, telemetry, start_time):
        self.operation = operation
        self.system = system
        self.model = model
        self.response_model = None
        self.span = span
        self.telemetry = telemetry
        self.start_time = start_time

    def record_usage(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: Optional[int] = None,
        cost_usd: float = 0.0,
        response_model: Optional[str] = None,
        finish_reason: Optional[str] = None
    ):
        """Record token usage and cost from LLM response."""
        if response_model:
            self.response_model = response_model
            self.span.set_attribute("gen_ai.response.model", response_model)

        if finish_reason:
            self.span.set_attribute("gen_ai.response.finish_reason", finish_reason)

        actual_total = total_tokens or (prompt_tokens + completion_tokens)

        self.span.set_attribute("gen_ai.usage.prompt_tokens", prompt_tokens)
        self.span.set_attribute("gen_ai.usage.completion_tokens", completion_tokens)
        self.span.set_attribute("gen_ai.usage.total_tokens", actual_total)
        self.span.set_attribute("gen_ai.usage.cost_usd", cost_usd)

        model = self.response_model or self.model
        duration = time.time() - self.start_time

        self.telemetry.record_usage_direct(
            operation=self.operation,
            system=self.system,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            duration_seconds=duration
        )

    def set_attribute(self, key: str, value: Any):
        self.span.set_attribute(key, value)


class _NoOpLLMContext:
    def record_usage(self, **kwargs): pass
    def set_attribute(self, key, value): pass
```

---

### 1.3 Modify `ia_modules/telemetry/__init__.py`

Add new exports.

```python
# ADD these imports after existing ones:

from .agent_telemetry import AgentTelemetry
from .llm_telemetry import LLMTelemetry

# ADD to __all__ list:
    # Agent Telemetry
    'AgentTelemetry',

    # LLM Telemetry
    'LLMTelemetry',
```

---

### 1.4 Modify `ia_modules/telemetry/integration.py`

Add global agent/LLM telemetry singletons alongside existing `_global_telemetry`.

```python
# ADD after existing imports at top:
from .agent_telemetry import AgentTelemetry
from .llm_telemetry import LLMTelemetry

# ADD after existing _global_telemetry variable (line ~427):
_global_agent_telemetry: Optional[AgentTelemetry] = None
_global_llm_telemetry: Optional[LLMTelemetry] = None


def get_agent_telemetry(
    collector: Optional[MetricsCollector] = None,
    tracer: Optional[Tracer] = None,
    enabled: bool = True
) -> AgentTelemetry:
    """Get or create global agent telemetry instance."""
    global _global_agent_telemetry
    if _global_agent_telemetry is None:
        _global_agent_telemetry = AgentTelemetry(
            collector=collector,
            tracer=tracer,
            enabled=enabled
        )
    return _global_agent_telemetry


def get_llm_telemetry(
    collector: Optional[MetricsCollector] = None,
    tracer: Optional[Tracer] = None,
    enabled: bool = True
) -> LLMTelemetry:
    """Get or create global LLM telemetry instance."""
    global _global_llm_telemetry
    if _global_llm_telemetry is None:
        _global_llm_telemetry = LLMTelemetry(
            collector=collector,
            tracer=tracer,
            enabled=enabled
        )
    return _global_llm_telemetry


def configure_agent_telemetry(
    collector: Optional[MetricsCollector] = None,
    tracer: Optional[Tracer] = None,
    enabled: bool = True
) -> AgentTelemetry:
    """Configure global agent telemetry instance."""
    global _global_agent_telemetry
    _global_agent_telemetry = AgentTelemetry(
        collector=collector, tracer=tracer, enabled=enabled
    )
    return _global_agent_telemetry


def configure_llm_telemetry(
    collector: Optional[MetricsCollector] = None,
    tracer: Optional[Tracer] = None,
    enabled: bool = True
) -> LLMTelemetry:
    """Configure global LLM telemetry instance."""
    global _global_llm_telemetry
    _global_llm_telemetry = LLMTelemetry(
        collector=collector, tracer=tracer, enabled=enabled
    )
    return _global_llm_telemetry
```

Also update the `__init__.py` exports to include:
```python
'get_agent_telemetry',
'get_llm_telemetry',
'configure_agent_telemetry',
'configure_llm_telemetry',
```

---

### 1.5 Modify `ia_modules/agents/core.py`

Instrument `BaseAgent` with optional telemetry.

```python
# ADD import at top:
from ia_modules.telemetry.integration import get_agent_telemetry

# MODIFY BaseAgent.__init__ (line ~59):
class BaseAgent(ABC):
    def __init__(self, role: AgentRole, state_manager: "StateManager",
                 enable_telemetry: bool = True):
        self.role = role
        self.state = state_manager
        self.logger = logging.getLogger(f"Agent.{role.name}")
        self._iteration_count = 0

        # Telemetry
        self.enable_telemetry = enable_telemetry
        self._telemetry = get_agent_telemetry() if enable_telemetry else None

    # WRAP read_state to track state operations:
    async def read_state(self, key: str, default: Any = None) -> Any:
        if self._telemetry:
            self._telemetry.record_state_operation(self.role.name, "read")
        return await self.state.get(key, default)

    async def write_state(self, key: str, value: Any) -> None:
        if self._telemetry:
            self._telemetry.record_state_operation(self.role.name, "write")
        await self.state.set(key, value)

    async def get_state_snapshot(self) -> Dict[str, Any]:
        if self._telemetry:
            self._telemetry.record_state_operation(self.role.name, "snapshot")
        return await self.state.snapshot()

    def increment_iteration(self) -> int:
        self._iteration_count += 1
        if self._telemetry:
            self._telemetry.record_iteration(self.role.name)
        return self._iteration_count
```

---

### 1.6 Modify `ia_modules/agents/base_agent.py`

Instrument `BaseCollaborativeAgent` message passing.

```python
# ADD import at top:
from ia_modules.telemetry.integration import get_agent_telemetry

# MODIFY __init__ (line ~45):
class BaseCollaborativeAgent(BaseAgent):
    def __init__(self, role: AgentRole, state_manager: StateManager,
                 message_bus: Optional[MessageBus] = None,
                 enable_telemetry: bool = True):
        super().__init__(role, state_manager, enable_telemetry=enable_telemetry)
        # ... existing code ...

    # MODIFY _handle_message (line ~108):
    async def _handle_message(self, message: AgentMessage) -> None:
        self.logger.debug(f"Received {message.message_type.value} from {message.sender}")

        # Track received message
        if self._telemetry:
            self._telemetry.record_message_received(
                self.agent_id, message.message_type.value
            )
            self._telemetry.update_pending_messages(
                self.agent_id, self._message_queue.qsize() + 1
            )

        await self._message_queue.put(message)

        handler = self._message_handlers.get(message.message_type)
        if handler:
            try:
                await handler(message)
            except Exception as e:
                self.logger.error(f"Error handling {message.message_type.value}: {e}", exc_info=True)
                if message.sender:
                    await self.send_error(message.sender, str(e), message.message_id)
        else:
            self.logger.warning(f"No handler for message type: {message.message_type.value}")

    # MODIFY send_message (line ~210):
    async def send_message(self, recipient: str, message_type: MessageType,
                          content: Any, reply_to: Optional[str] = None,
                          **kwargs) -> AgentMessage:
        message = AgentMessage(
            sender=self.agent_id,
            recipient=recipient,
            message_type=message_type,
            content=content,
            reply_to=reply_to,
            **kwargs
        )

        # Track with telemetry
        if self._telemetry:
            with self._telemetry.trace_message_send(
                sender=self.agent_id,
                recipient=recipient,
                message_type=message_type.value
            ):
                await self.message_bus.send(message)
        else:
            await self.message_bus.send(message)

        self.logger.debug(f"Sent {message_type.value} to {recipient}")
        return message
```

---

### 1.7 Modify `ia_modules/pipeline/llm_provider_service.py`

Instrument LLM calls with `LLMTelemetry`.

```python
# ADD import at top:
from ia_modules.telemetry.integration import get_llm_telemetry

# MODIFY __init__:
class LLMProviderService:
    def __init__(self, enable_telemetry: bool = True):
        # ... existing init code ...
        self._llm_telemetry = get_llm_telemetry() if enable_telemetry else None

    # MODIFY generate_completion:
    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        provider_name: Optional[str] = None,
        **litellm_params
    ) -> Dict[str, Any]:
        provider = self._get_provider(provider_name)
        model = provider["model"]

        # Determine system from model name
        system = self._detect_system(model)

        if self._llm_telemetry:
            with self._llm_telemetry.trace_llm_call(
                operation="chat",
                system=system,
                model=model
            ) as ctx:
                response = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    **litellm_params
                )

                # Extract usage from litellm response
                usage = response.usage
                cost = response._hidden_params.get("response_cost", 0.0)

                ctx.record_usage(
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                    cost_usd=cost,
                    response_model=response.model,
                    finish_reason=response.choices[0].finish_reason
                )

                return self._format_response(response, provider_name)
        else:
            # Original code path without telemetry
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                **litellm_params
            )
            return self._format_response(response, provider_name)

    def _detect_system(self, model: str) -> str:
        """Detect gen_ai.system from model name."""
        model_lower = model.lower()
        if any(x in model_lower for x in ["gpt", "o1", "o3"]):
            return "openai"
        elif any(x in model_lower for x in ["claude", "anthropic"]):
            return "anthropic"
        elif any(x in model_lower for x in ["gemini", "palm"]):
            return "google"
        elif "ollama" in model_lower or "/" in model:
            return "ollama"
        return "unknown"
```

---

### 1.8 Instrument Collaboration Patterns

Modify `ia_modules/agents/collaboration_patterns/hierarchical.py` (and others):

```python
# ADD import at top:
from ia_modules.telemetry.integration import get_agent_telemetry

# MODIFY HierarchicalCollaboration.execute():
async def execute(self, task_description: Dict[str, Any],
                 strategy: DecompositionStrategy = DecompositionStrategy.PARALLEL) -> Dict[str, Any]:

    telemetry = get_agent_telemetry()
    participants = [self.leader.agent_id] + [w.agent_id for w in self.workers]

    with telemetry.trace_collaboration(
        pattern="hierarchical",
        participants=participants
    ):
        # ... existing execution logic ...
        return result
```

Apply same pattern to `consensus.py`, `debate.py`, `peer_to_peer.py` with their respective pattern names.

---

### 1.9 Create `tests/integration/test_agent_telemetry.py` (NEW)

```python
"""Integration tests for agent telemetry"""

import pytest
import asyncio
from ia_modules.agents import BaseAgent, AgentRole, StateManager, MessageBus
from ia_modules.agents.base_agent import BaseCollaborativeAgent
from ia_modules.telemetry import MetricsCollector, SimpleTracer
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
```

---

### 1.10 Create `tests/integration/test_llm_telemetry.py` (NEW)

```python
"""Integration tests for LLM telemetry"""

import pytest
from ia_modules.telemetry import MetricsCollector, SimpleTracer
from ia_modules.telemetry.llm_telemetry import LLMTelemetry


@pytest.fixture
def llm_telemetry():
    collector = MetricsCollector()
    tracer = SimpleTracer()
    return LLMTelemetry(collector=collector, tracer=tracer, enabled=True)


class TestLLMTelemetry:

    def test_trace_llm_call_creates_span(self, llm_telemetry):
        with llm_telemetry.trace_llm_call("chat", "openai", "gpt-4o") as ctx:
            ctx.record_usage(
                prompt_tokens=150,
                completion_tokens=50,
                cost_usd=0.003,
                response_model="gpt-4o-2024-08-06",
                finish_reason="stop"
            )

        spans = llm_telemetry.get_spans()
        assert len(spans) == 1
        assert spans[0].name == "gen_ai.chat"
        assert spans[0].attributes["gen_ai.system"] == "openai"
        assert spans[0].attributes["gen_ai.usage.prompt_tokens"] == 150
        assert spans[0].attributes["gen_ai.usage.completion_tokens"] == 50

    def test_token_usage_metrics(self, llm_telemetry):
        with llm_telemetry.trace_llm_call("chat", "anthropic", "claude-sonnet-4-20250514") as ctx:
            ctx.record_usage(prompt_tokens=200, completion_tokens=100, cost_usd=0.005)

        metrics = llm_telemetry.get_metrics()
        token_metrics = [m for m in metrics if "token_usage" in m.name]
        assert len(token_metrics) > 0

        cost_metrics = [m for m in metrics if "cost_usd" in m.name]
        assert len(cost_metrics) > 0

    def test_error_tracking(self, llm_telemetry):
        with pytest.raises(ValueError):
            with llm_telemetry.trace_llm_call("chat", "openai", "gpt-4o") as ctx:
                raise ValueError("Rate limit exceeded")

        spans = llm_telemetry.get_spans()
        assert spans[0].status == "error"

        metrics = llm_telemetry.get_metrics()
        req_metrics = [m for m in metrics if "requests_total" in m.name]
        error_reqs = [m for m in req_metrics if m.labels.get("status") == "error"]
        assert len(error_reqs) > 0

    def test_direct_usage_recording(self, llm_telemetry):
        llm_telemetry.record_usage_direct(
            operation="chat",
            system="openai",
            model="gpt-4o-mini",
            prompt_tokens=50,
            completion_tokens=25,
            cost_usd=0.0001,
            duration_seconds=0.5
        )

        metrics = llm_telemetry.get_metrics()
        assert len(metrics) > 0

        tps_metrics = [m for m in metrics if "tokens_per_second" in m.name]
        assert len(tps_metrics) > 0

    def test_multiple_providers(self, llm_telemetry):
        # OpenAI call
        with llm_telemetry.trace_llm_call("chat", "openai", "gpt-4o") as ctx:
            ctx.record_usage(prompt_tokens=100, completion_tokens=50, cost_usd=0.002)

        # Anthropic call
        with llm_telemetry.trace_llm_call("chat", "anthropic", "claude-sonnet-4-20250514") as ctx:
            ctx.record_usage(prompt_tokens=100, completion_tokens=80, cost_usd=0.004)

        # Google call
        with llm_telemetry.trace_llm_call("chat", "google", "gemini-2.0-flash") as ctx:
            ctx.record_usage(prompt_tokens=100, completion_tokens=60, cost_usd=0.001)

        metrics = llm_telemetry.get_metrics()
        req_metrics = [m for m in metrics if "requests_total" in m.name]
        # Should have 3 success entries for 3 different providers
        assert len(req_metrics) >= 3
```

---

## Phase 2: Enhanced OTLP Export & API Endpoints

### 2.1 Modify `showcase_app/backend/services/telemetry_service.py`

Add agent and LLM telemetry data aggregation.

```python
# ADD to TelemetryService class:

class TelemetryService:
    def __init__(self, telemetry=None, tracer=None,
                 agent_telemetry=None, llm_telemetry=None):
        self.telemetry = telemetry
        self.tracer = tracer
        self.agent_telemetry = agent_telemetry
        self.llm_telemetry = llm_telemetry
        logger.info("Telemetry service initialized")

    # --- NEW: Agent metrics ---

    async def get_agent_metrics(self) -> Dict[str, Any]:
        """Get aggregated agent metrics."""
        if not self.agent_telemetry:
            return {"agents": [], "summary": {}}

        metrics = self.agent_telemetry.get_metrics()

        # Group by agent name
        agents = {}
        for m in metrics:
            agent_name = m.labels.get("agent_name", "unknown")
            if agent_name not in agents:
                agents[agent_name] = {
                    "name": agent_name,
                    "role": m.labels.get("agent_role", "unknown"),
                    "executions": 0,
                    "errors": 0,
                    "messages_sent": 0,
                    "messages_received": 0,
                    "state_reads": 0,
                    "state_writes": 0,
                }

            if "executions_total" in m.name and m.labels.get("status") == "success":
                agents[agent_name]["executions"] += m.value
            elif "errors_total" in m.name:
                agents[agent_name]["errors"] += m.value
            elif "messages_sent" in m.name and m.labels.get("sender") == agent_name:
                agents[agent_name]["messages_sent"] += m.value
            elif "messages_received" in m.name:
                agents[agent_name]["messages_received"] += m.value
            elif "state_operations" in m.name:
                op = m.labels.get("operation", "")
                if op == "read":
                    agents[agent_name]["state_reads"] += m.value
                elif op == "write":
                    agents[agent_name]["state_writes"] += m.value

        return {
            "agents": list(agents.values()),
            "summary": {
                "total_agents": len(agents),
                "total_executions": sum(a["executions"] for a in agents.values()),
                "total_errors": sum(a["errors"] for a in agents.values()),
                "total_messages": sum(a["messages_sent"] for a in agents.values()),
            }
        }

    # --- NEW: LLM metrics ---

    async def get_llm_metrics(self) -> Dict[str, Any]:
        """Get aggregated LLM usage metrics."""
        if not self.llm_telemetry:
            return {"models": [], "summary": {}}

        metrics = self.llm_telemetry.get_metrics()

        models = {}
        total_cost = 0.0
        total_requests = 0

        for m in metrics:
            model = m.labels.get("gen_ai_response_model", "unknown")
            system = m.labels.get("gen_ai_system", "unknown")
            key = f"{system}/{model}"

            if key not in models:
                models[key] = {
                    "model": model,
                    "system": system,
                    "requests": 0,
                    "errors": 0,
                    "total_cost_usd": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                }

            if "requests_total" in m.name:
                if m.labels.get("status") == "success":
                    models[key]["requests"] += m.value
                    total_requests += m.value
                elif m.labels.get("status") == "error":
                    models[key]["errors"] += m.value

            elif "cost_usd" in m.name:
                models[key]["total_cost_usd"] += m.value
                total_cost += m.value

        return {
            "models": list(models.values()),
            "summary": {
                "total_requests": total_requests,
                "total_cost_usd": round(total_cost, 4),
                "model_count": len(models),
            }
        }

    # --- NEW: Time-series data ---

    async def get_metrics_timeseries(self, metric_name: str, hours: int = 24) -> List[Dict]:
        """Get time-series data points for a specific metric.

        In production, this queries a time-series DB. For now, collect from
        the in-memory collector with timestamps.
        """
        if not self.telemetry:
            return []

        metrics = self.telemetry.get_metrics()
        points = []
        for m in metrics:
            if metric_name in m.name:
                points.append({
                    "timestamp": m.timestamp,
                    "value": m.value if isinstance(m.value, (int, float)) else 0,
                    "labels": m.labels
                })

        return sorted(points, key=lambda p: p["timestamp"])
```

---

### 2.2 Add new API endpoints to `showcase_app/backend/api/telemetry.py`

```python
# ADD these new endpoints after existing ones:

@router.get("/agents")
async def get_agent_metrics(service=Depends(get_telemetry_service)):
    """Get aggregated agent telemetry metrics."""
    try:
        return await service.get_agent_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/llm/usage")
async def get_llm_usage(service=Depends(get_telemetry_service)):
    """Get LLM usage metrics (tokens, cost, requests by model)."""
    try:
        return await service.get_llm_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/timeseries/{metric_name}")
async def get_metric_timeseries(
    metric_name: str,
    hours: int = 24,
    service=Depends(get_telemetry_service)
):
    """Get time-series data for a specific metric."""
    try:
        return await service.get_metrics_timeseries(metric_name, hours)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

### 2.3 Wire up in `showcase_app/backend/main.py`

```python
# MODIFY service initialization in lifespan (add agent/llm telemetry):
from ia_modules.telemetry.integration import (
    configure_agent_telemetry,
    configure_llm_telemetry
)

# Inside lifespan, after existing telemetry_service setup:
agent_telemetry = configure_agent_telemetry(
    collector=services.pipeline_service.telemetry.collector if services.pipeline_service.telemetry else None,
    tracer=services.pipeline_service.tracer
)
llm_telemetry = configure_llm_telemetry(
    collector=services.pipeline_service.telemetry.collector if services.pipeline_service.telemetry else None,
    tracer=services.pipeline_service.tracer
)

services.telemetry_service = TelemetryService(
    telemetry=services.pipeline_service.telemetry,
    tracer=services.pipeline_service.tracer,
    agent_telemetry=agent_telemetry,
    llm_telemetry=llm_telemetry
)
```

---

## Phase 3: Showcase Dashboard Charts

### 3.1 Add API methods to `showcase_app/frontend/src/services/api.js`

```javascript
// ADD to existing telemetryAPI or create new:

export const telemetryAPI = {
  getSpans: (jobId) => api.get(`/telemetry/spans/${jobId}`),
  getMetrics: (jobId) => api.get(`/telemetry/metrics/${jobId}`),
  getTimeline: (jobId) => api.get(`/telemetry/timeline/${jobId}`),
  // NEW:
  getAgentMetrics: () => api.get('/telemetry/agents'),
  getLLMUsage: () => api.get('/telemetry/llm/usage'),
  getTimeseries: (metric, hours = 24) => api.get(`/telemetry/timeseries/${metric}?hours=${hours}`),
}
```

---

### 3.2 Create `showcase_app/frontend/src/components/charts/MetricsTrendChart.jsx` (NEW)

Replaces the placeholder "Historical metrics visualization will appear here".

```jsx
import React from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer
} from 'recharts'

const METRIC_COLORS = {
  svr: '#22c55e',   // green
  cr: '#eab308',    // yellow
  hir: '#a855f7',   // purple
  ma: '#3b82f6',    // blue
  tcl: '#f97316',   // orange
  wct: '#6366f1',   // indigo
}

export default function MetricsTrendChart({ data, metrics = ['svr', 'cr', 'hir'] }) {
  if (!data || data.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-gray-500">
        <p>No time-series data available yet. Run pipelines to generate data.</p>
      </div>
    )
  }

  // Format timestamps for display
  const formatted = data.map(point => ({
    ...point,
    time: new Date(point.timestamp * 1000).toLocaleTimeString(),
  }))

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={formatted}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="time" fontSize={12} />
        <YAxis domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
        <Tooltip
          formatter={(value, name) => [`${(value * 100).toFixed(1)}%`, name.toUpperCase()]}
          labelFormatter={(label) => `Time: ${label}`}
        />
        <Legend />
        {metrics.map(metric => (
          <Line
            key={metric}
            type="monotone"
            dataKey={metric}
            stroke={METRIC_COLORS[metric] || '#8884d8'}
            strokeWidth={2}
            dot={false}
            name={metric.toUpperCase()}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}
```

---

### 3.3 Create `showcase_app/frontend/src/components/charts/AgentPerformanceChart.jsx` (NEW)

```jsx
import React from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, Cell
} from 'recharts'

const COLORS = ['#3b82f6', '#22c55e', '#f97316', '#a855f7', '#ef4444', '#06b6d4']

export default function AgentPerformanceChart({ agents }) {
  if (!agents || agents.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-gray-500">
        <p>No agent data available.</p>
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={agents} layout="vertical" margin={{ left: 80 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis type="number" />
        <YAxis type="category" dataKey="name" fontSize={12} />
        <Tooltip />
        <Legend />
        <Bar dataKey="executions" name="Executions" fill="#3b82f6" />
        <Bar dataKey="errors" name="Errors" fill="#ef4444" />
        <Bar dataKey="messages_sent" name="Messages Sent" fill="#22c55e" />
      </BarChart>
    </ResponsiveContainer>
  )
}
```

---

### 3.4 Create `showcase_app/frontend/src/components/charts/LLMUsageChart.jsx` (NEW)

```jsx
import React from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell
} from 'recharts'

const COLORS = ['#3b82f6', '#22c55e', '#f97316', '#a855f7', '#ef4444', '#06b6d4']

export default function LLMUsageChart({ models, view = 'cost' }) {
  if (!models || models.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-gray-500">
        <p>No LLM usage data available.</p>
      </div>
    )
  }

  if (view === 'cost') {
    return (
      <ResponsiveContainer width="100%" height={300}>
        <PieChart>
          <Pie
            data={models}
            cx="50%"
            cy="50%"
            outerRadius={100}
            dataKey="total_cost_usd"
            nameKey="model"
            label={({ model, total_cost_usd }) => `${model}: $${total_cost_usd.toFixed(3)}`}
          >
            {models.map((_, i) => (
              <Cell key={i} fill={COLORS[i % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip formatter={(value) => `$${value.toFixed(4)}`} />
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    )
  }

  // Requests view
  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={models}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="model" fontSize={11} angle={-20} textAnchor="end" height={60} />
        <YAxis />
        <Tooltip />
        <Legend />
        <Bar dataKey="requests" name="Requests" fill="#3b82f6" />
        <Bar dataKey="errors" name="Errors" fill="#ef4444" />
      </BarChart>
    </ResponsiveContainer>
  )
}
```

---

### 3.5 Create `showcase_app/frontend/src/components/charts/PipelineBreakdownChart.jsx` (NEW)

```jsx
import React from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell
} from 'recharts'

const COLORS = ['#3b82f6', '#22c55e', '#f97316', '#a855f7', '#ef4444', '#06b6d4', '#eab308']

export default function PipelineBreakdownChart({ steps }) {
  if (!steps || steps.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-gray-500">
        <p>No pipeline step data available.</p>
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={steps}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" fontSize={12} />
        <YAxis label={{ value: 'Duration (ms)', angle: -90, position: 'insideLeft' }} />
        <Tooltip formatter={(value) => `${value.toFixed(1)}ms`} />
        <Bar dataKey="duration_ms" name="Duration">
          {steps.map((_, i) => (
            <Cell key={i} fill={COLORS[i % COLORS.length]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
```

---

### 3.6 Create `showcase_app/frontend/src/components/charts/MessageFlowChart.jsx` (NEW)

Inter-agent message flow visualization using the existing `reactflow` dependency.

```jsx
import React, { useMemo } from 'react'
import ReactFlow, { Background, Controls, MarkerType } from 'reactflow'
import 'reactflow/dist/style.css'

export default function MessageFlowChart({ agents }) {
  if (!agents || agents.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-gray-500">
        <p>No message flow data available.</p>
      </div>
    )
  }

  const { nodes, edges } = useMemo(() => {
    const agentNodes = agents.map((agent, i) => ({
      id: agent.name,
      data: {
        label: (
          <div className="text-center">
            <div className="font-semibold">{agent.name}</div>
            <div className="text-xs text-gray-500">{agent.role}</div>
            <div className="text-xs mt-1">
              {agent.executions} runs | {agent.messages_sent} msgs
            </div>
          </div>
        )
      },
      position: {
        x: 150 + (i % 3) * 250,
        y: 50 + Math.floor(i / 3) * 150
      },
      style: {
        border: agent.errors > 0 ? '2px solid #ef4444' : '2px solid #3b82f6',
        borderRadius: '8px',
        padding: '10px',
        background: '#fff',
      }
    }))

    // Create edges for agents that communicated
    const agentEdges = []
    agents.forEach(agent => {
      if (agent.messages_sent > 0) {
        // Connect to other agents (simplified - in production, use actual message targets)
        agents.forEach(target => {
          if (target.name !== agent.name && target.messages_received > 0) {
            agentEdges.push({
              id: `${agent.name}-${target.name}`,
              source: agent.name,
              target: target.name,
              label: `${agent.messages_sent}`,
              markerEnd: { type: MarkerType.ArrowClosed },
              style: { stroke: '#94a3b8' },
              labelStyle: { fontSize: 10 }
            })
          }
        })
      }
    })

    return { nodes: agentNodes, edges: agentEdges }
  }, [agents])

  return (
    <div style={{ height: 400 }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        fitView
        attributionPosition="bottom-left"
      >
        <Background />
        <Controls />
      </ReactFlow>
    </div>
  )
}
```

---

### 3.7 Create `showcase_app/frontend/src/pages/AgentDashboard.jsx` (NEW)

```jsx
import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { telemetryAPI } from '../services/api'
import { Users, MessageSquare, Zap, AlertCircle } from 'lucide-react'
import AgentPerformanceChart from '../components/charts/AgentPerformanceChart'
import MessageFlowChart from '../components/charts/MessageFlowChart'

export default function AgentDashboard() {
  const { data: agentData } = useQuery({
    queryKey: ['agent-metrics'],
    queryFn: async () => {
      const response = await telemetryAPI.getAgentMetrics()
      return response.data
    },
    refetchInterval: 10000,
  })

  const summary = agentData?.summary || {}
  const agents = agentData?.agents || []

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-800">Agent Dashboard</h1>
        <p className="text-gray-600 mt-1">Real-time agent execution and communication metrics</p>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <SummaryCard
          label="Active Agents"
          value={summary.total_agents || 0}
          icon={<Users size={24} />}
          color="blue"
        />
        <SummaryCard
          label="Total Executions"
          value={summary.total_executions || 0}
          icon={<Zap size={24} />}
          color="green"
        />
        <SummaryCard
          label="Total Messages"
          value={summary.total_messages || 0}
          icon={<MessageSquare size={24} />}
          color="purple"
        />
        <SummaryCard
          label="Total Errors"
          value={summary.total_errors || 0}
          icon={<AlertCircle size={24} />}
          color="red"
        />
      </div>

      {/* Agent Performance Chart */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-bold text-gray-800 mb-4">Agent Performance</h2>
        <AgentPerformanceChart agents={agents} />
      </div>

      {/* Message Flow */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-bold text-gray-800 mb-4">Inter-Agent Message Flow</h2>
        <MessageFlowChart agents={agents} />
      </div>

      {/* Agent Table */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-bold text-gray-800 mb-4">Agent Details</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Agent</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Role</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Executions</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Errors</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Msgs Sent</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Msgs Recv</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">State R/W</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {agents.map(agent => (
                <tr key={agent.name} className="hover:bg-gray-50">
                  <td className="px-4 py-3 text-sm font-medium text-gray-900">{agent.name}</td>
                  <td className="px-4 py-3 text-sm text-gray-500">{agent.role}</td>
                  <td className="px-4 py-3 text-sm text-right text-gray-900">{agent.executions}</td>
                  <td className="px-4 py-3 text-sm text-right text-red-600">{agent.errors}</td>
                  <td className="px-4 py-3 text-sm text-right text-gray-900">{agent.messages_sent}</td>
                  <td className="px-4 py-3 text-sm text-right text-gray-900">{agent.messages_received}</td>
                  <td className="px-4 py-3 text-sm text-right text-gray-900">
                    {agent.state_reads}/{agent.state_writes}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

function SummaryCard({ label, value, icon, color }) {
  const colors = {
    blue: 'bg-blue-100 text-blue-600',
    green: 'bg-green-100 text-green-600',
    purple: 'bg-purple-100 text-purple-600',
    red: 'bg-red-100 text-red-600',
  }

  return (
    <div className="bg-white rounded-lg shadow p-4">
      <div className={`${colors[color]} rounded-lg p-2 w-fit mb-2`}>{icon}</div>
      <div className="text-2xl font-bold text-gray-800">{value}</div>
      <div className="text-sm text-gray-600">{label}</div>
    </div>
  )
}
```

---

### 3.8 Create `showcase_app/frontend/src/pages/LLMDashboard.jsx` (NEW)

```jsx
import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { telemetryAPI } from '../services/api'
import { Cpu, DollarSign, Zap, BarChart3 } from 'lucide-react'
import LLMUsageChart from '../components/charts/LLMUsageChart'

export default function LLMDashboard() {
  const [chartView, setChartView] = useState('cost')

  const { data: llmData } = useQuery({
    queryKey: ['llm-metrics'],
    queryFn: async () => {
      const response = await telemetryAPI.getLLMUsage()
      return response.data
    },
    refetchInterval: 10000,
  })

  const summary = llmData?.summary || {}
  const models = llmData?.models || []

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-800">LLM Usage Dashboard</h1>
        <p className="text-gray-600 mt-1">Token consumption, cost breakdown, and model performance</p>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white rounded-lg shadow p-4">
          <div className="bg-blue-100 text-blue-600 rounded-lg p-2 w-fit mb-2">
            <Zap size={24} />
          </div>
          <div className="text-2xl font-bold text-gray-800">{summary.total_requests || 0}</div>
          <div className="text-sm text-gray-600">Total Requests</div>
        </div>
        <div className="bg-white rounded-lg shadow p-4">
          <div className="bg-green-100 text-green-600 rounded-lg p-2 w-fit mb-2">
            <DollarSign size={24} />
          </div>
          <div className="text-2xl font-bold text-gray-800">
            ${(summary.total_cost_usd || 0).toFixed(4)}
          </div>
          <div className="text-sm text-gray-600">Total Cost</div>
        </div>
        <div className="bg-white rounded-lg shadow p-4">
          <div className="bg-purple-100 text-purple-600 rounded-lg p-2 w-fit mb-2">
            <Cpu size={24} />
          </div>
          <div className="text-2xl font-bold text-gray-800">{summary.model_count || 0}</div>
          <div className="text-sm text-gray-600">Models Used</div>
        </div>
      </div>

      {/* Cost / Requests Chart */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-gray-800">Usage by Model</h2>
          <div className="flex gap-2">
            <button
              onClick={() => setChartView('cost')}
              className={`px-3 py-1 rounded text-sm ${
                chartView === 'cost'
                  ? 'bg-blue-100 text-blue-700 font-medium'
                  : 'bg-gray-100 text-gray-600'
              }`}
            >
              Cost
            </button>
            <button
              onClick={() => setChartView('requests')}
              className={`px-3 py-1 rounded text-sm ${
                chartView === 'requests'
                  ? 'bg-blue-100 text-blue-700 font-medium'
                  : 'bg-gray-100 text-gray-600'
              }`}
            >
              Requests
            </button>
          </div>
        </div>
        <LLMUsageChart models={models} view={chartView} />
      </div>

      {/* Model Table */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-bold text-gray-800 mb-4">Model Details</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Provider</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Model</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Requests</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Errors</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Cost (USD)</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {models.map((model, i) => (
                <tr key={i} className="hover:bg-gray-50">
                  <td className="px-4 py-3 text-sm text-gray-900 capitalize">{model.system}</td>
                  <td className="px-4 py-3 text-sm font-mono text-gray-900">{model.model}</td>
                  <td className="px-4 py-3 text-sm text-right text-gray-900">{model.requests}</td>
                  <td className="px-4 py-3 text-sm text-right text-red-600">{model.errors}</td>
                  <td className="px-4 py-3 text-sm text-right text-gray-900">
                    ${model.total_cost_usd.toFixed(4)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
```

---

### 3.9 Modify `showcase_app/frontend/src/pages/MetricsPage.jsx`

Replace the placeholder trend chart (lines 261-271) with the real MetricsTrendChart.

```jsx
// ADD import at top:
import MetricsTrendChart from '../components/charts/MetricsTrendChart'

// ADD query for time-series data:
const { data: timeseries } = useQuery({
  queryKey: ['metrics-timeseries'],
  queryFn: async () => {
    const response = await metricsAPI.getHistory(24)
    return response.data?.datapoints || []
  },
  refetchInterval: 30000,
})

// REPLACE the placeholder section (lines 261-271) with:
<div className="bg-white rounded-lg shadow p-6">
  <h2 className="text-xl font-bold text-gray-800 mb-4">Metrics Trend (24h)</h2>
  <MetricsTrendChart data={timeseries} metrics={['svr', 'cr', 'hir', 'ma']} />
</div>
```

---

### 3.10 Modify `showcase_app/frontend/src/App.jsx`

Add new routes for Agent and LLM dashboards.

```jsx
// ADD imports:
import AgentDashboard from './pages/AgentDashboard'
import LLMDashboard from './pages/LLMDashboard'

// ADD routes inside <Routes> (after existing /metrics route):
<Route path="/agents" element={<AgentDashboard />} />
<Route path="/llm" element={<LLMDashboard />} />

// ADD sidebar navigation items (after existing Metrics link):
{ path: '/agents', label: 'Agents', icon: <Users size={20} /> },
{ path: '/llm', label: 'LLM Usage', icon: <Cpu size={20} /> },
```

---

## File Summary

### New Files (13)

| # | File Path | Phase | Purpose |
|---|-----------|-------|---------|
| 1 | `ia_modules/telemetry/agent_telemetry.py` | 1 | Agent execution/message/collaboration telemetry |
| 2 | `ia_modules/telemetry/llm_telemetry.py` | 1 | LLM call telemetry with gen_ai conventions |
| 3 | `tests/integration/test_agent_telemetry.py` | 1 | Agent telemetry integration tests |
| 4 | `tests/integration/test_llm_telemetry.py` | 1 | LLM telemetry integration tests |
| 5 | `showcase_app/frontend/src/components/charts/MetricsTrendChart.jsx` | 3 | Time-series line chart |
| 6 | `showcase_app/frontend/src/components/charts/AgentPerformanceChart.jsx` | 3 | Agent comparison bar chart |
| 7 | `showcase_app/frontend/src/components/charts/LLMUsageChart.jsx` | 3 | Cost pie + requests bar chart |
| 8 | `showcase_app/frontend/src/components/charts/PipelineBreakdownChart.jsx` | 3 | Step duration breakdown |
| 9 | `showcase_app/frontend/src/components/charts/MessageFlowChart.jsx` | 3 | ReactFlow agent message graph |
| 10 | `showcase_app/frontend/src/pages/AgentDashboard.jsx` | 3 | Agent metrics dashboard page |
| 11 | `showcase_app/frontend/src/pages/LLMDashboard.jsx` | 3 | LLM usage dashboard page |

### Modified Files (9)

| # | File Path | Phase | Change |
|---|-----------|-------|--------|
| 1 | `ia_modules/telemetry/__init__.py` | 1 | Export AgentTelemetry, LLMTelemetry |
| 2 | `ia_modules/telemetry/integration.py` | 1 | Add global agent/LLM telemetry singletons |
| 3 | `ia_modules/agents/core.py` | 1 | Instrument BaseAgent with telemetry |
| 4 | `ia_modules/agents/base_agent.py` | 1 | Instrument message passing |
| 5 | `ia_modules/pipeline/llm_provider_service.py` | 1 | Instrument LLM calls |
| 6 | `ia_modules/agents/collaboration_patterns/hierarchical.py` | 1 | Instrument collaboration |
| 7 | `showcase_app/backend/services/telemetry_service.py` | 2 | Add agent/LLM metric aggregation |
| 8 | `showcase_app/backend/api/telemetry.py` | 2 | Add new API endpoints |
| 9 | `showcase_app/backend/main.py` | 2 | Wire up agent/LLM telemetry services |
| 10 | `showcase_app/frontend/src/services/api.js` | 3 | Add telemetry API methods |
| 11 | `showcase_app/frontend/src/pages/MetricsPage.jsx` | 3 | Replace placeholder with real chart |
| 12 | `showcase_app/frontend/src/App.jsx` | 3 | Add Agent/LLM dashboard routes |

### No New Dependencies Required

The `[observability]` extra in `pyproject.toml` already includes all OpenTelemetry packages needed. The frontend already has `recharts` and `reactflow`.
