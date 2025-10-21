# Reliability & Observability System - Usage Guide

Complete guide for using the IA Modules Reliability and Observability System.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Decision Trails](#decision-trails)
3. [Replay System](#replay-system)
4. [Reliability Metrics](#reliability-metrics)
5. [SLO Tracking](#slo-tracking)
6. [Mode Enforcement](#mode-enforcement)
7. [Evidence Collection](#evidence-collection)
8. [Anomaly Detection](#anomaly-detection)
9. [Trend Analysis](#trend-analysis)
10. [Alert System](#alert-system)
11. [Circuit Breakers](#circuit-breakers)
12. [Cost Tracking](#cost-tracking)
13. [Storage Backends](#storage-backends)
14. [Auto-Evidence Hooks](#auto-evidence-hooks)
15. [Best Practices](#best-practices)

---

## Quick Start

```python
from reliability import (
    DecisionTrailBuilder,
    ReliabilityMetrics,
    SLOTracker,
    ModeEnforcer,
    AgentMode
)

# Initialize components
metrics = ReliabilityMetrics()
slo_tracker = SLOTracker()
mode_enforcer = ModeEnforcer()

# Set agent mode
mode_enforcer.set_mode("my_agent", AgentMode.EXECUTE)

# Record step
await metrics.record_step(
    agent="my_agent",
    success=True,
    mode="execute",
    declared_mode="execute"
)

# Get metrics report
report = await metrics.get_report()
print(f"System Health: {report.is_healthy()}")
print(f"SVR: {report.svr:.2%}")
```

---

## Decision Trails

### Complete Decision Reconstruction

```python
from reliability import DecisionTrailBuilder

# Initialize
trail_builder = DecisionTrailBuilder(
    state_manager=state_manager,
    tool_registry=tool_registry,
    checkpoint_manager=checkpoint_manager
)

# After workflow execution, build trail
trail = await trail_builder.build_trail(
    thread_id="workflow-123",
    checkpoint_id="checkpoint-456"
)

# Inspect trail
print(f"Goal: {trail.goal}")
print(f"Success: {trail.success}")
print(f"Steps: {len(trail.steps)}")
print(f"Tool calls: {len(trail.tool_calls)}")

# Generate human-readable explanation
explanation = await trail_builder.explain_decision(trail)
print(explanation)
```

### Evidence Levels

```python
# Examine evidence by confidence level
for evidence in trail.evidence:
    print(f"{evidence.confidence}: {evidence.type} from {evidence.source}")

# verified - From tools, databases, humans
# claimed - From agent reasoning
# inferred - Derived conclusions
```

---

## Replay System

### Strict Replay (Exact Reproduction)

```python
from reliability import Replayer, ReplayMode

replayer = Replayer(decision_trail)

# Replay with real tools
result = await replayer.strict_replay(
    orchestrator=orchestrator,
    tool_registry=tool_registry
)

if result.is_exact_match():
    print("Workflow reproduced exactly!")
else:
    print("Differences found:")
    for diff in result.differences:
        print(f"  {diff.field}: {diff.significance}")
```

### Simulated Replay (Fast Testing)

```python
# Replay with mocked tool responses
result = await replayer.simulated_replay(orchestrator=orchestrator)

# Fast - no real tool calls
# Perfect for unit testing workflow logic
```

### Counterfactual Replay (What-If Analysis)

```python
# Test different inputs
alternative_inputs = {"temperature": 0.5}

result = await replayer.counterfactual_replay(
    alternative_inputs,
    orchestrator=orchestrator,
    tool_registry=tool_registry
)

print(f"Original outcome: {trail.outcome}")
print(f"Alternative outcome: {result.outcome}")
```

---

## Reliability Metrics

### Five Core Metrics

```python
from reliability import ReliabilityMetrics

metrics = ReliabilityMetrics()

# Record steps
await metrics.record_step(
    agent="researcher",
    success=True,
    required_compensation=False,
    required_human=False,
    mode="execute",
    declared_mode="execute"
)

# Record workflow
await metrics.record_workflow(
    workflow_id="workflow-001",
    total_steps=10,
    total_retries=2,
    required_compensation=False,
    required_human=True,
    agents_involved=["researcher", "planner"]
)

# Get report
report = await metrics.get_report()

print(f"SVR (Step Validity Rate): {report.svr:.2%} (target >95%)")
print(f"CR (Compensation Rate): {report.cr:.2%} (target <10%)")
print(f"PC (Plan Churn): {report.pc:.1f} (target <2)")
print(f"HIR (Human Intervention Rate): {report.hir:.2%} (target <5%)")
print(f"MA (Mode Adherence): {report.ma:.2%} (target >90%)")

if not report.is_healthy():
    print("Violations:", report.get_violations())
```

### Performance Metrics (TCL, WCT)

```python
# Record step with tool timing
await metrics.record_step(
    agent="researcher",
    success=True,
    tool_duration_ms=250  # Tool took 250ms
)

# Record workflow with total duration
await metrics.record_workflow(
    workflow_id="workflow-002",
    total_steps=15,
    total_retries=0,
    duration_ms=5000  # Workflow took 5 seconds
)

# Get performance metrics
report = await metrics.get_report()

if report.tcl:
    print(f"TCL (Tool Call Latency): {report.tcl:.0f}ms")
if report.wct:
    print(f"WCT (Workflow Completion Time): {report.wct:.0f}ms")
```

---

## SLO Tracking

### Track MTTE (Mean Time to Explain)

```python
from reliability import SLOTracker

slo_tracker = SLOTracker()

# Record explanation attempt
start = time.time()
trail = await trail_builder.build_trail(thread_id, checkpoint_id)
explanation = await trail_builder.explain_decision(trail)
duration_ms = int((time.time() - start) * 1000)

await slo_tracker.record_mtte(
    thread_id=thread_id,
    checkpoint_id=checkpoint_id,
    duration_ms=duration_ms,
    success=True
)

# Get SLO report
slo_report = await slo_tracker.get_report()

print(f"MTTE P95: {slo_report.mtte_p95_ms/1000:.1f}s")
print(f"Target: {slo_report.mtte_target_ms/1000:.0f}s")
print(f"Compliant: {slo_report.is_mtte_compliant()}")
```

### Track RSR (Replay Success Rate)

```python
# Record replay attempt
result = await replayer.strict_replay(orchestrator, tool_registry)

await slo_tracker.record_rsr(
    thread_id=thread_id,
    checkpoint_id=checkpoint_id,
    replay_mode="strict",
    success=result.success
)

# Check compliance
slo_report = await slo_tracker.get_report()

print(f"RSR: {slo_report.rsr:.1%}")
print(f"Target: {slo_report.rsr_target:.1%}")
print(f"Compliant: {slo_report.is_rsr_compliant()}")
```

---

## Mode Enforcement

### Three Agent Modes

```python
from reliability import ModeEnforcer, AgentMode

enforcer = ModeEnforcer()

# EXPLORE mode - Read-only research
enforcer.set_mode("researcher", AgentMode.EXPLORE)

# EXECUTE mode - Take actions
enforcer.set_mode("executor", AgentMode.EXECUTE)

# ESCALATE mode - Request human approval
enforcer.set_mode("reviewer", AgentMode.ESCALATE)
enforcer.require_approval("reviewer", "approve_changes")
```

### Validate Actions

```python
# Check if action is allowed
if enforcer.can_execute("researcher", "database_write"):
    # Execute action
    pass
else:
    print("Action blocked - agent in EXPLORE mode")

# Check violations
violations = enforcer.get_violations(agent="researcher")
for v in violations:
    print(f"Violation: {v.action} at {v.timestamp}")
```

---

## Evidence Collection

### Manual Collection

```python
from reliability import EvidenceCollector

collector = EvidenceCollector()

# From tool
evidence = collector.from_tool_result("search_api", search_results)

# From database
evidence = collector.from_database_read("users_table", query_result)

# From API
evidence = collector.from_api_response("weather_api", api_response)

# From user
evidence = collector.from_user_input(user_input)

# From agent reasoning
evidence = collector.from_agent_output("planner", plan_output)

# Get all verified evidence
verified = collector.get_verified_evidence()
```

### Automatic Collection

```python
from reliability import AutoEvidenceHooks

# Install hooks
hooks = AutoEvidenceHooks(
    tool_registry=tool_registry,
    state_manager=state_manager,
    checkpoint_manager=checkpoint_manager
)
hooks.install()

# Evidence collected automatically during execution!

# Retrieve evidence
evidence = hooks.get_evidence(thread_id="thread-001")
verified = hooks.get_verified_evidence(thread_id="thread-001")

# Cleanup
hooks.uninstall()
```

---

## Anomaly Detection

### Threshold-Based Detection

```python
from reliability import AnomalyDetector, AnomalyThreshold, Severity

detector = AnomalyDetector(min_baseline_samples=10)

# Configure thresholds
detector.add_threshold(AnomalyThreshold(
    metric_name="error_rate",
    max_value=0.10,  # Max 10% errors
    severity=Severity.HIGH
))

detector.add_threshold(AnomalyThreshold(
    metric_name="success_rate",
    min_value=0.95,  # Min 95% success
    severity=Severity.HIGH
))

# Record values
detector.record_value("error_rate", 0.05)
detector.record_value("success_rate", 0.98)

# Detect anomalies
anomalies = detector.detect_anomalies()

for anomaly in anomalies:
    print(f"{anomaly.severity.value}: {anomaly.metric_name}")
    print(f"  Current: {anomaly.current_value}")
    print(f"  Expected: {anomaly.expected_value}")
```

### Statistical Anomaly Detection

```python
# Record baseline
for i in range(20):
    detector.record_value("latency", 100 + i)  # Normal: 100-120ms

# Record anomaly
detector.record_value("latency", 500)  # Spike!

anomalies = detector.detect_anomalies()

spike_anomalies = [a for a in anomalies if a.type == AnomalyType.SUDDEN_SPIKE]
print(f"Detected {len(spike_anomalies)} spikes")
```

---

## Trend Analysis

### Detect Degrading Trends

```python
from reliability import TrendAnalyzer, TrendDirection

analyzer = TrendAnalyzer(min_data_points=10)

# Record values over time
base_time = datetime.utcnow()
for i in range(20):
    timestamp = base_time + timedelta(hours=i)
    value = 0.99 - (i * 0.01)  # Degrading
    analyzer.record_value("success_rate", value, timestamp)

# Analyze
trend = analyzer.analyze("success_rate")

if trend.direction == TrendDirection.DEGRADING:
    print(f"WARNING: Success rate degrading!")
    print(f"  Slope: {trend.slope}")
    print(f"  Confidence: {trend.confidence:.2%}")
    print(f"  Current: {trend.current_value:.2%}")
    print(f"  Predicted next: {trend.predicted_value:.2%}")
```

### Forecast Future Values

```python
# Forecast 24 hours ahead
forecast = analyzer.forecast("success_rate", hours_ahead=24)

for timestamp, predicted_value in forecast:
    print(f"{timestamp}: {predicted_value:.2%}")
```

### Predict Degradation

```python
# Check if will drop below threshold
degradation = analyzer.detect_degradation(
    "success_rate",
    threshold=0.90,
    forecast_hours=24
)

if degradation:
    print(f"Will breach threshold in {degradation['hours_until_breach']:.1f} hours")
    print(f"Breach value: {degradation['breach_value']:.2%}")
```

---

## Alert System

### Setup Alerts

```python
from reliability import AlertManager, AlertSeverity, AlertType

manager = AlertManager()

# Add custom alert channel
def my_alert_handler(alert):
    print(f"ALERT: {alert.title}")
    # Send to Slack, email, etc.

from reliability import CallbackAlertChannel
manager.add_channel("custom", CallbackAlertChannel(my_alert_handler))

# Trigger alerts
await manager.trigger_metric_alert(
    metric_name="error_rate",
    current_value=0.15,
    threshold=0.10,
    severity=AlertSeverity.WARNING
)

# From SLO breach
await manager.trigger_slo_alert(slo_report, "mtte", compliant=False)

# From anomaly
await manager.trigger_anomaly_alert(anomaly)

# Get alerts
recent_alerts = manager.get_alerts(
    since=datetime.utcnow() - timedelta(hours=1),
    severity=AlertSeverity.HIGH
)
```

---

## Circuit Breakers

### Protect Against Failures

```python
from reliability import CircuitBreaker, CircuitBreakerConfig, CircuitState

breaker = CircuitBreaker(
    "external_api",
    CircuitBreakerConfig(
        failure_threshold=5,      # Open after 5 failures
        success_threshold=2,       # Close after 2 successes
        timeout_seconds=60         # Try recovery after 60s
    )
)

# Check before calling API
if breaker.can_execute():
    try:
        result = call_external_api()
        breaker.record_success()
    except Exception:
        breaker.record_failure()
else:
    print("Circuit open - service unavailable")

# Check state
if breaker.state == CircuitState.OPEN:
    print("Circuit breaker protecting system")
```

### Registry for Multiple Circuits

```python
from reliability import CircuitBreakerRegistry

registry = CircuitBreakerRegistry()

# Get/create circuit breakers
api_breaker = registry.get_breaker("external_api")
db_breaker = registry.get_breaker("database")

# Check status
status = registry.get_status()
print(f"API: {status['external_api']}")
print(f"DB: {status['database']}")

# Get open circuits
open_circuits = registry.get_open_breakers()
if open_circuits:
    print(f"Warning: {len(open_circuits)} circuits open")
```

---

## Cost Tracking

### Track LLM Costs

```python
from reliability import CostTracker, CostBudget

tracker = CostTracker()

# Set budget
budget = CostBudget(
    name="daily",
    total_limit=100.0,  # $100/day
    period_hours=24
)
tracker.set_budget(budget)

# Record LLM cost
tracker.record_llm_cost(
    agent="researcher",
    prompt_tokens=1000,
    completion_tokens=500,
    model="gpt-4",
    workflow_id="workflow-001"
)

# Record tool cost
tracker.record_tool_cost(
    agent="executor",
    tool_name="image_generation",
    cost=0.50,
    workflow_id="workflow-001"
)

# Check budget
if not tracker.is_within_budget():
    print("WARNING: Budget exceeded!")

# Get report
report = tracker.get_report()
print(f"Total cost: ${report.total_cost:.2f}")
print(f"Cost per workflow: ${report.cost_per_workflow:.2f}")
print(f"Tokens per workflow: {report.tokens_per_workflow:.0f}")
```

---

## Storage Backends

### SQL Storage

```python
from reliability import SQLMetricStorage
from database.interfaces import ConnectionConfig, DatabaseType

# PostgreSQL
config = ConnectionConfig(
    database_type=DatabaseType.POSTGRESQL,
    database_url="postgresql://localhost/metrics"
)

sql_storage = SQLMetricStorage(config)
await sql_storage.initialize()

# Use with metrics
metrics = ReliabilityMetrics(storage=sql_storage)
```

### Redis Storage

```python
from reliability import RedisMetricStorage

redis_storage = RedisMetricStorage(
    redis_url="redis://localhost:6379",
    ttl_days=90  # Auto-cleanup after 90 days
)
await redis_storage.initialize()

# Use with metrics
metrics = ReliabilityMetrics(storage=redis_storage)
```

---

## Auto-Evidence Hooks

### Automatic Integration

```python
from reliability import AutoEvidenceHooks

# Install once at app startup
hooks = AutoEvidenceHooks(
    tool_registry=tool_registry,
    state_manager=state_manager,
    checkpoint_manager=checkpoint_manager
)
hooks.install()

# Evidence collected automatically!
# - Every tool execution
# - Every state change
# - Every checkpoint

# Retrieve anytime
evidence = hooks.get_evidence("thread-123")
verified_only = hooks.get_verified_evidence("thread-123")

# Get stats
stats = hooks.get_stats()
print(f"Total evidence: {stats['total_evidence']}")
print(f"Across {stats['total_threads']} threads")
```

---

## Best Practices

### 1. Always Use Mode Enforcement

```python
# Set appropriate modes
enforcer.set_mode("researcher", AgentMode.EXPLORE)    # Safe research
enforcer.set_mode("executor", AgentMode.EXECUTE)      # Controlled actions
enforcer.set_mode("reviewer", AgentMode.ESCALATE)     # Human oversight
```

### 2. Track All Workflows

```python
# Record every step
await metrics.record_step(agent, success, mode, declared_mode)

# Record every workflow
await metrics.record_workflow(workflow_id, steps, retries)

# Track SLOs
await slo_tracker.record_mtte(thread_id, checkpoint_id, duration, success)
await slo_tracker.record_rsr(thread_id, checkpoint_id, mode, success)
```

### 3. Monitor Continuously

```python
# Set up anomaly detection
detector.add_threshold(AnomalyThreshold("svr", min_value=0.95))

# Set up alerting
manager.add_channel("ops", CallbackAlertChannel(notify_ops_team))

# Check health regularly
report = await metrics.get_report()
if not report.is_healthy():
    trigger_investigation(report.get_violations())
```

### 4. Use Circuit Breakers

```python
# Protect all external dependencies
api_breaker = registry.get_breaker("external_api")
db_breaker = registry.get_breaker("database")

if api_breaker.can_execute():
    # Safe to proceed
    pass
```

### 5. Track Costs

```python
# Monitor spend
tracker.record_llm_cost(...)
tracker.record_tool_cost(...)

if not tracker.is_within_budget():
    throttle_requests()
```

---

## Complete Example

```python
from reliability import *

# Setup
metrics = ReliabilityMetrics()
slo_tracker = SLOTracker()
enforcer = ModeEnforcer()
detector = AnomalyDetector()
analyzer = TrendAnalyzer()
alert_manager = AlertManager()
cost_tracker = CostTracker()

# Configure
enforcer.set_mode("agent", AgentMode.EXECUTE)
detector.add_threshold(AnomalyThreshold("svr", min_value=0.95))
cost_tracker.set_budget(CostBudget("daily", 100.0, 24))

# Execute workflow
if enforcer.can_execute("agent", "action"):
    start_time = time.time()

    try:
        result = execute_workflow()
        duration_ms = int((time.time() - start_time) * 1000)

        # Record success
        await metrics.record_step("agent", True, "execute", "execute")
        await slo_tracker.record_mtte(thread_id, checkpoint_id, duration_ms, True)
        cost_tracker.record_llm_cost("agent", 1000, 500, "gpt-4")

    except Exception as e:
        # Record failure
        await metrics.record_step("agent", False, "execute", "execute")

# Monitor
report = await metrics.get_report()
anomalies = detector.detect_anomalies()
trend = analyzer.analyze("svr")

if not report.is_healthy():
    await alert_manager.trigger_health_check_alert(
        report,
        report.get_violations()
    )
```

---

For more details, see the [Enterprise Reliability Framework Reference](Enterprise_RELIABILITY_FRAMEWORK_REFERENCE.md).
