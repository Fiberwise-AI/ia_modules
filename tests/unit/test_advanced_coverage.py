"""
Tests for advanced coverage of multiple modules.

Targets:
1. ia_modules/tools/advanced_executor.py
2. ia_modules/telemetry/opentelemetry_exporter.py
3. ia_modules/telemetry/tracing.py
4. ia_modules/telemetry/integration.py
5. ia_modules/benchmarking/framework.py
6. ia_modules/benchmarking/comparison.py
7. ia_modules/benchmarking/telemetry_bridge.py
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from ia_modules.benchmarking.models import BenchmarkConfig, BenchmarkResult


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def make_benchmark_result(
    name="test",
    iterations=10,
    mean_time=0.1,
    median_time=0.1,
    std_dev=0.01,
    min_time=0.05,
    max_time=0.2,
    p95_time=0.18,
    p99_time=0.19,
    total_time=1.0,
    operations_per_second=10.0,
    items_processed=0,
    api_calls_count=0,
    estimated_cost_usd=0.0,
    memory_per_operation_mb=0.0,
    cpu_per_operation_percent=0.0,
    memory_stats=None,
    cpu_stats=None,
    **kwargs,
):
    return BenchmarkResult(
        name=name,
        iterations=iterations,
        mean_time=mean_time,
        median_time=median_time,
        std_dev=std_dev,
        min_time=min_time,
        max_time=max_time,
        p95_time=p95_time,
        p99_time=p99_time,
        total_time=total_time,
        operations_per_second=operations_per_second,
        items_processed=items_processed,
        api_calls_count=api_calls_count,
        estimated_cost_usd=estimated_cost_usd,
        memory_per_operation_mb=memory_per_operation_mb,
        cpu_per_operation_percent=cpu_per_operation_percent,
        memory_stats=memory_stats,
        cpu_stats=cpu_stats,
        **kwargs,
    )


# ===========================================================================
# 1. tests for ia_modules/telemetry/tracing.py
# ===========================================================================

class TestSpan:
    """Cover Span helper methods."""

    def test_span_basic_attributes(self):
        from ia_modules.telemetry.tracing import Span
        span = Span(name="op", trace_id="t1", span_id="s1")
        assert span.name == "op"
        assert span.status == "unset"
        assert span.duration is None  # not finished yet

    def test_set_attribute(self):
        from ia_modules.telemetry.tracing import Span
        span = Span(name="op", trace_id="t1", span_id="s1")
        span.set_attribute("key", "val")
        assert span.attributes["key"] == "val"

    def test_add_event(self):
        from ia_modules.telemetry.tracing import Span
        span = Span(name="op", trace_id="t1", span_id="s1")
        span.add_event("evt", {"k": 1})
        assert len(span.events) == 1
        assert span.events[0]["name"] == "evt"
        assert span.events[0]["attributes"] == {"k": 1}

    def test_add_event_no_attrs(self):
        from ia_modules.telemetry.tracing import Span
        span = Span(name="op", trace_id="t1", span_id="s1")
        span.add_event("evt")
        assert span.events[0]["attributes"] == {}

    def test_set_status_with_description(self):
        from ia_modules.telemetry.tracing import Span
        span = Span(name="op", trace_id="t1", span_id="s1")
        span.set_status("error", "something bad")
        assert span.status == "error"
        assert span.attributes["status.description"] == "something bad"

    def test_set_status_without_description(self):
        from ia_modules.telemetry.tracing import Span
        span = Span(name="op", trace_id="t1", span_id="s1")
        span.set_status("ok")
        assert span.status == "ok"
        assert "status.description" not in span.attributes

    def test_finish_and_duration(self):
        from ia_modules.telemetry.tracing import Span
        span = Span(name="op", trace_id="t1", span_id="s1")
        span.finish()
        assert span.end_time is not None
        assert span.duration >= 0

    def test_finish_idempotent(self):
        from ia_modules.telemetry.tracing import Span
        span = Span(name="op", trace_id="t1", span_id="s1")
        span.finish()
        first_end = span.end_time
        span.finish()
        assert span.end_time == first_end  # should not change

    def test_to_dict(self):
        from ia_modules.telemetry.tracing import Span
        span = Span(name="op", trace_id="t1", span_id="s1", parent_span_id="p1",
                     attributes={"a": 1})
        span.finish()
        d = span.to_dict()
        assert d["name"] == "op"
        assert d["trace_id"] == "t1"
        assert d["span_id"] == "s1"
        assert d["parent_span_id"] == "p1"
        assert d["duration"] is not None

    def test_duration_none_when_not_finished(self):
        from ia_modules.telemetry.tracing import Span
        span = Span(name="op", trace_id="t1", span_id="s1")
        assert span.duration is None


class TestSimpleTracer:
    """Cover SimpleTracer."""

    def test_start_and_end_span(self):
        from ia_modules.telemetry.tracing import SimpleTracer
        tracer = SimpleTracer()
        span = tracer.start_span("op1", attributes={"k": "v"})
        assert span.attributes == {"k": "v"}
        tracer.end_span(span)
        assert len(tracer.spans) == 1
        assert span.duration is not None

    def test_parent_span(self):
        from ia_modules.telemetry.tracing import SimpleTracer
        tracer = SimpleTracer()
        parent = tracer.start_span("parent")
        child = tracer.start_span("child", parent=parent)
        assert child.trace_id == parent.trace_id
        assert child.parent_span_id == parent.span_id
        tracer.end_span(child)
        tracer.end_span(parent)

    def test_get_spans_by_trace_id(self):
        from ia_modules.telemetry.tracing import SimpleTracer
        tracer = SimpleTracer()
        s1 = tracer.start_span("a")
        tracer.end_span(s1)
        s2 = tracer.start_span("b")
        tracer.end_span(s2)
        filtered = tracer.get_spans(s1.trace_id)
        assert len(filtered) == 1
        assert filtered[0].name == "a"

    def test_get_all_spans(self):
        from ia_modules.telemetry.tracing import SimpleTracer
        tracer = SimpleTracer()
        s1 = tracer.start_span("a")
        tracer.end_span(s1)
        assert tracer.get_spans() == tracer.spans

    def test_clear(self):
        from ia_modules.telemetry.tracing import SimpleTracer
        tracer = SimpleTracer()
        s = tracer.start_span("a")
        tracer.end_span(s)
        tracer.clear()
        assert len(tracer.spans) == 0

    def test_generate_ids_increment(self):
        from ia_modules.telemetry.tracing import SimpleTracer
        tracer = SimpleTracer()
        s1 = tracer.start_span("a")
        s2 = tracer.start_span("b")
        # span ids should differ
        assert s1.span_id != s2.span_id


class TestTracedDecorator:
    """Cover the traced() decorator for sync and async."""

    async def test_traced_async_success(self):
        from ia_modules.telemetry.tracing import SimpleTracer, traced
        tracer = SimpleTracer()

        @traced(tracer, "my_async_op")
        async def my_func(x):
            return x * 2

        result = await my_func(5)
        assert result == 10
        assert len(tracer.spans) == 1
        assert tracer.spans[0].status == "ok"

    async def test_traced_async_error(self):
        from ia_modules.telemetry.tracing import SimpleTracer, traced
        tracer = SimpleTracer()

        @traced(tracer)
        async def failing():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            await failing()

        assert len(tracer.spans) == 1
        assert tracer.spans[0].status == "error"
        assert tracer.spans[0].attributes["error.type"] == "ValueError"

    def test_traced_sync_success(self):
        from ia_modules.telemetry.tracing import SimpleTracer, traced
        tracer = SimpleTracer()

        @traced(tracer, "sync_op")
        def sync_func(x):
            return x + 1

        result = sync_func(3)
        assert result == 4
        assert len(tracer.spans) == 1
        assert tracer.spans[0].status == "ok"

    def test_traced_sync_error(self):
        from ia_modules.telemetry.tracing import SimpleTracer, traced
        tracer = SimpleTracer()

        @traced(tracer)
        def failing():
            raise RuntimeError("sync boom")

        with pytest.raises(RuntimeError, match="sync boom"):
            failing()

        assert tracer.spans[0].status == "error"
        assert tracer.spans[0].attributes["error.type"] == "RuntimeError"


class TestTraceContext:
    """Cover the trace_context context manager."""

    def test_trace_context_success(self):
        from ia_modules.telemetry.tracing import SimpleTracer, trace_context
        tracer = SimpleTracer()

        with trace_context(tracer, "block", attributes={"a": 1}) as span:
            span.set_attribute("result", 42)

        assert len(tracer.spans) == 1
        assert tracer.spans[0].status == "ok"
        assert tracer.spans[0].attributes["result"] == 42

    def test_trace_context_error(self):
        from ia_modules.telemetry.tracing import SimpleTracer, trace_context
        tracer = SimpleTracer()

        with pytest.raises(ValueError):
            with trace_context(tracer, "block") as span:
                raise ValueError("ctx error")

        assert tracer.spans[0].status == "error"
        assert tracer.spans[0].attributes["error.type"] == "ValueError"


# ===========================================================================
# 2. tests for ia_modules/telemetry/integration.py
# ===========================================================================

class TestPipelineTelemetry:
    """Cover PipelineTelemetry including trace_pipeline and trace_step."""

    def test_disabled_telemetry_pipeline(self):
        from ia_modules.telemetry.integration import PipelineTelemetry
        tel = PipelineTelemetry(enabled=False)
        with tel.trace_pipeline("p") as ctx:
            ctx.set_result({"x": 1})
            ctx.add_event("e")
            ctx.set_attribute("k", "v")
            ctx.record_items(5)
            ctx.record_cost(api_calls=2, cost_usd=0.5)

    def test_disabled_telemetry_step(self):
        from ia_modules.telemetry.integration import PipelineTelemetry
        tel = PipelineTelemetry(enabled=False)
        with tel.trace_step("p", "s") as ctx:
            ctx.set_output({"y": 2})
            ctx.add_event("e")
            ctx.set_attribute("k", "v")

    def test_trace_pipeline_success(self):
        from ia_modules.telemetry.integration import PipelineTelemetry
        tel = PipelineTelemetry(enabled=True)
        with tel.trace_pipeline("mypipe", input_data={"x": 1}) as ctx:
            ctx.set_result({"answer": 42})
            ctx.add_event("processing", {"items": 5})
            ctx.set_attribute("custom", "val")
            ctx.record_items(10)
            ctx.record_cost(api_calls=3, cost_usd=0.01)

        spans = tel.get_spans()
        assert len(spans) == 1
        assert spans[0].status == "ok"

    def test_trace_pipeline_error(self):
        from ia_modules.telemetry.integration import PipelineTelemetry
        tel = PipelineTelemetry(enabled=True)
        with pytest.raises(RuntimeError):
            with tel.trace_pipeline("mypipe") as ctx:
                raise RuntimeError("pipeline fail")

        spans = tel.get_spans()
        assert spans[0].status == "error"

    def test_trace_step_success(self):
        from ia_modules.telemetry.integration import PipelineTelemetry
        tel = PipelineTelemetry(enabled=True)
        with tel.trace_step("pipe", "step1") as ctx:
            ctx.set_output({"data": [1, 2, 3]})
            ctx.add_event("computed")
            ctx.set_attribute("rows", 3)

        spans = tel.get_spans()
        assert len(spans) == 1
        assert spans[0].status == "ok"

    def test_trace_step_with_parent(self):
        from ia_modules.telemetry.integration import PipelineTelemetry
        from ia_modules.telemetry.tracing import SimpleTracer
        tracer = SimpleTracer()
        tel = PipelineTelemetry(tracer=tracer, enabled=True)
        parent = tracer.start_span("parent_span")
        with tel.trace_step("pipe", "step1", parent_span=parent) as ctx:
            ctx.set_output("ok")
        tracer.end_span(parent)

    def test_trace_step_error(self):
        from ia_modules.telemetry.integration import PipelineTelemetry
        tel = PipelineTelemetry(enabled=True)
        with pytest.raises(ValueError):
            with tel.trace_step("pipe", "step1") as ctx:
                raise ValueError("step fail")

        spans = tel.get_spans()
        assert spans[0].status == "error"

    def test_record_benchmark_result_disabled(self):
        from ia_modules.telemetry.integration import PipelineTelemetry
        tel = PipelineTelemetry(enabled=False)
        result = make_benchmark_result()
        tel.record_benchmark_result("pipe", result)  # should do nothing

    def test_record_benchmark_result_with_items_and_api_calls(self):
        from ia_modules.telemetry.integration import PipelineTelemetry
        tel = PipelineTelemetry(enabled=True)
        result = make_benchmark_result(
            total_time=2.0,
            items_processed=50,
            api_calls_count=10,
            estimated_cost_usd=0.5,
        )
        tel.record_benchmark_result("pipe", result)

    def test_record_benchmark_result_with_memory_stats(self):
        from ia_modules.telemetry.integration import PipelineTelemetry
        tel = PipelineTelemetry(enabled=True)
        result = make_benchmark_result(
            memory_stats={"delta_mb": 100.0},
        )
        tel.record_benchmark_result("pipe", result)

    def test_record_benchmark_result_with_memory_per_op(self):
        from ia_modules.telemetry.integration import PipelineTelemetry
        tel = PipelineTelemetry(enabled=True)
        result = make_benchmark_result(
            memory_per_operation_mb=5.0,
        )
        tel.record_benchmark_result("pipe", result)

    def test_record_benchmark_result_with_cpu_stats(self):
        from ia_modules.telemetry.integration import PipelineTelemetry
        tel = PipelineTelemetry(enabled=True)
        result = make_benchmark_result(
            cpu_stats={"average_cpu_percent": 45.0},
        )
        tel.record_benchmark_result("pipe", result)

    def test_record_benchmark_result_with_cpu_per_op(self):
        from ia_modules.telemetry.integration import PipelineTelemetry
        tel = PipelineTelemetry(enabled=True)
        result = make_benchmark_result(
            cpu_per_operation_percent=12.5,
        )
        tel.record_benchmark_result("pipe", result)

    def test_get_metrics(self):
        from ia_modules.telemetry.integration import PipelineTelemetry
        tel = PipelineTelemetry(enabled=True)
        metrics = tel.get_metrics()
        assert isinstance(metrics, list)

    def test_get_spans_with_trace_id(self):
        from ia_modules.telemetry.integration import PipelineTelemetry
        tel = PipelineTelemetry(enabled=True)
        spans = tel.get_spans(trace_id="nonexistent")
        assert spans == []

    def test_pipeline_context_record_cost_zero(self):
        """Cover record_cost when api_calls=0, cost_usd=0."""
        from ia_modules.telemetry.integration import PipelineTelemetry
        tel = PipelineTelemetry(enabled=True)
        with tel.trace_pipeline("pipe") as ctx:
            ctx.record_cost(api_calls=0, cost_usd=0.0)

    def test_pipeline_context_set_result_none(self):
        """Cover set_result with falsy result."""
        from ia_modules.telemetry.integration import PipelineTelemetry
        tel = PipelineTelemetry(enabled=True)
        with tel.trace_pipeline("pipe") as ctx:
            ctx.set_result(None)

    def test_step_context_set_output_none(self):
        """Cover set_output with falsy output."""
        from ia_modules.telemetry.integration import PipelineTelemetry
        tel = PipelineTelemetry(enabled=True)
        with tel.trace_step("pipe", "step") as ctx:
            ctx.set_output(None)


class TestNoOpContext:
    """Cover _NoOpContext methods."""

    def test_noop_all_methods(self):
        from ia_modules.telemetry.integration import _NoOpContext
        ctx = _NoOpContext()
        ctx.set_result("x")
        ctx.set_output("y")
        ctx.add_event("e", {"a": 1})
        ctx.set_attribute("k", "v")
        ctx.record_items(5)
        ctx.record_cost(api_calls=1, cost_usd=0.1)


class TestGlobalTelemetryFunctions:
    """Cover get_telemetry, configure_telemetry, and agent/llm variants."""

    def test_configure_and_get_telemetry(self):
        import ia_modules.telemetry.integration as mod
        # Reset global
        mod._global_telemetry = None
        t1 = mod.get_telemetry()
        t2 = mod.get_telemetry()
        assert t1 is t2  # singleton
        # Now configure replaces
        t3 = mod.configure_telemetry(enabled=False)
        assert t3 is not t1
        assert t3.enabled is False
        # Cleanup
        mod._global_telemetry = None

    def test_configure_and_get_agent_telemetry(self):
        import ia_modules.telemetry.integration as mod
        mod._global_agent_telemetry = None
        t1 = mod.get_agent_telemetry()
        t2 = mod.get_agent_telemetry()
        assert t1 is t2
        t3 = mod.configure_agent_telemetry(enabled=False)
        assert t3 is not t1
        mod._global_agent_telemetry = None

    def test_configure_and_get_llm_telemetry(self):
        import ia_modules.telemetry.integration as mod
        mod._global_llm_telemetry = None
        t1 = mod.get_llm_telemetry()
        t2 = mod.get_llm_telemetry()
        assert t1 is t2
        t3 = mod.configure_llm_telemetry(enabled=False)
        assert t3 is not t1
        mod._global_llm_telemetry = None


# ===========================================================================
# 3. tests for ia_modules/telemetry/opentelemetry_exporter.py
# ===========================================================================

class TestOpenTelemetryExporter:
    """Cover OpenTelemetryExporter with mocked otel libs."""

    def _make_exporter(self):
        """Create an exporter with all otel dependencies mocked."""
        with patch.dict("sys.modules", {
            "opentelemetry": MagicMock(),
            "opentelemetry.metrics": MagicMock(),
            "opentelemetry.sdk": MagicMock(),
            "opentelemetry.sdk.metrics": MagicMock(),
            "opentelemetry.sdk.metrics.export": MagicMock(),
            "opentelemetry.exporter": MagicMock(),
            "opentelemetry.exporter.otlp": MagicMock(),
            "opentelemetry.exporter.otlp.proto": MagicMock(),
            "opentelemetry.exporter.otlp.proto.grpc": MagicMock(),
            "opentelemetry.exporter.otlp.proto.grpc.metric_exporter": MagicMock(),
            "opentelemetry.exporter.otlp.proto.http": MagicMock(),
            "opentelemetry.exporter.otlp.proto.http.metric_exporter": MagicMock(),
            "opentelemetry.sdk.resources": MagicMock(),
        }):
            # Need to reload the module so it picks up mocked imports
            import importlib
            import ia_modules.telemetry.opentelemetry_exporter as otel_mod
            importlib.reload(otel_mod)
            # After reload, OTEL_AVAILABLE should be True
            assert otel_mod.OTEL_AVAILABLE is True
            exporter = otel_mod.OpenTelemetryExporter(
                endpoint="http://localhost:4317",
                protocol="grpc",
            )
            return exporter, otel_mod

    def test_init_grpc(self):
        exporter, _ = self._make_exporter()
        assert exporter.endpoint == "http://localhost:4317"
        assert exporter.protocol == "grpc"

    def test_init_http(self):
        with patch.dict("sys.modules", {
            "opentelemetry": MagicMock(),
            "opentelemetry.metrics": MagicMock(),
            "opentelemetry.sdk": MagicMock(),
            "opentelemetry.sdk.metrics": MagicMock(),
            "opentelemetry.sdk.metrics.export": MagicMock(),
            "opentelemetry.exporter": MagicMock(),
            "opentelemetry.exporter.otlp": MagicMock(),
            "opentelemetry.exporter.otlp.proto": MagicMock(),
            "opentelemetry.exporter.otlp.proto.grpc": MagicMock(),
            "opentelemetry.exporter.otlp.proto.grpc.metric_exporter": MagicMock(),
            "opentelemetry.exporter.otlp.proto.http": MagicMock(),
            "opentelemetry.exporter.otlp.proto.http.metric_exporter": MagicMock(),
            "opentelemetry.sdk.resources": MagicMock(),
        }):
            import importlib
            import ia_modules.telemetry.opentelemetry_exporter as otel_mod
            importlib.reload(otel_mod)
            exporter = otel_mod.OpenTelemetryExporter(protocol="http")
            assert exporter.protocol == "http"

    def test_init_invalid_protocol(self):
        with patch.dict("sys.modules", {
            "opentelemetry": MagicMock(),
            "opentelemetry.metrics": MagicMock(),
            "opentelemetry.sdk": MagicMock(),
            "opentelemetry.sdk.metrics": MagicMock(),
            "opentelemetry.sdk.metrics.export": MagicMock(),
            "opentelemetry.exporter": MagicMock(),
            "opentelemetry.exporter.otlp": MagicMock(),
            "opentelemetry.exporter.otlp.proto": MagicMock(),
            "opentelemetry.exporter.otlp.proto.grpc": MagicMock(),
            "opentelemetry.exporter.otlp.proto.grpc.metric_exporter": MagicMock(),
            "opentelemetry.exporter.otlp.proto.http": MagicMock(),
            "opentelemetry.exporter.otlp.proto.http.metric_exporter": MagicMock(),
            "opentelemetry.sdk.resources": MagicMock(),
        }):
            import importlib
            import ia_modules.telemetry.opentelemetry_exporter as otel_mod
            importlib.reload(otel_mod)
            with pytest.raises(ValueError, match="Unsupported protocol"):
                otel_mod.OpenTelemetryExporter(protocol="websocket")

    def test_export_metrics(self):
        from ia_modules.telemetry.metrics import Metric, MetricType
        exporter, _ = self._make_exporter()

        counter_metric = Metric(
            name="requests", metric_type=MetricType.COUNTER,
            value=10, labels={"env": "test"}
        )
        gauge_metric = Metric(
            name="connections", metric_type=MetricType.GAUGE,
            value=5, labels={}
        )
        hist_metric = Metric(
            name="latency", metric_type=MetricType.HISTOGRAM,
            value=0.123, labels={}
        )
        hist_dict_metric = Metric(
            name="latency2", metric_type=MetricType.HISTOGRAM,
            value={"observations": [0.1, 0.2, 0.3]}, labels={}
        )
        hist_sum_count = Metric(
            name="latency3", metric_type=MetricType.HISTOGRAM,
            value={"sum": 1.5, "count": 10}, labels={}
        )
        summary_metric = Metric(
            name="duration", metric_type=MetricType.SUMMARY,
            value=0.5, labels={}
        )

        exporter.export([
            counter_metric, gauge_metric, hist_metric,
            hist_dict_metric, hist_sum_count, summary_metric,
        ])

    def test_export_metric_error_logged(self):
        from ia_modules.telemetry.metrics import Metric, MetricType
        exporter, _ = self._make_exporter()
        # Make _export_metric raise
        exporter._export_metric = MagicMock(side_effect=RuntimeError("boom"))
        m = Metric(name="x", metric_type=MetricType.COUNTER, value=1)
        exporter.export([m])  # should not raise, error is logged

    def test_create_instrument_counter(self):
        from ia_modules.telemetry.metrics import MetricType
        exporter, _ = self._make_exporter()
        exporter._create_instrument("test", MetricType.COUNTER, "desc")

    def test_create_instrument_gauge(self):
        from ia_modules.telemetry.metrics import MetricType
        exporter, _ = self._make_exporter()
        exporter._create_instrument("test", MetricType.GAUGE, "desc")

    def test_create_instrument_histogram(self):
        from ia_modules.telemetry.metrics import MetricType
        exporter, _ = self._make_exporter()
        exporter._create_instrument("test", MetricType.HISTOGRAM, "desc")

    def test_create_instrument_summary(self):
        from ia_modules.telemetry.metrics import MetricType
        exporter, _ = self._make_exporter()
        exporter._create_instrument("test", MetricType.SUMMARY, "desc")

    def test_shutdown(self):
        exporter, _ = self._make_exporter()
        exporter.shutdown()

    def test_shutdown_no_provider(self):
        exporter, _ = self._make_exporter()
        del exporter.provider
        exporter.shutdown()  # should not raise

    def test_export_gauge_no_add(self):
        """Cover gauge branch where instrument has no 'add' method."""
        from ia_modules.telemetry.metrics import Metric, MetricType
        exporter, _ = self._make_exporter()
        # Create a gauge instrument that doesn't have 'add'
        gauge_metric = Metric(
            name="test_gauge_no_add", metric_type=MetricType.GAUGE,
            value=10, labels={}
        )
        # First export to create instrument, then remove 'add'
        exporter._export_metric(gauge_metric)
        key = f"{exporter.format_metric_name('test_gauge_no_add')}_gauge"
        instrument = exporter._instruments[key]
        if hasattr(instrument, 'add'):
            del instrument.add
        exporter._export_metric(gauge_metric)  # should hit else branch


class TestPrometheusRemoteWriteExporter:
    """Cover PrometheusRemoteWriteExporter with mocked prometheus_client."""

    def test_init_missing_prometheus_client(self):
        import ia_modules.telemetry.opentelemetry_exporter as otel_mod
        with patch.dict("sys.modules", {"prometheus_client": None}):
            with pytest.raises(ImportError):
                otel_mod.PrometheusRemoteWriteExporter(endpoint="http://prom:9091")

    def test_export_and_push(self):
        mock_prom = MagicMock()
        mock_registry = MagicMock()
        mock_prom.CollectorRegistry.return_value = mock_registry
        mock_counter = MagicMock()
        mock_gauge = MagicMock()
        mock_histogram = MagicMock()
        mock_prom.Counter = mock_counter
        mock_prom.Gauge = mock_gauge
        mock_prom.Histogram = mock_histogram

        with patch.dict("sys.modules", {"prometheus_client": mock_prom}):
            import importlib
            import ia_modules.telemetry.opentelemetry_exporter as otel_mod
            importlib.reload(otel_mod)

            exporter = otel_mod.PrometheusRemoteWriteExporter(endpoint="http://prom:9091")

            from ia_modules.telemetry.metrics import Metric, MetricType
            counter_m = Metric(name="req", metric_type=MetricType.COUNTER,
                               value=5, labels={"env": "test"})
            gauge_m = Metric(name="conn", metric_type=MetricType.GAUGE,
                             value=3, labels={})
            hist_m = Metric(name="lat", metric_type=MetricType.HISTOGRAM,
                            value={"observations": [0.1, 0.2]}, labels={})
            hist_simple = Metric(name="lat2", metric_type=MetricType.HISTOGRAM,
                                 value=0.5, labels={})
            exporter.export([counter_m, gauge_m, hist_m, hist_simple])

    def test_export_push_failure(self):
        mock_prom = MagicMock()
        mock_prom.CollectorRegistry.return_value = MagicMock()
        mock_prom.push_to_gateway.side_effect = RuntimeError("push failed")

        with patch.dict("sys.modules", {"prometheus_client": mock_prom}):
            import importlib
            import ia_modules.telemetry.opentelemetry_exporter as otel_mod
            importlib.reload(otel_mod)
            exporter = otel_mod.PrometheusRemoteWriteExporter(endpoint="http://prom:9091")
            from ia_modules.telemetry.metrics import Metric, MetricType
            m = Metric(name="x", metric_type=MetricType.COUNTER, value=1)
            exporter.export([m])  # should not raise

    def test_create_collector_unsupported_type(self):
        mock_prom = MagicMock()
        mock_prom.CollectorRegistry.return_value = MagicMock()

        with patch.dict("sys.modules", {"prometheus_client": mock_prom}):
            import importlib
            import ia_modules.telemetry.opentelemetry_exporter as otel_mod
            importlib.reload(otel_mod)
            exporter = otel_mod.PrometheusRemoteWriteExporter(endpoint="http://prom:9091")
            # Create a mock metric type that doesn't match
            from ia_modules.telemetry.metrics import MetricType
            with pytest.raises(ValueError, match="Unsupported metric type"):
                # Pass a string that won't match any branch
                fake_type = MagicMock()
                fake_type.value = "unknown"
                exporter._create_collector("n", fake_type, "d", [])


# ===========================================================================
# 4. tests for ia_modules/benchmarking/framework.py
# ===========================================================================

class TestBenchmarkRunner:
    """Cover BenchmarkRunner edge cases."""

    async def test_run_basic(self):
        from ia_modules.benchmarking.framework import BenchmarkRunner
        config = BenchmarkConfig(iterations=5, warmup_iterations=1)
        runner = BenchmarkRunner(config)

        async def fast_func():
            pass

        result = await runner.run("basic", fast_func)
        assert result.name == "basic"
        assert result.iterations == 5
        assert result.mean_time >= 0

    async def test_run_with_timeout(self):
        from ia_modules.benchmarking.framework import BenchmarkRunner
        config = BenchmarkConfig(iterations=2, warmup_iterations=0, timeout=0.001)
        runner = BenchmarkRunner(config)

        async def slow_func():
            await asyncio.sleep(10)

        result = await runner.run("slow", slow_func)
        # Should have timeout values
        assert result.iterations == 2

    async def test_run_with_exception(self):
        from ia_modules.benchmarking.framework import BenchmarkRunner
        config = BenchmarkConfig(iterations=2, warmup_iterations=0)
        runner = BenchmarkRunner(config)

        async def failing_func():
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError, match="fail"):
            await runner.run("failing", failing_func)

    async def test_run_collect_intermediate(self):
        from ia_modules.benchmarking.framework import BenchmarkRunner
        config = BenchmarkConfig(iterations=3, warmup_iterations=0, collect_intermediate=True)
        runner = BenchmarkRunner(config)

        async def func():
            pass

        result = await runner.run("intermediate", func)
        assert result.raw_times is not None
        assert len(result.raw_times) == 3

    async def test_run_no_warmup(self):
        from ia_modules.benchmarking.framework import BenchmarkRunner
        config = BenchmarkConfig(iterations=2, warmup_iterations=0)
        runner = BenchmarkRunner(config)

        async def func():
            pass

        result = await runner.run("no_warmup", func)
        assert result.iterations == 2

    def test_calculate_statistics_empty(self):
        from ia_modules.benchmarking.framework import BenchmarkRunner
        runner = BenchmarkRunner()
        with pytest.raises(ValueError, match="No timing data"):
            runner._calculate_statistics("empty", [])

    def test_calculate_statistics_single(self):
        from ia_modules.benchmarking.framework import BenchmarkRunner
        runner = BenchmarkRunner()
        result = runner._calculate_statistics("single", [0.5])
        assert result.iterations == 1
        assert result.std_dev == 0.0
        assert result.p95_time == 0.5
        assert result.p99_time == 0.5

    def test_calculate_statistics_multiple(self):
        from ia_modules.benchmarking.framework import BenchmarkRunner
        runner = BenchmarkRunner()
        times = [0.1, 0.2, 0.3, 0.15, 0.25]
        result = runner._calculate_statistics("multi", times)
        assert result.iterations == 5
        assert result.min_time == 0.1
        assert result.max_time == 0.3
        assert result.operations_per_second > 0

    async def test_profile_memory_no_psutil(self):
        from ia_modules.benchmarking.framework import BenchmarkRunner
        runner = BenchmarkRunner()

        async def func():
            pass

        with patch.dict("sys.modules", {"psutil": None}):
            result = await runner._profile_memory(func)
            assert result == {}

    async def test_profile_cpu_no_psutil(self):
        from ia_modules.benchmarking.framework import BenchmarkRunner
        runner = BenchmarkRunner()

        async def func():
            pass

        with patch.dict("sys.modules", {"psutil": None}):
            result = await runner._profile_cpu(func)
            assert result == {}

    async def test_run_with_memory_profiling(self):
        from ia_modules.benchmarking.framework import BenchmarkRunner
        config = BenchmarkConfig(iterations=2, warmup_iterations=0, profile_memory=True)
        runner = BenchmarkRunner(config)

        # Mock psutil
        mock_process = MagicMock()
        mock_mem_info = MagicMock()
        mock_mem_info.rss = 100 * 1024 * 1024  # 100 MB
        mock_process.memory_info.return_value = mock_mem_info

        async def func():
            pass

        with patch("ia_modules.benchmarking.framework.BenchmarkRunner._profile_memory",
                    return_value={"before_mb": 100, "after_mb": 105, "delta_mb": 5, "peak_mb": 105}):
            result = await runner.run("mem", func)
            assert result.memory_stats is not None
            assert result.memory_per_operation_mb > 0

    async def test_run_with_cpu_profiling(self):
        from ia_modules.benchmarking.framework import BenchmarkRunner
        config = BenchmarkConfig(iterations=2, warmup_iterations=0, profile_cpu=True)
        runner = BenchmarkRunner(config)

        async def func():
            pass

        with patch("ia_modules.benchmarking.framework.BenchmarkRunner._profile_cpu",
                    return_value={"cpu_percent_before": 10, "cpu_percent_after": 20,
                                  "duration_seconds": 1.0, "cpu_time_seconds": 0.2,
                                  "average_cpu_percent": 15}):
            result = await runner.run("cpu", func)
            assert result.cpu_stats is not None
            assert result.cpu_per_operation_percent == 15


class TestBenchmarkSuite:
    """Cover BenchmarkSuite."""

    async def test_add_benchmark(self):
        from ia_modules.benchmarking.framework import BenchmarkSuite
        config = BenchmarkConfig(iterations=3, warmup_iterations=0)
        suite = BenchmarkSuite("test_suite", config)

        async def func():
            pass

        result = await suite.add_benchmark("bench1", func)
        assert result.name == "bench1"
        assert len(suite.get_results()) == 1

    def test_get_summary_empty(self):
        from ia_modules.benchmarking.framework import BenchmarkSuite
        suite = BenchmarkSuite("empty_suite")
        summary = suite.get_summary()
        assert "No benchmarks run" in summary

    async def test_get_summary_with_results(self):
        from ia_modules.benchmarking.framework import BenchmarkSuite
        config = BenchmarkConfig(iterations=3, warmup_iterations=0)
        suite = BenchmarkSuite("suite", config)

        async def func():
            pass

        await suite.add_benchmark("b1", func)
        summary = suite.get_summary()
        assert "suite Benchmark Suite" in summary
        assert "b1" in summary

    def test_clear_results(self):
        from ia_modules.benchmarking.framework import BenchmarkSuite
        suite = BenchmarkSuite("suite")
        suite.results.append(make_benchmark_result())
        suite.clear_results()
        assert len(suite.results) == 0


# ===========================================================================
# 5. tests for ia_modules/benchmarking/comparison.py
# ===========================================================================

class TestComparisonResult:
    """Cover ComparisonResult.get_summary."""

    def test_summary_improved(self):
        from ia_modules.benchmarking.comparison import (
            ComparisonResult, ComparisonMetric, PerformanceChange
        )
        cr = ComparisonResult(
            metric=ComparisonMetric.MEAN_TIME,
            baseline_value=0.2,
            current_value=0.1,
            delta=-0.1,
            percent_change=-50.0,
            change_classification=PerformanceChange.IMPROVED,
            is_significant=True,
        )
        s = cr.get_summary()
        assert "mean_time" in s
        assert "-50.00%" in s

    def test_summary_regressed(self):
        from ia_modules.benchmarking.comparison import (
            ComparisonResult, ComparisonMetric, PerformanceChange
        )
        cr = ComparisonResult(
            metric=ComparisonMetric.MEAN_TIME,
            baseline_value=0.1,
            current_value=0.2,
            delta=0.1,
            percent_change=100.0,
            change_classification=PerformanceChange.REGRESSED,
            is_significant=True,
        )
        s = cr.get_summary()
        assert "+100.00%" in s

    def test_summary_unchanged(self):
        from ia_modules.benchmarking.comparison import (
            ComparisonResult, ComparisonMetric, PerformanceChange
        )
        cr = ComparisonResult(
            metric=ComparisonMetric.MEAN_TIME,
            baseline_value=0.1,
            current_value=0.1,
            delta=0.0,
            percent_change=0.0,
            change_classification=PerformanceChange.UNCHANGED,
            is_significant=False,
        )
        s = cr.get_summary()
        assert "+0.00%" in s


class TestBenchmarkComparator:
    """Cover BenchmarkComparator."""

    def test_compare_default_metrics(self):
        from ia_modules.benchmarking.comparison import BenchmarkComparator
        cmp = BenchmarkComparator()
        baseline = make_benchmark_result(mean_time=0.1, median_time=0.1, p95_time=0.18, p99_time=0.19)
        current = make_benchmark_result(mean_time=0.2, median_time=0.2, p95_time=0.36, p99_time=0.38)
        results = cmp.compare(baseline, current)
        assert len(results) == 4  # 4 default metrics

    def test_compare_specific_metrics(self):
        from ia_modules.benchmarking.comparison import BenchmarkComparator, ComparisonMetric
        cmp = BenchmarkComparator()
        baseline = make_benchmark_result(max_time=0.2)
        current = make_benchmark_result(max_time=0.1)
        results = cmp.compare(baseline, current, metrics=[ComparisonMetric.MAX_TIME])
        assert len(results) == 1
        assert results[0].change_classification.value == "improved"

    def test_compare_memory_delta(self):
        from ia_modules.benchmarking.comparison import BenchmarkComparator, ComparisonMetric
        cmp = BenchmarkComparator()
        baseline = make_benchmark_result(memory_stats={"delta_mb": 10.0})
        current = make_benchmark_result(memory_stats={"delta_mb": 15.0})
        results = cmp.compare(baseline, current, metrics=[ComparisonMetric.MEMORY_DELTA])
        assert len(results) == 1

    def test_compare_cpu_average(self):
        from ia_modules.benchmarking.comparison import BenchmarkComparator, ComparisonMetric
        cmp = BenchmarkComparator()
        baseline = make_benchmark_result(cpu_stats={"average_cpu_percent": 20.0})
        current = make_benchmark_result(cpu_stats={"average_cpu_percent": 30.0})
        results = cmp.compare(baseline, current, metrics=[ComparisonMetric.CPU_AVERAGE])
        assert len(results) == 1

    def test_compare_none_values(self):
        """When metric is not available, should return None and skip."""
        from ia_modules.benchmarking.comparison import BenchmarkComparator, ComparisonMetric
        cmp = BenchmarkComparator()
        baseline = make_benchmark_result()
        current = make_benchmark_result()
        results = cmp.compare(baseline, current, metrics=[ComparisonMetric.MEMORY_DELTA])
        assert len(results) == 0

    def test_compare_zero_baseline(self):
        """Cover division by zero branch."""
        from ia_modules.benchmarking.comparison import BenchmarkComparator, ComparisonMetric
        cmp = BenchmarkComparator()
        baseline = make_benchmark_result(mean_time=0.0)
        current = make_benchmark_result(mean_time=0.1)
        results = cmp.compare(baseline, current, metrics=[ComparisonMetric.MEAN_TIME])
        assert len(results) == 1
        assert results[0].percent_change == float('inf')

    def test_compare_zero_baseline_zero_delta(self):
        from ia_modules.benchmarking.comparison import BenchmarkComparator, ComparisonMetric
        cmp = BenchmarkComparator()
        baseline = make_benchmark_result(mean_time=0.0)
        current = make_benchmark_result(mean_time=0.0)
        results = cmp.compare(baseline, current, metrics=[ComparisonMetric.MEAN_TIME])
        assert results[0].percent_change == 0.0

    def test_unchanged_classification(self):
        """Small change below threshold -> UNCHANGED."""
        from ia_modules.benchmarking.comparison import BenchmarkComparator, ComparisonMetric, PerformanceChange
        cmp = BenchmarkComparator(significance_threshold=10.0)
        baseline = make_benchmark_result(mean_time=1.0)
        current = make_benchmark_result(mean_time=1.05)
        results = cmp.compare(baseline, current, metrics=[ComparisonMetric.MEAN_TIME])
        assert results[0].change_classification == PerformanceChange.UNCHANGED

    def test_non_time_metric_regression(self):
        """Cover non-time metric branch (increase = regression)."""
        from ia_modules.benchmarking.comparison import BenchmarkComparator, ComparisonMetric, PerformanceChange
        cmp = BenchmarkComparator(significance_threshold=5.0)
        baseline = make_benchmark_result(memory_stats={"delta_mb": 10.0})
        current = make_benchmark_result(memory_stats={"delta_mb": 20.0})
        results = cmp.compare(baseline, current, metrics=[ComparisonMetric.MEMORY_DELTA])
        assert results[0].change_classification == PerformanceChange.REGRESSED

    def test_non_time_metric_improvement(self):
        """Cover non-time metric decrease = improvement."""
        from ia_modules.benchmarking.comparison import BenchmarkComparator, ComparisonMetric, PerformanceChange
        cmp = BenchmarkComparator(significance_threshold=5.0)
        baseline = make_benchmark_result(memory_stats={"delta_mb": 20.0})
        current = make_benchmark_result(memory_stats={"delta_mb": 10.0})
        results = cmp.compare(baseline, current, metrics=[ComparisonMetric.MEMORY_DELTA])
        assert results[0].change_classification == PerformanceChange.IMPROVED

    def test_non_time_metric_unchanged(self):
        """Cover non-time metric unchanged branch."""
        from ia_modules.benchmarking.comparison import BenchmarkComparator, ComparisonMetric, PerformanceChange
        cmp = BenchmarkComparator(significance_threshold=50.0)
        baseline = make_benchmark_result(memory_stats={"delta_mb": 10.0})
        current = make_benchmark_result(memory_stats={"delta_mb": 11.0})
        results = cmp.compare(baseline, current, metrics=[ComparisonMetric.MEMORY_DELTA])
        assert results[0].change_classification == PerformanceChange.UNCHANGED

    def test_has_regression(self):
        from ia_modules.benchmarking.comparison import BenchmarkComparator
        cmp = BenchmarkComparator(regression_threshold=10.0)
        baseline = make_benchmark_result(mean_time=0.1)
        current = make_benchmark_result(mean_time=0.5)
        results = cmp.compare(baseline, current)
        assert cmp.has_regression(results) is True

    def test_no_regression(self):
        from ia_modules.benchmarking.comparison import BenchmarkComparator
        cmp = BenchmarkComparator()
        baseline = make_benchmark_result(mean_time=0.1)
        current = make_benchmark_result(mean_time=0.1)
        results = cmp.compare(baseline, current)
        assert cmp.has_regression(results) is False

    def test_get_summary_empty(self):
        from ia_modules.benchmarking.comparison import BenchmarkComparator
        cmp = BenchmarkComparator()
        assert cmp.get_summary([]) == "No comparison data"

    def test_get_summary_with_regression(self):
        from ia_modules.benchmarking.comparison import BenchmarkComparator
        cmp = BenchmarkComparator(regression_threshold=5.0)
        baseline = make_benchmark_result(mean_time=0.1)
        current = make_benchmark_result(mean_time=0.5)
        results = cmp.compare(baseline, current)
        summary = cmp.get_summary(results)
        assert "WARNING" in summary
        assert "Regressions:" in summary
        assert "Improvements:" in summary

    def test_get_summary_no_regression(self):
        from ia_modules.benchmarking.comparison import BenchmarkComparator
        cmp = BenchmarkComparator()
        baseline = make_benchmark_result(mean_time=0.1)
        current = make_benchmark_result(mean_time=0.1)
        results = cmp.compare(baseline, current)
        summary = cmp.get_summary(results)
        assert "WARNING" not in summary


class TestHistoricalComparator:
    """Cover HistoricalComparator."""

    def test_get_trend_empty(self):
        from ia_modules.benchmarking.comparison import HistoricalComparator, ComparisonMetric
        hc = HistoricalComparator()
        assert hc.get_trend(ComparisonMetric.MEAN_TIME) == {}

    def test_get_trend_single_result(self):
        from ia_modules.benchmarking.comparison import HistoricalComparator, ComparisonMetric
        hc = HistoricalComparator()
        hc.add_result(make_benchmark_result(mean_time=0.1))
        trend = hc.get_trend(ComparisonMetric.MEAN_TIME)
        assert trend["samples"] == 1
        assert trend["mean"] == 0.1

    def test_get_trend_multiple_results(self):
        from ia_modules.benchmarking.comparison import HistoricalComparator, ComparisonMetric
        hc = HistoricalComparator()
        for t in [0.1, 0.2, 0.3, 0.15, 0.25]:
            hc.add_result(make_benchmark_result(mean_time=t))
        trend = hc.get_trend(ComparisonMetric.MEAN_TIME)
        assert trend["samples"] == 5
        assert "slope" in trend
        assert "trend_direction" in trend
        assert "recent_change_percent" in trend

    def test_get_trend_with_window(self):
        from ia_modules.benchmarking.comparison import HistoricalComparator, ComparisonMetric
        hc = HistoricalComparator()
        for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
            hc.add_result(make_benchmark_result(mean_time=t))
        trend = hc.get_trend(ComparisonMetric.MEAN_TIME, window_size=3)
        assert trend["samples"] == 3

    def test_get_trend_zero_denominator(self):
        """Slope calculation with all same x differences canceling."""
        from ia_modules.benchmarking.comparison import HistoricalComparator, ComparisonMetric
        hc = HistoricalComparator()
        # If only 1 value after filtering, no slope
        hc.add_result(make_benchmark_result(mean_time=0.5))
        trend = hc.get_trend(ComparisonMetric.MEAN_TIME)
        assert "slope" not in trend

    def test_get_trend_stable(self):
        """All same values -> slope=0 -> stable."""
        from ia_modules.benchmarking.comparison import HistoricalComparator, ComparisonMetric
        hc = HistoricalComparator()
        for _ in range(5):
            hc.add_result(make_benchmark_result(mean_time=0.1))
        trend = hc.get_trend(ComparisonMetric.MEAN_TIME)
        assert trend["slope"] == 0
        assert trend["trend_direction"] == "stable"

    def test_get_trend_decreasing(self):
        from ia_modules.benchmarking.comparison import HistoricalComparator, ComparisonMetric
        hc = HistoricalComparator()
        for t in [0.5, 0.4, 0.3, 0.2, 0.1]:
            hc.add_result(make_benchmark_result(mean_time=t))
        trend = hc.get_trend(ComparisonMetric.MEAN_TIME)
        assert trend["trend_direction"] == "decreasing"

    def test_get_trend_recent_change_zero_prev(self):
        from ia_modules.benchmarking.comparison import HistoricalComparator, ComparisonMetric
        hc = HistoricalComparator()
        hc.add_result(make_benchmark_result(mean_time=0.0))
        hc.add_result(make_benchmark_result(mean_time=0.1))
        trend = hc.get_trend(ComparisonMetric.MEAN_TIME)
        assert trend["recent_change_percent"] == 0

    def test_get_trend_no_values(self):
        """When metric extraction returns None for all."""
        from ia_modules.benchmarking.comparison import HistoricalComparator, ComparisonMetric
        hc = HistoricalComparator()
        hc.add_result(make_benchmark_result())
        trend = hc.get_trend(ComparisonMetric.MEMORY_DELTA)
        assert trend == {}

    def test_detect_anomalies_too_few(self):
        from ia_modules.benchmarking.comparison import HistoricalComparator, ComparisonMetric
        hc = HistoricalComparator()
        hc.add_result(make_benchmark_result(mean_time=0.1))
        hc.add_result(make_benchmark_result(mean_time=0.2))
        assert hc.detect_anomalies(ComparisonMetric.MEAN_TIME) == []

    def test_detect_anomalies_no_anomalies(self):
        from ia_modules.benchmarking.comparison import HistoricalComparator, ComparisonMetric
        hc = HistoricalComparator()
        for _ in range(5):
            hc.add_result(make_benchmark_result(mean_time=0.1))
        anomalies = hc.detect_anomalies(ComparisonMetric.MEAN_TIME)
        assert anomalies == []  # all same -> std_dev = 0, returns []

    def test_detect_anomalies_with_outlier(self):
        from ia_modules.benchmarking.comparison import HistoricalComparator, ComparisonMetric
        hc = HistoricalComparator()
        for t in [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 10.0]:
            hc.add_result(make_benchmark_result(mean_time=t))
        anomalies = hc.detect_anomalies(ComparisonMetric.MEAN_TIME)
        assert len(anomalies) >= 1
        # The outlier (10.0) should be detected
        assert anomalies[-1][2] > 2.0  # z-score > threshold

    def test_detect_anomalies_no_values(self):
        from ia_modules.benchmarking.comparison import HistoricalComparator, ComparisonMetric
        hc = HistoricalComparator()
        for _ in range(5):
            hc.add_result(make_benchmark_result())
        assert hc.detect_anomalies(ComparisonMetric.MEMORY_DELTA) == []

    def test_clear_history(self):
        from ia_modules.benchmarking.comparison import HistoricalComparator
        hc = HistoricalComparator()
        hc.add_result(make_benchmark_result())
        hc.clear_history()
        assert len(hc.results_history) == 0


# ===========================================================================
# 6. tests for ia_modules/benchmarking/telemetry_bridge.py
# ===========================================================================

class TestBenchmarkTelemetryBridge:
    """Cover BenchmarkTelemetryBridge."""

    def test_export_result(self):
        from ia_modules.benchmarking.telemetry_bridge import BenchmarkTelemetryBridge
        from ia_modules.telemetry.integration import PipelineTelemetry
        tel = PipelineTelemetry(enabled=True)
        bridge = BenchmarkTelemetryBridge(tel)
        result = make_benchmark_result(total_time=2.0, items_processed=10)
        bridge.export_result("pipe", result)

    def test_export_result_disabled(self):
        from ia_modules.benchmarking.telemetry_bridge import BenchmarkTelemetryBridge
        from ia_modules.telemetry.integration import PipelineTelemetry
        tel = PipelineTelemetry(enabled=False)
        bridge = BenchmarkTelemetryBridge(tel)
        result = make_benchmark_result()
        bridge.export_result("pipe", result)  # should do nothing

    def test_export_result_error(self):
        from ia_modules.benchmarking.telemetry_bridge import BenchmarkTelemetryBridge
        tel = MagicMock()
        tel.enabled = True
        tel.record_benchmark_result.side_effect = RuntimeError("export fail")
        bridge = BenchmarkTelemetryBridge(tel)
        result = make_benchmark_result()
        bridge.export_result("pipe", result)  # should not raise

    def test_export_results_multiple(self):
        from ia_modules.benchmarking.telemetry_bridge import BenchmarkTelemetryBridge
        from ia_modules.telemetry.integration import PipelineTelemetry
        tel = PipelineTelemetry(enabled=True)
        bridge = BenchmarkTelemetryBridge(tel)
        results = [make_benchmark_result(name=f"r{i}", total_time=1.0) for i in range(3)]
        bridge.export_results("pipe", results)


class TestTelemetryBridgeGlobalFunctions:
    """Cover get_bridge and configure_bridge."""

    def test_get_bridge(self):
        import ia_modules.benchmarking.telemetry_bridge as mod
        mod._global_bridge = None
        from ia_modules.telemetry.integration import PipelineTelemetry
        tel = PipelineTelemetry(enabled=True)
        b1 = mod.get_bridge(tel)
        b2 = mod.get_bridge(tel)
        assert b1 is b2
        mod._global_bridge = None

    def test_configure_bridge(self):
        import ia_modules.benchmarking.telemetry_bridge as mod
        mod._global_bridge = None
        from ia_modules.telemetry.integration import PipelineTelemetry
        tel = PipelineTelemetry(enabled=True)
        b1 = mod.get_bridge(tel)
        b2 = mod.configure_bridge(tel)
        assert b2 is not b1
        mod._global_bridge = None


# ===========================================================================
# 7. tests for ia_modules/tools/advanced_executor.py
# ===========================================================================

class TestAdvancedToolExecutor:
    """Cover AdvancedToolExecutor methods."""

    def _make_executor(self):
        from ia_modules.tools.advanced_executor import AdvancedToolExecutor
        executor = AdvancedToolExecutor(
            enable_caching=True,
            max_concurrent=5,
        )
        return executor

    def test_init_defaults(self):
        from ia_modules.tools.advanced_executor import AdvancedToolExecutor
        executor = AdvancedToolExecutor()
        assert executor.resource_limits.max_concurrent == 10
        assert executor.default_retry_config.max_attempts == 3

    def test_init_custom_configs(self):
        from ia_modules.tools.advanced_executor import AdvancedToolExecutor
        from ia_modules.tools.error_handling import RetryConfig, CircuitBreakerConfig
        retry = RetryConfig(max_attempts=5)
        cb = CircuitBreakerConfig(failure_threshold=10)
        executor = AdvancedToolExecutor(
            default_retry_config=retry,
            default_circuit_breaker_config=cb,
        )
        assert executor.default_retry_config.max_attempts == 5
        assert executor.default_circuit_breaker_config.failure_threshold == 10

    def test_register_tool(self):
        executor = self._make_executor()
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        executor.registry = MagicMock()
        executor.register_tool(mock_tool, version="2.0.0", capabilities=["search"])
        executor.registry.register_versioned.assert_called_once_with(
            mock_tool, version="2.0.0", capabilities=["search"], set_as_default=True
        )

    async def test_execute_tool_no_retry(self):
        executor = self._make_executor()
        executor.registry = MagicMock()
        executor.registry.execute = AsyncMock(return_value="result")
        result = await executor.execute_tool(
            "tool1", {"param": "val"}, retry=False,
            use_circuit_breaker=False, fallback_tools=None,
        )
        assert result == "result"

    async def test_execute_tool_with_cache(self):
        executor = self._make_executor()
        executor.registry = MagicMock()
        executor.registry.execute_cached = AsyncMock(return_value="cached_result")
        # retry=False so it goes through execute_func directly
        result = await executor.execute_tool(
            "tool1", {"param": "val"}, retry=False,
            cache_ttl=60.0,
        )
        assert result == "cached_result"

    async def test_execute_tool_with_retry(self):
        executor = self._make_executor()
        executor.registry = MagicMock()
        executor.registry.execute = AsyncMock(return_value="retry_result")

        # Mock CompositeErrorHandler.execute
        with patch("ia_modules.tools.advanced_executor.CompositeErrorHandler") as MockHandler:
            handler_instance = MagicMock()
            handler_instance.execute = AsyncMock(return_value="retry_result")
            MockHandler.return_value = handler_instance

            result = await executor.execute_tool(
                "tool1", {"param": "val"}, retry=True,
            )
            assert result == "retry_result"

    async def test_execute_tool_with_fallbacks(self):
        executor = self._make_executor()
        executor.registry = MagicMock()
        executor.registry.execute = AsyncMock(return_value="fallback_result")

        with patch("ia_modules.tools.advanced_executor.CompositeErrorHandler") as MockHandler:
            handler_instance = MagicMock()
            handler_instance.execute = AsyncMock(return_value="fallback_result")
            MockHandler.return_value = handler_instance

            result = await executor.execute_tool(
                "tool1", {"param": "val"}, retry=False,
                fallback_tools=["tool2", "tool3"],
            )
            assert result == "fallback_result"

    async def test_execute_tool_with_circuit_breaker(self):
        executor = self._make_executor()
        executor.default_circuit_breaker_config = MagicMock()
        executor.registry = MagicMock()
        executor.registry.execute = AsyncMock(return_value="cb_result")

        with patch("ia_modules.tools.advanced_executor.CompositeErrorHandler") as MockHandler:
            handler_instance = MagicMock()
            handler_instance.execute = AsyncMock(return_value="cb_result")
            MockHandler.return_value = handler_instance

            result = await executor.execute_tool(
                "tool1", {"param": "val"}, retry=False,
                use_circuit_breaker=True,
            )
            assert result == "cb_result"

    async def test_execute_chain(self):
        executor = self._make_executor()
        mock_chain_result = MagicMock()
        mock_chain_result.success = True

        with patch("ia_modules.tools.advanced_executor.ToolChain") as MockChain:
            chain_instance = MagicMock()
            chain_instance.execute = AsyncMock(return_value=mock_chain_result)
            chain_instance.steps = []
            MockChain.return_value = chain_instance

            steps = [
                {
                    "tool_name": "t1",
                    "input_mapping": {"a": "b"},
                    "output_key": "out1",
                    "on_error": "skip",
                },
            ]
            result = await executor.execute_chain(steps, initial_context={"b": "val"})
            assert result.success is True

    async def test_execute_chain_with_condition(self):
        executor = self._make_executor()
        mock_chain_result = MagicMock()
        mock_chain_result.success = True

        with patch("ia_modules.tools.advanced_executor.ToolChain") as MockChain:
            chain_instance = MagicMock()
            chain_instance.execute = AsyncMock(return_value=mock_chain_result)
            chain_instance.steps = []
            MockChain.return_value = chain_instance

            steps = [
                {
                    "tool_name": "t1",
                    "input_mapping": {"a": "b"},
                    "output_key": "out1",
                    "condition": lambda ctx: True,
                    "parallel_group": "g1",
                },
            ]
            result = await executor.execute_chain(steps)
            assert result.success is True

    async def test_execute_parallel(self):
        executor = self._make_executor()

        with patch("ia_modules.tools.advanced_executor.ParallelExecutor") as MockPE:
            pe_instance = MagicMock()
            pe_instance.execute_all = AsyncMock(return_value={"task_0": "r0", "task_1": "r1"})
            MockPE.return_value = pe_instance

            tasks = [("tool1", {"p": 1}), ("tool2", {"p": 2})]
            results = await executor.execute_parallel(tasks)
            assert "task_0" in results

    async def test_execute_parallel_with_deps(self):
        executor = self._make_executor()

        with patch("ia_modules.tools.advanced_executor.ParallelExecutor") as MockPE:
            pe_instance = MagicMock()
            pe_instance.execute_all = AsyncMock(return_value={"task_0": "r0", "task_1": "r1"})
            MockPE.return_value = pe_instance

            tasks = [("tool1", {"p": 1}), ("tool2", {"p": 2})]
            deps = {"1": [0]}
            results = await executor.execute_parallel(tasks, dependencies=deps)
            assert "task_1" in results

    async def test_execute_task(self):
        executor = self._make_executor()

        mock_plan = MagicMock()
        mock_plan.steps = []
        mock_plan.complexity.value = "simple"
        mock_plan.estimated_time = 1.0
        mock_plan.confidence = 0.9

        mock_chain_result = MagicMock()
        mock_chain_result.success = True
        mock_chain_result.context = {"out": "val"}
        mock_chain_result.steps_executed = ["s1"]
        mock_chain_result.steps_skipped = []
        mock_chain_result.errors = {}

        executor.planner = MagicMock()
        executor.planner.create_plan = AsyncMock(return_value=mock_plan)
        executor.planner.validate_plan.return_value = (True, [])
        executor.planner.optimize_plan.return_value = mock_plan

        executor.execute_plan = AsyncMock(return_value=mock_chain_result)

        result = await executor.execute_task(
            "do something", requirements=["search"],
            context={"k": "v"}, optimize_plan=True, max_alternatives=2,
        )
        assert result["success"] is True
        assert result["plan"]["complexity"] == "simple"

    async def test_execute_task_invalid_plan(self):
        executor = self._make_executor()

        mock_plan = MagicMock()
        executor.planner = MagicMock()
        executor.planner.create_plan = AsyncMock(return_value=mock_plan)
        executor.planner.validate_plan.return_value = (False, ["missing tool"])

        with pytest.raises(ValueError, match="Invalid execution plan"):
            await executor.execute_task("bad task")

    async def test_execute_task_no_optimize(self):
        executor = self._make_executor()

        mock_plan = MagicMock()
        mock_plan.steps = []
        mock_plan.complexity.value = "simple"
        mock_plan.estimated_time = 0.0
        mock_plan.confidence = 1.0

        mock_chain_result = MagicMock()
        mock_chain_result.success = True
        mock_chain_result.context = {}
        mock_chain_result.steps_executed = []
        mock_chain_result.steps_skipped = []
        mock_chain_result.errors = {}

        executor.planner = MagicMock()
        executor.planner.create_plan = AsyncMock(return_value=mock_plan)
        executor.planner.validate_plan.return_value = (True, [])

        executor.execute_plan = AsyncMock(return_value=mock_chain_result)

        result = await executor.execute_task("task", optimize_plan=False)
        executor.planner.optimize_plan.assert_not_called()

    async def test_execute_plan(self):
        executor = self._make_executor()

        from ia_modules.tools.tool_planner import ExecutionPlan, Task, TaskComplexity

        task = Task(description="test task")
        plan = ExecutionPlan(
            task=task,
            steps=[
                {"tool_name": "t1", "capability": "search", "index": 0,
                 "input_mapping": {"q": "query"}},
                {"tool_name": "t2", "capability": "summarize", "index": 1,
                 "dependencies": [0]},
            ],
            complexity=TaskComplexity.MODERATE,
        )

        mock_chain_result = MagicMock()
        mock_chain_result.success = True

        with patch("ia_modules.tools.advanced_executor.ToolChain") as MockChain:
            chain_instance = MagicMock()
            chain_instance.execute = AsyncMock(return_value=mock_chain_result)
            chain_instance.steps = [MagicMock(), MagicMock()]
            MockChain.return_value = chain_instance

            result = await executor.execute_plan(plan, context={"query": "AI"})
            assert result.success is True

    def test_infer_input_mapping_no_tool(self):
        executor = self._make_executor()
        executor.registry = MagicMock()
        executor.registry.get_tool.return_value = None

        from ia_modules.tools.tool_planner import ExecutionPlan, Task
        plan = ExecutionPlan(task=Task(description="test"), steps=[])
        mapping = executor._infer_input_mapping({"tool_name": "missing"}, plan)
        assert mapping == {}

    def test_infer_input_mapping_with_tool_and_deps(self):
        executor = self._make_executor()
        mock_tool = MagicMock()
        mock_tool.parameters = {"query": {"type": "string"}, "limit": {"type": "int"}}
        executor.registry = MagicMock()
        executor.registry.get_tool.return_value = mock_tool

        from ia_modules.tools.tool_planner import ExecutionPlan, Task
        plan = ExecutionPlan(
            task=Task(description="test"),
            steps=[
                {"tool_name": "t0", "capability": "search", "index": 0},
                {"tool_name": "t1", "capability": "summarize", "index": 1,
                 "dependencies": [0]},
            ],
        )
        step = plan.steps[1]
        mapping = executor._infer_input_mapping(step, plan)
        assert len(mapping) == 2

    def test_infer_input_mapping_no_deps(self):
        executor = self._make_executor()
        mock_tool = MagicMock()
        mock_tool.parameters = {"query": {"type": "string"}}
        executor.registry = MagicMock()
        executor.registry.get_tool.return_value = mock_tool

        from ia_modules.tools.tool_planner import ExecutionPlan, Task
        plan = ExecutionPlan(task=Task(description="test"), steps=[])
        step = {"tool_name": "t0", "dependencies": []}
        mapping = executor._infer_input_mapping(step, plan)
        assert mapping == {"query": "query"}

    def test_get_statistics(self):
        executor = self._make_executor()
        executor.registry = MagicMock()
        executor.registry.get_statistics.return_value = {"calls": 5}
        executor.registry.cache = {"k": "v"}
        executor.registry.tools = {"t1": ["v1"]}
        executor.registry.list_capabilities.return_value = ["search"]

        stats = executor.get_statistics()
        assert stats["cache_size"] == 1
        assert stats["total_tools"] == 1
        assert stats["capabilities"] == ["search"]

    def test_clear_cache(self):
        executor = self._make_executor()
        executor.registry = MagicMock()
        executor.registry.clear_cache.return_value = 5
        cleared = executor.clear_cache()
        assert cleared == 5

    def test_list_tools(self):
        executor = self._make_executor()
        executor.registry = MagicMock()
        executor.registry.list_tools.return_value = [{"name": "t1"}]
        tools = executor.list_tools(capability="search", include_deprecated=True)
        executor.registry.list_tools.assert_called_once_with(
            include_deprecated=True, capability="search"
        )

    def test_export_catalog(self):
        executor = self._make_executor()
        executor.registry = MagicMock()
        executor.registry.export_tool_catalog.return_value = {"tools": []}
        catalog = executor.export_catalog()
        assert "tools" in catalog

    def test_create_tool_executor(self):
        executor = self._make_executor()
        func = executor._create_tool_executor()
        assert callable(func)
