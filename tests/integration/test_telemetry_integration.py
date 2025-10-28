"""
Integration tests for telemetry system with pipelines
"""

import pytest
import asyncio
from typing import Dict, Any

from ia_modules.pipeline.core import Step, Pipeline
from ia_modules.pipeline.test_utils import create_test_execution_context
from ia_modules.pipeline.services import ServiceRegistry
from ia_modules.telemetry import (
    PipelineTelemetry,
    MetricsCollector,
    SimpleTracer,
    PrometheusExporter
)
from ia_modules.telemetry.integration import get_telemetry, configure_telemetry
from ia_modules.benchmarking import BenchmarkResult, BenchmarkTelemetryBridge


class SimpleStep(Step):
    """Simple test step"""

    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.01)  # Simulate work
        # Get value from either 'input' or 'result' field
        value = data.get("result", data.get("input", 0))
        return {
            "result": value * 2,
            "step_name": self.name
        }


class FailingStep(Step):
    """Step that always fails"""

    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        raise ValueError("Intentional test failure")


@pytest.fixture
def telemetry():
    """Create fresh telemetry instance for each test"""
    collector = MetricsCollector()
    tracer = SimpleTracer()
    return PipelineTelemetry(collector=collector, tracer=tracer, enabled=True)


@pytest.fixture
def simple_pipeline(telemetry):
    """Create a simple test pipeline"""
    steps = [
        SimpleStep("step1", {}),
        SimpleStep("step2", {}),
        SimpleStep("step3", {})
    ]

    flow = {
        "start_at": "step1",
        "paths": [
            {"from": "step1", "to": "step2"},
            {"from": "step2", "to": "step3"},
            {"from": "step3", "to": "end_with_success"}
        ]
    }

    services = ServiceRegistry()

    # Configure global telemetry
    configure_telemetry(collector=telemetry.collector, tracer=telemetry.tracer)

    return Pipeline("test_pipeline", steps, flow, services, enable_telemetry=True)


class TestPipelineTelemetryIntegration:
    """Test telemetry integration with pipeline execution"""

    @pytest.mark.asyncio
    async def test_pipeline_execution_metrics(self, simple_pipeline, telemetry):
        """Test that pipeline execution generates metrics"""
        # Run pipeline
        input_data = {"input": 5}
        result = await simple_pipeline.run(input_data)

        # Verify pipeline completed (step1: 5*2=10, step2: 10*2=20, step3: 20*2=40)
        assert "output" in result
        assert result["output"]["result"] == 40  # 5 * 2 * 2 * 2

        # Verify metrics were collected
        metrics = telemetry.get_metrics()

        # Should have execution counter
        execution_metrics = [m for m in metrics if "executions" in m.name]
        assert len(execution_metrics) > 0

        # Should have duration histogram
        duration_metrics = [m for m in metrics if "duration" in m.name]
        assert len(duration_metrics) > 0

    @pytest.mark.asyncio
    async def test_pipeline_execution_traces(self, simple_pipeline, telemetry):
        """Test that pipeline execution generates traces"""
        # Run pipeline
        input_data = {"input": 10}
        await simple_pipeline.run(input_data)

        # Get all spans
        spans = telemetry.get_spans()

        # Should have pipeline span + 3 step spans
        assert len(spans) >= 4

        # Verify pipeline span
        pipeline_spans = [s for s in spans if s.name.startswith("pipeline.")]
        assert len(pipeline_spans) == 1
        assert pipeline_spans[0].status == "ok"

        # Verify step spans
        step_spans = [s for s in spans if s.name.startswith("step.")]
        assert len(step_spans) == 3

        # All spans should be successful
        for span in step_spans:
            assert span.status == "ok"

    @pytest.mark.asyncio
    async def test_pipeline_error_tracking(self, telemetry):
        """Test that errors are tracked in telemetry"""
        # Create pipeline with failing step
        steps = [
            SimpleStep("step1", {}),
            FailingStep("failing_step", {"error_handling": {"continue_on_error": False}})
        ]

        flow = {
            "start_at": "step1",
            "paths": [
                {"from": "step1", "to": "failing_step"},
                {"from": "failing_step", "to": "end_with_success"}
            ]
        }

        # Configure global telemetry
        configure_telemetry(collector=telemetry.collector, tracer=telemetry.tracer)

        pipeline = Pipeline("error_pipeline", steps, flow, ServiceRegistry(), enable_telemetry=True)

        # Run pipeline and expect failure
        with pytest.raises(Exception):
            await pipeline.run({"input": 1}, create_test_execution_context())

        # Get spans
        spans = telemetry.get_spans()

        # Pipeline span should show error
        pipeline_spans = [s for s in spans if s.name.startswith("pipeline.")]
        assert len(pipeline_spans) == 1
        assert pipeline_spans[0].status == "error"

        # Failing step should show error
        failing_spans = [s for s in spans if "failing_step" in s.name]
        assert len(failing_spans) >= 1
        assert failing_spans[0].status == "error"

        # Check metrics for errors
        metrics = telemetry.get_metrics()
        error_metrics = [m for m in metrics if "error" in m.name]
        assert len(error_metrics) > 0

    @pytest.mark.asyncio
    async def test_nested_span_hierarchy(self, simple_pipeline, telemetry):
        """Test that spans are properly nested"""
        # Run pipeline
        await simple_pipeline.run({"input": 1}, create_test_execution_context())

        # Get all spans
        spans = telemetry.get_spans()

        # Get pipeline span
        pipeline_spans = [s for s in spans if s.name.startswith("pipeline.")]
        assert len(pipeline_spans) == 1
        pipeline_span = pipeline_spans[0]

        # Get step spans
        step_spans = [s for s in spans if s.name.startswith("step.")]

        # Step spans should have pipeline span as parent
        for step_span in step_spans:
            assert step_span.parent_span_id == pipeline_span.span_id
            assert step_span.trace_id == pipeline_span.trace_id


class TestBenchmarkTelemetryBridge:
    """Test benchmark to telemetry bridge"""

    def test_export_benchmark_result(self, telemetry):
        """Test exporting benchmark result to telemetry"""
        # Create benchmark result
        result = BenchmarkResult(
            name="test_benchmark",
            iterations=100,
            mean_time=0.05,
            median_time=0.05,
            std_dev=0.005,
            min_time=0.04,
            max_time=0.06,
            p95_time=0.058,
            p99_time=0.059,
            total_time=5.0,
            operations_per_second=20.0,
            items_processed=1000,
            items_per_second=200.0,
            api_calls_count=500,
            estimated_cost_usd=2.50,
            cost_per_operation=0.025
        )

        # Create bridge
        bridge = BenchmarkTelemetryBridge(telemetry)

        # Export result
        bridge.export_result("test_pipeline", result)

        # Verify metrics were recorded
        metrics = telemetry.get_metrics()

        # Should have duration metric
        duration_metrics = [m for m in metrics if "duration" in m.name]
        assert len(duration_metrics) > 0

        # Should have cost metric
        cost_metrics = [m for m in metrics if "cost" in m.name]
        assert len(cost_metrics) > 0

        # Should have items processed
        items_metrics = [m for m in metrics if "items_processed" in m.name]
        assert len(items_metrics) > 0

    def test_export_multiple_results(self, telemetry):
        """Test exporting multiple benchmark results"""
        results = [
            BenchmarkResult(
                name="test1",
                iterations=50,
                mean_time=0.05,
                median_time=0.05,
                std_dev=0.005,
                min_time=0.04,
                max_time=0.06,
                p95_time=0.058,
                p99_time=0.059,
                total_time=2.5,
                operations_per_second=20.0,
                items_processed=500,
                estimated_cost_usd=1.25
            ),
            BenchmarkResult(
                name="test2",
                iterations=50,
                mean_time=0.046,
                median_time=0.046,
                std_dev=0.004,
                min_time=0.04,
                max_time=0.055,
                p95_time=0.053,
                p99_time=0.054,
                total_time=2.3,
                operations_per_second=21.7,
                items_processed=520,
                estimated_cost_usd=1.15
            )
        ]

        bridge = BenchmarkTelemetryBridge(telemetry)
        bridge.export_results("benchmark_pipeline", results)

        # Verify metrics
        metrics = telemetry.get_metrics()
        assert len(metrics) > 0

        # Cost should be sum of both
        cost_metrics = [m for m in metrics if "cost" in m.name and m.metric_type.value == "counter"]
        if cost_metrics:
            # Should have accumulated costs
            assert cost_metrics[0].value >= 2.40  # 1.25 + 1.15


class TestPrometheusExport:
    """Test Prometheus export of telemetry data"""

    @pytest.mark.asyncio
    async def test_prometheus_export_after_pipeline(self, simple_pipeline, telemetry):
        """Test exporting telemetry to Prometheus format"""
        # Run pipeline
        await simple_pipeline.run({"input": 5}, create_test_execution_context())

        # Export to Prometheus
        exporter = PrometheusExporter(prefix="test")
        exporter.export(telemetry.get_metrics())

        # Get metrics text
        metrics_text = exporter.get_metrics_text()

        # Verify Prometheus format
        assert "# HELP" in metrics_text
        assert "# TYPE" in metrics_text
        assert "test_pipeline_executions_total" in metrics_text
        assert "test_pipeline_duration_seconds" in metrics_text

    @pytest.mark.asyncio
    async def test_prometheus_labels(self, simple_pipeline, telemetry):
        """Test that Prometheus export includes labels"""
        # Run pipeline
        await simple_pipeline.run({"input": 3}, create_test_execution_context())

        # Export to Prometheus
        exporter = PrometheusExporter(prefix="app")
        exporter.export(telemetry.get_metrics())

        metrics_text = exporter.get_metrics_text()

        # Should include pipeline name label
        assert 'pipeline_name="test_pipeline"' in metrics_text

        # Should include status label
        assert 'status="success"' in metrics_text


class TestTelemetryDisabled:
    """Test pipeline execution with telemetry disabled"""

    @pytest.mark.asyncio
    async def test_pipeline_without_telemetry(self):
        """Test that pipeline works with telemetry disabled"""
        steps = [SimpleStep("step1", {})]

        flow = {
            "start_at": "step1",
            "paths": [
                {"from": "step1", "to": "end_with_success"}
            ]
        }

        # Create pipeline with telemetry disabled
        pipeline = Pipeline(
            "no_telemetry_pipeline",
            steps,
            flow,
            ServiceRegistry(),
            enable_telemetry=False
        )

        # Should run normally
        result = await pipeline.run({"input": 10}, create_test_execution_context())
        assert result["output"]["result"] == 20


class TestTelemetryPerformance:
    """Test telemetry performance impact"""

    @pytest.mark.asyncio
    async def test_telemetry_overhead(self):
        """Test that telemetry has minimal overhead"""
        import time

        steps = [SimpleStep(f"step{i}", {}) for i in range(5)]

        flow = {
            "start_at": "step0",
            "paths": [
                {"from": f"step{i}", "to": f"step{i+1}"}
                for i in range(4)
            ] + [{"from": "step4", "to": "end_with_success"}]
        }

        # Run with telemetry
        telemetry = PipelineTelemetry(enabled=True)
        configure_telemetry(collector=telemetry.collector, tracer=telemetry.tracer)

        pipeline_with_telemetry = Pipeline(
            "telemetry_pipeline",
            steps,
            flow,
            ServiceRegistry(),
            enable_telemetry=True
        )

        start = time.time()
        await pipeline_with_telemetry.run({"input": 1})
        with_telemetry_time = time.time() - start

        # Run without telemetry
        pipeline_without_telemetry = Pipeline(
            "no_telemetry_pipeline",
            [SimpleStep(f"step{i}", {}) for i in range(5)],
            flow,
            ServiceRegistry(),
            enable_telemetry=False
        )

        start = time.time()
        await pipeline_without_telemetry.run({"input": 1})
        without_telemetry_time = time.time() - start

        # Telemetry overhead should be minimal (< 20%)
        overhead = (with_telemetry_time - without_telemetry_time) / without_telemetry_time
        assert overhead < 0.2, f"Telemetry overhead too high: {overhead * 100:.1f}%"
