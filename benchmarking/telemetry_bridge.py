"""
Telemetry Bridge for Benchmarking

Bridges benchmark results to telemetry exporters for production monitoring.
"""

from typing import Optional, List, TYPE_CHECKING
import logging

from .framework import BenchmarkResult

if TYPE_CHECKING:
    from ..telemetry.integration import PipelineTelemetry

logger = logging.getLogger(__name__)


class BenchmarkTelemetryBridge:
    """
    Bridge benchmark results to telemetry system.

    Automatically exports benchmark metrics to configured telemetry exporters.
    """

    def __init__(self, telemetry: Optional['PipelineTelemetry'] = None):
        """
        Initialize telemetry bridge.

        Args:
            telemetry: PipelineTelemetry instance (creates/gets global if None)
        """
        if telemetry is None:
            from ..telemetry.integration import get_telemetry
            telemetry = get_telemetry()
        self.telemetry = telemetry

    def export_result(
        self,
        pipeline_name: str,
        result: BenchmarkResult
    ):
        """
        Export benchmark result to telemetry.

        Args:
            pipeline_name: Name of the pipeline
            result: Benchmark result to export
        """
        if not self.telemetry or not self.telemetry.enabled:
            return

        try:
            self.telemetry.record_benchmark_result(pipeline_name, result)
            logger.debug(f"Exported benchmark result for {pipeline_name} to telemetry")

        except Exception as e:
            logger.error(f"Failed to export benchmark result to telemetry: {e}")

    def export_results(
        self,
        pipeline_name: str,
        results: List[BenchmarkResult]
    ):
        """
        Export multiple benchmark results to telemetry.

        Args:
            pipeline_name: Name of the pipeline
            results: List of benchmark results
        """
        for result in results:
            self.export_result(pipeline_name, result)


# Global bridge instance
_global_bridge: Optional[BenchmarkTelemetryBridge] = None


def get_bridge(telemetry: Optional['PipelineTelemetry'] = None) -> BenchmarkTelemetryBridge:
    """
    Get or create global telemetry bridge.

    Args:
        telemetry: PipelineTelemetry instance

    Returns:
        BenchmarkTelemetryBridge instance
    """
    global _global_bridge

    if _global_bridge is None:
        _global_bridge = BenchmarkTelemetryBridge(telemetry)

    return _global_bridge


def configure_bridge(telemetry: Optional['PipelineTelemetry'] = None) -> BenchmarkTelemetryBridge:
    """
    Configure global telemetry bridge.

    Args:
        telemetry: PipelineTelemetry instance

    Returns:
        New BenchmarkTelemetryBridge instance
    """
    global _global_bridge
    _global_bridge = BenchmarkTelemetryBridge(telemetry)
    return _global_bridge
