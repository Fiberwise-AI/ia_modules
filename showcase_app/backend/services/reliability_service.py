"""Reliability and observability service using ia_modules library"""

import logging
from typing import Dict, Any, Optional, List
from ia_modules.reliability import (
    ReliabilityMetrics,
    SQLMetricStorage,
    SLOTracker,
    AnomalyDetector,
    AlertManager,
    CircuitBreakerRegistry,
    CostTracker
)

logger = logging.getLogger(__name__)


class ReliabilityService:
    """Service for reliability metrics and observability using ia_modules"""

    def __init__(self, db_manager):
        self.db_manager = db_manager

        logger.info("Initializing reliability service with ia_modules library...")

        # Initialize metric storage using library's SQLMetricStorage
        storage = SQLMetricStorage(self.db_manager)

        # Initialize reliability metrics
        self.metrics = ReliabilityMetrics(storage=storage)

        # Initialize SLO tracker
        self.slo_tracker = SLOTracker()

        # Initialize anomaly detector
        self.anomaly_detector = AnomalyDetector()

        # Initialize alert manager
        self.alert_manager = AlertManager()

        # Initialize circuit breaker registry
        self.circuit_breaker_registry = CircuitBreakerRegistry()

        # Initialize cost tracker
        self.cost_tracker = CostTracker()

        logger.info("Reliability service initialized")

    async def get_metrics(self, pipeline_id: Optional[str] = None) -> Dict[str, Any]:
        """Get reliability metrics (SR, CR, PC, HIR, MA, TCL, WCT)"""
        try:
            # Get metrics from library
            report = await self.metrics.get_report(pipeline_id=pipeline_id)

            return {
                "success_rate": report.success_rate,
                "compensation_rate": report.compensation_rate,
                "pass_confidence": report.pass_confidence,
                "human_intervention_rate": report.human_intervention_rate,
                "model_accuracy": report.model_accuracy,
                "tool_call_latency_ms": report.tool_call_latency_ms,
                "workflow_completion_time_ms": report.workflow_completion_time_ms,
                "total_executions": report.total_executions,
                "successful_executions": report.successful_executions,
                "failed_executions": report.failed_executions,
                "compensated_executions": report.compensated_executions
            }
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return {}

    async def get_slo_status(self, pipeline_id: Optional[str] = None) -> Dict[str, Any]:
        """Get SLO compliance status"""
        try:
            slo_report = await self.slo_tracker.get_report(pipeline_id=pipeline_id)

            return {
                "compliance_status": slo_report.compliance_status,
                "slos": slo_report.slos,
                "violations": slo_report.violations,
                "mtte": slo_report.mtte,
                "rsr": slo_report.rsr
            }
        except Exception as e:
            logger.error(f"Failed to get SLO status: {e}")
            return {}

    async def get_anomalies(
        self,
        pipeline_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get detected anomalies"""
        try:
            anomalies = await self.anomaly_detector.get_anomalies(
                pipeline_id=pipeline_id,
                limit=limit
            )

            return [
                {
                    "id": anomaly.id,
                    "type": anomaly.type.value,
                    "severity": anomaly.severity.value,
                    "metric_name": anomaly.metric_name,
                    "detected_value": anomaly.detected_value,
                    "expected_range": anomaly.expected_range,
                    "timestamp": anomaly.timestamp.isoformat(),
                    "pipeline_id": anomaly.pipeline_id
                }
                for anomaly in anomalies
            ]
        except Exception as e:
            logger.error(f"Failed to get anomalies: {e}")
            return []

    async def get_alerts(
        self,
        pipeline_id: Optional[str] = None,
        active_only: bool = True,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get active alerts"""
        try:
            alerts = await self.alert_manager.get_alerts(
                pipeline_id=pipeline_id,
                active_only=active_only,
                limit=limit
            )

            return [
                {
                    "id": alert.id,
                    "type": alert.type.value,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "pipeline_id": alert.pipeline_id,
                    "timestamp": alert.timestamp.isoformat(),
                    "resolved": alert.resolved
                }
                for alert in alerts
            ]
        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            return []

    async def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get circuit breaker status for all pipelines"""
        try:
            status = self.circuit_breaker_registry.get_all_status()
            return status
        except Exception as e:
            logger.error(f"Failed to get circuit breaker status: {e}")
            return {}

    async def get_cost_metrics(self, pipeline_id: Optional[str] = None) -> Dict[str, Any]:
        """Get cost tracking metrics"""
        try:
            report = await self.cost_tracker.get_report(pipeline_id=pipeline_id)

            return {
                "total_cost_usd": report.total_cost_usd,
                "cost_by_category": report.cost_by_category,
                "cost_by_pipeline": report.cost_by_pipeline,
                "budget_status": report.budget_status,
                "alerts": report.alerts
            }
        except Exception as e:
            logger.error(f"Failed to get cost metrics: {e}")
            return {}

    async def get_trend_analysis(
        self,
        metric_name: str,
        pipeline_id: Optional[str] = None,
        window_size: int = 10
    ) -> Dict[str, Any]:
        """Get trend analysis for a specific metric"""
        try:
            # This would use the TrendAnalyzer from the library
            # For now, return placeholder
            return {
                "metric_name": metric_name,
                "pipeline_id": pipeline_id,
                "trend_direction": "stable",
                "slope": 0.0,
                "confidence": 0.95,
                "prediction": 0.0
            }
        except Exception as e:
            logger.error(f"Failed to get trend analysis: {e}")
            return {}
