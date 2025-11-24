"""Telemetry service for retrieving execution spans and metrics"""

import logging
from typing import List, Dict, Any
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class TelemetryService:
    """Service for retrieving telemetry data from ia_modules"""

    def __init__(self, telemetry=None, tracer=None):
        """
        Initialize telemetry service
        
        Args:
            telemetry: PipelineTelemetry instance from ia_modules
            tracer: SimpleTracer instance from ia_modules
        """
        self.telemetry = telemetry
        self.tracer = tracer
        logger.info("Telemetry service initialized")

    async def get_execution_spans(self, job_id: str) -> List[Dict[str, Any]]:
        """
        Get all telemetry spans for a specific execution
        
        Args:
            job_id: Execution job ID
            
        Returns:
            List of span dictionaries with execution details
        """
        if not self.tracer:
            logger.warning("No tracer available for telemetry")
            return []

        try:
            # Get all spans from tracer
            all_spans = self.tracer.get_spans()
            
            # Filter spans for this job_id (check attributes)
            job_spans = []
            for span in all_spans:
                try:
                    span_dict = self._span_to_dict(span)
                    
                    # Check if span belongs to this job
                    if span_dict.get("attributes", {}).get("job_id") == job_id:
                        job_spans.append(span_dict)
                except Exception as e:
                    logger.warning(f"Error processing span: {e}")
                    continue
            
            # Sort by start time
            job_spans.sort(key=lambda s: s.get("start_time", ""))
            
            logger.info(f"Retrieved {len(job_spans)} spans for job {job_id}")
            return job_spans

        except Exception as e:
            logger.error(f"Error retrieving spans for job {job_id}: {e}", exc_info=True)
            return []

    async def get_execution_metrics(self, job_id: str) -> Dict[str, Any]:
        """
        Get aggregated metrics for a specific execution
        
        Args:
            job_id: Execution job ID
            
        Returns:
            Dictionary of aggregated metrics
        """
        spans = await self.get_execution_spans(job_id)
        
        if not spans:
            return {
                "total_spans": 0,
                "total_duration_ms": 0,
                "step_count": 0,
                "error_count": 0
            }

        # Aggregate metrics from spans
        total_duration = 0
        step_count = 0
        error_count = 0
        
        for span in spans:
            if span.get("duration_ms"):
                total_duration += span["duration_ms"]
            
            # Count step spans (not pipeline-level)
            if "step" in span.get("name", "").lower():
                step_count += 1
            
            # Count errors
            if span.get("status") == "error":
                error_count += 1

        return {
            "total_spans": len(spans),
            "total_duration_ms": total_duration,
            "step_count": step_count,
            "error_count": error_count,
            "avg_step_duration_ms": total_duration / step_count if step_count > 0 else 0
        }

    async def get_span_timeline(self, job_id: str) -> List[Dict[str, Any]]:
        """
        Get spans formatted for timeline visualization
        
        Args:
            job_id: Execution job ID
            
        Returns:
            List of spans with timeline data (start, end, duration, depth)
        """
        spans = await self.get_execution_spans(job_id)
        
        if not spans:
            return []

        # Calculate depth for nested spans
        timeline = []
        for span in spans:
            timeline_entry = {
                "span_id": span["span_id"],
                "parent_id": span.get("parent_id"),
                "name": span["name"],
                "start_time": span.get("start_time"),
                "end_time": span.get("end_time"),
                "duration_ms": span.get("duration_ms", 0),
                "status": span.get("status", "ok"),
                "depth": self._calculate_span_depth(span, spans),
                "attributes": span.get("attributes", {})
            }
            timeline.append(timeline_entry)

        return timeline

    def _span_to_dict(self, span) -> Dict[str, Any]:
        """Convert span object to dictionary"""
        if isinstance(span, dict):
            return span

        # Convert span object to dict
        span_dict = {
            "span_id": getattr(span, "span_id", None),
            "parent_id": getattr(span, "parent_id", None),
            "name": getattr(span, "name", "unknown"),
            "start_time": None,
            "end_time": None,
            "duration_ms": None,
            "attributes": getattr(span, "attributes", {}),
            "status": getattr(span, "status", "ok")
        }

        # Handle timestamps - may be datetime objects or Unix timestamps (float)
        if hasattr(span, "start_time") and span.start_time:
            if isinstance(span.start_time, datetime):
                span_dict["start_time"] = span.start_time.isoformat()
            elif isinstance(span.start_time, (int, float)):
                span_dict["start_time"] = datetime.fromtimestamp(span.start_time, tz=timezone.utc).isoformat()
            else:
                span_dict["start_time"] = str(span.start_time)

        if hasattr(span, "end_time") and span.end_time:
            if isinstance(span.end_time, datetime):
                span_dict["end_time"] = span.end_time.isoformat()
            elif isinstance(span.end_time, (int, float)):
                span_dict["end_time"] = datetime.fromtimestamp(span.end_time, tz=timezone.utc).isoformat()
            else:
                span_dict["end_time"] = str(span.end_time)

        # Calculate duration
        if hasattr(span, "duration_ms"):
            span_dict["duration_ms"] = span.duration_ms
        elif hasattr(span, "start_time") and hasattr(span, "end_time"):
            if span.start_time and span.end_time:
                # Handle both datetime and Unix timestamp formats
                if isinstance(span.start_time, (int, float)) and isinstance(span.end_time, (int, float)):
                    span_dict["duration_ms"] = (span.end_time - span.start_time) * 1000
                elif isinstance(span.start_time, datetime) and isinstance(span.end_time, datetime):
                    delta = span.end_time - span.start_time
                    span_dict["duration_ms"] = delta.total_seconds() * 1000

        return span_dict

    def _calculate_span_depth(self, span: Dict, all_spans: List[Dict]) -> int:
        """Calculate nesting depth of span"""
        depth = 0
        current_parent = span.get("parent_id")
        
        while current_parent:
            depth += 1
            # Find parent span
            parent_span = next(
                (s for s in all_spans if s.get("span_id") == current_parent),
                None
            )
            if parent_span:
                current_parent = parent_span.get("parent_id")
            else:
                break
        
        return depth
