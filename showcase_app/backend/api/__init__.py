"""API package"""

# Export all routers for easy import
from .pipelines import router as pipelines_router
from .execution import router as execution_router
from .metrics import router as metrics_router
from .websocket import router as websocket_router
from .checkpoints import router as checkpoints_router
from .reliability import router as reliability_router
from .scheduler import router as scheduler_router
from .benchmarking import router as benchmarking_router
from .telemetry import router as telemetry_router
from .memory import router as memory_router
from .patterns import router as patterns_router
from .multi_agent import router as multi_agent_router

__all__ = [
    'pipelines_router',
    'execution_router',
    'metrics_router',
    'websocket_router',
    'checkpoints_router',
    'reliability_router',
    'scheduler_router',
    'benchmarking_router',
    'telemetry_router',
    'memory_router',
    'patterns_router',
    'multi_agent_router'
]
