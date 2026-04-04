"""API package"""

import sys
from pathlib import Path

# Remove conflicting ia_modules/pipeline from sys.path (has a services.py that conflicts)
sys.path = [p for p in sys.path if 'ia_modules' not in p or 'showcase_app' in p]

# Add backend to sys.path for services imports
_backend_dir = str(Path(__file__).parent.parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

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
from .patterns import router as patterns_router  # Re-enabled after sys.path fix
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
