"""
IA Modules Showcase App - FastAPI Backend
Main application entry point
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import asyncio
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from pathlib import Path

from ia_modules.database.manager import DatabaseManager
from ia_modules.reliability.decision_trail import DecisionTrailBuilder
from ia_modules.pipeline.importer import PipelineImportService

from api.pipelines import router as pipelines_router
from api.execution import router as execution_router
from api.metrics import router as metrics_router
from api.websocket import router as websocket_router
from api.checkpoints import router as checkpoints_router
from api.reliability import router as reliability_router
from api.scheduler import router as scheduler_router
from api.benchmarking import router as benchmarking_router
from api.telemetry import router as telemetry_router
from api.memory import router as memory_router
from api.patterns import router as patterns_router
from api.multi_agent import router as multi_agent_router
from api.constitutional_ai_api import router as constitutional_ai_router
from api.advanced_memory_api import router as advanced_memory_router
from api.multimodal_api import router as multimodal_router
from api.agents_api import router as agents_router
from api.prompt_optimization_api import router as prompt_optimization_router
from api.advanced_tools_api import router as advanced_tools_router
from api.step_modules import router as step_modules_router
from api.hitl import router as hitl_router
from services.container import ServiceContainer
from services.metrics_service import MetricsService
from services.pipeline_service import PipelineService
from services.reliability_service import ReliabilityService
from services.scheduler_service import SchedulerService
from services.benchmark_service import BenchmarkService
from services.telemetry_service import TelemetryService
from services.checkpoint_service import CheckpointService
from services.memory_service import MemoryService
from services.replay_service import ReplayService
from services.decision_trail_service import DecisionTrailService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting IA Modules Showcase App...")

    # Initialize service container
    services = ServiceContainer()

    # Initialize database with migrations
    db_url = os.getenv('DATABASE_URL', 'postgresql://localhost/showcase_app')
    services.db_manager = DatabaseManager(db_url)

    # Use built-in migration system - no app-specific migrations needed
    if not await services.db_manager.initialize(apply_schema=True, app_migration_paths=None):
        raise RuntimeError(f"Database initialization failed. Check DATABASE_URL in .env: {db_url}")
    logger.info(f"✓ Database initialized ({services.db_manager.config.database_type.value})")

    # Initialize services
    services.metrics_service = MetricsService(services.db_manager)
    services.pipeline_service = PipelineService(
        services.metrics_service, 
        services.db_manager
    )
    services.reliability_service = ReliabilityService(services.db_manager)
    services.benchmark_service = BenchmarkService(services.pipeline_service, services.db_manager)
    
    # Initialize telemetry service with tracer from pipeline_service
    services.telemetry_service = TelemetryService(
        telemetry=services.pipeline_service.telemetry,
        tracer=services.pipeline_service.tracer
    )
    
    # Initialize checkpoint service with checkpointer from pipeline_service
    services.checkpoint_service = CheckpointService(
        checkpointer=services.pipeline_service.checkpointer,
        pipeline_service=services.pipeline_service
    )
    
    # Initialize memory service (memory backend can be added later if needed)
    services.memory_service = MemoryService(memory_backend=None)
    
    # Initialize replay service
    services.replay_service = ReplayService(
        reliability_metrics=services.reliability_service,
        pipeline_service=services.pipeline_service
    )
    
    # Initialize decision trail builder from ia_modules
    decision_trail_builder = DecisionTrailBuilder(
        state_manager=None,  # Can be added if state management is needed
        tool_registry=None,   # Can be added if tool tracking is needed
        checkpointer=services.pipeline_service.checkpointer
    )
    
    # Initialize decision trail service
    services.decision_trail_service = DecisionTrailService(
        decision_trail_builder=decision_trail_builder,
        reliability_metrics=services.reliability_service
    )

    logger.info("✓ Services initialized successfully")

    # Import test pipelines on startup
    tests_dir = Path(__file__).parent.parent.parent / "tests" / "pipelines"
    importer = PipelineImportService(services.db_manager, str(tests_dir))
    import_results = await importer.import_all_pipelines()
    logger.info(f"✓ Pipeline import: {import_results['imported']} imported, {import_results['updated']} updated, {import_results['skipped']} skipped")

    # Store services on app state
    app.state.services = services

    yield

    # Shutdown
    logger.info("Shutting down services...")
    try:
        if services.scheduler_service:
            await services.scheduler_service.cleanup()
        await services.metrics_service.cleanup()
        if services.db_manager:
            services.db_manager.disconnect()
        logger.info("Shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    finally:
        os._exit(0)


# Create FastAPI app
app = FastAPI(
    title="IA Modules Showcase API",
    description="REST API for IA Modules demonstration app",
    version="0.0.3",
    lifespan=lifespan
)

# CORS middleware - MUST be added BEFORE routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    db_type = "unknown"
    services_status = {}
    
    if hasattr(app.state, 'services'):
        services = app.state.services
        if services.db_manager:
            db_type = services.db_manager.config.database_type.value.upper()
            if db_type == "SQLITE":
                db_type = "SQLite"
            elif db_type == "POSTGRESQL":
                db_type = "PostgreSQL"
            elif db_type == "MEMORY":
                db_type = "In-Memory"
        
        services_status = {
            "metrics": services.metrics_service is not None,
            "pipelines": services.pipeline_service is not None,
            "database": services.db_manager is not None,
        }
    
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "0.0.3",
        "database": db_type,
        "services": services_status
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "IA Modules Showcase API",
        "version": "0.0.3",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "pipelines": "/api/pipelines",
            "execution": "/api/execute",
            "metrics": "/api/metrics",
            "checkpoints": "/api/checkpoints",
            "reliability": "/api/reliability",
            "scheduler": "/api/scheduler",
            "benchmarking": "/api/benchmarking",
            "websocket": "/ws"
        }
    }


# Include routers
app.include_router(pipelines_router, prefix="/api/pipelines", tags=["Pipelines"])
app.include_router(step_modules_router, tags=["Step Modules"])
app.include_router(execution_router, prefix="/api/execute", tags=["Execution"])
app.include_router(metrics_router, prefix="/api/metrics", tags=["Metrics"])
app.include_router(checkpoints_router, prefix="/api/checkpoints", tags=["Checkpoints"])
app.include_router(reliability_router, prefix="/api/reliability", tags=["Reliability"])
app.include_router(scheduler_router, prefix="/api/scheduler", tags=["Scheduler"])
app.include_router(benchmarking_router, prefix="/api/benchmarking", tags=["Benchmarking"])
app.include_router(telemetry_router, prefix="/api/telemetry", tags=["Telemetry"])
app.include_router(memory_router, prefix="/api/memory", tags=["Memory"])
app.include_router(patterns_router, tags=["Patterns"])
app.include_router(multi_agent_router, tags=["Multi-Agent"])
app.include_router(websocket_router, prefix="/ws", tags=["WebSocket"])
app.include_router(hitl_router, prefix="/api/hitl", tags=["Human-in-the-Loop"])

# Advanced AI Features
app.include_router(constitutional_ai_router, prefix="/api/constitutional-ai", tags=["Constitutional AI"])
app.include_router(advanced_memory_router, prefix="/api/advanced-memory", tags=["Advanced Memory"])
app.include_router(multimodal_router, prefix="/api/multimodal", tags=["Multimodal"])
app.include_router(agents_router, prefix="/api/agents", tags=["Agents"])
app.include_router(prompt_optimization_router, prefix="/api/prompt-optimization", tags=["Prompt Optimization"])
app.include_router(advanced_tools_router, prefix="/api/tools", tags=["Advanced Tools"])


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle uncaught exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "type": type(exc).__name__
        }
    )


# Make services accessible to routers
def get_metrics_service() -> MetricsService:
    """Get metrics service instance"""
    return app.state.services.metrics_service


def get_pipeline_service() -> PipelineService:
    """Get pipeline service instance"""
    return app.state.services.pipeline_service


def get_reliability_service() -> ReliabilityService:
    """Get reliability service instance"""
    return app.state.services.reliability_service


def get_scheduler_service() -> SchedulerService:
    """Get scheduler service instance"""
    return app.state.services.scheduler_service


def get_benchmark_service() -> BenchmarkService:
    """Get benchmark service instance"""
    return app.state.services.benchmark_service


def get_memory_service() -> MemoryService:
    """Get memory service instance"""
    return app.state.services.memory_service


def get_replay_service() -> ReplayService:
    """Get replay service instance"""
    return app.state.services.replay_service


def get_decision_trail_service() -> DecisionTrailService:
    """Get decision trail service instance"""
    return app.state.services.decision_trail_service


def get_db_manager() -> DatabaseManager:
    """Get database manager instance"""
    return app.state.services.db_manager


if __name__ == "__main__":
    import uvicorn
    import sys
    import os

    # On Windows, uvicorn sometimes doesn't handle Ctrl+C well
    # Pass app object directly and use try/except
    try:
        uvicorn.run(
            app,  # Pass app object directly instead of "main:app" string
            host="0.0.0.0",
            port=5555,
            reload=False,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("\nServer stopped by user")
        os._exit(0)  # Force immediate exit
