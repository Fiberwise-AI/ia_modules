"""Service Container - Dependency Injection Container for Showcase App"""

from typing import Optional
from ia_modules.database.manager import DatabaseManager
from services.metrics_service import MetricsService
from services.pipeline_service import PipelineService
from services.reliability_service import ReliabilityService
from services.scheduler_service import SchedulerService
from services.benchmark_service import BenchmarkService


class ServiceContainer:
    """Container holding all application services"""

    def __init__(self):
        self.db_manager: Optional[DatabaseManager] = None
        self.metrics_service: Optional[MetricsService] = None
        self.pipeline_service: Optional[PipelineService] = None
        self.reliability_service: Optional[ReliabilityService] = None
        self.scheduler_service: Optional[SchedulerService] = None
        self.benchmark_service: Optional[BenchmarkService] = None
