"""
Compatibility shim for pipeline models.

This module re-exports the canonical models and helper functions from
`ia_modules.pipeline.pipeline_models` so older import paths continue to work
while keeping a single source of truth.
"""
from ia_modules.pipeline.pipeline_models import (
    PipelineConfiguration,
    PipelineConfigurationCreate,
    PipelineConfigurationSummary,
    PipelineExecution,
    PipelineExecutionCreate,
    PipelineExecutionSummary,
    PipelineExecutionUpdate,
    PipelineExecutionRequest,
    PipelineExecutionResponse,
    row_to_configuration,
    row_to_configuration_summary,
    row_to_execution,
    row_to_execution_summary,
)

__all__ = [
    'PipelineConfiguration',
    'PipelineConfigurationCreate',
    'PipelineConfigurationSummary',
    'PipelineExecution',
    'PipelineExecutionCreate',
    'PipelineExecutionSummary',
    'PipelineExecutionUpdate',
    'PipelineExecutionRequest',
    'PipelineExecutionResponse',
    'row_to_configuration',
    'row_to_configuration_summary',
    'row_to_execution',
    'row_to_execution_summary',
]
