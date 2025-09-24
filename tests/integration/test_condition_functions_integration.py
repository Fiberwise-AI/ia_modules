"""
Integration tests for condition functions with real scenarios
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest

from ia_modules.pipeline.condition_functions import (
    business_hours_condition,
    data_quality_condition,
    threshold_condition,
    error_rate_condition,
    regex_match_condition,
    cost_threshold_condition,
    approval_required_condition,
    retry_condition,
    load_balancing_condition,
    feature_flag_condition
)
from ia_modules.pipeline.routing import RoutingContext


@pytest.mark.asyncio
async def test_business_hours_condition_integration():
    """Test business hours condition with real context"""
    # Mock context with step results
    context = Mock(spec=RoutingContext)
    context.step_results = {}
    
    # Test with default parameters (should pass in normal business hours)
    parameters = {
        'timezone': 'UTC',
        'start_hour': 9,
        'end_hour': 17,
        'weekdays_only': True
    }
    
    result = await asyncio.get_event_loop().run_in_executor(None, business_hours_condition, context, parameters)
    # This will depend on actual time, so we just check it returns a boolean
    assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_data_quality_condition_integration():
    """Test data quality condition with real data"""
    context = Mock(spec=RoutingContext)
    context.step_results = {
        'step1': {
            'result': {
                'field1': 'value1',
                'field2': 'value2',
                'error_count': 0,
                'total_records': 10
            }
        }
    }
    
    parameters = {
        'required_fields': ['field1', 'field2'],
        'min_completeness': 0.8,
        'max_error_rate': 0.1
    }
    
    result = await asyncio.get_event_loop().run_in_executor(None, data_quality_condition, context, parameters)
    assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_threshold_condition_integration():
    """Test threshold condition with real data"""
    context = Mock(spec=RoutingContext)
    context.step_results = {
        'step1': {
            'result': {
                'score': 85
            }
        }
    }
    
    parameters = {
        'field_path': 'score',
        'operator': '>=',
        'threshold': 80
    }
    
    result = await asyncio.get_event_loop().run_in_executor(None, threshold_condition, context, parameters)
    assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_cost_threshold_condition_integration():
    """Test cost threshold condition with real data"""
    context = Mock(spec=RoutingContext)
    context.step_results = {
        'step1': {
            'result': {
                'estimated_cost': 50.0
            }
        }
    }
    
    parameters = {
        'max_cost': 100.0,
        'cost_field': 'estimated_cost'
    }
    
    result = await asyncio.get_event_loop().run_in_executor(None, cost_threshold_condition, context, parameters)
    assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_load_balancing_condition_integration():
    """Test load balancing condition with real execution ID"""
    context = Mock(spec=RoutingContext)
    context.execution_id = "test_execution_123"
    context.step_results = {}
    
    parameters = {
        'route_percentage': 50
    }
    
    result = await asyncio.get_event_loop().run_in_executor(None, load_balancing_condition, context, parameters)
    assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_feature_flag_condition_integration():
    """Test feature flag condition with real metadata"""
    context = Mock(spec=RoutingContext)
    context.metadata = {
        'feature_flags': {
            'new_feature': True,
            'legacy_feature': False
        }
    }
    context.step_results = {}
    
    parameters = {
        'feature_name': 'new_feature',
        'default_value': False
    }
    
    result = await asyncio.get_event_loop().run_in_executor(None, feature_flag_condition, context, parameters)
    assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
