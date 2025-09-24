"""
Unit tests for condition functions
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


def test_business_hours_condition():
    """Test business hours condition"""
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
    
    result = business_hours_condition(context, parameters)
    # This will depend on actual time, so we just check it returns a boolean
    assert isinstance(result, bool)


def test_data_quality_condition():
    """Test data quality condition"""
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
    
    result = data_quality_condition(context, parameters)
    assert isinstance(result, bool)


def test_threshold_condition():
    """Test threshold condition"""
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
    
    result = threshold_condition(context, parameters)
    assert isinstance(result, bool)


def test_error_rate_condition():
    """Test error rate condition"""
    context = Mock(spec=RoutingContext)
    context.step_results = {
        'step1': {
            'result': {'error': 'some_error'}
        },
        'step2': {
            'result': {}
        }
    }
    
    parameters = {
        'max_error_rate': 0.5,
        'window_minutes': 60
    }
    
    result = error_rate_condition(context, parameters)
    assert isinstance(result, bool)


def test_regex_match_condition():
    """Test regex match condition"""
    context = Mock(spec=RoutingContext)
    context.step_results = {
        'step1': {
            'result': {
                'email': 'test@example.com'
            }
        }
    }
    
    parameters = {
        'field_path': 'email',
        'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    }
    
    result = regex_match_condition(context, parameters)
    assert isinstance(result, bool)


def test_cost_threshold_condition():
    """Test cost threshold condition"""
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
    
    result = cost_threshold_condition(context, parameters)
    assert isinstance(result, bool)


def test_approval_required_condition():
    """Test approval required condition"""
    context = Mock(spec=RoutingContext)
    context.step_results = {
        'step1': {
            'result': {
                'risk_score': 0.8,
                'amount': 1500.0
            }
        }
    }
    
    parameters = {
        'risk_threshold': 0.7,
        'auto_approve_limit': 1000.0
    }
    
    result = approval_required_condition(context, parameters)
    assert isinstance(result, bool)


def test_load_balancing_condition():
    """Test load balancing condition"""
    context = Mock(spec=RoutingContext)
    context.execution_id = "test_execution_123"
    context.step_results = {}
    
    parameters = {
        'route_percentage': 50
    }
    
    result = load_balancing_condition(context, parameters)
    assert isinstance(result, bool)


def test_feature_flag_condition():
    """Test feature flag condition"""
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
    
    result = feature_flag_condition(context, parameters)
    assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
