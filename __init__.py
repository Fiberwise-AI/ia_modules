"""
Intelligent Automation Modules

A modular framework for building intelligent automation solutions.
Provides modules for agents, databases, pipelines, and other automation components.
"""

__version__ = "0.2.0"

# Import main modules
from . import database
from . import pipeline 
from . import auth


__all__ = [
    'database',
    'pipeline',
    'auth',
]