"""
Prompt Optimization Module

This module provides a comprehensive prompt optimization system with multiple strategies:
- Genetic algorithms for evolutionary prompt improvement
- Reinforcement learning for adaptive prompt selection
- A/B testing with multi-armed bandit algorithms
- Automatic evaluation metrics
- Template library and composition system
"""

from .optimizer import PromptOptimizer, OptimizationStrategy
from .genetic import GeneticOptimizer, GeneticConfig
from .reinforcement import RLOptimizer, RLConfig
from .ab_testing import ABTester, ABTestConfig
from .evaluators import (
    PromptEvaluator,
    AccuracyEvaluator,
    CoherenceEvaluator,
    RelevanceEvaluator,
    CompositeEvaluator,
)
from .templates import (
    PromptTemplate,
    TemplateLibrary,
    TemplateComposer,
    TemplateVariable,
)

__all__ = [
    # Base classes
    "PromptOptimizer",
    "OptimizationStrategy",
    # Optimizers
    "GeneticOptimizer",
    "GeneticConfig",
    "RLOptimizer",
    "RLConfig",
    "ABTester",
    "ABTestConfig",
    # Evaluators
    "PromptEvaluator",
    "AccuracyEvaluator",
    "CoherenceEvaluator",
    "RelevanceEvaluator",
    "CompositeEvaluator",
    # Templates
    "PromptTemplate",
    "TemplateLibrary",
    "TemplateComposer",
    "TemplateVariable",
]

__version__ = "1.0.0"
