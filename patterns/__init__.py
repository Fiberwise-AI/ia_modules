"""
Advanced AI agent patterns for IA Modules.

This module provides production-ready implementations of advanced AI patterns:
- Chain-of-Thought (CoT): Explicit step-by-step reasoning
- Self-Consistency: Multiple sampling with consensus
- ReAct: Reasoning and acting in a loop
- Tree of Thoughts: Exploration of reasoning paths
- Constitutional AI: Self-critique and improvement
"""

from .chain_of_thought import ChainOfThoughtStep, CoTConfig
from .self_consistency import SelfConsistencyStep, SelfConsistencyConfig, VotingStrategy
from .react_agent import ReActAgent, ReActConfig, AgentState
from .tree_of_thoughts import TreeOfThoughtsStep, ToTConfig, PruningStrategy, ToTNode
from .constitutional_ai import (
    ConstitutionalAIStep,
    ConstitutionalConfig,
    Principle,
    PrincipleCategory,
    CritiqueResult,
    apply_constitutional_ai
)

__all__ = [
    'ChainOfThoughtStep',
    'CoTConfig',
    'SelfConsistencyStep',
    'SelfConsistencyConfig',
    'VotingStrategy',
    'ReActAgent',
    'ReActConfig',
    'AgentState',
    'TreeOfThoughtsStep',
    'ToTConfig',
    'PruningStrategy',
    'ToTNode',
    'ConstitutionalAIStep',
    'ConstitutionalConfig',
    'Principle',
    'PrincipleCategory',
    'CritiqueResult',
    'apply_constitutional_ai',
]
