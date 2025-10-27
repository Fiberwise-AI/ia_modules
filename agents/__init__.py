"""
Multi-Agent Orchestration and Collaboration System

Provides specialized agents with role-based design, centralized state management,
graph-based orchestration, and advanced collaboration patterns for complex AI workflows.

Core Components:
- BaseAgent & AgentRole: Foundation for agent creation
- StateManager: Centralized state with versioning
- AgentOrchestrator: Graph-based workflow execution
- BaseCollaborativeAgent: Enhanced agent with message passing
- Communication: MessageBus and AgentMessage for inter-agent communication
- Task Decomposition: Strategies for breaking down complex tasks
- Specialist Agents: Ready-to-use agents (Research, Analysis, Synthesis, Critic)
- Collaboration Patterns: Hierarchical, P2P, Debate, Consensus

Example Usage:
    >>> # Basic orchestration
    >>> from ia_modules.agents import AgentOrchestrator, StateManager, AgentRole, BaseAgent
    >>>
    >>> # Collaborative agents
    >>> from ia_modules.agents import (
    ...     BaseCollaborativeAgent,
    ...     MessageBus,
    ...     ResearchAgent,
    ...     AnalysisAgent
    ... )
    >>>
    >>> # Collaboration patterns
    >>> from ia_modules.agents.collaboration_patterns import (
    ...     HierarchicalCollaboration,
    ...     PeerToPeerCollaboration,
    ...     DebateCollaboration,
    ...     ConsensusCollaboration
    ... )
"""

# Core agent infrastructure
from .core import AgentRole, BaseAgent
from .state import StateManager
from .orchestrator import AgentOrchestrator, Edge

# Communication infrastructure
from .communication import (
    MessageBus,
    AgentMessage,
    MessageType
)

# Collaborative agent base
from .base_agent import BaseCollaborativeAgent

# Task decomposition
from .task_decomposition import (
    Task,
    TaskStatus,
    TaskDecomposer,
    DependencyGraph,
    DecompositionStrategy
)

# Specialist agents
from .specialist_agents import (
    ResearchAgent,
    AnalysisAgent,
    SynthesisAgent,
    CriticAgent
)

# Legacy role-based agents (from roles.py)
from .roles import (
    PlannerAgent,
    ResearcherAgent,
    CoderAgent,
    FormatterAgent
)


__all__ = [
    # Core components
    "AgentRole",
    "BaseAgent",
    "StateManager",
    "AgentOrchestrator",
    "Edge",

    # Communication
    "MessageBus",
    "AgentMessage",
    "MessageType",

    # Collaborative agents
    "BaseCollaborativeAgent",

    # Task decomposition
    "Task",
    "TaskStatus",
    "TaskDecomposer",
    "DependencyGraph",
    "DecompositionStrategy",

    # Specialist collaborative agents
    "ResearchAgent",
    "AnalysisAgent",
    "SynthesisAgent",
    "CriticAgent",

    # Legacy role-based agents
    "PlannerAgent",
    "ResearcherAgent",
    "CoderAgent",
    "FormatterAgent",
]
