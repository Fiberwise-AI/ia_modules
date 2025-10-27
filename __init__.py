"""
Intelligent Automation Modules

A modular framework for building intelligent automation solutions with advanced AI capabilities.

Core Modules:
- database: Database interfaces and management
- pipeline: Pipeline orchestration and execution
- auth: Authentication and authorization
- utils: Utility functions

Advanced AI Features:
- patterns: AI reasoning patterns (CoT, ReAct, ToT, Constitutional AI)
- memory: Advanced memory strategies (semantic, episodic, working)
- multimodal: Multi-modal AI support (text, image, audio, video)
- agents: Multi-agent collaboration patterns
- prompt_optimization: Automated prompt engineering
- tools: Advanced tool calling with planning and execution
- guardrails: LLM safety and control (input/output rails, jailbreak detection)
- rag: Retrieval-Augmented Generation with RAPTOR and Self-RAG
"""

__version__ = "0.1.0"

# Import core modules
from . import database
from . import pipeline
from . import auth
from . import utils

# Import advanced AI modules
from . import patterns
from . import memory
from . import multimodal
from . import agents
from . import prompt_optimization
from . import tools
from . import guardrails
from . import rag

__all__ = [
    # Core modules
    'database',
    'pipeline',
    'auth',
    'utils',
    # Advanced AI modules
    'patterns',
    'memory',
    'multimodal',
    'agents',
    'prompt_optimization',
    'tools',
    'guardrails',
    'rag',
]