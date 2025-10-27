"""
Memory System for IA Modules

Provides:
- Conversation memory: Short-term (thread) and long-term (user) memory
- Advanced memory strategies: Semantic, episodic, and working memory
"""

from .core import ConversationMemory, Message, MessageRole
from .memory_backend import MemoryConversationMemory

# Advanced memory system
from .memory_manager import MemoryManager, MemoryConfig, Memory, MemoryType
from .semantic_memory import SemanticMemory
from .episodic_memory import EpisodicMemory
from .working_memory import WorkingMemory
from .compression import MemoryCompressor, CompressionStrategy

__all__ = [
    'ConversationMemory',
    'Message',
    'MessageRole',
    'MemoryConversationMemory',
    # Advanced memory
    'MemoryManager',
    'MemoryConfig',
    'Memory',
    'MemoryType',
    'SemanticMemory',
    'EpisodicMemory',
    'WorkingMemory',
    'MemoryCompressor',
    'CompressionStrategy',
]

# Optional backends
try:
    from .sql import SQLConversationMemory
    __all__.append('SQLConversationMemory')
except ImportError:
    pass

try:
    from .redis import RedisConversationMemory
    __all__.append('RedisConversationMemory')
except ImportError:
    pass
