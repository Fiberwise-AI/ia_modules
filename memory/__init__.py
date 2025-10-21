"""
Conversation Memory System for IA Modules

Provides short-term (thread) and long-term (user) memory for AI agents.
"""

from .core import ConversationMemory, Message, MessageRole
from .memory_backend import MemoryConversationMemory

__all__ = [
    'ConversationMemory',
    'Message',
    'MessageRole',
    'MemoryConversationMemory',
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
