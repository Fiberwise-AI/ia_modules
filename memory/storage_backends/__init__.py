"""Storage backends for memory persistence."""

from .in_memory_backend import InMemoryBackend

__all__ = ['InMemoryBackend']

# Optional backends
try:
    from .sqlite_backend import SQLiteBackend  # noqa: F401
    __all__.append('SQLiteBackend')
except ImportError:
    pass

try:
    from .vector_backend import VectorBackend  # noqa: F401
    __all__.append('VectorBackend')
except ImportError:
    pass
