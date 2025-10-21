"""
RAG (Retrieval-Augmented Generation) system.

Provides vector storage and document retrieval for grounding agent responses.
"""

from .core import Document, VectorStore

__all__ = ["Document", "VectorStore"]
