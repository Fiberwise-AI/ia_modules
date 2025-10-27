"""
Advanced Memory API endpoints.

This module provides REST API endpoints for demonstrating Advanced Memory features,
including semantic memory, episodic memory, working memory, and memory compression.
"""

from fastapi import APIRouter, HTTPException, Depends, Request, Query
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response Models

class MemoryType(str, Enum):
    """Types of memories."""
    SEMANTIC = "semantic"  # Long-term knowledge
    EPISODIC = "episodic"  # Event sequences
    WORKING = "working"    # Short-term buffer
    COMPRESSED = "compressed"  # Compressed old memories


class MemoryModel(BaseModel):
    """A memory entry."""
    content: str = Field(..., description="Memory content")
    memory_type: MemoryType = Field(..., description="Type of memory")
    importance: float = Field(0.5, ge=0, le=1, description="Importance score (0-1)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "content": "The user prefers Python over JavaScript",
                "memory_type": "semantic",
                "importance": 0.8,
                "metadata": {"topic": "programming_preferences"}
            }
        }


class MemoryResponse(BaseModel):
    """Response with memory details."""
    id: str = Field(..., description="Memory ID")
    content: str = Field(..., description="Memory content")
    memory_type: MemoryType = Field(..., description="Type of memory")
    timestamp: float = Field(..., description="Creation timestamp")
    importance: float = Field(..., description="Importance score (0-1)")
    access_count: int = Field(..., description="Number of times accessed")
    last_accessed: Optional[float] = Field(None, description="Last access timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class AddMemoryRequest(BaseModel):
    """Request to add a memory."""
    memory: MemoryModel = Field(..., description="Memory to add")
    session_id: Optional[str] = Field(None, description="Session ID for grouping memories")

    class Config:
        json_schema_extra = {
            "example": {
                "memory": {
                    "content": "User completed Python tutorial",
                    "memory_type": "episodic",
                    "importance": 0.7,
                    "metadata": {"event": "tutorial_completion"}
                },
                "session_id": "session_123"
            }
        }


class AddMemoryResponse(BaseModel):
    """Response after adding a memory."""
    success: bool = Field(..., description="Whether memory was added successfully")
    memory_id: str = Field(..., description="ID of the added memory")
    memory_type: MemoryType = Field(..., description="Type of memory stored")
    message: str = Field(..., description="Success message")


class RetrieveMemoryRequest(BaseModel):
    """Request to retrieve memories."""
    query: Optional[str] = Field(None, description="Search query for semantic retrieval")
    memory_type: Optional[MemoryType] = Field(None, description="Filter by memory type")
    limit: int = Field(10, ge=1, le=100, description="Maximum memories to return")
    min_importance: float = Field(0.0, ge=0, le=1, description="Minimum importance threshold")
    session_id: Optional[str] = Field(None, description="Filter by session ID")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "programming preferences",
                "memory_type": "semantic",
                "limit": 5,
                "min_importance": 0.5
            }
        }


class RetrieveMemoryResponse(BaseModel):
    """Response with retrieved memories."""
    memories: List[MemoryResponse] = Field(..., description="Retrieved memories")
    total_count: int = Field(..., description="Total number of memories found")
    query: Optional[str] = Field(None, description="Original search query")


class MemoryStatsResponse(BaseModel):
    """Statistics about the memory system."""
    total_memories: int = Field(..., description="Total number of memories")
    by_type: Dict[str, int] = Field(..., description="Count by memory type")
    total_tokens: int = Field(..., description="Approximate total tokens")
    working_memory_size: int = Field(..., description="Current working memory size")
    working_memory_capacity: int = Field(..., description="Working memory capacity")
    compression_threshold: int = Field(..., description="Threshold for compression")
    compression_enabled: bool = Field(..., description="Whether compression is enabled")
    oldest_memory: Optional[float] = Field(None, description="Timestamp of oldest memory")
    newest_memory: Optional[float] = Field(None, description="Timestamp of newest memory")
    avg_importance: float = Field(..., description="Average importance score")
    total_accesses: int = Field(..., description="Total memory access count")


class CompressMemoriesRequest(BaseModel):
    """Request to compress memories."""
    memory_type: Optional[MemoryType] = Field(None, description="Type of memories to compress")
    threshold_age_hours: float = Field(24.0, ge=0, description="Compress memories older than this (hours)")
    max_memories: Optional[int] = Field(None, description="Maximum memories to compress")
    preserve_important: bool = Field(True, description="Preserve high-importance memories")
    importance_threshold: float = Field(0.7, ge=0, le=1, description="Importance threshold for preservation")

    class Config:
        json_schema_extra = {
            "example": {
                "memory_type": "episodic",
                "threshold_age_hours": 48.0,
                "preserve_important": True,
                "importance_threshold": 0.8
            }
        }


class CompressMemoriesResponse(BaseModel):
    """Response after memory compression."""
    success: bool = Field(..., description="Whether compression succeeded")
    compressed_count: int = Field(..., description="Number of memories compressed")
    preserved_count: int = Field(..., description="Number of memories preserved")
    new_memories_created: int = Field(..., description="Number of new compressed memories")
    tokens_saved: int = Field(..., description="Approximate tokens saved")
    message: str = Field(..., description="Success message")


class ClearMemoriesRequest(BaseModel):
    """Request to clear memories."""
    memory_type: Optional[MemoryType] = Field(None, description="Type of memories to clear (all if not specified)")
    session_id: Optional[str] = Field(None, description="Clear only memories from this session")
    confirm: bool = Field(False, description="Must be true to confirm deletion")


class ClearMemoriesResponse(BaseModel):
    """Response after clearing memories."""
    success: bool = Field(..., description="Whether memories were cleared")
    deleted_count: int = Field(..., description="Number of memories deleted")
    message: str = Field(..., description="Success message")


# In-memory storage for demo
# In production, this would use the actual MemoryManager from ia_modules
demo_memories: Dict[str, MemoryResponse] = {}
memory_counter = 0


# Dependency injection
def get_memory_service(request: Request):
    """Get memory service from app state."""
    # For now, return None. In production, return request.app.state.services.memory_service
    return None


# API Endpoints

@router.post("/add", response_model=AddMemoryResponse)
async def add_memory(
    request: AddMemoryRequest,
    service=Depends(get_memory_service)
) -> AddMemoryResponse:
    """
    Add a memory to the memory system.

    Memories can be of different types:
    - SEMANTIC: Long-term knowledge and facts
    - EPISODIC: Event sequences and experiences
    - WORKING: Short-term buffer for immediate context
    - COMPRESSED: Compressed versions of older memories

    Example:
        ```python
        response = await client.post("/api/memory/add", json={
            "memory": {
                "content": "User prefers dark mode",
                "memory_type": "semantic",
                "importance": 0.8
            }
        })
        ```
    """
    global memory_counter

    try:
        import time

        # Create memory ID
        memory_counter += 1
        memory_id = f"mem_{memory_counter}"

        # Create memory response
        memory_response = MemoryResponse(
            id=memory_id,
            content=request.memory.content,
            memory_type=request.memory.memory_type,
            timestamp=time.time(),
            importance=request.memory.importance,
            access_count=0,
            last_accessed=None,
            metadata={
                **request.memory.metadata,
                "session_id": request.session_id
            } if request.session_id else request.memory.metadata
        )

        # Store memory
        demo_memories[memory_id] = memory_response

        logger.info(f"Added memory {memory_id} of type {request.memory.memory_type}")

        return AddMemoryResponse(
            success=True,
            memory_id=memory_id,
            memory_type=request.memory.memory_type,
            message=f"Memory added successfully with ID {memory_id}"
        )

    except Exception as e:
        logger.error(f"Error adding memory: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add memory: {str(e)}"
        )


@router.post("/retrieve", response_model=RetrieveMemoryResponse)
async def retrieve_memories(
    request: RetrieveMemoryRequest,
    service=Depends(get_memory_service)
) -> RetrieveMemoryResponse:
    """
    Retrieve memories based on query and filters.

    Supports:
    - Semantic search (when query is provided)
    - Filtering by memory type
    - Filtering by importance threshold
    - Session-based filtering
    - Limit on results

    Example:
        ```python
        response = await client.post("/api/memory/retrieve", json={
            "query": "user preferences",
            "memory_type": "semantic",
            "limit": 5,
            "min_importance": 0.6
        })
        ```
    """
    try:
        import time

        # Filter memories
        filtered_memories = []

        for memory in demo_memories.values():
            # Apply filters
            if request.memory_type and memory.memory_type != request.memory_type:
                continue

            if memory.importance < request.min_importance:
                continue

            if request.session_id:
                session_id = memory.metadata.get("session_id")
                if session_id != request.session_id:
                    continue

            # Simple text matching for query (in production, use embeddings)
            if request.query:
                if request.query.lower() not in memory.content.lower():
                    continue

            filtered_memories.append(memory)

        # Sort by importance and timestamp
        filtered_memories.sort(
            key=lambda m: (m.importance, m.timestamp),
            reverse=True
        )

        # Apply limit
        limited_memories = filtered_memories[:request.limit]

        # Update access count and last_accessed
        for memory in limited_memories:
            memory.access_count += 1
            memory.last_accessed = time.time()

        logger.info(f"Retrieved {len(limited_memories)} memories (query: {request.query})")

        return RetrieveMemoryResponse(
            memories=limited_memories,
            total_count=len(filtered_memories),
            query=request.query
        )

    except Exception as e:
        logger.error(f"Error retrieving memories: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve memories: {str(e)}"
        )


@router.get("/stats", response_model=MemoryStatsResponse)
async def get_memory_stats(
    service=Depends(get_memory_service)
) -> MemoryStatsResponse:
    """
    Get statistics about the memory system.

    Returns counts, token usage, and other metrics about the memory system.

    Example:
        ```python
        stats = await client.get("/api/memory/stats")
        print(f"Total memories: {stats.total_memories}")
        print(f"Average importance: {stats.avg_importance}")
        ```
    """
    try:
        # Calculate stats
        total_memories = len(demo_memories)

        if total_memories == 0:
            return MemoryStatsResponse(
                total_memories=0,
                by_type={},
                total_tokens=0,
                working_memory_size=0,
                working_memory_capacity=10,
                compression_threshold=50,
                compression_enabled=True,
                oldest_memory=None,
                newest_memory=None,
                avg_importance=0.0,
                total_accesses=0
            )

        # Count by type
        by_type = {}
        for memory in demo_memories.values():
            type_str = memory.memory_type.value
            by_type[type_str] = by_type.get(type_str, 0) + 1

        # Estimate tokens (rough approximation)
        total_tokens = sum(len(m.content.split()) * 1.3 for m in demo_memories.values())

        # Get timestamps
        timestamps = [m.timestamp for m in demo_memories.values()]
        oldest = min(timestamps) if timestamps else None
        newest = max(timestamps) if timestamps else None

        # Average importance
        avg_importance = sum(m.importance for m in demo_memories.values()) / total_memories

        # Total accesses
        total_accesses = sum(m.access_count for m in demo_memories.values())

        # Working memory size (count WORKING type)
        working_memory_size = by_type.get("working", 0)

        return MemoryStatsResponse(
            total_memories=total_memories,
            by_type=by_type,
            total_tokens=int(total_tokens),
            working_memory_size=working_memory_size,
            working_memory_capacity=10,
            compression_threshold=50,
            compression_enabled=True,
            oldest_memory=oldest,
            newest_memory=newest,
            avg_importance=round(avg_importance, 3),
            total_accesses=total_accesses
        )

    except Exception as e:
        logger.error(f"Error getting memory stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get memory stats: {str(e)}"
        )


@router.post("/compress", response_model=CompressMemoriesResponse)
async def compress_memories(
    request: CompressMemoriesRequest,
    service=Depends(get_memory_service)
) -> CompressMemoriesResponse:
    """
    Compress old memories to save space and tokens.

    Memory compression:
    - Combines multiple old memories into summarized versions
    - Preserves important memories based on importance threshold
    - Reduces token usage while maintaining key information
    - Can be filtered by memory type and age

    Example:
        ```python
        response = await client.post("/api/memory/compress", json={
            "memory_type": "episodic",
            "threshold_age_hours": 48,
            "preserve_important": True,
            "importance_threshold": 0.8
        })
        ```
    """
    try:
        import time

        current_time = time.time()
        threshold_seconds = request.threshold_age_hours * 3600

        # Find memories to compress
        to_compress = []
        to_preserve = []

        for memory in demo_memories.values():
            # Skip if wrong type
            if request.memory_type and memory.memory_type != request.memory_type:
                continue

            # Skip if too new
            age = current_time - memory.timestamp
            if age < threshold_seconds:
                continue

            # Check if should be preserved
            if request.preserve_important and memory.importance >= request.importance_threshold:
                to_preserve.append(memory)
            else:
                to_compress.append(memory)

        # Apply max_memories limit
        if request.max_memories and len(to_compress) > request.max_memories:
            # Sort by importance (compress least important first)
            to_compress.sort(key=lambda m: m.importance)
            to_compress = to_compress[:request.max_memories]

        # Estimate tokens before compression
        tokens_before = sum(len(m.content.split()) * 1.3 for m in to_compress)

        # Create compressed memory
        new_memories_created = 0
        if to_compress:
            # Combine contents
            combined_content = " | ".join([m.content for m in to_compress])
            summary = f"[COMPRESSED] Summary of {len(to_compress)} memories: {combined_content[:200]}..."

            # Create compressed memory
            global memory_counter
            memory_counter += 1
            compressed_id = f"mem_{memory_counter}_compressed"

            compressed_memory = MemoryResponse(
                id=compressed_id,
                content=summary,
                memory_type=MemoryType.COMPRESSED,
                timestamp=current_time,
                importance=max(m.importance for m in to_compress),
                access_count=0,
                last_accessed=None,
                metadata={
                    "original_count": len(to_compress),
                    "compressed_at": current_time,
                    "original_ids": [m.id for m in to_compress]
                }
            )

            demo_memories[compressed_id] = compressed_memory
            new_memories_created = 1

            # Remove compressed memories
            for memory in to_compress:
                del demo_memories[memory.id]

        # Estimate tokens after
        tokens_after = len(summary.split()) * 1.3 if to_compress else 0
        tokens_saved = int(tokens_before - tokens_after)

        logger.info(
            f"Compressed {len(to_compress)} memories, "
            f"preserved {len(to_preserve)}, "
            f"saved ~{tokens_saved} tokens"
        )

        return CompressMemoriesResponse(
            success=True,
            compressed_count=len(to_compress),
            preserved_count=len(to_preserve),
            new_memories_created=new_memories_created,
            tokens_saved=tokens_saved,
            message=f"Successfully compressed {len(to_compress)} memories, saved ~{tokens_saved} tokens"
        )

    except Exception as e:
        logger.error(f"Error compressing memories: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compress memories: {str(e)}"
        )


@router.delete("/clear", response_model=ClearMemoriesResponse)
async def clear_memories(
    request: ClearMemoriesRequest,
    service=Depends(get_memory_service)
) -> ClearMemoriesResponse:
    """
    Clear memories from the system.

    Can clear:
    - All memories (if no filters specified)
    - Memories of a specific type
    - Memories from a specific session

    Requires confirmation flag to prevent accidental deletion.

    Example:
        ```python
        response = await client.delete("/api/memory/clear", json={
            "memory_type": "working",
            "confirm": True
        })
        ```
    """
    try:
        if not request.confirm:
            raise HTTPException(
                status_code=400,
                detail="Must set 'confirm: true' to delete memories"
            )

        # Find memories to delete
        to_delete = []

        for memory_id, memory in demo_memories.items():
            # Apply filters
            if request.memory_type and memory.memory_type != request.memory_type:
                continue

            if request.session_id:
                session_id = memory.metadata.get("session_id")
                if session_id != request.session_id:
                    continue

            to_delete.append(memory_id)

        # Delete memories
        for memory_id in to_delete:
            del demo_memories[memory_id]

        logger.info(f"Cleared {len(to_delete)} memories")

        return ClearMemoriesResponse(
            success=True,
            deleted_count=len(to_delete),
            message=f"Successfully deleted {len(to_delete)} memories"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing memories: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear memories: {str(e)}"
        )


@router.get("/retrieve", response_model=RetrieveMemoryResponse)
async def retrieve_memories_get(
    query: Optional[str] = Query(None, description="Search query"),
    memory_type: Optional[MemoryType] = Query(None, description="Filter by type"),
    limit: int = Query(10, ge=1, le=100, description="Max results"),
    min_importance: float = Query(0.0, ge=0, le=1, description="Min importance"),
    session_id: Optional[str] = Query(None, description="Session ID filter"),
    service=Depends(get_memory_service)
) -> RetrieveMemoryResponse:
    """
    Retrieve memories using GET method (convenience endpoint).

    Same as POST /api/memory/retrieve but using query parameters.

    Example:
        ```python
        memories = await client.get(
            "/api/memory/retrieve?query=preferences&memory_type=semantic&limit=5"
        )
        ```
    """
    request_obj = RetrieveMemoryRequest(
        query=query,
        memory_type=memory_type,
        limit=limit,
        min_importance=min_importance,
        session_id=session_id
    )
    return await retrieve_memories(request_obj, service)
