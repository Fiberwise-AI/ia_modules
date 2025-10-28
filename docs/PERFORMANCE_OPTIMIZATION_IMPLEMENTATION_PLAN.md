# Performance & Optimization Implementation Plan

**Date**: 2025-10-25
**Status**: Planning Phase
**Priority**: High - Production Performance

---

## Table of Contents

1. [Intelligent Caching](#1-intelligent-caching)
2. [Batch Processing Optimization](#2-batch-processing-optimization)
3. [Cost Optimization](#3-cost-optimization)
4. [Query Optimization](#4-query-optimization)
5. [Implementation Timeline](#implementation-timeline)
6. [Dependencies & Prerequisites](#dependencies--prerequisites)

---

## 1. Intelligent Caching

### Overview
Multi-level caching system with in-memory, distributed, and database caching layers. Provides intelligent cache warming, invalidation strategies, and cross-worker coherency.

### Requirements

#### 1.1 Cache Manager Core

```python
# ia_modules/cache/manager.py

from typing import Any, Optional, Callable, Union, List, Dict
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import hashlib
import json
import pickle
import asyncio
from enum import Enum
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class CacheBackend(str, Enum):
    """Cache backend types"""
    MEMORY = "memory"
    REDIS = "redis"
    DATABASE = "database"


class InvalidationPolicy(str, Enum):
    """Cache invalidation policies"""
    TTL = "ttl"  # Time to live
    LRU = "lru"  # Least recently used
    LFU = "lfu"  # Least frequently used
    EVENT = "event"  # Event-based invalidation


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    compressed: bool = False
    size_bytes: int = 0

    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl_seconds is None:
            return False
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl_seconds

    def touch(self):
        """Update access metadata"""
        self.accessed_at = datetime.now()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    entry_count: int = 0
    avg_access_time_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class CacheBackendInterface(ABC):
    """Abstract cache backend interface"""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass

    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Set value in cache"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        pass

    @abstractmethod
    async def delete_by_tags(self, tags: List[str]) -> int:
        """Delete all entries with given tags"""
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """Clear entire cache"""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass

    @abstractmethod
    async def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        pass


class MemoryCacheBackend(CacheBackendInterface):
    """In-memory cache backend with LRU/LFU support"""

    def __init__(
        self,
        max_size_mb: int = 512,
        policy: InvalidationPolicy = InvalidationPolicy.LRU
    ):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.policy = policy
        self.entries: Dict[str, CacheEntry] = {}
        self.stats = CacheStats()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        start = datetime.now()

        async with self._lock:
            entry = self.entries.get(key)

            if entry is None:
                self.stats.misses += 1
                return None

            if entry.is_expired():
                await self._delete_entry(key)
                self.stats.misses += 1
                return None

            # Update access metadata
            entry.touch()
            self.stats.hits += 1

            # Update average access time
            duration_ms = (datetime.now() - start).total_seconds() * 1000
            self.stats.avg_access_time_ms = (
                (self.stats.avg_access_time_ms * (self.stats.hits - 1) + duration_ms)
                / self.stats.hits
            )

            return entry.value

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Set value in cache"""
        async with self._lock:
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except Exception:
                size_bytes = 0

            # Check if we need to evict entries
            while (
                self.stats.size_bytes + size_bytes > self.max_size_bytes
                and self.entries
            ):
                await self._evict_entry()

            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                accessed_at=datetime.now(),
                ttl_seconds=ttl_seconds,
                tags=tags or [],
                size_bytes=size_bytes
            )

            # Remove old entry if exists
            if key in self.entries:
                old_size = self.entries[key].size_bytes
                self.stats.size_bytes -= old_size

            # Add new entry
            self.entries[key] = entry
            self.stats.size_bytes += size_bytes
            self.stats.entry_count = len(self.entries)

            return True

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        async with self._lock:
            return await self._delete_entry(key)

    async def _delete_entry(self, key: str) -> bool:
        """Internal delete without lock"""
        if key in self.entries:
            entry = self.entries.pop(key)
            self.stats.size_bytes -= entry.size_bytes
            self.stats.entry_count = len(self.entries)
            return True
        return False

    async def delete_by_tags(self, tags: List[str]) -> int:
        """Delete all entries with given tags"""
        async with self._lock:
            keys_to_delete = [
                key for key, entry in self.entries.items()
                if any(tag in entry.tags for tag in tags)
            ]

            count = 0
            for key in keys_to_delete:
                if await self._delete_entry(key):
                    count += 1

            return count

    async def clear(self) -> bool:
        """Clear entire cache"""
        async with self._lock:
            self.entries.clear()
            self.stats = CacheStats()
            return True

    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        async with self._lock:
            entry = self.entries.get(key)
            if entry and entry.is_expired():
                await self._delete_entry(key)
                return False
            return entry is not None

    async def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        return self.stats

    async def _evict_entry(self):
        """Evict entry based on policy"""
        if not self.entries:
            return

        if self.policy == InvalidationPolicy.LRU:
            # Evict least recently used
            key_to_evict = min(
                self.entries.keys(),
                key=lambda k: self.entries[k].accessed_at
            )
        elif self.policy == InvalidationPolicy.LFU:
            # Evict least frequently used
            key_to_evict = min(
                self.entries.keys(),
                key=lambda k: self.entries[k].access_count
            )
        else:
            # Default to oldest
            key_to_evict = min(
                self.entries.keys(),
                key=lambda k: self.entries[k].created_at
            )

        await self._delete_entry(key_to_evict)
        self.stats.evictions += 1


class RedisCacheBackend(CacheBackendInterface):
    """Redis distributed cache backend"""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self.client = None
        self.stats = CacheStats()
        self._connected = False

    async def _ensure_connected(self):
        """Ensure Redis connection"""
        if not self._connected:
            import redis.asyncio as redis
            self.client = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False
            )
            self._connected = True

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        await self._ensure_connected()
        start = datetime.now()

        try:
            value = await self.client.get(key)
            duration_ms = (datetime.now() - start).total_seconds() * 1000

            if value is None:
                self.stats.misses += 1
                return None

            # Deserialize
            result = pickle.loads(value)
            self.stats.hits += 1
            self.stats.avg_access_time_ms = (
                (self.stats.avg_access_time_ms * (self.stats.hits - 1) + duration_ms)
                / self.stats.hits
            )

            return result
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self.stats.misses += 1
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Set value in cache"""
        await self._ensure_connected()

        try:
            # Serialize
            serialized = pickle.dumps(value)

            # Set value with optional TTL
            if ttl_seconds:
                await self.client.setex(key, ttl_seconds, serialized)
            else:
                await self.client.set(key, serialized)

            # Store tags if provided
            if tags:
                for tag in tags:
                    tag_key = f"tag:{tag}"
                    await self.client.sadd(tag_key, key)

            self.stats.entry_count += 1
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        await self._ensure_connected()

        try:
            result = await self.client.delete(key)
            if result > 0:
                self.stats.entry_count = max(0, self.stats.entry_count - 1)
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False

    async def delete_by_tags(self, tags: List[str]) -> int:
        """Delete all entries with given tags"""
        await self._ensure_connected()

        count = 0
        try:
            for tag in tags:
                tag_key = f"tag:{tag}"
                keys = await self.client.smembers(tag_key)

                if keys:
                    deleted = await self.client.delete(*keys)
                    count += deleted
                    await self.client.delete(tag_key)

            self.stats.entry_count = max(0, self.stats.entry_count - count)
            return count
        except Exception as e:
            logger.error(f"Redis delete by tags error: {e}")
            return count

    async def clear(self) -> bool:
        """Clear entire cache"""
        await self._ensure_connected()

        try:
            await self.client.flushdb()
            self.stats = CacheStats()
            return True
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        await self._ensure_connected()

        try:
            return await self.client.exists(key) > 0
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False

    async def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        await self._ensure_connected()

        try:
            info = await self.client.info("memory")
            self.stats.size_bytes = info.get("used_memory", 0)

            # Get key count
            dbsize = await self.client.dbsize()
            self.stats.entry_count = dbsize

            return self.stats
        except Exception as e:
            logger.error(f"Redis stats error: {e}")
            return self.stats


class CacheManager:
    """Multi-level cache manager"""

    def __init__(
        self,
        l1_enabled: bool = True,
        l1_size_mb: int = 512,
        l2_enabled: bool = False,
        l2_redis_url: str = "redis://localhost:6379/0",
        default_ttl_seconds: int = 3600
    ):
        self.l1_enabled = l1_enabled
        self.l2_enabled = l2_enabled
        self.default_ttl_seconds = default_ttl_seconds

        # Initialize backends
        self.l1: Optional[MemoryCacheBackend] = None
        self.l2: Optional[RedisCacheBackend] = None

        if l1_enabled:
            self.l1 = MemoryCacheBackend(max_size_mb=l1_size_mb)

        if l2_enabled:
            self.l2 = RedisCacheBackend(redis_url=l2_redis_url)

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache (L1 -> L2)"""
        # Try L1 first
        if self.l1:
            value = await self.l1.get(key)
            if value is not None:
                return value

        # Try L2 if L1 miss
        if self.l2:
            value = await self.l2.get(key)
            if value is not None:
                # Populate L1 for future hits
                if self.l1:
                    await self.l1.set(key, value, self.default_ttl_seconds)
                return value

        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Set value in all cache levels"""
        ttl = ttl_seconds or self.default_ttl_seconds
        success = True

        # Set in L1
        if self.l1:
            success = await self.l1.set(key, value, ttl, tags) and success

        # Set in L2
        if self.l2:
            success = await self.l2.set(key, value, ttl, tags) and success

        return success

    async def delete(self, key: str) -> bool:
        """Delete key from all cache levels"""
        success = True

        if self.l1:
            success = await self.l1.delete(key) and success

        if self.l2:
            success = await self.l2.delete(key) and success

        return success

    async def delete_by_tags(self, tags: List[str]) -> int:
        """Delete entries by tags from all levels"""
        count = 0

        if self.l1:
            count += await self.l1.delete_by_tags(tags)

        if self.l2:
            count += await self.l2.delete_by_tags(tags)

        return count

    async def clear(self) -> bool:
        """Clear all cache levels"""
        success = True

        if self.l1:
            success = await self.l1.clear() and success

        if self.l2:
            success = await self.l2.clear() and success

        return success

    async def get_stats(self) -> Dict[str, CacheStats]:
        """Get statistics from all levels"""
        stats = {}

        if self.l1:
            stats["l1"] = await self.l1.get_stats()

        if self.l2:
            stats["l2"] = await self.l2.get_stats()

        return stats

    def generate_key(
        self,
        prefix: str,
        *args,
        **kwargs
    ) -> str:
        """Generate cache key from arguments"""
        # Create deterministic key from args and kwargs
        key_data = {
            "args": args,
            "kwargs": sorted(kwargs.items())
        }
        key_json = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_json.encode()).hexdigest()[:16]

        return f"{prefix}:{key_hash}"


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def configure_cache(
    l1_enabled: bool = True,
    l1_size_mb: int = 512,
    l2_enabled: bool = False,
    l2_redis_url: str = "redis://localhost:6379/0",
    default_ttl_seconds: int = 3600
):
    """Configure global cache manager"""
    global _cache_manager
    _cache_manager = CacheManager(
        l1_enabled=l1_enabled,
        l1_size_mb=l1_size_mb,
        l2_enabled=l2_enabled,
        l2_redis_url=l2_redis_url,
        default_ttl_seconds=default_ttl_seconds
    )
```

#### 1.2 Cache Decorators

```python
# ia_modules/cache/decorators.py

from typing import Callable, Optional, List, Any
from functools import wraps
import asyncio
import inspect
from .manager import get_cache_manager


def cached(
    ttl_seconds: Optional[int] = None,
    key_prefix: Optional[str] = None,
    tags: Optional[List[str]] = None,
    cache_none: bool = False
):
    """
    Decorator to cache function results

    Args:
        ttl_seconds: Time to live in seconds
        key_prefix: Custom key prefix (defaults to function name)
        tags: Tags for cache invalidation
        cache_none: Whether to cache None results
    """
    def decorator(func: Callable) -> Callable:
        prefix = key_prefix or f"{func.__module__}.{func.__name__}"
        is_async = asyncio.iscoroutinefunction(func)

        if is_async:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                cache = get_cache_manager()

                # Generate cache key
                cache_key = cache.generate_key(prefix, *args, **kwargs)

                # Try to get from cache
                cached_value = await cache.get(cache_key)
                if cached_value is not None:
                    return cached_value

                # Execute function
                result = await func(*args, **kwargs)

                # Cache result if not None or cache_none is True
                if result is not None or cache_none:
                    await cache.set(cache_key, result, ttl_seconds, tags)

                return result

            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                cache = get_cache_manager()

                # Generate cache key
                cache_key = cache.generate_key(prefix, *args, **kwargs)

                # Try to get from cache (sync version)
                cached_value = asyncio.run(cache.get(cache_key))
                if cached_value is not None:
                    return cached_value

                # Execute function
                result = func(*args, **kwargs)

                # Cache result
                if result is not None or cache_none:
                    asyncio.run(cache.set(cache_key, result, ttl_seconds, tags))

                return result

            return sync_wrapper

    return decorator


def cache_invalidate(tags: List[str]):
    """
    Decorator to invalidate cache by tags after function execution

    Args:
        tags: Tags to invalidate
    """
    def decorator(func: Callable) -> Callable:
        is_async = asyncio.iscoroutinefunction(func)

        if is_async:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                result = await func(*args, **kwargs)

                # Invalidate cache
                cache = get_cache_manager()
                await cache.delete_by_tags(tags)

                return result

            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                result = func(*args, **kwargs)

                # Invalidate cache
                cache = get_cache_manager()
                asyncio.run(cache.delete_by_tags(tags))

                return result

            return sync_wrapper

    return decorator


class CacheAside:
    """Cache-aside pattern implementation"""

    def __init__(self, cache_manager: Optional[Any] = None):
        self.cache = cache_manager or get_cache_manager()

    async def get_or_compute(
        self,
        key: str,
        compute_func: Callable,
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> Any:
        """
        Get value from cache or compute it

        Args:
            key: Cache key
            compute_func: Function to compute value if cache miss
            ttl_seconds: TTL for cached value
            tags: Tags for invalidation

        Returns:
            Cached or computed value
        """
        # Try cache first
        value = await self.cache.get(key)
        if value is not None:
            return value

        # Compute value
        if asyncio.iscoroutinefunction(compute_func):
            value = await compute_func()
        else:
            value = compute_func()

        # Store in cache
        await self.cache.set(key, value, ttl_seconds, tags)

        return value
```

#### 1.3 Cache Warming & Prefetching

```python
# ia_modules/cache/warming.py

from typing import List, Dict, Any, Callable, Optional
import asyncio
from datetime import datetime
import logging
from dataclasses import dataclass
from .manager import get_cache_manager

logger = logging.getLogger(__name__)


@dataclass
class WarmingTask:
    """Cache warming task definition"""
    key_prefix: str
    compute_func: Callable
    args_list: List[tuple]
    kwargs_list: List[dict]
    ttl_seconds: int
    tags: Optional[List[str]] = None
    priority: int = 0


class CacheWarmer:
    """Cache warming and prefetching manager"""

    def __init__(
        self,
        cache_manager: Optional[Any] = None,
        max_concurrent: int = 10
    ):
        self.cache = cache_manager or get_cache_manager()
        self.max_concurrent = max_concurrent
        self.warming_tasks: List[WarmingTask] = []

    def register_task(
        self,
        key_prefix: str,
        compute_func: Callable,
        args_list: List[tuple],
        kwargs_list: Optional[List[dict]] = None,
        ttl_seconds: int = 3600,
        tags: Optional[List[str]] = None,
        priority: int = 0
    ):
        """Register a cache warming task"""
        task = WarmingTask(
            key_prefix=key_prefix,
            compute_func=compute_func,
            args_list=args_list,
            kwargs_list=kwargs_list or [{}] * len(args_list),
            ttl_seconds=ttl_seconds,
            tags=tags,
            priority=priority
        )
        self.warming_tasks.append(task)

    async def warm_cache(
        self,
        task: WarmingTask,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """Warm cache for a specific task"""
        start_time = datetime.now()
        total = len(task.args_list)
        completed = 0
        errors = 0

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_item(args: tuple, kwargs: dict):
            nonlocal completed, errors

            async with semaphore:
                try:
                    # Generate cache key
                    cache_key = self.cache.generate_key(
                        task.key_prefix,
                        *args,
                        **kwargs
                    )

                    # Check if already cached
                    if await self.cache.l1.exists(cache_key):
                        completed += 1
                        if progress_callback:
                            progress_callback(completed, total)
                        return

                    # Compute value
                    if asyncio.iscoroutinefunction(task.compute_func):
                        value = await task.compute_func(*args, **kwargs)
                    else:
                        value = task.compute_func(*args, **kwargs)

                    # Store in cache
                    await self.cache.set(
                        cache_key,
                        value,
                        task.ttl_seconds,
                        task.tags
                    )

                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total)

                except Exception as e:
                    logger.error(
                        f"Error warming cache for {task.key_prefix}: {e}"
                    )
                    errors += 1
                    completed += 1

        # Process all items concurrently
        tasks = [
            process_item(args, kwargs)
            for args, kwargs in zip(task.args_list, task.kwargs_list)
        ]
        await asyncio.gather(*tasks)

        duration = (datetime.now() - start_time).total_seconds()

        return {
            "task": task.key_prefix,
            "total": total,
            "completed": completed - errors,
            "errors": errors,
            "duration_seconds": duration,
            "items_per_second": total / duration if duration > 0 else 0
        }

    async def warm_all(
        self,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> List[Dict[str, Any]]:
        """Warm cache for all registered tasks"""
        # Sort by priority (higher first)
        sorted_tasks = sorted(
            self.warming_tasks,
            key=lambda t: t.priority,
            reverse=True
        )

        results = []
        for task in sorted_tasks:
            logger.info(f"Warming cache for {task.key_prefix}")

            def task_progress(completed: int, total: int):
                if progress_callback:
                    progress_callback(task.key_prefix, completed, total)

            result = await self.warm_cache(task, task_progress)
            results.append(result)

        return results

    async def prefetch(
        self,
        key_prefix: str,
        compute_func: Callable,
        args_list: List[tuple],
        ttl_seconds: int = 3600
    ) -> int:
        """
        Prefetch data into cache

        Args:
            key_prefix: Cache key prefix
            compute_func: Function to compute values
            args_list: List of arguments to prefetch
            ttl_seconds: TTL for cached values

        Returns:
            Number of items prefetched
        """
        count = 0

        for args in args_list:
            try:
                cache_key = self.cache.generate_key(key_prefix, *args)

                # Skip if already cached
                if await self.cache.l1.exists(cache_key):
                    continue

                # Compute and cache
                if asyncio.iscoroutinefunction(compute_func):
                    value = await compute_func(*args)
                else:
                    value = compute_func(*args)

                await self.cache.set(cache_key, value, ttl_seconds)
                count += 1

            except Exception as e:
                logger.error(f"Error prefetching {cache_key}: {e}")

        return count


# Example usage
async def example_cache_warming():
    """Example of cache warming"""
    from .manager import configure_cache

    # Configure cache
    configure_cache(
        l1_enabled=True,
        l1_size_mb=1024,
        l2_enabled=True,
        l2_redis_url="redis://localhost:6379/0"
    )

    # Create warmer
    warmer = CacheWarmer(max_concurrent=20)

    # Register tasks
    async def fetch_user_data(user_id: int) -> dict:
        # Simulate API call
        await asyncio.sleep(0.1)
        return {"id": user_id, "name": f"User {user_id}"}

    # Warm cache for users 1-1000
    warmer.register_task(
        key_prefix="user_data",
        compute_func=fetch_user_data,
        args_list=[(i,) for i in range(1, 1001)],
        ttl_seconds=3600,
        tags=["users"],
        priority=10
    )

    # Warm cache
    def progress(task: str, completed: int, total: int):
        print(f"{task}: {completed}/{total} ({completed/total*100:.1f}%)")

    results = await warmer.warm_all(progress)

    for result in results:
        print(f"""
Task: {result['task']}
Completed: {result['completed']}/{result['total']}
Errors: {result['errors']}
Duration: {result['duration_seconds']:.2f}s
Rate: {result['items_per_second']:.2f} items/s
        """)
```

#### 1.4 Cache Analytics

```python
# ia_modules/cache/analytics.py

from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict
from .manager import get_cache_manager, CacheStats


@dataclass
class CacheMetrics:
    """Aggregated cache metrics"""
    timestamp: datetime
    hit_rate: float
    miss_rate: float
    total_requests: int
    hits: int
    misses: int
    evictions: int
    size_bytes: int
    entry_count: int
    avg_access_time_ms: float
    hot_keys: List[str]
    cold_keys: List[str]


class CacheAnalytics:
    """Cache analytics and monitoring"""

    def __init__(self, cache_manager: Optional[Any] = None):
        self.cache = cache_manager or get_cache_manager()
        self.metrics_history: List[CacheMetrics] = []
        self.key_access_counts: Dict[str, int] = defaultdict(int)

    async def collect_metrics(self) -> CacheMetrics:
        """Collect current cache metrics"""
        stats_dict = await self.cache.get_stats()

        # Aggregate stats from all levels
        total_hits = 0
        total_misses = 0
        total_evictions = 0
        total_size = 0
        total_entries = 0
        avg_access_times = []

        for level, stats in stats_dict.items():
            total_hits += stats.hits
            total_misses += stats.misses
            total_evictions += stats.evictions
            total_size += stats.size_bytes
            total_entries += stats.entry_count
            if stats.avg_access_time_ms > 0:
                avg_access_times.append(stats.avg_access_time_ms)

        total_requests = total_hits + total_misses
        hit_rate = total_hits / total_requests if total_requests > 0 else 0
        miss_rate = 1 - hit_rate
        avg_access_time = (
            sum(avg_access_times) / len(avg_access_times)
            if avg_access_times else 0
        )

        # Get hot and cold keys
        hot_keys = self._get_hot_keys(top_n=10)
        cold_keys = self._get_cold_keys(top_n=10)

        metrics = CacheMetrics(
            timestamp=datetime.now(),
            hit_rate=hit_rate,
            miss_rate=miss_rate,
            total_requests=total_requests,
            hits=total_hits,
            misses=total_misses,
            evictions=total_evictions,
            size_bytes=total_size,
            entry_count=total_entries,
            avg_access_time_ms=avg_access_time,
            hot_keys=hot_keys,
            cold_keys=cold_keys
        )

        self.metrics_history.append(metrics)

        # Keep only last 1000 metrics
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]

        return metrics

    def _get_hot_keys(self, top_n: int = 10) -> List[str]:
        """Get most frequently accessed keys"""
        sorted_keys = sorted(
            self.key_access_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [key for key, _ in sorted_keys[:top_n]]

    def _get_cold_keys(self, top_n: int = 10) -> List[str]:
        """Get least frequently accessed keys"""
        sorted_keys = sorted(
            self.key_access_counts.items(),
            key=lambda x: x[1]
        )
        return [key for key, _ in sorted_keys[:top_n]]

    async def get_metrics_summary(
        self,
        period_minutes: int = 60
    ) -> Dict[str, Any]:
        """Get metrics summary for a time period"""
        cutoff = datetime.now() - timedelta(minutes=period_minutes)
        recent_metrics = [
            m for m in self.metrics_history
            if m.timestamp >= cutoff
        ]

        if not recent_metrics:
            return {}

        # Calculate averages
        avg_hit_rate = sum(m.hit_rate for m in recent_metrics) / len(recent_metrics)
        avg_access_time = sum(m.avg_access_time_ms for m in recent_metrics) / len(recent_metrics)
        total_evictions = sum(m.evictions for m in recent_metrics)

        # Current values
        current = recent_metrics[-1]

        return {
            "period_minutes": period_minutes,
            "samples": len(recent_metrics),
            "current": {
                "hit_rate": current.hit_rate,
                "size_mb": current.size_bytes / 1024 / 1024,
                "entry_count": current.entry_count,
                "avg_access_time_ms": current.avg_access_time_ms
            },
            "averages": {
                "hit_rate": avg_hit_rate,
                "avg_access_time_ms": avg_access_time
            },
            "totals": {
                "requests": sum(m.total_requests for m in recent_metrics),
                "evictions": total_evictions
            },
            "hot_keys": current.hot_keys,
            "cold_keys": current.cold_keys
        }

    async def start_monitoring(
        self,
        interval_seconds: int = 60,
        callback: Optional[Callable[[CacheMetrics], None]] = None
    ):
        """Start continuous monitoring"""
        while True:
            try:
                metrics = await self.collect_metrics()

                if callback:
                    callback(metrics)

                await asyncio.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"Error collecting cache metrics: {e}")
                await asyncio.sleep(interval_seconds)

    def export_metrics(self) -> List[Dict[str, Any]]:
        """Export metrics history"""
        return [asdict(m) for m in self.metrics_history]
```

---

## 2. Batch Processing Optimization

### Overview
Efficient batch processing with intelligent chunking, parallel execution, memory monitoring, and resilient error handling.

### Requirements

#### 2.1 Batch Processor Core

```python
# ia_modules/batch/processor.py

from typing import (
    List, Any, Callable, Optional, TypeVar, Generic,
    AsyncIterator, Dict, Tuple
)
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import logging
from abc import ABC, abstractmethod
import psutil
import math

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class ChunkingStrategy(str, Enum):
    """Chunking strategy types"""
    FIXED = "fixed"
    DYNAMIC = "dynamic"
    ADAPTIVE = "adaptive"
    MEMORY_BASED = "memory_based"


class BatchStatus(str, Enum):
    """Batch processing status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class BatchResult(Generic[R]):
    """Result of batch processing"""
    status: BatchStatus
    total_items: int
    processed_items: int
    failed_items: int
    results: List[R]
    errors: List[Tuple[int, Exception]]
    duration_seconds: float
    throughput: float  # items per second
    memory_peak_mb: float
    checkpoints: List[int] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_items == 0:
            return 0.0
        return self.processed_items / self.total_items


@dataclass
class BatchConfig:
    """Batch processing configuration"""
    chunk_size: int = 100
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.FIXED
    max_concurrent: int = 10
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    checkpoint_interval: int = 100
    memory_threshold_mb: float = 1000.0
    fail_fast: bool = False
    preserve_order: bool = True


class ChunkerInterface(ABC, Generic[T]):
    """Abstract chunker interface"""

    @abstractmethod
    def chunk(self, items: List[T]) -> List[List[T]]:
        """Chunk items into batches"""
        pass


class FixedChunker(ChunkerInterface[T]):
    """Fixed-size chunking"""

    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size

    def chunk(self, items: List[T]) -> List[List[T]]:
        """Chunk items into fixed-size batches"""
        return [
            items[i:i + self.chunk_size]
            for i in range(0, len(items), self.chunk_size)
        ]


class DynamicChunker(ChunkerInterface[T]):
    """Dynamic chunking based on system load"""

    def __init__(
        self,
        base_chunk_size: int,
        min_chunk_size: int = 10,
        max_chunk_size: int = 1000
    ):
        self.base_chunk_size = base_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def chunk(self, items: List[T]) -> List[List[T]]:
        """Chunk items dynamically based on system resources"""
        # Get current CPU and memory usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent

        # Adjust chunk size based on load
        load_factor = 1.0 - (cpu_percent + memory_percent) / 200.0
        adjusted_size = int(self.base_chunk_size * max(0.1, load_factor))

        # Clamp to min/max
        chunk_size = max(
            self.min_chunk_size,
            min(self.max_chunk_size, adjusted_size)
        )

        logger.debug(
            f"Dynamic chunking: CPU={cpu_percent:.1f}%, "
            f"Memory={memory_percent:.1f}%, "
            f"chunk_size={chunk_size}"
        )

        return [
            items[i:i + chunk_size]
            for i in range(0, len(items), chunk_size)
        ]


class AdaptiveChunker(ChunkerInterface[T]):
    """Adaptive chunking that learns from processing time"""

    def __init__(
        self,
        initial_chunk_size: int = 100,
        min_chunk_size: int = 10,
        max_chunk_size: int = 1000,
        target_chunk_time_seconds: float = 5.0
    ):
        self.chunk_size = initial_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.target_chunk_time = target_chunk_time_seconds
        self.processing_times: List[float] = []

    def chunk(self, items: List[T]) -> List[List[T]]:
        """Chunk items adaptively"""
        # Adjust chunk size based on recent processing times
        if len(self.processing_times) >= 3:
            avg_time = sum(self.processing_times[-3:]) / 3
            items_per_second = self.chunk_size / avg_time if avg_time > 0 else 1

            # Calculate optimal chunk size
            optimal_size = int(items_per_second * self.target_chunk_time)
            self.chunk_size = max(
                self.min_chunk_size,
                min(self.max_chunk_size, optimal_size)
            )

            logger.debug(
                f"Adaptive chunking: avg_time={avg_time:.2f}s, "
                f"chunk_size={self.chunk_size}"
            )

        return [
            items[i:i + self.chunk_size]
            for i in range(0, len(items), self.chunk_size)
        ]

    def record_processing_time(self, duration_seconds: float):
        """Record processing time for adaptation"""
        self.processing_times.append(duration_seconds)
        # Keep only last 10 times
        if len(self.processing_times) > 10:
            self.processing_times = self.processing_times[-10:]


class MemoryBasedChunker(ChunkerInterface[T]):
    """Memory-based chunking"""

    def __init__(
        self,
        max_memory_mb: float = 1000.0,
        estimated_item_size_kb: float = 10.0
    ):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.estimated_item_size = estimated_item_size_kb * 1024

    def chunk(self, items: List[T]) -> List[List[T]]:
        """Chunk items based on memory constraints"""
        # Calculate chunk size based on memory
        chunk_size = max(1, int(self.max_memory_bytes / self.estimated_item_size))

        logger.debug(
            f"Memory-based chunking: "
            f"max_memory={self.max_memory_bytes/1024/1024:.0f}MB, "
            f"chunk_size={chunk_size}"
        )

        return [
            items[i:i + chunk_size]
            for i in range(0, len(items), chunk_size)
        ]


class BatchProcessor(Generic[T, R]):
    """High-performance batch processor"""

    def __init__(self, config: Optional[BatchConfig] = None):
        self.config = config or BatchConfig()
        self.chunker = self._create_chunker()
        self.checkpoint_data: Dict[str, Any] = {}

    def _create_chunker(self) -> ChunkerInterface[T]:
        """Create chunker based on strategy"""
        strategy = self.config.chunking_strategy

        if strategy == ChunkingStrategy.FIXED:
            return FixedChunker(self.config.chunk_size)
        elif strategy == ChunkingStrategy.DYNAMIC:
            return DynamicChunker(self.config.chunk_size)
        elif strategy == ChunkingStrategy.ADAPTIVE:
            return AdaptiveChunker(
                initial_chunk_size=self.config.chunk_size,
                target_chunk_time_seconds=5.0
            )
        elif strategy == ChunkingStrategy.MEMORY_BASED:
            return MemoryBasedChunker(
                max_memory_mb=self.config.memory_threshold_mb
            )
        else:
            return FixedChunker(self.config.chunk_size)

    async def process(
        self,
        items: List[T],
        process_func: Callable[[T], R],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> BatchResult[R]:
        """
        Process items in batches

        Args:
            items: Items to process
            process_func: Function to process each item
            progress_callback: Optional progress callback

        Returns:
            Batch processing result
        """
        start_time = datetime.now()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        peak_memory = start_memory

        total_items = len(items)
        processed_items = 0
        failed_items = 0
        results: List[R] = [None] * total_items if self.config.preserve_order else []
        errors: List[Tuple[int, Exception]] = []
        checkpoints: List[int] = []

        # Chunk items
        chunks = self.chunker.chunk(items)
        logger.info(
            f"Processing {total_items} items in {len(chunks)} chunks "
            f"(strategy: {self.config.chunking_strategy})"
        )

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async def process_chunk(chunk_idx: int, chunk: List[T]):
            nonlocal processed_items, failed_items, peak_memory

            async with semaphore:
                chunk_start = datetime.now()
                chunk_offset = sum(len(chunks[i]) for i in range(chunk_idx))

                for item_idx, item in enumerate(chunk):
                    global_idx = chunk_offset + item_idx
                    retries = 0
                    success = False

                    while retries <= self.config.max_retries and not success:
                        try:
                            # Process item
                            if asyncio.iscoroutinefunction(process_func):
                                result = await process_func(item)
                            else:
                                result = process_func(item)

                            # Store result
                            if self.config.preserve_order:
                                results[global_idx] = result
                            else:
                                results.append(result)

                            processed_items += 1
                            success = True

                            # Update progress
                            if progress_callback:
                                progress_callback(processed_items, total_items)

                            # Checkpoint
                            if processed_items % self.config.checkpoint_interval == 0:
                                checkpoints.append(processed_items)
                                await self._save_checkpoint(
                                    processed_items,
                                    results,
                                    errors
                                )

                        except Exception as e:
                            retries += 1
                            if retries <= self.config.max_retries:
                                logger.warning(
                                    f"Error processing item {global_idx}, "
                                    f"retry {retries}/{self.config.max_retries}: {e}"
                                )
                                await asyncio.sleep(
                                    self.config.retry_delay_seconds * retries
                                )
                            else:
                                logger.error(
                                    f"Failed to process item {global_idx} "
                                    f"after {self.config.max_retries} retries: {e}"
                                )
                                errors.append((global_idx, e))
                                failed_items += 1

                                if self.config.fail_fast:
                                    raise

                    # Monitor memory
                    current_memory = (
                        psutil.Process().memory_info().rss / 1024 / 1024
                    )
                    peak_memory = max(peak_memory, current_memory)

                    # Check memory threshold
                    if current_memory > self.config.memory_threshold_mb:
                        logger.warning(
                            f"Memory usage ({current_memory:.0f}MB) exceeds "
                            f"threshold ({self.config.memory_threshold_mb:.0f}MB)"
                        )

                # Record chunk processing time for adaptive chunking
                chunk_duration = (
                    datetime.now() - chunk_start
                ).total_seconds()
                if isinstance(self.chunker, AdaptiveChunker):
                    self.chunker.record_processing_time(chunk_duration)

        # Process all chunks
        try:
            chunk_tasks = [
                process_chunk(idx, chunk)
                for idx, chunk in enumerate(chunks)
            ]
            await asyncio.gather(*chunk_tasks)

            status = BatchStatus.COMPLETED
            if failed_items > 0:
                status = BatchStatus.PARTIAL

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            status = BatchStatus.FAILED

        # Calculate metrics
        duration = (datetime.now() - start_time).total_seconds()
        throughput = processed_items / duration if duration > 0 else 0

        return BatchResult(
            status=status,
            total_items=total_items,
            processed_items=processed_items,
            failed_items=failed_items,
            results=results,
            errors=errors,
            duration_seconds=duration,
            throughput=throughput,
            memory_peak_mb=peak_memory,
            checkpoints=checkpoints
        )

    async def _save_checkpoint(
        self,
        processed_count: int,
        results: List[R],
        errors: List[Tuple[int, Exception]]
    ):
        """Save checkpoint data"""
        self.checkpoint_data = {
            "processed_count": processed_count,
            "results": results[:processed_count],
            "errors": errors,
            "timestamp": datetime.now().isoformat()
        }
        logger.info(f"Checkpoint saved: {processed_count} items processed")

    async def resume_from_checkpoint(
        self,
        items: List[T],
        process_func: Callable[[T], R],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> BatchResult[R]:
        """Resume processing from last checkpoint"""
        if not self.checkpoint_data:
            logger.info("No checkpoint found, starting from beginning")
            return await self.process(items, process_func, progress_callback)

        processed_count = self.checkpoint_data["processed_count"]
        logger.info(f"Resuming from checkpoint: {processed_count} items")

        # Process remaining items
        remaining_items = items[processed_count:]
        partial_result = await self.process(
            remaining_items,
            process_func,
            progress_callback
        )

        # Merge results
        checkpoint_results = self.checkpoint_data["results"]
        checkpoint_errors = self.checkpoint_data["errors"]

        merged_results = checkpoint_results + partial_result.results
        merged_errors = checkpoint_errors + [
            (idx + processed_count, err)
            for idx, err in partial_result.errors
        ]

        return BatchResult(
            status=partial_result.status,
            total_items=len(items),
            processed_items=processed_count + partial_result.processed_items,
            failed_items=len(merged_errors),
            results=merged_results,
            errors=merged_errors,
            duration_seconds=partial_result.duration_seconds,
            throughput=partial_result.throughput,
            memory_peak_mb=partial_result.memory_peak_mb,
            checkpoints=partial_result.checkpoints
        )


async def process_in_batches(
    items: List[T],
    process_func: Callable[[T], R],
    chunk_size: int = 100,
    max_concurrent: int = 10,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> BatchResult[R]:
    """
    Convenience function for batch processing

    Args:
        items: Items to process
        process_func: Function to process each item
        chunk_size: Size of each chunk
        max_concurrent: Maximum concurrent chunks
        progress_callback: Optional progress callback

    Returns:
        Batch processing result
    """
    config = BatchConfig(
        chunk_size=chunk_size,
        max_concurrent=max_concurrent,
        chunking_strategy=ChunkingStrategy.FIXED
    )

    processor = BatchProcessor[T, R](config)
    return await processor.process(items, process_func, progress_callback)
```

#### 2.2 Streaming Batch Processor

```python
# ia_modules/batch/streaming.py

from typing import AsyncIterator, Callable, Optional, TypeVar, List
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class StreamConfig:
    """Streaming configuration"""
    buffer_size: int = 1000
    max_concurrent: int = 10
    flush_interval_seconds: float = 5.0
    max_memory_mb: float = 500.0


class StreamingBatchProcessor:
    """Memory-efficient streaming batch processor"""

    def __init__(self, config: Optional[StreamConfig] = None):
        self.config = config or StreamConfig()
        self.buffer: List = []
        self.processed_count = 0
        self.last_flush = datetime.now()

    async def process_stream(
        self,
        item_stream: AsyncIterator[T],
        process_func: Callable[[List[T]], List[R]],
        output_callback: Optional[Callable[[List[R]], None]] = None
    ) -> int:
        """
        Process items from an async stream

        Args:
            item_stream: Async iterator of items
            process_func: Function to process batches
            output_callback: Callback for processed results

        Returns:
            Total items processed
        """
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async def process_batch(batch: List[T]):
            async with semaphore:
                try:
                    # Process batch
                    if asyncio.iscoroutinefunction(process_func):
                        results = await process_func(batch)
                    else:
                        results = process_func(batch)

                    # Send results to callback
                    if output_callback:
                        output_callback(results)

                    return len(results)

                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    return 0

        pending_tasks = []

        # Stream items
        async for item in item_stream:
            self.buffer.append(item)

            # Flush buffer if full or time elapsed
            should_flush = (
                len(self.buffer) >= self.config.buffer_size or
                (datetime.now() - self.last_flush).total_seconds() >=
                self.config.flush_interval_seconds
            )

            if should_flush and self.buffer:
                # Create batch task
                batch = self.buffer.copy()
                self.buffer.clear()
                self.last_flush = datetime.now()

                task = asyncio.create_task(process_batch(batch))
                pending_tasks.append(task)

                # Clean up completed tasks
                pending_tasks = [t for t in pending_tasks if not t.done()]

        # Flush remaining buffer
        if self.buffer:
            task = asyncio.create_task(process_batch(self.buffer))
            pending_tasks.append(task)

        # Wait for all tasks to complete
        if pending_tasks:
            results = await asyncio.gather(*pending_tasks)
            self.processed_count += sum(results)

        return self.processed_count


# Example usage
async def example_streaming():
    """Example of streaming batch processing"""

    # Create async stream
    async def item_generator():
        for i in range(10000):
            await asyncio.sleep(0.001)  # Simulate streaming
            yield {"id": i, "value": i * 2}

    # Process function
    async def process_batch(items: List[dict]) -> List[dict]:
        # Simulate processing
        await asyncio.sleep(0.1)
        return [{"id": item["id"], "processed": True} for item in items]

    # Output callback
    def handle_results(results: List[dict]):
        print(f"Processed {len(results)} items")

    # Process stream
    processor = StreamingBatchProcessor(
        config=StreamConfig(buffer_size=100, max_concurrent=5)
    )

    total = await processor.process_stream(
        item_generator(),
        process_batch,
        handle_results
    )

    print(f"Total processed: {total}")
```

---

## 3. Cost Optimization

### Overview
Comprehensive LLM and resource cost tracking with token usage monitoring, cost estimation, budget enforcement, and optimization recommendations.

### Requirements

#### 3.1 Cost Tracker Core

```python
# ia_modules/cost/tracker.py

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class ModelProvider(str, Enum):
    """LLM provider types"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    CUSTOM = "custom"


@dataclass
class ModelPricing:
    """Model pricing information"""
    provider: ModelProvider
    model_name: str
    input_cost_per_1k: float  # Cost per 1000 input tokens
    output_cost_per_1k: float  # Cost per 1000 output tokens
    context_window: int
    supports_streaming: bool = True


# Default pricing (as of 2025-10-25)
DEFAULT_PRICING = {
    "gpt-4": ModelPricing(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4",
        input_cost_per_1k=0.03,
        output_cost_per_1k=0.06,
        context_window=8192
    ),
    "gpt-4-32k": ModelPricing(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4-32k",
        input_cost_per_1k=0.06,
        output_cost_per_1k=0.12,
        context_window=32768
    ),
    "gpt-3.5-turbo": ModelPricing(
        provider=ModelProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        input_cost_per_1k=0.0015,
        output_cost_per_1k=0.002,
        context_window=4096
    ),
    "claude-3-opus": ModelPricing(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-opus-20240229",
        input_cost_per_1k=0.015,
        output_cost_per_1k=0.075,
        context_window=200000
    ),
    "claude-3-sonnet": ModelPricing(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-sonnet-20240229",
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
        context_window=200000
    ),
    "claude-3-haiku": ModelPricing(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-haiku-20240307",
        input_cost_per_1k=0.00025,
        output_cost_per_1k=0.00125,
        context_window=200000
    ),
    "gemini-pro": ModelPricing(
        provider=ModelProvider.GOOGLE,
        model_name="gemini-pro",
        input_cost_per_1k=0.00025,
        output_cost_per_1k=0.0005,
        context_window=30720
    ),
}


@dataclass
class TokenUsage:
    """Token usage information"""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: 'TokenUsage') -> 'TokenUsage':
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens
        )


@dataclass
class CostEstimate:
    """Cost estimate for LLM usage"""
    model_name: str
    token_usage: TokenUsage
    input_cost: float
    output_cost: float
    total_cost: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "input_tokens": self.token_usage.input_tokens,
            "output_tokens": self.token_usage.output_tokens,
            "total_tokens": self.token_usage.total_tokens,
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "total_cost": self.total_cost,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class BudgetAlert:
    """Budget alert"""
    threshold_type: str  # "daily", "weekly", "monthly", "total"
    threshold_amount: float
    current_amount: float
    percentage_used: float
    triggered_at: datetime


class CostTracker:
    """LLM cost tracking and budget management"""

    def __init__(
        self,
        custom_pricing: Optional[Dict[str, ModelPricing]] = None
    ):
        self.pricing = {**DEFAULT_PRICING, **(custom_pricing or {})}
        self.usage_history: List[CostEstimate] = []
        self.budgets: Dict[str, float] = {}
        self.alerts: List[BudgetAlert] = []

        # Usage aggregations
        self.usage_by_model: Dict[str, TokenUsage] = defaultdict(TokenUsage)
        self.usage_by_user: Dict[str, TokenUsage] = defaultdict(TokenUsage)
        self.usage_by_pipeline: Dict[str, TokenUsage] = defaultdict(TokenUsage)

    def estimate_cost(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int
    ) -> CostEstimate:
        """
        Estimate cost for token usage

        Args:
            model_name: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost estimate
        """
        pricing = self.pricing.get(model_name)
        if not pricing:
            logger.warning(f"No pricing found for model {model_name}")
            pricing = ModelPricing(
                provider=ModelProvider.CUSTOM,
                model_name=model_name,
                input_cost_per_1k=0.001,
                output_cost_per_1k=0.002,
                context_window=4096
            )

        input_cost = (input_tokens / 1000) * pricing.input_cost_per_1k
        output_cost = (output_tokens / 1000) * pricing.output_cost_per_1k
        total_cost = input_cost + output_cost

        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens
        )

        return CostEstimate(
            model_name=model_name,
            token_usage=usage,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost
        )

    def track_usage(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        user_id: Optional[str] = None,
        pipeline_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CostEstimate:
        """
        Track LLM usage

        Args:
            model_name: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            user_id: Optional user ID
            pipeline_id: Optional pipeline ID
            metadata: Optional metadata

        Returns:
            Cost estimate
        """
        estimate = self.estimate_cost(model_name, input_tokens, output_tokens)

        # Store in history
        self.usage_history.append(estimate)

        # Update aggregations
        usage = estimate.token_usage
        self.usage_by_model[model_name] += usage

        if user_id:
            self.usage_by_user[user_id] += usage

        if pipeline_id:
            self.usage_by_pipeline[pipeline_id] += usage

        # Check budgets
        self._check_budgets()

        return estimate

    def set_budget(
        self,
        period: str,  # "daily", "weekly", "monthly", "total"
        amount: float
    ):
        """Set budget for a period"""
        self.budgets[period] = amount
        logger.info(f"Budget set: {period} = ${amount:.2f}")

    def _check_budgets(self):
        """Check if any budgets are exceeded"""
        now = datetime.now()

        for period, budget in self.budgets.items():
            current_cost = self._get_period_cost(period, now)
            percentage = (current_cost / budget * 100) if budget > 0 else 0

            # Alert thresholds: 80%, 90%, 100%
            for threshold in [80, 90, 100]:
                if percentage >= threshold:
                    alert = BudgetAlert(
                        threshold_type=period,
                        threshold_amount=budget,
                        current_amount=current_cost,
                        percentage_used=percentage,
                        triggered_at=now
                    )
                    self.alerts.append(alert)
                    logger.warning(
                        f"Budget alert: {period} budget at {percentage:.1f}% "
                        f"(${current_cost:.2f} / ${budget:.2f})"
                    )
                    break

    def _get_period_cost(self, period: str, reference_time: datetime) -> float:
        """Get cost for a specific period"""
        if period == "total":
            return sum(e.total_cost for e in self.usage_history)

        # Calculate period start
        if period == "daily":
            start = reference_time.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "weekly":
            start = reference_time - timedelta(days=reference_time.weekday())
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "monthly":
            start = reference_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            return 0.0

        # Sum costs in period
        return sum(
            e.total_cost for e in self.usage_history
            if e.timestamp >= start
        )

    def get_usage_report(
        self,
        period: Optional[str] = None,
        group_by: str = "model"  # "model", "user", "pipeline", "day"
    ) -> Dict[str, Any]:
        """
        Get usage report

        Args:
            period: Optional period filter
            group_by: Grouping dimension

        Returns:
            Usage report
        """
        # Filter by period
        if period:
            now = datetime.now()
            history = [
                e for e in self.usage_history
                if self._is_in_period(e.timestamp, period, now)
            ]
        else:
            history = self.usage_history

        # Group by dimension
        if group_by == "model":
            grouped = defaultdict(lambda: {"cost": 0.0, "tokens": 0})
            for estimate in history:
                grouped[estimate.model_name]["cost"] += estimate.total_cost
                grouped[estimate.model_name]["tokens"] += estimate.token_usage.total_tokens
        else:
            # Implement other groupings as needed
            grouped = {}

        total_cost = sum(e.total_cost for e in history)
        total_tokens = sum(e.token_usage.total_tokens for e in history)

        return {
            "period": period or "all",
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "request_count": len(history),
            "grouped": dict(grouped),
            "generated_at": datetime.now().isoformat()
        }

    def _is_in_period(
        self,
        timestamp: datetime,
        period: str,
        reference_time: datetime
    ) -> bool:
        """Check if timestamp is in period"""
        if period == "daily":
            start = reference_time.replace(hour=0, minute=0, second=0, microsecond=0)
            return timestamp >= start
        elif period == "weekly":
            start = reference_time - timedelta(days=reference_time.weekday())
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            return timestamp >= start
        elif period == "monthly":
            start = reference_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            return timestamp >= start
        return True

    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get cost optimization recommendations"""
        recommendations = []

        # Analyze model usage
        total_cost = sum(e.total_cost for e in self.usage_history)
        model_costs = defaultdict(float)

        for estimate in self.usage_history:
            model_costs[estimate.model_name] += estimate.total_cost

        # Recommend cheaper models
        for model, cost in model_costs.items():
            if cost / total_cost > 0.3:  # Model accounts for >30% of costs
                if "gpt-4" in model:
                    recommendations.append({
                        "type": "model_substitution",
                        "current_model": model,
                        "suggested_model": "gpt-3.5-turbo",
                        "potential_savings": cost * 0.95,  # ~95% savings
                        "reason": "GPT-3.5-turbo is 95% cheaper and suitable for many tasks"
                    })
                elif "claude-3-opus" in model:
                    recommendations.append({
                        "type": "model_substitution",
                        "current_model": model,
                        "suggested_model": "claude-3-sonnet",
                        "potential_savings": cost * 0.80,  # ~80% savings
                        "reason": "Claude 3 Sonnet is 80% cheaper with similar capabilities"
                    })

        # Recommend caching
        duplicate_inputs = self._find_duplicate_inputs()
        if duplicate_inputs > 10:
            cache_savings = sum(
                e.total_cost for e in self.usage_history[:duplicate_inputs]
            )
            recommendations.append({
                "type": "enable_caching",
                "duplicate_requests": duplicate_inputs,
                "potential_savings": cache_savings,
                "reason": f"Found {duplicate_inputs} duplicate requests that could be cached"
            })

        # Recommend prompt optimization
        avg_input_tokens = sum(
            e.token_usage.input_tokens for e in self.usage_history
        ) / len(self.usage_history) if self.usage_history else 0

        if avg_input_tokens > 2000:
            recommendations.append({
                "type": "prompt_optimization",
                "current_avg_tokens": avg_input_tokens,
                "target_avg_tokens": avg_input_tokens * 0.7,
                "potential_savings": total_cost * 0.3,
                "reason": "Large prompts detected. Optimization could reduce costs by 30%"
            })

        return recommendations

    def _find_duplicate_inputs(self) -> int:
        """Find number of duplicate input requests"""
        # Simplified implementation
        # In production, would hash inputs and count duplicates
        return max(0, len(self.usage_history) // 10)

    def export_usage(self, filepath: str):
        """Export usage history to JSON"""
        data = {
            "exported_at": datetime.now().isoformat(),
            "total_cost": sum(e.total_cost for e in self.usage_history),
            "total_requests": len(self.usage_history),
            "usage": [e.to_dict() for e in self.usage_history]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Usage exported to {filepath}")


# Global cost tracker instance
_cost_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """Get global cost tracker instance"""
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker
```

#### 3.2 Token Counter

```python
# ia_modules/cost/tokens.py

from typing import Optional
import tiktoken
import logging

logger = logging.getLogger(__name__)


class TokenCounter:
    """Token counting utilities"""

    def __init__(self):
        self.encoders = {}

    def get_encoder(self, model: str):
        """Get tokenizer encoder for model"""
        if model not in self.encoders:
            try:
                self.encoders[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                # Default to cl100k_base for unknown models
                logger.warning(
                    f"Unknown model {model}, using cl100k_base encoding"
                )
                self.encoders[model] = tiktoken.get_encoding("cl100k_base")

        return self.encoders[model]

    def count_tokens(
        self,
        text: str,
        model: str = "gpt-3.5-turbo"
    ) -> int:
        """Count tokens in text"""
        encoder = self.get_encoder(model)
        return len(encoder.encode(text))

    def count_messages_tokens(
        self,
        messages: list,
        model: str = "gpt-3.5-turbo"
    ) -> int:
        """
        Count tokens in chat messages

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name

        Returns:
            Total token count
        """
        encoder = self.get_encoder(model)

        # Tokens per message overhead
        if model.startswith("gpt-4"):
            tokens_per_message = 3
            tokens_per_name = 1
        elif model.startswith("gpt-3.5"):
            tokens_per_message = 4
            tokens_per_name = -1
        else:
            tokens_per_message = 3
            tokens_per_name = 1

        num_tokens = 0

        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoder.encode(str(value)))
                if key == "name":
                    num_tokens += tokens_per_name

        num_tokens += 3  # Every reply is primed with assistant

        return num_tokens

    def estimate_max_output_tokens(
        self,
        input_text: str,
        model: str = "gpt-3.5-turbo",
        max_context: Optional[int] = None
    ) -> int:
        """
        Estimate maximum output tokens available

        Args:
            input_text: Input text
            model: Model name
            max_context: Maximum context window (auto-detected if None)

        Returns:
            Maximum output tokens
        """
        from .tracker import DEFAULT_PRICING

        input_tokens = self.count_tokens(input_text, model)

        # Get context window
        if max_context is None:
            pricing = DEFAULT_PRICING.get(model)
            if pricing:
                max_context = pricing.context_window
            else:
                max_context = 4096  # Default

        return max(0, max_context - input_tokens)


# Global token counter instance
_token_counter: Optional[TokenCounter] = None


def get_token_counter() -> TokenCounter:
    """Get global token counter instance"""
    global _token_counter
    if _token_counter is None:
        _token_counter = TokenCounter()
    return _token_counter


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Convenience function to count tokens"""
    counter = get_token_counter()
    return counter.count_tokens(text, model)
```

---

## 4. Query Optimization

### Overview
Database query performance optimization with query plan analysis, index recommendations, N+1 detection, and automatic caching.

### Requirements

#### 4.1 Query Optimizer

```python
# ia_modules/database/query_optimizer.py

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import logging
import re
import hashlib
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class QueryPlan:
    """Database query execution plan"""
    query_hash: str
    sql: str
    plan: List[Dict[str, Any]]
    estimated_cost: float
    estimated_rows: int
    execution_time_ms: float
    timestamp: datetime


@dataclass
class IndexRecommendation:
    """Index recommendation"""
    table_name: str
    column_names: List[str]
    index_type: str  # "btree", "hash", "gin", etc.
    estimated_benefit: float
    reason: str
    create_sql: str


@dataclass
class QueryIssue:
    """Query performance issue"""
    query_hash: str
    issue_type: str  # "slow", "n+1", "missing_index", "full_scan"
    severity: str  # "low", "medium", "high", "critical"
    description: str
    recommendation: str
    detected_at: datetime


class QueryOptimizer:
    """Database query optimization analyzer"""

    def __init__(self, db_manager):
        self.db = db_manager
        self.query_plans: Dict[str, QueryPlan] = {}
        self.slow_queries: List[QueryPlan] = []
        self.index_recommendations: List[IndexRecommendation] = []
        self.query_issues: List[QueryIssue] = []

        # N+1 detection
        self.query_execution_log: List[tuple] = []
        self.n_plus_one_threshold = 10

        # Slow query threshold (ms)
        self.slow_query_threshold_ms = 1000

    def analyze_query(
        self,
        sql: str,
        params: Optional[tuple] = None
    ) -> QueryPlan:
        """
        Analyze query execution plan

        Args:
            sql: SQL query
            params: Query parameters

        Returns:
            Query execution plan
        """
        query_hash = self._hash_query(sql)
        start_time = datetime.now()

        try:
            # Get execution plan
            explain_sql = f"EXPLAIN (FORMAT JSON, ANALYZE) {sql}"

            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                if params:
                    cursor.execute(explain_sql, params)
                else:
                    cursor.execute(explain_sql)

                plan_data = cursor.fetchone()[0]

            execution_time = (
                datetime.now() - start_time
            ).total_seconds() * 1000

            # Parse plan
            plan = plan_data[0]["Plan"] if isinstance(plan_data, list) else plan_data["Plan"]

            query_plan = QueryPlan(
                query_hash=query_hash,
                sql=sql,
                plan=[plan],
                estimated_cost=plan.get("Total Cost", 0),
                estimated_rows=plan.get("Plan Rows", 0),
                execution_time_ms=execution_time,
                timestamp=datetime.now()
            )

            # Store plan
            self.query_plans[query_hash] = query_plan

            # Check if slow
            if execution_time > self.slow_query_threshold_ms:
                self.slow_queries.append(query_plan)
                self._record_issue(
                    query_hash,
                    "slow",
                    "high",
                    f"Query took {execution_time:.0f}ms (threshold: {self.slow_query_threshold_ms}ms)",
                    "Consider adding indexes or optimizing the query"
                )

            # Analyze plan for issues
            self._analyze_plan_issues(query_plan)

            return query_plan

        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            raise

    def _hash_query(self, sql: str) -> str:
        """Generate hash for query"""
        # Normalize query (remove whitespace, lowercase)
        normalized = re.sub(r'\s+', ' ', sql.strip().lower())
        return hashlib.md5(normalized.encode()).hexdigest()

    def _analyze_plan_issues(self, query_plan: QueryPlan):
        """Analyze query plan for issues"""
        plan = query_plan.plan[0]

        # Check for sequential scans
        if self._has_sequential_scan(plan):
            self._record_issue(
                query_plan.query_hash,
                "full_scan",
                "medium",
                "Query performs sequential scan on table",
                "Consider adding an index on the filtered columns"
            )

            # Generate index recommendation
            table_name = self._extract_table_from_seq_scan(plan)
            filter_columns = self._extract_filter_columns(query_plan.sql)

            if table_name and filter_columns:
                self._recommend_index(table_name, filter_columns)

    def _has_sequential_scan(self, plan: Dict[str, Any]) -> bool:
        """Check if plan contains sequential scan"""
        if plan.get("Node Type") == "Seq Scan":
            return True

        # Check children
        for child in plan.get("Plans", []):
            if self._has_sequential_scan(child):
                return True

        return False

    def _extract_table_from_seq_scan(self, plan: Dict[str, Any]) -> Optional[str]:
        """Extract table name from sequential scan"""
        if plan.get("Node Type") == "Seq Scan":
            return plan.get("Relation Name")

        for child in plan.get("Plans", []):
            table = self._extract_table_from_seq_scan(child)
            if table:
                return table

        return None

    def _extract_filter_columns(self, sql: str) -> List[str]:
        """Extract filter columns from WHERE clause"""
        # Simple regex-based extraction
        # In production, use a proper SQL parser
        where_match = re.search(r'WHERE\s+(.+?)(?:GROUP|ORDER|LIMIT|$)', sql, re.IGNORECASE)

        if not where_match:
            return []

        where_clause = where_match.group(1)

        # Extract column names (simplified)
        columns = re.findall(r'(\w+)\s*[=<>]', where_clause)

        return list(set(columns))

    def _recommend_index(
        self,
        table_name: str,
        column_names: List[str],
        index_type: str = "btree"
    ):
        """Recommend index creation"""
        # Check if recommendation already exists
        for rec in self.index_recommendations:
            if (rec.table_name == table_name and
                set(rec.column_names) == set(column_names)):
                return

        index_name = f"idx_{table_name}_{'_'.join(column_names)}"
        columns_sql = ", ".join(column_names)

        recommendation = IndexRecommendation(
            table_name=table_name,
            column_names=column_names,
            index_type=index_type,
            estimated_benefit=0.5,  # Simplified
            reason=f"Sequential scan detected on {table_name}",
            create_sql=f"CREATE INDEX {index_name} ON {table_name} ({columns_sql});"
        )

        self.index_recommendations.append(recommendation)
        logger.info(f"Index recommended: {recommendation.create_sql}")

    def _record_issue(
        self,
        query_hash: str,
        issue_type: str,
        severity: str,
        description: str,
        recommendation: str
    ):
        """Record query issue"""
        issue = QueryIssue(
            query_hash=query_hash,
            issue_type=issue_type,
            severity=severity,
            description=description,
            recommendation=recommendation,
            detected_at=datetime.now()
        )

        self.query_issues.append(issue)

    def detect_n_plus_one(
        self,
        query: str,
        execution_context: Optional[str] = None
    ):
        """
        Detect N+1 query pattern

        Args:
            query: SQL query
            execution_context: Optional execution context (e.g., endpoint)
        """
        timestamp = datetime.now()
        self.query_execution_log.append((query, execution_context, timestamp))

        # Keep only recent queries (last 10 seconds)
        cutoff = timestamp - timedelta(seconds=10)
        self.query_execution_log = [
            (q, ctx, ts) for q, ctx, ts in self.query_execution_log
            if ts >= cutoff
        ]

        # Group by context and query pattern
        if execution_context:
            context_queries = [
                q for q, ctx, _ in self.query_execution_log
                if ctx == execution_context
            ]

            # Check for repeated similar queries
            query_counts = defaultdict(int)
            for q in context_queries:
                # Normalize query (remove literals)
                normalized = re.sub(r'\b\d+\b', '?', q)
                query_counts[normalized] += 1

            # Detect N+1
            for normalized_query, count in query_counts.items():
                if count >= self.n_plus_one_threshold:
                    query_hash = self._hash_query(normalized_query)
                    self._record_issue(
                        query_hash,
                        "n+1",
                        "critical",
                        f"N+1 query detected: {count} similar queries in {execution_context}",
                        "Use eager loading or batch queries to reduce database roundtrips"
                    )

    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report"""
        return {
            "summary": {
                "total_queries_analyzed": len(self.query_plans),
                "slow_queries": len(self.slow_queries),
                "index_recommendations": len(self.index_recommendations),
                "total_issues": len(self.query_issues)
            },
            "slow_queries": [
                {
                    "sql": qp.sql,
                    "execution_time_ms": qp.execution_time_ms,
                    "estimated_cost": qp.estimated_cost,
                    "timestamp": qp.timestamp.isoformat()
                }
                for qp in sorted(
                    self.slow_queries,
                    key=lambda x: x.execution_time_ms,
                    reverse=True
                )[:10]
            ],
            "index_recommendations": [
                {
                    "table": rec.table_name,
                    "columns": rec.column_names,
                    "type": rec.index_type,
                    "create_sql": rec.create_sql,
                    "reason": rec.reason
                }
                for rec in self.index_recommendations
            ],
            "issues_by_severity": {
                severity: [
                    {
                        "type": issue.issue_type,
                        "description": issue.description,
                        "recommendation": issue.recommendation
                    }
                    for issue in self.query_issues
                    if issue.severity == severity
                ]
                for severity in ["critical", "high", "medium", "low"]
            },
            "generated_at": datetime.now().isoformat()
        }


class QueryCache:
    """Query result caching"""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, tuple] = {}  # hash -> (result, timestamp)

    def get(self, sql: str, params: Optional[tuple] = None) -> Optional[Any]:
        """Get cached query result"""
        cache_key = self._make_key(sql, params)

        if cache_key in self.cache:
            result, timestamp = self.cache[cache_key]

            # Check if expired
            age = (datetime.now() - timestamp).total_seconds()
            if age <= self.ttl_seconds:
                return result
            else:
                del self.cache[cache_key]

        return None

    def set(self, sql: str, params: Optional[tuple], result: Any):
        """Cache query result"""
        cache_key = self._make_key(sql, params)

        # Evict oldest if cache full
        if len(self.cache) >= self.max_size:
            oldest_key = min(
                self.cache.keys(),
                key=lambda k: self.cache[k][1]
            )
            del self.cache[oldest_key]

        self.cache[cache_key] = (result, datetime.now())

    def _make_key(self, sql: str, params: Optional[tuple]) -> str:
        """Generate cache key"""
        key_data = f"{sql}:{params}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def clear(self):
        """Clear cache"""
        self.cache.clear()
```

---

## Implementation Timeline

### Phase 1: Caching Infrastructure (Weeks 1-2)
- [ ] Implement cache manager core (L1/L2)
- [ ] Create cache backends (Memory, Redis)
- [ ] Build cache decorators
- [ ] Add cache warming system
- [ ] Implement cache analytics
- [ ] Write comprehensive tests

### Phase 2: Batch Processing (Weeks 3-4)
- [ ] Build batch processor core
- [ ] Implement chunking strategies
- [ ] Add streaming processor
- [ ] Create checkpoint/resume system
- [ ] Add memory monitoring
- [ ] Write performance benchmarks

### Phase 3: Cost Optimization (Weeks 5-6)
- [ ] Implement cost tracker
- [ ] Build token counter
- [ ] Add budget management
- [ ] Create optimization analyzer
- [ ] Build usage dashboard
- [ ] Add alerting system

### Phase 4: Query Optimization (Weeks 7-8)
- [ ] Build query optimizer
- [ ] Implement plan analyzer
- [ ] Add N+1 detection
- [ ] Create index recommender
- [ ] Build query cache
- [ ] Add monitoring dashboard

### Phase 5: Integration & Testing (Week 9)
- [ ] Integrate all systems
- [ ] Performance benchmarking
- [ ] Load testing
- [ ] Documentation
- [ ] Production deployment

---

## Dependencies & Prerequisites

### Python Packages
```txt
# Caching
redis>=4.5.0
pickle5>=0.0.12

# Batch Processing
psutil>=5.9.0

# Cost Tracking
tiktoken>=0.4.0

# Query Optimization
sqlparse>=0.4.4
```

### Infrastructure
- Redis server (for L2 cache)
- PostgreSQL (for query optimization)
- Monitoring stack (Prometheus/Grafana)

### Configuration
```yaml
# config/performance.yaml
cache:
  l1:
    enabled: true
    size_mb: 512
    policy: lru
  l2:
    enabled: true
    redis_url: redis://localhost:6379/0
    ttl_seconds: 3600

batch:
  default_chunk_size: 100
  max_concurrent: 10
  chunking_strategy: adaptive
  memory_threshold_mb: 1000

cost:
  budget:
    daily: 100.0
    weekly: 500.0
    monthly: 2000.0
  alert_thresholds: [80, 90, 100]

query:
  slow_query_threshold_ms: 1000
  enable_auto_indexing: false
  cache_ttl_seconds: 300
```

---

## Benchmarks & Metrics

### Caching Performance
```python
# Expected metrics:
# - L1 hit rate: >90%
# - L1 access time: <1ms
# - L2 access time: <5ms
# - Cache warming: 10,000 items/second
# - Memory efficiency: <1KB overhead per entry
```

### Batch Processing Performance
```python
# Expected metrics:
# - Fixed chunking: 5,000 items/second
# - Adaptive chunking: 7,000 items/second
# - Memory-based chunking: Stable <1GB memory
# - Checkpoint overhead: <5% performance impact
```

### Cost Optimization Results
```python
# Expected savings:
# - Model substitution: 50-80% cost reduction
# - Caching: 30-50% reduction in API calls
# - Prompt optimization: 20-30% token reduction
# - Batch requests: 15-25% cost reduction
```

### Query Optimization Results
```python
# Expected improvements:
# - Index recommendations: 10-100x query speedup
# - N+1 detection: 90% reduction in query count
# - Query caching: 50-80% faster repeat queries
# - Plan analysis: Identify 95% of slow queries
```

---

**Document Version**: 1.0
**Last Updated**: 2025-10-25
**Status**: Ready for Implementation
