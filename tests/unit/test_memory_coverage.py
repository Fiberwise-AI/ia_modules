"""
Comprehensive unit tests for memory backends.

Covers: RedisConversationMemory, SQLConversationMemory, MemoryManager,
SemanticMemory, InMemoryBackend, SQLiteBackend, VectorBackend.
"""

import asyncio
import json
import math
import os
import time
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from ia_modules.memory.core import ConversationMemory, Message, MessageRole
from ia_modules.memory.memory_manager import Memory, MemoryConfig, MemoryManager, MemoryType
from ia_modules.memory.semantic_memory import SemanticMemory
from ia_modules.memory.storage_backends.in_memory_backend import InMemoryBackend
from ia_modules.memory.storage_backends.sqlite_backend import SQLiteBackend
from ia_modules.memory.storage_backends.vector_backend import VectorBackend
from ia_modules.memory.redis import RedisConversationMemory
from ia_modules.memory.sql import SQLConversationMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_memory(content="test", memory_type=MemoryType.WORKING, importance=0.5, **kwargs):
    """Create a Memory object for testing."""
    return Memory(
        content=content,
        memory_type=memory_type,
        timestamp=kwargs.pop("timestamp", time.time()),
        importance=importance,
        metadata=kwargs.pop("metadata", {}),
        access_count=kwargs.pop("access_count", 0),
        last_accessed=kwargs.pop("last_accessed", None),
        embedding=kwargs.pop("embedding", None),
        id=kwargs.pop("id", None),
    )


class _TestableRedisConversationMemory(RedisConversationMemory):
    """Concrete subclass of RedisConversationMemory for testing.

    Implements the abstract methods add_message and get_messages using the
    mock Redis client so the rest of the inherited behaviour can be tested.
    """

    async def add_message(
        self, thread_id, role, content, user_id=None, metadata=None,
        function_call=None, function_name=None,
    ):
        msg_id = str(uuid.uuid4())
        msg = Message(
            message_id=msg_id, thread_id=thread_id, user_id=user_id,
            role=MessageRole(role), content=content,
            metadata=metadata or {},
            function_call=function_call, function_name=function_name,
        )
        ts = time.time()
        thread_key = f"thread:{thread_id}"
        message_key = f"message:{thread_id}:{msg_id}"
        await self.redis.zadd(thread_key, {msg_id: ts})
        await self.redis.set(message_key, json.dumps(msg.to_dict()))
        await self.redis.expire(thread_key, self.ttl)
        await self.redis.expire(message_key, self.ttl)
        return msg_id

    async def get_messages(self, thread_id, limit=10, offset=0):
        thread_key = f"thread:{thread_id}"
        message_ids = await self.redis.zrevrange(thread_key, offset, offset + limit - 1)
        if not message_ids:
            return []
        messages = []
        for mid in message_ids:
            if isinstance(mid, bytes):
                mid = mid.decode("utf-8")
            message_key = f"message:{thread_id}:{mid}"
            msg_json = await self.redis.get(message_key)
            if msg_json:
                if isinstance(msg_json, bytes):
                    msg_json = msg_json.decode("utf-8")
                messages.append(self._dict_to_message(json.loads(msg_json)))
        return messages


# ===================================================================
# InMemoryBackend Tests
# ===================================================================

class TestInMemoryBackend:

    async def test_store_and_retrieve(self):
        backend = InMemoryBackend()
        mem = _make_memory(content="hello")
        await backend.store(mem)
        result = await backend.retrieve(mem.id)
        assert result is mem

    async def test_retrieve_missing(self):
        backend = InMemoryBackend()
        assert await backend.retrieve("nonexistent") is None

    async def test_delete_existing(self):
        backend = InMemoryBackend()
        mem = _make_memory()
        await backend.store(mem)
        assert await backend.delete(mem.id) is True
        assert await backend.retrieve(mem.id) is None

    async def test_delete_missing(self):
        backend = InMemoryBackend()
        assert await backend.delete("nope") is False

    async def test_list_all(self):
        backend = InMemoryBackend()
        m1 = _make_memory(content="a")
        m2 = _make_memory(content="b")
        await backend.store(m1)
        await backend.store(m2)
        all_mems = await backend.list_all()
        assert len(all_mems) == 2

    async def test_clear(self):
        backend = InMemoryBackend()
        await backend.store(_make_memory())
        await backend.clear()
        assert await backend.count() == 0

    async def test_count(self):
        backend = InMemoryBackend()
        assert await backend.count() == 0
        await backend.store(_make_memory())
        assert await backend.count() == 1


# ===================================================================
# SQLiteBackend Tests
# ===================================================================

class TestSQLiteBackend:

    @pytest.fixture
    def backend(self, tmp_path):
        db_path = str(tmp_path / "test_mem.db")
        return SQLiteBackend(db_path=db_path)

    async def test_store_and_retrieve(self, backend):
        mem = _make_memory(content="sqlite test", importance=0.7)
        await backend.store(mem)
        result = await backend.retrieve(mem.id)
        assert result is not None
        assert result.content == "sqlite test"
        assert result.importance == 0.7

    async def test_retrieve_missing(self, backend):
        assert await backend.retrieve("no-such-id") is None

    async def test_store_with_embedding(self, backend):
        mem = _make_memory(content="embedded", embedding=[0.1, 0.2, 0.3])
        await backend.store(mem)
        result = await backend.retrieve(mem.id)
        assert result.embedding == [0.1, 0.2, 0.3]

    async def test_store_upsert(self, backend):
        mem = _make_memory(content="v1", id="fixed-id")
        await backend.store(mem)
        mem2 = _make_memory(content="v2", id="fixed-id")
        await backend.store(mem2)
        result = await backend.retrieve("fixed-id")
        assert result.content == "v2"

    async def test_delete(self, backend):
        mem = _make_memory()
        await backend.store(mem)
        assert await backend.delete(mem.id) is True
        assert await backend.retrieve(mem.id) is None

    async def test_delete_nonexistent(self, backend):
        assert await backend.delete("nope") is False

    async def test_list_all(self, backend):
        await backend.store(_make_memory(content="a"))
        await backend.store(_make_memory(content="b"))
        all_mems = await backend.list_all()
        assert len(all_mems) == 2

    async def test_clear(self, backend):
        await backend.store(_make_memory())
        await backend.clear()
        assert await backend.count() == 0

    async def test_count(self, backend):
        assert await backend.count() == 0
        await backend.store(_make_memory())
        assert await backend.count() == 1

    async def test_store_error_handling(self, backend):
        """Store should handle errors gracefully (logs error, rolls back)."""
        mem = _make_memory()
        # Create a mock connection whose cursor.execute raises inside the try block
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("forced error")
        mock_conn.cursor.return_value = mock_cursor
        with patch("ia_modules.memory.storage_backends.sqlite_backend.sqlite3.connect", return_value=mock_conn):
            # Should not raise, just log error and rollback
            await backend.store(mem)
        mock_conn.rollback.assert_called_once()
        mock_conn.close.assert_called_once()

    async def test_row_to_memory_types(self, backend):
        """Verify _row_to_memory correctly reconstructs Memory with all types."""
        mem = _make_memory(
            content="typed",
            memory_type=MemoryType.SEMANTIC,
            metadata={"key": "val"},
            embedding=None,
        )
        await backend.store(mem)
        result = await backend.retrieve(mem.id)
        assert result.memory_type == MemoryType.SEMANTIC
        assert result.metadata == {"key": "val"}


# ===================================================================
# VectorBackend Tests (fallback mode, no chromadb)
# ===================================================================

class TestVectorBackendFallback:
    """Test VectorBackend with ChromaDB unavailable (fallback mode)."""

    @pytest.fixture
    def backend(self):
        # Force fallback mode by patching chromadb import to fail
        with patch.dict("sys.modules", {"chromadb": None}):
            vb = VectorBackend.__new__(VectorBackend)
            vb.available = False
            vb.fallback_storage = {}
            return vb

    async def test_store_and_retrieve_fallback(self, backend):
        mem = _make_memory(content="fallback")
        await backend.store(mem)
        result = await backend.retrieve(mem.id)
        assert result is mem

    async def test_retrieve_missing_fallback(self, backend):
        assert await backend.retrieve("nope") is None

    async def test_search_fallback(self, backend):
        m1 = _make_memory(content="first")
        m2 = _make_memory(content="second")
        await backend.store(m1)
        await backend.store(m2)
        results = await backend.search([0.1, 0.2], k=1)
        assert len(results) == 1

    async def test_delete_fallback(self, backend):
        mem = _make_memory()
        await backend.store(mem)
        assert await backend.delete(mem.id) is True
        assert await backend.delete(mem.id) is False

    async def test_list_all_fallback(self, backend):
        await backend.store(_make_memory(content="a"))
        await backend.store(_make_memory(content="b"))
        assert len(await backend.list_all()) == 2

    async def test_clear_fallback(self, backend):
        await backend.store(_make_memory())
        await backend.clear()
        assert await backend.count() == 0

    async def test_count_fallback(self, backend):
        assert await backend.count() == 0
        await backend.store(_make_memory())
        assert await backend.count() == 1


class TestVectorBackendWithChroma:
    """Test VectorBackend when ChromaDB is 'available' (mocked)."""

    @pytest.fixture
    def backend(self):
        vb = VectorBackend.__new__(VectorBackend)
        vb.available = True
        vb.client = MagicMock()
        vb.collection = MagicMock()
        return vb

    async def test_store_with_embedding(self, backend):
        mem = _make_memory(content="emb", embedding=[0.1, 0.2])
        await backend.store(mem)
        backend.collection.upsert.assert_called_once()
        call_kwargs = backend.collection.upsert.call_args
        assert call_kwargs[1]["embeddings"] == [[0.1, 0.2]]

    async def test_store_without_embedding(self, backend):
        mem = _make_memory(content="no-emb", embedding=None)
        await backend.store(mem)
        backend.collection.upsert.assert_called_once()
        call_kwargs = backend.collection.upsert.call_args
        assert "embeddings" not in call_kwargs[1]

    async def test_store_error(self, backend):
        backend.collection.upsert.side_effect = Exception("db error")
        mem = _make_memory()
        # Should not raise
        await backend.store(mem)

    async def test_retrieve_found(self, backend):
        backend.collection.get.return_value = {
            "ids": ["id1"],
            "documents": ["content"],
            "metadatas": [{"memory_type": "working", "timestamp": 1.0, "importance": 0.5, "access_count": 0}],
            "embeddings": [[0.1]],
        }
        result = await backend.retrieve("id1")
        assert result is not None
        assert result.content == "content"

    async def test_retrieve_not_found(self, backend):
        backend.collection.get.return_value = {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
        assert await backend.retrieve("nope") is None

    async def test_retrieve_error(self, backend):
        backend.collection.get.side_effect = Exception("fail")
        assert await backend.retrieve("x") is None

    async def test_search_with_results(self, backend):
        backend.collection.query.return_value = {
            "ids": [["id1", "id2"]],
            "documents": [["doc1", "doc2"]],
            "metadatas": [[
                {"memory_type": "semantic", "timestamp": 1.0, "importance": 0.8, "access_count": 1},
                {"memory_type": "working", "timestamp": 2.0, "importance": 0.5, "access_count": 0},
            ]],
            "embeddings": [[[0.1], [0.2]]],
        }
        results = await backend.search([0.1], k=2)
        assert len(results) == 2

    async def test_search_empty(self, backend):
        backend.collection.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "embeddings": [[]]}
        results = await backend.search([0.1])
        assert results == []

    async def test_search_error(self, backend):
        backend.collection.query.side_effect = Exception("search fail")
        assert await backend.search([0.1]) == []

    async def test_delete_success(self, backend):
        assert await backend.delete("id1") is True
        backend.collection.delete.assert_called_once()

    async def test_delete_error(self, backend):
        backend.collection.delete.side_effect = Exception("del fail")
        assert await backend.delete("x") is False

    async def test_list_all(self, backend):
        backend.collection.get.return_value = {
            "ids": ["id1"],
            "documents": ["doc1"],
            "metadatas": [{"memory_type": "working", "timestamp": 1.0, "importance": 0.5, "access_count": 0}],
            "embeddings": [[0.1]],
        }
        results = await backend.list_all()
        assert len(results) == 1

    async def test_list_all_empty(self, backend):
        backend.collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}
        results = await backend.list_all()
        assert results == []

    async def test_list_all_error(self, backend):
        backend.collection.get.side_effect = Exception("list fail")
        assert await backend.list_all() == []

    async def test_clear(self, backend):
        backend.collection.name = "memories"
        await backend.clear()
        backend.client.delete_collection.assert_called_once()
        backend.client.create_collection.assert_called_once()

    async def test_clear_error(self, backend):
        backend.collection.name = "memories"
        backend.client.delete_collection.side_effect = Exception("clear fail")
        # Should not raise
        await backend.clear()

    async def test_count(self, backend):
        backend.collection.count.return_value = 42
        assert await backend.count() == 42

    async def test_count_error(self, backend):
        backend.collection.count.side_effect = Exception("count fail")
        assert await backend.count() == 0


# ===================================================================
# SemanticMemory Tests
# ===================================================================

class TestSemanticMemory:

    def test_init_no_embeddings(self):
        sm = SemanticMemory(enable_embeddings=False)
        assert sm.embed_fn is None
        assert sm.memories == {}

    async def test_add_without_embeddings(self):
        sm = SemanticMemory(enable_embeddings=False)
        mem = _make_memory(content="knowledge")
        await sm.add(mem)
        assert mem.id in sm.memories

    async def test_retrieve_empty(self):
        sm = SemanticMemory(enable_embeddings=False)
        results = await sm.retrieve("anything")
        assert results == []

    async def test_keyword_search(self):
        sm = SemanticMemory(enable_embeddings=False)
        m1 = _make_memory(content="the cat sat on the mat")
        m2 = _make_memory(content="the dog ran in the park")
        m3 = _make_memory(content="unrelated stuff here")
        await sm.add(m1)
        await sm.add(m2)
        await sm.add(m3)
        results = await sm.retrieve("cat mat", k=2)
        assert len(results) > 0
        assert results[0].content == "the cat sat on the mat"

    async def test_keyword_search_no_match(self):
        sm = SemanticMemory(enable_embeddings=False)
        await sm.add(_make_memory(content="something"))
        results = await sm.retrieve("zzzznotfound")
        assert results == []

    async def test_add_with_sync_embed_fn(self):
        """Test with a synchronous embedding function."""
        sm = SemanticMemory(enable_embeddings=False)
        sm.embed_fn = lambda text: [0.1, 0.2, 0.3]
        mem = _make_memory(content="test")
        await sm.add(mem)
        assert mem.id in sm.embeddings
        assert sm.embeddings[mem.id] == [0.1, 0.2, 0.3]

    async def test_add_with_async_embed_fn(self):
        """Test with an async embedding function."""
        sm = SemanticMemory(enable_embeddings=False)

        async def async_embed(text):
            return [0.4, 0.5, 0.6]

        sm.embed_fn = async_embed
        mem = _make_memory(content="async test")
        await sm.add(mem)
        assert sm.embeddings[mem.id] == [0.4, 0.5, 0.6]

    async def test_add_embed_fn_fails(self):
        """Embedding failure should not prevent memory storage."""
        sm = SemanticMemory(enable_embeddings=False)
        sm.embed_fn = lambda text: (_ for _ in ()).throw(RuntimeError("embed fail"))
        mem = _make_memory(content="still stored")
        await sm.add(mem)
        assert mem.id in sm.memories
        assert mem.id not in sm.embeddings

    async def test_retrieve_with_sync_embeddings(self):
        sm = SemanticMemory(enable_embeddings=False)
        sm.embed_fn = lambda text: [1.0, 0.0] if "cat" in text else [0.0, 1.0]
        m1 = _make_memory(content="cat")
        m2 = _make_memory(content="dog")
        await sm.add(m1)
        await sm.add(m2)
        results = await sm.retrieve("cat", k=1)
        assert len(results) == 1
        assert results[0].content == "cat"

    async def test_retrieve_with_async_embeddings(self):
        sm = SemanticMemory(enable_embeddings=False)

        async def async_embed(text):
            return [1.0, 0.0] if "cat" in text else [0.0, 1.0]

        sm.embed_fn = async_embed
        m1 = _make_memory(content="cat")
        m2 = _make_memory(content="dog")
        await sm.add(m1)
        await sm.add(m2)
        results = await sm.retrieve("cat", k=1)
        assert len(results) == 1
        assert results[0].content == "cat"

    async def test_retrieve_embed_fails_falls_back_to_keyword(self):
        """If embed fails during retrieval, fall back to keyword search."""
        sm = SemanticMemory(enable_embeddings=False)

        call_count = 0

        def flaky_embed(text):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return [0.1, 0.2]
            raise RuntimeError("embed fail")

        sm.embed_fn = flaky_embed
        m1 = _make_memory(content="machine learning topic")
        m2 = _make_memory(content="cooking recipes")
        await sm.add(m1)
        await sm.add(m2)
        # Now embed_fn will fail on retrieval, should fall back to keyword
        results = await sm.retrieve("machine learning")
        assert len(results) > 0

    def test_cosine_similarity(self):
        sm = SemanticMemory(enable_embeddings=False)
        assert sm._cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
        assert sm._cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
        assert sm._cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_cosine_similarity_zero_vector(self):
        sm = SemanticMemory(enable_embeddings=False)
        assert sm._cosine_similarity([0, 0], [1, 0]) == 0.0

    async def test_clear(self):
        sm = SemanticMemory(enable_embeddings=False)
        await sm.add(_make_memory(content="x"))
        await sm.clear()
        assert sm.memories == {}
        assert sm.embeddings == {}

    def test_init_with_embeddings_no_library(self):
        """When enable_embeddings=True but no library available, should raise ImportError."""
        import builtins
        real_import = builtins.__import__

        def _blocked_import(name, *args, **kwargs):
            if name == "openai" or name.startswith("openai."):
                raise ImportError(f"Blocked {name} for testing")
            return real_import(name, *args, **kwargs)

        with patch("ia_modules.memory.semantic_memory.SentenceTransformer", None):
            with patch("builtins.__import__", side_effect=_blocked_import):
                with pytest.raises(ImportError):
                    SemanticMemory(enable_embeddings=True)


# ===================================================================
# MemoryManager Tests
# ===================================================================

class TestMemoryManager:

    @pytest.fixture
    def config(self):
        return MemoryConfig(
            semantic_enabled=False,
            episodic_enabled=False,
            compression_enabled=False,
            enable_embeddings=False,
            storage_backend="in_memory",
            working_memory_size=5,
        )

    @pytest.fixture
    def manager(self, config):
        return MemoryManager(config)

    async def test_add_memory(self, manager):
        mem_id = await manager.add("hello world")
        assert mem_id is not None
        assert mem_id in manager.memories

    async def test_add_with_metadata(self, manager):
        mem_id = await manager.add("test", metadata={"importance": 0.9})
        assert manager.memories[mem_id].importance == 0.9

    async def test_retrieve_working_memory(self, manager):
        await manager.add("working item")
        results = await manager.retrieve("working")
        assert len(results) > 0

    async def test_retrieve_with_min_importance(self, manager):
        await manager.add("low", metadata={"importance": 0.1})
        await manager.add("high", metadata={"importance": 0.9})
        results = await manager.retrieve("item", min_importance=0.5)
        assert all(r.importance >= 0.5 for r in results)

    async def test_retrieve_updates_access_stats(self, manager):
        mem_id = await manager.add("stats test")
        results = await manager.retrieve("stats")
        for r in results:
            assert r.access_count >= 1
            assert r.last_accessed is not None

    async def test_retrieve_with_memory_type_filter(self, manager):
        await manager.add("item")
        results = await manager.retrieve("item", memory_types=[MemoryType.SEMANTIC])
        # With semantic disabled, should still not crash
        assert isinstance(results, list)

    async def test_clear_all(self, manager):
        await manager.add("a")
        await manager.add("b")
        await manager.clear()
        assert len(manager.memories) == 0

    async def test_clear_specific_type(self, manager):
        await manager.add("working item")
        await manager.clear(memory_types=[MemoryType.WORKING])
        working_count = sum(1 for m in manager.memories.values() if m.memory_type == MemoryType.WORKING)
        assert working_count == 0

    async def test_get_stats_empty(self, manager):
        stats = await manager.get_stats()
        assert stats["total_memories"] == 0
        assert stats["avg_importance"] == 0.0
        assert stats["total_accesses"] == 0

    async def test_get_stats_with_data(self, manager):
        await manager.add("a", metadata={"importance": 0.6})
        await manager.add("b", metadata={"importance": 0.4})
        stats = await manager.get_stats()
        assert stats["total_memories"] == 2
        assert stats["avg_importance"] == pytest.approx(0.5)

    async def test_get_context_window_empty(self, manager):
        ctx = await manager.get_context_window()
        assert ctx == ""

    async def test_get_context_window_with_working_memory(self, manager):
        await manager.add("recent context item")
        ctx = await manager.get_context_window()
        assert "Recent Context" in ctx
        assert "recent context item" in ctx

    async def test_get_context_window_with_query(self):
        config = MemoryConfig(
            semantic_enabled=True,
            episodic_enabled=True,
            compression_enabled=False,
            enable_embeddings=False,
            storage_backend="in_memory",
        )
        mgr = MemoryManager(config)
        await mgr.add("working item")
        await mgr.add("semantic fact", metadata={"is_fact": True})
        await mgr.add("episodic event", metadata={"is_event": True})
        ctx = await mgr.get_context_window(query="semantic fact")
        assert "Recent Context" in ctx or "Relevant Background" in ctx

    async def test_get_context_window_truncation(self, manager):
        await manager.add("x" * 20000)
        ctx = await manager.get_context_window(max_tokens=100)
        assert ctx.endswith("...")

    async def test_determine_memory_type_from_metadata(self, manager):
        # Explicit memory_type in metadata
        t = manager._determine_memory_type("", {"memory_type": "semantic"})
        assert t == MemoryType.SEMANTIC

    async def test_determine_memory_type_fact(self, manager):
        t = manager._determine_memory_type("", {"is_fact": True})
        assert t == MemoryType.SEMANTIC

    async def test_determine_memory_type_knowledge(self, manager):
        t = manager._determine_memory_type("", {"is_knowledge": True})
        assert t == MemoryType.SEMANTIC

    async def test_determine_memory_type_event(self, manager):
        t = manager._determine_memory_type("", {"is_event": True})
        assert t == MemoryType.EPISODIC

    async def test_determine_memory_type_timestamp(self, manager):
        t = manager._determine_memory_type("", {"timestamp": 123.0})
        assert t == MemoryType.EPISODIC

    async def test_determine_memory_type_default(self, manager):
        t = manager._determine_memory_type("hello", {})
        assert t == MemoryType.WORKING

    async def test_rank_memories(self, manager):
        now = time.time()
        m1 = _make_memory(content="a", importance=0.9, access_count=5, last_accessed=now)
        m2 = _make_memory(content="b", importance=0.1, access_count=0, last_accessed=None)
        m3 = _make_memory(content="c", importance=0.5, memory_type=MemoryType.WORKING)
        ranked = manager._rank_memories([m1, m2, m3], "test")
        # m1 has high importance + access, should be first or near top
        assert ranked[0].importance >= ranked[-1].importance or ranked[0].content == "a"

    def test_init_storage_backend_unknown(self):
        config = MemoryConfig(storage_backend="unknown_backend", enable_embeddings=False,
                              semantic_enabled=False, episodic_enabled=False, compression_enabled=False)
        with pytest.raises(ValueError, match="Unknown storage backend"):
            MemoryManager(config)

    def test_init_storage_backend_sqlite(self, tmp_path):
        config = MemoryConfig(
            storage_backend="sqlite",
            enable_embeddings=False,
            semantic_enabled=False,
            episodic_enabled=False,
            compression_enabled=False,
        )
        mgr = MemoryManager(config)
        assert isinstance(mgr.storage, SQLiteBackend)

    async def test_compress_no_compressor(self, manager):
        result = await manager.compress()
        assert result == 0

    async def test_compress_no_eligible_memories(self):
        config = MemoryConfig(
            compression_enabled=True,
            enable_embeddings=False,
            semantic_enabled=False,
            episodic_enabled=False,
            storage_backend="in_memory",
            compression_threshold=1000,
        )
        mgr = MemoryManager(config)
        # Add a recent memory (not old enough to compress)
        await mgr.add("recent")
        result = await mgr.compress()
        assert result == 0

    async def test_compress_old_memories(self):
        config = MemoryConfig(
            compression_enabled=True,
            enable_embeddings=False,
            semantic_enabled=False,
            episodic_enabled=False,
            storage_backend="in_memory",
            compression_threshold=10000,  # High so auto-compress doesn't trigger
        )
        mgr = MemoryManager(config)

        # Manually add old memories
        for i in range(5):
            mem = Memory(
                content=f"old memory {i}",
                memory_type=MemoryType.WORKING,
                timestamp=time.time() - 200000,  # >24h ago
                importance=0.3,  # < 0.8 threshold
            )
            mgr.memories[mem.id] = mem
            await mgr.storage.store(mem)

        result = await mgr.compress()
        assert result == 5

    async def test_auto_compression_on_threshold(self):
        config = MemoryConfig(
            compression_enabled=True,
            enable_embeddings=False,
            semantic_enabled=False,
            episodic_enabled=False,
            storage_backend="in_memory",
            compression_threshold=3,
        )
        mgr = MemoryManager(config)
        # Add enough memories to trigger compression
        for i in range(3):
            await mgr.add(f"item {i}")
        # Compression was called, but recent memories won't be compressed
        # Just verify no crash

    async def test_add_routes_to_semantic(self):
        config = MemoryConfig(
            semantic_enabled=True,
            episodic_enabled=False,
            compression_enabled=False,
            enable_embeddings=False,
            storage_backend="in_memory",
        )
        mgr = MemoryManager(config)
        await mgr.add("fact", metadata={"is_fact": True})
        assert len(mgr.semantic.memories) == 1

    async def test_add_routes_to_episodic(self):
        config = MemoryConfig(
            semantic_enabled=False,
            episodic_enabled=True,
            compression_enabled=False,
            enable_embeddings=False,
            storage_backend="in_memory",
        )
        mgr = MemoryManager(config)
        await mgr.add("event", metadata={"is_event": True})
        assert len(mgr.episodic.memories) == 1

    async def test_retrieve_from_all_subsystems(self):
        config = MemoryConfig(
            semantic_enabled=True,
            episodic_enabled=True,
            compression_enabled=False,
            enable_embeddings=False,
            storage_backend="in_memory",
        )
        mgr = MemoryManager(config)
        await mgr.add("fact knowledge", metadata={"is_fact": True})
        await mgr.add("event happened", metadata={"is_event": True})
        await mgr.add("working item")
        results = await mgr.retrieve("knowledge event working", k=10)
        assert len(results) >= 1


# ===================================================================
# Memory dataclass Tests
# ===================================================================

class TestMemoryDataclass:

    def test_auto_id_generation(self):
        mem = Memory(content="test", memory_type=MemoryType.WORKING, timestamp=1.0)
        assert mem.id is not None
        assert len(mem.id) > 0

    def test_explicit_id(self):
        mem = Memory(content="test", memory_type=MemoryType.WORKING, timestamp=1.0, id="my-id")
        assert mem.id == "my-id"

    def test_default_fields(self):
        mem = Memory(content="test", memory_type=MemoryType.WORKING, timestamp=1.0)
        assert mem.metadata == {}
        assert mem.importance == 0.5
        assert mem.access_count == 0
        assert mem.last_accessed is None
        assert mem.embedding is None


class TestMemoryType:

    def test_all_types(self):
        assert MemoryType.SEMANTIC.value == "semantic"
        assert MemoryType.EPISODIC.value == "episodic"
        assert MemoryType.WORKING.value == "working"
        assert MemoryType.COMPRESSED.value == "compressed"


class TestMemoryConfig:

    def test_defaults(self):
        cfg = MemoryConfig()
        assert cfg.semantic_enabled is True
        assert cfg.episodic_enabled is True
        assert cfg.working_memory_size == 10
        assert cfg.compression_threshold == 50
        assert cfg.storage_backend == "in_memory"


# ===================================================================
# RedisConversationMemory Tests
# ===================================================================

class TestRedisConversationMemory:

    @pytest.fixture
    def mock_redis(self):
        return AsyncMock()

    @pytest.fixture
    def redis_mem(self, mock_redis):
        return _TestableRedisConversationMemory(redis_client=mock_redis, ttl=3600)

    async def test_init(self, redis_mem, mock_redis):
        assert redis_mem.redis is mock_redis
        assert redis_mem.ttl == 3600

    async def test_get_user_messages_empty(self, redis_mem, mock_redis):
        mock_redis.zrevrange.return_value = []
        result = await redis_mem.get_user_messages("user1")
        assert result == []

    async def test_get_user_messages_with_data(self, redis_mem, mock_redis):
        mock_redis.zrevrange.return_value = [b"msg-1"]
        # scan returns (cursor, keys) -- first call finds thread, cursor 0 stops
        mock_redis.scan.return_value = (0, [b"user:user1:thread:thread-1"])
        mock_redis.sismember.return_value = True
        msg_data = {
            "message_id": "msg-1",
            "thread_id": "thread-1",
            "user_id": "user1",
            "role": "user",
            "content": "hello",
            "metadata": {},
            "timestamp": datetime.now().isoformat(),
        }
        mock_redis.get.return_value = json.dumps(msg_data).encode("utf-8")
        result = await redis_mem.get_user_messages("user1")
        assert len(result) == 1
        assert result[0].content == "hello"

    async def test_get_user_messages_string_keys(self, redis_mem, mock_redis):
        """Test with string (non-bytes) message IDs and keys."""
        mock_redis.zrevrange.return_value = ["msg-1"]
        mock_redis.scan.return_value = (0, ["user:user1:thread:thread-1"])
        mock_redis.sismember.return_value = True
        msg_data = {
            "message_id": "msg-1",
            "thread_id": "thread-1",
            "role": "user",
            "content": "hi",
            "metadata": {},
            "timestamp": datetime.now().isoformat(),
        }
        mock_redis.get.return_value = json.dumps(msg_data)
        result = await redis_mem.get_user_messages("user1")
        assert len(result) == 1

    async def test_get_user_messages_no_thread_found(self, redis_mem, mock_redis):
        mock_redis.zrevrange.return_value = [b"msg-1"]
        mock_redis.scan.return_value = (0, [])
        result = await redis_mem.get_user_messages("user1")
        assert result == []

    async def test_get_user_messages_message_not_found(self, redis_mem, mock_redis):
        mock_redis.zrevrange.return_value = [b"msg-1"]
        mock_redis.scan.return_value = (0, [b"user:user1:thread:thread-1"])
        mock_redis.sismember.return_value = True
        mock_redis.get.return_value = None
        result = await redis_mem.get_user_messages("user1")
        assert result == []

    async def test_search_messages_global(self, redis_mem, mock_redis):
        """Search all threads (no thread_id, no user_id)."""
        msg_data = {
            "message_id": "msg-1",
            "thread_id": "t1",
            "role": "user",
            "content": "hello world",
            "metadata": {},
            "timestamp": datetime.now().isoformat(),
        }
        mock_redis.scan.return_value = (0, [b"message:t1:msg-1"])
        mock_redis.get.return_value = json.dumps(msg_data).encode("utf-8")
        results = await redis_mem.search_messages("hello")
        assert len(results) == 1
        assert results[0].content == "hello world"

    async def test_search_messages_global_string_keys(self, redis_mem, mock_redis):
        msg_data = {
            "message_id": "msg-1",
            "thread_id": "t1",
            "role": "user",
            "content": "findme",
            "metadata": {},
            "timestamp": datetime.now().isoformat(),
        }
        mock_redis.scan.return_value = (0, ["message:t1:msg-1"])
        mock_redis.get.return_value = json.dumps(msg_data)
        results = await redis_mem.search_messages("findme")
        assert len(results) == 1

    async def test_search_messages_global_no_match(self, redis_mem, mock_redis):
        msg_data = {
            "message_id": "msg-1",
            "thread_id": "t1",
            "role": "user",
            "content": "nothing here",
            "metadata": {},
            "timestamp": datetime.now().isoformat(),
        }
        mock_redis.scan.return_value = (0, [b"message:t1:msg-1"])
        mock_redis.get.return_value = json.dumps(msg_data).encode("utf-8")
        results = await redis_mem.search_messages("zzz_not_found")
        assert results == []

    async def test_search_messages_global_null_message(self, redis_mem, mock_redis):
        mock_redis.scan.return_value = (0, [b"message:t1:msg-1"])
        mock_redis.get.return_value = None
        results = await redis_mem.search_messages("anything")
        assert results == []

    async def test_search_messages_by_user(self, redis_mem, mock_redis):
        """Search by user_id delegates to get_user_messages."""
        mock_redis.zrevrange.return_value = [b"msg-1"]
        mock_redis.scan.return_value = (0, [b"user:u1:thread:t1"])
        mock_redis.sismember.return_value = True
        msg_data = {
            "message_id": "msg-1",
            "thread_id": "t1",
            "user_id": "u1",
            "role": "user",
            "content": "match this",
            "metadata": {},
            "timestamp": datetime.now().isoformat(),
        }
        mock_redis.get.return_value = json.dumps(msg_data).encode("utf-8")
        results = await redis_mem.search_messages("match", user_id="u1")
        assert len(results) == 1

    async def test_delete_thread(self, redis_mem, mock_redis):
        mock_redis.zrange.return_value = [b"msg-1", b"msg-2"]
        mock_redis.delete.return_value = 1
        count = await redis_mem.delete_thread("thread-1")
        assert count == 2

    async def test_delete_thread_string_ids(self, redis_mem, mock_redis):
        mock_redis.zrange.return_value = ["msg-1"]
        mock_redis.delete.return_value = 1
        count = await redis_mem.delete_thread("thread-1")
        assert count == 1

    async def test_delete_thread_empty(self, redis_mem, mock_redis):
        mock_redis.zrange.return_value = []
        count = await redis_mem.delete_thread("thread-1")
        assert count == 0

    async def test_delete_thread_not_found(self, redis_mem, mock_redis):
        mock_redis.zrange.return_value = [b"msg-1"]
        mock_redis.delete.return_value = 0  # message key not found
        count = await redis_mem.delete_thread("thread-1")
        assert count == 0

    async def test_get_thread_stats_empty(self, redis_mem, mock_redis):
        mock_redis.zcard.return_value = 0
        stats = await redis_mem.get_thread_stats("thread-1")
        assert stats["message_count"] == 0
        assert stats["thread_id"] == "thread-1"

    async def test_get_thread_stats_with_data(self, redis_mem, mock_redis):
        mock_redis.zcard.return_value = 5
        ts = time.time()
        mock_redis.zrange.return_value = [(b"msg-1", ts - 100)]
        mock_redis.zrevrange.return_value = [(b"msg-5", ts)]
        stats = await redis_mem.get_thread_stats("thread-1")
        assert stats["message_count"] == 5
        assert stats["first_message"] is not None
        assert stats["last_message"] is not None

    async def test_close(self, redis_mem, mock_redis):
        await redis_mem.close()
        mock_redis.close.assert_called_once()

    def test_dict_to_message(self, redis_mem):
        data = {
            "message_id": "m1",
            "thread_id": "t1",
            "user_id": "u1",
            "role": "assistant",
            "content": "hi",
            "metadata": {"key": "val"},
            "timestamp": datetime.now().isoformat(),
            "function_call": {"name": "fn"},
            "function_name": "fn",
        }
        msg = redis_mem._dict_to_message(data)
        assert msg.message_id == "m1"
        assert msg.role == MessageRole.ASSISTANT
        assert msg.function_call == {"name": "fn"}


# ===================================================================
# SQLConversationMemory Tests
# ===================================================================

class TestSQLConversationMemory:

    @pytest.fixture
    def mock_db(self):
        db = AsyncMock()
        return db

    @pytest.fixture
    def sql_mem(self, mock_db):
        return SQLConversationMemory(db_manager=mock_db)

    async def test_add_message(self, sql_mem, mock_db):
        msg_id = await sql_mem.add_message(
            thread_id="t1",
            role="user",
            content="hello",
            user_id="u1",
            metadata={"key": "val"},
        )
        assert msg_id is not None
        mock_db.execute_async.assert_called_once()

    async def test_add_message_with_function_call(self, sql_mem, mock_db):
        msg_id = await sql_mem.add_message(
            thread_id="t1",
            role="function",
            content="result",
            function_call={"name": "fn", "args": {}},
            function_name="fn",
        )
        assert msg_id is not None
        call_args = mock_db.execute_async.call_args
        params = call_args[0][1]
        assert params["function_name"] == "fn"
        assert params["function_call"] is not None

    async def test_add_message_minimal(self, sql_mem, mock_db):
        msg_id = await sql_mem.add_message(
            thread_id="t1",
            role="user",
            content="bare message",
        )
        assert msg_id is not None
        params = mock_db.execute_async.call_args[0][1]
        assert params["user_id"] is None
        assert params["metadata"] == "{}"
        assert params["function_call"] is None

    async def test_get_messages(self, sql_mem, mock_db):
        mock_result = MagicMock()
        mock_result.data = [
            {
                "message_id": "m1",
                "thread_id": "t1",
                "user_id": "u1",
                "role": "user",
                "content": "hi",
                "metadata": "{}",
                "timestamp": datetime.now().isoformat(),
                "function_call": None,
                "function_name": None,
            }
        ]
        mock_db.fetch_all.return_value = mock_result
        messages = await sql_mem.get_messages("t1")
        assert len(messages) == 1
        assert messages[0].content == "hi"

    async def test_get_messages_empty(self, sql_mem, mock_db):
        mock_result = MagicMock()
        mock_result.data = []
        mock_db.fetch_all.return_value = mock_result
        messages = await sql_mem.get_messages("t1")
        assert messages == []

    async def test_get_user_messages(self, sql_mem, mock_db):
        mock_result = MagicMock()
        mock_result.data = [
            {
                "message_id": "m1",
                "thread_id": "t1",
                "user_id": "u1",
                "role": "assistant",
                "content": "reply",
                "metadata": {"k": "v"},
                "timestamp": datetime.now().isoformat(),
                "function_call": None,
                "function_name": None,
            }
        ]
        mock_db.fetch_all.return_value = mock_result
        messages = await sql_mem.get_user_messages("u1")
        assert len(messages) == 1
        assert messages[0].role == MessageRole.ASSISTANT

    async def test_search_messages_all_filters(self, sql_mem, mock_db):
        mock_result = MagicMock()
        mock_result.data = [
            {
                "message_id": "m1",
                "thread_id": "t1",
                "user_id": "u1",
                "role": "user",
                "content": "search result",
                "metadata": "{}",
                "timestamp": datetime.now().isoformat(),
                "function_call": None,
                "function_name": None,
            }
        ]
        mock_db.fetch_all.return_value = mock_result
        results = await sql_mem.search_messages("search", thread_id="t1", user_id="u1")
        assert len(results) == 1
        # Check that the query had all conditions
        call_args = mock_db.fetch_all.call_args[0]
        assert "thread_id" in call_args[1]
        assert "user_id" in call_args[1]

    async def test_search_messages_no_filters(self, sql_mem, mock_db):
        mock_result = MagicMock()
        mock_result.data = []
        mock_db.fetch_all.return_value = mock_result
        results = await sql_mem.search_messages("query")
        assert results == []

    async def test_delete_thread(self, sql_mem, mock_db):
        mock_result = MagicMock()
        mock_result.row_count = 5
        mock_db.execute_async.return_value = mock_result
        count = await sql_mem.delete_thread("t1")
        assert count == 5

    async def test_get_thread_stats(self, sql_mem, mock_db):
        mock_result = MagicMock()
        mock_result.get_first_row.return_value = {
            "count": 10,
            "first_msg": datetime.now().isoformat(),
            "last_msg": datetime.now().isoformat(),
            "participant_count": 2,
        }
        mock_db.fetch_one.return_value = mock_result
        stats = await sql_mem.get_thread_stats("t1")
        assert stats["message_count"] == 10
        assert stats["first_message"] is not None
        assert stats["thread_id"] == "t1"

    async def test_get_thread_stats_empty(self, sql_mem, mock_db):
        mock_result = MagicMock()
        mock_result.get_first_row.return_value = {
            "count": 0,
            "first_msg": None,
            "last_msg": None,
            "participant_count": 0,
        }
        mock_db.fetch_one.return_value = mock_result
        stats = await sql_mem.get_thread_stats("t1")
        assert stats["message_count"] == 0
        assert stats["first_message"] is None

    async def test_close(self, sql_mem, mock_db):
        await sql_mem.close()
        mock_db.disconnect.assert_called_once()

    def test_row_to_message_with_string_metadata(self, sql_mem):
        row = {
            "message_id": "m1",
            "thread_id": "t1",
            "user_id": "u1",
            "role": "user",
            "content": "test",
            "metadata": '{"key": "val"}',
            "timestamp": datetime.now().isoformat(),
            "function_call": '{"name": "fn"}',
            "function_name": "fn",
        }
        msg = sql_mem._row_to_message(row)
        assert msg.metadata == {"key": "val"}
        assert msg.function_call == {"name": "fn"}

    def test_row_to_message_with_dict_metadata(self, sql_mem):
        row = {
            "message_id": "m1",
            "thread_id": "t1",
            "user_id": None,
            "role": "system",
            "content": "system msg",
            "metadata": {"already": "parsed"},
            "timestamp": datetime.now(),
            "function_call": None,
            "function_name": None,
        }
        msg = sql_mem._row_to_message(row)
        assert msg.metadata == {"already": "parsed"}
        assert msg.function_call is None
        assert msg.role == MessageRole.SYSTEM


# ===================================================================
# Integration-style: MemoryManager with SQLite backend
# ===================================================================

class TestMemoryManagerWithSQLite:

    @pytest.fixture
    def manager(self, tmp_path):
        # Patch SQLiteBackend to use tmp_path
        db_path = str(tmp_path / "mem_test.db")
        config = MemoryConfig(
            semantic_enabled=False,
            episodic_enabled=False,
            compression_enabled=False,
            enable_embeddings=False,
            storage_backend="in_memory",  # Will swap after init
        )
        mgr = MemoryManager(config)
        mgr.storage = SQLiteBackend(db_path=db_path)
        return mgr

    async def test_add_and_retrieve_sqlite(self, manager):
        mem_id = await manager.add("stored in sqlite")
        assert mem_id in manager.memories
        stored = await manager.storage.retrieve(mem_id)
        assert stored is not None
        assert stored.content == "stored in sqlite"

    async def test_clear_with_sqlite(self, manager):
        await manager.add("item1")
        await manager.add("item2")
        await manager.clear()
        assert await manager.storage.count() == 0


# ===================================================================
# Edge cases and additional coverage
# ===================================================================

class TestRedisSearchByThread:
    """Test search_messages with thread_id (calls get_messages which is abstract)."""

    async def test_search_by_thread_calls_get_messages(self):
        """RedisConversationMemory.search_messages with thread_id calls self.get_messages."""
        mock_redis = AsyncMock()
        rmem = _TestableRedisConversationMemory(redis_client=mock_redis)

        # Since get_messages is abstract and not implemented on Redis,
        # we need to mock it
        msg = Message(
            message_id="m1", thread_id="t1", content="hello world",
            role=MessageRole.USER, user_id="u1",
        )
        rmem.get_messages = AsyncMock(return_value=[msg])

        results = await rmem.search_messages("hello", thread_id="t1")
        assert len(results) == 1
        rmem.get_messages.assert_called_once()

    async def test_search_by_thread_filters_user(self):
        mock_redis = AsyncMock()
        rmem = _TestableRedisConversationMemory(redis_client=mock_redis)
        msg1 = Message(message_id="m1", thread_id="t1", content="hello", role=MessageRole.USER, user_id="u1")
        msg2 = Message(message_id="m2", thread_id="t1", content="hello", role=MessageRole.USER, user_id="u2")
        rmem.get_messages = AsyncMock(return_value=[msg1, msg2])

        results = await rmem.search_messages("hello", thread_id="t1", user_id="u1")
        assert len(results) == 1
        assert results[0].user_id == "u1"

    async def test_search_by_thread_respects_limit(self):
        mock_redis = AsyncMock()
        rmem = _TestableRedisConversationMemory(redis_client=mock_redis)
        msgs = [
            Message(message_id=f"m{i}", thread_id="t1", content="hello", role=MessageRole.USER)
            for i in range(20)
        ]
        rmem.get_messages = AsyncMock(return_value=msgs)

        results = await rmem.search_messages("hello", thread_id="t1", limit=3)
        assert len(results) == 3

    async def test_search_by_user_respects_limit(self):
        mock_redis = AsyncMock()
        rmem = _TestableRedisConversationMemory(redis_client=mock_redis)
        msgs = [
            Message(message_id=f"m{i}", thread_id="t1", content="findme", role=MessageRole.USER, user_id="u1")
            for i in range(20)
        ]
        rmem.get_user_messages = AsyncMock(return_value=msgs)

        results = await rmem.search_messages("findme", user_id="u1", limit=5)
        assert len(results) == 5


class TestRedisGetUserMessagesMultiScan:
    """Test get_user_messages when scan requires multiple iterations."""

    async def test_multi_scan_iterations(self):
        mock_redis = AsyncMock()
        rmem = _TestableRedisConversationMemory(redis_client=mock_redis)

        mock_redis.zrevrange.return_value = [b"msg-1"]
        # First scan returns cursor > 0 (more to scan), second returns cursor 0
        mock_redis.scan.side_effect = [
            (42, []),  # First iteration: no keys found, cursor not 0
            (0, [b"user:u1:thread:t1"]),  # Second iteration: found the thread
        ]
        mock_redis.sismember.return_value = True
        msg_data = {
            "message_id": "msg-1",
            "thread_id": "t1",
            "role": "user",
            "content": "multi-scan",
            "metadata": {},
            "timestamp": datetime.now().isoformat(),
        }
        mock_redis.get.return_value = json.dumps(msg_data).encode("utf-8")

        results = await rmem.get_user_messages("u1")
        assert len(results) == 1
        assert mock_redis.scan.call_count == 2


class TestSearchMessagesGlobalLimit:
    """Test that global search respects the limit parameter."""

    async def test_global_search_limit(self):
        mock_redis = AsyncMock()
        rmem = _TestableRedisConversationMemory(redis_client=mock_redis)

        # Return many message keys
        keys = [f"message:t1:msg-{i}".encode() for i in range(20)]
        mock_redis.scan.return_value = (0, keys)

        msg_template = {
            "message_id": "msg-{i}",
            "thread_id": "t1",
            "role": "user",
            "content": "match",
            "metadata": {},
            "timestamp": datetime.now().isoformat(),
        }

        async def get_side_effect(key):
            return json.dumps({**msg_template, "message_id": key}).encode("utf-8")

        mock_redis.get.side_effect = get_side_effect

        results = await rmem.search_messages("match", limit=3)
        assert len(results) == 3
