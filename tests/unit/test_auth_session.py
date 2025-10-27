"""
Tests for auth.session module
"""

import pytest
from ia_modules.pipeline.test_utils import create_test_execution_context
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from ia_modules.auth.session import SessionManager


class QueryResultSimple:
    """Simple query result for testing"""
    def __init__(self, success: bool, data=None, row_count=0):
        self.success = success
        self.data = data
        self.row_count = row_count


class MockDBProvider:
    """Mock database provider for testing"""

    def __init__(self):
        self.execute_calls = []
        self.fetch_calls = []
        self.execute_results = []
        self.fetch_results = []

    async def execute_async(self, query, params=None):
        self.execute_calls.append((query, params))
        if self.execute_results:
            return self.execute_results.pop(0)
        return QueryResultSimple(success=True, row_count=1)

    async def fetch_one(self, query, params=None):
        self.fetch_calls.append((query, params))
        if self.fetch_results:
            return self.fetch_results.pop(0)
        return QueryResultSimple(success=False, data=None)

    def add_execute_result(self, success=True, row_count=1):
        self.execute_results.append(QueryResultSimple(success=success, row_count=row_count))

    def add_fetch_result(self, success=True, data=None):
        self.fetch_results.append(QueryResultSimple(success=success, data=data))


@pytest.mark.asyncio
class TestSessionManager:
    """Test SessionManager class"""

    async def test_init(self):
        """Test SessionManager initialization"""
        db = MockDBProvider()
        manager = SessionManager(db)

        assert manager.db_provider is db

    async def test_generate_session_token(self):
        """Test session token generation"""
        token = SessionManager.generate_session_token()

        assert isinstance(token, str)
        assert len(token) > 0

    async def test_generate_session_token_unique(self):
        """Test that session tokens are unique"""
        tokens = [SessionManager.generate_session_token() for _ in range(100)]
        assert len(set(tokens)) == 100

    async def test_create_session_success(self):
        """Test successful session creation"""
        db = MockDBProvider()
        db.add_execute_result(success=True, row_count=1)
        manager = SessionManager(db)

        session_token = await manager.create_session(user_id=1, expires_in_hours=24)

        assert session_token is not None
        assert isinstance(session_token, str)
        assert len(db.execute_calls) == 1

        query, params = db.execute_calls[0]
        assert "INSERT INTO user_sessions" in query
        assert params['user_id'] == 1  # user_id as dict key

    async def test_create_session_custom_expiry(self):
        """Test session creation with custom expiry"""
        db = MockDBProvider()
        db.add_execute_result(success=True, row_count=1)
        manager = SessionManager(db)

        session_token = await manager.create_session(user_id=1, expires_in_hours=48)

        assert session_token is not None
        query, params = db.execute_calls[0]
        assert params['user_id'] == 1  # user_id as dict key
        assert 'expires_at' in params  # expiry datetime included

    async def test_create_session_failure(self):
        """Test session creation failure"""
        db = MockDBProvider()
        db.add_execute_result(success=False, row_count=0)
        manager = SessionManager(db)

        session_token = await manager.create_session(user_id=1, expires_in_hours=24)

        assert session_token is None

    async def test_create_session_no_db_provider(self):
        """Test session creation without db provider"""
        manager = SessionManager(None)

        session_token = await manager.create_session(user_id=1, expires_in_hours=24)

        assert session_token is None

    async def test_create_session_exception(self):
        """Test session creation with exception"""
        db = MockDBProvider()

        async def raise_exception(*args, **kwargs):
            raise Exception("Database error")

        db.execute_async = raise_exception
        manager = SessionManager(db)

        session_token = await manager.create_session(user_id=1, expires_in_hours=24)

        assert session_token is None

    async def test_validate_session_success(self):
        """Test successful session validation"""
        db = MockDBProvider()
        db.add_fetch_result(success=True, data=[{"user_id": 1}])
        manager = SessionManager(db)

        user_id = await manager.validate_session("valid_token_123")

        assert user_id == 1
        assert len(db.fetch_calls) == 1

        query, params = db.fetch_calls[0]
        assert "SELECT user_id FROM user_sessions" in query
        assert params['session_token'] == "valid_token_123"

    async def test_validate_session_expired(self):
        """Test validation of expired session"""
        db = MockDBProvider()
        db.add_fetch_result(success=True, data=[])  # No data = expired
        manager = SessionManager(db)

        user_id = await manager.validate_session("expired_token_123")

        assert user_id is None

    async def test_validate_session_invalid(self):
        """Test validation of invalid session"""
        db = MockDBProvider()
        db.add_fetch_result(success=False, data=None)
        manager = SessionManager(db)

        user_id = await manager.validate_session("invalid_token_123")

        assert user_id is None

    async def test_validate_session_no_db_provider(self):
        """Test session validation without db provider"""
        manager = SessionManager(None)

        user_id = await manager.validate_session("token_123")

        assert user_id is None

    async def test_validate_session_exception(self):
        """Test session validation with exception"""
        db = MockDBProvider()

        async def raise_exception(*args, **kwargs):
            raise Exception("Database error")

        db.fetch_one = raise_exception
        manager = SessionManager(db)

        user_id = await manager.validate_session("token_123")

        assert user_id is None

    async def test_destroy_session_success(self):
        """Test successful session destruction"""
        db = MockDBProvider()
        db.add_execute_result(success=True, row_count=1)
        manager = SessionManager(db)

        result = await manager.destroy_session("token_123")

        assert result is True
        assert len(db.execute_calls) == 1

        query, params = db.execute_calls[0]
        assert "DELETE FROM user_sessions WHERE session_token = :session_token" in query
        assert isinstance(params, dict)
        assert params['session_token'] == "token_123"

    async def test_destroy_session_failure(self):
        """Test session destruction failure"""
        db = MockDBProvider()
        db.add_execute_result(success=False, row_count=0)
        manager = SessionManager(db)

        result = await manager.destroy_session("token_123")

        assert result is False

    async def test_destroy_session_no_db_provider(self):
        """Test session destruction without db provider"""
        manager = SessionManager(None)

        result = await manager.destroy_session("token_123")

        assert result is False

    async def test_destroy_session_exception(self):
        """Test session destruction with exception"""
        db = MockDBProvider()

        async def raise_exception(*args, **kwargs):
            raise Exception("Database error")

        db.execute_async = raise_exception
        manager = SessionManager(db)

        result = await manager.destroy_session("token_123")

        assert result is False

    async def test_delete_user_sessions_success(self):
        """Test successful deletion of all user sessions"""
        db = MockDBProvider()
        db.add_execute_result(success=True, row_count=3)
        manager = SessionManager(db)

        result = await manager.delete_user_sessions(user_id=1)

        assert result is True
        assert len(db.execute_calls) == 1

        query, params = db.execute_calls[0]
        assert "DELETE FROM user_sessions WHERE user_id = :user_id" in query
        assert isinstance(params, dict)
        assert params['user_id'] == 1

    async def test_delete_user_sessions_failure(self):
        """Test deletion of user sessions failure"""
        db = MockDBProvider()
        db.add_execute_result(success=False, row_count=0)
        manager = SessionManager(db)

        result = await manager.delete_user_sessions(user_id=1)

        assert result is False

    async def test_delete_user_sessions_no_db_provider(self):
        """Test deletion of user sessions without db provider"""
        manager = SessionManager(None)

        result = await manager.delete_user_sessions(user_id=1)

        assert result is False

    async def test_delete_user_sessions_exception(self):
        """Test deletion of user sessions with exception"""
        db = MockDBProvider()

        async def raise_exception(*args, **kwargs):
            raise Exception("Database error")

        db.execute_async = raise_exception
        manager = SessionManager(db)

        result = await manager.delete_user_sessions(user_id=1)

        assert result is False

    async def test_cleanup_expired_sessions_success(self):
        """Test successful cleanup of expired sessions"""
        db = MockDBProvider()
        db.add_execute_result(success=True, row_count=5)
        manager = SessionManager(db)

        count = await manager.cleanup_expired_sessions()

        assert count == 5
        assert len(db.execute_calls) == 1

        query, params = db.execute_calls[0]
        assert "DELETE FROM user_sessions WHERE expires_at <= datetime('now')" in query

    async def test_cleanup_expired_sessions_none_expired(self):
        """Test cleanup when no sessions are expired"""
        db = MockDBProvider()
        db.add_execute_result(success=True, row_count=0)
        manager = SessionManager(db)

        count = await manager.cleanup_expired_sessions()

        assert count == 0

    async def test_cleanup_expired_sessions_failure(self):
        """Test cleanup of expired sessions failure"""
        db = MockDBProvider()
        db.add_execute_result(success=False, row_count=0)
        manager = SessionManager(db)

        count = await manager.cleanup_expired_sessions()

        assert count == 0

    async def test_cleanup_expired_sessions_no_db_provider(self):
        """Test cleanup without db provider"""
        manager = SessionManager(None)

        count = await manager.cleanup_expired_sessions()

        assert count == 0

    async def test_cleanup_expired_sessions_exception(self):
        """Test cleanup with exception"""
        db = MockDBProvider()

        async def raise_exception(*args, **kwargs):
            raise Exception("Database error")

        db.execute_async = raise_exception
        manager = SessionManager(db)

        count = await manager.cleanup_expired_sessions()

        assert count == 0

    async def test_get_user_by_email_success(self):
        """Test successful get user by email"""
        db = MockDBProvider()
        user_data = {
            "id": 1,
            "email": "test@example.com",
            "password_hash": "hash",
            "first_name": "John",
            "last_name": "Doe"
        }
        db.add_fetch_result(success=True, data=[user_data])
        manager = SessionManager(db)

        user = await manager.get_user_by_email("test@example.com")

        assert user == user_data
        assert len(db.fetch_calls) == 1

        query, params = db.fetch_calls[0]
        assert "SELECT * FROM users WHERE email = :email" in query
        assert isinstance(params, dict)
        assert params['email'] == "test@example.com"

    async def test_get_user_by_email_not_found(self):
        """Test get user by email when not found"""
        db = MockDBProvider()
        db.add_fetch_result(success=True, data=[])
        manager = SessionManager(db)

        user = await manager.get_user_by_email("notfound@example.com")

        assert user is None

    async def test_get_user_by_email_failure(self):
        """Test get user by email failure"""
        db = MockDBProvider()
        db.add_fetch_result(success=False, data=None)
        manager = SessionManager(db)

        user = await manager.get_user_by_email("test@example.com")

        assert user is None

    async def test_get_user_by_email_no_db_provider(self):
        """Test get user without db provider"""
        manager = SessionManager(None)

        user = await manager.get_user_by_email("test@example.com")

        assert user is None

    async def test_get_user_by_email_exception(self):
        """Test get user with exception"""
        db = MockDBProvider()

        async def raise_exception(*args, **kwargs):
            raise Exception("Database error")

        db.fetch_one = raise_exception
        manager = SessionManager(db)

        user = await manager.get_user_by_email("test@example.com")

        assert user is None

    async def test_create_user_success(self):
        """Test successful user creation"""
        db = MockDBProvider()
        db.add_execute_result(success=True, row_count=1)
        db.add_fetch_result(success=True, data=[{"id": 1}])
        manager = SessionManager(db)

        user_id = await manager.create_user(
            email="new@example.com",
            password="password123",
            first_name="Jane",
            last_name="Smith",
            role="user"
        )

        assert user_id == 1
        assert len(db.execute_calls) == 1
        assert len(db.fetch_calls) == 1

        query, params = db.execute_calls[0]
        assert "INSERT INTO users" in query
        assert params['email'] == "new@example.com"
        assert params['first_name'] == "Jane"
        assert params['last_name'] == "Smith"
        assert params['role'] == "user"

    async def test_create_user_default_role(self):
        """Test user creation with default role"""
        db = MockDBProvider()
        db.add_execute_result(success=True, row_count=1)
        db.add_fetch_result(success=True, data=[{"id": 1}])
        manager = SessionManager(db)

        user_id = await manager.create_user(
            email="new@example.com",
            password="password123",
            first_name="Jane",
            last_name="Smith"
        )

        assert user_id == 1

        query, params = db.execute_calls[0]
        assert params['role'] == "user"  # default role

    async def test_create_user_failure(self):
        """Test user creation failure"""
        db = MockDBProvider()
        db.add_execute_result(success=False, row_count=0)
        manager = SessionManager(db)

        user_id = await manager.create_user(
            email="new@example.com",
            password="password123",
            first_name="Jane",
            last_name="Smith"
        )

        assert user_id is None

    async def test_create_user_no_db_provider(self):
        """Test user creation without db provider"""
        manager = SessionManager(None)

        user_id = await manager.create_user(
            email="new@example.com",
            password="password123",
            first_name="Jane",
            last_name="Smith"
        )

        assert user_id is None

    async def test_create_user_exception(self):
        """Test user creation with exception"""
        db = MockDBProvider()

        async def raise_exception(*args, **kwargs):
            raise Exception("Database error")

        db.execute_async = raise_exception
        manager = SessionManager(db)

        user_id = await manager.create_user(
            email="new@example.com",
            password="password123",
            first_name="Jane",
            last_name="Smith"
        )

        assert user_id is None
