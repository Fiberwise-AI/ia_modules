"""
Tests for auth.middleware module
"""

import pytest
from ia_modules.pipeline.test_utils import create_test_execution_context
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import Request, HTTPException
from fastapi.responses import RedirectResponse
from ia_modules.auth.middleware import (
    init_auth,
    get_session_manager,
    AuthMiddleware,
    get_current_user,
    login_user,
    require_auth
)
from ia_modules.auth.models import CurrentUser
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
        self.fetch_results = []

    async def fetch_one(self, query, params=None):
        if self.fetch_results:
            return self.fetch_results.pop(0)
        return QueryResultSimple(success=False, data=None)

    async def execute_async(self, query, params=None):
        return QueryResultSimple(success=True, row_count=1)

    def add_fetch_result(self, success=True, data=None):
        self.fetch_results.append(QueryResultSimple(success=success, data=data))


def create_mock_request(path="/", cookies=None):
    """Create a mock request object that passes isinstance(obj, Request)"""
    # Create a dynamic class that inherits from Request
    class MockRequest(Request):
        def __init__(self, path_val, cookies_dict):
            # Don't call super().__init__ - just set needed attributes
            self._path = path_val
            self._cookies_dict = cookies_dict or {}
            # Add scope for __len__ method
            self.scope = {"type": "http", "path": path_val}

        @property
        def url(self):
            url_mock = MagicMock()
            url_mock.path = self._path
            return url_mock

        @property
        def cookies(self):
            cookies_mock = MagicMock()
            cookies_mock.get = lambda key, default=None: self._cookies_dict.get(key, default)
            return cookies_mock

    return MockRequest(path, cookies)


@pytest.mark.asyncio
class TestAuthInit:
    """Test auth initialization"""

    async def test_init_auth(self):
        """Test auth initialization"""
        db = MockDBProvider()

        init_auth(db)

        manager = get_session_manager()
        assert isinstance(manager, SessionManager)
        assert manager.db_provider is db

    async def test_get_session_manager_not_initialized(self):
        """Test get_session_manager raises error when not initialized"""
        import ia_modules.auth.middleware as middleware_module

        # Save original value
        original_manager = middleware_module._session_manager

        try:
            # Clear the manager
            middleware_module._session_manager = None

            with pytest.raises(RuntimeError, match="Authentication system not initialized"):
                get_session_manager()
        finally:
            # Restore original value
            middleware_module._session_manager = original_manager


@pytest.mark.asyncio
class TestAuthMiddleware:
    """Test AuthMiddleware class"""

    async def test_middleware_init(self):
        """Test middleware initialization"""
        app = MagicMock()
        middleware = AuthMiddleware(app)

        assert middleware.excluded_paths == []

    async def test_middleware_init_with_excluded_paths(self):
        """Test middleware initialization with excluded paths"""
        app = MagicMock()
        excluded = ["/public", "/api/public"]
        middleware = AuthMiddleware(app, excluded_paths=excluded)

        assert middleware.excluded_paths == excluded

    async def test_middleware_dispatch_excluded_path(self):
        """Test middleware allows excluded paths"""
        app = MagicMock()
        middleware = AuthMiddleware(app, excluded_paths=["/public"])

        request = create_mock_request(path="/public/page")
        call_next = AsyncMock(return_value="response")

        response = await middleware.dispatch(request, call_next)

        assert response == "response"
        call_next.assert_called_once_with(request)

    async def test_middleware_dispatch_static_files(self):
        """Test middleware allows static files"""
        app = MagicMock()
        middleware = AuthMiddleware(app)

        static_paths = ["/static/style.css", "/docs", "/redoc", "/openapi.json", "/health"]

        for path in static_paths:
            request = create_mock_request(path=path)
            call_next = AsyncMock(return_value="response")

            response = await middleware.dispatch(request, call_next)

            assert response == "response"
            call_next.assert_called_once_with(request)

    async def test_middleware_dispatch_api_no_auth(self):
        """Test middleware returns 401 for unauthenticated API requests"""
        db = MockDBProvider()
        init_auth(db)

        app = MagicMock()
        middleware = AuthMiddleware(app)

        request = create_mock_request(path="/api/data")
        call_next = AsyncMock(return_value="response")

        with pytest.raises(HTTPException) as exc_info:
            await middleware.dispatch(request, call_next)

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Authentication required"

    async def test_middleware_dispatch_web_no_auth(self):
        """Test middleware redirects to login for unauthenticated web requests"""
        db = MockDBProvider()
        init_auth(db)

        app = MagicMock()
        middleware = AuthMiddleware(app)

        request = create_mock_request(path="/dashboard")
        call_next = AsyncMock(return_value="response")

        response = await middleware.dispatch(request, call_next)

        assert isinstance(response, RedirectResponse)
        assert response.headers["location"] == "/auth/login"
        assert response.status_code == 302

    async def test_middleware_dispatch_auth_path_no_redirect(self):
        """Test middleware doesn't redirect auth paths"""
        db = MockDBProvider()
        init_auth(db)

        app = MagicMock()
        middleware = AuthMiddleware(app)

        request = create_mock_request(path="/auth/login")
        call_next = AsyncMock(return_value="response")

        response = await middleware.dispatch(request, call_next)

        assert response == "response"
        call_next.assert_called_once_with(request)

    async def test_middleware_dispatch_authenticated_user(self):
        """Test middleware allows authenticated users"""
        db = MockDBProvider()
        user_data = {
            "id": 1,
            "email": "test@example.com",
            "password_hash": "hash",
            "first_name": "John",
            "last_name": "Doe",
            "role": "user",
            "facility_id": None,
            "active": 1,
            "username": "johndoe"
        }
        db.add_fetch_result(success=True, data=[{"user_id": 1}])  # Session validation
        db.add_fetch_result(success=True, data=[user_data])  # User lookup

        init_auth(db)

        app = MagicMock()
        middleware = AuthMiddleware(app)

        request = create_mock_request(path="/dashboard", cookies={"session_id": "valid_token"})
        call_next = AsyncMock(return_value="response")

        response = await middleware.dispatch(request, call_next)

        assert response == "response"
        call_next.assert_called_once_with(request)


@pytest.mark.asyncio
class TestGetCurrentUser:
    """Test get_current_user function"""

    async def test_get_current_user_no_session_manager(self):
        """Test get_current_user returns None when session manager not initialized"""
        import ia_modules.auth.middleware as middleware_module

        original_manager = middleware_module._session_manager

        try:
            middleware_module._session_manager = None

            request = create_mock_request(cookies={"session_id": "token"})
            user = await get_current_user(request)

            assert user is None
        finally:
            middleware_module._session_manager = original_manager

    async def test_get_current_user_no_token(self):
        """Test get_current_user returns None when no token in cookies"""
        db = MockDBProvider()
        init_auth(db)

        request = create_mock_request()
        user = await get_current_user(request)

        assert user is None

    async def test_get_current_user_invalid_session(self):
        """Test get_current_user returns None for invalid session"""
        db = MockDBProvider()
        db.add_fetch_result(success=False, data=None)
        init_auth(db)

        request = create_mock_request(cookies={"session_id": "invalid_token"})
        user = await get_current_user(request)

        assert user is None

    async def test_get_current_user_valid_session(self):
        """Test get_current_user returns user for valid session"""
        db = MockDBProvider()
        user_data = {
            "id": 1,
            "email": "test@example.com",
            "password_hash": "hash",
            "first_name": "John",
            "last_name": "Doe",
            "role": "user",
            "facility_id": None,
            "active": 1,
            "username": "johndoe"
        }
        db.add_fetch_result(success=True, data=[{"user_id": 1}])  # Session validation
        db.add_fetch_result(success=True, data=[user_data])  # User lookup

        init_auth(db)

        request = create_mock_request(cookies={"session_id": "valid_token"})
        user = await get_current_user(request)

        assert isinstance(user, CurrentUser)
        assert user.id == 1
        assert user.email == "test@example.com"
        assert user.first_name == "John"
        assert user.last_name == "Doe"
        assert user.role == "user"

    async def test_get_current_user_inactive_user(self):
        """Test get_current_user returns None for inactive user"""
        db = MockDBProvider()
        db.add_fetch_result(success=True, data=[{"user_id": 1}])  # Session validation
        db.add_fetch_result(success=True, data=[])  # User lookup returns empty (filtered by SQL)

        init_auth(db)

        request = create_mock_request(cookies={"session_id": "valid_token"})
        user = await get_current_user(request)

        # The SQL query filters for active = 1, so inactive users return no data
        assert user is None

    async def test_get_current_user_exception(self):
        """Test get_current_user handles exceptions gracefully"""
        db = MockDBProvider()

        async def raise_exception(*args, **kwargs):
            raise Exception("Database error")

        db.fetch_one = raise_exception
        init_auth(db)

        request = create_mock_request(cookies={"session_id": "valid_token"})
        user = await get_current_user(request)

        assert user is None


@pytest.mark.asyncio
class TestLoginUser:
    """Test login_user function"""

    async def test_login_user_no_session_manager(self):
        """Test login_user returns None when session manager not initialized"""
        import ia_modules.auth.middleware as middleware_module

        original_manager = middleware_module._session_manager

        try:
            middleware_module._session_manager = None

            user, token = await login_user("test@example.com", "password123")

            assert user is None
            assert token is None
        finally:
            middleware_module._session_manager = original_manager

    async def test_login_user_user_not_found(self):
        """Test login_user returns None when user not found"""
        db = MockDBProvider()
        db.add_fetch_result(success=False, data=None)
        init_auth(db)

        user, token = await login_user("notfound@example.com", "password123")

        assert user is None
        assert token is None

    async def test_login_user_wrong_password(self):
        """Test login_user returns None with wrong password"""
        db = MockDBProvider()

        from ia_modules.auth.security import hash_password
        password_hash = hash_password("correctpassword")

        user_data = {
            "id": 1,
            "email": "test@example.com",
            "password_hash": password_hash,
            "first_name": "John",
            "last_name": "Doe",
            "role": "user",
            "facility_id": None,
            "active": 1,
            "username": "johndoe"
        }
        db.add_fetch_result(success=True, data=[user_data])
        init_auth(db)

        user, token = await login_user("test@example.com", "wrongpassword")

        assert user is None
        assert token is None

    async def test_login_user_success(self):
        """Test successful user login"""
        db = MockDBProvider()

        from ia_modules.auth.security import hash_password
        password_hash = hash_password("password123")

        user_data = {
            "id": 1,
            "email": "test@example.com",
            "password_hash": password_hash,
            "first_name": "John",
            "last_name": "Doe",
            "role": "user",
            "facility_id": None,
            "active": 1,
            "username": "johndoe"
        }
        db.add_fetch_result(success=True, data=[user_data])  # User lookup
        init_auth(db)

        user, token = await login_user("test@example.com", "password123")

        assert isinstance(user, CurrentUser)
        assert user.id == 1
        assert user.email == "test@example.com"
        assert isinstance(token, str)
        assert len(token) > 0

    async def test_login_user_remember_me(self):
        """Test user login with remember_me flag"""
        db = MockDBProvider()

        from ia_modules.auth.security import hash_password
        password_hash = hash_password("password123")

        user_data = {
            "id": 1,
            "email": "test@example.com",
            "password_hash": password_hash,
            "first_name": "John",
            "last_name": "Doe",
            "role": "user",
            "facility_id": None,
            "active": 1,
            "username": "johndoe"
        }
        db.add_fetch_result(success=True, data=[user_data])
        init_auth(db)

        user, token = await login_user("test@example.com", "password123", remember_me=True)

        assert isinstance(user, CurrentUser)
        assert isinstance(token, str)

    async def test_login_user_session_creation_failure(self):
        """Test login_user when session creation fails"""
        db = MockDBProvider()

        from ia_modules.auth.security import hash_password
        password_hash = hash_password("password123")

        user_data = {
            "id": 1,
            "email": "test@example.com",
            "password_hash": password_hash,
            "first_name": "John",
            "last_name": "Doe",
            "role": "user",
            "facility_id": None,
            "active": 1,
            "username": "johndoe"
        }
        db.add_fetch_result(success=True, data=[user_data])

        async def execute_fail(*args, **kwargs):
            return DatabaseResult(success=False, row_count=0)

        db.execute_async = execute_fail
        init_auth(db)

        user, token = await login_user("test@example.com", "password123")

        assert user is None
        assert token is None

    async def test_login_user_exception(self):
        """Test login_user handles exceptions gracefully"""
        db = MockDBProvider()

        async def raise_exception(*args, **kwargs):
            raise Exception("Database error")

        db.fetch_one = raise_exception
        init_auth(db)

        user, token = await login_user("test@example.com", "password123")

        assert user is None
        assert token is None


@pytest.mark.asyncio
class TestRequireAuth:
    """Test require_auth decorator"""

    async def test_require_auth_with_authenticated_user(self):
        """Test require_auth allows authenticated users"""
        db = MockDBProvider()
        user_data = {
            "id": 1,
            "email": "test@example.com",
            "password_hash": "hash",
            "first_name": "John",
            "last_name": "Doe",
            "role": "user",
            "facility_id": None,
            "active": 1,
            "username": "johndoe"
        }
        db.add_fetch_result(success=True, data=[{"user_id": 1}])  # Session validation
        db.add_fetch_result(success=True, data=[user_data])  # User lookup

        init_auth(db)

        @require_auth
        async def protected_function(request: Request):
            return "success"

        request = create_mock_request(cookies={"session_id": "valid_token"})
        result = await protected_function(request)

        assert result == "success"

    async def test_require_auth_without_authenticated_user(self):
        """Test require_auth raises HTTPException for unauthenticated users"""
        db = MockDBProvider()
        init_auth(db)

        @require_auth
        async def protected_function(request: Request):
            return "success"

        request = create_mock_request()

        with pytest.raises(HTTPException) as exc_info:
            await protected_function(request)

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Authentication required"

    async def test_require_auth_no_request_in_args(self):
        """Test require_auth raises error when no Request object found"""
        db = MockDBProvider()
        init_auth(db)

        @require_auth
        async def protected_function():
            return "success"

        with pytest.raises(HTTPException) as exc_info:
            await protected_function()

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Request object not found"

    async def test_require_auth_with_kwargs(self):
        """Test require_auth works with kwargs"""
        db = MockDBProvider()
        user_data = {
            "id": 1,
            "email": "test@example.com",
            "password_hash": "hash",
            "first_name": "John",
            "last_name": "Doe",
            "role": "user",
            "facility_id": None,
            "active": 1,
            "username": "johndoe"
        }
        db.add_fetch_result(success=True, data=[{"user_id": 1}])
        db.add_fetch_result(success=True, data=[user_data])

        init_auth(db)

        @require_auth
        async def protected_function(request: Request, data: str = "default"):
            return f"success-{data}"

        request = create_mock_request(cookies={"session_id": "valid_token"})
        result = await protected_function(request, data="custom")

        assert result == "success-custom"
