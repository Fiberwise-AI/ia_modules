"""
Authentication middleware for IA Modules
"""

from fastapi import Request, HTTPException
from fastapi.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import List, Optional, Callable
from .models import CurrentUser
from .session import SessionManager
from .security import verify_password
import logging

logger = logging.getLogger(__name__)

# Global database provider and session manager
_db_provider = None
_session_manager = None


def init_auth(db_provider):
    """Initialize authentication system with database provider"""
    global _db_provider, _session_manager
    _db_provider = db_provider
    _session_manager = SessionManager(db_provider)
    logger.info("IA Modules authentication system initialized")


def get_session_manager() -> SessionManager:
    """Get the session manager instance"""
    if not _session_manager:
        raise RuntimeError("Authentication system not initialized")
    return _session_manager


class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication middleware to protect routes"""
    
    def __init__(self, app, excluded_paths: List[str] = None):
        super().__init__(app)
        self.excluded_paths = excluded_paths or []
        
    async def dispatch(self, request: Request, call_next):
        # Check if path should be excluded from authentication
        path = request.url.path
        
        # Skip authentication for excluded paths
        for excluded_path in self.excluded_paths:
            if path.startswith(excluded_path):
                return await call_next(request)
        
        # Skip authentication for static files and health checks
        if any(path.startswith(p) for p in ["/static", "/docs", "/redoc", "/openapi.json", "/health"]):
            return await call_next(request)
        
        # Get current user
        current_user = await get_current_user(request)
        
        # If no user and this is an API endpoint, return 401
        if not current_user and path.startswith("/api"):
            raise HTTPException(status_code=401, detail="Authentication required")
        
        # If no user and this is a protected web page, redirect to login
        if not current_user and not path.startswith("/auth"):
            return RedirectResponse(url="/auth/login", status_code=302)
        
        # User is authenticated, continue with request
        return await call_next(request)


async def get_current_user(request: Request) -> Optional[CurrentUser]:
    """Get current authenticated user from session"""
    if not _session_manager:
        return None
    
    # Try session token from cookies
    session_token = request.cookies.get("session_id")
    if not session_token:
        return None
    
    try:
        # Validate session and get user ID
        user_id = await _session_manager.validate_session(session_token)
        if not user_id:
            return None
        
        # Get user details
        query = "SELECT * FROM users WHERE id = :user_id AND active = 1"
        result = await _db_provider.fetch_one(query, {'user_id': user_id})
        
        if not result.success or not result.data:
            return None
        
        user_data = result.data[0]
        return CurrentUser(
            id=user_data['id'],
            email=user_data['email'],
            first_name=user_data.get('first_name'),
            last_name=user_data.get('last_name'),
            role=user_data.get('role', 'user'),
            facility_id=user_data.get('facility_id'),
            active=user_data.get('active', True),
            username=user_data.get('username') or user_data['email']
        )
    except Exception as e:
        logger.error(f"Error getting current user: {e}")
        return None


async def login_user(email: str, password: str, remember_me: bool = False) -> tuple[Optional[CurrentUser], Optional[str]]:
    """Login user with email and password, return user and session ID"""
    if not _session_manager:
        return None, None
    
    try:
        # Get user by email
        query = "SELECT * FROM users WHERE email = :email AND active = 1"
        result = await _db_provider.fetch_one(query, {'email': email})
        
        if not result.success or not result.data:
            return None, None
        
        user_data = result.data[0]
        
        # Verify password
        if not verify_password(password, user_data['password_hash']):
            return None, None
        
        # Create session
        expires_hours = 24 * 30 if remember_me else 24  # 30 days vs 24 hours
        session_token = await _session_manager.create_session(user_data['id'], expires_hours)
        
        if not session_token:
            return None, None
        
        user = CurrentUser(
            id=user_data['id'],
            email=user_data['email'],
            first_name=user_data.get('first_name'),
            last_name=user_data.get('last_name'),
            role=user_data.get('role', 'user'),
            facility_id=user_data.get('facility_id'),
            active=user_data.get('active', True),
            username=user_data.get('username') or user_data['email']
        )
        
        return user, session_token
        
    except Exception as e:
        logger.error(f"Error during login: {e}")
        return None, None


def require_auth(func: Callable) -> Callable:
    """Decorator to require authentication for a function"""
    async def wrapper(*args, **kwargs):
        # Extract request from args or kwargs
        request = None
        for arg in args:
            if isinstance(arg, Request):
                request = arg
                break
        
        if not request:
            raise HTTPException(status_code=500, detail="Request object not found")
        
        current_user = await get_current_user(request)
        if not current_user:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        return await func(*args, **kwargs)
    
    return wrapper
