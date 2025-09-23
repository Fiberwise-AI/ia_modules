"""
IA Modules Authentication Package
"""

from .middleware import AuthMiddleware, init_auth, get_current_user, login_user
from .session import SessionManager
from .models import CurrentUser
from .security import generate_session_token, create_session_cookie

__all__ = ['AuthMiddleware', 'init_auth', 'get_current_user', 'login_user', 'SessionManager', 'CurrentUser', 'UserRole', 'generate_session_token', 'create_session_cookie']
