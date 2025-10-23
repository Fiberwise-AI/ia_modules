"""
IA Modules Authentication Package

Optional authentication utilities for FastAPI applications.
Requires: pip install ia_modules[web]
"""

# Import only if fastapi is available
try:
    from .middleware import AuthMiddleware, init_auth, get_current_user, login_user
    from .session import SessionManager
    from .models import CurrentUser
    from .security import generate_session_token, create_session_cookie

    __all__ = [
        'AuthMiddleware',
        'init_auth',
        'get_current_user',
        'login_user',
        'SessionManager',
        'CurrentUser',
        'generate_session_token',
        'create_session_cookie'
    ]
except ImportError:
    # FastAPI not installed - auth module not available
    __all__ = []
