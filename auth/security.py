"""
Security utilities for IA Modules authentication
"""

import hashlib
import secrets
from typing import Dict, Any


def hash_password(password: str) -> str:
    """Hash a password using SHA-256 with salt"""
    salt = secrets.token_hex(16)
    password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return f"{salt}:{password_hash}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against stored hash"""
    try:
        salt, stored_password_hash = stored_hash.split(":", 1)
        password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return password_hash == stored_password_hash
    except ValueError:
        return False


def generate_session_token() -> str:
    """Generate a secure session token"""
    return secrets.token_urlsafe(32)


def create_session_cookie(session_token: str, remember_me: bool = False) -> Dict[str, Any]:
    """Create session cookie parameters"""
    cookie_params = {
        "key": "session_id",
        "value": session_token,
        "httponly": True,
        "secure": False,  # Set to True for HTTPS in production
        "samesite": "lax"
    }
    
    if remember_me:
        # Remember for 30 days
        cookie_params["max_age"] = 30 * 24 * 60 * 60
    else:
        # Session cookie (expires when browser closes)
        cookie_params["max_age"] = None
    
    return cookie_params
