"""
Tests for auth.security module
"""

import pytest
from ia_modules.pipeline.test_utils import create_test_execution_context
from ia_modules.auth.security import (
    hash_password,
    verify_password,
    generate_session_token,
    create_session_cookie
)


class TestPasswordHashing:
    """Test password hashing functionality"""

    def test_hash_password_creates_hash_with_salt(self):
        """Test that hash_password creates a hash with salt"""
        password = "testpassword123"
        hashed = hash_password(password)

        assert ":" in hashed
        parts = hashed.split(":")
        assert len(parts) == 2
        assert len(parts[0]) == 32  # Salt should be 16 bytes hex = 32 chars
        assert len(parts[1]) == 64  # SHA-256 hash = 64 hex chars

    def test_hash_password_different_salts(self):
        """Test that same password gets different hashes due to different salts"""
        password = "testpassword123"
        hash1 = hash_password(password)
        hash2 = hash_password(password)

        assert hash1 != hash2

    def test_verify_password_correct(self):
        """Test password verification with correct password"""
        password = "testpassword123"
        hashed = hash_password(password)

        assert verify_password(password, hashed) is True

    def test_verify_password_incorrect(self):
        """Test password verification with incorrect password"""
        password = "testpassword123"
        wrong_password = "wrongpassword456"
        hashed = hash_password(password)

        assert verify_password(wrong_password, hashed) is False

    def test_verify_password_empty_password(self):
        """Test password verification with empty password"""
        password = ""
        hashed = hash_password(password)

        assert verify_password(password, hashed) is True
        assert verify_password("notEmpty", hashed) is False

    def test_verify_password_malformed_hash(self):
        """Test password verification with malformed hash"""
        password = "testpassword123"

        # Hash without separator
        assert verify_password(password, "malformedhash") is False

        # Hash with multiple separators
        assert verify_password(password, "salt:hash:extra") is False

        # Empty hash
        assert verify_password(password, "") is False

    def test_verify_password_special_characters(self):
        """Test password hashing and verification with special characters"""
        password = "P@ssw0rd!#$%^&*()"
        hashed = hash_password(password)

        assert verify_password(password, hashed) is True
        assert verify_password("P@ssw0rd!#$%^&*", hashed) is False

    def test_verify_password_unicode(self):
        """Test password hashing with unicode characters"""
        password = "пароль密码🔐"
        hashed = hash_password(password)

        assert verify_password(password, hashed) is True
        assert verify_password("пароль", hashed) is False


class TestSessionToken:
    """Test session token generation"""

    def test_generate_session_token_returns_string(self):
        """Test that session token is a string"""
        token = generate_session_token()
        assert isinstance(token, str)
        assert len(token) > 0

    def test_generate_session_token_unique(self):
        """Test that session tokens are unique"""
        tokens = [generate_session_token() for _ in range(100)]
        assert len(set(tokens)) == 100

    def test_generate_session_token_url_safe(self):
        """Test that session token is URL-safe"""
        token = generate_session_token()
        # URL-safe base64 should only contain these characters
        allowed_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")
        assert all(c in allowed_chars for c in token)


class TestSessionCookie:
    """Test session cookie creation"""

    def test_create_session_cookie_basic(self):
        """Test basic session cookie creation"""
        token = "test_session_token_123"
        cookie = create_session_cookie(token)

        assert cookie["key"] == "session_id"
        assert cookie["value"] == token
        assert cookie["httponly"] is True
        assert cookie["secure"] is False
        assert cookie["samesite"] == "lax"
        assert cookie["max_age"] is None  # Session cookie by default

    def test_create_session_cookie_remember_me(self):
        """Test session cookie with remember_me flag"""
        token = "test_session_token_123"
        cookie = create_session_cookie(token, remember_me=True)

        assert cookie["key"] == "session_id"
        assert cookie["value"] == token
        assert cookie["httponly"] is True
        assert cookie["max_age"] == 30 * 24 * 60 * 60  # 30 days in seconds

    def test_create_session_cookie_not_remember_me(self):
        """Test session cookie without remember_me flag"""
        token = "test_session_token_123"
        cookie = create_session_cookie(token, remember_me=False)

        assert cookie["max_age"] is None  # Session cookie

    def test_create_session_cookie_security_flags(self):
        """Test that security flags are set correctly"""
        token = "test_session_token_123"
        cookie = create_session_cookie(token)

        # httponly prevents JavaScript access
        assert cookie["httponly"] is True

        # secure should be True in production (HTTPS)
        # Currently False for development
        assert cookie["secure"] is False

        # samesite prevents CSRF
        assert cookie["samesite"] == "lax"
