"""
Session management for IA Modules
"""

from typing import Optional
from datetime import datetime, timedelta
import secrets
import hashlib
import logging

logger = logging.getLogger(__name__)


class SessionManager:
    """Manage user sessions using database provider"""
    
    def __init__(self, db_provider):
        """Initialize with database provider"""
        self.db_provider = db_provider
    
    async def create_session(self, user_id: int, expires_in_hours: int = 24) -> Optional[str]:
        """Create a new session for user"""
        if not self.db_provider:
            return None
        
        try:
            session_token = self.generate_session_token()
            expires_at = datetime.now() + timedelta(hours=expires_in_hours)
            
            query = """
            INSERT INTO user_sessions (user_id, session_token, expires_at)
            VALUES (?, ?, ?)
            """
            result = await self.db_provider.execute_query(query, (user_id, session_token, expires_at.isoformat()))
            
            return session_token if result.success else None
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return None
    
    async def validate_session(self, session_token: str) -> Optional[int]:
        """Validate session and return user ID if valid"""
        if not self.db_provider:
            return None
        
        try:
            query = """
            SELECT user_id FROM user_sessions
            WHERE session_token = ? AND expires_at > datetime('now')
            """
            result = await self.db_provider.fetch_one(query, (session_token,))
            
            return result.data[0]['user_id'] if result.success and result.data else None
        except Exception as e:
            logger.error(f"Error validating session: {e}")
            return None
    
    async def destroy_session(self, session_token: str) -> bool:
        """Delete a session"""
        if not self.db_provider:
            return False
        
        try:
            result = await self.db_provider.execute_query("DELETE FROM user_sessions WHERE session_token = ?", (session_token,))
            return result.success
        except Exception as e:
            logger.error(f"Error deleting session: {e}")
            return False
    
    async def delete_user_sessions(self, user_id: int) -> bool:
        """Delete all sessions for a user"""
        if not self.db_provider:
            return False
        
        try:
            result = await self.db_provider.execute_query("DELETE FROM user_sessions WHERE user_id = ?", (user_id,))
            return result.success
        except Exception as e:
            logger.error(f"Error deleting user sessions: {e}")
            return False
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions and return count of deleted sessions"""
        if not self.db_provider:
            return 0
        
        try:
            result = await self.db_provider.execute_query("DELETE FROM user_sessions WHERE expires_at <= datetime('now')")
            return result.row_count if result.success else 0
        except Exception as e:
            logger.error(f"Error cleaning up sessions: {e}")
            return 0
    
    async def get_user_by_email(self, email: str):
        """Get user by email"""
        if not self.db_provider:
            return None
        
        try:
            query = "SELECT * FROM users WHERE email = ?"
            result = await self.db_provider.fetch_one(query, (email,))
            
            return result.data[0] if result.success and result.data else None
        except Exception as e:
            logger.error(f"Error getting user by email: {e}")
            return None
    
    async def create_user(self, email: str, password: str, first_name: str, last_name: str, role: str = "user") -> Optional[int]:
        """Create a new user"""
        if not self.db_provider:
            return None

        try:
            import uuid
            from .security import hash_password

            user_uuid = str(uuid.uuid4())
            password_hash = hash_password(password)

            query = """
            INSERT INTO users (uuid, email, password_hash, first_name, last_name, role, active)
            VALUES (?, ?, ?, ?, ?, ?, 1)
            """
            result = await self.db_provider.execute_query(query, (user_uuid, email, password_hash, first_name, last_name, role))
            
            if result.success:
                # Get the created user ID
                user_result = await self.db_provider.fetch_one("SELECT id FROM users WHERE email = ?", (email,))
                return user_result.data[0]['id'] if user_result.success and user_result.data else None
            
            return None
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return None
    
    @staticmethod
    def generate_session_token() -> str:
        """Generate a secure session token"""
        return secrets.token_urlsafe(32)
