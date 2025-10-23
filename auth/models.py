"""
Authentication models for IA Modules
"""

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, Dict, Any


class UserRole(Enum):
    """User roles enum"""
    USER = "user"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"
    FACILITY_ADMIN = "facility_admin"


@dataclass
class CurrentUser:
    """Current authenticated user model"""
    id: int
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    role: str = "user"
    facility_id: Optional[int] = None
    active: bool = True
    is_admin: bool = False
    is_super_admin: bool = False
    is_facility_admin: bool = False
    username: Optional[str] = None

    @property
    def full_name(self) -> str:
        """Get full name"""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        elif self.first_name:
            return self.first_name
        elif self.last_name:
            return self.last_name
        else:
            return self.email
    
    def __post_init__(self):
        """Set convenience properties based on role"""
        if self.role == UserRole.SUPER_ADMIN.value:
            self.is_super_admin = True
            self.is_admin = True
        elif self.role == UserRole.ADMIN.value:
            self.is_admin = True
        elif self.role == UserRole.FACILITY_ADMIN.value:
            self.is_facility_admin = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

