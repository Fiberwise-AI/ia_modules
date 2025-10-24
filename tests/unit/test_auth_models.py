"""
Tests for auth.models module
"""

import pytest
from ia_modules.auth.models import UserRole, CurrentUser


class TestUserRole:
    """Test UserRole enum"""

    def test_user_role_values(self):
        """Test that UserRole enum has correct values"""
        assert UserRole.USER.value == "user"
        assert UserRole.ADMIN.value == "admin"
        assert UserRole.SUPER_ADMIN.value == "super_admin"
        assert UserRole.FACILITY_ADMIN.value == "facility_admin"

    def test_user_role_enum_members(self):
        """Test UserRole enum members"""
        roles = list(UserRole)
        assert len(roles) == 4
        assert UserRole.USER in roles
        assert UserRole.ADMIN in roles
        assert UserRole.SUPER_ADMIN in roles
        assert UserRole.FACILITY_ADMIN in roles


class TestCurrentUser:
    """Test CurrentUser dataclass"""

    def test_current_user_basic_creation(self):
        """Test basic CurrentUser creation"""
        user = CurrentUser(
            id=1,
            email="test@example.com"
        )

        assert user.id == 1
        assert user.email == "test@example.com"
        assert user.first_name is None
        assert user.last_name is None
        assert user.role == "user"
        assert user.facility_id is None
        assert user.active is True
        assert user.is_admin is False
        assert user.is_super_admin is False
        assert user.is_facility_admin is False
        assert user.username is None

    def test_current_user_full_creation(self):
        """Test CurrentUser creation with all fields"""
        user = CurrentUser(
            id=1,
            email="test@example.com",
            first_name="John",
            last_name="Doe",
            role="admin",
            facility_id=5,
            active=True,
            username="johndoe"
        )

        assert user.id == 1
        assert user.email == "test@example.com"
        assert user.first_name == "John"
        assert user.last_name == "Doe"
        assert user.role == "admin"
        assert user.facility_id == 5
        assert user.active is True
        assert user.username == "johndoe"

    def test_current_user_full_name_both_names(self):
        """Test full_name property with both first and last name"""
        user = CurrentUser(
            id=1,
            email="test@example.com",
            first_name="John",
            last_name="Doe"
        )

        assert user.full_name == "John Doe"

    def test_current_user_full_name_first_only(self):
        """Test full_name property with only first name"""
        user = CurrentUser(
            id=1,
            email="test@example.com",
            first_name="John"
        )

        assert user.full_name == "John"

    def test_current_user_full_name_last_only(self):
        """Test full_name property with only last name"""
        user = CurrentUser(
            id=1,
            email="test@example.com",
            last_name="Doe"
        )

        assert user.full_name == "Doe"

    def test_current_user_full_name_no_names(self):
        """Test full_name property with no names (falls back to email)"""
        user = CurrentUser(
            id=1,
            email="test@example.com"
        )

        assert user.full_name == "test@example.com"

    def test_current_user_super_admin_role(self):
        """Test that super_admin role sets both admin flags"""
        user = CurrentUser(
            id=1,
            email="admin@example.com",
            role=UserRole.SUPER_ADMIN.value
        )

        assert user.is_super_admin is True
        assert user.is_admin is True
        assert user.is_facility_admin is False

    def test_current_user_admin_role(self):
        """Test that admin role sets admin flag"""
        user = CurrentUser(
            id=1,
            email="admin@example.com",
            role=UserRole.ADMIN.value
        )

        assert user.is_admin is True
        assert user.is_super_admin is False
        assert user.is_facility_admin is False

    def test_current_user_facility_admin_role(self):
        """Test that facility_admin role sets facility_admin flag"""
        user = CurrentUser(
            id=1,
            email="fadmin@example.com",
            role=UserRole.FACILITY_ADMIN.value
        )

        assert user.is_facility_admin is True
        assert user.is_admin is False
        assert user.is_super_admin is False

    def test_current_user_regular_user_role(self):
        """Test that regular user role doesn't set any admin flags"""
        user = CurrentUser(
            id=1,
            email="user@example.com",
            role=UserRole.USER.value
        )

        assert user.is_admin is False
        assert user.is_super_admin is False
        assert user.is_facility_admin is False

    def test_current_user_to_dict(self):
        """Test CurrentUser to_dict method"""
        user = CurrentUser(
            id=1,
            email="test@example.com",
            first_name="John",
            last_name="Doe",
            role="admin",
            facility_id=5,
            active=True,
            username="johndoe"
        )

        user_dict = user.to_dict()

        assert isinstance(user_dict, dict)
        assert user_dict["id"] == 1
        assert user_dict["email"] == "test@example.com"
        assert user_dict["first_name"] == "John"
        assert user_dict["last_name"] == "Doe"
        assert user_dict["role"] == "admin"
        assert user_dict["facility_id"] == 5
        assert user_dict["active"] is True
        assert user_dict["username"] == "johndoe"
        assert user_dict["is_admin"] is True
        assert user_dict["is_super_admin"] is False
        assert user_dict["is_facility_admin"] is False

    def test_current_user_to_dict_minimal(self):
        """Test CurrentUser to_dict with minimal data"""
        user = CurrentUser(
            id=1,
            email="test@example.com"
        )

        user_dict = user.to_dict()

        assert user_dict["id"] == 1
        assert user_dict["email"] == "test@example.com"
        assert user_dict["first_name"] is None
        assert user_dict["last_name"] is None
        assert user_dict["role"] == "user"
        assert user_dict["facility_id"] is None
        assert user_dict["active"] is True
        assert user_dict["username"] is None

    def test_current_user_inactive(self):
        """Test CurrentUser with inactive status"""
        user = CurrentUser(
            id=1,
            email="test@example.com",
            active=False
        )

        assert user.active is False

    def test_current_user_with_facility_id(self):
        """Test CurrentUser with facility_id"""
        user = CurrentUser(
            id=1,
            email="test@example.com",
            facility_id=42
        )

        assert user.facility_id == 42

    def test_current_user_custom_role_string(self):
        """Test CurrentUser with custom role string (not from enum)"""
        user = CurrentUser(
            id=1,
            email="test@example.com",
            role="custom_role"
        )

        assert user.role == "custom_role"
        assert user.is_admin is False
        assert user.is_super_admin is False
        assert user.is_facility_admin is False
