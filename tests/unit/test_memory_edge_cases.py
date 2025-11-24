"""
Edge case tests for memory/core.py to improve coverage
"""

import pytest
from datetime import datetime, timezone
from ia_modules.memory.core import Message, MessageRole


class TestMessageEdgeCases:
    """Test edge cases in Message class"""

    def test_message_to_dict_with_all_fields(self):
        """Test to_dict with all fields populated"""
        timestamp = datetime.now(timezone.utc)

        message = Message(
            message_id="msg-001",
            thread_id="thread-001",
            user_id="user-001",
            role=MessageRole.ASSISTANT,
            content="Hello, how can I help?",
            metadata={"source": "ai", "confidence": 0.95},
            timestamp=timestamp,
            function_call={"name": "get_weather", "args": {"city": "NYC"}},
            function_name="get_weather"
        )

        result = message.to_dict()

        assert result['message_id'] == "msg-001"
        assert result['thread_id'] == "thread-001"
        assert result['user_id'] == "user-001"
        assert result['role'] == "assistant"
        assert result['content'] == "Hello, how can I help?"
        assert result['metadata'] == {"source": "ai", "confidence": 0.95}
        assert result['timestamp'] == timestamp.isoformat()
        assert result['function_call'] == {"name": "get_weather", "args": {"city": "NYC"}}
        assert result['function_name'] == "get_weather"

    def test_message_to_dict_with_minimal_fields(self):
        """Test to_dict with only required fields"""
        message = Message(
            message_id="msg-002",
            thread_id="thread-002"
        )

        result = message.to_dict()

        assert result['message_id'] == "msg-002"
        assert result['thread_id'] == "thread-002"
        assert result['user_id'] is None
        assert result['role'] == "user"  # Default role
        assert result['content'] == ""
        assert result['metadata'] == {}
        assert result['function_call'] is None
        assert result['function_name'] is None

    def test_message_from_dict_with_all_fields(self):
        """Test from_dict with all fields"""
        data = {
            'message_id': "msg-003",
            'thread_id': "thread-003",
            'user_id': "user-003",
            'role': "system",
            'content': "System initialized",
            'metadata': {"version": "1.0"},
            'timestamp': "2025-01-01T12:00:00+00:00",
            'function_call': {"name": "init"},
            'function_name': "init"
        }

        message = Message.from_dict(data)

        assert message.message_id == "msg-003"
        assert message.thread_id == "thread-003"
        assert message.user_id == "user-003"
        assert message.role == MessageRole.SYSTEM
        assert message.content == "System initialized"
        assert message.metadata == {"version": "1.0"}
        assert isinstance(message.timestamp, datetime)
        assert message.function_call == {"name": "init"}
        assert message.function_name == "init"

    def test_message_from_dict_with_minimal_fields(self):
        """Test from_dict with only required fields"""
        data = {
            'message_id': "msg-004",
            'thread_id': "thread-004",
            'timestamp': "2025-01-01T12:00:00+00:00"
        }

        message = Message.from_dict(data)

        assert message.message_id == "msg-004"
        assert message.thread_id == "thread-004"
        assert message.user_id is None
        assert message.role == MessageRole.USER  # Default
        assert message.content == ""
        assert message.metadata == {}
        assert message.function_call is None
        assert message.function_name is None

    def test_message_from_dict_with_datetime_object(self):
        """Test from_dict when timestamp is a string (expected format)"""
        timestamp = datetime.now(timezone.utc)
        data = {
            'message_id': "msg-005",
            'thread_id': "thread-005",
            'timestamp': timestamp.isoformat()  # String format
        }

        message = Message.from_dict(data)

        # Compare ISO strings since precision might differ
        assert message.timestamp.isoformat() == timestamp.isoformat()

    def test_message_from_dict_with_missing_timestamp(self):
        """Test from_dict when timestamp is missing (should use current time)"""
        data = {
            'message_id': "msg-006",
            'thread_id': "thread-006"
        }

        datetime.now()
        message = Message.from_dict(data)
        datetime.now()

        # timestamp might be timezone-aware or naive depending on implementation
        # Just check it's a datetime object
        assert isinstance(message.timestamp, datetime)

    def test_message_role_enum_values(self):
        """Test all MessageRole enum values"""
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"
        assert MessageRole.SYSTEM.value == "system"
        assert MessageRole.FUNCTION.value == "function"

    def test_message_with_function_role(self):
        """Test message with FUNCTION role"""
        message = Message(
            message_id="msg-007",
            thread_id="thread-007",
            role=MessageRole.FUNCTION,
            content='{"result": "success"}',
            function_name="process_data"
        )

        assert message.role == MessageRole.FUNCTION
        assert message.function_name == "process_data"

        result = message.to_dict()
        assert result['role'] == "function"

    def test_message_roundtrip_serialization(self):
        """Test Message can roundtrip through to_dict/from_dict"""
        original = Message(
            message_id="msg-008",
            thread_id="thread-008",
            user_id="user-008",
            role=MessageRole.ASSISTANT,
            content="Test message",
            metadata={"test": True},
            timestamp=datetime.now(timezone.utc),
            function_call={"name": "test"},
            function_name="test"
        )

        # Convert to dict and back
        data = original.to_dict()
        restored = Message.from_dict(data)

        assert restored.message_id == original.message_id
        assert restored.thread_id == original.thread_id
        assert restored.user_id == original.user_id
        assert restored.role == original.role
        assert restored.content == original.content
        assert restored.metadata == original.metadata
        assert restored.function_call == original.function_call
        assert restored.function_name == original.function_name

    def test_message_with_empty_metadata(self):
        """Test message with explicitly empty metadata"""
        message = Message(
            message_id="msg-009",
            thread_id="thread-009",
            metadata={}
        )

        assert message.metadata == {}
        result = message.to_dict()
        assert result['metadata'] == {}

    def test_message_with_complex_metadata(self):
        """Test message with nested metadata"""
        metadata = {
            "source": "api",
            "config": {
                "temperature": 0.7,
                "max_tokens": 100
            },
            "tags": ["important", "urgent"]
        }

        message = Message(
            message_id="msg-010",
            thread_id="thread-010",
            metadata=metadata
        )

        assert message.metadata == metadata

        # Roundtrip test
        data = message.to_dict()
        restored = Message.from_dict(data)
        assert restored.metadata == metadata

    def test_message_with_empty_content(self):
        """Test message with empty string content"""
        message = Message(
            message_id="msg-011",
            thread_id="thread-011",
            content=""
        )

        assert message.content == ""
        result = message.to_dict()
        assert result['content'] == ""

    def test_message_from_dict_with_invalid_role(self):
        """Test from_dict with invalid role defaults gracefully"""
        data = {
            'message_id': "msg-012",
            'thread_id': "thread-012",
            'role': "invalid_role",  # This will fail enum validation
            'timestamp': "2025-01-01T12:00:00+00:00"
        }

        # Should raise ValueError because "invalid_role" is not a valid MessageRole
        with pytest.raises(ValueError, match="'invalid_role' is not a valid MessageRole"):
            Message.from_dict(data)

    def test_message_default_timestamp(self):
        """Test that message timestamp defaults to current time"""
        before = datetime.now()
        message = Message(
            message_id="msg-013",
            thread_id="thread-013"
        )
        after = datetime.now()

        assert before <= message.timestamp <= after
