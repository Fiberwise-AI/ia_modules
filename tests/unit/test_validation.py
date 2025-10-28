"""
Unit tests for structured output validation.

Tests StructuredOutputValidator.
"""
import pytest
from ia_modules.pipeline.test_utils import create_test_execution_context
from pydantic import BaseModel, ValidationError, Field
from typing import List, Optional
from ia_modules.validation.core import StructuredOutputValidator


class SimpleModel(BaseModel):
    """Simple test model."""
    name: str
    age: int


class ComplexModel(BaseModel):
    """Complex test model."""
    title: str
    items: List[str]
    score: float
    metadata: Optional[dict] = None


class NestedModel(BaseModel):
    """Nested test model."""
    user: SimpleModel
    status: str


@pytest.mark.asyncio
class TestStructuredOutputValidator:
    """Test StructuredOutputValidator."""

    async def test_validator_creation(self):
        """Validator can be created."""
        validator = StructuredOutputValidator()

        assert validator is not None

    async def test_validate_dict_valid(self):
        """Validator accepts valid dict."""
        validator = StructuredOutputValidator()

        data = {"name": "Alice", "age": 30}
        result = await validator.validate(data, SimpleModel)

        assert isinstance(result, SimpleModel)
        assert result.name == "Alice"
        assert result.age == 30

    async def test_validate_json_string_valid(self):
        """Validator accepts valid JSON string."""
        validator = StructuredOutputValidator()

        json_str = '{"name": "Bob", "age": 25}'
        result = await validator.validate(json_str, SimpleModel)

        assert isinstance(result, SimpleModel)
        assert result.name == "Bob"
        assert result.age == 25

    async def test_validate_dict_invalid(self):
        """Validator rejects invalid dict."""
        validator = StructuredOutputValidator()

        data = {"name": "Charlie"}  # Missing required 'age'

        with pytest.raises(ValidationError):
            await validator.validate(data, SimpleModel)

    async def test_validate_dict_wrong_type(self):
        """Validator rejects wrong type."""
        validator = StructuredOutputValidator()

        data = {"name": "David", "age": "not a number"}

        with pytest.raises(ValidationError):
            await validator.validate(data, SimpleModel)

    async def test_validate_complex_model(self):
        """Validator handles complex models."""
        validator = StructuredOutputValidator()

        data = {
            "title": "Test",
            "items": ["a", "b", "c"],
            "score": 9.5,
            "metadata": {"key": "value"}
        }

        result = await validator.validate(data, ComplexModel)

        assert result.title == "Test"
        assert len(result.items) == 3
        assert result.score == 9.5
        assert result.metadata == {"key": "value"}

    async def test_validate_nested_model(self):
        """Validator handles nested models."""
        validator = StructuredOutputValidator()

        data = {
            "user": {"name": "Eve", "age": 28},
            "status": "active"
        }

        result = await validator.validate(data, NestedModel)

        assert result.user.name == "Eve"
        assert result.user.age == 28
        assert result.status == "active"

    async def test_validate_and_retry_success_first_try(self):
        """validate_and_retry succeeds on first try."""
        validator = StructuredOutputValidator()

        data = {"name": "Frank", "age": 35}

        async def retry_func(error_feedback: str = None):
            return data  # Not called on success

        result = await validator.validate_and_retry(
            data, SimpleModel, retry_func
        )

        assert result.name == "Frank"
        assert result.age == 35

    async def test_validate_and_retry_success_second_try(self):
        """validate_and_retry succeeds on second try."""
        validator = StructuredOutputValidator()

        invalid_data = {"name": "Grace"}  # Missing age
        valid_data = {"name": "Grace", "age": 40}

        async def retry_func(error_feedback: str = None):
            assert error_feedback is not None  # Should get error feedback
            return valid_data

        result = await validator.validate_and_retry(
            invalid_data, SimpleModel, retry_func, max_retries=3
        )

        assert result.name == "Grace"
        assert result.age == 40

    async def test_validate_and_retry_max_retries(self):
        """validate_and_retry fails after max retries."""
        validator = StructuredOutputValidator()

        invalid_data = {"name": "Henry"}  # Always missing age

        async def retry_func(error_feedback: str = None):
            return invalid_data  # Always return invalid

        with pytest.raises(ValidationError):
            await validator.validate_and_retry(
                invalid_data, SimpleModel, retry_func, max_retries=2
            )

    async def test_validate_and_retry_error_formatting(self):
        """validate_and_retry formats errors clearly."""
        validator = StructuredOutputValidator()

        invalid_data = {"name": "Ivy", "age": "not_a_number"}
        error_received = None

        async def retry_func(error_feedback: str = None):
            nonlocal error_received
            error_received = error_feedback
            # Fix the error
            return {"name": "Ivy", "age": 25}

        result = await validator.validate_and_retry(
            invalid_data, SimpleModel, retry_func
        )

        assert error_received is not None
        assert "age" in error_received  # Error mentions the problematic field
        assert result.name == "Ivy"
        assert result.age == 25

    async def test_extract_json_from_text(self):
        """Validator can extract JSON from text."""
        validator = StructuredOutputValidator()

        text = """
        Here is the response:
        {"name": "Jack", "age": 32}
        That's the data.
        """

        result = await validator.validate(text, SimpleModel)

        assert result.name == "Jack"
        assert result.age == 32

    async def test_extract_json_with_code_block(self):
        """Validator can extract JSON from markdown code block."""
        validator = StructuredOutputValidator()

        text = """
        ```json
        {"name": "Kate", "age": 29}
        ```
        """

        result = await validator.validate(text, SimpleModel)

        assert result.name == "Kate"
        assert result.age == 29

    async def test_to_json_schema(self):
        """Validator can generate JSON schema from model."""
        validator = StructuredOutputValidator()

        schema = validator.to_json_schema(SimpleModel)

        assert "properties" in schema
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]
        assert schema["properties"]["name"]["type"] == "string"
        assert schema["properties"]["age"]["type"] == "integer"
