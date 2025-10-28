# API Connectors Implementation Plan

## Overview

This document provides a comprehensive implementation plan for building a flexible, extensible API connector system within ia_modules. This enables pipelines to integrate with external services via REST, GraphQL, gRPC, and WebSocket protocols.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Connector Interface](#connector-interface)
3. [REST API Connector](#rest-api-connector)
4. [GraphQL Connector](#graphql-connector)
5. [gRPC Connector](#grpc-connector)
6. [WebSocket Connector](#websocket-connector)
7. [Authentication & Security](#authentication--security)
8. [Rate Limiting & Retry](#rate-limiting--retry)
9. [Connector Registry](#connector-registry)
10. [Pipeline Integration](#pipeline-integration)
11. [Testing Strategy](#testing-strategy)

---

## 1. Architecture Overview

### 1.1 Design Principles

- **Protocol Agnostic**: Support multiple API protocols (REST, GraphQL, gRPC, WebSocket)
- **Authentication**: Flexible auth strategies (API key, OAuth2, JWT, custom)
- **Resilience**: Built-in retry logic, circuit breakers, timeouts
- **Rate Limiting**: Respect API rate limits, implement backoff
- **Registry**: Centralized connector registry for discovery and reuse
- **Type Safety**: Full type hints and Pydantic validation

### 1.2 Component Architecture

```
ia_modules/
├── connectors/
│   ├── __init__.py
│   ├── models.py              # Data models
│   ├── base.py                # Abstract connector interface
│   ├── auth.py                # Authentication strategies
│   ├── protocols/
│   │   ├── __init__.py
│   │   ├── rest.py            # REST connector
│   │   ├── graphql.py         # GraphQL connector
│   │   ├── grpc.py            # gRPC connector
│   │   └── websocket.py       # WebSocket connector
│   ├── registry.py            # Connector registry
│   ├── rate_limiter.py        # Rate limiting
│   └── circuit_breaker.py    # Circuit breaker pattern
├── pipeline/
│   └── steps/
│       └── api_call.py        # API call pipeline step
└── tests/
    └── integration/
        └── test_connectors.py
```

---

## 2. Connector Interface

### 2.1 Data Models

**File**: `ia_modules/connectors/models.py`

```python
"""Data models for API connectors."""
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl
from enum import Enum


class HTTPMethod(str, Enum):
    """HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class AuthType(str, Enum):
    """Authentication types."""
    NONE = "none"
    API_KEY = "api_key"
    BEARER = "bearer"
    BASIC = "basic"
    OAUTH2 = "oauth2"
    JWT = "jwt"
    CUSTOM = "custom"


class ApiRequest(BaseModel):
    """API request specification."""
    method: HTTPMethod = Field(HTTPMethod.GET, description="HTTP method")
    endpoint: str = Field(..., description="API endpoint path")
    headers: Dict[str, str] = Field(default_factory=dict)
    query_params: Dict[str, Any] = Field(default_factory=dict)
    body: Optional[Union[Dict, str]] = None
    timeout: float = Field(30.0, description="Request timeout in seconds")

    class Config:
        json_schema_extra = {
            "example": {
                "method": "POST",
                "endpoint": "/api/v1/search",
                "headers": {"Content-Type": "application/json"},
                "query_params": {"limit": 10},
                "body": {"query": "machine learning"}
            }
        }


class ApiResponse(BaseModel):
    """API response."""
    status_code: int = Field(..., description="HTTP status code")
    headers: Dict[str, str] = Field(default_factory=dict)
    body: Optional[Union[Dict, str, bytes]] = None
    execution_time_ms: float = 0.0
    success: bool = Field(..., description="Whether request succeeded")
    error: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "status_code": 200,
                "headers": {"content-type": "application/json"},
                "body": {"results": []},
                "execution_time_ms": 123.45,
                "success": True
            }
        }


class ConnectorConfig(BaseModel):
    """Configuration for API connector."""
    name: str = Field(..., description="Connector identifier")
    base_url: HttpUrl = Field(..., description="Base URL for API")
    auth_type: AuthType = Field(AuthType.NONE)

    # Authentication credentials
    api_key: Optional[str] = None
    api_key_header: str = Field("X-API-Key", description="Header name for API key")
    bearer_token: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None

    # OAuth2
    oauth2_client_id: Optional[str] = None
    oauth2_client_secret: Optional[str] = None
    oauth2_token_url: Optional[HttpUrl] = None

    # Rate limiting
    rate_limit: Optional[int] = Field(None, description="Max requests per minute")
    rate_limit_window: int = Field(60, description="Rate limit window in seconds")

    # Retry configuration
    max_retries: int = Field(3, ge=0)
    retry_backoff: float = Field(1.0, description="Exponential backoff base")

    # Timeouts
    default_timeout: float = Field(30.0, gt=0)
    connection_timeout: float = Field(10.0, gt=0)

    # Headers
    default_headers: Dict[str, str] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "name": "example_api",
                "base_url": "https://api.example.com",
                "auth_type": "api_key",
                "api_key": "sk-xxxxx",
                "rate_limit": 60
            }
        }


class RateLimitInfo(BaseModel):
    """Rate limit information."""
    limit: int
    remaining: int
    reset_at: datetime
    retry_after: Optional[int] = None


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if recovered
```

### 2.2 Abstract Base Class

**File**: `ia_modules/connectors/base.py`

```python
"""Abstract base class for API connectors."""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import time
from .models import ApiRequest, ApiResponse, ConnectorConfig


class ConnectorBase(ABC):
    """Abstract base class for API connectors."""

    def __init__(self, config: ConnectorConfig):
        """Initialize connector."""
        self.config = config
        self._session = None

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection/session."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection/session."""
        pass

    @abstractmethod
    async def execute(self, request: ApiRequest) -> ApiResponse:
        """
        Execute API request.

        Args:
            request: Request specification

        Returns:
            API response
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if API is healthy.

        Returns:
            True if healthy, False otherwise
        """
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
        return False


class ConnectorError(Exception):
    """Base exception for connector errors."""
    pass


class AuthenticationError(ConnectorError):
    """Authentication failed."""
    pass


class RateLimitError(ConnectorError):
    """Rate limit exceeded."""
    pass


class ConnectionError(ConnectorError):
    """Connection failed."""
    pass


class TimeoutError(ConnectorError):
    """Request timeout."""
    pass
```

---

## 3. REST API Connector

### 3.1 REST Implementation

**File**: `ia_modules/connectors/protocols/rest.py`

```python
"""REST API connector implementation."""
from typing import Optional, Dict, Any
import asyncio
import time
import httpx
from ..base import ConnectorBase, RateLimitError, TimeoutError, AuthenticationError
from ..models import ApiRequest, ApiResponse, ConnectorConfig, HTTPMethod
from ..auth import AuthStrategy, get_auth_strategy
from ..rate_limiter import RateLimiter
from ..circuit_breaker import CircuitBreaker


class RestConnector(ConnectorBase):
    """REST API connector with authentication, rate limiting, and retry."""

    def __init__(self, config: ConnectorConfig):
        """Initialize REST connector."""
        super().__init__(config)
        self._client: Optional[httpx.AsyncClient] = None
        self._auth_strategy: Optional[AuthStrategy] = None
        self._rate_limiter: Optional[RateLimiter] = None
        self._circuit_breaker: Optional[CircuitBreaker] = None

    async def connect(self) -> None:
        """Initialize HTTP client and components."""
        # Create HTTP client
        self._client = httpx.AsyncClient(
            base_url=str(self.config.base_url),
            timeout=httpx.Timeout(
                connect=self.config.connection_timeout,
                read=self.config.default_timeout,
                write=self.config.default_timeout,
                pool=self.config.default_timeout
            ),
            follow_redirects=True
        )

        # Setup authentication
        self._auth_strategy = get_auth_strategy(self.config)

        # Setup rate limiter
        if self.config.rate_limit:
            self._rate_limiter = RateLimiter(
                max_requests=self.config.rate_limit,
                window_seconds=self.config.rate_limit_window
            )

        # Setup circuit breaker
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0,
            expected_exception=httpx.HTTPError
        )

    async def disconnect(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def execute(self, request: ApiRequest) -> ApiResponse:
        """Execute REST API request with retry logic."""
        if not self._client:
            raise ConnectorError("Not connected")

        # Check circuit breaker
        if self._circuit_breaker and not self._circuit_breaker.can_execute():
            raise ConnectorError("Circuit breaker is open")

        # Check rate limit
        if self._rate_limiter:
            await self._rate_limiter.acquire()

        # Retry loop
        last_error = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = await self._execute_once(request)

                # Success - close circuit breaker
                if self._circuit_breaker:
                    self._circuit_breaker.record_success()

                return response

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_error = e

                # Record failure in circuit breaker
                if self._circuit_breaker:
                    self._circuit_breaker.record_failure()

                # Retry with exponential backoff
                if attempt < self.config.max_retries:
                    delay = self.config.retry_backoff * (2 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    raise TimeoutError(f"Request failed after {attempt + 1} attempts: {e}")

            except httpx.HTTPStatusError as e:
                # Don't retry client errors (4xx)
                if 400 <= e.response.status_code < 500:
                    raise ConnectorError(f"Client error: {e}")

                # Retry server errors (5xx)
                if attempt < self.config.max_retries:
                    delay = self.config.retry_backoff * (2 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    raise ConnectorError(f"Server error after {attempt + 1} attempts: {e}")

        raise ConnectorError(f"Request failed: {last_error}")

    async def _execute_once(self, request: ApiRequest) -> ApiResponse:
        """Execute single request attempt."""
        start_time = time.time()

        # Build headers
        headers = {**self.config.default_headers, **request.headers}

        # Add authentication
        if self._auth_strategy:
            headers.update(await self._auth_strategy.get_headers())

        # Prepare request kwargs
        request_kwargs = {
            "method": request.method.value,
            "url": request.endpoint,
            "headers": headers,
            "params": request.query_params,
            "timeout": request.timeout
        }

        # Add body if present
        if request.body is not None:
            if isinstance(request.body, dict):
                request_kwargs["json"] = request.body
            else:
                request_kwargs["content"] = request.body

        # Execute request
        try:
            response = await self._client.request(**request_kwargs)
            response.raise_for_status()

            # Parse response body
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                body = response.json()
            elif "text/" in content_type:
                body = response.text
            else:
                body = response.content

            execution_time = (time.time() - start_time) * 1000

            return ApiResponse(
                status_code=response.status_code,
                headers=dict(response.headers),
                body=body,
                execution_time_ms=execution_time,
                success=True
            )

        except httpx.HTTPStatusError as e:
            execution_time = (time.time() - start_time) * 1000

            # Handle rate limit
            if e.response.status_code == 429:
                raise RateLimitError("Rate limit exceeded")

            # Handle authentication
            if e.response.status_code == 401:
                raise AuthenticationError("Authentication failed")

            return ApiResponse(
                status_code=e.response.status_code,
                headers=dict(e.response.headers),
                body=e.response.text,
                execution_time_ms=execution_time,
                success=False,
                error=str(e)
            )

    async def health_check(self) -> bool:
        """Check API health."""
        try:
            request = ApiRequest(
                method=HTTPMethod.GET,
                endpoint="/health",  # Common health endpoint
                timeout=5.0
            )
            response = await self.execute(request)
            return response.success and response.status_code == 200
        except Exception:
            return False


class RestApiBuilder:
    """Fluent builder for REST API requests."""

    def __init__(self, connector: RestConnector):
        """Initialize builder."""
        self.connector = connector
        self._request = ApiRequest(endpoint="/")

    def get(self, endpoint: str) -> "RestApiBuilder":
        """GET request."""
        self._request.method = HTTPMethod.GET
        self._request.endpoint = endpoint
        return self

    def post(self, endpoint: str) -> "RestApiBuilder":
        """POST request."""
        self._request.method = HTTPMethod.POST
        self._request.endpoint = endpoint
        return self

    def put(self, endpoint: str) -> "RestApiBuilder":
        """PUT request."""
        self._request.method = HTTPMethod.PUT
        self._request.endpoint = endpoint
        return self

    def delete(self, endpoint: str) -> "RestApiBuilder":
        """DELETE request."""
        self._request.method = HTTPMethod.DELETE
        self._request.endpoint = endpoint
        return self

    def headers(self, headers: Dict[str, str]) -> "RestApiBuilder":
        """Add headers."""
        self._request.headers.update(headers)
        return self

    def params(self, params: Dict[str, Any]) -> "RestApiBuilder":
        """Add query parameters."""
        self._request.query_params.update(params)
        return self

    def json(self, body: Dict[str, Any]) -> "RestApiBuilder":
        """Add JSON body."""
        self._request.body = body
        return self

    def timeout(self, seconds: float) -> "RestApiBuilder":
        """Set timeout."""
        self._request.timeout = seconds
        return self

    async def execute(self) -> ApiResponse:
        """Execute the request."""
        return await self.connector.execute(self._request)
```

### 3.2 REST Usage Example

```python
"""Example: Using REST connector."""
import asyncio
from ia_modules.connectors.protocols.rest import RestConnector, RestApiBuilder
from ia_modules.connectors.models import ConnectorConfig, AuthType, HTTPMethod


async def main():
    # Configure connector
    config = ConnectorConfig(
        name="github_api",
        base_url="https://api.github.com",
        auth_type=AuthType.BEARER,
        bearer_token="ghp_xxxxx",
        rate_limit=60,
        max_retries=3
    )

    # Use connector
    async with RestConnector(config) as connector:
        # Method 1: Direct execution
        from ia_modules.connectors.models import ApiRequest

        request = ApiRequest(
            method=HTTPMethod.GET,
            endpoint="/users/octocat",
            headers={"Accept": "application/vnd.github.v3+json"}
        )
        response = await connector.execute(request)
        print(f"User: {response.body}")

        # Method 2: Fluent builder
        builder = RestApiBuilder(connector)
        response = await (
            builder
            .get("/repos/octocat/Hello-World")
            .headers({"Accept": "application/vnd.github.v3+json"})
            .execute()
        )
        print(f"Repo: {response.body['name']}")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 4. GraphQL Connector

### 4.1 GraphQL Implementation

**File**: `ia_modules/connectors/protocols/graphql.py`

```python
"""GraphQL connector implementation."""
from typing import Optional, Dict, Any, List
import httpx
from gql import gql, Client
from gql.transport.httpx import HTTPXAsyncTransport
from ..base import ConnectorBase
from ..models import ApiResponse, ConnectorConfig
from ..auth import get_auth_strategy


class GraphQLConnector(ConnectorBase):
    """GraphQL API connector."""

    def __init__(self, config: ConnectorConfig, endpoint: str = "/graphql"):
        """
        Initialize GraphQL connector.

        Args:
            config: Connector configuration
            endpoint: GraphQL endpoint path
        """
        super().__init__(config)
        self.endpoint = endpoint
        self._client: Optional[Client] = None
        self._transport: Optional[HTTPXAsyncTransport] = None

    async def connect(self) -> None:
        """Initialize GraphQL client."""
        # Build URL
        url = f"{self.config.base_url}{self.endpoint}"

        # Setup headers
        headers = {**self.config.default_headers}

        # Add authentication
        auth_strategy = get_auth_strategy(self.config)
        if auth_strategy:
            headers.update(await auth_strategy.get_headers())

        # Create transport
        self._transport = HTTPXAsyncTransport(
            url=url,
            headers=headers,
            timeout=self.config.default_timeout
        )

        # Create client
        self._client = Client(
            transport=self._transport,
            fetch_schema_from_transport=True
        )

    async def disconnect(self) -> None:
        """Close GraphQL client."""
        if self._client:
            await self._client.close_async()
            self._client = None
        self._transport = None

    async def query(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute GraphQL query.

        Args:
            query: GraphQL query string
            variables: Query variables

        Returns:
            Query result data
        """
        if not self._client:
            raise ConnectorError("Not connected")

        # Parse query
        document = gql(query)

        # Execute query
        result = await self._client.execute_async(
            document,
            variable_values=variables
        )

        return result

    async def mutation(
        self,
        mutation: str,
        variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute GraphQL mutation.

        Args:
            mutation: GraphQL mutation string
            variables: Mutation variables

        Returns:
            Mutation result data
        """
        return await self.query(mutation, variables)

    async def execute(self, request: Any) -> ApiResponse:
        """Execute GraphQL request (for compatibility with base class)."""
        # Extract query and variables from request
        if isinstance(request, dict):
            query_str = request.get("query", "")
            variables = request.get("variables", {})
        else:
            query_str = str(request)
            variables = {}

        # Execute query
        start_time = time.time()
        try:
            result = await self.query(query_str, variables)
            execution_time = (time.time() - start_time) * 1000

            return ApiResponse(
                status_code=200,
                headers={},
                body=result,
                execution_time_ms=execution_time,
                success=True
            )
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000

            return ApiResponse(
                status_code=500,
                headers={},
                body=None,
                execution_time_ms=execution_time,
                success=False,
                error=str(e)
            )

    async def health_check(self) -> bool:
        """Check GraphQL API health."""
        try:
            # Simple introspection query
            query = """
            query {
                __schema {
                    queryType {
                        name
                    }
                }
            }
            """
            await self.query(query)
            return True
        except Exception:
            return False


class GraphQLQueryBuilder:
    """Builder for GraphQL queries."""

    def __init__(self):
        """Initialize query builder."""
        self._operation = "query"
        self._name: Optional[str] = None
        self._fields: List[str] = []
        self._variables: Dict[str, str] = {}
        self._arguments: Dict[str, Any] = {}

    def query(self, name: str) -> "GraphQLQueryBuilder":
        """Start query operation."""
        self._operation = "query"
        self._name = name
        return self

    def mutation(self, name: str) -> "GraphQLQueryBuilder":
        """Start mutation operation."""
        self._operation = "mutation"
        self._name = name
        return self

    def variable(self, name: str, type: str) -> "GraphQLQueryBuilder":
        """Add variable."""
        self._variables[name] = type
        return self

    def argument(self, name: str, value: Any) -> "GraphQLQueryBuilder":
        """Add argument."""
        self._arguments[name] = value
        return self

    def field(self, field: str) -> "GraphQLQueryBuilder":
        """Add field to selection."""
        self._fields.append(field)
        return self

    def build(self) -> str:
        """Build GraphQL query string."""
        parts = []

        # Operation
        if self._variables:
            vars_str = ", ".join([f"${k}: {v}" for k, v in self._variables.items()])
            parts.append(f"{self._operation} {self._name}({vars_str})")
        else:
            parts.append(f"{self._operation}")

        parts.append("{")

        # Query body
        if self._arguments:
            args_str = ", ".join([f"{k}: {self._format_value(v)}" for k, v in self._arguments.items()])
            parts.append(f"  {self._name}({args_str})")
        else:
            parts.append(f"  {self._name}")

        parts.append("  {")

        # Fields
        for field in self._fields:
            parts.append(f"    {field}")

        parts.append("  }")
        parts.append("}")

        return "\n".join(parts)

    @staticmethod
    def _format_value(value: Any) -> str:
        """Format value for GraphQL."""
        if isinstance(value, str):
            if value.startswith("$"):
                return value  # Variable reference
            return f'"{value}"'
        elif isinstance(value, bool):
            return "true" if value else "false"
        else:
            return str(value)
```

### 4.2 GraphQL Usage Example

```python
"""Example: Using GraphQL connector."""
import asyncio
from ia_modules.connectors.protocols.graphql import GraphQLConnector, GraphQLQueryBuilder
from ia_modules.connectors.models import ConnectorConfig, AuthType


async def main():
    # Configure connector
    config = ConnectorConfig(
        name="github_graphql",
        base_url="https://api.github.com",
        auth_type=AuthType.BEARER,
        bearer_token="ghp_xxxxx"
    )

    async with GraphQLConnector(config) as connector:
        # Method 1: Direct query
        query = """
        query {
            viewer {
                login
                name
                email
            }
        }
        """
        result = await connector.query(query)
        print(f"User: {result['viewer']['login']}")

        # Method 2: Query with variables
        query = """
        query($owner: String!, $name: String!) {
            repository(owner: $owner, name: $name) {
                name
                description
                stargazerCount
            }
        }
        """
        variables = {"owner": "octocat", "name": "Hello-World"}
        result = await connector.query(query, variables)
        print(f"Stars: {result['repository']['stargazerCount']}")

        # Method 3: Query builder
        builder = GraphQLQueryBuilder()
        query_str = (
            builder
            .query("GetUser")
            .variable("login", "String!")
            .argument("login", "$login")
            .field("name")
            .field("email")
            .field("bio")
            .build()
        )
        result = await connector.query(query_str, {"login": "octocat"})


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 5. gRPC Connector

### 5.1 gRPC Implementation

**File**: `ia_modules/connectors/protocols/grpc.py`

```python
"""gRPC connector implementation."""
from typing import Optional, Any, Dict
import grpc
from ..base import ConnectorBase
from ..models import ApiResponse, ConnectorConfig


class GrpcConnector(ConnectorBase):
    """gRPC API connector."""

    def __init__(
        self,
        config: ConnectorConfig,
        stub_class: type,
        proto_module: Any
    ):
        """
        Initialize gRPC connector.

        Args:
            config: Connector configuration
            stub_class: gRPC stub class
            proto_module: Generated protobuf module
        """
        super().__init__(config)
        self.stub_class = stub_class
        self.proto_module = proto_module
        self._channel: Optional[grpc.aio.Channel] = None
        self._stub: Optional[Any] = None

    async def connect(self) -> None:
        """Establish gRPC channel."""
        # Extract host and port from base_url
        url_parts = str(self.config.base_url).replace("grpc://", "").replace("grpcs://", "")
        target = url_parts

        # Create channel (secure or insecure)
        if str(self.config.base_url).startswith("grpcs://"):
            credentials = grpc.ssl_channel_credentials()
            self._channel = grpc.aio.secure_channel(target, credentials)
        else:
            self._channel = grpc.aio.insecure_channel(target)

        # Create stub
        self._stub = self.stub_class(self._channel)

    async def disconnect(self) -> None:
        """Close gRPC channel."""
        if self._channel:
            await self._channel.close()
            self._channel = None
        self._stub = None

    async def call(
        self,
        method_name: str,
        request: Any,
        timeout: Optional[float] = None
    ) -> Any:
        """
        Call gRPC method.

        Args:
            method_name: Method name on stub
            request: Protobuf request message
            timeout: Request timeout

        Returns:
            Protobuf response message
        """
        if not self._stub:
            raise ConnectorError("Not connected")

        # Get method from stub
        method = getattr(self._stub, method_name)

        # Call with timeout
        timeout = timeout or self.config.default_timeout
        try:
            response = await method(request, timeout=timeout)
            return response
        except grpc.RpcError as e:
            raise ConnectorError(f"gRPC call failed: {e.code()} - {e.details()}")

    async def call_stream(
        self,
        method_name: str,
        request: Any,
        timeout: Optional[float] = None
    ):
        """
        Call gRPC streaming method.

        Args:
            method_name: Method name on stub
            request: Protobuf request message
            timeout: Request timeout

        Yields:
            Protobuf response messages
        """
        if not self._stub:
            raise ConnectorError("Not connected")

        method = getattr(self._stub, method_name)
        timeout = timeout or self.config.default_timeout

        try:
            async for response in method(request, timeout=timeout):
                yield response
        except grpc.RpcError as e:
            raise ConnectorError(f"gRPC stream failed: {e.code()} - {e.details()}")

    async def execute(self, request: Any) -> ApiResponse:
        """Execute gRPC request (for compatibility)."""
        # This is a simplified adapter - in practice, you'd need more context
        # about which RPC method to call
        start_time = time.time()

        try:
            # Assuming request contains method name and proto message
            method_name = request.get("method", "")
            proto_request = request.get("request")

            response = await self.call(method_name, proto_request)
            execution_time = (time.time() - start_time) * 1000

            return ApiResponse(
                status_code=200,
                headers={},
                body=response,
                execution_time_ms=execution_time,
                success=True
            )
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000

            return ApiResponse(
                status_code=500,
                headers={},
                body=None,
                execution_time_ms=execution_time,
                success=False,
                error=str(e)
            )

    async def health_check(self) -> bool:
        """Check gRPC service health."""
        try:
            # Use gRPC health checking protocol if available
            from grpc_health.v1 import health_pb2, health_pb2_grpc

            health_stub = health_pb2_grpc.HealthStub(self._channel)
            request = health_pb2.HealthCheckRequest()
            response = await health_stub.Check(request, timeout=5.0)

            return response.status == health_pb2.HealthCheckResponse.SERVING
        except Exception:
            return False
```

---

## 6. WebSocket Connector

### 6.1 WebSocket Implementation

**File**: `ia_modules/connectors/protocols/websocket.py`

```python
"""WebSocket connector implementation."""
from typing import Optional, AsyncIterator, Any
import asyncio
import json
import websockets
from ..base import ConnectorBase
from ..models import ConnectorConfig


class WebSocketConnector(ConnectorBase):
    """WebSocket connector for real-time communication."""

    def __init__(self, config: ConnectorConfig, endpoint: str = "/ws"):
        """
        Initialize WebSocket connector.

        Args:
            config: Connector configuration
            endpoint: WebSocket endpoint path
        """
        super().__init__(config)
        self.endpoint = endpoint
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False

    async def connect(self) -> None:
        """Establish WebSocket connection."""
        # Build WebSocket URL
        ws_url = str(self.config.base_url).replace("http://", "ws://").replace("https://", "wss://")
        url = f"{ws_url}{self.endpoint}"

        # Add headers
        headers = {**self.config.default_headers}

        # Add authentication
        from ..auth import get_auth_strategy
        auth_strategy = get_auth_strategy(self.config)
        if auth_strategy:
            headers.update(await auth_strategy.get_headers())

        # Connect
        try:
            self._websocket = await websockets.connect(
                url,
                extra_headers=headers,
                ping_interval=20,
                ping_timeout=10
            )
            self._connected = True
        except Exception as e:
            raise ConnectionError(f"WebSocket connection failed: {e}")

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
        self._connected = False

    async def send(self, message: Any) -> None:
        """
        Send message over WebSocket.

        Args:
            message: Message to send (will be JSON-encoded if dict)
        """
        if not self._websocket or not self._connected:
            raise ConnectorError("Not connected")

        # Encode message
        if isinstance(message, dict):
            message_str = json.dumps(message)
        else:
            message_str = str(message)

        await self._websocket.send(message_str)

    async def receive(self, timeout: Optional[float] = None) -> Any:
        """
        Receive message from WebSocket.

        Args:
            timeout: Receive timeout

        Returns:
            Received message (parsed as JSON if possible)
        """
        if not self._websocket or not self._connected:
            raise ConnectorError("Not connected")

        try:
            if timeout:
                message = await asyncio.wait_for(
                    self._websocket.recv(),
                    timeout=timeout
                )
            else:
                message = await self._websocket.recv()

            # Try to parse as JSON
            try:
                return json.loads(message)
            except json.JSONDecodeError:
                return message

        except asyncio.TimeoutError:
            raise TimeoutError("Receive timeout")
        except websockets.ConnectionClosed:
            self._connected = False
            raise ConnectionError("WebSocket connection closed")

    async def stream(self) -> AsyncIterator[Any]:
        """
        Stream messages from WebSocket.

        Yields:
            Messages as they arrive
        """
        if not self._websocket or not self._connected:
            raise ConnectorError("Not connected")

        try:
            async for message in self._websocket:
                # Try to parse as JSON
                try:
                    yield json.loads(message)
                except json.JSONDecodeError:
                    yield message
        except websockets.ConnectionClosed:
            self._connected = False
            raise ConnectionError("WebSocket connection closed")

    async def execute(self, request: Any) -> Any:
        """Execute request-response over WebSocket."""
        await self.send(request)
        return await self.receive(timeout=self.config.default_timeout)

    async def health_check(self) -> bool:
        """Check WebSocket connection health."""
        return self._connected and self._websocket and self._websocket.open
```

---

## 7. Authentication & Security

### 7.1 Authentication Strategies

**File**: `ia_modules/connectors/auth.py`

```python
"""Authentication strategies for API connectors."""
from abc import ABC, abstractmethod
from typing import Dict, Optional
import base64
from .models import ConnectorConfig, AuthType


class AuthStrategy(ABC):
    """Abstract authentication strategy."""

    @abstractmethod
    async def get_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        pass


class NoAuth(AuthStrategy):
    """No authentication."""

    async def get_headers(self) -> Dict[str, str]:
        """No auth headers."""
        return {}


class ApiKeyAuth(AuthStrategy):
    """API key authentication."""

    def __init__(self, api_key: str, header_name: str = "X-API-Key"):
        """Initialize API key auth."""
        self.api_key = api_key
        self.header_name = header_name

    async def get_headers(self) -> Dict[str, str]:
        """Get API key header."""
        return {self.header_name: self.api_key}


class BearerAuth(AuthStrategy):
    """Bearer token authentication."""

    def __init__(self, token: str):
        """Initialize bearer auth."""
        self.token = token

    async def get_headers(self) -> Dict[str, str]:
        """Get bearer token header."""
        return {"Authorization": f"Bearer {self.token}"}


class BasicAuth(AuthStrategy):
    """HTTP Basic authentication."""

    def __init__(self, username: str, password: str):
        """Initialize basic auth."""
        self.username = username
        self.password = password

    async def get_headers(self) -> Dict[str, str]:
        """Get basic auth header."""
        credentials = f"{self.username}:{self.password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return {"Authorization": f"Basic {encoded}"}


class OAuth2Auth(AuthStrategy):
    """OAuth2 authentication with token refresh."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        token_url: str,
        scope: Optional[str] = None
    ):
        """Initialize OAuth2 auth."""
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self.scope = scope
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None

    async def get_headers(self) -> Dict[str, str]:
        """Get OAuth2 token header."""
        if not self._access_token:
            await self._fetch_token()

        return {"Authorization": f"Bearer {self._access_token}"}

    async def _fetch_token(self) -> None:
        """Fetch OAuth2 access token."""
        import httpx

        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }
        if self.scope:
            data["scope"] = self.scope

        async with httpx.AsyncClient() as client:
            response = await client.post(self.token_url, data=data)
            response.raise_for_status()

            token_data = response.json()
            self._access_token = token_data["access_token"]
            self._refresh_token = token_data.get("refresh_token")


def get_auth_strategy(config: ConnectorConfig) -> Optional[AuthStrategy]:
    """Create authentication strategy from config."""
    if config.auth_type == AuthType.NONE:
        return NoAuth()

    elif config.auth_type == AuthType.API_KEY:
        if not config.api_key:
            raise ValueError("API key required for api_key auth")
        return ApiKeyAuth(config.api_key, config.api_key_header)

    elif config.auth_type == AuthType.BEARER:
        if not config.bearer_token:
            raise ValueError("Bearer token required for bearer auth")
        return BearerAuth(config.bearer_token)

    elif config.auth_type == AuthType.BASIC:
        if not config.username or not config.password:
            raise ValueError("Username and password required for basic auth")
        return BasicAuth(config.username, config.password)

    elif config.auth_type == AuthType.OAUTH2:
        if not config.oauth2_client_id or not config.oauth2_client_secret:
            raise ValueError("OAuth2 credentials required")
        return OAuth2Auth(
            client_id=config.oauth2_client_id,
            client_secret=config.oauth2_client_secret,
            token_url=str(config.oauth2_token_url)
        )

    return None
```

---

## 8. Rate Limiting & Retry

### 8.1 Rate Limiter

**File**: `ia_modules/connectors/rate_limiter.py`

```python
"""Rate limiting for API connectors."""
import asyncio
from datetime import datetime, timedelta
from collections import deque


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, max_requests: int, window_seconds: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_requests: Max requests per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window = timedelta(seconds=window_seconds)
        self._requests: deque = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire permission to make request (blocks if rate limit exceeded)."""
        async with self._lock:
            now = datetime.now()

            # Remove old requests outside window
            while self._requests and now - self._requests[0] > self.window:
                self._requests.popleft()

            # Check if we can proceed
            if len(self._requests) >= self.max_requests:
                # Calculate wait time
                oldest = self._requests[0]
                wait_until = oldest + self.window
                wait_seconds = (wait_until - now).total_seconds()

                if wait_seconds > 0:
                    await asyncio.sleep(wait_seconds)

                # Remove old request
                self._requests.popleft()

            # Record this request
            self._requests.append(datetime.now())
```

### 8.2 Circuit Breaker

**File**: `ia_modules/connectors/circuit_breaker.py`

```python
"""Circuit breaker pattern for API connectors."""
import asyncio
from datetime import datetime, timedelta
from .models import CircuitState


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            expected_exception: Exception type to catch
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = timedelta(seconds=recovery_timeout)
        self.expected_exception = expected_exception

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._lock = asyncio.Lock()

    def can_execute(self) -> bool:
        """Check if requests can be executed."""
        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.OPEN:
            # Check if we should transition to half-open
            if self._last_failure_time:
                if datetime.now() - self._last_failure_time > self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    return True
            return False

        # HALF_OPEN state
        return True

    def record_success(self) -> None:
        """Record successful request."""
        if self._state == CircuitState.HALF_OPEN:
            # Recovery successful
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None

    def record_failure(self) -> None:
        """Record failed request."""
        self._failure_count += 1
        self._last_failure_time = datetime.now()

        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
```

---

## 9. Connector Registry

### 9.1 Registry Implementation

**File**: `ia_modules/connectors/registry.py`

```python
"""Registry for managing API connectors."""
from typing import Dict, Optional, Type
from .base import ConnectorBase
from .models import ConnectorConfig


class ConnectorRegistry:
    """Centralized registry for API connectors."""

    def __init__(self):
        """Initialize registry."""
        self._connectors: Dict[str, ConnectorBase] = {}
        self._configs: Dict[str, ConnectorConfig] = {}

    def register(
        self,
        name: str,
        connector: ConnectorBase,
        config: ConnectorConfig
    ) -> None:
        """
        Register connector.

        Args:
            name: Connector identifier
            connector: Connector instance
            config: Connector configuration
        """
        self._connectors[name] = connector
        self._configs[name] = config

    def get(self, name: str) -> Optional[ConnectorBase]:
        """Get connector by name."""
        return self._connectors.get(name)

    def get_config(self, name: str) -> Optional[ConnectorConfig]:
        """Get connector configuration."""
        return self._configs.get(name)

    def list_connectors(self) -> list[str]:
        """List all registered connectors."""
        return list(self._connectors.keys())

    async def close_all(self) -> None:
        """Close all connectors."""
        for connector in self._connectors.values():
            await connector.disconnect()


# Global registry instance
_registry = ConnectorRegistry()


def get_registry() -> ConnectorRegistry:
    """Get global connector registry."""
    return _registry
```

---

## 10. Pipeline Integration

### 10.1 API Call Pipeline Step

**File**: `ia_modules/pipeline/steps/api_call.py`

```python
"""Pipeline step for API calls."""
from typing import Dict, Any
from ...connectors.registry import get_registry
from ...connectors.models import ApiRequest, HTTPMethod


async def api_call_step(
    context: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute API call in pipeline.

    Config:
        connector: Connector name from registry
        method: HTTP method
        endpoint: API endpoint
        body_field: Context field for request body
        output_field: Field to store response

    Example:
        {
            "connector": "github_api",
            "method": "GET",
            "endpoint": "/users/{username}",
            "output_field": "user_data"
        }
    """
    # Get connector
    connector_name = config["connector"]
    registry = get_registry()
    connector = registry.get(connector_name)

    if not connector:
        raise ValueError(f"Connector '{connector_name}' not found")

    # Build request
    endpoint = config["endpoint"]

    # Substitute path parameters from context
    import re
    for match in re.finditer(r"\{(\w+)\}", endpoint):
        param_name = match.group(1)
        param_value = context.get(param_name, "")
        endpoint = endpoint.replace(f"{{{param_name}}}", str(param_value))

    # Get body if specified
    body = None
    if "body_field" in config:
        body = context.get(config["body_field"])

    # Create request
    request = ApiRequest(
        method=HTTPMethod(config.get("method", "GET")),
        endpoint=endpoint,
        body=body
    )

    # Execute
    response = await connector.execute(request)

    # Store response
    output_field = config.get("output_field", "api_response")
    context[output_field] = response.body

    return context
```

---

## 11. Testing Strategy

### 11.1 Integration Tests

**File**: `ia_modules/tests/integration/test_connectors.py`

```python
"""Integration tests for API connectors."""
import pytest
from ia_modules.connectors.protocols.rest import RestConnector
from ia_modules.connectors.models import ConnectorConfig, AuthType, ApiRequest, HTTPMethod


@pytest.fixture
def rest_config():
    """REST connector configuration."""
    return ConnectorConfig(
        name="test_api",
        base_url="https://jsonplaceholder.typicode.com",
        auth_type=AuthType.NONE
    )


@pytest.mark.asyncio
async def test_rest_get_request(rest_config):
    """Test REST GET request."""
    async with RestConnector(rest_config) as connector:
        request = ApiRequest(
            method=HTTPMethod.GET,
            endpoint="/posts/1"
        )

        response = await connector.execute(request)

        assert response.success
        assert response.status_code == 200
        assert isinstance(response.body, dict)
        assert "userId" in response.body


@pytest.mark.asyncio
async def test_rest_post_request(rest_config):
    """Test REST POST request."""
    async with RestConnector(rest_config) as connector:
        request = ApiRequest(
            method=HTTPMethod.POST,
            endpoint="/posts",
            body={
                "title": "Test Post",
                "body": "This is a test",
                "userId": 1
            }
        )

        response = await connector.execute(request)

        assert response.success
        assert response.status_code == 201
```

---

## Summary

This implementation plan provides:

✅ **Multi-protocol support** (REST, GraphQL, gRPC, WebSocket)
✅ **Flexible authentication** (API key, Bearer, Basic, OAuth2)
✅ **Resilience patterns** (retry, circuit breaker, rate limiting)
✅ **Connector registry** for centralized management
✅ **Pipeline integration** as reusable steps
✅ **Type safety** with Pydantic models
✅ **Testing strategy** with integration tests

All 5 advanced integration implementation files complete:
1. ✅ VECTOR_DATABASE_INTEGRATION.md
2. ✅ EMBEDDING_MANAGEMENT.md
3. ✅ HYBRID_SEARCH.md
4. ✅ KNOWLEDGE_GRAPH_INTEGRATION.md
5. ✅ API_CONNECTORS.md
