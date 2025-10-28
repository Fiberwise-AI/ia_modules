# Knowledge Graph Integration Implementation Plan

## Overview

This document provides a comprehensive implementation plan for integrating knowledge graph capabilities into the ia_modules pipeline system. Knowledge graphs enable rich relationship modeling, graph-based reasoning, and enhanced RAG with structured knowledge.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Graph Database Interface](#graph-database-interface)
3. [Neo4j Integration](#neo4j-integration)
4. [Graph Schema Management](#graph-schema-management)
5. [Graph Query Builder](#graph-query-builder)
6. [Graph-Based RAG](#graph-based-rag)
7. [Entity Extraction & Linking](#entity-extraction--linking)
8. [Graph Algorithms](#graph-algorithms)
9. [Pipeline Integration](#pipeline-integration)
10. [Testing Strategy](#testing-strategy)

---

## 1. Architecture Overview

### 1.1 Design Principles

- **Provider Agnostic**: Abstract interface for different graph databases (Neo4j, ArangoDB, etc.)
- **Rich Schema**: Support for typed nodes, relationships, and properties
- **Cypher Queries**: Full Cypher query language support for Neo4j
- **Graph RAG**: Combine vector search with graph traversal for context
- **Entity Management**: Extract, link, and store entities with relationships
- **Scalable**: Batch operations, connection pooling, query optimization

### 1.2 Component Architecture

```
ia_modules/
├── graph/
│   ├── __init__.py
│   ├── models.py              # Data models
│   ├── base.py                # Abstract interface
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── neo4j.py           # Neo4j implementation
│   │   └── memory.py          # In-memory graph (testing)
│   ├── schema.py              # Schema management
│   ├── query_builder.py       # Query builder DSL
│   ├── rag.py                 # Graph-based RAG
│   ├── entity_extractor.py    # Entity extraction
│   └── algorithms.py          # Graph algorithms
├── pipeline/
│   └── steps/
│       ├── graph_query.py     # Query graph
│       └── entity_linking.py  # Link entities
└── tests/
    └── integration/
        └── test_graph.py
```

---

## 2. Graph Database Interface

### 2.1 Data Models

**File**: `ia_modules/graph/models.py`

```python
"""Data models for knowledge graph operations."""
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class NodeType(str, Enum):
    """Common node types."""
    DOCUMENT = "Document"
    ENTITY = "Entity"
    CONCEPT = "Concept"
    PERSON = "Person"
    ORGANIZATION = "Organization"
    LOCATION = "Location"
    EVENT = "Event"


class RelationType(str, Enum):
    """Common relationship types."""
    MENTIONS = "MENTIONS"
    RELATED_TO = "RELATED_TO"
    PART_OF = "PART_OF"
    LOCATED_IN = "LOCATED_IN"
    WORKS_FOR = "WORKS_FOR"
    KNOWS = "KNOWS"
    CAUSED_BY = "CAUSED_BY"
    SIMILAR_TO = "SIMILAR_TO"


class GraphNode(BaseModel):
    """Node in knowledge graph."""
    id: Optional[str] = Field(None, description="Node ID (assigned by DB)")
    labels: List[str] = Field(default_factory=list, description="Node labels/types")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Node properties")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "123",
                "labels": ["Entity", "Person"],
                "properties": {
                    "name": "Ada Lovelace",
                    "born": 1815,
                    "occupation": "Mathematician"
                }
            }
        }


class GraphRelationship(BaseModel):
    """Relationship between nodes."""
    id: Optional[str] = Field(None, description="Relationship ID")
    type: str = Field(..., description="Relationship type")
    start_node: Union[str, GraphNode] = Field(..., description="Start node ID or object")
    end_node: Union[str, GraphNode] = Field(..., description="End node ID or object")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Relationship properties")

    class Config:
        json_schema_extra = {
            "example": {
                "type": "WORKS_FOR",
                "start_node": "person_123",
                "end_node": "org_456",
                "properties": {
                    "role": "Engineer",
                    "since": 2020
                }
            }
        }


class GraphQuery(BaseModel):
    """Graph query specification."""
    query: str = Field(..., description="Query string (Cypher for Neo4j)")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Query parameters")
    limit: Optional[int] = Field(None, description="Result limit")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "MATCH (p:Person)-[:WORKS_FOR]->(o:Organization) WHERE o.name = $org_name RETURN p",
                "parameters": {"org_name": "Acme Corp"},
                "limit": 10
            }
        }


class GraphQueryResult(BaseModel):
    """Result from graph query."""
    nodes: List[GraphNode] = Field(default_factory=list)
    relationships: List[GraphRelationship] = Field(default_factory=list)
    records: List[Dict[str, Any]] = Field(default_factory=list, description="Raw query results")
    execution_time_ms: float = 0.0


class GraphPath(BaseModel):
    """Path through graph (sequence of nodes and relationships)."""
    nodes: List[GraphNode]
    relationships: List[GraphRelationship]
    length: int

    def to_text(self) -> str:
        """Convert path to human-readable text."""
        parts = []
        for i, node in enumerate(self.nodes):
            node_name = node.properties.get("name", f"Node{i}")
            parts.append(node_name)

            if i < len(self.relationships):
                rel = self.relationships[i]
                parts.append(f"-[{rel.type}]->")

        return " ".join(parts)


class GraphConfig(BaseModel):
    """Configuration for graph database."""
    provider: str = Field(..., description="Provider: neo4j, arangodb, memory")
    uri: str = Field(..., description="Database URI")
    username: Optional[str] = None
    password: Optional[str] = None
    database: str = Field("neo4j", description="Database name")

    # Connection pooling
    max_connection_pool_size: int = Field(50, gt=0)
    connection_timeout: float = Field(30.0, gt=0)

    # Query
    default_limit: int = Field(100, gt=0)
    query_timeout: float = Field(60.0, gt=0)
```

### 2.2 Abstract Base Class

**File**: `ia_modules/graph/base.py`

```python
"""Abstract base class for graph database providers."""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, AsyncIterator
from contextlib import asynccontextmanager
import time
from .models import (
    GraphNode,
    GraphRelationship,
    GraphQuery,
    GraphQueryResult,
    GraphPath,
    GraphConfig
)


class GraphDatabaseBase(ABC):
    """Abstract base class for graph database implementations."""

    def __init__(self, config: GraphConfig):
        """Initialize graph database."""
        self.config = config
        self._connected = False

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to graph database."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to graph database."""
        pass

    @abstractmethod
    async def create_node(
        self,
        labels: List[str],
        properties: Dict[str, Any]
    ) -> GraphNode:
        """
        Create a node in the graph.

        Args:
            labels: Node labels/types
            properties: Node properties

        Returns:
            Created node with ID
        """
        pass

    @abstractmethod
    async def create_relationship(
        self,
        start_node_id: str,
        end_node_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> GraphRelationship:
        """
        Create relationship between nodes.

        Args:
            start_node_id: Start node ID
            end_node_id: End node ID
            rel_type: Relationship type
            properties: Optional relationship properties

        Returns:
            Created relationship
        """
        pass

    @abstractmethod
    async def get_node(
        self,
        node_id: str
    ) -> Optional[GraphNode]:
        """
        Get node by ID.

        Args:
            node_id: Node ID

        Returns:
            Node or None if not found
        """
        pass

    @abstractmethod
    async def find_nodes(
        self,
        labels: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[GraphNode]:
        """
        Find nodes matching criteria.

        Args:
            labels: Filter by labels
            properties: Filter by properties
            limit: Max results

        Returns:
            List of matching nodes
        """
        pass

    @abstractmethod
    async def execute_query(
        self,
        query: GraphQuery
    ) -> GraphQueryResult:
        """
        Execute raw query.

        Args:
            query: Query specification

        Returns:
            Query results
        """
        pass

    @abstractmethod
    async def find_paths(
        self,
        start_node_id: str,
        end_node_id: str,
        max_depth: int = 5,
        relationship_types: Optional[List[str]] = None
    ) -> List[GraphPath]:
        """
        Find paths between nodes.

        Args:
            start_node_id: Start node
            end_node_id: End node
            max_depth: Maximum path length
            relationship_types: Filter by relationship types

        Returns:
            List of paths
        """
        pass

    @abstractmethod
    async def get_neighbors(
        self,
        node_id: str,
        relationship_types: Optional[List[str]] = None,
        direction: str = "both",  # incoming, outgoing, both
        depth: int = 1
    ) -> List[GraphNode]:
        """
        Get neighboring nodes.

        Args:
            node_id: Node ID
            relationship_types: Filter by relationship types
            direction: Relationship direction
            depth: Traversal depth

        Returns:
            List of neighbor nodes
        """
        pass

    @abstractmethod
    async def delete_node(self, node_id: str) -> bool:
        """Delete node and its relationships."""
        pass

    @abstractmethod
    async def delete_relationship(self, relationship_id: str) -> bool:
        """Delete relationship."""
        pass

    @abstractmethod
    async def create_index(
        self,
        label: str,
        property_key: str
    ) -> None:
        """Create index on node property."""
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
        return False


class GraphDatabaseError(Exception):
    """Base exception for graph database operations."""
    pass


class NodeNotFoundError(GraphDatabaseError):
    """Node not found in graph."""
    pass


class QueryExecutionError(GraphDatabaseError):
    """Error executing graph query."""
    pass
```

---

## 3. Neo4j Integration

### 3.1 Neo4j Implementation

**File**: `ia_modules/graph/providers/neo4j.py`

```python
"""Neo4j graph database implementation."""
from typing import List, Optional, Dict, Any
import asyncio
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from neo4j.exceptions import ServiceUnavailable, ClientError
from ..base import GraphDatabaseBase, NodeNotFoundError, QueryExecutionError
from ..models import (
    GraphNode,
    GraphRelationship,
    GraphQuery,
    GraphQueryResult,
    GraphPath,
    GraphConfig
)


class Neo4jGraphDatabase(GraphDatabaseBase):
    """Neo4j graph database implementation."""

    def __init__(self, config: GraphConfig):
        """Initialize Neo4j client."""
        super().__init__(config)
        self._driver: Optional[AsyncDriver] = None

    async def connect(self) -> None:
        """Establish connection to Neo4j."""
        if self._connected:
            return

        try:
            self._driver = AsyncGraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
                max_connection_pool_size=self.config.max_connection_pool_size,
                connection_timeout=self.config.connection_timeout
            )

            # Verify connectivity
            await self._driver.verify_connectivity()
            self._connected = True
        except ServiceUnavailable as e:
            raise GraphDatabaseError(f"Failed to connect to Neo4j: {e}")

    async def disconnect(self) -> None:
        """Close Neo4j connection."""
        if self._driver:
            await self._driver.close()
            self._driver = None
        self._connected = False

    async def create_node(
        self,
        labels: List[str],
        properties: Dict[str, Any]
    ) -> GraphNode:
        """Create node in Neo4j."""
        if not self._driver:
            raise GraphDatabaseError("Not connected to Neo4j")

        # Build Cypher query
        labels_str = ":".join(labels)
        query = f"""
        CREATE (n:{labels_str} $props)
        RETURN n
        """

        async with self._driver.session(database=self.config.database) as session:
            result = await session.run(query, props=properties)
            record = await result.single()

            if not record:
                raise GraphDatabaseError("Failed to create node")

            node_data = dict(record["n"])
            node_id = str(record["n"].element_id)

            return GraphNode(
                id=node_id,
                labels=labels,
                properties=node_data
            )

    async def create_relationship(
        self,
        start_node_id: str,
        end_node_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> GraphRelationship:
        """Create relationship in Neo4j."""
        if not self._driver:
            raise GraphDatabaseError("Not connected to Neo4j")

        properties = properties or {}

        query = """
        MATCH (start), (end)
        WHERE elementId(start) = $start_id AND elementId(end) = $end_id
        CREATE (start)-[r:%s $props]->(end)
        RETURN r
        """ % rel_type

        async with self._driver.session(database=self.config.database) as session:
            result = await session.run(
                query,
                start_id=start_node_id,
                end_id=end_node_id,
                props=properties
            )
            record = await result.single()

            if not record:
                raise GraphDatabaseError("Failed to create relationship")

            rel_data = dict(record["r"])
            rel_id = str(record["r"].element_id)

            return GraphRelationship(
                id=rel_id,
                type=rel_type,
                start_node=start_node_id,
                end_node=end_node_id,
                properties=rel_data
            )

    async def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get node by ID from Neo4j."""
        if not self._driver:
            raise GraphDatabaseError("Not connected to Neo4j")

        query = """
        MATCH (n)
        WHERE elementId(n) = $node_id
        RETURN n, labels(n) AS labels
        """

        async with self._driver.session(database=self.config.database) as session:
            result = await session.run(query, node_id=node_id)
            record = await result.single()

            if not record:
                return None

            return GraphNode(
                id=node_id,
                labels=record["labels"],
                properties=dict(record["n"])
            )

    async def find_nodes(
        self,
        labels: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[GraphNode]:
        """Find nodes in Neo4j."""
        if not self._driver:
            raise GraphDatabaseError("Not connected to Neo4j")

        # Build query
        where_clauses = []
        params = {"limit": limit}

        if labels:
            label_str = ":".join(labels)
            match_clause = f"MATCH (n:{label_str})"
        else:
            match_clause = "MATCH (n)"

        if properties:
            for key, value in properties.items():
                param_name = f"prop_{key}"
                where_clauses.append(f"n.{key} = ${param_name}")
                params[param_name] = value

        where_clause = " AND ".join(where_clauses) if where_clauses else "true"

        query = f"""
        {match_clause}
        WHERE {where_clause}
        RETURN n, labels(n) AS labels
        LIMIT $limit
        """

        async with self._driver.session(database=self.config.database) as session:
            result = await session.run(query, **params)
            records = await result.data()

            nodes = []
            for record in records:
                node = GraphNode(
                    id=str(record["n"].element_id),
                    labels=record["labels"],
                    properties=dict(record["n"])
                )
                nodes.append(node)

            return nodes

    async def execute_query(self, query: GraphQuery) -> GraphQueryResult:
        """Execute Cypher query."""
        if not self._driver:
            raise GraphDatabaseError("Not connected to Neo4j")

        start_time = time.time()

        try:
            async with self._driver.session(database=self.config.database) as session:
                result = await session.run(query.query, **query.parameters)
                records = await result.data()

                # Parse nodes and relationships from results
                nodes = []
                relationships = []

                for record in records:
                    for value in record.values():
                        if hasattr(value, 'labels'):  # Node
                            node = GraphNode(
                                id=str(value.element_id),
                                labels=list(value.labels),
                                properties=dict(value)
                            )
                            nodes.append(node)
                        elif hasattr(value, 'type'):  # Relationship
                            rel = GraphRelationship(
                                id=str(value.element_id),
                                type=value.type,
                                start_node=str(value.start_node.element_id),
                                end_node=str(value.end_node.element_id),
                                properties=dict(value)
                            )
                            relationships.append(rel)

                execution_time = (time.time() - start_time) * 1000

                return GraphQueryResult(
                    nodes=nodes,
                    relationships=relationships,
                    records=records,
                    execution_time_ms=execution_time
                )

        except ClientError as e:
            raise QueryExecutionError(f"Query execution failed: {e}")

    async def find_paths(
        self,
        start_node_id: str,
        end_node_id: str,
        max_depth: int = 5,
        relationship_types: Optional[List[str]] = None
    ) -> List[GraphPath]:
        """Find paths between nodes in Neo4j."""
        if not self._driver:
            raise GraphDatabaseError("Not connected to Neo4j")

        # Build relationship type filter
        if relationship_types:
            rel_filter = "|".join(relationship_types)
            rel_pattern = f"[:{rel_filter}*1..{max_depth}]"
        else:
            rel_pattern = f"[*1..{max_depth}]"

        query = f"""
        MATCH path = (start)-{rel_pattern}-(end)
        WHERE elementId(start) = $start_id AND elementId(end) = $end_id
        RETURN path
        LIMIT 10
        """

        async with self._driver.session(database=self.config.database) as session:
            result = await session.run(
                query,
                start_id=start_node_id,
                end_id=end_node_id
            )
            records = await result.data()

            paths = []
            for record in records:
                path_obj = record["path"]

                # Extract nodes
                nodes = [
                    GraphNode(
                        id=str(node.element_id),
                        labels=list(node.labels),
                        properties=dict(node)
                    )
                    for node in path_obj.nodes
                ]

                # Extract relationships
                relationships = [
                    GraphRelationship(
                        id=str(rel.element_id),
                        type=rel.type,
                        start_node=str(rel.start_node.element_id),
                        end_node=str(rel.end_node.element_id),
                        properties=dict(rel)
                    )
                    for rel in path_obj.relationships
                ]

                path = GraphPath(
                    nodes=nodes,
                    relationships=relationships,
                    length=len(relationships)
                )
                paths.append(path)

            return paths

    async def get_neighbors(
        self,
        node_id: str,
        relationship_types: Optional[List[str]] = None,
        direction: str = "both",
        depth: int = 1
    ) -> List[GraphNode]:
        """Get neighbors of a node."""
        if not self._driver:
            raise GraphDatabaseError("Not connected to Neo4j")

        # Build relationship pattern
        if relationship_types:
            rel_filter = "|".join(relationship_types)
            rel_type_pattern = f":{rel_filter}"
        else:
            rel_type_pattern = ""

        # Build direction pattern
        if direction == "outgoing":
            pattern = f"-[{rel_type_pattern}*1..{depth}]->"
        elif direction == "incoming":
            pattern = f"<-[{rel_type_pattern}*1..{depth}]-"
        else:  # both
            pattern = f"-[{rel_type_pattern}*1..{depth}]-"

        query = f"""
        MATCH (start){pattern}(neighbor)
        WHERE elementId(start) = $node_id
        RETURN DISTINCT neighbor, labels(neighbor) AS labels
        """

        async with self._driver.session(database=self.config.database) as session:
            result = await session.run(query, node_id=node_id)
            records = await result.data()

            neighbors = []
            for record in records:
                neighbor = GraphNode(
                    id=str(record["neighbor"].element_id),
                    labels=record["labels"],
                    properties=dict(record["neighbor"])
                )
                neighbors.append(neighbor)

            return neighbors

    async def delete_node(self, node_id: str) -> bool:
        """Delete node and relationships."""
        if not self._driver:
            raise GraphDatabaseError("Not connected to Neo4j")

        query = """
        MATCH (n)
        WHERE elementId(n) = $node_id
        DETACH DELETE n
        RETURN count(n) AS deleted
        """

        async with self._driver.session(database=self.config.database) as session:
            result = await session.run(query, node_id=node_id)
            record = await result.single()
            return record["deleted"] > 0

    async def delete_relationship(self, relationship_id: str) -> bool:
        """Delete relationship."""
        if not self._driver:
            raise GraphDatabaseError("Not connected to Neo4j")

        query = """
        MATCH ()-[r]->()
        WHERE elementId(r) = $rel_id
        DELETE r
        RETURN count(r) AS deleted
        """

        async with self._driver.session(database=self.config.database) as session:
            result = await session.run(query, rel_id=relationship_id)
            record = await result.single()
            return record["deleted"] > 0

    async def create_index(self, label: str, property_key: str) -> None:
        """Create index on node property."""
        if not self._driver:
            raise GraphDatabaseError("Not connected to Neo4j")

        query = f"""
        CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.{property_key})
        """

        async with self._driver.session(database=self.config.database) as session:
            await session.run(query)

    async def create_full_text_index(
        self,
        index_name: str,
        labels: List[str],
        properties: List[str]
    ) -> None:
        """Create full-text search index."""
        if not self._driver:
            raise GraphDatabaseError("Not connected to Neo4j")

        labels_str = ", ".join([f"'{label}'" for label in labels])
        props_str = ", ".join([f"'{prop}'" for prop in properties])

        query = f"""
        CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS
        FOR (n:{"|".join(labels)})
        ON EACH [{props_str}]
        """

        async with self._driver.session(database=self.config.database) as session:
            await session.run(query)
```

---

## 4. Graph Schema Management

### 4.1 Schema Manager

**File**: `ia_modules/graph/schema.py`

```python
"""Schema management for knowledge graphs."""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from .base import GraphDatabaseBase
from .models import NodeType, RelationType


class NodeSchema(BaseModel):
    """Schema definition for node type."""
    label: str
    required_properties: List[str] = Field(default_factory=list)
    optional_properties: List[str] = Field(default_factory=list)
    property_types: Dict[str, str] = Field(default_factory=dict)
    indexes: List[str] = Field(default_factory=list)


class RelationshipSchema(BaseModel):
    """Schema definition for relationship type."""
    type: str
    start_node_labels: List[str]
    end_node_labels: List[str]
    required_properties: List[str] = Field(default_factory=list)
    optional_properties: List[str] = Field(default_factory=list)


class GraphSchema(BaseModel):
    """Complete graph schema."""
    nodes: Dict[str, NodeSchema] = Field(default_factory=dict)
    relationships: Dict[str, RelationshipSchema] = Field(default_factory=dict)


class SchemaManager:
    """Manage knowledge graph schema."""

    def __init__(self, graph_db: GraphDatabaseBase):
        """Initialize schema manager."""
        self.graph_db = graph_db
        self.schema = self._default_schema()

    @staticmethod
    def _default_schema() -> GraphSchema:
        """Create default schema for common use cases."""
        return GraphSchema(
            nodes={
                "Document": NodeSchema(
                    label="Document",
                    required_properties=["id", "content"],
                    optional_properties=["title", "created_at", "source"],
                    property_types={
                        "id": "string",
                        "content": "string",
                        "title": "string",
                        "created_at": "datetime"
                    },
                    indexes=["id", "title"]
                ),
                "Entity": NodeSchema(
                    label="Entity",
                    required_properties=["name", "type"],
                    optional_properties=["description", "aliases"],
                    property_types={
                        "name": "string",
                        "type": "string",
                        "description": "string"
                    },
                    indexes=["name", "type"]
                ),
                "Concept": NodeSchema(
                    label="Concept",
                    required_properties=["name"],
                    optional_properties=["definition", "domain"],
                    property_types={
                        "name": "string",
                        "definition": "string",
                        "domain": "string"
                    },
                    indexes=["name"]
                )
            },
            relationships={
                "MENTIONS": RelationshipSchema(
                    type="MENTIONS",
                    start_node_labels=["Document"],
                    end_node_labels=["Entity", "Concept"],
                    optional_properties=["count", "confidence"]
                ),
                "RELATED_TO": RelationshipSchema(
                    type="RELATED_TO",
                    start_node_labels=["Entity", "Concept"],
                    end_node_labels=["Entity", "Concept"],
                    optional_properties=["strength", "reason"]
                )
            }
        )

    async def apply_schema(self) -> None:
        """Apply schema to graph database (create indexes, etc.)."""
        # Create indexes for all node types
        for node_schema in self.schema.nodes.values():
            for index_prop in node_schema.indexes:
                await self.graph_db.create_index(
                    label=node_schema.label,
                    property_key=index_prop
                )

    def validate_node(
        self,
        labels: List[str],
        properties: Dict[str, Any]
    ) -> bool:
        """Validate node against schema."""
        for label in labels:
            if label not in self.schema.nodes:
                continue  # Unknown labels are allowed

            node_schema = self.schema.nodes[label]

            # Check required properties
            for required_prop in node_schema.required_properties:
                if required_prop not in properties:
                    raise ValueError(
                        f"Missing required property '{required_prop}' for node type '{label}'"
                    )

        return True
```

---

## 5. Graph Query Builder

### 5.1 Query Builder DSL

**File**: `ia_modules/graph/query_builder.py`

```python
"""Query builder DSL for graph queries."""
from typing import List, Dict, Any, Optional
from .models import GraphQuery


class CypherQueryBuilder:
    """Fluent interface for building Cypher queries."""

    def __init__(self):
        """Initialize query builder."""
        self._match_clauses: List[str] = []
        self._where_clauses: List[str] = []
        self._return_clause: Optional[str] = None
        self._order_by: Optional[str] = None
        self._limit: Optional[int] = None
        self._parameters: Dict[str, Any] = {}

    def match(
        self,
        pattern: str,
        **params
    ) -> "CypherQueryBuilder":
        """Add MATCH clause."""
        self._match_clauses.append(pattern)
        self._parameters.update(params)
        return self

    def where(
        self,
        condition: str,
        **params
    ) -> "CypherQueryBuilder":
        """Add WHERE condition."""
        self._where_clauses.append(condition)
        self._parameters.update(params)
        return self

    def return_nodes(self, *node_vars: str) -> "CypherQueryBuilder":
        """Add RETURN clause for nodes."""
        self._return_clause = ", ".join(node_vars)
        return self

    def order_by(self, expression: str) -> "CypherQueryBuilder":
        """Add ORDER BY clause."""
        self._order_by = expression
        return self

    def limit(self, count: int) -> "CypherQueryBuilder":
        """Add LIMIT clause."""
        self._limit = count
        return self

    def build(self) -> GraphQuery:
        """Build final Cypher query."""
        parts = []

        # MATCH clauses
        if self._match_clauses:
            parts.append("MATCH " + ", ".join(self._match_clauses))

        # WHERE clause
        if self._where_clauses:
            parts.append("WHERE " + " AND ".join(self._where_clauses))

        # RETURN clause
        if self._return_clause:
            parts.append(f"RETURN {self._return_clause}")

        # ORDER BY
        if self._order_by:
            parts.append(f"ORDER BY {self._order_by}")

        # LIMIT
        if self._limit:
            parts.append(f"LIMIT {self._limit}")

        query_string = "\n".join(parts)

        return GraphQuery(
            query=query_string,
            parameters=self._parameters,
            limit=self._limit
        )


# Example usage helper functions

def find_entity_by_name(name: str) -> GraphQuery:
    """Find entity by name."""
    return (
        CypherQueryBuilder()
        .match("(e:Entity)", name=name)
        .where("e.name = $name")
        .return_nodes("e")
        .build()
    )


def find_related_entities(entity_id: str, depth: int = 2) -> GraphQuery:
    """Find entities related to given entity."""
    return (
        CypherQueryBuilder()
        .match(f"(e:Entity)-[:RELATED_TO*1..{depth}]-(related:Entity)")
        .where("elementId(e) = $entity_id", entity_id=entity_id)
        .return_nodes("related")
        .limit(20)
        .build()
    )


def find_documents_mentioning_entity(entity_id: str) -> GraphQuery:
    """Find documents mentioning an entity."""
    return (
        CypherQueryBuilder()
        .match("(d:Document)-[:MENTIONS]->(e:Entity)")
        .where("elementId(e) = $entity_id", entity_id=entity_id)
        .return_nodes("d")
        .order_by("d.created_at DESC")
        .limit(10)
        .build()
    )
```

---

## 6. Graph-Based RAG

### 6.1 Graph RAG Implementation

**File**: `ia_modules/graph/rag.py`

```python
"""Graph-based Retrieval Augmented Generation."""
from typing import List, Dict, Any, Optional
from ..embeddings.base import EmbeddingProviderBase
from ..vector_stores.base import VectorStoreBase
from .base import GraphDatabaseBase
from .models import GraphNode, GraphPath


class GraphRAG:
    """
    Graph-based RAG combining vector search with graph traversal.

    Process:
    1. Vector search for relevant documents/entities
    2. Graph traversal to find related entities and relationships
    3. Construct rich context from graph neighborhood
    4. Generate response using LLM with graph context
    """

    def __init__(
        self,
        graph_db: GraphDatabaseBase,
        vector_store: VectorStoreBase,
        embedding_provider: EmbeddingProviderBase
    ):
        """Initialize Graph RAG."""
        self.graph_db = graph_db
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider

    async def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
        graph_depth: int = 2
    ) -> Dict[str, Any]:
        """
        Retrieve context for query using graph and vectors.

        Args:
            query: User query
            top_k: Number of initial results
            graph_depth: Depth of graph traversal

        Returns:
            Rich context dictionary
        """
        # 1. Generate query embedding
        embedding_response = await self.embedding_provider.generate_embeddings([query])
        query_vector = embedding_response.embeddings[0]

        # 2. Vector search for relevant documents
        vector_results = await self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )

        # 3. For each result, find corresponding graph node
        graph_nodes = []
        for result in vector_results.results:
            doc_id = result.id
            nodes = await self.graph_db.find_nodes(
                labels=["Document"],
                properties={"id": doc_id}
            )
            if nodes:
                graph_nodes.extend(nodes)

        # 4. Graph traversal to find related entities
        related_entities = []
        entity_relationships = []

        for node in graph_nodes:
            # Get neighboring entities
            neighbors = await self.graph_db.get_neighbors(
                node_id=node.id,
                relationship_types=["MENTIONS", "RELATED_TO"],
                depth=graph_depth
            )
            related_entities.extend(neighbors)

            # Get paths to related entities (for explanation)
            for neighbor in neighbors[:3]:  # Limit to avoid explosion
                paths = await self.graph_db.find_paths(
                    start_node_id=node.id,
                    end_node_id=neighbor.id,
                    max_depth=graph_depth
                )
                if paths:
                    entity_relationships.extend(paths)

        # 5. Construct context
        context = {
            "query": query,
            "documents": [
                {
                    "id": r.id,
                    "content": r.metadata.get("text", ""),
                    "score": r.score
                }
                for r in vector_results.results
            ],
            "entities": [
                {
                    "name": e.properties.get("name", ""),
                    "type": e.properties.get("type", ""),
                    "description": e.properties.get("description", "")
                }
                for e in related_entities[:10]
            ],
            "relationships": [
                {
                    "path": path.to_text(),
                    "length": path.length
                }
                for path in entity_relationships[:5]
            ]
        }

        return context

    def format_context_for_llm(self, context: Dict[str, Any]) -> str:
        """Format graph context for LLM prompt."""
        parts = []

        # Add documents
        parts.append("## Relevant Documents")
        for i, doc in enumerate(context["documents"], 1):
            parts.append(f"{i}. {doc['content'][:500]}... (score: {doc['score']:.3f})")

        # Add entities
        if context["entities"]:
            parts.append("\n## Related Entities")
            for entity in context["entities"]:
                parts.append(
                    f"- {entity['name']} ({entity['type']}): {entity.get('description', '')}"
                )

        # Add relationships
        if context["relationships"]:
            parts.append("\n## Knowledge Graph Relationships")
            for rel in context["relationships"]:
                parts.append(f"- {rel['path']}")

        return "\n".join(parts)
```

---

## 7. Entity Extraction & Linking

### 7.1 Entity Extractor

**File**: `ia_modules/graph/entity_extractor.py`

```python
"""Entity extraction and linking for knowledge graphs."""
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from openai import AsyncOpenAI
from .base import GraphDatabaseBase
from .models import GraphNode


class EntityExtractor:
    """Extract entities from text and link to knowledge graph."""

    def __init__(
        self,
        llm_client: AsyncOpenAI,
        graph_db: GraphDatabaseBase,
        model: str = "gpt-4o-mini"
    ):
        """Initialize entity extractor."""
        self.llm_client = llm_client
        self.graph_db = graph_db
        self.model = model

    async def extract_entities(
        self,
        text: str,
        entity_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract entities from text using LLM.

        Args:
            text: Input text
            entity_types: Specific entity types to extract

        Returns:
            List of extracted entities
        """
        entity_types_str = ", ".join(entity_types) if entity_types else "any type"

        prompt = f"""Extract entities of type {entity_types_str} from the following text.

Text:
{text}

Return a JSON array of entities with this format:
[
  {{"name": "entity name", "type": "entity type", "description": "brief description"}}
]

Only return the JSON array, no other text."""

        response = await self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert at entity extraction."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )

        # Parse JSON response
        import json
        try:
            entities = json.loads(response.choices[0].message.content)
            return entities
        except json.JSONDecodeError:
            return []

    async def link_entity(
        self,
        entity_name: str,
        entity_type: str
    ) -> Optional[GraphNode]:
        """
        Link entity to existing node in graph or create new.

        Args:
            entity_name: Entity name
            entity_type: Entity type

        Returns:
            Graph node (existing or new)
        """
        # Try to find existing entity
        existing = await self.graph_db.find_nodes(
            labels=["Entity"],
            properties={"name": entity_name, "type": entity_type},
            limit=1
        )

        if existing:
            return existing[0]

        # Create new entity node
        node = await self.graph_db.create_node(
            labels=["Entity", entity_type],
            properties={
                "name": entity_name,
                "type": entity_type,
                "created_at": datetime.now().isoformat()
            }
        )

        return node

    async def process_document(
        self,
        document_id: str,
        document_text: str,
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process document: create node, extract entities, create relationships.

        Args:
            document_id: Document ID
            document_text: Document content
            document_metadata: Optional metadata

        Returns:
            Processing results
        """
        # Create document node
        doc_properties = {
            "id": document_id,
            "content": document_text[:1000],  # Store truncated
            "created_at": datetime.now().isoformat()
        }
        if document_metadata:
            doc_properties.update(document_metadata)

        doc_node = await self.graph_db.create_node(
            labels=["Document"],
            properties=doc_properties
        )

        # Extract entities
        entities = await self.extract_entities(document_text)

        # Link entities to document
        linked_entities = []
        for entity_data in entities:
            # Link or create entity
            entity_node = await self.link_entity(
                entity_name=entity_data["name"],
                entity_type=entity_data["type"]
            )

            # Create MENTIONS relationship
            await self.graph_db.create_relationship(
                start_node_id=doc_node.id,
                end_node_id=entity_node.id,
                rel_type="MENTIONS",
                properties={"confidence": 0.9}
            )

            linked_entities.append(entity_node)

        return {
            "document_node": doc_node,
            "entities": linked_entities,
            "entity_count": len(linked_entities)
        }
```

---

## 8. Graph Algorithms

### 8.1 Common Graph Algorithms

**File**: `ia_modules/graph/algorithms.py`

```python
"""Graph algorithms for knowledge graphs."""
from typing import List, Dict, Any, Set
from collections import defaultdict, deque
from .base import GraphDatabaseBase
from .models import GraphNode


class GraphAlgorithms:
    """Common graph algorithms."""

    def __init__(self, graph_db: GraphDatabaseBase):
        """Initialize graph algorithms."""
        self.graph_db = graph_db

    async def page_rank(
        self,
        node_labels: List[str],
        relationship_type: str,
        max_iterations: int = 20,
        damping_factor: float = 0.85
    ) -> Dict[str, float]:
        """
        Calculate PageRank for nodes (using Neo4j GDS if available).

        Args:
            node_labels: Node types to rank
            relationship_type: Relationship type to follow
            max_iterations: Max iterations
            damping_factor: Damping factor (0-1)

        Returns:
            Dict mapping node ID to PageRank score
        """
        # Simplified PageRank using Cypher
        # In production, use Neo4j Graph Data Science library
        query = f"""
        CALL gds.pageRank.stream({{
            nodeLabels: {node_labels},
            relationshipTypes: ['{relationship_type}'],
            maxIterations: {max_iterations},
            dampingFactor: {damping_factor}
        }})
        YIELD nodeId, score
        RETURN gds.util.asNode(nodeId).id AS node_id, score
        ORDER BY score DESC
        """

        from .models import GraphQuery
        result = await self.graph_db.execute_query(
            GraphQuery(query=query, parameters={})
        )

        return {
            record["node_id"]: record["score"]
            for record in result.records
        }

    async def find_communities(
        self,
        node_labels: List[str],
        relationship_type: str
    ) -> Dict[str, int]:
        """
        Detect communities using Louvain algorithm.

        Returns:
            Dict mapping node ID to community ID
        """
        query = f"""
        CALL gds.louvain.stream({{
            nodeLabels: {node_labels},
            relationshipTypes: ['{relationship_type}']
        }})
        YIELD nodeId, communityId
        RETURN gds.util.asNode(nodeId).id AS node_id, communityId
        """

        from .models import GraphQuery
        result = await self.graph_db.execute_query(
            GraphQuery(query=query, parameters={})
        )

        return {
            record["node_id"]: record["communityId"]
            for record in result.records
        }

    async def find_similar_nodes(
        self,
        node_id: str,
        similarity_threshold: float = 0.7,
        top_k: int = 10
    ) -> List[tuple[GraphNode, float]]:
        """
        Find similar nodes based on neighborhood overlap (Jaccard similarity).

        Args:
            node_id: Reference node
            similarity_threshold: Minimum similarity score
            top_k: Number of results

        Returns:
            List of (node, similarity_score) tuples
        """
        # Get neighbors of reference node
        ref_neighbors = await self.graph_db.get_neighbors(
            node_id=node_id,
            depth=1
        )
        ref_neighbor_ids = {n.id for n in ref_neighbors}

        # Find candidates (neighbors of neighbors)
        candidates = set()
        for neighbor in ref_neighbors:
            second_hop = await self.graph_db.get_neighbors(
                node_id=neighbor.id,
                depth=1
            )
            candidates.update(n.id for n in second_hop if n.id != node_id)

        # Calculate Jaccard similarity for each candidate
        similar_nodes = []
        for candidate_id in candidates:
            candidate_neighbors = await self.graph_db.get_neighbors(
                node_id=candidate_id,
                depth=1
            )
            candidate_neighbor_ids = {n.id for n in candidate_neighbors}

            # Jaccard similarity
            intersection = len(ref_neighbor_ids & candidate_neighbor_ids)
            union = len(ref_neighbor_ids | candidate_neighbor_ids)
            similarity = intersection / union if union > 0 else 0.0

            if similarity >= similarity_threshold:
                node = await self.graph_db.get_node(candidate_id)
                if node:
                    similar_nodes.append((node, similarity))

        # Sort by similarity and return top K
        similar_nodes.sort(key=lambda x: x[1], reverse=True)
        return similar_nodes[:top_k]
```

---

## 9. Pipeline Integration

### 9.1 Graph Query Pipeline Step

**File**: `ia_modules/pipeline/steps/graph_query.py`

```python
"""Pipeline step for querying knowledge graph."""
from typing import Dict, Any
from ...graph.base import GraphDatabaseBase
from ...graph.models import GraphQuery


async def graph_query_step(
    context: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute graph query.

    Config:
        query: Cypher query string
        parameters: Query parameters from context
        output_field: Field to store results

    Example:
        {
            "query": "MATCH (e:Entity {name: $entity_name})-[:RELATED_TO]-(related) RETURN related",
            "parameters": {"entity_name": "context.entity"},
            "output_field": "related_entities"
        }
    """
    graph_db: GraphDatabaseBase = context.get("graph_db")
    if not graph_db:
        raise ValueError("graph_db not found in context")

    # Build query
    query_str = config["query"]
    params = {}

    # Resolve parameters from context
    for param_name, param_path in config.get("parameters", {}).items():
        if param_path.startswith("context."):
            field = param_path[8:]  # Remove "context."
            params[param_name] = context.get(field)
        else:
            params[param_name] = param_path

    # Execute query
    query = GraphQuery(query=query_str, parameters=params)
    result = await graph_db.execute_query(query)

    # Store results
    output_field = config.get("output_field", "graph_results")
    context[output_field] = result.records

    return context
```

---

## 10. Testing Strategy

### 10.1 Integration Tests

**File**: `ia_modules/tests/integration/test_graph.py`

```python
"""Integration tests for knowledge graph."""
import pytest
from ia_modules.graph.providers.neo4j import Neo4jGraphDatabase
from ia_modules.graph.models import GraphConfig


@pytest.fixture
async def neo4j_db():
    """Neo4j database fixture."""
    config = GraphConfig(
        provider="neo4j",
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password",
        database="neo4j"
    )

    db = Neo4jGraphDatabase(config)
    await db.connect()
    yield db
    await db.disconnect()


@pytest.mark.asyncio
async def test_create_and_query_nodes(neo4j_db):
    """Test node creation and querying."""
    # Create node
    node = await neo4j_db.create_node(
        labels=["Person"],
        properties={"name": "Alice", "age": 30}
    )

    assert node.id is not None
    assert "Person" in node.labels

    # Find node
    found = await neo4j_db.find_nodes(
        labels=["Person"],
        properties={"name": "Alice"}
    )

    assert len(found) >= 1
    assert found[0].properties["name"] == "Alice"


@pytest.mark.asyncio
async def test_create_relationship(neo4j_db):
    """Test relationship creation."""
    # Create two nodes
    alice = await neo4j_db.create_node(
        labels=["Person"],
        properties={"name": "Alice"}
    )

    bob = await neo4j_db.create_node(
        labels=["Person"],
        properties={"name": "Bob"}
    )

    # Create relationship
    rel = await neo4j_db.create_relationship(
        start_node_id=alice.id,
        end_node_id=bob.id,
        rel_type="KNOWS",
        properties={"since": 2020}
    )

    assert rel.id is not None
    assert rel.type == "KNOWS"
    assert rel.properties["since"] == 2020
```

---

## Summary

This implementation plan provides:

✅ **Abstract graph interface** for multiple providers
✅ **Complete Neo4j integration** with Cypher support
✅ **Schema management** for typed graphs
✅ **Query builder DSL** for fluent query construction
✅ **Graph-based RAG** combining vector + graph search
✅ **Entity extraction & linking** using LLMs
✅ **Graph algorithms** (PageRank, communities, similarity)
✅ **Pipeline integration** as reusable steps
✅ **Type safety** with Pydantic models

Next: [API_CONNECTORS.md](API_CONNECTORS.md) for external API integration.
