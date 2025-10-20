"""
Quick test to verify the Dashboard API works
"""

import asyncio
from fastapi.testclient import TestClient
from .main import app

client = TestClient(app)


def test_health_check():
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_get_stats():
    """Test stats endpoint"""
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_pipelines" in data
    assert "active_executions" in data


def test_list_pipelines():
    """Test list pipelines"""
    response = client.get("/api/pipelines")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_list_plugins():
    """Test list plugins"""
    response = client.get("/api/plugins")
    assert response.status_code == 200
    data = response.json()
    assert "plugins" in data


def test_create_pipeline():
    """Test create pipeline"""
    pipeline = {
        "name": "Test Pipeline",
        "description": "A test pipeline",
        "config": {
            "name": "test",
            "steps": [],
            "flow": {"start_at": "test", "paths": []}
        },
        "tags": ["test"]
    }

    response = client.post("/api/pipelines", json=pipeline)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test Pipeline"
    assert "id" in data

    # Clean up - delete the pipeline
    pipeline_id = data["id"]
    response = client.delete(f"/api/pipelines/{pipeline_id}")
    assert response.status_code == 204


if __name__ == "__main__":
    print("Running Dashboard API tests...")

    test_health_check()
    print("✓ Health check passed")

    test_get_stats()
    print("✓ Stats endpoint passed")

    test_list_pipelines()
    print("✓ List pipelines passed")

    test_list_plugins()
    print("✓ List plugins passed")

    test_create_pipeline()
    print("✓ Create pipeline passed")

    print("\n✅ All tests passed!")
