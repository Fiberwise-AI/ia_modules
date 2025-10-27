"""WebSocket endpoints for real-time updates"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import Dict, Set
import asyncio
import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)
router = APIRouter()

# Active WebSocket connections
active_connections: Dict[str, Set[WebSocket]] = {
    "metrics": set(),
    "executions": {}
}


class ConnectionManager:
    """Manage WebSocket connections"""

    def __init__(self):
        self.metrics_connections: Set[WebSocket] = set()
        self.execution_connections: Dict[str, Set[WebSocket]] = {}
        self.hitl_connections: Dict[str, Set[WebSocket]] = {}  # user_id -> connections

    async def connect_metrics(self, websocket: WebSocket):
        """Connect to metrics stream"""
        await websocket.accept()
        self.metrics_connections.add(websocket)
        logger.info(f"Metrics WebSocket connected. Total: {len(self.metrics_connections)}")

    async def connect_execution(self, job_id: str, websocket: WebSocket):
        """Connect to execution stream"""
        await websocket.accept()

        if job_id not in self.execution_connections:
            self.execution_connections[job_id] = set()

        self.execution_connections[job_id].add(websocket)
        logger.info(f"Execution WebSocket connected for job {job_id}")

    def disconnect_metrics(self, websocket: WebSocket):
        """Disconnect from metrics stream"""
        self.metrics_connections.discard(websocket)
        logger.info(f"Metrics WebSocket disconnected. Remaining: {len(self.metrics_connections)}")

    def disconnect_execution(self, job_id: str, websocket: WebSocket):
        """Disconnect from execution stream"""
        if job_id in self.execution_connections:
            self.execution_connections[job_id].discard(websocket)

            if not self.execution_connections[job_id]:
                del self.execution_connections[job_id]

        logger.info(f"Execution WebSocket disconnected for job {job_id}")

    async def broadcast_metrics(self, message: dict):
        """Broadcast to all metrics connections"""
        if not self.metrics_connections:
            return

        disconnected = set()

        for connection in self.metrics_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to metrics WebSocket: {e}")
                disconnected.add(connection)

        # Clean up disconnected
        for connection in disconnected:
            self.metrics_connections.discard(connection)

    async def broadcast_execution(self, job_id: str, message: dict):
        """Broadcast to execution connections for specific job"""
        if job_id not in self.execution_connections:
            return

        disconnected = set()

        for connection in self.execution_connections[job_id]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to execution WebSocket: {e}")
                disconnected.add(connection)

        # Clean up disconnected
        for connection in disconnected:
            self.execution_connections[job_id].discard(connection)

    async def connect_hitl(self, user_id: str, websocket: WebSocket):
        """Connect to HITL notifications stream for a specific user"""
        await websocket.accept()

        if user_id not in self.hitl_connections:
            self.hitl_connections[user_id] = set()

        self.hitl_connections[user_id].add(websocket)
        logger.info(f"HITL WebSocket connected for user {user_id}")

    def disconnect_hitl(self, user_id: str, websocket: WebSocket):
        """Disconnect from HITL stream"""
        if user_id in self.hitl_connections:
            self.hitl_connections[user_id].discard(websocket)

            if not self.hitl_connections[user_id]:
                del self.hitl_connections[user_id]

        logger.info(f"HITL WebSocket disconnected for user {user_id}")

    async def broadcast_hitl_notification(self, user_ids: list, message: dict):
        """Broadcast HITL notification to specific users"""
        disconnected = []

        for user_id in user_ids:
            if user_id not in self.hitl_connections:
                continue

            for connection in self.hitl_connections[user_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error sending HITL notification to {user_id}: {e}")
                    disconnected.append((user_id, connection))

        # Clean up disconnected
        for user_id, connection in disconnected:
            self.hitl_connections[user_id].discard(connection)


manager = ConnectionManager()


@router.websocket("/metrics")
async def websocket_metrics_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time metrics updates"""
    await manager.connect_metrics(websocket)

    try:
        # Send initial message
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to metrics stream",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        # Keep connection alive and listen for messages
        while True:
            try:
                # Wait for message with timeout
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )

                # Echo back (for ping/pong)
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_json({
                    "type": "keepalive",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

    except WebSocketDisconnect:
        manager.disconnect_metrics(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        manager.disconnect_metrics(websocket)


@router.websocket("/execution/{job_id}")
async def websocket_execution_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time execution updates"""
    await manager.connect_execution(job_id, websocket)

    try:
        # Send initial message
        await websocket.send_json({
            "type": "connected",
            "job_id": job_id,
            "message": f"Connected to execution stream for job {job_id}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        # Keep connection alive
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )

                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

            except asyncio.TimeoutError:
                await websocket.send_json({
                    "type": "keepalive",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

    except WebSocketDisconnect:
        manager.disconnect_execution(job_id, websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        manager.disconnect_execution(job_id, websocket)


# Background task to broadcast metrics updates
async def metrics_broadcaster():
    """Background task to broadcast metrics updates"""
    from main import get_metrics_service

    while True:
        try:
            await asyncio.sleep(5)  # Update every 5 seconds

            service = get_metrics_service()
            if service and manager.metrics_connections:
                report = await service.get_report()

                await manager.broadcast_metrics({
                    "type": "metrics_update",
                    "data": report,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

        except Exception as e:
            logger.error(f"Error in metrics broadcaster: {e}", exc_info=True)
            await asyncio.sleep(5)


@router.websocket("/hitl/{user_id}")
async def websocket_hitl_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time HITL notifications for a specific user"""
    await manager.connect_hitl(user_id, websocket)

    try:
        # Send initial message
        await websocket.send_json({
            "type": "connected",
            "message": f"Connected to HITL notifications for user {user_id}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        # Keep connection alive and listen for messages
        while True:
            try:
                # Wait for message with timeout
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )

                # Parse message
                try:
                    msg = json.loads(data)
                    if msg.get("type") == "ping":
                        await websocket.send_json({
                            "type": "pong",
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        })
                except json.JSONDecodeError:
                    pass

            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_json({
                    "type": "keepalive",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

    except WebSocketDisconnect:
        manager.disconnect_hitl(user_id, websocket)
    except Exception as e:
        logger.error(f"HITL WebSocket error for user {user_id}: {e}", exc_info=True)
        manager.disconnect_hitl(user_id, websocket)


# Export manager for use in other modules
def get_ws_manager():
    """Get WebSocket manager instance"""
    return manager
