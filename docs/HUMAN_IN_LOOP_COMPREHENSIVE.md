# Human-in-the-Loop (HITL) Comprehensive Design (WIP)

## Overview

A complete Human-in-the-Loop system for IA Modules that allows humans to interact with pipelines at various stages through multiple interaction patterns.

## HITL Interaction Patterns

### 1. **Pause-and-Resume Pattern**
Pipeline pauses at specific points, waits for human input, then continues.

```python
class PauseForInputStep(Step):
    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Create pending interaction record
        interaction_id = str(uuid.uuid4())

        # Store pipeline state
        await self._save_pipeline_state(interaction_id, data)

        # Return special response indicating human input needed
        return {
            "status": "human_input_required",
            "interaction_id": interaction_id,
            "ui_schema": self.config.get("ui_schema", {}),
            "prompt": self.config.get("prompt", "Please provide input"),
            "timeout_seconds": self.config.get("timeout", 3600),
            "data": data  # Pass through current data
        }

    async def _save_pipeline_state(self, interaction_id: str, data: Dict[str, Any]):
        # Save to database or cache for later retrieval
        pass
```

### 2. **Interactive Review Pattern**
Human reviews and approves/rejects pipeline outputs.

```python
class ReviewAndApproveStep(Step):
    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        content_to_review = data.get("generated_content", "")

        return {
            "status": "awaiting_review",
            "interaction_id": str(uuid.uuid4()),
            "review_data": {
                "content": content_to_review,
                "metadata": data.get("metadata", {}),
                "suggestions": await self._generate_review_suggestions(content_to_review)
            },
            "ui_schema": {
                "type": "review",
                "fields": [
                    {
                        "name": "decision",
                        "type": "radio",
                        "options": ["approve", "reject", "request_changes"],
                        "required": True
                    },
                    {
                        "name": "feedback",
                        "type": "textarea",
                        "label": "Feedback/Comments",
                        "required": False
                    },
                    {
                        "name": "suggested_changes",
                        "type": "textarea",
                        "label": "Suggested Changes",
                        "required": False
                    }
                ]
            }
        }
```

### 3. **Real-time Collaboration Pattern**
Multiple humans can interact with the pipeline simultaneously.

```python
class CollaborativeDecisionStep(Step):
    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        decision_id = str(uuid.uuid4())

        # Notify all stakeholders
        stakeholders = self.config.get("stakeholders", [])

        await self._notify_stakeholders(decision_id, stakeholders, data)

        return {
            "status": "collaborative_decision_pending",
            "decision_id": decision_id,
            "stakeholders": stakeholders,
            "decision_type": self.config.get("decision_type", "consensus"),
            "voting_options": self.config.get("voting_options", ["approve", "reject"]),
            "timeout": self.config.get("decision_timeout", 24 * 3600),  # 24 hours
            "data": data
        }
```

### 4. **Exception Handling Pattern**
When automated steps fail, escalate to human intervention.

```python
class AutoWithHumanFallbackStep(Step):
    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Attempt automated processing
            result = await self._automated_processing(data)

            # Check if result meets quality threshold
            confidence = result.get("confidence", 0)
            threshold = self.config.get("confidence_threshold", 0.8)

            if confidence >= threshold:
                return {
                    "status": "automated_success",
                    "result": result,
                    "processing_method": "automated"
                }
            else:
                # Escalate to human
                return await self._escalate_to_human(data, result, "low_confidence")

        except Exception as e:
            # Escalate to human on error
            return await self._escalate_to_human(data, None, f"error: {str(e)}")

    async def _escalate_to_human(self, data: Dict[str, Any], partial_result: Any, reason: str):
        return {
            "status": "escalated_to_human",
            "interaction_id": str(uuid.uuid4()),
            "escalation_reason": reason,
            "partial_result": partial_result,
            "original_data": data,
            "ui_schema": {
                "type": "manual_processing",
                "context": f"Automated processing failed: {reason}",
                "fields": [
                    {
                        "name": "manual_result",
                        "type": "textarea",
                        "label": "Manual Processing Result",
                        "required": True
                    },
                    {
                        "name": "confidence",
                        "type": "number",
                        "label": "Confidence Level (0-1)",
                        "min": 0,
                        "max": 1,
                        "step": 0.1,
                        "default": 0.9
                    }
                ]
            }
        }
```

### 5. **Progressive Enhancement Pattern**
Human iteratively improves automated results.

```python
class IterativeRefinementStep(Step):
    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        iteration = data.get("iteration", 1)
        max_iterations = self.config.get("max_iterations", 3)

        if iteration > max_iterations:
            return {
                "status": "max_iterations_reached",
                "final_result": data.get("current_result"),
                "iterations_completed": iteration - 1
            }

        current_result = data.get("current_result", data)

        return {
            "status": "refinement_needed",
            "interaction_id": str(uuid.uuid4()),
            "iteration": iteration,
            "current_result": current_result,
            "refinement_history": data.get("refinement_history", []),
            "ui_schema": {
                "type": "iterative_refinement",
                "fields": [
                    {
                        "name": "refined_result",
                        "type": "textarea",
                        "label": f"Refine Result (Iteration {iteration})",
                        "default": json.dumps(current_result, indent=2),
                        "required": True
                    },
                    {
                        "name": "refinement_notes",
                        "type": "textarea",
                        "label": "What changes did you make?",
                        "required": False
                    },
                    {
                        "name": "continue_refining",
                        "type": "checkbox",
                        "label": "Continue to next iteration?",
                        "default": False
                    }
                ]
            }
        }
```

## HITL Infrastructure Components

### 1. Pipeline State Manager

```python
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

class PipelineStateManager:
    def __init__(self, db_manager, cache_service=None):
        self.db_manager = db_manager
        self.cache_service = cache_service

    async def save_pipeline_state(
        self,
        interaction_id: str,
        pipeline_name: str,
        step_name: str,
        data: Dict[str, Any],
        timeout_seconds: int = 3600
    ):
        """Save pipeline state for later resumption"""
        state_record = {
            "interaction_id": interaction_id,
            "pipeline_name": pipeline_name,
            "step_name": step_name,
            "data": json.dumps(data),
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(seconds=timeout_seconds),
            "status": "pending"
        }

        await self.db_manager.execute_query("""
            INSERT INTO pipeline_states
            (interaction_id, pipeline_name, step_name, data, created_at, expires_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            state_record["interaction_id"],
            state_record["pipeline_name"],
            state_record["step_name"],
            state_record["data"],
            state_record["created_at"],
            state_record["expires_at"],
            state_record["status"]
        ))

        # Also cache for faster access
        if self.cache_service:
            await self.cache_service.set(
                f"pipeline_state:{interaction_id}",
                state_record,
                timeout_seconds
            )

    async def get_pipeline_state(self, interaction_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve pipeline state"""
        # Try cache first
        if self.cache_service:
            cached_state = await self.cache_service.get(f"pipeline_state:{interaction_id}")
            if cached_state:
                return cached_state

        # Fallback to database
        result = await self.db_manager.fetch_one("""
            SELECT * FROM pipeline_states
            WHERE interaction_id = ? AND status = 'pending' AND expires_at > ?
        """, (interaction_id, datetime.now()))

        if result:
            return {
                "interaction_id": result["interaction_id"],
                "pipeline_name": result["pipeline_name"],
                "step_name": result["step_name"],
                "data": json.loads(result["data"]),
                "created_at": result["created_at"],
                "expires_at": result["expires_at"],
                "status": result["status"]
            }

        return None

    async def resume_pipeline(
        self,
        interaction_id: str,
        human_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resume pipeline with human input"""
        state = await self.get_pipeline_state(interaction_id)

        if not state:
            raise ValueError(f"Pipeline state not found for interaction {interaction_id}")

        # Mark as completed
        await self.db_manager.execute_query("""
            UPDATE pipeline_states
            SET status = 'completed', completed_at = ?, human_input = ?
            WHERE interaction_id = ?
        """, (datetime.now(), json.dumps(human_input), interaction_id))

        # Merge human input with original data
        merged_data = {**state["data"], **human_input}

        # Continue pipeline execution
        return await self._continue_pipeline_execution(
            state["pipeline_name"],
            state["step_name"],
            merged_data
        )
```

### 2. Human Interaction API

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional

class HumanInputRequest(BaseModel):
    interaction_id: str
    user_input: Dict[str, Any]
    user_id: Optional[str] = None

class InteractionResponse(BaseModel):
    interaction_id: str
    status: str
    ui_schema: Dict[str, Any]
    data: Dict[str, Any]
    created_at: datetime
    expires_at: datetime

app = FastAPI()

@app.get("/hitl/pending/{user_id}")
async def get_pending_interactions(user_id: str) -> List[InteractionResponse]:
    """Get all pending interactions for a user"""
    result = await db_manager.fetch_all("""
        SELECT ps.*, pa.user_id
        FROM pipeline_states ps
        JOIN pipeline_assignments pa ON ps.interaction_id = pa.interaction_id
        WHERE pa.user_id = ? AND ps.status = 'pending' AND ps.expires_at > ?
        ORDER BY ps.created_at
    """, (user_id, datetime.now()))

    return [InteractionResponse(**row) for row in result]

@app.post("/hitl/respond")
async def respond_to_interaction(
    request: HumanInputRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, str]:
    """Submit human response to interaction"""

    # Validate interaction exists and is pending
    state = await state_manager.get_pipeline_state(request.interaction_id)
    if not state:
        raise HTTPException(status_code=404, detail="Interaction not found")

    # Process response in background
    background_tasks.add_task(
        state_manager.resume_pipeline,
        request.interaction_id,
        request.user_input
    )

    return {"status": "accepted", "message": "Response submitted successfully"}

@app.get("/hitl/status/{interaction_id}")
async def get_interaction_status(interaction_id: str) -> Dict[str, Any]:
    """Get status of specific interaction"""
    state = await state_manager.get_pipeline_state(interaction_id)

    if not state:
        raise HTTPException(status_code=404, detail="Interaction not found")

    return {
        "interaction_id": interaction_id,
        "status": state["status"],
        "created_at": state["created_at"],
        "expires_at": state["expires_at"]
    }
```

### 3. WebSocket Real-time Updates

```python
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import json

class HITLWebSocketManager:
    def __init__(self):
        self.user_connections: Dict[str, List[WebSocket]] = {}
        self.interaction_subscribers: Dict[str, List[WebSocket]] = {}

    async def connect_user(self, websocket: WebSocket, user_id: str):
        """Connect user to HITL updates"""
        await websocket.accept()

        if user_id not in self.user_connections:
            self.user_connections[user_id] = []

        self.user_connections[user_id].append(websocket)

        # Send pending interactions on connect
        pending = await get_pending_interactions(user_id)
        await websocket.send_text(json.dumps({
            "type": "pending_interactions",
            "data": [interaction.dict() for interaction in pending]
        }))

    async def disconnect_user(self, websocket: WebSocket, user_id: str):
        """Disconnect user"""
        if user_id in self.user_connections:
            if websocket in self.user_connections[user_id]:
                self.user_connections[user_id].remove(websocket)

    async def notify_new_interaction(self, user_id: str, interaction_data: Dict[str, Any]):
        """Notify user of new interaction"""
        if user_id in self.user_connections:
            message = json.dumps({
                "type": "new_interaction",
                "data": interaction_data
            })

            # Send to all user's connections
            disconnected = []
            for websocket in self.user_connections[user_id]:
                try:
                    await websocket.send_text(message)
                except:
                    disconnected.append(websocket)

            # Clean up disconnected websockets
            for ws in disconnected:
                self.user_connections[user_id].remove(ws)

websocket_manager = HITLWebSocketManager()

@app.websocket("/ws/hitl/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await websocket_manager.connect_user(websocket, user_id)

    try:
        while True:
            # Keep connection alive and handle any client messages
            data = await websocket.receive_text()

            # Handle ping/pong or other client messages
            if data == "ping":
                await websocket.send_text("pong")

    except WebSocketDisconnect:
        await websocket_manager.disconnect_user(websocket, user_id)
```

### 4. Frontend Integration Examples

#### React Component for HITL Interactions

```typescript
import React, { useState, useEffect } from 'react';

interface HITLInteraction {
  interaction_id: string;
  status: string;
  ui_schema: any;
  data: any;
  created_at: string;
  expires_at: string;
}

const HITLDashboard: React.FC<{ userId: string }> = ({ userId }) => {
  const [interactions, setInteractions] = useState<HITLInteraction[]>([]);
  const [websocket, setWebsocket] = useState<WebSocket | null>(null);

  useEffect(() => {
    // Connect to WebSocket for real-time updates
    const ws = new WebSocket(`ws://localhost:8000/ws/hitl/${userId}`);

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);

      if (message.type === 'pending_interactions') {
        setInteractions(message.data);
      } else if (message.type === 'new_interaction') {
        setInteractions(prev => [...prev, message.data]);
      }
    };

    setWebsocket(ws);

    return () => {
      ws.close();
    };
  }, [userId]);

  const handleInteractionResponse = async (
    interactionId: string,
    response: any
  ) => {
    await fetch('/hitl/respond', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        interaction_id: interactionId,
        user_input: response
      })
    });

    // Remove from pending list
    setInteractions(prev =>
      prev.filter(i => i.interaction_id !== interactionId)
    );
  };

  return (
    <div className="hitl-dashboard">
      <h2>Pending Human Interactions ({interactions.length})</h2>

      {interactions.map(interaction => (
        <InteractionCard
          key={interaction.interaction_id}
          interaction={interaction}
          onRespond={handleInteractionResponse}
        />
      ))}
    </div>
  );
};
```

### 5. Advanced HITL Patterns

#### Time-based Decision Making

```python
class TimeBasedDecisionStep(Step):
    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        decision_timeout = self.config.get("decision_timeout", 300)  # 5 minutes
        default_action = self.config.get("default_action", "proceed")

        interaction_id = str(uuid.uuid4())

        # Start async timer
        asyncio.create_task(
            self._handle_timeout(interaction_id, default_action, decision_timeout)
        )

        return {
            "status": "time_sensitive_decision",
            "interaction_id": interaction_id,
            "timeout_seconds": decision_timeout,
            "default_action": default_action,
            "ui_schema": {
                "type": "urgent_decision",
                "fields": [
                    {
                        "name": "decision",
                        "type": "radio",
                        "options": ["proceed", "abort", "modify"],
                        "required": True
                    },
                    {
                        "name": "urgency_acknowledged",
                        "type": "checkbox",
                        "label": f"I understand this decision expires in {decision_timeout//60} minutes",
                        "required": True
                    }
                ]
            },
            "data": data
        }

    async def _handle_timeout(self, interaction_id: str, default_action: str, timeout: int):
        await asyncio.sleep(timeout)

        # Check if decision was made
        state = await state_manager.get_pipeline_state(interaction_id)
        if state and state["status"] == "pending":
            # Apply default action
            await state_manager.resume_pipeline(
                interaction_id,
                {"decision": default_action, "timeout_applied": True}
            )
```

#### Multi-stakeholder Approval

```python
class MultiApprovalStep(Step):
    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        approvers = self.config.get("approvers", [])
        approval_type = self.config.get("approval_type", "all")  # all, majority, any

        approval_id = str(uuid.uuid4())

        # Create approval records for each approver
        for approver in approvers:
            await self._create_approval_request(approval_id, approver, data)

        return {
            "status": "multi_approval_pending",
            "approval_id": approval_id,
            "approvers": approvers,
            "approval_type": approval_type,
            "data": data
        }

    async def _create_approval_request(self, approval_id: str, approver: str, data: Dict[str, Any]):
        # Create individual approval request
        interaction_id = f"{approval_id}_{approver}"

        await state_manager.save_pipeline_state(
            interaction_id,
            self.pipeline_name,
            self.name,
            {
                "approval_id": approval_id,
                "approver": approver,
                "approval_data": data
            }
        )

        # Notify approver
        await websocket_manager.notify_new_interaction(approver, {
            "interaction_id": interaction_id,
            "type": "approval_request",
            "ui_schema": {
                "type": "approval",
                "fields": [
                    {
                        "name": "decision",
                        "type": "radio",
                        "options": ["approve", "reject"],
                        "required": True
                    },
                    {
                        "name": "comments",
                        "type": "textarea",
                        "label": "Comments (optional)"
                    }
                ]
            }
        })
```

## Database Schema for HITL

```sql
-- Pipeline states for paused executions
CREATE TABLE pipeline_states (
    interaction_id TEXT PRIMARY KEY,
    pipeline_name TEXT NOT NULL,
    step_name TEXT NOT NULL,
    data TEXT NOT NULL,  -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'completed', 'expired', 'cancelled')),
    completed_at TIMESTAMP,
    human_input TEXT  -- JSON
);

-- User assignments for interactions
CREATE TABLE pipeline_assignments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    role TEXT DEFAULT 'reviewer',
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (interaction_id) REFERENCES pipeline_states(interaction_id)
);

-- Approval tracking
CREATE TABLE multi_approvals (
    approval_id TEXT NOT NULL,
    approver_id TEXT NOT NULL,
    decision TEXT CHECK (decision IN ('approve', 'reject', 'pending')),
    comments TEXT,
    decided_at TIMESTAMP,
    PRIMARY KEY (approval_id, approver_id)
);

-- Indexes for performance
CREATE INDEX idx_pipeline_states_status ON pipeline_states(status);
CREATE INDEX idx_pipeline_states_expires ON pipeline_states(expires_at);
CREATE INDEX idx_pipeline_assignments_user ON pipeline_assignments(user_id);
CREATE INDEX idx_multi_approvals_id ON multi_approvals(approval_id);
```

This comprehensive HITL system provides:

1. **Multiple interaction patterns** for different use cases
2. **State management** for paused pipelines
3. **Real-time notifications** via WebSocket
4. **RESTful API** for human responses
5. **Frontend integration** examples
6. **Database persistence** for reliability
7. **Time-based decisions** with automatic fallbacks
8. **Multi-stakeholder approvals** with flexible voting

The system is designed to be flexible, scalable, and integrate seamlessly with the existing IA Modules pipeline architecture.