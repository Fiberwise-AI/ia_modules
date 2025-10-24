# Multi-Agent API Documentation

## REST Endpoints

All endpoints follow RESTful conventions with proper resource naming and HTTP methods.

### Workflows

#### Create Workflow
```
POST /api/multi-agent/workflows
```
Creates a new multi-agent workflow with agents and edges.

**Request Body:**
```json
{
  "workflow_id": "my_workflow",
  "agents": [
    {
      "id": "agent1",
      "role": "planner",
      "description": "Creates execution plan"
    }
  ],
  "edges": [
    {
      "from": "agent1",
      "to": "agent2",
      "condition": "optional_condition"
    }
  ],
  "feedback_loops": [
    {
      "from": "agent1",
      "to": "agent2",
      "max_iterations": 3
    }
  ]
}
```

#### List Workflows
```
GET /api/multi-agent/workflows
```
Returns all created workflows.

**Response:**
```json
{
  "workflows": [...],
  "total": 5
}
```

#### Get Workflow State
```
GET /api/multi-agent/workflows/{workflow_id}/state
```
Returns current state, statistics, and agent information for a specific workflow.

**Response:**
```json
{
  "workflow_id": "my_workflow",
  "status": "ready",
  "agent_stats": {
    "agent1": {
      "executions": 5,
      "total_duration": 12.5,
      "average_duration": 2.5
    }
  },
  "execution_count": 10,
  "last_execution": "2025-10-24T10:30:00Z"
}
```

#### Get Workflow Communications
```
GET /api/multi-agent/workflows/{workflow_id}/communications?execution_id={optional}
```
Returns communication logs for a workflow. Optionally filter by execution ID.

**Response:**
```json
{
  "workflow_id": "my_workflow",
  "execution_id": "exec_123",
  "communications": [
    {
      "timestamp": "2025-10-24T10:30:00Z",
      "type": "agent_activated",
      "agent": "agent1",
      "data": {...},
      "duration": 2.5
    }
  ]
}
```

### Executions

#### Execute Workflow (Create Execution)
```
POST /api/multi-agent/workflows/{workflow_id}/executions
```
Executes a workflow, creating a new execution record.

**Request Body:**
```json
{
  "start_agent": "planner",
  "initial_data": {
    "task": "Demo task"
  }
}
```

**Response:**
```json
{
  "execution_id": "exec_123",
  "workflow_id": "my_workflow",
  "status": "completed",
  "result": {...},
  "communications": [...]
}
```

#### List Executions
```
GET /api/multi-agent/executions?limit=50
```
Lists recent executions across all workflows.

**Response:**
```json
{
  "executions": [
    {
      "execution_id": "exec_123",
      "workflow_id": "my_workflow",
      "timestamp": "2025-10-24T10:30:00Z",
      "status": "completed"
    }
  ],
  "total": 25
}
```

### Templates

#### Get Workflow Templates
```
GET /api/multi-agent/templates
```
Returns pre-built workflow templates.

**Response:**
```json
{
  "templates": {
    "simple_sequence": {
      "name": "Simple Sequence",
      "description": "Linear workflow",
      "agents": [...],
      "edges": [...]
    }
  }
}
```

## REST Conventions Applied

1. ✅ **Resources as nouns**: `/workflows`, `/executions`, `/templates`
2. ✅ **HTTP methods for actions**: 
   - `POST` for creating
   - `GET` for retrieving
   - Resource nesting: `/workflows/{id}/executions`
3. ✅ **Proper status codes**:
   - `200` for successful GET
   - `404` for not found
   - `500` for server errors
4. ✅ **Consistent response formats**: All responses return objects with descriptive keys
5. ✅ **Query parameters for filtering**: `?limit=50`, `?execution_id=123`

## Changes Made

### Before (Non-RESTful):
- ❌ `POST /workflows/execute` - Action verb in URL
- ❌ `GET /history` - Vague resource name

### After (RESTful):
- ✅ `POST /workflows/{workflow_id}/executions` - Creates execution resource
- ✅ `GET /executions` - Lists execution resources

## Frontend Integration

The frontend components have been updated to use the corrected endpoints:

```typescript
// Execute workflow
const response = await fetch(
  `${API_BASE}/workflows/${workflowId}/executions`,
  {
    method: 'POST',
    body: JSON.stringify({
      start_agent: 'planner',
      initial_data: { task: 'Demo' }
    })
  }
);
```
