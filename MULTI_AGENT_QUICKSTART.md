# Multi-Agent Feature Quick Reference

## 🚀 Quick Start

### 1. Start the Backend
```bash
cd showcase_app/backend
python main.py
# Server starts on http://localhost:5555
```

### 2. Start the Frontend
```bash
cd showcase_app/frontend
npm run dev
# Frontend starts on http://localhost:5173
```

### 3. Access Multi-Agent Dashboard
Navigate to: `http://localhost:5173/multi-agent`

## 📋 Workflow Templates

Choose from 12 pre-built templates:

| Template | Use Case | Agents | Complexity |
|----------|----------|--------|------------|
| Simple Sequence | Linear workflow | 3 | ⭐ |
| Feedback Loop | Iterative improvement | 2 | ⭐⭐ |
| Conditional Routing | Dynamic paths | 3 | ⭐⭐ |
| Complex Workflow | Multi-stage pipeline | 5 | ⭐⭐⭐ |
| Customer Service | Support automation | 5 | ⭐⭐⭐ |
| Code Review | Multi-perspective review | 5 | ⭐⭐⭐ |
| Content Pipeline | Content generation | 5 | ⭐⭐⭐ |
| Data Analysis | ETL + Analysis | 6 | ⭐⭐⭐⭐ |
| Debate System | Consensus building | 5 | ⭐⭐⭐⭐ |
| Q&A System | Question answering | 5 | ⭐⭐⭐ |
| Creative Writing | Story generation | 5 | ⭐⭐⭐⭐ |
| Research Paper | Academic writing | 6 | ⭐⭐⭐⭐⭐ |

## 🔌 API Quick Reference

### Base URL
```
http://localhost:5555/api/multi-agent
```

### Create Workflow
```bash
POST /workflows
{
  "workflow_id": "my_workflow",
  "agents": [...],
  "edges": [...]
}
```

### Execute Workflow
```bash
POST /workflows/{id}/executions
{
  "start_agent": "agent1",
  "initial_data": {}
}
```

### Save Workflow
```bash
POST /workflows/{id}/save
{
  "name": "My Workflow",
  "description": "Description"
}
```

### Load Workflow
```bash
POST /workflows/{id}/load
```

### Export Workflow
```bash
GET /workflows/{id}/export
```

### Import Workflow
```bash
POST /workflows/import
{
  "workflow_id": "...",
  "agents": [...],
  "edges": [...]
}
```

### List Saved Workflows
```bash
GET /workflows/saved
```

### WebSocket Connection
```javascript
ws://localhost:5555/api/multi-agent/ws/{workflow_id}
```

## 🔧 Code Examples

### Python - Create and Execute
```python
from showcase_app.backend.services.multi_agent_service import MultiAgentService

service = MultiAgentService()

# Create workflow
await service.create_workflow(
    workflow_id="demo",
    agents=[
        {"id": "agent1", "role": "planner", "description": "Plans"},
        {"id": "agent2", "role": "executor", "description": "Executes"}
    ],
    edges=[
        {"from": "agent1", "to": "agent2"}
    ]
)

# Execute
result = await service.execute_workflow(
    workflow_id="demo",
    start_agent="agent1",
    initial_data={"task": "Build a feature"}
)

print(result["communication_log"])
```

### JavaScript - WebSocket Monitoring
```javascript
const ws = new WebSocket('ws://localhost:5555/api/multi-agent/ws/demo');

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  
  switch(message.type) {
    case 'agent_start':
      console.log(`Agent ${message.data.agent} started`);
      break;
    case 'agent_complete':
      console.log(`Agent ${message.data.agent} completed in ${message.data.duration_seconds}s`);
      break;
    case 'agent_error':
      console.error(`Agent ${message.data.agent} error: ${message.data.error}`);
      break;
  }
};

// Keepalive
setInterval(() => {
  ws.send(JSON.stringify({ type: 'ping' }));
}, 30000);
```

### cURL - Save and Load
```bash
# Save workflow
curl -X POST http://localhost:5555/api/multi-agent/workflows/demo/save \
  -H "Content-Type: application/json" \
  -d '{"name": "Demo Workflow", "description": "A demo"}'

# Load workflow
curl -X POST http://localhost:5555/api/multi-agent/workflows/demo/load

# List saved
curl http://localhost:5555/api/multi-agent/workflows/saved
```

## 🧪 Testing

### Run All Tests
```bash
cd ia_modules
python -m pytest tests/unit/test_multi_agent_service.py -v
```

### Run Specific Test Category
```bash
# Workflow creation tests
pytest tests/unit/test_multi_agent_service.py -k "create" -v

# Persistence tests
pytest tests/unit/test_multi_agent_service.py -k "save or load" -v

# WebSocket tests
pytest tests/unit/test_multi_agent_service.py -k "websocket" -v
```

### Test with Coverage
```bash
pytest tests/unit/test_multi_agent_service.py --cov=showcase_app/backend/services
```

## 📊 Event Types

### WebSocket Events

| Event Type | Triggered When | Data Fields |
|------------|----------------|-------------|
| `agent_start` | Agent begins execution | `agent`, `input_data`, `timestamp` |
| `agent_complete` | Agent finishes successfully | `agent`, `output_data`, `duration_seconds`, `timestamp` |
| `agent_error` | Agent encounters error | `agent`, `error`, `timestamp` |
| `pong` | Response to ping | `type: "pong"` |

## 🗂️ File Structure

```
ia_modules/
├── showcase_app/
│   ├── backend/
│   │   ├── api/
│   │   │   └── multi_agent.py          # API endpoints
│   │   └── services/
│   │       └── multi_agent_service.py  # Core service
│   └── frontend/
│       └── src/
│           └── components/
│               └── MultiAgent/         # UI components
├── tests/
│   └── unit/
│       └── test_multi_agent_service.py # Test suite
└── workflows/                          # Saved workflows (auto-created)
```

## 🐛 Troubleshooting

### Issue: Tests failing with "StateManager has no attribute 'get_snapshot'"
**Solution:** Use `await state.snapshot()` instead of `state.get_snapshot()`

### Issue: WebSocket connection refused
**Solution:** Ensure backend is running on port 5555:
```bash
cd showcase_app/backend
python main.py
```

### Issue: "Workflow not found" error
**Solution:** Create workflow before executing:
```python
await service.create_workflow(workflow_id="...", agents=[...], edges=[...])
```

### Issue: JSON serialization error with Edge objects
**Solution:** Access edge properties correctly:
```python
edge.to  # ✅ Correct
edge.to_agent  # ❌ Wrong
```

## 📈 Performance Tips

1. **Use WebSocket for real-time updates** instead of polling
2. **Save frequently used workflows** to avoid recreation
3. **Export workflows as backups** before major changes
4. **Monitor execution logs** for bottlenecks
5. **Use feedback loops sparingly** (they increase execution time)

## 🔒 Security Notes

- Workflows stored in `./workflows/` directory (local filesystem)
- No authentication on WebSocket connections (add in production)
- Sanitize user input in workflow configurations
- Validate imported workflows before execution

## 📚 Additional Resources

- **Full Documentation:** `MULTI_AGENT_ENHANCEMENTS.md`
- **API Reference:** `MULTI_AGENT_API.md`
- **Architecture Review:** `MULTI_AGENT_REVIEW.md`
- **Implementation Summary:** `MULTI_AGENT_SUMMARY.md`

## ⚡ Power User Tips

### Chain Multiple Workflows
```python
# Workflow 1: Data collection
result1 = await service.execute_workflow("data_collector", "scraper", {})

# Workflow 2: Data processing (use result1)
result2 = await service.execute_workflow("data_processor", "cleaner", result1["result"])

# Workflow 3: Report generation
result3 = await service.execute_workflow("report_gen", "writer", result2["result"])
```

### Template Customization
```python
# Load template
template = (await fetch_templates())["templates"]["customer_service"]

# Customize
template["agents"].append({
    "id": "escalation_handler",
    "role": "escalation",
    "description": "Handles escalations"
})

# Create custom workflow
await service.create_workflow(
    workflow_id="custom_support",
    **template
)
```

### Batch Operations
```python
# Save multiple workflows
workflows = ["wf1", "wf2", "wf3"]
for wf_id in workflows:
    await service.save_workflow(wf_id, f"Workflow {wf_id}", "Batch save")

# Load all saved workflows
saved = await service.list_saved_workflows()
for wf in saved:
    await service.load_workflow(wf["workflow_id"])
```

---
*Quick Reference Guide - v1.0*
*Last Updated: October 24, 2025*
