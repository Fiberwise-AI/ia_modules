# Showcase App Improvements

## Current Issues

### 1. Pipeline Visualization
- ❌ Parallel pipeline shows steps sequentially (no visual branching)
- ❌ Conditional pipeline doesn't show branch paths
- ❌ No graph visualization of pipeline flow
- ❌ Missing step dependencies/connections

### 2. Execution Details
- ❌ No telemetry spans displayed
- ❌ No checkpoint data shown
- ❌ Missing step-level timing breakdown
- ❌ No retry/fallback indicators
- ❌ No parallel execution visualization

### 3. Test Inputs
- ⚠️ Inputs work but could be more descriptive
- ⚠️ No explanation of what each pipeline demonstrates

### 4. Real-Time Updates
- ❌ WebSocket exists but not fully integrated
- ❌ No live progress indicators during execution
- ❌ No streaming step updates

### 5. Missing ia_modules Features
- ❌ Checkpointing not showcased
- ❌ Reliability metrics not visible per-execution
- ❌ Benchmarking not integrated
- ❌ Scheduler not shown
- ❌ Loop detection not demonstrated
- ❌ HITL pipeline not executable

## Required Improvements

### Phase 1: Visualization (HIGH PRIORITY)

**Pipeline Flow Graph**
```javascript
// Use ReactFlow or similar to render:
- Nodes for each step
- Edges showing data flow
- Parallel branches side-by-side
- Conditional branches with decision nodes
- Loop indicators
```

**Execution Visualization**
- Color-code steps by status (pending/running/completed/failed)
- Show active step highlighted
- Display parallel execution in real-time
- Animate transitions between steps

### Phase 2: Telemetry Integration

**Tracing Display**
```javascript
<TracingPanel execution={execution}>
  {execution.spans.map(span => (
    <Span
      name={span.name}
      duration={span.duration}
      attributes={span.attributes}
      children={span.children}
    />
  ))}
</TracingPanel>
```

**What to Show:**
- Span tree with parent/child relationships
- Duration per span
- Attributes (inputs/outputs)
- Status codes
- Timestamps

### Phase 3: Checkpoint Integration

**Checkpoint UI**
```javascript
<CheckpointPanel execution={execution}>
  <CheckpointList checkpoints={execution.checkpoints}>
    {checkpoints.map(cp => (
      <Checkpoint
        id={cp.id}
        state={cp.state}
        timestamp={cp.created_at}
        onResume={() => resumeFromCheckpoint(cp.id)}
      />
    ))}
  </CheckpointList>
</CheckpointPanel>
```

**Features:**
- List all checkpoints for execution
- View checkpoint state/data
- Resume from checkpoint button
- Checkpoint creation timestamps

### Phase 4: Step Details

**Step Execution Panel**
```javascript
<StepDetails step={step}>
  <StepHeader
    name={step.name}
    status={step.status}
    duration={step.duration_ms}
  />
  <StepInput data={step.input_data} />
  <StepOutput data={step.output_data} />
  <StepMetrics
    retries={step.retries}
    tokens={step.tokens}
    cost={step.cost}
  />
  <StepSpans spans={step.spans} />
</StepDetails>
```

### Phase 5: Real-Time Progress

**WebSocket Integration**
- Live execution progress bar
- Step-by-step updates as they complete
- Real-time metrics
- Live logs streaming

### Phase 6: Feature Showcase

**Add Pages/Sections:**
- Benchmarking - compare pipeline runs
- Scheduling - view scheduled jobs
- Checkpoints - pause/resume demos
- Reliability - show metrics (SR, CR, PC, etc.)
- HITL - human-in-the-loop demo with approval UI

## Implementation Priority

1. **CRITICAL** - Fix parallel pipeline visualization
2. **CRITICAL** - Add telemetry spans display
3. **HIGH** - Add checkpoint UI
4. **HIGH** - Show step-level details
5. **MEDIUM** - Real-time WebSocket updates
6. **MEDIUM** - Benchmarking page
7. **LOW** - Scheduler page
8. **LOW** - HITL demo

## Technical Requirements

### Backend Changes Needed

**Add to execution response:**
```python
{
  "job_id": "...",
  "status": "completed",
  "steps": [...],
  "output_data": {...},

  # Add these:
  "spans": [...],  # Telemetry spans
  "checkpoints": [...],  # Checkpoint data
  "graph": {  # Pipeline structure
    "nodes": [...],
    "edges": [...]
  }
}
```

### Frontend Packages Needed

```json
{
  "reactflow": "^11.0.0",  // For pipeline visualization
  "recharts": "^2.0.0",    // For metrics charts
  "lucide-react": "^0.263.0"  // Already have
}
```

### API Endpoints Needed

```
GET /api/executions/{job_id}/spans
GET /api/executions/{job_id}/checkpoints
POST /api/executions/{job_id}/resume/{checkpoint_id}
GET /api/benchmarks
GET /api/scheduled-jobs
```

## Example Visualizations

### Parallel Pipeline Graph
```
         ┌──────────────┐
         │ Data Splitter│
         └──────┬───────┘
                │
        ┌───────┼───────┐
        │       │       │
   ┌────▼───┐ ┌▼────┐ ┌▼────┐
   │Stream 1│ │Str 2│ │Str 3│ (parallel)
   └────┬───┘ └┬────┘ └┬────┘
        │      │       │
        └──────┼───────┘
               │
        ┌──────▼────────┐
        │Result Merger  │
        └───────────────┘
```

### Conditional Pipeline Graph
```
     ┌──────────────┐
     │Data Ingestor │
     └──────┬───────┘
            │
     ┌──────▼────────┐
     │Quality Checker│
     └──────┬────────┘
            │
     ┌──────▼────────┐ quality >= 0.8?
     │   Decision    │
     └──┬────────┬───┘
        │        │
   HIGH │        │ LOW
        │        │
   ┌────▼───┐  ┌▼────┐
   │High  │  │Low  │
   │Proc  │  │Proc │
   └────┬───┘  └┬────┘
        │       │
        └───┬───┘
            │
     ┌──────▼─────────┐
     │Results Aggregat│
     └────────────────┘
```

## Success Criteria

✅ Parallel branches visible in UI
✅ Conditional branches visible in UI
✅ Telemetry spans displayed
✅ Checkpoint data accessible
✅ Step timing breakdown shown
✅ Real-time execution progress
✅ All ia_modules features demonstrated

## Next Steps

1. Add ReactFlow dependency
2. Create PipelineGraph component
3. Update backend to return graph structure
4. Add spans endpoint
5. Create TracingPanel component
6. Add CheckpointPanel
7. Integrate WebSocket for real-time
8. Add benchmarking page

---

**Goal**: Make showcase app truly showcase ALL ia_modules capabilities with visual, interactive demonstrations.
