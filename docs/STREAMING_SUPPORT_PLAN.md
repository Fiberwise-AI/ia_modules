# Streaming Support - Implementation Plan

## Overview

This document outlines the implementation plan for real-time streaming of pipeline outputs in IA Modules v0.0.4. This feature will enable consumers to receive pipeline step outputs as they are generated, rather than waiting for the entire pipeline to complete.

## Motivation

### Current Limitations
- **Blocking Execution**: Clients must wait for entire pipeline completion
- **No Progress Feedback**: Users cannot see intermediate results or progress
- **Poor UX for Long-Running Pipelines**: Multi-minute pipelines provide no feedback
- **Large Result Sets**: Memory-intensive to accumulate all outputs before returning
- **LLM Token Streaming**: Cannot stream LLM-generated tokens to users in real-time

### Use Cases
1. **Real-time LLM Output**: Stream AI-generated text token-by-token
2. **Progress Monitoring**: Show step-by-step progress for long pipelines
3. **Early Error Detection**: Alert users immediately when steps fail
4. **Data Processing Pipelines**: Stream results as they're computed
5. **Human-in-the-Loop**: Show real-time status while waiting for human input
6. **Debugging**: Live visibility into pipeline execution for troubleshooting

## Architecture

### Core Components

#### 1. **StreamingContext** (New)
```python
class StreamingContext:
    """Manages streaming output for a pipeline execution"""

    def __init__(self, execution_id: str):
        self.execution_id = execution_id
        self.subscribers: List[StreamSubscriber] = []
        self.buffer: Queue = asyncio.Queue()

    async def emit(self, event: StreamEvent):
        """Emit an event to all subscribers"""

    async def subscribe(self, subscriber: StreamSubscriber):
        """Add a new subscriber to receive events"""
```

#### 2. **StreamEvent** (New)
```python
@dataclass
class StreamEvent:
    """A single streaming event from pipeline execution"""

    event_type: str  # step_start, step_output, step_complete, step_error, pipeline_complete
    execution_id: str
    step_id: Optional[str]
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
```

Event types:
- `pipeline_start`: Pipeline execution begins
- `step_start`: Step execution begins
- `step_progress`: Step emits progress update (e.g., LLM tokens)
- `step_output`: Step produces output data
- `step_complete`: Step finishes successfully
- `step_error`: Step encounters error
- `pipeline_complete`: Pipeline finishes
- `pipeline_error`: Pipeline-level error
- `checkpoint_saved`: Checkpoint created

#### 3. **StreamSubscriber** (Interface)
```python
class StreamSubscriber(ABC):
    """Abstract base for streaming subscribers"""

    @abstractmethod
    async def on_event(self, event: StreamEvent):
        """Handle a streaming event"""
        pass
```

Implementations:
- `WebSocketSubscriber`: Stream to WebSocket connections
- `SSESubscriber`: Stream via Server-Sent Events
- `CallbackSubscriber`: Stream to async callback function
- `FileSubscriber`: Write events to file (debugging)
- `MetricsSubscriber`: Update reliability metrics in real-time

#### 4. **Modified Pipeline & Step Classes**

```python
class Step:
    """Enhanced with streaming support"""

    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute step - can now emit streaming events"""

        # Emit progress updates
        if self.streaming_context:
            await self.streaming_context.emit(StreamEvent(
                event_type="step_progress",
                data={"progress": 0.5, "message": "Processing..."}
            ))
```

```python
class Pipeline:
    """Enhanced with streaming context"""

    def __init__(self, ..., streaming_context: Optional[StreamingContext] = None):
        self.streaming_context = streaming_context

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute with streaming support"""

        if self.streaming_context:
            await self.streaming_context.emit(
                StreamEvent(event_type="pipeline_start", ...)
            )
```

### Integration Points

#### 1. **FastAPI WebSocket Integration**
```python
# api/routes/streaming.py

@router.websocket("/ws/pipeline/{execution_id}")
async def pipeline_stream(
    websocket: WebSocket,
    execution_id: str,
    user_id: int = Depends(get_current_user)
):
    """Stream pipeline execution over WebSocket"""

    await websocket.accept()
    subscriber = WebSocketSubscriber(websocket)

    # Register subscriber with execution's streaming context
    streaming_context = get_streaming_context(execution_id)
    await streaming_context.subscribe(subscriber)

    try:
        # Keep connection alive
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await streaming_context.unsubscribe(subscriber)
```

#### 2. **Server-Sent Events (SSE) Integration**
```python
@router.get("/stream/pipeline/{execution_id}")
async def pipeline_stream_sse(
    execution_id: str,
    user_id: int = Depends(get_current_user)
):
    """Stream pipeline execution via SSE"""

    async def event_generator():
        subscriber = SSESubscriber()
        streaming_context = get_streaming_context(execution_id)

        await streaming_context.subscribe(subscriber)

        async for event in subscriber:
            yield {
                "event": event.event_type,
                "data": json.dumps(event.data),
                "id": str(event.timestamp.timestamp())
            }

    return EventSourceResponse(event_generator())
```

#### 3. **Runner Service Integration**
```python
# api/services/runner_service.py

async def execute_pipeline_with_streaming(
    pipeline_config: Dict[str, Any],
    input_data: Dict[str, Any],
    user_id: int,
    execution_id: str
) -> StreamingContext:
    """Execute pipeline with streaming support"""

    # Create streaming context
    streaming_context = StreamingContext(execution_id)

    # Add metrics subscriber for real-time reliability tracking
    metrics_subscriber = MetricsSubscriber(reliability_metrics)
    await streaming_context.subscribe(metrics_subscriber)

    # Create pipeline with streaming
    services = get_services()
    services.register('streaming_context', streaming_context)

    pipeline = create_pipeline_from_json(pipeline_config, services)

    # Run in background task
    asyncio.create_task(pipeline.run(input_data))

    return streaming_context
```

### Data Flow

```
┌─────────────┐
│  Pipeline   │
│  Execution  │
└──────┬──────┘
       │ emit events
       ▼
┌──────────────────┐
│ StreamingContext │
└──────┬───────────┘
       │ fan out
       ├─────────┬─────────┬──────────┐
       ▼         ▼         ▼          ▼
  WebSocket    SSE    Callback   Metrics
  Subscriber   Sub     Sub        Sub
       │         │         │          │
       ▼         ▼         ▼          ▼
   Browser   Browser   Queue    Reliability
   Client    Client             Tracker
```

## Implementation Phases

### Phase 1: Core Streaming Infrastructure (Week 1)
**Goal**: Build foundational streaming components

**Tasks**:
1. Create `StreamingContext` class
2. Define `StreamEvent` dataclass with all event types
3. Implement `StreamSubscriber` interface
4. Add streaming context to `Pipeline` class
5. Add streaming context to `Step` class
6. Implement event emission in pipeline execution flow
7. Unit tests for streaming components

**Files to Create**:
- `ia_modules/pipeline/streaming.py` - Core streaming classes
- `ia_modules/tests/unit/test_streaming.py` - Unit tests

**Files to Modify**:
- `ia_modules/pipeline/core.py` - Add streaming to Step
- `ia_modules/pipeline/runner.py` - Add streaming to Pipeline

### Phase 2: Subscriber Implementations (Week 2)
**Goal**: Build concrete subscriber implementations

**Tasks**:
1. Implement `WebSocketSubscriber`
2. Implement `SSESubscriber`
3. Implement `CallbackSubscriber`
4. Implement `FileSubscriber` (for debugging)
5. Implement `MetricsSubscriber` (real-time reliability tracking)
6. Add subscriber lifecycle management (subscribe/unsubscribe)
7. Handle subscriber errors gracefully
8. Unit tests for each subscriber

**Files to Create**:
- `ia_modules/pipeline/subscribers.py` - Subscriber implementations
- `ia_modules/tests/unit/test_subscribers.py` - Unit tests

### Phase 3: API Integration (Week 3)
**Goal**: Expose streaming via FastAPI endpoints

**Tasks**:
1. Create WebSocket endpoint for streaming
2. Create SSE endpoint for streaming
3. Integrate with existing `runner_service.py`
4. Add execution ID tracking for streaming contexts
5. Implement connection lifecycle management
6. Add authentication/authorization for streaming endpoints
7. Integration tests with real pipeline execution
8. Update WebSocket manager to support streaming

**Files to Create**:
- `app_knowledge_base/api/routes/streaming.py` - Streaming endpoints

**Files to Modify**:
- `app_knowledge_base/api/services/runner_service.py` - Streaming integration
- `app_knowledge_base/api/websockets.py` - Enhanced WebSocket support

### Phase 4: LLM Token Streaming (Week 4)
**Goal**: Enable real-time streaming of LLM-generated tokens

**Tasks**:
1. Modify `LLMProviderService` to support streaming
2. Implement token-by-token emission in LLM steps
3. Handle streaming for Google Gemini
4. Handle streaming for OpenAI
5. Handle streaming for Anthropic
6. Add buffering strategy for token accumulation
7. Tests for LLM streaming

**Files to Modify**:
- `ia_modules/pipeline/llm_provider_service.py` - Add streaming support

### Phase 5: Frontend Integration (Week 5)
**Goal**: Build UI components for consuming streams

**Tasks**:
1. Create `StreamingPipelineMonitor` Vue component
2. Implement WebSocket connection management
3. Display real-time step progress
4. Show streaming LLM output
5. Handle reconnection logic
6. Visual progress indicators
7. Error state handling

**Files to Create**:
- `app_knowledge_base/src/components/pipeline/streaming-monitor.js`

### Phase 6: Documentation & Examples (Week 6)
**Goal**: Complete documentation and examples

**Tasks**:
1. API documentation for streaming endpoints
2. Usage guide for streaming in pipelines
3. Example pipeline with streaming
4. Frontend integration guide
5. Performance tuning guide
6. Update FEATURES.md
7. Update API_REFERENCE.md

**Files to Create**:
- `ia_modules/docs/STREAMING_GUIDE.md`
- `ia_modules/tests/pipelines/streaming_example/`

**Files to Modify**:
- `ia_modules/docs/FEATURES.md`
- `ia_modules/docs/API_REFERENCE.md`

## Technical Considerations

### Performance
- **Event Buffering**: Queue events to prevent blocking pipeline execution
- **Backpressure**: Handle slow consumers gracefully
- **Memory Management**: Limit event buffer size, implement circular buffer
- **Connection Limits**: Cap max concurrent streaming connections per user

### Error Handling
- **Subscriber Failures**: Pipeline continues if subscriber fails
- **Network Issues**: Automatic reconnection for WebSocket/SSE
- **Partial Results**: Clear indication when stream terminates early
- **Timeout Handling**: Close stale streaming connections

### Security
- **Authentication**: Verify user identity for streaming endpoints
- **Authorization**: Ensure users can only stream their own executions
- **Rate Limiting**: Prevent abuse of streaming endpoints
- **Input Validation**: Validate execution IDs and parameters

### Reliability Integration
- **Real-time Metrics**: Update SR, CR, TCL, WCT as events stream
- **Event Replay**: Store streaming events for debugging
- **Checkpoint Integration**: Emit checkpoint events in stream
- **SLO Monitoring**: Track streaming latency and throughput

### Backward Compatibility
- **Optional Feature**: Streaming is opt-in, existing code unchanged
- **Fallback Mode**: Non-streaming execution remains default
- **Graceful Degradation**: If streaming fails, fall back to blocking execution

## Configuration

### Pipeline Configuration
```json
{
  "name": "Streaming Pipeline",
  "streaming": {
    "enabled": true,
    "emit_progress": true,
    "buffer_size": 1000,
    "flush_interval_ms": 100
  },
  "steps": [...]
}
```

### Step Configuration
```python
class MyStep(Step):
    async def run(self, data):
        # Emit progress updates
        for i in range(10):
            await self.emit_progress(i / 10, f"Processing item {i}")
            result = await process_item(i)

        return {"results": all_results}
```

## API Specification

### WebSocket API
```
WS /ws/pipeline/{execution_id}

Message Format:
{
  "event_type": "step_complete",
  "execution_id": "exec_123",
  "step_id": "step_1",
  "timestamp": "2025-10-22T10:30:00Z",
  "data": {
    "output": {...},
    "duration_ms": 1234
  }
}
```

### SSE API
```
GET /stream/pipeline/{execution_id}

Event Format:
event: step_complete
data: {"step_id": "step_1", "output": {...}}
id: 1729594200
```

## Testing Strategy

### Unit Tests
- StreamingContext event emission
- StreamSubscriber interface implementations
- Event serialization/deserialization
- Buffer management and backpressure

### Integration Tests
- Full pipeline execution with streaming
- Multiple concurrent subscribers
- Subscriber add/remove during execution
- Error handling and recovery

### End-to-End Tests
- WebSocket streaming from browser
- SSE streaming from browser
- LLM token streaming
- Long-running pipeline streaming

### Performance Tests
- Streaming overhead measurement
- High-frequency event emission
- Memory usage with large event streams
- Connection scaling (100+ concurrent streams)

## Success Metrics

### Functional
- ✅ Stream events for all pipeline steps
- ✅ WebSocket endpoint functional
- ✅ SSE endpoint functional
- ✅ LLM token streaming working
- ✅ Zero test regressions

### Performance
- ⚡ Streaming overhead < 5% vs non-streaming
- ⚡ Event latency < 50ms
- ⚡ Support 100+ concurrent streams per instance
- ⚡ Memory usage < 100MB for typical pipeline stream

### Quality
- 📊 100% test coverage for streaming components
- 📊 Zero critical bugs in streaming code
- 📚 Complete API documentation
- 📚 Example pipelines with streaming

## Dependencies

### External Libraries (Already Available)
- `fastapi` - WebSocket and SSE support
- `websockets` - WebSocket protocol
- `asyncio` - Async event handling
- `dataclasses` - Event definitions

### Internal Dependencies
- `ia_modules.pipeline.core` - Pipeline and Step classes
- `ia_modules.pipeline.runner` - Pipeline execution
- `ia_modules.reliability.metrics` - Real-time metrics
- `app_knowledge_base.api.websockets` - WebSocket manager

## Risks & Mitigation

### Risk: Performance Impact
- **Mitigation**: Make streaming opt-in, benchmark overhead, optimize hot paths
- **Fallback**: Disable streaming for slow pipelines

### Risk: Memory Leaks
- **Mitigation**: Bounded buffers, aggressive cleanup, memory profiling
- **Monitoring**: Track memory usage metrics

### Risk: WebSocket Stability
- **Mitigation**: Reconnection logic, SSE alternative, health checks
- **Fallback**: Polling-based updates

### Risk: Breaking Changes
- **Mitigation**: 100% backward compatible, streaming is additive
- **Testing**: Run full test suite with and without streaming

## Future Enhancements (Post v0.0.4)

### v0.0.5+
- **Replay Streaming**: Re-stream historical executions
- **Stream Filtering**: Client-side event filtering
- **Compression**: Compress event payloads for bandwidth
- **Multiplexing**: Multiple pipeline streams over one connection
- **Binary Streaming**: Stream binary data (images, files)
- **GraphQL Subscriptions**: Alternative to WebSocket/SSE
- **Distributed Streaming**: Stream from multi-node pipelines

## Alternatives Considered

### 1. Polling
**Pros**: Simple, no persistent connections
**Cons**: Higher latency, more server load, poor UX
**Decision**: Rejected - poor real-time experience

### 2. Message Queue (e.g., Redis Pub/Sub)
**Pros**: Scalable, decoupled
**Cons**: External dependency, complexity
**Decision**: Future enhancement, not MVP

### 3. gRPC Streaming
**Pros**: Efficient, type-safe
**Cons**: Browser support requires gRPC-Web, adds complexity
**Decision**: Rejected - WebSocket/SSE sufficient for MVP

## References

- [WebSocket RFC 6455](https://tools.ietf.org/html/rfc6455)
- [Server-Sent Events (SSE) Spec](https://html.spec.whatwg.org/multipage/server-sent-events.html)
- [FastAPI WebSocket Documentation](https://fastapi.tiangolo.com/advanced/websockets/)
- [LangGraph Streaming](https://python.langchain.com/docs/langgraph/how-tos/streaming-tokens)

## Appendix: Example Usage

### Python API
```python
from ia_modules.pipeline.streaming import StreamingContext, CallbackSubscriber

async def on_event(event):
    print(f"[{event.event_type}] {event.data}")

# Create streaming context
streaming_context = StreamingContext("exec_123")
subscriber = CallbackSubscriber(on_event)
await streaming_context.subscribe(subscriber)

# Run pipeline with streaming
pipeline = create_pipeline_from_json(config)
pipeline.streaming_context = streaming_context
result = await pipeline.run(input_data)
```

### JavaScript/TypeScript Frontend
```javascript
// Connect to streaming endpoint
const ws = new WebSocket(`ws://localhost:8000/ws/pipeline/${executionId}`);

ws.onmessage = (event) => {
  const streamEvent = JSON.parse(event.data);

  switch (streamEvent.event_type) {
    case 'step_start':
      console.log(`Step ${streamEvent.step_id} started`);
      break;
    case 'step_progress':
      updateProgressBar(streamEvent.data.progress);
      break;
    case 'step_complete':
      displayResults(streamEvent.data.output);
      break;
  }
};
```

---

**Document Version**: 1.0
**Created**: 2025-10-22
**Author**: Claude
**Status**: Draft - Pending Review
**Target Release**: v0.0.4
