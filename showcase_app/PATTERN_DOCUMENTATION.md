# Agentic Design Patterns - Complete Documentation

## Architecture Formula

```
AI Agent = LLM + System Prompt + Tools + Memory + Reasoning Pattern
```

## Pattern Catalog

### 1. Reflection Pattern
**Self-critique and iterative improvement**

**How it works:**
1. Generate initial output
2. LLM critiques its own output against quality criteria
3. Extract improvement suggestions
4. Apply improvements to create refined output
5. Repeat until quality threshold met or max iterations reached

**Environment Variables:**
- `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` - At least one required
- `DEFAULT_LLM_PROVIDER` - Which provider to use (defaults to first available)
- `OPENAI_MODEL` / `ANTHROPIC_MODEL` / `GEMINI_MODEL` - Model selection per provider
- `PATTERN_TEMPERATURE=0.7` - Lower = more consistent critiques
- `PATTERN_MAX_TOKENS=2000` - Token limit for responses

**API Endpoint:** `POST /api/patterns/reflection`

**Request:**
```json
{
  "initial_output": "Your draft text here",
  "criteria": {
    "clarity": "Text should be clear and concise",
    "accuracy": "Information must be factual",
    "completeness": "All key points covered"
  },
  "max_iterations": 3
}
```

**Response:**
```json
{
  "pattern": "reflection",
  "initial_output": "...",
  "final_output": "...",
  "iterations": [
    {
      "iteration": 1,
      "output": "...",
      "critique": "LLM-generated critique",
      "quality_score": 0.65,
      "improvements_suggested": ["Add more detail", "Fix grammar"],
      "timestamp": "2025-10-24T..."
    }
  ],
  "total_iterations": 3,
  "final_quality_score": 0.87
}
```

---

### 2. Planning Pattern
**Multi-step goal decomposition**

**How it works:**
1. LLM receives high-level goal and constraints
2. Breaks down goal into 3-5 actionable steps
3. For each step: provides reasoning, duration estimate, dependencies, success criteria
4. Returns structured execution plan

**Environment Variables:**
- Same LLM provider configuration as Reflection
- `PATTERN_TEMPERATURE=0.7` - Affects planning creativity vs consistency
- `PATTERN_MAX_TOKENS=2000` - Must be sufficient for multi-step plans

**API Endpoint:** `POST /api/patterns/planning`

**Request:**
```json
{
  "goal": "Research and write a technical blog post",
  "constraints": {
    "time_available": "4 hours",
    "budget": "none",
    "resources": "internet access only"
  }
}
```

**Response:**
```json
{
  "pattern": "planning",
  "goal": "Research and write a technical blog post",
  "constraints": {...},
  "total_steps": 4,
  "estimated_total_time": 120,
  "plan": [
    {
      "step_number": 1,
      "subgoal": "Define topic and scope",
      "reasoning": "Clear scope prevents scope creep",
      "estimated_duration": 15,
      "dependencies": [],
      "success_criteria": ["Topic chosen", "Outline created"]
    }
  ],
  "created_at": "2025-10-24T..."
}
```

---

### 3. Tool Use Pattern
**Dynamic capability selection**

**How it works:**
1. Agent analyzes task requirements
2. Matches requirements to available tool capabilities
3. Selects optimal tool(s) with reasoning
4. Plans execution sequence

**Environment Variables:**
- Same LLM provider configuration
- No pattern-specific env vars currently

**API Endpoint:** `POST /api/patterns/tool-use`

**Request:**
```json
{
  "task": "Find and summarize recent papers on neural networks",
  "available_tools": ["web_search", "pdf_reader", "summarizer", "calculator"]
}
```

**Response:**
```json
{
  "pattern": "tool_use",
  "task": "...",
  "available_tools": [...],
  "analysis": {
    "task_type": "research_and_synthesis",
    "required_capabilities": ["search", "read", "summarize"],
    "complexity": "medium"
  },
  "selected_tools": [
    {
      "tool": "web_search",
      "reasoning": "Required to find recent papers",
      "priority": 1
    },
    {
      "tool": "pdf_reader",
      "reasoning": "Needed to extract paper content",
      "priority": 2
    },
    {
      "tool": "summarizer",
      "reasoning": "Final step to create summary",
      "priority": 3
    }
  ],
  "execution_plan": {...}
}
```

---

### 4. Agentic RAG Pattern
**Query refinement and relevance evaluation**

**How it works:**
1. Retrieve documents for initial query
2. LLM evaluates relevance of each document
3. If average relevance < threshold, refine query
4. Retrieve again with refined query
5. Repeat until good results or max refinements

**Environment Variables:**
- Same LLM provider configuration
- Future: `RAG_RELEVANCE_THRESHOLD=0.7` - Minimum acceptable relevance
- Future: `RAG_MAX_REFINEMENTS=3` - Max query refinement iterations

**API Endpoint:** `POST /api/patterns/agentic-rag`

**Request:**
```json
{
  "query": "machine learning best practices",
  "max_refinements": 3
}
```

**Response:**
```json
{
  "pattern": "agentic_rag",
  "initial_query": "machine learning best practices",
  "final_query": "machine learning model training and deployment best practices 2025",
  "total_iterations": 2,
  "iterations": [
    {
      "iteration": 1,
      "query": "machine learning best practices",
      "documents": [...],
      "relevance_scores": [0.6, 0.5, 0.4],
      "average_relevance": 0.5,
      "refinement_reasoning": "Query too broad, adding specificity"
    }
  ],
  "final_relevance": 0.82
}
```

---

### 5. Metacognition Pattern
**Self-monitoring and strategy adjustment**

**How it works:**
1. Agent receives execution trace and performance metrics
2. Analyzes patterns in execution (bottlenecks, errors, inefficiencies)
3. Assesses overall performance against goals
4. Identifies specific issues
5. Suggests strategy adjustments

**Environment Variables:**
- Same LLM provider configuration
- Future: `METACOG_MIN_TRACE_LENGTH=5` - Minimum execution steps needed

**API Endpoint:** `POST /api/patterns/metacognition`

**Request:**
```json
{
  "execution_trace": [
    {"step": "retrieve_data", "duration": 1.2, "success": true},
    {"step": "process_data", "duration": 5.8, "success": true},
    {"step": "generate_output", "duration": 0.3, "success": false}
  ],
  "performance_metrics": {
    "total_duration": 7.3,
    "success_rate": 0.67,
    "error_count": 1
  }
}
```

**Response:**
```json
{
  "pattern": "metacognition",
  "performance_assessment": "Below target performance",
  "patterns_detected": [
    "Processing step is bottleneck (80% of time)",
    "Output generation failing consistently"
  ],
  "issues_identified": [
    {
      "issue": "Slow data processing",
      "severity": "high",
      "impact": "80% of execution time"
    }
  ],
  "strategy_adjustments": [
    "Implement caching for processed data",
    "Add error handling in output generation",
    "Consider parallel processing for data step"
  ]
}
```

---

## Environment Variable Impact Matrix

| Variable | Reflection | Planning | Tool Use | Agentic RAG | Metacognition |
|----------|-----------|----------|----------|-------------|---------------|
| `OPENAI_API_KEY` | ✅ Required | ✅ Required | ✅ Required | ✅ Required | ✅ Required |
| `ANTHROPIC_API_KEY` | ✅ Alternative | ✅ Alternative | ✅ Alternative | ✅ Alternative | ✅ Alternative |
| `GEMINI_API_KEY` | ✅ Alternative | ✅ Alternative | ✅ Alternative | ✅ Alternative | ✅ Alternative |
| `DEFAULT_LLM_PROVIDER` | ✅ Used | ✅ Used | ✅ Used | ✅ Used | ✅ Used |
| `*_MODEL` | ✅ Used | ✅ Used | ✅ Used | ✅ Used | ✅ Used |
| `PATTERN_TEMPERATURE` | ✅ Used | ✅ Used | ⚠️ Not yet | ⚠️ Not yet | ⚠️ Not yet |
| `PATTERN_MAX_TOKENS` | ✅ Used | ✅ Used | ⚠️ Not yet | ⚠️ Not yet | ⚠️ Not yet |

**Legend:**
- ✅ Currently implemented and used
- ⚠️ Defined but not yet applied to pattern
- ❌ Not applicable

---

## Future Code Requirements

### 1. Environment Variable Integration

**Missing implementations:**
```python
# In pattern_service.py __init__
self.default_temperature = float(os.getenv("PATTERN_TEMPERATURE", "0.7"))
self.default_max_tokens = int(os.getenv("PATTERN_MAX_TOKENS", "2000"))

# Apply to all pattern methods
response = await self.llm_service.generate_completion(
    prompt=prompt,
    temperature=self.default_temperature,  # Currently hardcoded
    max_tokens=self.default_max_tokens     # Currently hardcoded
)
```

**New env vars to add:**
```bash
# RAG-specific
RAG_RELEVANCE_THRESHOLD=0.7
RAG_MAX_REFINEMENTS=3
RAG_CHUNK_SIZE=1000

# Metacognition-specific
METACOG_MIN_TRACE_LENGTH=5
METACOG_PERFORMANCE_THRESHOLD=0.8

# Tool Use-specific
TOOL_USE_MAX_TOOLS=5
TOOL_USE_PARALLEL_ALLOWED=true

# Per-pattern temperature overrides
REFLECTION_TEMPERATURE=0.3  # Lower for consistency
PLANNING_TEMPERATURE=0.7
TOOL_USE_TEMPERATURE=0.5
RAG_TEMPERATURE=0.6
METACOG_TEMPERATURE=0.8  # Higher for creative solutions
```

### 2. Provider Selection API

**Endpoint needed:** `POST /api/patterns/{pattern_name}/execute`

```python
# Allow per-request provider selection
class PatternExecutionRequest(BaseModel):
    provider: Optional[str] = None  # "openai", "anthropic", "gemini"
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    # ... pattern-specific params

# In pattern service
async def execute_pattern(
    self,
    pattern_name: str,
    params: Dict,
    provider_override: Optional[str] = None
):
    provider = provider_override or self.default_provider
    # Use specified provider for this execution
```

### 3. Pattern Chaining

**Not yet implemented:**
```python
# Combine patterns in sequences
class PatternChain:
    """Execute multiple patterns in sequence"""
    
    async def execute_chain(
        self,
        steps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Example:
        [
            {"pattern": "planning", "goal": "Write blog post"},
            {"pattern": "tool_use", "task": "Research topic"},
            {"pattern": "reflection", "output": "$previous.result"}
        ]
        """
        results = []
        context = {}
        
        for step in steps:
            pattern = step["pattern"]
            params = self._resolve_params(step, context)
            result = await self.execute_pattern(pattern, params)
            context[f"step_{len(results)}"] = result
            results.append(result)
        
        return {
            "chain": steps,
            "results": results,
            "final_output": results[-1]
        }
```

### 4. Streaming Support

**Not yet implemented:**
```python
# Streaming LLM responses for real-time feedback
async def reflection_example_stream(
    self,
    initial_output: str,
    criteria: Dict[str, str]
) -> AsyncGenerator[Dict[str, Any], None]:
    """Stream reflection iterations as they complete"""
    for i in range(max_iterations):
        # Stream critique generation
        critique_chunks = []
        async for chunk in self.llm_service.generate_completion_stream(prompt):
            critique_chunks.append(chunk)
            yield {"type": "critique_chunk", "content": chunk}
        
        critique = "".join(critique_chunks)
        yield {"type": "critique_complete", "critique": critique}
        
        # Stream improvement application
        # ...
```

### 5. Pattern Analytics

**New service needed:**
```python
class PatternAnalyticsService:
    """Track pattern usage and performance"""
    
    async def record_execution(
        self,
        pattern: str,
        provider: str,
        duration: float,
        token_usage: Dict,
        success: bool
    ):
        """Record pattern execution metrics"""
        pass
    
    async def get_pattern_stats(
        self,
        pattern: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Returns:
        {
            "reflection": {
                "total_executions": 150,
                "avg_duration": 3.2,
                "avg_iterations": 2.1,
                "avg_quality_improvement": 0.23,
                "provider_breakdown": {
                    "openai": 100,
                    "anthropic": 50
                }
            }
        }
        """
        pass
```

### 6. Batch Processing

**Not yet implemented:**
```python
# Process multiple inputs through same pattern
@router.post("/api/patterns/batch/{pattern_name}")
async def batch_execute_pattern(
    pattern_name: str,
    requests: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Execute same pattern on multiple inputs
    Useful for: batch reflection, batch planning, etc.
    """
    results = await asyncio.gather(*[
        pattern_service.execute_pattern(pattern_name, req)
        for req in requests
    ])
    
    return {
        "pattern": pattern_name,
        "total_requests": len(requests),
        "results": results,
        "summary": _calculate_batch_summary(results)
    }
```

---

## UI/UX Requirements

### 1. Pattern Selection Dashboard

**Component:** `PatternDashboard.tsx`

```tsx
interface Pattern {
  id: string;
  name: string;
  description: string;
  level: string; // "3. Tool-Using", "5. ReAct Agent", etc.
  useCases: string[];
  status: 'available' | 'requires_api_key' | 'disabled';
}

// Visual grid of pattern cards
<div className="pattern-grid">
  {patterns.map(pattern => (
    <PatternCard
      key={pattern.id}
      pattern={pattern}
      onClick={() => navigateToPattern(pattern.id)}
      disabled={pattern.status !== 'available'}
    />
  ))}
</div>
```

**Features needed:**
- Visual indicators for available vs disabled patterns
- API key status warnings
- Quick launch buttons
- Pattern comparison view

### 2. Pattern Execution Interface

**Component:** `PatternExecutor.tsx`

```tsx
<PatternExecutor pattern="reflection">
  {/* Provider Selection */}
  <ProviderSelector
    available={['openai', 'anthropic', 'gemini']}
    selected={selectedProvider}
    onChange={setSelectedProvider}
    showModelSelection={true}
  />
  
  {/* Pattern-specific inputs */}
  <ReflectionInputs
    onSubmit={executePattern}
    showAdvancedOptions={true}
  />
  
  {/* Advanced settings */}
  <AdvancedSettings>
    <TemperatureSlider value={temperature} />
    <MaxTokensInput value={maxTokens} />
    <MaxIterationsInput value={maxIterations} />
  </AdvancedSettings>
  
  {/* Execution progress */}
  <ExecutionProgress
    iterations={iterations}
    currentIteration={current}
    realtime={true}
  />
  
  {/* Results visualization */}
  <ResultsViewer results={results} />
</PatternExecutor>
```

**Features needed:**
- Real-time iteration progress
- Quality score visualization
- Diff view for improvements
- Export results

### 3. Provider Configuration Panel

**Component:** `ProviderConfig.tsx`

```tsx
<ProviderConfig>
  {/* API Key Management */}
  <ApiKeySection>
    <ApiKeyInput
      provider="openai"
      stored={hasOpenAIKey}
      onSave={saveApiKey}
      testConnection={testOpenAI}
    />
    <ApiKeyInput provider="anthropic" />
    <ApiKeyInput provider="gemini" />
  </ApiKeySection>
  
  {/* Default Settings */}
  <DefaultSettings>
    <Select
      label="Default Provider"
      options={['openai', 'anthropic', 'gemini']}
      value={defaultProvider}
    />
    
    <ModelSelector
      provider={defaultProvider}
      models={availableModels}
      selected={defaultModel}
    />
  </DefaultSettings>
  
  {/* Per-Pattern Overrides */}
  <PatternOverrides>
    {patterns.map(pattern => (
      <PatternSettings
        key={pattern}
        pattern={pattern}
        temperature={getTemp(pattern)}
        maxTokens={getTokens(pattern)}
      />
    ))}
  </PatternOverrides>
</ProviderConfig>
```

**Features needed:**
- Secure API key storage (local only, never sent to backend)
- Connection testing
- Provider availability status
- Cost estimation per provider

### 4. Real-time Execution Viewer

**Component:** `LiveExecutionViewer.tsx`

```tsx
// For Reflection Pattern
<ReflectionLiveView>
  <Timeline>
    {iterations.map((iter, i) => (
      <TimelineItem key={i} active={i === current}>
        <IterationNumber>{i + 1}</IterationNumber>
        <IterationContent>
          <OutputPreview text={iter.output} />
          <CritiquePanel critique={iter.critique} />
          <QualityScore score={iter.quality_score} />
          <ImprovementsList items={iter.improvements_suggested} />
        </IterationContent>
      </TimelineItem>
    ))}
  </Timeline>
  
  <DiffView
    before={iterations[current - 1]?.output}
    after={iterations[current]?.output}
    highlights={improvements}
  />
  
  <MetricsPanel>
    <Chart type="line" data={qualityScores} />
    <Stat label="Total Iterations" value={iterations.length} />
    <Stat label="Quality Improvement" value={`+${improvement}%`} />
  </MetricsPanel>
</ReflectionLiveView>
```

**Features needed:**
- Streaming updates
- Iteration timeline
- Before/after comparison
- Quality trend visualization

### 5. Pattern Comparison Tool

**Component:** `PatternComparison.tsx`

```tsx
<PatternComparison>
  <CompareSelector>
    <PatternSelect value={pattern1} onChange={setPattern1} />
    <PatternSelect value={pattern2} onChange={setPattern2} />
  </CompareSelector>
  
  <ComparisonTable>
    <Row>
      <Cell>Provider</Cell>
      <Cell>{pattern1Results.provider}</Cell>
      <Cell>{pattern2Results.provider}</Cell>
    </Row>
    <Row>
      <Cell>Execution Time</Cell>
      <Cell>{pattern1Results.duration}s</Cell>
      <Cell>{pattern2Results.duration}s</Cell>
    </Row>
    <Row>
      <Cell>Tokens Used</Cell>
      <Cell>{pattern1Results.tokens}</Cell>
      <Cell>{pattern2Results.tokens}</Cell>
    </Row>
    <Row>
      <Cell>Quality Score</Cell>
      <Cell>{pattern1Results.quality}</Cell>
      <Cell>{pattern2Results.quality}</Cell>
    </Row>
  </ComparisonTable>
  
  <SideBySideOutput>
    <OutputPanel results={pattern1Results} />
    <OutputPanel results={pattern2Results} />
  </SideBySideOutput>
</PatternComparison>
```

**Features needed:**
- Multi-pattern comparison
- Multi-provider comparison
- Cost analysis
- Quality metrics

### 6. Pattern Chain Builder

**Component:** `PatternChainBuilder.tsx`

```tsx
<ChainBuilder>
  {/* Visual workflow builder */}
  <Canvas>
    {steps.map((step, i) => (
      <ChainStep
        key={i}
        pattern={step.pattern}
        params={step.params}
        onEdit={() => editStep(i)}
        onDelete={() => removeStep(i)}
      />
    ))}
    
    <AddStepButton onClick={addStep} />
  </Canvas>
  
  {/* Parameter binding */}
  <ParameterBinding>
    <BindingRule
      from="step_0.result.final_output"
      to="step_1.params.initial_output"
    />
  </ParameterBinding>
  
  {/* Execute chain */}
  <ExecuteChain
    chain={steps}
    onExecute={runChain}
    showProgress={true}
  />
</ChainBuilder>
```

**Features needed:**
- Drag-and-drop workflow
- Parameter passing between steps
- Conditional branching
- Loop support

### 7. Analytics Dashboard

**Component:** `PatternAnalytics.tsx`

```tsx
<AnalyticsDashboard>
  <MetricsOverview>
    <StatCard
      title="Total Executions"
      value={totalExecutions}
      trend="+12%"
    />
    <StatCard
      title="Avg Quality Score"
      value={avgQuality}
      trend="+5%"
    />
    <StatCard
      title="Cost (30 days)"
      value={`$${totalCost}`}
    />
  </MetricsOverview>
  
  <Charts>
    <ProviderUsageChart data={providerStats} />
    <PatternPopularityChart data={patternStats} />
    <QualityTrendChart data={qualityTrends} />
  </Charts>
  
  <ExecutionHistory
    executions={history}
    filters={['pattern', 'provider', 'date']}
  />
</AnalyticsDashboard>
```

**Features needed:**
- Usage tracking
- Cost monitoring
- Quality trends
- Provider performance comparison

### 8. Mobile-Responsive Pattern Interface

**Component:** `MobilePatternViewer.tsx`

```tsx
<MobileView>
  {/* Swipeable pattern cards */}
  <SwipeablePatterns>
    {patterns.map(pattern => (
      <PatternCard
        key={pattern.id}
        swipeable={true}
        onSwipe={handleSwipe}
      />
    ))}
  </SwipeablePatterns>
  
  {/* Bottom sheet for execution */}
  <BottomSheet>
    <QuickExecute
      pattern={selectedPattern}
      simplified={true}
    />
  </BottomSheet>
</MobileView>
```

---

## Summary

**Current State:**
- ✅ 5 patterns implemented with basic LLM integration
- ✅ Environment variable configuration for API keys and providers
- ✅ Basic REST API endpoints
- ⚠️ Limited use of temperature/max_tokens env vars
- ❌ No UI implementation yet

**Immediate Needs:**
1. Apply `PATTERN_TEMPERATURE` and `PATTERN_MAX_TOKENS` to all patterns
2. Add per-pattern temperature overrides
3. Build basic pattern execution UI
4. Add provider selection interface

**Future Enhancements:**
1. Pattern chaining
2. Streaming responses
3. Batch processing
4. Analytics dashboard
5. Mobile interface
6. Cost tracking
7. Pattern comparison tools
