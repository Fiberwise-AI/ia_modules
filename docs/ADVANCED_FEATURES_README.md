# Advanced AI Features for IA Modules

This document describes the six advanced AI features implemented in `ia_modules`:

1. [Constitutional AI](#1-constitutional-ai)
2. [Advanced Memory Strategies](#2-advanced-memory-strategies)
3. [Multi-Modal Support](#3-multi-modal-support)
4. [Agent Collaboration](#4-agent-collaboration-patterns)
5. [Prompt Optimization](#5-prompt-optimization)
6. [Advanced Tool Calling](#6-advanced-tool-calling)

---

## 1. Constitutional AI

**Self-critique pattern** where AI critiques and improves its own outputs based on predefined principles.

### Features

- Define custom constitutional principles
- Iterative refinement based on critique
- Pre-built constitutions (harmless, helpful, honest)
- Parallel critique evaluation
- Quality scoring and convergence detection

### Quick Start

```python
from ia_modules.patterns import ConstitutionalAIStep, ConstitutionalConfig
from ia_modules.patterns.constitutions import harmless_principles

config = ConstitutionalConfig(
    principles=harmless_principles,
    max_revisions=3,
    min_quality_score=0.8
)

step = ConstitutionalAIStep(
    name="safe_assistant",
    prompt="Provide advice on {topic}",
    config=config
)

result = await step.execute({"topic": "stress management"})
print(result['response'])  # Refined, safe response
```

### Pre-built Constitutions

- **harmless_principles**: Safety and ethics
- **helpful_principles**: Utility and clarity
- **honest_principles**: Truthfulness and accuracy

### Custom Principles

```python
from ia_modules.patterns import Principle, PrincipleCategory

custom = Principle(
    name="technical_accuracy",
    description="Ensure technical correctness",
    critique_prompt="Rate technical accuracy 0-10...",
    weight=2.0,
    min_score=0.9
)
```

---

## 2. Advanced Memory Strategies

Sophisticated memory management with **semantic**, **episodic**, and **working** memory.

### Features

- **Semantic Memory**: Long-term knowledge with vector search
- **Episodic Memory**: Event sequences with temporal indexing
- **Working Memory**: Short-term buffer with LRU eviction
- **Memory Compression**: Automatic summarization of old memories
- Multiple storage backends (in-memory, SQLite, ChromaDB)

### Quick Start

```python
from ia_modules.memory import MemoryManager, MemoryConfig

config = MemoryConfig(
    semantic_enabled=True,
    episodic_enabled=True,
    working_memory_size=10,
    compression_threshold=50
)

memory = MemoryManager(config)

# Add memories
await memory.add(
    "User prefers Python",
    metadata={"importance": 0.9, "type": "preference"}
)

# Retrieve relevant memories
results = await memory.retrieve("programming language", k=5)

# Get formatted context for LLM
context = await memory.get_context_window(
    query="What does user like?",
    max_tokens=2000
)
```

### Memory Types

- **Semantic**: Facts and knowledge
- **Episodic**: Events and experiences
- **Working**: Recent context
- **Compressed**: Summarized old memories

### Storage Backends

- **InMemoryBackend**: Fast, not persistent
- **SQLiteBackend**: Persistent, file-based
- **VectorBackend**: Semantic search with ChromaDB

---

## 3. Multi-Modal Support

Process **text**, **images**, **audio**, and **video** with unified interface.

### Features

- **Image Processing**: Vision models (GPT-4V, Claude Vision, Gemini)
- **Audio Processing**: Speech-to-text (Whisper), text-to-speech
- **Video Processing**: Frame extraction and analysis
- **Modality Fusion**: Combine information across modalities

### Quick Start

```python
from ia_modules.multimodal import (
    MultiModalProcessor,
    MultiModalInput,
    ModalityType
)

processor = MultiModalProcessor()

# Process image
image_data = open("photo.jpg", "rb").read()
result = await processor.process_image(
    image=image_data,
    prompt="What's in this image?"
)

# Process multiple modalities
inputs = [
    MultiModalInput(
        content="Analyze this diagram",
        modality=ModalityType.TEXT
    ),
    MultiModalInput(
        content=image_data,
        modality=ModalityType.IMAGE
    )
]

result = await processor.process(inputs)
print(result.result)  # Fused analysis
```

### Supported Providers

- **OpenAI**: GPT-4 Vision, Whisper, TTS
- **Anthropic**: Claude Vision
- **Gemini**: Gemini Vision

### Vision Models

```python
from ia_modules.multimodal import MultiModalConfig

config = MultiModalConfig(
    image_model="gpt-4-vision-preview",
    vision_provider="openai",  # or "anthropic", "gemini"
    max_image_size=2048
)
```

---

## 4. Agent Collaboration Patterns

Multiple agents working together with different collaboration patterns.

### Features

- **Specialist Agents**: Research, Analysis, Synthesis, Critic
- **Collaboration Patterns**: Hierarchical, Peer-to-peer, Debate, Consensus
- **Message Bus**: Async agent communication
- **Task Decomposition**: Automatic task splitting
- **Result Synthesis**: Combine agent outputs

### Quick Start

```python
from ia_modules.agents import (
    AgentOrchestrator,
    StateManager,
    ResearchAgent,
    AnalysisAgent,
    SynthesisAgent
)

# Create state manager and orchestrator
state = StateManager(thread_id="user-123")
orchestrator = AgentOrchestrator(state)

# Register agents
orchestrator.add_agent("researcher", ResearchAgent())
orchestrator.add_agent("analyst", AnalysisAgent())
orchestrator.add_agent("synthesizer", SynthesisAgent())

# Build workflow: researcher → analyst → synthesizer
orchestrator.add_edge("researcher", "analyst")
orchestrator.add_edge("analyst", "synthesizer")

# Execute workflow
result = await orchestrator.run(
    start_agent="researcher",
    initial_input={"task": "Analyze AI market trends in healthcare"}
)
```

### Collaboration Patterns

- **Hierarchical**: Leader assigns tasks to workers
- **Peer-to-peer**: Equal collaboration between agents
- **Debate**: Adversarial discussion for best solution
- **Consensus**: Agreement-based decision making

### Specialist Agents

- **ResearchAgent**: Information gathering
- **AnalysisAgent**: Data analysis
- **SynthesisAgent**: Result combination
- **CriticAgent**: Quality checking

---

## 5. Prompt Optimization

Automated prompt engineering using various optimization strategies.

### Features

- **Genetic Algorithm**: Mutation and crossover
- **Reinforcement Learning**: Q-learning for prompt selection
- **A/B Testing**: Statistical significance testing
- **Multi-Armed Bandit**: Exploration/exploitation
- **Automatic Evaluation**: Quality metrics

### Quick Start

```python
from ia_modules.prompt_optimization import (
    GeneticOptimizer,
    GeneticConfig,
    AccuracyEvaluator
)

# Define test cases
test_cases = [
    {"input": "What is 2+2?", "expected": "4"},
    {"input": "Capital of France?", "expected": "Paris"},
]

# Configure optimizer
config = GeneticConfig(
    population_size=20,
    max_generations=50,
    mutation_rate=0.2,
    crossover_rate=0.7
)

# Create evaluator and optimizer
evaluator = AccuracyEvaluator(test_cases)
optimizer = GeneticOptimizer(
    evaluator=evaluator,
    config=config,
    verbose=True
)

# Optimize prompt
result = await optimizer.optimize(
    initial_prompt="Answer: {input}",
    target_metric="accuracy"
)

print(f"Best prompt: {result.best_prompt}")
print(f"Score: {result.best_score}")
```

### Optimization Strategies

- **Genetic**: Evolution-based optimization
- **Reinforcement Learning**: Learn from outcomes
- **A/B Testing**: Statistical comparison
- **Grid Search**: Exhaustive search

### Custom Evaluators

```python
from ia_modules.prompt_optimization import PromptEvaluator

class CustomEvaluator(PromptEvaluator):
    async def evaluate(self, response: str, expected: str) -> float:
        # Custom evaluation logic
        return score
```

---

## 6. Advanced Tool Calling

Sophisticated tool orchestration with planning, parallel execution, and error handling.

### Features

- **Tool Registry**: Discovery and version management
- **Execution Planning**: Automatic task decomposition
- **Parallel Execution**: Concurrent tool calls
- **Tool Chains**: Compose tools with data flow
- **Error Handling**: Retry, fallback, circuit breaker
- **Built-in Tools**: Web search, calculator, code executor, file ops, API caller

### Quick Start

```python
from ia_modules.tools import (
    AdvancedToolExecutor,
    ToolDefinition,
    RetryConfig
)
from ia_modules.tools.builtin_tools import register_all_builtin_tools

# Configure executor
executor = AdvancedToolExecutor(
    enable_caching=True,
    max_concurrent=5,
    default_retry_config=RetryConfig(max_attempts=3)
)

# Register built-in tools (web_search, calculator, etc.)
register_all_builtin_tools(executor.registry)

# Or register a custom tool
custom_tool = ToolDefinition(
    name="web_search",
    description="Search the web",
    parameters={"query": {"type": "string"}},
    execute=search_function
)
executor.register_tool(custom_tool, capabilities=["web_search"])

# Execute task with automatic planning
result = await executor.execute_task(
    "Find latest AI news and summarize",
    context={"max_results": 5}
)
```

### Tool Chains

```python
from ia_modules.tools import ToolChain

chain = ToolChain("research_pipeline")
chain.add_tool(search_tool)
chain.add_tool(extract_tool)
chain.add_tool(summarize_tool)

result = await chain.execute("AI in healthcare")
```

### Built-in Tools

- **web_search**: Search the web
- **calculator**: Math operations
- **code_executor**: Safe Python execution
- **file_ops**: File operations
- **api_caller**: HTTP API calls

### Error Handling

```python
from ia_modules.tools import RetryStrategy, CircuitBreaker

# Retry with exponential backoff
retry = RetryStrategy(
    max_attempts=3,
    backoff="exponential"
)

# Circuit breaker for failing tools
breaker = CircuitBreaker(
    failure_threshold=5,
    timeout=60
)
```

---

## Installation

### Core Dependencies

```bash
pip install ia_modules
```

### Optional Dependencies

For full functionality, install optional dependencies:

```bash
# Memory features
pip install sentence-transformers chromadb

# Multi-modal features
pip install Pillow pydub opencv-python

# Optimization features
pip install numpy scipy

# All features
pip install ia_modules[all]
```

---

## Examples

Complete examples are available in the `examples/` directory:

- `constitutional_ai_example.py`: Self-critique patterns
- `memory_example.py`: Advanced memory usage
- `guardrails_example.py`: Input/output guardrails and validation

**Note:** For multi-modal, agent collaboration, prompt optimization, and tool calling examples, refer to the test suites in `tests/unit/` and `tests/integration/` directories, which contain comprehensive usage examples.

---

## Testing

Run tests for all features:

```bash
# Unit tests
pytest ia_modules/tests/unit/

# Integration tests
pytest ia_modules/tests/integration/

# All tests
pytest ia_modules/tests/
```

---

## Architecture

### Module Structure

```
ia_modules/
├── patterns/              # AI reasoning patterns
│   ├── constitutional_ai.py
│   └── constitutions/
├── memory/               # Memory strategies
│   ├── memory_manager.py
│   ├── semantic_memory.py
│   ├── episodic_memory.py
│   ├── working_memory.py
│   └── storage_backends/
├── multimodal/           # Multi-modal support
│   ├── processor.py
│   ├── image_processor.py
│   ├── audio_processor.py
│   └── video_processor.py
├── agents/               # Agent collaboration
│   ├── orchestrator.py
│   ├── base_agent.py
│   ├── specialist_agents.py
│   └── collaboration_patterns/
├── prompt_optimization/  # Prompt engineering
│   ├── optimizer.py
│   ├── genetic.py
│   ├── reinforcement.py
│   └── ab_testing.py
└── tools/                # Tool calling
    ├── advanced_executor.py
    ├── tool_registry.py
    ├── tool_chain.py
    └── builtin_tools/
```

### Design Principles

- **Modularity**: Each feature is independent and composable
- **Async/Await**: Full async support for concurrent operations
- **Extensibility**: Easy to add custom implementations
- **Production-Ready**: Comprehensive error handling and logging

---

## Performance Considerations

### Memory

- Use `enable_embeddings=False` for faster, non-semantic memory
- Adjust `compression_threshold` based on memory requirements
- Use `InMemoryBackend` for fastest access

### Multi-Modal

- Vision API calls can be expensive; cache results
- Resize images before processing
- Process video frames in batches

### Agent Collaboration

- Limit `max_rounds` to prevent infinite loops
- Use hierarchical pattern for fastest execution
- Enable timeouts for long-running tasks

### Prompt Optimization

- Start with small population sizes (10-20)
- Use A/B testing for quick comparisons
- Cache evaluation results

### Tool Calling

- Enable `enable_caching` for repeated operations
- Adjust `max_parallel` based on resource limits
- Use circuit breakers for unreliable tools

---

## API Reference

Full API documentation is available at: [docs/api/](docs/api/)

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

See [LICENSE](LICENSE) file for details.

---

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/ia_modules/issues)
- **Documentation**: [Full Docs](https://ia-modules.readthedocs.io)
- **Examples**: [examples/](examples/)

---

## Changelog

### Version 0.1.0 (2025-10-25)

**Added:**
- Constitutional AI with self-critique pattern
- Advanced memory strategies (semantic, episodic, working)
- Multi-modal support (text, image, audio, video)
- Agent collaboration patterns
- Prompt optimization with multiple strategies
- Advanced tool calling with planning

**Core Modules:**
- `patterns`: AI reasoning patterns
- `memory`: Memory management
- `multimodal`: Multi-modal processing
- `agents`: Multi-agent collaboration
- `prompt_optimization`: Automated prompt engineering
- `tools`: Advanced tool orchestration

---

## Roadmap

### Upcoming Features

- **Memory**: Graph-based memory networks
- **Multi-Modal**: Live video streaming support
- **Agents**: Distributed agent execution
- **Optimization**: Bayesian optimization for prompts
- **Tools**: Tool marketplace and sharing

---

## Acknowledgments

Built on top of:
- OpenAI GPT models
- Anthropic Claude
- Google Gemini
- Sentence Transformers
- ChromaDB

---

**Happy Building! 🚀**
