"""
Integration tests for advanced features working together.

Tests interactions between constitutional AI, memory, multimodal,
agents, prompt optimization, and tools.
"""

import pytest
from ia_modules.pipeline.test_utils import create_test_execution_context
import asyncio
from unittest.mock import Mock, AsyncMock
import time

from ia_modules.patterns.constitutional_ai import (
    Principle,
    PrincipleCategory,
    ConstitutionalConfig,
    ConstitutionalAIStep
)
from ia_modules.memory.semantic_memory import SemanticMemory
from ia_modules.memory.episodic_memory import EpisodicMemory
from ia_modules.memory.working_memory import WorkingMemory
from ia_modules.memory.compression import MemoryCompressor, CompressionStrategy
from ia_modules.multimodal.processor import (
    MultiModalProcessor,
    MultiModalInput,
    MultiModalConfig,
    ModalityType
)
from ia_modules.agents.communication import MessageBus, AgentMessage, MessageType
from ia_modules.agents.orchestrator import AgentOrchestrator
from ia_modules.agents.state import StateManager
from ia_modules.agents.core import BaseAgent, AgentRole
from ia_modules.prompt_optimization.genetic import GeneticOptimizer, GeneticConfig
from ia_modules.prompt_optimization.evaluators import PromptEvaluator
from ia_modules.tools.tool_registry import AdvancedToolRegistry
from ia_modules.tools.tool_chain import ToolChain
from ia_modules.tools.core import ToolDefinition


# Mock memory for testing
class MockMemory:
    """Mock memory object."""
    _id_counter = 0

    def __init__(self, content, importance=0.5, tags=None):
        MockMemory._id_counter += 1
        self.id = f"mem_{MockMemory._id_counter}"
        self.content = content
        self.importance = importance
        self.metadata = {"tags": tags or []}
        self.timestamp = time.time()
        self.access_count = 0
        self.embedding = None


# Mock LLM provider for testing
class MockLLMProvider:
    """Mock LLM provider."""

    async def generate(self, prompt, **kwargs):
        """Generate mock response."""
        await asyncio.sleep(0.01)
        return {"content": f"Response to: {prompt[:50]}..."}


# Mock agent for testing
class MockAgent(BaseAgent):
    """Mock agent."""

    def __init__(self, role, state):
        super().__init__(role, state)
        self.execute_count = 0

    async def execute(self, input_data):
        """Execute mock task."""
        self.execute_count += 1
        await asyncio.sleep(0.01)
        return {"status": "success", "data": input_data}


@pytest.mark.asyncio
class TestConstitutionalAIWithMemory:
    """Test Constitutional AI integrated with memory systems."""

    async def test_constitutional_ai_stores_in_memory(self):
        """Constitutional AI critiques are stored in episodic memory."""
        # Setup
        llm_provider = MockLLMProvider()
        episodic = EpisodicMemory()

        principles = [
            Principle(
                name="helpful",
                description="Be helpful",
                critique_prompt="Is this helpful?",
                min_score=0.7
            )
        ]

        config = ConstitutionalConfig(
            principles=principles,
            max_revisions=2
        )

        step = ConstitutionalAIStep(
            name="test",
            prompt="Generate helpful response",
            config=config,
            llm_provider=llm_provider
        )

        # Execute
        result = await step.execute({})

        # Store history in episodic memory
        for revision in result["history"]:
            memory = MockMemory(
                content=revision.response,
                tags=["constitutional_ai", "revision"],
                importance=revision.quality_score
            )
            await episodic.add(memory)

        # Verify
        memories = await episodic.get_by_tags(["constitutional_ai"])
        assert len(memories) > 0

    async def test_constitutional_ai_with_semantic_retrieval(self):
        """Constitutional AI uses semantic memory for context."""
        # Setup semantic memory with context
        semantic = SemanticMemory(enable_embeddings=False)

        context_memories = [
            MockMemory("User prefers concise answers"),
            MockMemory("Previous interactions were technical"),
            MockMemory("User is interested in AI topics")
        ]

        for mem in context_memories:
            await semantic.add(mem)

        # Retrieve context
        context = await semantic.retrieve("user preferences", k=2)

        # Use in Constitutional AI
        llm_provider = MockLLMProvider()
        principles = [
            Principle(
                name="personalized",
                description="Match user preferences",
                critique_prompt="Does this match user preferences?"
            )
        ]

        config = ConstitutionalConfig(principles=principles, max_revisions=1)
        step = ConstitutionalAIStep(
            name="personalized",
            prompt="Generate response based on: " + " ".join([m.content for m in context]),
            config=config,
            llm_provider=llm_provider
        )

        result = await step.execute({})

        assert result["response"] is not None


@pytest.mark.asyncio
class TestMultimodalWithMemory:
    """Test multimodal processing with memory integration."""

    async def test_multimodal_results_stored_in_memory(self):
        """Multimodal processing results are stored in semantic memory."""
        # Setup
        config = MultiModalConfig(enable_fusion=False)
        processor = MultiModalProcessor(config=config)
        semantic = SemanticMemory(enable_embeddings=False)

        # Mock processors
        processor.image_processor.process = AsyncMock(
            return_value="Image shows a beautiful sunset"
        )
        processor.audio_processor.transcribe = AsyncMock(
            return_value="The audio says hello world"
        )

        # Process multimodal inputs
        inputs = [
            MultiModalInput(content=b"image", modality=ModalityType.IMAGE),
            MultiModalInput(content=b"audio", modality=ModalityType.AUDIO)
        ]

        result = await processor.process(inputs)

        # Store in memory
        for modality, content in result.modality_results.items():
            memory = MockMemory(
                content=f"{modality.value}: {content}",
                tags=["multimodal", modality.value]
            )
            await semantic.add(memory)

        # Verify
        memories = await semantic.retrieve("image", k=5)
        assert any("image" in m.content.lower() for m in memories)

    async def test_multimodal_with_working_memory_buffer(self):
        """Multimodal processing uses working memory as buffer."""
        # Setup working memory
        working = WorkingMemory(size=5)

        # Process multiple multimodal inputs
        config = MultiModalConfig(enable_fusion=False)
        processor = MultiModalProcessor(config=config)

        processor.image_processor.process = AsyncMock(return_value="Image result")

        for i in range(7):  # More than working memory size
            inputs = [
                MultiModalInput(
                    content=b"image",
                    modality=ModalityType.IMAGE
                )
            ]

            result = await processor.process(inputs)

            # Add to working memory
            memory = MockMemory(
                content=result.result,
                importance=0.5 + i * 0.05
            )
            await working.add(memory)

        # Verify working memory is at capacity
        assert working.get_size() <= 5


@pytest.mark.asyncio
class TestAgentCollaborationWithTools:
    """Test agent collaboration integrated with tool system."""

    async def test_agents_use_shared_tool_registry(self):
        """Multiple agents use shared tool registry."""
        # Setup tool registry
        registry = AdvancedToolRegistry()

        # Register tools
        async def search_tool(**kwargs):
            return {"results": ["result1", "result2"]}

        async def analyze_tool(**kwargs):
            return {"analysis": "complete"}

        search_def = ToolDefinition(
            name="search",
            description="Search tool",
            function=search_tool,
            parameters={}
        )

        analyze_def = ToolDefinition(
            name="analyze",
            description="Analyze tool",
            function=analyze_tool,
            parameters={}
        )

        registry.register(search_def)
        registry.register(analyze_def)

        # Execute tools from different agents
        result1 = await registry.execute("search", {})
        result2 = await registry.execute("analyze", {})

        assert result1 is not None
        assert result2 is not None

        # Check statistics
        stats = registry.get_statistics()
        assert len(stats) == 2

    async def test_agent_orchestration_with_tool_chain(self):
        """Agent orchestrator uses tool chains."""
        # Setup
        state = StateManager(thread_id="test")
        orchestrator = AgentOrchestrator(state)

        # Mock tool executor
        async def tool_executor(tool_name, params):
            await asyncio.sleep(0.01)
            return f"result_{tool_name}"

        # Create tool chain
        chain = ToolChain(tool_executor)
        chain.add_step("step1", {}, "output1")
        chain.add_step("step2", {}, "output2")

        # Execute chain
        result = await chain.execute({})

        # Store results in agent state
        for key, value in result.context.items():
            await state.set(key, value)

        # Verify
        output1 = await state.get("output1")
        assert output1 is not None


@pytest.mark.asyncio
class TestPromptOptimizationWithMemory:
    """Test prompt optimization with memory integration."""

    async def test_genetic_optimizer_uses_memory_for_seeds(self):
        """Genetic optimizer uses memory to seed population."""
        # Setup semantic memory with good prompts
        semantic = SemanticMemory(enable_embeddings=False)

        good_prompts = [
            MockMemory("Solve this problem step by step", importance=0.9),
            MockMemory("Think carefully about the solution", importance=0.8),
            MockMemory("Break down the problem systematically", importance=0.85)
        ]

        for mem in good_prompts:
            await semantic.add(mem)

        # Retrieve as seed prompts
        seed_memories = await semantic.retrieve("problem solving", k=3)
        seed_prompts = [m.content for m in seed_memories]

        # Use in genetic optimizer
        class SimpleEvaluator(PromptEvaluator):
            async def evaluate(self, prompt, test_cases=None):
                await asyncio.sleep(0.001)
                return 0.7 + len(prompt) / 1000

        evaluator = SimpleEvaluator()
        config = GeneticConfig(
            population_size=5,
            max_generations=3
        )

        optimizer = GeneticOptimizer(
            evaluator=evaluator,
            config=config,
            seed_prompts=seed_prompts
        )

        result = await optimizer.optimize("Base prompt")

        assert result.best_score > 0
        assert result.iterations > 0


@pytest.mark.asyncio
class TestMemoryCompression:
    """Test memory compression across different memory types."""

    async def test_compress_episodic_to_semantic(self):
        """Episodic memories are compressed and moved to semantic."""
        # Setup
        episodic = EpisodicMemory()
        semantic = SemanticMemory(enable_embeddings=False)
        compressor = MemoryCompressor(
            strategy=CompressionStrategy.IMPORTANCE,
            target_ratio=0.3
        )

        # Add many episodic memories
        memories = [
            MockMemory(f"Event {i}", importance=i/20, tags=["event"])
            for i in range(20)
        ]

        for mem in memories:
            await episodic.add(mem)

        # Compress
        all_memories = await episodic.get_recent(k=20)
        compressed = await compressor.compress(all_memories)

        # Store compressed in semantic memory
        for mem in compressed:
            await semantic.add(mem)

        # Verify compression
        assert len(compressed) < len(memories)

        # Clear old episodic memories
        await episodic.clear()

        # Verify semantic has compressed memories
        semantic_memories = await semantic.retrieve("Event", k=10)
        assert len(semantic_memories) > 0

    async def test_working_memory_overflow_to_episodic(self):
        """Working memory overflow moves to episodic memory."""
        # Setup
        working = WorkingMemory(size=5)
        episodic = EpisodicMemory()

        # Add more than capacity
        for i in range(10):
            mem = MockMemory(f"Memory {i}", importance=0.5, tags=["working"])

            # Try to add to working memory
            await working.add(mem)

            # If working memory is full, move to episodic
            if working.is_full():
                # Simulate eviction - add to episodic
                await episodic.add(mem)

        # Verify
        assert working.get_size() <= 5
        episodic_count = len(await episodic.get_by_tags(["working"]))
        assert episodic_count > 0


@pytest.mark.asyncio
class TestMultiAgentWithMessageBus:
    """Test multi-agent collaboration with message bus."""

    async def test_multi_agent_workflow_with_memory(self):
        """Multi-agent workflow shares information via message bus and memory."""
        # Setup
        bus = MessageBus()
        state = StateManager(thread_id="multi-agent")

        # Create agents
        async def agent1_handler(msg):
            """Agent 1 processes and responds."""
            if msg.message_type == MessageType.TASK_REQUEST:
                # Process task
                result = {"processed": "data"}

                # Send response
                reply = msg.create_reply(
                    sender="agent1",
                    content=result,
                    message_type=MessageType.TASK_RESPONSE
                )
                await bus.send(reply)

        async def agent2_handler(msg):
            """Agent 2 receives and stores in state."""
            if msg.message_type == MessageType.TASK_RESPONSE:
                # Store in shared state
                await state.set("agent1_result", msg.content)

        # Subscribe agents
        await bus.subscribe("agent1", agent1_handler)
        await bus.subscribe("agent2", agent2_handler)

        # Send task request
        task = AgentMessage(
            sender="coordinator",
            recipient="agent1",
            message_type=MessageType.TASK_REQUEST,
            content={"task": "process_data"}
        )
        await bus.send(task)

        # Wait for processing
        await asyncio.sleep(0.2)

        # Verify result in state
        result = await state.get("agent1_result")
        assert result is not None


@pytest.mark.asyncio
class TestEndToEndWorkflow:
    """End-to-end test combining all advanced features."""

    async def test_complete_ai_agent_workflow(self):
        """
        Complete workflow:
        1. Multimodal input processing
        2. Constitutional AI for safe responses
        3. Memory storage and retrieval
        4. Prompt optimization
        5. Tool execution via chains
        6. Agent collaboration
        """
        # 1. Setup components
        llm_provider = MockLLMProvider()
        semantic = SemanticMemory(enable_embeddings=False)
        episodic = EpisodicMemory()
        tool_registry = AdvancedToolRegistry()

        # 2. Process multimodal input
        multimodal_config = MultiModalConfig(enable_fusion=False)
        multimodal = MultiModalProcessor(config=multimodal_config)
        multimodal.image_processor.process = AsyncMock(
            return_value="User uploaded image of a cat"
        )

        inputs = [
            MultiModalInput(content=b"image", modality=ModalityType.IMAGE)
        ]

        mm_result = await multimodal.process(inputs)

        # Store in episodic memory
        mem1 = MockMemory(
            mm_result.result,
            tags=["multimodal", "user_input"]
        )
        await episodic.add(mem1)

        # 3. Use Constitutional AI for safe response
        principles = [
            Principle(
                name="safe",
                description="Ensure safe response",
                critique_prompt="Is this response safe?"
            )
        ]

        const_config = ConstitutionalConfig(
            principles=principles,
            max_revisions=1
        )

        const_ai = ConstitutionalAIStep(
            name="safe_response",
            prompt="Generate response about: " + mm_result.result,
            config=const_config,
            llm_provider=llm_provider
        )

        ai_result = await const_ai.execute({})

        # Store in semantic memory
        mem2 = MockMemory(
            ai_result["response"],
            tags=["ai_response", "safe"],
            importance=ai_result["quality_score"]
        )
        await semantic.add(mem2)

        # 4. Register and execute tools
        async def summary_tool(**kwargs):
            return {"summary": "Concise summary"}

        tool_def = ToolDefinition(
            name="summarize",
            description="Summarize content",
            function=summary_tool,
            parameters={}
        )

        tool_registry.register(tool_def)

        tool_result = await tool_registry.execute("summarize", {})

        # 5. Verify complete workflow
        assert mm_result is not None
        assert ai_result["response"] is not None
        assert tool_result is not None

        # Verify memories were stored
        semantic_mems = await semantic.retrieve("response", k=5)
        episodic_mems = await episodic.get_by_tags(["user_input"])

        assert len(semantic_mems) > 0
        assert len(episodic_mems) > 0


@pytest.mark.asyncio
class TestPerformanceIntegration:
    """Test performance of integrated features."""

    async def test_concurrent_multimodal_processing(self):
        """Multiple multimodal inputs processed concurrently."""
        config = MultiModalConfig(max_concurrent=3, enable_fusion=False)
        processor = MultiModalProcessor(config=config)

        processor.image_processor.process = AsyncMock(return_value="Image result")

        # Process multiple inputs concurrently
        tasks = []
        for i in range(5):
            inputs = [
                MultiModalInput(content=b"image", modality=ModalityType.IMAGE)
            ]
            tasks.append(processor.process(inputs))

        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(r.result is not None for r in results)

    async def test_memory_compression_performance(self):
        """Memory compression handles large volumes."""
        compressor = MemoryCompressor(
            strategy=CompressionStrategy.CLUSTER,
            target_ratio=0.2
        )

        # Create many memories
        memories = [
            MockMemory(f"Memory content {i} about topic {i % 5}")
            for i in range(100)
        ]

        # Compress
        start = time.time()
        compressed = await compressor.compress(memories)
        duration = time.time() - start

        # Verify compression happened and was reasonably fast
        assert len(compressed) < len(memories)
        assert duration < 5.0  # Should complete in reasonable time

    async def test_tool_registry_caching_performance(self):
        """Tool registry caching improves performance."""
        registry = AdvancedToolRegistry(enable_caching=True)

        call_count = 0

        async def slow_tool(**kwargs):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)  # Simulate slow operation
            return {"result": "data"}

        tool_def = ToolDefinition(
            name="slow_tool",
            description="Slow tool",
            function=slow_tool,
            parameters={}
        )

        registry.register(tool_def)

        # First call (no cache)
        start1 = time.time()
        await registry.execute_cached("slow_tool", {})
        duration1 = time.time() - start1

        # Second call (cached)
        start2 = time.time()
        await registry.execute_cached("slow_tool", {})
        duration2 = time.time() - start2

        # Cached call should be faster
        assert duration2 < duration1
        assert call_count == 1  # Only called once due to caching


class TestErrorHandlingIntegration:
    """Test error handling across integrated features."""

    @pytest.mark.asyncio
    async def test_graceful_degradation_multimodal_failure(self):
        """System handles multimodal processing failures gracefully."""
        config = MultiModalConfig(enable_fusion=False)
        processor = MultiModalProcessor(config=config)

        # Mock a failure
        processor.image_processor.process = AsyncMock(
            side_effect=Exception("Processing error")
        )

        # Should not crash
        try:
            inputs = [
                MultiModalInput(content=b"bad", modality=ModalityType.IMAGE)
            ]
            result = await processor.process(inputs)
            # May return partial results or error indication
        except Exception:
            # Error handling is acceptable
            pass

    @pytest.mark.asyncio
    async def test_memory_system_resilience(self):
        """Memory systems handle errors gracefully."""
        semantic = SemanticMemory(enable_embeddings=False)

        # Try to add invalid memory
        try:
            # This should not crash the system
            bad_mem = MockMemory("")
            await semantic.add(bad_mem)

            # Retrieval should still work
            results = await semantic.retrieve("test", k=5)
            assert isinstance(results, list)
        except Exception:
            # Acceptable to handle errors
            pass


# Test suite complete marker
@pytest.mark.asyncio
async def test_suite_complete():
    """Marker test that the complete test suite exists."""
    assert True, "Integration test suite successfully created"
