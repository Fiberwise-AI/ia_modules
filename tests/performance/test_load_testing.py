"""
Load testing for ia_modules.

Tests system behavior under concurrent load:
- 100 concurrent executions should complete in <30 seconds
- Memory usage should stay under 1GB
- Error rate should be <1%
"""

import pytest
from ia_modules.pipeline.test_utils import create_test_execution_context
import asyncio
import time
import psutil
import os
from typing import List

from ia_modules.patterns import ConstitutionalAIStep, ConstitutionalConfig, Principle
from ia_modules.memory import MemoryManager, MemoryConfig
from ia_modules.agents import AgentOrchestrator, StateManager


@pytest.fixture
def mock_llm():
    """Mock LLM provider for performance testing."""
    class MockLLM:
        async def generate(self, prompt, **kwargs):
            await asyncio.sleep(0.01)  # Simulate minimal processing
            return {"content": f"Response to {prompt[:50]}"}
    return MockLLM()


@pytest.mark.performance
@pytest.mark.asyncio
async def test_100_concurrent_constitutional_ai(mock_llm):
    """Test 100 concurrent Constitutional AI executions."""
    principles = [
        Principle(
            name="test",
            description="Test principle",
            critique_prompt="Rate this",
            min_score=0.7
        )
    ]

    config = ConstitutionalConfig(
        principles=principles,
        max_revisions=1
    )

    async def execute_one():
        step = ConstitutionalAIStep(
            name="load_test",
            prompt="Test prompt",
            config=config,
            llm_provider=mock_llm
        )
        return await step.execute({})

    # Track performance
    start_time = time.time()
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Execute 100 concurrent
    tasks = [execute_one() for _ in range(100)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Verify performance
    duration = end_time - start_time
    memory_used = end_memory - start_memory
    errors = sum(1 for r in results if isinstance(r, Exception))
    error_rate = errors / len(results)

    print(f"\n100 Concurrent Executions:")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Memory Used: {memory_used:.2f}MB")
    print(f"  Error Rate: {error_rate:.2%}")

    # Assertions
    assert duration < 30, f"Took {duration:.2f}s, should be <30s"
    assert memory_used < 1024, f"Used {memory_used:.2f}MB, should be <1GB"
    assert error_rate < 0.01, f"Error rate {error_rate:.2%}, should be <1%"


@pytest.mark.performance
@pytest.mark.asyncio
async def test_1000_concurrent_memory_operations():
    """Test 1000 concurrent memory operations."""
    config = MemoryConfig(
        semantic_enabled=True,
        episodic_enabled=True,
        working_memory_size=10,
        enable_embeddings=False
    )

    memory = MemoryManager(config)

    async def add_and_retrieve(i):
        await memory.add(f"Memory {i}", metadata={"importance": 0.5})
        results = await memory.retrieve(f"Memory {i % 10}", k=5)
        return len(results)

    start_time = time.time()
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / 1024 / 1024

    # Execute 1000 operations
    tasks = [add_and_retrieve(i) for i in range(1000)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024

    duration = end_time - start_time
    memory_used = end_memory - start_memory
    errors = sum(1 for r in results if isinstance(r, Exception))

    print(f"\n1000 Memory Operations:")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Memory Used: {memory_used:.2f}MB")
    print(f"  Errors: {errors}")

    assert duration < 60, "Should complete in <60s"
    assert memory_used < 1024, "Should use <1GB"
    assert errors == 0, "Should have no errors"


@pytest.mark.performance
@pytest.mark.asyncio
async def test_sustained_load():
    """Test sustained load over time."""
    config = MemoryConfig(enable_embeddings=False)
    memory = MemoryManager(config)

    start_time = time.time()
    operations = 0
    errors = 0
    max_duration = 10  # 10 seconds for testing (would be 3600 in production)

    while time.time() - start_time < max_duration:
        try:
            await memory.add(f"Load test {operations}", metadata={"importance": 0.5})
            operations += 1
            await asyncio.sleep(0.01)  # Throttle to ~100 ops/sec
        except Exception:
            errors += 1

    duration = time.time() - start_time
    ops_per_sec = operations / duration
    error_rate = errors / operations if operations > 0 else 0

    print(f"\nSustained Load Test:")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Operations: {operations}")
    print(f"  Ops/sec: {ops_per_sec:.2f}")
    print(f"  Error Rate: {error_rate:.2%}")

    assert ops_per_sec > 50, "Should handle >50 ops/sec"
    assert error_rate < 0.001, "Error rate should be <0.1%"


@pytest.mark.performance
@pytest.mark.asyncio
async def test_memory_leak_detection():
    """Test for memory leaks over repeated operations."""
    config = MemoryConfig(enable_embeddings=False)
    memory = MemoryManager(config)

    process = psutil.Process(os.getpid())

    # Baseline
    for _ in range(100):
        await memory.add("warmup", metadata={"importance": 0.5})
    baseline_memory = process.memory_info().rss / 1024 / 1024

    # Test iterations
    memory_samples = []
    for iteration in range(10):
        for _ in range(100):
            await memory.add(f"Iteration {iteration}", metadata={"importance": 0.5})

        current_memory = process.memory_info().rss / 1024 / 1024
        memory_samples.append(current_memory)

        # Clear some memory
        await memory.compress()

    # Check for linear growth (memory leak)
    if len(memory_samples) >= 2:
        growth = memory_samples[-1] - memory_samples[0]
        growth_per_iteration = growth / len(memory_samples)

        print(f"\nMemory Leak Detection:")
        print(f"  Baseline: {baseline_memory:.2f}MB")
        print(f"  Final: {memory_samples[-1]:.2f}MB")
        print(f"  Growth: {growth:.2f}MB")
        print(f"  Growth per iteration: {growth_per_iteration:.2f}MB")

        assert growth_per_iteration < 10, "Memory leak detected (>10MB/iteration)"


@pytest.mark.performance
@pytest.mark.asyncio
async def test_parallel_agent_execution():
    """Test parallel agent execution performance."""
    state_manager = StateManager(thread_id="test-performance")
    orchestrator = AgentOrchestrator(state_manager)

    # Mock agents
    from ia_modules.agents import ResearchAgent, AnalysisAgent
    orchestrator.register_agent(ResearchAgent(name="researcher"))
    orchestrator.register_agent(AnalysisAgent(name="analyst"))

    async def execute_task(i):
        return await orchestrator.execute(task=f"Task {i}")

    start_time = time.time()

    # Execute 50 tasks in parallel
    tasks = [execute_task(i) for i in range(50)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    duration = time.time() - start_time
    errors = sum(1 for r in results if isinstance(r, Exception))

    print(f"\n50 Parallel Agent Tasks:")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Errors: {errors}")

    assert duration < 120, "Should complete in <2 minutes"
    assert errors < 5, "Should have <10% error rate"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance"])
