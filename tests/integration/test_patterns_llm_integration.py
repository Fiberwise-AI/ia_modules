"""
Integration tests for AI Patterns with real LLM APIs.
Run with: pytest tests/integration/test_patterns_llm_integration.py -v
"""

import pytest
import os
from ia_modules.patterns import (
    ChainOfThoughtStep,
    CoTConfig,
    SelfConsistencyStep,
    SelfConsistencyConfig,
    VotingStrategy,
    ReActAgent,
    ReActConfig,
    TreeOfThoughtsStep,
    ToTConfig,
    PruningStrategy,
)
from ia_modules.pipeline.llm_provider_service import (
    LLMProviderService,
    LLMProvider,
)
from ia_modules.utils import LLMProviderAdapter


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cot_with_openai():
    """Test Chain-of-Thought with OpenAI."""
    service = LLMProviderService()
    service.register_provider(
        "openai",
        LLMProvider.OPENAI,
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-3.5-turbo",
        temperature=0.0
    )
    
    # Use the official adapter
    adapter = LLMProviderAdapter(service)
    context = {'services': {'llm': adapter}}
    
    step = ChainOfThoughtStep(
        name="math",
        prompt="If a train travels 60 miles in 1 hour, how far in 2.5 hours?",
        config=CoTConfig(reasoning_depth=3)
    )
    
    result = await step.execute(context)
    
    assert 'answer' in result
    assert '150' in result['answer'] or '2.5 hours' in result['answer']
    print(f"✓ OpenAI CoT: {result['answer']}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cot_with_anthropic():
    """Test Chain-of-Thought with Anthropic."""
    service = LLMProviderService()
    service.register_provider(
        "anthropic",
        LLMProvider.ANTHROPIC,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-3-haiku-20240307",
        temperature=0.0
    )
    
    adapter = LLMProviderAdapter(service)
    context = {'services': {'llm': adapter}}
    
    step = ChainOfThoughtStep(
        name="logic",
        prompt="All birds can fly. Penguins are birds. Can penguins fly?",
        config=CoTConfig(reasoning_depth=3, format="numbered")
    )
    
    result = await step.execute(context)
    
    assert 'answer' in result
    print(f"✓ Anthropic CoT: {result['answer']}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cot_with_google():
    """Test Chain-of-Thought with Google."""
    service = LLMProviderService()
    service.register_provider(
        "google",
        LLMProvider.GOOGLE,
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-2.0-flash-exp",
        temperature=0.0
    )
    
    adapter = LLMProviderAdapter(service)
    context = {'services': {'llm': adapter}}
    
    step = ChainOfThoughtStep(
        name="word_problem",
        prompt="Sarah has 3 apples. She buys 5 more, then gives away 2. How many does she have?",
        config=CoTConfig(reasoning_depth=3)
    )
    
    result = await step.execute(context)
    
    assert 'answer' in result
    assert '6' in str(result['answer'])
    print(f"✓ Google CoT: {result['answer']}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_self_consistency_openai():
    """Test Self-Consistency with OpenAI."""
    service = LLMProviderService()
    service.register_provider(
        "openai",
        LLMProvider.OPENAI,
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-3.5-turbo"
    )
    
    adapter = LLMProviderAdapter(service)
    context = {'services': {'llm': adapter}}
    
    step = SelfConsistencyStep(
        name="factual",
        prompt="What is the capital of France?",
        config=SelfConsistencyConfig(
            num_samples=5,
            voting_strategy=VotingStrategy.MAJORITY,
            temperature=0.7
        )
    )
    
    result = await step.execute(context)
    
    assert 'answer' in result
    assert 'paris' in result['answer'].lower()
    assert result['confidence'] >= 0.6
    print(f"✓ OpenAI Self-Consistency: {result['answer']} (confidence: {result['confidence']:.0%})")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_self_consistency_anthropic():
    """Test Self-Consistency with Anthropic."""
    service = LLMProviderService()
    service.register_provider(
        "anthropic",
        LLMProvider.ANTHROPIC,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-3-haiku-20240307"
    )
    
    adapter = LLMProviderAdapter(service)
    context = {'services': {'llm': adapter}}
    
    step = SelfConsistencyStep(
        name="calculation",
        prompt="What is 15 * 8?",
        config=SelfConsistencyConfig(
            num_samples=3,
            voting_strategy=VotingStrategy.MAJORITY,
            temperature=0.3
        )
    )
    
    result = await step.execute(context)
    
    assert 'answer' in result
    assert '120' in str(result['answer'])
    print(f"✓ Anthropic Self-Consistency: {result['answer']}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_self_consistency_google():
    """Test Self-Consistency with Google."""
    service = LLMProviderService()
    service.register_provider(
        "google",
        LLMProvider.GOOGLE,
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-2.0-flash-exp"
    )
    
    adapter = LLMProviderAdapter(service)
    context = {'services': {'llm': adapter}}
    
    step = SelfConsistencyStep(
        name="threshold",
        prompt="What is 10 + 5?",
        config=SelfConsistencyConfig(
            num_samples=3,
            voting_strategy=VotingStrategy.CONFIDENCE_THRESHOLD,
            confidence_threshold=0.5,
            temperature=0.0
        )
    )
    
    result = await step.execute(context)
    
    assert 'answer' in result
    assert '15' in str(result['answer'])
    print(f"✓ Google Self-Consistency: {result['answer']}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_react_agent_openai():
    """Test ReAct Agent with OpenAI."""
    service = LLMProviderService()
    service.register_provider(
        "openai",
        LLMProvider.OPENAI,
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-3.5-turbo",
        temperature=0.0
    )
    
    class LLMAdapter:
        async def generate(self, prompt: str, model=None, temperature=0.7):
            response = await service.generate_completion(prompt, temperature=temperature, max_tokens=500)
            return response.content
    
    class ToolAdapter:
        async def use_tool(self, tool_name: str, tool_input: str) -> str:
            if tool_name == "calculator":
                try:
                    result = eval(tool_input, {"__builtins__": {}}, {})
                    return f"Result: {result}"
                except:
                    return "Error calculating"
            return f"Tool {tool_name} not available"
    
    context = {'services': {'llm': LLMAdapter(), 'tools': ToolAdapter()}}
    
    agent = ReActAgent(
        name="calc",
        task="Calculate 25 * 4",
        config=ReActConfig(max_iterations=5, tools=["calculator"])
    )
    
    result = await agent.execute(context)
    
    assert 'answer' in result
    assert '100' in str(result['answer'])
    print(f"✓ OpenAI ReAct: {result['answer']}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_react_agent_anthropic():
    """Test ReAct Agent with Anthropic."""
    service = LLMProviderService()
    service.register_provider(
        "anthropic",
        LLMProvider.ANTHROPIC,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-3-haiku-20240307",
        temperature=0.0
    )
    
    class LLMAdapter:
        async def generate(self, prompt: str, model=None, temperature=0.7):
            response = await service.generate_completion(prompt, temperature=temperature, max_tokens=500)
            return response.content
    
    class ToolAdapter:
        async def use_tool(self, tool_name: str, tool_input: str) -> str:
            if tool_name == "calculator":
                try:
                    result = eval(tool_input, {"__builtins__": {}}, {})
                    return f"Result: {result}"
                except:
                    return "Error calculating"
            return f"Tool {tool_name} not available"
    
    context = {'services': {'llm': LLMAdapter(), 'tools': ToolAdapter()}}
    
    agent = ReActAgent(
        name="calc",
        task="What is 12 + 8?",
        config=ReActConfig(max_iterations=5, tools=["calculator"])
    )
    
    result = await agent.execute(context)
    
    assert 'answer' in result
    assert '20' in str(result['answer'])
    print(f"✓ Anthropic ReAct: {result['answer']}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_react_agent_google():
    """Test ReAct Agent with Google."""
    service = LLMProviderService()
    service.register_provider(
        "google",
        LLMProvider.GOOGLE,
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-2.0-flash-exp",
        temperature=0.0
    )
    
    class LLMAdapter:
        async def generate(self, prompt: str, model=None, temperature=0.7):
            response = await service.generate_completion(prompt, temperature=temperature, max_tokens=500)
            return response.content
    
    class ToolAdapter:
        async def use_tool(self, tool_name: str, tool_input: str) -> str:
            if tool_name == "calculator":
                try:
                    result = eval(tool_input, {"__builtins__": {}}, {})
                    return f"Result: {result}"
                except:
                    return "Error calculating"
            return f"Tool {tool_name} not available"
    
    context = {'services': {'llm': LLMAdapter(), 'tools': ToolAdapter()}}
    
    agent = ReActAgent(
        name="calc",
        task="If a product costs $50 with 20% discount, what's the final price?",
        config=ReActConfig(max_iterations=5, tools=["calculator"])
    )
    
    result = await agent.execute(context)
    
    assert 'answer' in result
    assert '40' in str(result['answer'])
    print(f"✓ Google ReAct: {result['answer']}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tree_of_thoughts_openai():
    """Test Tree of Thoughts with OpenAI."""
    service = LLMProviderService()
    service.register_provider(
        "openai",
        LLMProvider.OPENAI,
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-3.5-turbo",
        temperature=0.7
    )
    
    adapter = LLMProviderAdapter(service)
    context = {'services': {'llm': adapter}}
    
    step = TreeOfThoughtsStep(
        name="problem_solver",
        prompt="How can we reduce plastic waste in oceans?",
        config=ToTConfig(
            branching_factor=3,
            max_depth=2,
            pruning_strategy=PruningStrategy.THRESHOLD,
            threshold=0.6
        )
    )
    
    result = await step.execute(context)
    
    assert 'solution' in result
    assert 'tree_depth' in result
    assert result['tree_depth'] > 0
    assert len(result['solution']) > 20
    print(f"✓ OpenAI ToT: Depth {result['tree_depth']}, confidence: {result['confidence']:.2f}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tree_of_thoughts_anthropic():
    """Test Tree of Thoughts with Anthropic."""
    service = LLMProviderService()
    service.register_provider(
        "anthropic",
        LLMProvider.ANTHROPIC,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-3-haiku-20240307",
        temperature=0.8
    )
    
    adapter = LLMProviderAdapter(service)
    context = {'services': {'llm': adapter}}
    
    step = TreeOfThoughtsStep(
        name="creative",
        prompt="Design a mobile app feature to help people learn languages",
        config=ToTConfig(
            branching_factor=2,
            max_depth=2,
            pruning_strategy=PruningStrategy.BEST_FIRST
        )
    )
    
    result = await step.execute(context)
    
    assert 'solution' in result
    assert 'confidence' in result
    assert result['confidence'] > 0
    print(f"✓ Anthropic ToT: Confidence {result['confidence']:.2f}")


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skip(reason="Google API quota limit - use when quota available")
async def test_tree_of_thoughts_google():
    """Test Tree of Thoughts with Google."""
    service = LLMProviderService()
    service.register_provider(
        "google",
        LLMProvider.GOOGLE,
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-2.0-flash-exp",
        temperature=0.7
    )
    
    adapter = LLMProviderAdapter(service)
    context = {'services': {'llm': adapter}}
    
    step = TreeOfThoughtsStep(
        name="beam",
        prompt="What are the best strategies to improve team productivity?",
        config=ToTConfig(
            branching_factor=3,
            max_depth=2,
            pruning_strategy=PruningStrategy.BEAM_SEARCH,
            beam_width=2
        )
    )
    
    result = await step.execute(context)
    
    assert 'solution' in result
    assert 'tree_depth' in result
    print(f"✓ Google ToT: Depth {result['tree_depth']}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cot_validation():
    """Test CoT with validation enabled."""
    service = LLMProviderService()
    service.register_provider(
        "openai",
        LLMProvider.OPENAI,
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-3.5-turbo",
        temperature=0.0
    )
    
    adapter = LLMProviderAdapter(service)
    context = {'services': {'llm': adapter}}
    
    step = ChainOfThoughtStep(
        name="validated",
        prompt="What is 8 * 7?",
        config=CoTConfig(reasoning_depth=3, validation_step=True)
    )
    
    result = await step.execute(context)
    
    assert 'answer' in result
    assert '56' in result['answer']
    print(f"✓ CoT with validation: {result['answer']}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_self_consistency_parallel():
    """Test Self-Consistency with parallel execution."""
    service = LLMProviderService()
    service.register_provider(
        "openai",
        LLMProvider.OPENAI,
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-3.5-turbo"
    )
    
    adapter = LLMProviderAdapter(service)
    context = {'services': {'llm': adapter}}
    
    step = SelfConsistencyStep(
        name="parallel",
        prompt="What is the capital of Japan?",
        config=SelfConsistencyConfig(
            num_samples=5,
            voting_strategy=VotingStrategy.MAJORITY,
            temperature=0.7,
            parallel_execution=True
        )
    )
    
    result = await step.execute(context)
    
    assert 'answer' in result
    assert 'tokyo' in result['answer'].lower()
    print(f"✓ Parallel Self-Consistency: {result['answer']} ({result['confidence']:.0%})")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_self_consistency_weighted_voting():
    """Test Self-Consistency with weighted voting."""
    service = LLMProviderService()
    service.register_provider(
        "anthropic",
        LLMProvider.ANTHROPIC,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-3-haiku-20240307"
    )
    
    adapter = LLMProviderAdapter(service)
    context = {'services': {'llm': adapter}}
    
    step = SelfConsistencyStep(
        name="weighted",
        prompt="What is 9 * 9?",
        config=SelfConsistencyConfig(
            num_samples=5,
            voting_strategy=VotingStrategy.WEIGHTED,
            temperature=0.5
        )
    )
    
    result = await step.execute(context)
    
    assert 'answer' in result
    assert '81' in str(result['answer'])
    print(f"✓ Weighted voting: {result['answer']}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cot_different_formats():
    """Test CoT with different reasoning formats."""
    service = LLMProviderService()
    service.register_provider(
        "google",
        LLMProvider.GOOGLE,
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-2.0-flash-exp",
        temperature=0.0
    )
    
    adapter = LLMProviderAdapter(service)
    context = {'services': {'llm': adapter}}
    
    # Test bullet format
    step = ChainOfThoughtStep(
        name="bullet",
        prompt="What is 5 * 3?",
        config=CoTConfig(reasoning_depth=2, format="bullet")
    )
    
    result = await step.execute(context)
    
    assert 'answer' in result
    assert '15' in str(result['answer'])
    print(f"✓ CoT bullet format: {result['answer']}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cross_provider_consistency():
    """Test that all providers give consistent answers to basic math."""
    providers = [
        ("openai", LLMProvider.OPENAI, "gpt-3.5-turbo", os.getenv("OPENAI_API_KEY")),
        ("anthropic", LLMProvider.ANTHROPIC, "claude-3-haiku-20240307", os.getenv("ANTHROPIC_API_KEY")),
        ("google", LLMProvider.GOOGLE, "gemini-2.0-flash-exp", os.getenv("GOOGLE_API_KEY")),
    ]
    
    results = {}
    
    for name, provider_type, model, api_key in providers:
        service = LLMProviderService()
        service.register_provider(name, provider_type, api_key=api_key, model=model, temperature=0.0)
        
        adapter = LLMProviderAdapter(service)
        context = {'services': {'llm': adapter}}
        
        step = ChainOfThoughtStep(
            name=f"test_{name}",
            prompt="What is 7 * 8?",
            config=CoTConfig(reasoning_depth=3)
        )
        
        result = await step.execute(context)
        results[name] = result['answer']
        assert '56' in result['answer'], f"{name} should get 56"
    
    print(f"✓ Cross-provider consistency: All got 56")
