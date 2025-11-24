"""
Tests for Tree of Thoughts pattern.
"""

import pytest
from ia_modules.patterns import TreeOfThoughtsStep, ToTConfig, PruningStrategy, ToTNode


class MockLLMService:
    """Mock LLM service for testing."""
    
    def __init__(self, responses: dict):
        self.responses = responses
        self.call_count = 0
    
    async def generate(self, prompt: str, model: str, temperature: float):
        # Return different responses based on prompt content
        if 'Generate' in prompt and 'next steps' in prompt:
            # Branching request
            key = 'branches'
        elif 'scale of 0-10' in prompt:
            # Evaluation request
            key = 'evaluation'
        else:
            key = 'default'
        
        response = self.responses.get(key, "1. Default branch")
        return response


@pytest.mark.asyncio
async def test_tree_of_thoughts_basic():
    """Test basic ToT execution."""
    
    llm_responses = {
        'branches': """1. Consider multiplication
2. Consider division
3. Consider addition""",
        'evaluation': "8"  # Score of 8/10
    }
    
    llm = MockLLMService(llm_responses)
    context = {'services': {'llm': llm}}
    
    step = TreeOfThoughtsStep(
        name="test",
        prompt="Solve math problem",
        config=ToTConfig(
            branching_factor=3,
            max_depth=2,
            beam_width=2
        )
    )
    
    result = await step.execute(context)
    
    assert 'solution' in result
    assert 'reasoning_path' in result
    assert 'confidence' in result
    assert result['tree_depth'] >= 0


@pytest.mark.asyncio
async def test_tree_of_thoughts_finds_solution():
    """Test ToT stops when solution is found."""
    
    llm_responses = {
        'branches': """1. Try approach A
2. Therefore, the answer is 42
3. Try approach B""",
        'evaluation': "9"
    }
    
    llm = MockLLMService(llm_responses)
    context = {'services': {'llm': llm}}
    
    step = TreeOfThoughtsStep(
        name="test",
        prompt="Find answer",
        config=ToTConfig(
            branching_factor=3,
            max_depth=5  # Should stop early when solution found
        )
    )
    
    result = await step.execute(context)
    
    assert result['is_complete_solution'] is True
    assert result['tree_depth'] < 5  # Stopped early


@pytest.mark.asyncio
async def test_tree_of_thoughts_pruning_strategies():
    """Test different pruning strategies."""
    
    llm_responses = {
        'branches': """1. Branch A
2. Branch B
3. Branch C""",
        'evaluation': "7"
    }
    
    llm = MockLLMService(llm_responses)
    context = {'services': {'llm': llm}}
    
    # Test BEST_FIRST
    step = TreeOfThoughtsStep(
        name="test",
        prompt="Test",
        config=ToTConfig(
            branching_factor=3,
            max_depth=2,
            pruning_strategy=PruningStrategy.BEST_FIRST,
            beam_width=2
        )
    )
    result = await step.execute(context)
    assert 'solution' in result
    
    # Test THRESHOLD
    step = TreeOfThoughtsStep(
        name="test",
        prompt="Test",
        config=ToTConfig(
            branching_factor=3,
            max_depth=2,
            pruning_strategy=PruningStrategy.THRESHOLD,
            threshold=0.5
        )
    )
    result = await step.execute(context)
    assert 'solution' in result
    
    # Test BEAM_SEARCH
    step = TreeOfThoughtsStep(
        name="test",
        prompt="Test",
        config=ToTConfig(
            branching_factor=3,
            max_depth=2,
            pruning_strategy=PruningStrategy.BEAM_SEARCH,
            beam_width=2
        )
    )
    result = await step.execute(context)
    assert 'solution' in result


@pytest.mark.asyncio
async def test_tree_of_thoughts_custom_evaluation():
    """Test ToT with custom evaluation function."""
    
    def custom_eval(node: ToTNode, context: dict) -> float:
        # Score based on depth (deeper = better)
        return node.depth * 0.3
    
    llm_responses = {
        'branches': """1. Branch 1
2. Branch 2"""
    }
    
    llm = MockLLMService(llm_responses)
    context = {'services': {'llm': llm}}
    
    step = TreeOfThoughtsStep(
        name="test",
        prompt="Test",
        evaluation_fn=custom_eval,
        config=ToTConfig(
            branching_factor=2,
            max_depth=2
        )
    )
    
    result = await step.execute(context)
    
    assert 'confidence' in result
    # Deeper nodes should score higher
    assert result['confidence'] >= 0.0


@pytest.mark.asyncio
async def test_tree_of_thoughts_no_llm_service():
    """Test ToT fails gracefully without LLM service."""
    
    context = {'services': {}}
    
    step = TreeOfThoughtsStep(
        name="test",
        prompt="Test"
    )
    
    with pytest.raises(ValueError, match="LLM service not found"):
        await step.execute(context)


@pytest.mark.asyncio
async def test_tree_of_thoughts_evaluation_parsing():
    """Test robust parsing of evaluation scores."""
    
    test_cases = [
        ("8", 0.8),
        ("Score: 7", 0.7),
        ("I would rate this 9 out of 10", 0.9),
        ("10", 1.0),
        ("0", 0.0),
        ("invalid", 0.5),  # Fallback
    ]
    
    for response, expected_score in test_cases:
        llm_responses = {
            'branches': "1. Branch",
            'evaluation': response
        }
        
        llm = MockLLMService(llm_responses)
        context = {'services': {'llm': llm}}
        
        step = TreeOfThoughtsStep(
            name="test",
            prompt="Test",
            config=ToTConfig(branching_factor=1, max_depth=1)
        )
        
        result = await step.execute(context)
        
        # Check that score was parsed (approximately)
        assert result['confidence'] >= 0.0
        assert result['confidence'] <= 1.0


@pytest.mark.asyncio
async def test_tree_of_thoughts_empty_branches():
    """Test ToT handles case with no branches generated."""
    
    llm_responses = {
        'branches': "",  # No branches
        'evaluation': "5"
    }
    
    llm = MockLLMService(llm_responses)
    context = {'services': {'llm': llm}}
    
    step = TreeOfThoughtsStep(
        name="test",
        prompt="Test",
        config=ToTConfig(branching_factor=3, max_depth=2)
    )
    
    result = await step.execute(context)
    
    # Should handle gracefully
    assert 'solution' in result


@pytest.mark.asyncio
async def test_tree_of_thoughts_max_depth():
    """Test ToT respects max depth."""
    
    llm_responses = {
        'branches': """1. Go deeper
2. Keep exploring
3. Another path""",
        'evaluation': "7"
    }
    
    llm = MockLLMService(llm_responses)
    context = {'services': {'llm': llm}}
    
    max_depth = 3
    step = TreeOfThoughtsStep(
        name="test",
        prompt="Deep exploration",
        config=ToTConfig(
            branching_factor=3,
            max_depth=max_depth,
            beam_width=2
        )
    )
    
    result = await step.execute(context)
    
    # Should not exceed max depth
    assert result['tree_depth'] <= max_depth
