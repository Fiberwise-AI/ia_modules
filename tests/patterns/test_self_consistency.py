"""
Tests for Self-Consistency pattern.
"""

import pytest
from ia_modules.pipeline.test_utils import create_test_execution_context
from ia_modules.patterns import SelfConsistencyStep, SelfConsistencyConfig, VotingStrategy


class MockLLMService:
    """Mock LLM service for testing."""
    
    def __init__(self, responses: list):
        self.responses = responses
        self.call_count = 0
    
    async def generate(self, prompt: str, model: str, temperature: float):
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response


@pytest.mark.asyncio
async def test_self_consistency_majority_vote():
    """Test self-consistency with majority voting."""
    
    # Majority should agree on Canberra
    responses = [
        "The capital of Australia is Canberra",
        "Answer: Canberra",
        "Therefore, the answer is Canberra",
        "The capital is Canberra",
        "Answer: Canberra"
    ]
    
    llm = MockLLMService(responses)
    context = {'services': {'llm': llm}}
    
    step = SelfConsistencyStep(
        name="test",
        prompt="What is the capital of Australia?",
        config=SelfConsistencyConfig(
            num_samples=5,
            voting_strategy=VotingStrategy.MAJORITY
        )
    )
    
    result = await step.execute(context)
    
    assert 'answer' in result
    assert 'canberra' in result['answer'].lower()
    assert result['confidence'] >= 0.6  # At least 60% agreement
    assert result['num_samples'] == 5


@pytest.mark.asyncio
async def test_self_consistency_unanimous():
    """Test unanimous voting requirement."""
    
    # All agree on same answer
    responses = [
        "Answer: 42",
        "Answer: 42",
        "Answer: 42",
    ]
    
    llm = MockLLMService(responses)
    context = {'services': {'llm': llm}}
    
    step = SelfConsistencyStep(
        name="test",
        prompt="What is the meaning of life?",
        config=SelfConsistencyConfig(
            num_samples=3,
            voting_strategy=VotingStrategy.UNANIMOUS
        )
    )
    
    result = await step.execute(context)
    
    assert result['confidence'] == 1.0
    assert '42' in result['answer']


@pytest.mark.asyncio
async def test_self_consistency_no_consensus():
    """Test case with no unanimous consensus."""
    
    responses = [
        "Answer: Option A",
        "Answer: Option B",
        "Answer: Option C",
    ]
    
    llm = MockLLMService(responses)
    context = {'services': {'llm': llm}}
    
    step = SelfConsistencyStep(
        name="test",
        prompt="Pick one",
        config=SelfConsistencyConfig(
            num_samples=3,
            voting_strategy=VotingStrategy.UNANIMOUS
        )
    )
    
    result = await step.execute(context)
    
    assert 'No consensus' in result['answer']
    assert result['confidence'] < 1.0


@pytest.mark.asyncio
async def test_self_consistency_threshold_pass():
    """Test threshold voting when threshold is met."""
    
    responses = [
        "Answer: Yes",
        "Answer: Yes",
        "Answer: Yes",
        "Answer: No",
    ]
    
    llm = MockLLMService(responses)
    context = {'services': {'llm': llm}}
    
    step = SelfConsistencyStep(
        name="test",
        prompt="Is the sky blue?",
        config=SelfConsistencyConfig(
            num_samples=4,
            voting_strategy=VotingStrategy.CONFIDENCE_THRESHOLD,
            confidence_threshold=0.6
        )
    )
    
    result = await step.execute(context)
    
    # 3/4 = 0.75 > 0.6 threshold
    assert result['confidence'] == 0.75
    assert 'yes' in result['answer'].lower()
    assert 'Low confidence' not in result['answer']


@pytest.mark.asyncio
async def test_self_consistency_threshold_fail():
    """Test threshold voting when threshold is not met."""
    
    responses = [
        "Answer: Yes",
        "Answer: No",
        "Answer: Maybe",
    ]
    
    llm = MockLLMService(responses)
    context = {'services': {'llm': llm}}
    
    step = SelfConsistencyStep(
        name="test",
        prompt="Question?",
        config=SelfConsistencyConfig(
            num_samples=3,
            voting_strategy=VotingStrategy.CONFIDENCE_THRESHOLD,
            confidence_threshold=0.6
        )
    )
    
    result = await step.execute(context)
    
    # 1/3 = 0.33 < 0.6 threshold
    assert 'Low confidence' in result['answer']


@pytest.mark.asyncio
async def test_self_consistency_weighted_vote():
    """Test weighted voting based on reasoning quality."""
    
    # Longer, more detailed reasoning should score higher
    responses = [
        "Answer: A",  # Short
        "Let me think: First... Second... Third... Therefore: B",  # Detailed
        "Answer: A",  # Short
    ]
    
    llm = MockLLMService(responses)
    context = {'services': {'llm': llm}}
    
    step = SelfConsistencyStep(
        name="test",
        prompt="Choose",
        config=SelfConsistencyConfig(
            num_samples=3,
            voting_strategy=VotingStrategy.WEIGHTED
        )
    )
    
    result = await step.execute(context)
    
    # Weighted vote should favor the detailed reasoning
    assert 'answer' in result
    assert result['confidence'] > 0.0


@pytest.mark.asyncio
async def test_self_consistency_parallel_execution():
    """Test parallel vs sequential execution."""
    
    responses = ["Answer: Test"] * 5
    
    llm = MockLLMService(responses)
    context = {'services': {'llm': llm}}
    
    # Parallel
    step_parallel = SelfConsistencyStep(
        name="test",
        prompt="Test",
        config=SelfConsistencyConfig(
            num_samples=5,
            parallel_execution=True
        )
    )
    
    result = await step_parallel.execute(context)
    assert result['num_samples'] == 5
    
    # Sequential
    llm.call_count = 0
    step_sequential = SelfConsistencyStep(
        name="test",
        prompt="Test",
        config=SelfConsistencyConfig(
            num_samples=5,
            parallel_execution=False
        )
    )
    
    result = await step_sequential.execute(context)
    assert result['num_samples'] == 5


@pytest.mark.asyncio
async def test_self_consistency_no_llm_service():
    """Test fails gracefully without LLM service."""
    
    context = {'services': {}}
    
    step = SelfConsistencyStep(
        name="test",
        prompt="Test"
    )
    
    with pytest.raises(ValueError, match="LLM service not found"):
        await step.execute(context)


@pytest.mark.asyncio
async def test_self_consistency_answer_extraction():
    """Test various answer extraction patterns."""
    
    responses = [
        "Therefore, the answer is Tokyo",
        "Answer: Tokyo",
        "Conclusion: Tokyo",
        "Result: Tokyo",
        "Final answer: Tokyo",
    ]
    
    llm = MockLLMService(responses)
    context = {'services': {'llm': llm}}
    
    step = SelfConsistencyStep(
        name="test",
        prompt="What is the capital of Japan?",
        config=SelfConsistencyConfig(num_samples=5)
    )
    
    result = await step.execute(context)
    
    assert result['confidence'] >= 0.6  # Strong agreement
    assert 'tokyo' in result['answer'].lower()
