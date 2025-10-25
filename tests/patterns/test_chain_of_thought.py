"""
Tests for Chain-of-Thought pattern.
"""

import pytest
from ia_modules.patterns import ChainOfThoughtStep, CoTConfig


class MockLLMService:
    """Mock LLM service for testing."""
    
    def __init__(self, response: str):
        self.response = response
        self.calls = []
    
    async def generate(self, prompt: str, model: str, temperature: float):
        self.calls.append({
            'prompt': prompt,
            'model': model,
            'temperature': temperature
        })
        return self.response


@pytest.mark.asyncio
async def test_chain_of_thought_basic():
    """Test basic CoT execution."""
    
    llm_response = """1. We need to find 15% of 80
2. Convert 15% to decimal: 0.15
3. Multiply: 0.15 × 80 = 12
Therefore, the answer is 12"""
    
    llm = MockLLMService(llm_response)
    context = {
        'services': {'llm': llm}
    }
    
    step = ChainOfThoughtStep(
        name="test",
        prompt="What is 15% of 80?",
        model="gpt-4",
        config=CoTConfig(show_reasoning=True, validation_step=False)
    )
    
    result = await step.execute(context)
    
    assert 'answer' in result
    assert 'reasoning' in result
    assert len(result['reasoning']) > 0
    assert result['model_used'] == 'gpt-4'
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_chain_of_thought_with_validation():
    """Test CoT with validation."""
    
    llm_response = """1. Start with 120 apples
2. Morning: sell 30% = 36 apples
3. Remaining: 120 - 36 = 84
4. Afternoon: sell 25% of 84 = 21
5. Final: 84 - 21 = 63
Therefore, 63 apples remain"""
    
    llm = MockLLMService(llm_response)
    context = {
        'services': {'llm': llm}
    }
    
    step = ChainOfThoughtStep(
        name="test",
        prompt="Store has 120 apples. Sells 30% morning, 25% of remainder afternoon. How many left?",
        config=CoTConfig(validation_step=True, retry_on_validation_failure=False)
    )
    
    result = await step.execute(context)
    
    assert 'answer' in result
    assert '63' in result['answer']


@pytest.mark.asyncio
async def test_chain_of_thought_formats():
    """Test different CoT formats."""
    
    for format_type in ['numbered', 'bullet', 'freeform']:
        llm = MockLLMService("Step 1\nStep 2\nFinal answer: 42")
        context = {'services': {'llm': llm}}
        
        step = ChainOfThoughtStep(
            name="test",
            prompt="Test",
            config=CoTConfig(format=format_type, validation_step=False)
        )
        
        result = await step.execute(context)
        assert 'answer' in result


@pytest.mark.asyncio
async def test_chain_of_thought_invalid_reasoning():
    """Test CoT with invalid reasoning triggers retry."""
    
    # First response is uncertain
    llm = MockLLMService("I don't know the answer")
    context = {'services': {'llm': llm}}
    
    step = ChainOfThoughtStep(
        name="test",
        prompt="What is 2+2?",
        config=CoTConfig(validation_step=True, retry_on_validation_failure=True)
    )
    
    result = await step.execute(context)
    
    # Should have made 2 calls (original + retry)
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_chain_of_thought_no_llm_service():
    """Test CoT fails gracefully without LLM service."""
    
    context = {'services': {}}
    
    step = ChainOfThoughtStep(
        name="test",
        prompt="Test"
    )
    
    with pytest.raises(ValueError, match="LLM service not found"):
        await step.execute(context)


@pytest.mark.asyncio
async def test_chain_of_thought_context_formatting():
    """Test CoT with context variable formatting."""
    
    llm = MockLLMService("The answer is 10")
    context = {
        'services': {'llm': llm},
        'number': 5
    }
    
    step = ChainOfThoughtStep(
        name="test",
        prompt="What is {number} times 2?",
        config=CoTConfig(validation_step=False)
    )
    
    result = await step.execute(context)
    
    assert 'What is 5 times 2?' in llm.calls[0]['prompt']
