"""
Tests for ReAct agent pattern.
"""

import pytest
from ia_modules.pipeline.test_utils import create_test_execution_context
from ia_modules.patterns import ReActAgent, ReActConfig, AgentState


class MockLLMService:
    """Mock LLM service for testing."""
    
    def __init__(self, responses: list):
        self.responses = responses
        self.call_count = 0
    
    async def generate(self, prompt: str, model: str, temperature: float):
        response = self.responses[self.call_count]
        self.call_count += 1
        return response


class MockToolService:
    """Mock tool service for testing."""
    
    def __init__(self):
        self.executions = []
    
    async def execute(self, tool: str, inputs: dict, context: dict):
        self.executions.append({
            'tool': tool,
            'inputs': inputs
        })
        
        # Return mock results
        if tool == "search":
            return "Tokyo has 14 million people"
        elif tool == "calculator":
            expr = inputs.get('expression', '')
            if '14' in expr and '8.3' in expr:
                return "1.686"
            return "42"
        elif tool == "wikipedia":
            return "Wikipedia article content"
        
        return f"Result from {tool}"


@pytest.mark.asyncio
async def test_react_agent_basic():
    """Test basic ReAct agent execution."""
    
    llm_responses = [
        """Thought: I need to search for Tokyo population
Action: search
Action Input: {"query": "Tokyo population"}""",
        
        """Thought: Now I have the answer
Action: finish
Action Input: {"answer": "Tokyo has 14 million people"}"""
    ]
    
    llm = MockLLMService(llm_responses)
    tools = MockToolService()
    context = {
        'services': {
            'llm': llm,
            'tools': tools
        }
    }
    
    agent = ReActAgent(
        name="test",
        task="What is the population of Tokyo?",
        tools=["search"],
        config=ReActConfig(max_iterations=10, verbose=False)
    )
    
    result = await agent.execute(context)
    
    assert result['success'] is True
    assert 'Tokyo' in result['answer']
    assert len(result['trajectory']) == 1  # One search action
    assert tools.executions[0]['tool'] == 'search'


@pytest.mark.asyncio
async def test_react_agent_multi_step():
    """Test ReAct agent with multiple steps."""
    
    llm_responses = [
        """Thought: First search for Tokyo
Action: search
Action Input: {"query": "Tokyo population"}""",
        
        """Thought: Now search for NYC
Action: search
Action Input: {"query": "NYC population"}""",
        
        """Thought: Now compare them with calculator
Action: calculator
Action Input: {"expression": "14 / 8.3"}""",
        
        """Thought: Tokyo is 1.7x larger
Action: finish
Action Input: {"answer": "Tokyo (14M) is 1.7x larger than NYC (8.3M)"}"""
    ]
    
    llm = MockLLMService(llm_responses)
    tools = MockToolService()
    context = {
        'services': {
            'llm': llm,
            'tools': tools
        }
    }
    
    agent = ReActAgent(
        name="test",
        task="Compare Tokyo and NYC populations",
        tools=["search", "calculator"],
        config=ReActConfig(max_iterations=10, verbose=False)
    )
    
    result = await agent.execute(context)
    
    assert result['success'] is True
    assert len(result['trajectory']) == 3
    assert len(tools.executions) == 3
    assert tools.executions[0]['tool'] == 'search'
    assert tools.executions[1]['tool'] == 'search'
    assert tools.executions[2]['tool'] == 'calculator'


@pytest.mark.asyncio
async def test_react_agent_max_iterations():
    """Test ReAct agent hitting max iterations."""
    
    # Agent never finishes
    llm_responses = [
        """Thought: Keep thinking
Action: search
Action Input: {"query": "test"}"""
    ] * 5
    
    llm = MockLLMService(llm_responses)
    tools = MockToolService()
    context = {
        'services': {
            'llm': llm,
            'tools': tools
        }
    }
    
    agent = ReActAgent(
        name="test",
        task="Never-ending task",
        tools=["search"],
        config=ReActConfig(max_iterations=3, verbose=False)
    )
    
    result = await agent.execute(context)
    
    assert result['success'] is False
    assert 'incomplete' in result['answer'].lower()
    assert result['iterations'] == 3


@pytest.mark.asyncio
async def test_react_agent_unknown_tool():
    """Test ReAct agent with unknown tool."""
    
    llm_responses = [
        """Thought: Use unknown tool
Action: unknown_tool
Action Input: {"test": "value"}""",
        
        """Thought: Let me finish
Action: finish
Action Input: {"answer": "Done"}"""
    ]
    
    llm = MockLLMService(llm_responses)
    tools = MockToolService()
    context = {
        'services': {
            'llm': llm,
            'tools': tools
        }
    }
    
    agent = ReActAgent(
        name="test",
        task="Test",
        tools=["search"],
        config=ReActConfig(verbose=False)
    )
    
    result = await agent.execute(context)
    
    # Should handle error and continue
    assert result['success'] is True
    assert 'Error: Unknown tool' in result['trajectory'][0]['observation']


@pytest.mark.asyncio
async def test_react_agent_tool_error():
    """Test ReAct agent handling tool execution error."""
    
    class ErrorToolService:
        async def execute(self, tool: str, inputs: dict, context: dict):
            raise ValueError("Tool failed")
    
    llm_responses = [
        """Thought: Try search
Action: search
Action Input: {"query": "test"}""",
        
        """Thought: Finish anyway
Action: finish
Action Input: {"answer": "Done despite error"}"""
    ]
    
    llm = MockLLMService(llm_responses)
    tools = ErrorToolService()
    context = {
        'services': {
            'llm': llm,
            'tools': tools
        }
    }
    
    agent = ReActAgent(
        name="test",
        task="Test",
        tools=["search"],
        config=ReActConfig(verbose=False)
    )
    
    result = await agent.execute(context)
    
    # Should catch error and continue
    assert 'Error executing search' in result['trajectory'][0]['observation']


@pytest.mark.asyncio
async def test_react_agent_trajectory_tracking():
    """Test that agent properly tracks trajectory."""
    
    llm_responses = [
        """Thought: Step 1
Action: search
Action Input: {"query": "test1"}""",
        
        """Thought: Step 2
Action: search
Action Input: {"query": "test2"}""",
        
        """Thought: Done
Action: finish
Action Input: {"answer": "Complete"}"""
    ]
    
    llm = MockLLMService(llm_responses)
    tools = MockToolService()
    context = {
        'services': {
            'llm': llm,
            'tools': tools
        }
    }
    
    agent = ReActAgent(
        name="test",
        task="Test",
        tools=["search"],
        config=ReActConfig(verbose=False)
    )
    
    result = await agent.execute(context)
    
    assert len(result['trajectory']) == 2
    assert result['trajectory'][0]['thought'] == 'Step 1'
    assert result['trajectory'][0]['action'] == 'search'
    assert result['trajectory'][0]['action_input'] == {"query": "test1"}
    assert result['trajectory'][0]['observation'] is not None


@pytest.mark.asyncio
async def test_react_agent_no_services():
    """Test ReAct agent fails gracefully without services."""
    
    agent = ReActAgent(
        name="test",
        task="Test",
        tools=["search"]
    )
    
    # No LLM service
    with pytest.raises(ValueError, match="LLM service not found"):
        await agent.execute({'services': {'tools': MockToolService()}})
    
    # No tool service
    with pytest.raises(ValueError, match="Tool service not found"):
        await agent.execute({'services': {'llm': MockLLMService([])}})
