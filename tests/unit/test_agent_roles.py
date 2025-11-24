"""
Tests for agents.roles module
"""

import pytest
from ia_modules.agents.core import AgentRole
from ia_modules.agents.roles import (
    PlannerAgent,
    ResearcherAgent,
    CoderAgent,
    CriticAgent,
    FormatterAgent
)


class MockStateManager:
    """Mock state manager for testing"""

    def __init__(self):
        self.state = {}
        self.read_calls = []
        self.write_calls = []

    async def get(self, key, default=None):
        """Get value from state"""
        self.read_calls.append((key, default))
        return self.state.get(key, default)

    async def set(self, key, value):
        """Set value in state"""
        self.write_calls.append((key, value))
        self.state[key] = value

    async def snapshot(self):
        """Get state snapshot"""
        return self.state.copy()


@pytest.mark.asyncio
class TestPlannerAgent:
    """Test PlannerAgent"""

    async def test_planner_basic_execution(self):
        """Test basic planner execution"""
        role = AgentRole(
            name="planner",
            description="Plans tasks",
            system_prompt="You are a planner"
        )
        state_manager = MockStateManager()
        agent = PlannerAgent(role, state_manager)

        result = await agent.execute({"request": "Build a web app"})

        assert result["status"] == "plan_created"
        assert result["steps"] == 4
        assert "plan" in state_manager.state
        assert "plan_metadata" in state_manager.state

    async def test_planner_from_state(self):
        """Test planner using request from state"""
        role = AgentRole(name="planner", description="Plans", system_prompt="prompt")
        state_manager = MockStateManager()
        state_manager.state["original_request"] = "Test request from state"
        agent = PlannerAgent(role, state_manager)

        result = await agent.execute({})

        assert result["status"] == "plan_created"
        assert result["steps"] > 0

    async def test_planner_no_request(self):
        """Test planner with no request"""
        role = AgentRole(name="planner", description="Plans", system_prompt="prompt")
        state_manager = MockStateManager()
        agent = PlannerAgent(role, state_manager)

        result = await agent.execute({})

        assert result["status"] == "no_request"
        assert result["steps"] == 0

    async def test_planner_creates_plan(self):
        """Test plan creation"""
        role = AgentRole(name="planner", description="Plans", system_prompt="prompt")
        state_manager = MockStateManager()
        agent = PlannerAgent(role, state_manager)

        plan = await agent._create_plan("Test request")

        assert isinstance(plan, list)
        assert len(plan) == 4
        assert all("step" in s for s in plan)
        assert all("action" in s for s in plan)
        assert all("description" in s for s in plan)

    async def test_planner_complexity_simple(self):
        """Test complexity estimation - simple"""
        role = AgentRole(name="planner", description="Plans", system_prompt="prompt")
        agent = PlannerAgent(role, MockStateManager())

        plan = [{"step": 1}, {"step": 2}]
        complexity = agent._estimate_complexity(plan)

        assert complexity == "simple"

    async def test_planner_complexity_moderate(self):
        """Test complexity estimation - moderate"""
        role = AgentRole(name="planner", description="Plans", system_prompt="prompt")
        agent = PlannerAgent(role, MockStateManager())

        plan = [{"step": i} for i in range(1, 5)]
        complexity = agent._estimate_complexity(plan)

        assert complexity == "moderate"

    async def test_planner_complexity_complex(self):
        """Test complexity estimation - complex"""
        role = AgentRole(name="planner", description="Plans", system_prompt="prompt")
        agent = PlannerAgent(role, MockStateManager())

        plan = [{"step": i} for i in range(1, 7)]
        complexity = agent._estimate_complexity(plan)

        assert complexity == "complex"

    async def test_planner_writes_metadata(self):
        """Test that planner writes metadata to state"""
        role = AgentRole(name="planner", description="Plans", system_prompt="prompt")
        state_manager = MockStateManager()
        agent = PlannerAgent(role, state_manager)

        await agent.execute({"request": "Test"})

        metadata = state_manager.state["plan_metadata"]
        assert "total_steps" in metadata
        assert "complexity" in metadata
        assert metadata["total_steps"] == 4
        assert metadata["complexity"] == "moderate"


@pytest.mark.asyncio
class TestResearcherAgent:
    """Test ResearcherAgent"""

    async def test_researcher_basic_execution(self):
        """Test basic researcher execution"""
        role = AgentRole(
            name="researcher",
            description="Researches topics",
            allowed_tools=["web_search"]
        )
        state_manager = MockStateManager()
        agent = ResearcherAgent(role, state_manager)

        result = await agent.execute({"topic": "Python programming"})

        assert result["status"] == "research_complete"
        assert result["findings_count"] > 0
        assert "research_findings" in state_manager.state
        assert "research_sources" in state_manager.state

    async def test_researcher_from_plan(self):
        """Test researcher using plan from state"""
        role = AgentRole(name="researcher", description="Researches", system_prompt="prompt")
        state_manager = MockStateManager()
        state_manager.state["plan"] = [
            {"step": 1, "description": "Research topic A"},
            {"step": 2, "description": "Research topic B"}
        ]
        state_manager.state["current_step"] = 0
        agent = ResearcherAgent(role, state_manager)

        result = await agent.execute({})

        assert result["status"] == "research_complete"
        assert result["findings_count"] > 0

    async def test_researcher_from_original_request(self):
        """Test researcher falls back to original_request"""
        role = AgentRole(name="researcher", description="Researches", system_prompt="prompt")
        state_manager = MockStateManager()
        state_manager.state["original_request"] = "Find info about AI"
        agent = ResearcherAgent(role, state_manager)

        result = await agent.execute({})

        assert result["status"] == "research_complete"

    async def test_researcher_conduct_research(self):
        """Test research generation"""
        role = AgentRole(name="researcher", description="Researches", system_prompt="prompt")
        agent = ResearcherAgent(role, MockStateManager())

        findings = await agent._conduct_research("AI topics")

        assert "topic" in findings
        assert "facts" in findings
        assert "sources" in findings
        assert "confidence" in findings
        assert len(findings["facts"]) == 3
        assert len(findings["sources"]) == 2

    async def test_researcher_writes_sources(self):
        """Test that researcher writes sources separately"""
        role = AgentRole(name="researcher", description="Researches", system_prompt="prompt")
        state_manager = MockStateManager()
        agent = ResearcherAgent(role, state_manager)

        await agent.execute({"topic": "Test topic"})

        assert "research_sources" in state_manager.state
        assert len(state_manager.state["research_sources"]) == 2


@pytest.mark.asyncio
class TestCoderAgent:
    """Test CoderAgent"""

    async def test_coder_basic_execution(self):
        """Test basic coder execution"""
        role = AgentRole(
            name="coder",
            description="Writes code",
            allowed_tools=["python_exec"]
        )
        state_manager = MockStateManager()
        state_manager.state["specifications"] = {"language": "python"}
        agent = CoderAgent(role, state_manager)

        result = await agent.execute({})

        assert result["status"] == "code_generated"
        assert result["lines_of_code"] > 0
        assert "code_snippets" in state_manager.state
        assert "code_metadata" in state_manager.state

    async def test_coder_from_plan(self):
        """Test coder using plan when no specifications"""
        role = AgentRole(name="coder", description="Codes", system_prompt="prompt")
        state_manager = MockStateManager()
        state_manager.state["plan"] = [{"step": 1, "action": "write_code"}]
        agent = CoderAgent(role, state_manager)

        result = await agent.execute({})

        assert result["status"] == "code_generated"

    async def test_coder_generate_code(self):
        """Test code generation"""
        role = AgentRole(name="coder", description="Codes", system_prompt="prompt")
        agent = CoderAgent(role, MockStateManager())

        code = await agent._generate_code({"requirements": "test"})

        assert "language" in code
        assert "files" in code
        assert "lines" in code
        assert code["language"] == "python"
        assert len(code["files"]) == 2

    async def test_coder_writes_metadata(self):
        """Test that coder writes metadata"""
        role = AgentRole(name="coder", description="Codes", system_prompt="prompt")
        state_manager = MockStateManager()
        agent = CoderAgent(role, state_manager)

        await agent.execute({})

        metadata = state_manager.state["code_metadata"]
        assert "language" in metadata
        assert "lines_of_code" in metadata
        assert "files" in metadata
        assert metadata["language"] == "python"
        assert len(metadata["files"]) == 2


@pytest.mark.asyncio
class TestCriticAgent:
    """Test CriticAgent"""

    async def test_critic_basic_execution(self):
        """Test basic critic execution"""
        role = AgentRole(
            name="critic",
            description="Reviews work",
            metadata={"criteria": ["quality", "accuracy"]}
        )
        state_manager = MockStateManager()
        state_manager.state["code_snippets"] = {
            "language": "python",
            "files": {"main.py": "code"}
        }
        agent = CriticAgent(role, state_manager)

        result = await agent.execute({"artifact_key": "code_snippets"})

        assert result["status"] == "review_complete"
        assert "issues_found" in result
        assert "approved" in result
        assert "critique" in state_manager.state
        assert "approved" in state_manager.state

    async def test_critic_custom_criteria(self):
        """Test critic with custom criteria"""
        role = AgentRole(name="critic", description="Reviews", system_prompt="prompt")
        state_manager = MockStateManager()
        state_manager.state["artifact"] = {"data": "test"}
        agent = CriticAgent(role, state_manager, criteria=["custom1", "custom2"])

        assert agent.criteria == ["custom1", "custom2"]

    async def test_critic_no_artifact(self):
        """Test critic when artifact not found"""
        role = AgentRole(name="critic", description="Reviews", system_prompt="prompt")
        state_manager = MockStateManager()
        agent = CriticAgent(role, state_manager)

        result = await agent.execute({"artifact_key": "missing_key"})

        assert result["status"] == "no_artifact"
        assert result["approved"] is False

    async def test_critic_finds_language_issue(self):
        """Test critic finds missing language"""
        role = AgentRole(name="critic", description="Reviews", system_prompt="prompt")
        agent = CriticAgent(role, MockStateManager())

        artifact = {"files": {"main.py": "code"}}
        issues = await agent._review(artifact, ["language"])

        assert len(issues) == 1
        assert issues[0]["criterion"] == "language"
        assert issues[0]["severity"] == "high"

    async def test_critic_finds_completeness_issue(self):
        """Test critic finds missing files"""
        role = AgentRole(name="critic", description="Reviews", system_prompt="prompt")
        agent = CriticAgent(role, MockStateManager())

        artifact = {"language": "python"}
        issues = await agent._review(artifact, ["completeness"])

        assert len(issues) == 1
        assert issues[0]["criterion"] == "completeness"

    async def test_critic_approves_complete_artifact(self):
        """Test critic approves complete artifact"""
        role = AgentRole(name="critic", description="Reviews", system_prompt="prompt")
        state_manager = MockStateManager()
        state_manager.state["code"] = {
            "language": "python",
            "files": {"main.py": "code"}
        }
        agent = CriticAgent(role, state_manager, criteria=["quality"])

        result = await agent.execute({"artifact_key": "code"})

        assert result["approved"] is True
        assert result["issues_found"] == 0

    async def test_critic_default_criteria(self):
        """Test critic uses default criteria"""
        role = AgentRole(name="critic", description="Reviews", system_prompt="prompt")
        state_manager = MockStateManager()
        agent = CriticAgent(role, state_manager)

        assert agent.criteria == ["quality", "accuracy"]

    async def test_critic_metadata_criteria(self):
        """Test critic uses criteria from metadata"""
        role = AgentRole(
            name="critic",
            description="Reviews",
            system_prompt="prompt",
            metadata={"criteria": ["test1", "test2"]}
        )
        state_manager = MockStateManager()
        agent = CriticAgent(role, state_manager)

        assert agent.criteria == ["test1", "test2"]


@pytest.mark.asyncio
class TestFormatterAgent:
    """Test FormatterAgent"""

    async def test_formatter_basic_execution(self):
        """Test basic formatter execution"""
        role = AgentRole(
            name="formatter",
            description="Formats output",
            metadata={"default_format": "markdown"}
        )
        state_manager = MockStateManager()
        state_manager.state["plan"] = [{"description": "Step 1"}]
        agent = FormatterAgent(role, state_manager)

        result = await agent.execute({"format": "markdown"})

        assert result["status"] == "formatted"
        assert result["format"] == "markdown"
        assert result["length"] > 0
        assert "final_answer" in state_manager.state

    async def test_formatter_default_format(self):
        """Test formatter uses default format"""
        role = AgentRole(
            name="formatter",
            description="Formats",
            system_prompt="prompt",
            metadata={"default_format": "json"}
        )
        state_manager = MockStateManager()
        state_manager.state["data"] = "test"
        agent = FormatterAgent(role, state_manager)

        result = await agent.execute({})

        assert result["format"] == "json"

    async def test_formatter_json_output(self):
        """Test JSON formatting"""
        role = AgentRole(name="formatter", description="Formats", system_prompt="prompt")
        agent = FormatterAgent(role, MockStateManager())

        state = {"key": "value", "number": 42}
        output = await agent._format_output(state, "json")

        assert '"key": "value"' in output
        assert '"number": 42' in output

    async def test_formatter_markdown_output(self):
        """Test markdown formatting"""
        role = AgentRole(name="formatter", description="Formats", system_prompt="prompt")
        agent = FormatterAgent(role, MockStateManager())

        state = {
            "plan": [
                {"description": "Step 1"},
                {"description": "Step 2"}
            ],
            "research_findings": {
                "facts": ["Fact 1", "Fact 2"]
            },
            "code_snippets": {
                "language": "python",
                "files": {
                    "main.py": "print('hello')"
                }
            }
        }
        output = await agent._format_output(state, "markdown")

        assert "# Agent Workflow Results" in output
        assert "## Plan" in output
        assert "Step 1" in output
        assert "## Research Findings" in output
        assert "Fact 1" in output
        assert "## Generated Code" in output
        assert "main.py" in output
        assert "```python" in output

    async def test_formatter_markdown_minimal_state(self):
        """Test markdown formatting with minimal state"""
        role = AgentRole(name="formatter", description="Formats", system_prompt="prompt")
        agent = FormatterAgent(role, MockStateManager())

        state = {"some_data": "value"}
        output = await agent._format_output(state, "markdown")

        assert "# Agent Workflow Results" in output

    async def test_formatter_text_output(self):
        """Test plain text formatting"""
        role = AgentRole(name="formatter", description="Formats", system_prompt="prompt")
        agent = FormatterAgent(role, MockStateManager())

        state = {"key": "value"}
        output = await agent._format_output(state, "text")

        assert "key" in output
        assert "value" in output

    async def test_formatter_uses_state_snapshot(self):
        """Test formatter gets state snapshot"""
        role = AgentRole(name="formatter", description="Formats", system_prompt="prompt")
        state_manager = MockStateManager()
        state_manager.state = {
            "plan": [{"step": 1}],
            "research": "data",
            "code": "snippet"
        }
        agent = FormatterAgent(role, state_manager)

        await agent.execute({"format": "json"})

        final_answer = state_manager.state["final_answer"]
        assert "plan" in final_answer
        assert "research" in final_answer
        assert "code" in final_answer
