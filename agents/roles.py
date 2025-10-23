"""
Built-in specialized agent roles.

Provides ready-to-use agents for common tasks:
- PlannerAgent: Decomposes requests into plans
- ResearcherAgent: Gathers information
- CoderAgent: Writes code
- CriticAgent: Reviews outputs
- FormatterAgent: Formats final output
"""

from typing import Dict, Any, List, Optional
from .core import BaseAgent, AgentRole


class PlannerAgent(BaseAgent):
    """
    Decomposes complex requests into actionable step-by-step plans.

    Reads from state:
    - original_request: The user's request

    Writes to state:
    - plan: List of steps to execute
    - plan_metadata: Additional planning info

    Example:
        >>> role = AgentRole(
        ...     name="planner",
        ...     description="Breaks down requests into steps",
        ...     system_prompt="You are a planning agent..."
        ... )
        >>> agent = PlannerAgent(role, state_manager)
        >>> result = await agent.execute({})
    """

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create step-by-step plan from request.

        Returns:
            Dict with status and number of steps
        """
        # Read original request
        request = await self.read_state("original_request")
        if not request:
            request = input_data.get("request", "")

        if not request:
            self.logger.warning("No request provided for planning")
            return {"status": "no_request", "steps": 0}

        self.logger.info(f"Creating plan for: {request[:100]}...")

        # Create plan (simplified - in real implementation, would use LLM)
        plan = await self._create_plan(request)

        # Write plan to state
        await self.write_state("plan", plan)
        await self.write_state("plan_metadata", {
            "total_steps": len(plan),
            "complexity": self._estimate_complexity(plan)
        })

        self.logger.info(f"Created plan with {len(plan)} steps")

        return {"status": "plan_created", "steps": len(plan)}

    async def _create_plan(self, request: str) -> List[Dict[str, Any]]:
        """
        Create plan from request.

        In real implementation, this would call an LLM.
        For now, creates a simple generic plan.

        Args:
            request: User request

        Returns:
            List of plan steps
        """
        # Simplified planning logic
        # Real implementation would use LLM with system prompt:
        # "Break down the following request into concrete steps..."

        steps = [
            {
                "step": 1,
                "action": "analyze_requirements",
                "description": "Analyze and understand the request",
                "agent": "analyzer"
            },
            {
                "step": 2,
                "action": "gather_information",
                "description": "Research and gather necessary information",
                "agent": "researcher"
            },
            {
                "step": 3,
                "action": "execute_task",
                "description": "Execute the main task",
                "agent": "worker"
            },
            {
                "step": 4,
                "action": "validate_results",
                "description": "Validate and verify results",
                "agent": "validator"
            }
        ]

        return steps

    def _estimate_complexity(self, plan: List[Dict[str, Any]]) -> str:
        """Estimate plan complexity."""
        num_steps = len(plan)
        if num_steps <= 2:
            return "simple"
        elif num_steps <= 5:
            return "moderate"
        else:
            return "complex"


class ResearcherAgent(BaseAgent):
    """
    Gathers information using search and knowledge tools.

    Reads from state:
    - plan: Plan to understand what to research
    - current_step: Current step being executed

    Writes to state:
    - research_findings: Dictionary of discovered facts
    - research_sources: List of sources consulted

    Example:
        >>> role = AgentRole(
        ...     name="researcher",
        ...     description="Gathers information",
        ...     allowed_tools=["web_search", "database_query"]
        ... )
        >>> agent = ResearcherAgent(role, state_manager)
    """

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Conduct research based on plan.

        Returns:
            Dict with findings count
        """
        # Get research topic
        topic = input_data.get("topic")
        if not topic:
            plan = await self.read_state("plan", [])
            current_step = await self.read_state("current_step", 0)
            if plan and current_step < len(plan):
                topic = plan[current_step].get("description", "")

        if not topic:
            topic = await self.read_state("original_request", "")

        self.logger.info(f"Researching: {topic[:100]}...")

        # Conduct research
        findings = await self._conduct_research(topic)

        # Write findings to state
        await self.write_state("research_findings", findings)
        await self.write_state("research_sources", findings.get("sources", []))

        self.logger.info(f"Research complete: {len(findings.get('facts', []))} findings")

        return {
            "status": "research_complete",
            "findings_count": len(findings.get("facts", []))
        }

    async def _conduct_research(self, topic: str) -> Dict[str, Any]:
        """
        Conduct research on topic.

        In real implementation, would use tools like web_search.

        Args:
            topic: Research topic

        Returns:
            Research findings
        """
        # Simplified research
        # Real implementation would use tools from self.role.allowed_tools

        findings = {
            "topic": topic,
            "facts": [
                f"Fact 1 about {topic}",
                f"Fact 2 about {topic}",
                f"Fact 3 about {topic}"
            ],
            "sources": [
                "source1.com",
                "source2.com"
            ],
            "confidence": 0.85
        }

        return findings


class CoderAgent(BaseAgent):
    """
    Writes code based on specifications.

    Reads from state:
    - plan: Plan with code requirements
    - research_findings: Research to inform code
    - specifications: Code specifications

    Writes to state:
    - code_snippets: Dictionary of code by language/file
    - code_metadata: Metadata about generated code

    Example:
        >>> role = AgentRole(
        ...     name="coder",
        ...     description="Writes code",
        ...     allowed_tools=["python_exec", "code_validator"]
        ... )
        >>> agent = CoderAgent(role, state_manager)
    """

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate code based on specifications.

        Returns:
            Dict with code status
        """
        # Get specifications
        specifications = await self.read_state("specifications", {})
        if not specifications:
            plan = await self.read_state("plan", [])
            specifications = {"plan": plan}

        self.logger.info("Generating code...")

        # Generate code
        code = await self._generate_code(specifications)

        # Write code to state
        await self.write_state("code_snippets", code)
        await self.write_state("code_metadata", {
            "language": code.get("language", "python"),
            "lines_of_code": code.get("lines", 0),
            "files": list(code.get("files", {}).keys())
        })

        self.logger.info(f"Code generated: {code.get('lines', 0)} lines")

        return {
            "status": "code_generated",
            "lines_of_code": code.get("lines", 0)
        }

    async def _generate_code(self, specifications: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate code from specifications.

        Args:
            specifications: Code requirements

        Returns:
            Generated code
        """
        # Simplified code generation
        # Real implementation would use LLM with system prompt:
        # "Generate code based on these specifications..."

        code = {
            "language": "python",
            "files": {
                "main.py": "def main():\n    pass\n",
                "utils.py": "def helper():\n    pass\n"
            },
            "lines": 4,
            "complexity": "low"
        }

        return code


class CriticAgent(BaseAgent):
    """
    Reviews outputs from other agents for quality and accuracy.

    Reads from state:
    - artifact_key: Key of artifact to review
    - criteria: Review criteria (from config)

    Writes to state:
    - critique: List of issues found
    - approved: Boolean indicating if artifact passes

    Example:
        >>> role = AgentRole(
        ...     name="critic",
        ...     description="Reviews code quality",
        ...     metadata={"criteria": ["accuracy", "completeness", "format"]}
        ... )
        >>> agent = CriticAgent(role, state_manager)
    """

    def __init__(self, role: AgentRole, state_manager: "StateManager",
                 criteria: Optional[List[str]] = None):
        """
        Initialize critic agent.

        Args:
            role: Agent role
            state_manager: State manager
            criteria: Review criteria (overrides role.metadata)
        """
        super().__init__(role, state_manager)
        self.criteria = criteria or role.metadata.get("criteria", ["quality", "accuracy"])

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Review artifact against criteria.

        Returns:
            Dict with review results
        """
        # Get artifact to review
        artifact_key = input_data.get("artifact_key", "code_snippets")
        artifact = await self.read_state(artifact_key)

        if not artifact:
            self.logger.warning(f"No artifact found at key: {artifact_key}")
            return {"status": "no_artifact", "approved": False}

        self.logger.info(f"Reviewing {artifact_key} against {len(self.criteria)} criteria")

        # Conduct review
        issues = await self._review(artifact, self.criteria)

        # Write critique to state
        await self.write_state("critique", issues)
        await self.write_state("approved", len(issues) == 0)

        approved = len(issues) == 0
        self.logger.info(f"Review complete: {'APPROVED' if approved else f'{len(issues)} issues found'}")

        return {
            "status": "review_complete",
            "issues_found": len(issues),
            "approved": approved
        }

    async def _review(self, artifact: Any, criteria: List[str]) -> List[Dict[str, str]]:
        """
        Review artifact against criteria.

        Args:
            artifact: Artifact to review
            criteria: Review criteria

        Returns:
            List of issues found
        """
        # Simplified review logic
        # Real implementation would use LLM with system prompt:
        # "Review the following against these criteria: {criteria}"

        issues = []

        # Example checks
        if isinstance(artifact, dict):
            if "language" in criteria and "language" not in artifact:
                issues.append({
                    "criterion": "language",
                    "issue": "Language not specified",
                    "severity": "high"
                })

            if "completeness" in criteria and not artifact.get("files"):
                issues.append({
                    "criterion": "completeness",
                    "issue": "No files generated",
                    "severity": "high"
                })

        return issues


class FormatterAgent(BaseAgent):
    """
    Formats final output in desired format (JSON, markdown, etc.).

    Reads from state:
    - All relevant state for synthesis
    - output_format: Desired format (from input or config)

    Writes to state:
    - final_answer: Formatted output

    Example:
        >>> role = AgentRole(
        ...     name="formatter",
        ...     description="Formats final output",
        ...     metadata={"default_format": "markdown"}
        ... )
        >>> agent = FormatterAgent(role, state_manager)
    """

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format final output.

        Returns:
            Dict with formatting status
        """
        # Get desired format
        output_format = input_data.get("format")
        if not output_format:
            output_format = self.role.metadata.get("default_format", "markdown")

        self.logger.info(f"Formatting output as {output_format}")

        # Get all relevant data from state
        state_snapshot = await self.get_state_snapshot()

        # Format output
        formatted = await self._format_output(state_snapshot, output_format)

        # Write final answer to state
        await self.write_state("final_answer", formatted)

        self.logger.info(f"Output formatted ({len(formatted)} chars)")

        return {
            "status": "formatted",
            "format": output_format,
            "length": len(formatted)
        }

    async def _format_output(self, state: Dict[str, Any], format_type: str) -> str:
        """
        Format output from state.

        Args:
            state: Current state
            format_type: Output format (json, markdown, text)

        Returns:
            Formatted output string
        """
        if format_type == "json":
            import json
            return json.dumps(state, indent=2)

        elif format_type == "markdown":
            # Format as markdown
            lines = ["# Agent Workflow Results\n"]

            if "plan" in state:
                lines.append("## Plan")
                plan = state["plan"]
                for step in plan:
                    lines.append(f"- {step.get('description', 'N/A')}")
                lines.append("")

            if "research_findings" in state:
                lines.append("## Research Findings")
                findings = state["research_findings"]
                for fact in findings.get("facts", []):
                    lines.append(f"- {fact}")
                lines.append("")

            if "code_snippets" in state:
                lines.append("## Generated Code")
                code = state["code_snippets"]
                for filename, content in code.get("files", {}).items():
                    lines.append(f"### {filename}")
                    lines.append(f"```{code.get('language', '')}")
                    lines.append(content)
                    lines.append("```\n")

            return "\n".join(lines)

        else:
            # Plain text
            return str(state)
