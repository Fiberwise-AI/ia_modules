"""
Debate Collaboration Example: Code Review

Demonstrates adversarial debate pattern where agents argue different
perspectives on code quality and design decisions.

Use Case:
- Proponent argues for accepting code changes
- Opponent argues against or identifies issues
- Moderator ensures balanced discussion
- Result synthesizes best practices

Pattern: Debate (Adversarial)
"""

import asyncio
import logging
from typing import Dict, Any, List

from ..core import AgentRole
from ..state import StateManager
from ..communication import MessageBus, MessageType
from ..collaboration_patterns.debate import (
    DebateCollaboration,
    DebateRole,
)
from ..base_agent import BaseCollaborativeAgent


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class CodeReviewDebateAgent(BaseCollaborativeAgent):
    """
    Debate agent specialized for code review discussions.

    Can take either proponent or opponent role.
    """

    def __init__(self, role: AgentRole, state_manager: StateManager,
                 message_bus: MessageBus, debate_role: DebateRole,
                 expertise: str = "general"):
        """
        Initialize code review debate agent.

        Args:
            role: Agent role
            state_manager: State manager
            message_bus: Message bus
            debate_role: PROPONENT or OPPONENT
            expertise: Area of expertise (security/performance/maintainability/general)
        """
        super().__init__(role, state_manager, message_bus)
        self.debate_role = debate_role
        self.expertise = expertise

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute debate turn (opening/argument/closing).

        Args:
            input_data: Contains turn_type, topic, previous_arguments

        Returns:
            Argument or statement
        """
        turn_type = input_data.get("turn_type", "argument")
        topic = input_data.get("topic", "")
        previous_args = input_data.get("previous_arguments", [])

        if turn_type == "opening":
            return await self._opening_statement(topic)
        elif turn_type == "argument":
            return await self._make_argument(topic, previous_args)
        elif turn_type == "closing":
            return await self._closing_statement(topic, previous_args)
        else:
            return {"status": "unknown_turn_type"}

    async def _opening_statement(self, topic: str) -> Dict[str, Any]:
        """Generate opening statement."""
        await asyncio.sleep(0.1)  # Simulate thinking

        statements = {
            (DebateRole.PROPONENT, "security"): f"""
OPENING STATEMENT - PROPONENT (Security Focus):

I argue IN FAVOR of accepting this code change.

Security Analysis:
✓ Input validation is properly implemented
✓ SQL injection risks mitigated through parameterized queries
✓ Authentication checks are in place
✓ No sensitive data exposed in logs

The security posture of this code meets our standards.
""",
            (DebateRole.PROPONENT, "performance"): f"""
OPENING STATEMENT - PROPONENT (Performance Focus):

I argue IN FAVOR of accepting this code change.

Performance Analysis:
✓ Time complexity is O(n log n), acceptable for expected data sizes
✓ Database queries use proper indexing
✓ Caching strategy is appropriate
✓ No memory leaks identified

The performance characteristics are sound.
""",
            (DebateRole.OPPONENT, "security"): f"""
OPENING STATEMENT - OPPONENT (Security Focus):

I argue AGAINST accepting this code change in its current form.

Security Concerns:
✗ Error messages leak system information
✗ Rate limiting not implemented on API endpoints
✗ CSRF tokens missing from forms
✗ Dependency versions have known vulnerabilities

These security gaps must be addressed before merging.
""",
            (DebateRole.OPPONENT, "performance"): f"""
OPENING STATEMENT - OPPONENT (Performance Focus):

I argue AGAINST accepting this code change in its current form.

Performance Concerns:
✗ N+1 query problem in user data loading
✗ Missing database indexes on frequently queried columns
✗ Large payload sizes without compression
✗ No query result caching

These issues will cause scalability problems.
""",
        }

        key = (self.debate_role, self.expertise)
        statement = statements.get(key, f"Opening statement from {self.debate_role.value}")

        return {
            "status": "success",
            "role": self.debate_role.value,
            "expertise": self.expertise,
            "statement": statement,
            "agent": self.agent_id
        }

    async def _make_argument(self, topic: str, previous_args: List[Dict]) -> Dict[str, Any]:
        """Generate argument based on debate role and previous arguments."""
        await asyncio.sleep(0.1)  # Simulate thinking

        # Extract key points from opponent
        opponent_points = []
        for arg in previous_args:
            if arg.get("role") != self.debate_role.value:
                opponent_points.append(arg.get("statement", ""))

        if self.debate_role == DebateRole.PROPONENT:
            argument = self._proponent_rebuttal(opponent_points)
        else:
            argument = self._opponent_rebuttal(opponent_points)

        return {
            "status": "success",
            "role": self.debate_role.value,
            "expertise": self.expertise,
            "statement": argument,
            "agent": self.agent_id
        }

    def _proponent_rebuttal(self, opponent_points: List[str]) -> str:
        """Generate proponent rebuttal."""
        if self.expertise == "security":
            return """
REBUTTAL - PROPONENT (Security):

While the opponent raises valid concerns, I counter with:

1. Error Messages: These are DEBUG-only, not exposed in production
2. Rate Limiting: Already handled at load balancer level
3. CSRF Tokens: SameSite cookies provide equivalent protection
4. Dependencies: Vulnerabilities are in dev dependencies, not production

The security concerns are either mitigated or not applicable to production.
"""
        else:  # performance
            return """
REBUTTAL - PROPONENT (Performance):

The performance concerns are overstated:

1. N+1 Queries: Only occurs with <10 items, negligible impact
2. Missing Indexes: Existing indexes cover 95% of query patterns
3. Payload Size: Clients request full data, compression would waste CPU
4. Caching: Cold start cost is minimal, premature optimization

The current performance is acceptable for our use case.
"""

    def _opponent_rebuttal(self, opponent_points: List[str]) -> str:
        """Generate opponent rebuttal."""
        if self.expertise == "security":
            return """
REBUTTAL - OPPONENT (Security):

The proponent's justifications are insufficient:

1. Debug-only errors: Configuration mistakes can expose these in production
2. Load balancer rate limiting: Application-level limits are still needed
3. SameSite cookies: Not supported in all browsers we must support
4. Dev dependencies: Still create supply chain risks

Defense in depth requires addressing ALL these issues.
"""
        else:  # performance
            return """
REBUTTAL - OPPONENT (Performance):

The proponent underestimates real-world impact:

1. N+1 Queries: As data grows, this will cause exponential slowdown
2. Index Coverage: That 5% creates timeout issues for some users
3. Compression: CPU cost is minimal vs network transfer savings
4. Caching: First users after deploy experience unacceptable delays

These issues must be fixed NOW before they become technical debt.
"""

    async def _closing_statement(self, topic: str, all_arguments: List[Dict]) -> Dict[str, Any]:
        """Generate closing statement."""
        await asyncio.sleep(0.1)

        if self.debate_role == DebateRole.PROPONENT:
            closing = """
CLOSING STATEMENT - PROPONENT:

This code change represents a net positive:
- Core functionality is solid
- Raised concerns are mitigatable
- Blocking merge delays valuable features
- Issues can be addressed in follow-up PRs

Recommendation: ACCEPT with minor follow-up tasks
"""
        else:
            closing = """
CLOSING STATEMENT - OPPONENT:

This code change requires revisions before merge:
- Identified issues create real risks
- "Fix later" often becomes "never fixed"
- Quality standards must be maintained
- Proper review prevents future incidents

Recommendation: REQUEST CHANGES, then re-review
"""

        return {
            "status": "success",
            "role": self.debate_role.value,
            "expertise": self.expertise,
            "statement": closing,
            "agent": self.agent_id
        }


class CodeReviewModeratorAgent(BaseCollaborativeAgent):
    """Moderator for code review debates."""

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Moderate debate turn."""
        turn_type = input_data.get("turn_type", "")

        if turn_type == "introduction":
            return {
                "status": "success",
                "message": "Moderator: Let's begin the code review debate. Proponents will argue for accepting the changes, opponents will argue against. Let's keep discussion focused on technical merits."
            }
        elif turn_type == "transition":
            return {
                "status": "success",
                "message": "Moderator: Thank you for those arguments. Let's move to rebuttals."
            }
        else:
            return {
                "status": "success",
                "message": "Moderator: Please proceed."
            }


async def run_code_review_debate_example():
    """
    Run code review debate example.

    Demonstrates:
    - Adversarial argumentation
    - Multiple perspectives (security, performance)
    - Structured debate rounds
    - Synthesis of best arguments
    """
    print("\n" + "="*80)
    print("DEBATE COLLABORATION EXAMPLE: Code Review")
    print("="*80 + "\n")

    # Setup infrastructure
    state = StateManager(thread_id="code_review_debate_001")
    bus = MessageBus()

    # Create proponent agents (arguing FOR accepting code)
    proponent_security = CodeReviewDebateAgent(
        role=AgentRole(
            name="proponent_security",
            description="Argues for code acceptance (security focus)",
            system_prompt="You evaluate code security and argue for acceptance when security is adequate."
        ),
        state_manager=state,
        message_bus=bus,
        debate_role=DebateRole.PROPONENT,
        expertise="security"
    )

    proponent_performance = CodeReviewDebateAgent(
        role=AgentRole(
            name="proponent_performance",
            description="Argues for code acceptance (performance focus)",
            system_prompt="You evaluate code performance and argue for acceptance when performance is acceptable."
        ),
        state_manager=state,
        message_bus=bus,
        debate_role=DebateRole.PROPONENT,
        expertise="performance"
    )

    # Create opponent agents (arguing AGAINST accepting code)
    opponent_security = CodeReviewDebateAgent(
        role=AgentRole(
            name="opponent_security",
            description="Identifies security issues",
            system_prompt="You critically examine code for security vulnerabilities."
        ),
        state_manager=state,
        message_bus=bus,
        debate_role=DebateRole.OPPONENT,
        expertise="security"
    )

    opponent_performance = CodeReviewDebateAgent(
        role=AgentRole(
            name="opponent_performance",
            description="Identifies performance issues",
            system_prompt="You critically examine code for performance problems."
        ),
        state_manager=state,
        message_bus=bus,
        debate_role=DebateRole.OPPONENT,
        expertise="performance"
    )

    # Create moderator
    moderator = CodeReviewModeratorAgent(
        role=AgentRole(
            name="moderator",
            description="Facilitates code review debate",
            system_prompt="You moderate code review discussions, ensuring balanced perspectives."
        ),
        state_manager=state,
        message_bus=bus
    )

    # Create debate collaboration
    debate = DebateCollaboration(
        proponents=[proponent_security, proponent_performance],
        opponents=[opponent_security, opponent_performance],
        moderator=moderator,
        message_bus=bus,
        state_manager=state
    )

    # Initialize
    print("Initializing code review debate team...")
    await debate.initialize()
    print("✓ Team initialized: 2 proponents, 2 opponents, 1 moderator\n")

    # Execute debate
    code_change = "PR #1234: Add user data export API endpoint"

    print(f"Code Change: {code_change}")
    print("Debate Rounds: 2")
    print("\nStarting debate...\n")

    result = await debate.execute({
        "topic": code_change,
        "rounds": 2
    })

    # Display results
    print("\n" + "="*80)
    print("DEBATE RESULTS")
    print("="*80 + "\n")

    print(f"Topic: {result.get('topic', '')}")
    print(f"Total Rounds: {result.get('total_rounds', 0)}")
    print(f"Status: {result.get('status', '')}")

    if 'opening_statements' in result:
        print("\n" + "-"*80)
        print("OPENING STATEMENTS")
        print("-"*80 + "\n")
        for side, statements in result['opening_statements'].items():
            print(f"\n{side.upper()}:")
            for stmt in statements:
                print(stmt.get('statement', ''))
                print()

    if 'debate_rounds' in result:
        print("\n" + "-"*80)
        print("DEBATE ROUNDS")
        print("-"*80 + "\n")
        for round_data in result['debate_rounds']:
            round_num = round_data.get('round', 0)
            print(f"\nROUND {round_num}:")
            for arg in round_data.get('arguments', []):
                print(f"\n{arg.get('role', '').upper()}:")
                print(arg.get('statement', ''))

    if 'closing_statements' in result:
        print("\n" + "-"*80)
        print("CLOSING STATEMENTS")
        print("-"*80 + "\n")
        for side, statements in result['closing_statements'].items():
            print(f"\n{side.upper()}:")
            for stmt in statements:
                print(stmt.get('statement', ''))
                print()

    if 'synthesis' in result:
        print("\n" + "="*80)
        print("SYNTHESIS")
        print("="*80)
        print(result['synthesis'])

    # Shutdown
    await debate.shutdown()
    print("\n✓ Debate team shutdown complete")


async def main():
    """Main entry point for example."""
    await run_code_review_debate_example()


if __name__ == "__main__":
    # Run example
    asyncio.run(main())
