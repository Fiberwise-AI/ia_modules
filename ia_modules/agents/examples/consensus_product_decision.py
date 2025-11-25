"""
Consensus Collaboration Example: Product Decision

Demonstrates consensus-building pattern where stakeholder agents work
together to reach agreement on product decisions through voting and refinement.

Use Case:
- Product Manager proposes feature
- Stakeholders (Engineering, Design, Sales, Support) vote and discuss
- Iterative refinement until consensus
- Final decision reflects collective agreement

Pattern: Consensus (Agreement-Based)
"""

import asyncio
import logging
from typing import Dict, Any
from enum import Enum

from ..core import AgentRole
from ..state import StateManager
from ..communication import MessageBus
from ..collaboration_patterns.consensus import (
    ConsensusCollaboration,
    ConsensusStrategy,
    VoteType,
)
from ..base_agent import BaseCollaborativeAgent


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class StakeholderRole(Enum):
    """Different stakeholder perspectives."""
    ENGINEERING = "engineering"
    DESIGN = "design"
    SALES = "sales"
    SUPPORT = "support"
    PRODUCT = "product"


class StakeholderAgent(BaseCollaborativeAgent):
    """
    Stakeholder agent that participates in consensus decisions.

    Each stakeholder has a different perspective and priorities.
    """

    def __init__(self, role: AgentRole, state_manager: StateManager,
                 message_bus: MessageBus, stakeholder_role: StakeholderRole):
        """
        Initialize stakeholder agent.

        Args:
            role: Agent role
            state_manager: State manager
            message_bus: Message bus
            stakeholder_role: Type of stakeholder
        """
        super().__init__(role, state_manager, message_bus)
        self.stakeholder_role = stakeholder_role

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute consensus participation (discuss/vote/refine).

        Args:
            input_data: Contains action_type and proposal

        Returns:
            Discussion, vote, or refinement suggestion
        """
        action_type = input_data.get("action_type", "")
        proposal = input_data.get("proposal", {})

        if action_type == "discuss":
            return await self._discuss_proposal(proposal)
        elif action_type == "vote":
            return await self._vote_on_proposal(proposal)
        elif action_type == "suggest_refinement":
            return await self._suggest_refinement(proposal)
        else:
            return {"status": "unknown_action"}

    async def _discuss_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Discuss proposal from stakeholder perspective."""
        await asyncio.sleep(0.1)  # Simulate thinking

        feature = proposal.get("content", {})
        feature.get("name", "Unknown Feature")

        discussions = {
            StakeholderRole.ENGINEERING: {
                "perspective": "Technical Feasibility",
                "points": [
                    f"Estimated implementation: {self._estimate_effort(feature)} weeks",
                    f"Technical risk: {self._assess_tech_risk(feature)}",
                    "Need to consider: Database schema changes, API design, testing strategy",
                    "Concerns: Performance impact on existing features"
                ],
                "priority_factors": ["implementation time", "technical debt", "maintainability"]
            },
            StakeholderRole.DESIGN: {
                "perspective": "User Experience",
                "points": [
                    f"UX complexity: {self._assess_ux_complexity(feature)}",
                    "Design considerations: Consistency with existing UI, accessibility",
                    "User research needed: A/B testing plan, user feedback collection",
                    "Concerns: Learning curve for users"
                ],
                "priority_factors": ["user satisfaction", "design consistency", "accessibility"]
            },
            StakeholderRole.SALES: {
                "perspective": "Market Impact",
                "points": [
                    f"Revenue potential: {self._assess_revenue_impact(feature)}",
                    "Customer requests: Multiple enterprise clients asking for this",
                    "Competitive analysis: 3 of 5 competitors already have similar features",
                    "Concerns: Sales cycle length, pricing strategy"
                ],
                "priority_factors": ["revenue impact", "competitive advantage", "customer demand"]
            },
            StakeholderRole.SUPPORT: {
                "perspective": "Support Impact",
                "points": [
                    f"Support complexity: {self._assess_support_complexity(feature)}",
                    "Documentation needs: User guides, troubleshooting docs, FAQs",
                    "Training required: Support team onboarding, customer education",
                    "Concerns: Support ticket volume increase"
                ],
                "priority_factors": ["support burden", "documentation quality", "training needs"]
            },
            StakeholderRole.PRODUCT: {
                "perspective": "Product Strategy",
                "points": [
                    f"Strategic alignment: {self._assess_strategic_fit(feature)}",
                    "Roadmap impact: Fits Q2 objectives, may delay other features",
                    "User segment: Targets enterprise users, complements existing features",
                    "Concerns: Scope creep, timeline pressure"
                ],
                "priority_factors": ["strategic fit", "roadmap alignment", "user impact"]
            }
        }

        discussion = discussions.get(self.stakeholder_role, {
            "perspective": "General",
            "points": ["Need more information"],
            "priority_factors": []
        })

        return {
            "status": "success",
            "stakeholder": self.stakeholder_role.value,
            "perspective": discussion["perspective"],
            "points": discussion["points"],
            "priority_factors": discussion["priority_factors"],
            "agent": self.agent_id
        }

    async def _vote_on_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Vote on proposal based on stakeholder priorities."""
        await asyncio.sleep(0.1)

        feature = proposal.get("content", {})
        iteration = proposal.get("iteration", 1)

        # Different voting logic per stakeholder
        vote, reasoning = self._determine_vote(feature, iteration)

        return {
            "status": "success",
            "stakeholder": self.stakeholder_role.value,
            "vote": vote.value,
            "confidence": self._calculate_confidence(feature),
            "reasoning": reasoning,
            "conditions": self._get_approval_conditions(feature, vote),
            "agent": self.agent_id
        }

    def _determine_vote(self, feature: Dict[str, Any], iteration: int) -> tuple:
        """Determine vote based on stakeholder priorities."""
        effort = self._estimate_effort(feature)

        voting_logic = {
            StakeholderRole.ENGINEERING: (
                VoteType.APPROVE if effort <= 4 else VoteType.CONDITIONAL,
                f"Feasible within {effort} weeks" if effort <= 4 else f"Needs scope reduction ({effort} weeks)"
            ),
            StakeholderRole.DESIGN: (
                VoteType.APPROVE if self._assess_ux_complexity(feature) != "High" else VoteType.CONDITIONAL,
                "UX is well-designed" if self._assess_ux_complexity(feature) != "High" else "Needs UX simplification"
            ),
            StakeholderRole.SALES: (
                VoteType.APPROVE,  # Sales usually approves new features
                "High customer demand, competitive necessity"
            ),
            StakeholderRole.SUPPORT: (
                VoteType.CONDITIONAL if self._assess_support_complexity(feature) == "High" else VoteType.APPROVE,
                "Need documentation before launch" if self._assess_support_complexity(feature) == "High" else "Support impact is manageable"
            ),
            StakeholderRole.PRODUCT: (
                VoteType.APPROVE if iteration > 1 else VoteType.CONDITIONAL,
                "Aligns with strategy" if iteration > 1 else "Want to see stakeholder input first"
            )
        }

        return voting_logic.get(self.stakeholder_role, (VoteType.ABSTAIN, "Insufficient information"))

    def _suggest_refinement(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest refinements to proposal."""
        proposal.get("content", {})

        refinements = {
            StakeholderRole.ENGINEERING: {
                "suggestion": "Break into smaller phases",
                "details": "Phase 1: Core functionality (3 weeks), Phase 2: Advanced features (2 weeks)",
                "benefit": "Reduces risk, allows earlier release"
            },
            StakeholderRole.DESIGN: {
                "suggestion": "Simplify user interface",
                "details": "Remove advanced options from default view, add to 'Advanced' section",
                "benefit": "Better user experience, easier onboarding"
            },
            StakeholderRole.SALES: {
                "suggestion": "Add enterprise-tier pricing",
                "details": "Make feature available in Professional tier and above",
                "benefit": "Drives upgrades, justifies development cost"
            },
            StakeholderRole.SUPPORT: {
                "suggestion": "Include self-service documentation",
                "details": "Interactive tutorial, video guides, searchable FAQ",
                "benefit": "Reduces support burden, improves user satisfaction"
            },
            StakeholderRole.PRODUCT: {
                "suggestion": "Align with Q2 launch",
                "details": "Target release with quarterly product update",
                "benefit": "Coordinated marketing, better visibility"
            }
        }

        refinement = refinements.get(self.stakeholder_role, {
            "suggestion": "No specific refinement",
            "details": "",
            "benefit": ""
        })

        return {
            "status": "success",
            "stakeholder": self.stakeholder_role.value,
            "suggestion": refinement["suggestion"],
            "details": refinement["details"],
            "benefit": refinement["benefit"],
            "agent": self.agent_id
        }

    # Helper methods for simulation
    def _estimate_effort(self, feature: Dict[str, Any]) -> int:
        """Estimate implementation effort in weeks."""
        complexity = feature.get("complexity", "medium")
        effort_map = {"low": 2, "medium": 4, "high": 8}
        return effort_map.get(complexity, 4)

    def _assess_tech_risk(self, feature: Dict[str, Any]) -> str:
        """Assess technical risk."""
        complexity = feature.get("complexity", "medium")
        risk_map = {"low": "Low", "medium": "Medium", "high": "High"}
        return risk_map.get(complexity, "Medium")

    def _assess_ux_complexity(self, feature: Dict[str, Any]) -> str:
        """Assess UX complexity."""
        complexity = feature.get("complexity", "medium")
        return complexity.capitalize()

    def _assess_revenue_impact(self, feature: Dict[str, Any]) -> str:
        """Assess revenue impact."""
        complexity = feature.get("complexity", "medium")
        impact_map = {"low": "Moderate", "medium": "High", "high": "Very High"}
        return impact_map.get(complexity, "High")

    def _assess_support_complexity(self, feature: Dict[str, Any]) -> str:
        """Assess support complexity."""
        complexity = feature.get("complexity", "medium")
        return complexity.capitalize()

    def _assess_strategic_fit(self, feature: Dict[str, Any]) -> str:
        """Assess strategic fit."""
        return "Strong alignment with product vision"

    def _calculate_confidence(self, feature: Dict[str, Any]) -> float:
        """Calculate voting confidence."""
        return 0.8  # Simplified

    def _get_approval_conditions(self, feature: Dict[str, Any], vote: VoteType) -> list:
        """Get conditions for approval."""
        if vote != VoteType.CONDITIONAL:
            return []

        conditions_map = {
            StakeholderRole.ENGINEERING: ["Reduce scope to fit 4-week timeline", "Add comprehensive tests"],
            StakeholderRole.DESIGN: ["Conduct user testing", "Simplify advanced features"],
            StakeholderRole.SALES: [],
            StakeholderRole.SUPPORT: ["Complete documentation before launch", "Train support team"],
            StakeholderRole.PRODUCT: ["Get stakeholder alignment", "Confirm resource availability"]
        }

        return conditions_map.get(self.stakeholder_role, [])


async def run_product_decision_consensus_example():
    """
    Run consensus-based product decision example.

    Demonstrates:
    - Multi-stakeholder decision making
    - Voting and discussion
    - Iterative refinement
    - Consensus building
    """
    print("\n" + "="*80)
    print("CONSENSUS COLLABORATION EXAMPLE: Product Decision")
    print("="*80 + "\n")

    # Setup infrastructure
    state = StateManager(thread_id="product_decision_001")
    bus = MessageBus()

    # Create stakeholder agents
    stakeholders = []
    stakeholder_types = [
        (StakeholderRole.ENGINEERING, "Engineering"),
        (StakeholderRole.DESIGN, "Design/UX"),
        (StakeholderRole.SALES, "Sales"),
        (StakeholderRole.SUPPORT, "Customer Support"),
        (StakeholderRole.PRODUCT, "Product Management")
    ]

    for stakeholder_role, description in stakeholder_types:
        agent = StakeholderAgent(
            role=AgentRole(
                name=f"stakeholder_{stakeholder_role.value}",
                description=f"{description} stakeholder",
                system_prompt=f"You represent {description} perspective in product decisions."
            ),
            state_manager=state,
            message_bus=bus,
            stakeholder_role=stakeholder_role
        )
        stakeholders.append(agent)

    # Create consensus collaboration
    consensus = ConsensusCollaboration(
        agents=stakeholders,
        message_bus=bus,
        state_manager=state,
        strategy=ConsensusStrategy.MAJORITY,
        max_iterations=3
    )

    # Initialize
    print("Initializing product decision stakeholders...")
    await consensus.initialize()
    print(f"✓ Stakeholders initialized: {len(stakeholders)} participants\n")

    # Execute consensus process
    feature_proposal = {
        "name": "Real-time Collaboration Mode",
        "description": "Allow multiple users to edit documents simultaneously with live cursors and presence indicators",
        "complexity": "medium",
        "estimated_users": "All tiers",
        "strategic_priority": "high"
    }

    print(f"Feature Proposal: {feature_proposal['name']}")
    print(f"Description: {feature_proposal['description']}")
    print(f"Complexity: {feature_proposal['complexity']}")
    print("Consensus Strategy: MAJORITY")
    print("Max Iterations: 3\n")

    print("Starting consensus process...\n")

    result = await consensus.execute({
        "proposal": feature_proposal,
        "context": {
            "quarter": "Q2 2024",
            "budget": "available",
            "team_capacity": "80%"
        }
    })

    # Display results
    print("\n" + "="*80)
    print("CONSENSUS RESULTS")
    print("="*80 + "\n")

    print(f"Status: {result.get('status', '')}")
    print(f"Consensus Reached: {result.get('consensus_reached', False)}")
    print(f"Total Iterations: {result.get('iterations', 0)}")
    print(f"Final Agreement Level: {result.get('final_agreement_level', 0):.0%}")

    if 'iterations_detail' in result:
        print("\n" + "-"*80)
        print("ITERATION DETAILS")
        print("-"*80 + "\n")

        for iteration_data in result['iterations_detail']:
            iter_num = iteration_data.get('iteration', 0)
            print(f"\nITERATION {iter_num}:")

            # Discussion points
            if 'discussion' in iteration_data:
                print("\nStakeholder Perspectives:")
                for point in iteration_data['discussion']:
                    stakeholder = point.get('stakeholder', 'Unknown')
                    perspective = point.get('perspective', '')
                    print(f"\n  {stakeholder.upper()} - {perspective}:")
                    for p in point.get('points', []):
                        print(f"    • {p}")

            # Votes
            if 'votes' in iteration_data:
                print("\nVotes:")
                vote_summary = {}
                for vote in iteration_data['votes']:
                    vote_type = vote.get('vote', 'unknown')
                    vote_summary[vote_type] = vote_summary.get(vote_type, 0) + 1
                    stakeholder = vote.get('stakeholder', 'Unknown')
                    reasoning = vote.get('reasoning', '')
                    conditions = vote.get('conditions', [])

                    print(f"  {stakeholder.upper()}: {vote_type}")
                    print(f"    Reasoning: {reasoning}")
                    if conditions:
                        print(f"    Conditions: {', '.join(conditions)}")

                print(f"\n  Summary: {vote_summary}")
                print(f"  Agreement Level: {iteration_data.get('agreement_level', 0):.0%}")

            # Refinements
            if 'refinements' in iteration_data and iteration_data['refinements']:
                print("\nProposed Refinements:")
                for refinement in iteration_data['refinements']:
                    stakeholder = refinement.get('stakeholder', 'Unknown')
                    suggestion = refinement.get('suggestion', '')
                    details = refinement.get('details', '')
                    benefit = refinement.get('benefit', '')

                    print(f"\n  {stakeholder.upper()}: {suggestion}")
                    print(f"    Details: {details}")
                    print(f"    Benefit: {benefit}")

    if 'final_proposal' in result:
        print("\n" + "="*80)
        print("FINAL DECISION")
        print("="*80)
        print(f"\nFeature: {result['final_proposal'].get('content', {}).get('name', '')}")
        print(f"Status: {'APPROVED' if result.get('consensus_reached') else 'NEEDS MORE WORK'}")

        if result.get('consensus_reached'):
            print("\nNext Steps:")
            print("  1. Engineering team creates technical spec")
            print("  2. Design team creates mockups and prototypes")
            print("  3. Product team schedules for Q2 sprint")
            print("  4. Support team prepares documentation")
            print("  5. Sales team notified for customer communication")

    # Shutdown
    await consensus.shutdown()
    print("\n✓ Consensus collaboration shutdown complete")


async def main():
    """Main entry point for example."""
    await run_product_decision_consensus_example()


if __name__ == "__main__":
    # Run example
    asyncio.run(main())
