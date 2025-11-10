"""
Peer-to-Peer Collaboration Example: Knowledge Sharing

Demonstrates peer-to-peer pattern where expert agents share knowledge
as equals to build comprehensive understanding of a topic.

Use Case:
- Expert agents from different domains share knowledge
- Each peer contributes insights from their domain
- Peers build on each other's contributions
- Collective knowledge emerges through collaboration

Pattern: Peer-to-Peer (Equal Collaboration)
"""

import asyncio
import logging
from typing import Dict, Any, List
from enum import Enum

from ..core import AgentRole
from ..state import StateManager
from ..communication import MessageBus, MessageType
from ..collaboration_patterns.peer_to_peer import PeerToPeerCollaboration
from ..base_agent import BaseCollaborativeAgent


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class ExpertiseDomain(Enum):
    """Different domains of expertise."""
    COMPUTER_SCIENCE = "computer_science"
    MATHEMATICS = "mathematics"
    PHYSICS = "physics"
    BIOLOGY = "biology"


class ExpertPeerAgent(BaseCollaborativeAgent):
    """
    Expert peer agent with specific domain knowledge.

    Collaborates as equal with other expert peers.
    """

    def __init__(self, role: AgentRole, state_manager: StateManager,
                 message_bus: MessageBus, domain: ExpertiseDomain):
        """
        Initialize expert peer agent.

        Args:
            role: Agent role
            state_manager: State manager
            message_bus: Message bus
            domain: Area of expertise
        """
        super().__init__(role, state_manager, message_bus)
        self.domain = domain

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute peer collaboration action.

        Args:
            input_data: Contains action type and context

        Returns:
            Contribution, review, or refinement
        """
        action = input_data.get("action", "")

        if action == "contribute":
            return await self._make_contribution(input_data)
        elif action == "review":
            return await self._review_peer_work(input_data)
        elif action == "refine":
            return await self._refine_contribution(input_data)
        else:
            return {"status": "unknown_action"}

    async def _make_contribution(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Make contribution from domain expertise."""
        task = context.get("task", {})
        round_num = context.get("round", 1)
        previous_contributions = context.get("previous_contributions", [])

        topic = task.get("task", "Unknown Topic")

        await asyncio.sleep(0.1)  # Simulate thinking

        # Generate contribution based on domain and round
        if round_num == 1:
            # Initial contribution: domain perspective
            contribution = await self._initial_contribution(topic)
        else:
            # Build on previous round
            contribution = await self._build_on_previous(topic, previous_contributions)

        return {
            "status": "success",
            "domain": self.domain.value,
            "contribution_type": "initial" if round_num == 1 else "building",
            "content": contribution,
            "round": round_num,
            "agent": self.agent_id
        }

    async def _initial_contribution(self, topic: str) -> str:
        """Generate initial contribution from domain perspective."""
        contributions = {
            ExpertiseDomain.COMPUTER_SCIENCE: f"""
COMPUTER SCIENCE PERSPECTIVE on {topic}:

1. Computational Aspects:
   - Algorithmic complexity and computational feasibility
   - Data structures and processing pipelines
   - Scalability and distributed computing considerations
   - Machine learning and AI applications

2. Implementation Considerations:
   - Software architecture patterns
   - Performance optimization strategies
   - Security and reliability requirements
   - Testing and validation approaches

3. Current State of the Art:
   - Latest research in computational methods
   - Emerging technologies and frameworks
   - Open-source implementations available
   - Industry adoption and case studies

4. Connections to Other Domains:
   - Mathematical foundations for algorithms
   - Physical constraints on computation
   - Biological inspiration for algorithms (e.g., neural networks, genetic algorithms)
""",
            ExpertiseDomain.MATHEMATICS: f"""
MATHEMATICAL PERSPECTIVE on {topic}:

1. Mathematical Foundations:
   - Underlying mathematical theories and theorems
   - Formal proofs and logical frameworks
   - Statistical and probabilistic models
   - Optimization and game theory aspects

2. Quantitative Analysis:
   - Numerical methods and approximations
   - Error bounds and convergence properties
   - Complexity measures and bounds
   - Information-theoretic considerations

3. Abstract Structures:
   - Relevant algebraic structures (groups, rings, fields)
   - Topological and geometric properties
   - Category theory perspectives
   - Graph theory applications

4. Connections to Other Domains:
   - Computational complexity theory (CS)
   - Mathematical physics applications
   - Biomathematics and population models
""",
            ExpertiseDomain.PHYSICS: f"""
PHYSICS PERSPECTIVE on {topic}:

1. Physical Principles:
   - Fundamental laws governing the system
   - Energy considerations and thermodynamics
   - Quantum vs. classical descriptions
   - Relativistic effects (if applicable)

2. Physical Constraints:
   - Speed of light / information propagation limits
   - Energy requirements and efficiency
   - Physical resource limitations
   - Environmental factors and noise

3. Experimental Aspects:
   - Measurement techniques and instrumentation
   - Experimental validation methods
   - Precision and uncertainty considerations
   - Laboratory vs. real-world conditions

4. Connections to Other Domains:
   - Physical limits on computation (CS)
   - Mathematical physics frameworks
   - Biophysics applications
""",
            ExpertiseDomain.BIOLOGY: f"""
BIOLOGICAL PERSPECTIVE on {topic}:

1. Biological Systems:
   - Relevant biological processes and mechanisms
   - Evolutionary perspectives and adaptation
   - Cellular and molecular basis
   - Systems biology approaches

2. Bio-inspired Design:
   - Natural solutions to similar problems
   - Biomimicry opportunities
   - Self-organization and emergence
   - Robustness and resilience patterns

3. Applications in Life Sciences:
   - Medical and healthcare applications
   - Ecological and environmental impacts
   - Agricultural and biotechnology uses
   - Synthetic biology possibilities

4. Connections to Other Domains:
   - Bioinformatics and computational biology (CS)
   - Mathematical modeling of biological systems
   - Biophysics and biomechanics
"""
        }

        return contributions.get(
            self.domain,
            f"General perspective on {topic} from {self.domain.value}"
        )

    async def _build_on_previous(self, topic: str, previous_contributions: List[Dict]) -> str:
        """Build on previous round's contributions."""
        # Extract insights from other domains
        peer_insights = {}
        for contrib in previous_contributions:
            if contrib.get("domain") != self.domain.value:
                peer_insights[contrib.get("domain")] = contrib.get("content", "")

        building_content = f"BUILDING ON PEER CONTRIBUTIONS - {self.domain.value.upper()}:\n\n"

        # Cross-domain connections
        if self.domain == ExpertiseDomain.COMPUTER_SCIENCE:
            building_content += """
Cross-Domain Synthesis:

1. From Mathematics:
   - Implementing mathematical models computationally
   - Numerical algorithms for theoretical frameworks
   - Complexity analysis using mathematical tools

2. From Physics:
   - Physical computing paradigms (quantum, neuromorphic)
   - Constraints inform system design
   - Simulation of physical systems

3. From Biology:
   - Bio-inspired algorithms (genetic, swarm, neural)
   - Evolutionary computation methods
   - Modeling biological complexity
"""
        elif self.domain == ExpertiseDomain.MATHEMATICS:
            building_content += """
Cross-Domain Synthesis:

1. From Computer Science:
   - Computational complexity theory
   - Formal verification methods
   - Algorithmic information theory

2. From Physics:
   - Mathematical physics frameworks
   - Differential equations and dynamics
   - Statistical mechanics models

3. From Biology:
   - Population dynamics and epidemiology
   - Network theory in biological systems
   - Evolutionary game theory
"""
        elif self.domain == ExpertiseDomain.PHYSICS:
            building_content += """
Cross-Domain Synthesis:

1. From Computer Science:
   - Quantum computing and information
   - Physical limits of computation
   - Computational simulations of physics

2. From Mathematics:
   - Applying mathematical physics
   - Symmetry and conservation laws
   - Variational principles

3. From Biology:
   - Biophysics and biomechanics
   - Physical basis of life processes
   - Energy flows in living systems
"""
        else:  # BIOLOGY
            building_content += """
Cross-Domain Synthesis:

1. From Computer Science:
   - Computational biology and bioinformatics
   - AI for drug discovery
   - Biological data analysis

2. From Mathematics:
   - Mathematical biology and epidemiology
   - Network biology
   - Systems biology modeling

3. From Physics:
   - Biophysics and structural biology
   - Thermodynamics of living systems
   - Physical constraints on evolution
"""

        building_content += "\n4. Novel Insights:\n"
        building_content += f"   - Integration of {len(peer_insights)} domain perspectives creates holistic understanding\n"
        building_content += "   - Cross-pollination reveals unexpected connections\n"
        building_content += "   - Multi-disciplinary approach stronger than any single view\n"

        return building_content

    async def _review_peer_work(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Review work from peer agents."""
        contributions = context.get("contributions", [])

        reviews = []
        for contrib in contributions:
            if contrib.get("domain") != self.domain.value:
                reviews.append({
                    "reviewed_domain": contrib.get("domain"),
                    "feedback": f"Excellent insights from {contrib.get('domain')} perspective. Complements {self.domain.value} view well.",
                    "rating": 4.5
                })

        return {
            "status": "success",
            "reviewer": self.domain.value,
            "reviews": reviews,
            "agent": self.agent_id
        }

    async def _refine_contribution(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Refine contribution based on peer feedback."""
        original = context.get("original_contribution", "")
        feedback = context.get("peer_feedback", [])

        refined = f"{original}\n\nREFINED BASED ON PEER FEEDBACK:\n"
        refined += "- Incorporated cross-domain perspectives\n"
        refined += "- Clarified domain-specific terminology\n"
        refined += "- Strengthened connections to other fields\n"

        return {
            "status": "success",
            "domain": self.domain.value,
            "refined_content": refined,
            "agent": self.agent_id
        }


async def run_knowledge_sharing_example():
    """
    Run peer-to-peer knowledge sharing example.

    Demonstrates:
    - Equal collaboration among experts
    - Knowledge sharing across domains
    - Building on peer contributions
    - Collective intelligence emergence
    """
    print("\n" + "="*80)
    print("PEER-TO-PEER COLLABORATION EXAMPLE: Knowledge Sharing")
    print("="*80 + "\n")

    # Setup infrastructure
    state = StateManager(thread_id="knowledge_sharing_001")
    bus = MessageBus()

    # Create expert peer agents
    experts = []
    domains = [
        (ExpertiseDomain.COMPUTER_SCIENCE, "Computer Science Expert"),
        (ExpertiseDomain.MATHEMATICS, "Mathematics Expert"),
        (ExpertiseDomain.PHYSICS, "Physics Expert"),
        (ExpertiseDomain.BIOLOGY, "Biology Expert"),
    ]

    for domain, description in domains:
        expert = ExpertPeerAgent(
            role=AgentRole(
                name=f"expert_{domain.value}",
                description=description,
                system_prompt=f"You are a {description} sharing knowledge with peers from other domains."
            ),
            state_manager=state,
            message_bus=bus,
            domain=domain
        )
        experts.append(expert)

    # Create peer-to-peer collaboration
    collaboration = PeerToPeerCollaboration(
        peers=experts,
        message_bus=bus,
        state_manager=state
    )

    # Initialize
    print("Initializing expert peer network...")
    await collaboration.initialize()
    print(f"✓ Expert network initialized: {len(experts)} peers\n")

    # Execute knowledge sharing
    knowledge_topic = "The Nature and Implications of Consciousness"

    print(f"Knowledge Topic: {knowledge_topic}")
    print(f"Collaboration Rounds: 2")
    print(f"Peer Structure: Equal collaboration (no hierarchy)\n")

    print("Starting knowledge sharing...\n")

    result = await collaboration.execute(
        task={"task": knowledge_topic},
        rounds=2
    )

    # Display results
    print("\n" + "="*80)
    print("KNOWLEDGE SHARING RESULTS")
    print("="*80 + "\n")

    print(f"Topic: {result.get('task', {}).get('task', '')}")
    print(f"Total Rounds: {result.get('total_rounds', 0)}")
    print(f"Total Contributions: {result.get('total_contributions', 0)}")
    print(f"Participating Experts: {result.get('participating_peers', 0)}")

    if 'rounds_detail' in result:
        for round_data in result['rounds_detail']:
            round_num = round_data.get('round', 0)
            contributions = round_data.get('contributions', [])

            print(f"\n" + "-"*80)
            print(f"ROUND {round_num}")
            print("-"*80 + "\n")

            for contrib in contributions:
                domain = contrib.get('domain', 'Unknown')
                contrib_type = contrib.get('contribution_type', '')
                content = contrib.get('content', '')

                print(f"\n{domain.upper()} ({contrib_type}):")
                print(content)
                print()

    if 'synthesis' in result:
        print("\n" + "="*80)
        print("COLLECTIVE KNOWLEDGE SYNTHESIS")
        print("="*80)
        print(result['synthesis'])

    print("\n" + "-"*80)
    print("KEY INSIGHTS")
    print("-"*80)
    print("""
1. Multi-disciplinary Perspective:
   - Each domain contributes unique insights
   - Cross-domain connections reveal deeper understanding
   - Collective knowledge > sum of individual parts

2. Equal Collaboration Benefits:
   - No single authority, all voices heard
   - Peer review improves quality
   - Diverse perspectives reduce blind spots

3. Emergent Understanding:
   - Round 1: Individual domain perspectives
   - Round 2: Integration and synthesis
   - Final: Holistic multi-domain knowledge
""")

    # Shutdown
    await collaboration.shutdown()
    print("✓ Knowledge sharing network shutdown complete")


async def main():
    """Main entry point for example."""
    await run_knowledge_sharing_example()


if __name__ == "__main__":
    # Run example
    asyncio.run(main())
