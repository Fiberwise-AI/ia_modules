"""
Hierarchical Collaboration Example: Research Team

Demonstrates leader-worker pattern where a lead researcher coordinates
specialist researchers to analyze a complex topic.

Use Case:
- Lead researcher receives research topic
- Breaks down into specialized areas (algorithms, hardware, applications)
- Delegates to specialist researchers
- Synthesizes comprehensive research report

Pattern: Hierarchical (Leader-Worker)
"""

import asyncio
import logging
from typing import Dict, Any

from ..core import AgentRole
from ..state import StateManager
from ..communication import MessageBus
from ..collaboration_patterns.hierarchical import (
    HierarchicalCollaboration,
    LeaderAgent,
    WorkerAgent,
)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class ResearchLeaderAgent(LeaderAgent):
    """
    Lead researcher who coordinates research team.

    Responsibilities:
    - Decompose research topics into specialized areas
    - Coordinate specialist researchers
    - Synthesize findings into comprehensive report
    """

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute research coordination."""
        topic = input_data.get("topic", "")
        depth = input_data.get("depth", "standard")

        self.logger.info(f"Lead researcher coordinating research on: {topic}")

        # Announce research project to team
        await self.broadcast_message(
            message_type=MessageType.BROADCAST,
            content={
                "action": "project_started",
                "topic": topic,
                "depth": depth,
                "message": f"Starting comprehensive research on {topic}"
            }
        )

        # Store research context
        await self.write_state("research_topic", topic)
        await self.write_state("research_depth", depth)
        await self.write_state("research_status", "coordinating")

        return {
            "status": "coordinating",
            "topic": topic,
            "depth": depth
        }


class SpecialistResearcherAgent(WorkerAgent):
    """
    Specialist researcher focused on a specific area.

    Each researcher has expertise in:
    - Algorithms (theory, complexity, optimization)
    - Hardware (processors, memory, accelerators)
    - Applications (use cases, implementations, benchmarks)
    """

    def __init__(self, role: AgentRole, state_manager: StateManager,
                 message_bus: MessageBus, specialty: str):
        """
        Initialize specialist researcher.

        Args:
            role: Agent role
            state_manager: State manager
            message_bus: Message bus
            specialty: Area of expertise (algorithms/hardware/applications)
        """
        super().__init__(role, state_manager, message_bus)
        self.specialty = specialty

    async def _process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process research task in specialist area.

        Args:
            task_data: Research task parameters

        Returns:
            Research findings
        """
        description = task_data.get("description", "")
        topic = await self.read_state("research_topic", "Unknown")
        depth = await self.read_state("research_depth", "standard")

        self.logger.info(f"{self.specialty} researcher analyzing: {description}")

        # Simulate research work
        await asyncio.sleep(0.2)  # Simulate research time

        # Generate findings based on specialty
        findings = self._generate_findings(topic, description, depth)

        return {
            "status": "success",
            "task_id": task_data.get("task_id"),
            "specialty": self.specialty,
            "topic": topic,
            "findings": findings,
            "sources_consulted": self._get_sources(),
            "confidence": 0.85,
            "worker": self.agent_id
        }

    def _generate_findings(self, topic: str, description: str, depth: str) -> str:
        """Generate research findings based on specialty."""
        findings_templates = {
            "algorithms": f"""
Algorithmic Analysis of {topic}:

1. Theoretical Foundations:
   - Core algorithms and data structures involved
   - Computational complexity (time/space)
   - Optimization opportunities

2. Key Algorithms:
   - Primary algorithms used in {topic}
   - Trade-offs between different approaches
   - Recent algorithmic innovations

3. Performance Characteristics:
   - Expected performance bounds
   - Best/average/worst case scenarios
   - Scalability considerations
""",
            "hardware": f"""
Hardware Perspective on {topic}:

1. Hardware Requirements:
   - Processor requirements (CPU/GPU/TPU)
   - Memory requirements (RAM/VRAM)
   - Storage considerations

2. Hardware Optimizations:
   - Architecture-specific optimizations
   - Parallel processing opportunities
   - Accelerator utilization

3. Future Hardware Trends:
   - Emerging hardware technologies
   - Hardware-software co-design
   - Energy efficiency considerations
""",
            "applications": f"""
Applications and Use Cases for {topic}:

1. Current Applications:
   - Industry applications and deployments
   - Real-world use cases
   - Success stories and case studies

2. Implementation Patterns:
   - Common implementation approaches
   - Integration patterns
   - Best practices and anti-patterns

3. Future Directions:
   - Emerging applications
   - Research directions
   - Potential impact areas
"""
        }

        base_findings = findings_templates.get(
            self.specialty,
            f"General research findings on {topic}"
        )

        if depth == "comprehensive":
            base_findings += f"\n\n4. Deep Dive Analysis:\n   - Advanced {self.specialty} considerations\n   - Cutting-edge research\n   - Expert recommendations"

        return base_findings

    def _get_sources(self) -> list:
        """Get relevant sources based on specialty."""
        source_map = {
            "algorithms": [
                "ACM Computing Surveys",
                "Journal of Algorithms",
                "ArXiv CS.DS",
            ],
            "hardware": [
                "IEEE Micro",
                "Computer Architecture Letters",
                "ISCA Proceedings",
            ],
            "applications": [
                "Industry Reports",
                "Case Studies Database",
                "Production Deployment Surveys",
            ]
        }
        return source_map.get(self.specialty, ["General Academic Sources"])


async def run_research_team_example():
    """
    Run hierarchical research team example.

    Demonstrates:
    - Leader-worker coordination
    - Task delegation
    - Result synthesis
    - Specialized agent roles
    """
    print("\n" + "="*80)
    print("HIERARCHICAL COLLABORATION EXAMPLE: Research Team")
    print("="*80 + "\n")

    # Setup infrastructure
    state = StateManager(thread_id="research_team_001")
    bus = MessageBus()

    # Create lead researcher
    leader_role = AgentRole(
        name="lead_researcher",
        description="Coordinates research team and synthesizes findings",
        system_prompt="You are a lead researcher who coordinates specialists and synthesizes comprehensive research reports.",
        max_iterations=5
    )
    leader = ResearchLeaderAgent(
        role=leader_role,
        state_manager=state,
        message_bus=bus
    )

    # Create specialist researchers
    specialties = ["algorithms", "hardware", "applications"]
    workers = []

    for specialty in specialties:
        worker_role = AgentRole(
            name=f"researcher_{specialty}",
            description=f"Expert researcher in {specialty}",
            system_prompt=f"You are an expert researcher specializing in {specialty}.",
            max_iterations=3
        )
        worker = SpecialistResearcherAgent(
            role=worker_role,
            state_manager=state,
            message_bus=bus,
            specialty=specialty
        )
        workers.append(worker)

    # Create hierarchical collaboration
    collaboration = HierarchicalCollaboration(
        leader=leader,
        workers=workers,
        message_bus=bus
    )

    # Initialize all agents
    print("Initializing research team...")
    await collaboration.initialize()
    print(f"✓ Team initialized: 1 lead researcher, {len(workers)} specialists\n")

    # Execute research task
    research_topic = "Quantum Computing Applications in Machine Learning"

    print(f"Research Topic: {research_topic}")
    print(f"Research Depth: comprehensive")
    print("\nStarting research...\n")

    result = await collaboration.execute(
        task_description={
            "task": f"Conduct comprehensive research on {research_topic}",
            "topic": research_topic,
            "depth": "comprehensive"
        }
    )

    # Display results
    print("\n" + "="*80)
    print("RESEARCH RESULTS")
    print("="*80 + "\n")

    print(f"Topic: {result.get('task', '')}")
    print(f"Status: {result.get('status', '')}")
    print(f"Total Specialists: {result.get('total_workers', 0)}")
    print(f"Successful Contributions: {result.get('successful_workers', 0)}")
    print(f"Failed Contributions: {result.get('failed_workers', 0)}")

    if result.get('status') == 'success':
        print("\n" + "-"*80)
        print("SPECIALIST FINDINGS")
        print("-"*80 + "\n")

        for i, output in enumerate(result.get('worker_outputs', []), 1):
            if isinstance(output, dict):
                specialty = output.get('specialty', 'Unknown')
                findings = output.get('findings', '')
                confidence = output.get('confidence', 0)
                sources = output.get('sources_consulted', [])

                print(f"\n{i}. {specialty.upper()} RESEARCH")
                print(f"   Confidence: {confidence:.0%}")
                print(f"   Sources: {', '.join(sources)}")
                print(f"\n{findings}")
                print("\n" + "-"*80)

        print("\n" + "="*80)
        print("SYNTHESIS")
        print("="*80)
        print(result.get('summary', ''))

    # Shutdown
    await collaboration.shutdown()
    print("\n✓ Research team shutdown complete")


async def main():
    """Main entry point for example."""
    await run_research_team_example()


if __name__ == "__main__":
    # Import MessageType here to avoid circular imports
    from ..communication import MessageType

    # Run example
    asyncio.run(main())
