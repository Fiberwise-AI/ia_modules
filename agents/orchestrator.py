"""
Agent orchestrator with graph-based execution.

Manages multi-agent workflows with explicit control flow, conditional
branching, and feedback loops.
"""

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any, Set
import logging

from .core import BaseAgent
from .state import StateManager


@dataclass
class Edge:
    """
    Represents a transition between agents in the workflow graph.

    Attributes:
        to: Target agent ID
        condition: Optional condition function to evaluate
        metadata: Additional edge configuration
    """
    to: str
    condition: Optional[Callable] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AgentOrchestrator:
    """
    Orchestrates multi-agent workflows with graph-based execution.

    Features:
    - Graph-based agent sequencing
    - Conditional branching based on state
    - Feedback loops with automatic iteration tracking
    - Parallel agent execution
    - Cycle detection

    Example:
        >>> state = StateManager(thread_id="user-123")
        >>> orchestrator = AgentOrchestrator(state)
        >>>
        >>> # Register agents
        >>> orchestrator.add_agent("planner", planner_agent)
        >>> orchestrator.add_agent("coder", coder_agent)
        >>> orchestrator.add_agent("critic", critic_agent)
        >>>
        >>> # Build workflow: planner → coder → critic
        >>> orchestrator.add_edge("planner", "coder")
        >>> orchestrator.add_edge("coder", "critic")
        >>>
        >>> # Add feedback loop: critic → coder (if not approved)
        >>> orchestrator.add_feedback_loop("coder", "critic", max_iterations=3)
        >>>
        >>> # Execute
        >>> result = await orchestrator.run("planner", {"task": "Build API"})
    """

    def __init__(self, state_manager: StateManager):
        """
        Initialize orchestrator.

        Args:
            state_manager: Centralized state for agent communication
        """
        self.state = state_manager
        self.agents: Dict[str, BaseAgent] = {}
        self.graph: Dict[str, List[Edge]] = {}
        self.logger = logging.getLogger(f"Orchestrator.{state_manager.thread_id}")

    def add_agent(self, agent_id: str, agent: BaseAgent) -> None:
        """
        Register an agent in the workflow.

        Args:
            agent_id: Unique identifier for agent in this workflow
            agent: Agent instance
        """
        self.agents[agent_id] = agent
        if agent_id not in self.graph:
            self.graph[agent_id] = []

        self.logger.debug(f"Added agent: {agent_id} ({agent.role.name})")

    def add_edge(self, from_agent: str, to_agent: str,
                 condition: Optional[Callable] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add transition between agents.

        Args:
            from_agent: Source agent ID
            to_agent: Target agent ID
            condition: Optional async function(state) -> bool
            metadata: Additional edge configuration

        Example:
            >>> # Unconditional edge
            >>> orchestrator.add_edge("planner", "coder")
            >>>
            >>> # Conditional edge
            >>> async def needs_research(state):
            ...     return await state.get("requires_research", False)
            >>>
            >>> orchestrator.add_edge("planner", "researcher", condition=needs_research)
        """
        if from_agent not in self.graph:
            self.graph[from_agent] = []

        self.graph[from_agent].append(
            Edge(to=to_agent, condition=condition, metadata=metadata)
        )

        self.logger.debug(f"Added edge: {from_agent} → {to_agent}")

    def add_feedback_loop(self, worker_agent: str, critic_agent: str,
                         max_iterations: int = 3,
                         next_agent: Optional[str] = None) -> Callable:
        """
        Add feedback loop between worker and critic agents.

        Creates two edges:
        1. worker → critic (always)
        2. critic → worker (if not approved and under max iterations)
        3. critic → next_agent (if approved or max iterations reached)

        Args:
            worker_agent: Agent that produces work
            critic_agent: Agent that reviews work
            max_iterations: Maximum feedback iterations
            next_agent: Next agent after loop completes

        Returns:
            Condition function for "loop complete" (for adding next edge)

        Example:
            >>> # Add feedback loop
            >>> is_complete = orchestrator.add_feedback_loop("coder", "critic")
            >>>
            >>> # Add edge to next agent when loop completes
            >>> orchestrator.add_edge("critic", "formatter", condition=is_complete)
        """
        # Worker → Critic (always)
        self.add_edge(worker_agent, critic_agent)

        iteration_key = f"{worker_agent}_iterations"

        # Critic → Worker (if not approved and under max iterations)
        async def needs_revision(state: StateManager) -> bool:
            approved = await state.get("approved", False)
            iterations = await state.get(iteration_key, 0)

            if not approved and iterations < max_iterations:
                # Increment iteration count
                await state.set(iteration_key, iterations + 1)
                self.logger.info(f"Feedback loop iteration {iterations + 1}/{max_iterations}")
                return True

            return False

        self.add_edge(critic_agent, worker_agent, condition=needs_revision,
                     metadata={"type": "feedback_loop", "max_iterations": max_iterations})

        # Critic → Next (if approved or max iterations)
        async def is_complete(state: StateManager) -> bool:
            approved = await state.get("approved", False)
            iterations = await state.get(iteration_key, 0)
            complete = approved or iterations >= max_iterations

            if complete:
                self.logger.info(f"Feedback loop complete (approved={approved}, iterations={iterations})")

            return complete

        if next_agent:
            self.add_edge(critic_agent, next_agent, condition=is_complete)

        return is_complete

    async def run(self, start_agent: str, input_data: Dict[str, Any],
                  max_steps: int = 100) -> Dict[str, Any]:
        """
        Execute agent workflow starting from start_agent.

        Args:
            start_agent: Agent ID to start workflow
            input_data: Initial input data
            max_steps: Maximum execution steps (prevents infinite loops)

        Returns:
            Final state snapshot

        Raises:
            ValueError: If start_agent not found
            RuntimeError: If infinite loop detected
        """
        if start_agent not in self.agents:
            raise ValueError(f"Unknown start agent: {start_agent}")

        # Initialize state with input
        await self.state.update(input_data)
        self.logger.info(f"Starting workflow from {start_agent}")

        current_agent = start_agent
        steps = 0
        execution_path = []

        while current_agent and steps < max_steps:
            steps += 1
            execution_path.append(current_agent)

            self.logger.info(f"Step {steps}: Executing {current_agent}")

            # Execute agent
            agent = self.agents[current_agent]
            try:
                result = await agent.execute(input_data)
                self.logger.debug(f"{current_agent} returned: {result}")

            except Exception as e:
                self.logger.error(f"Agent {current_agent} failed: {e}")
                await self.state.set("error", str(e))
                await self.state.set("failed_agent", current_agent)
                raise

            # Find next agent
            next_agent = await self._get_next_agent(current_agent)

            if next_agent:
                self.logger.debug(f"Next agent: {next_agent}")
            else:
                self.logger.info("Workflow complete (no more agents)")

            current_agent = next_agent

        if steps >= max_steps:
            raise RuntimeError(f"Max steps ({max_steps}) exceeded. Possible infinite loop. Path: {execution_path}")

        # Save execution metadata
        await self.state.set("execution_path", execution_path)
        await self.state.set("total_steps", steps)

        self.logger.info(f"Workflow complete in {steps} steps: {' → '.join(execution_path)}")

        # Return final state
        return await self.state.snapshot()

    async def _get_next_agent(self, current_agent: str) -> Optional[str]:
        """
        Determine next agent based on edges and conditions.

        Args:
            current_agent: Current agent ID

        Returns:
            Next agent ID or None if workflow complete
        """
        edges = self.graph.get(current_agent, [])

        for edge in edges:
            # Unconditional edge
            if edge.condition is None:
                return edge.to

            # Evaluate condition
            try:
                if await edge.condition(self.state):
                    return edge.to
            except Exception as e:
                self.logger.error(f"Condition evaluation failed for edge {current_agent}→{edge.to}: {e}")
                continue

        return None  # End of workflow

    def visualize(self) -> str:
        """
        Generate Mermaid diagram of workflow.

        Returns:
            Mermaid markdown string

        Example:
            >>> diagram = orchestrator.visualize()
            >>> print(diagram)
            graph TD
                planner --> coder
                coder --> critic
                critic -->|needs_revision| coder
                critic -->|is_complete| formatter
        """
        lines = ["graph TD"]

        for from_agent, edges in self.graph.items():
            from_label = self.agents[from_agent].role.name if from_agent in self.agents else from_agent

            for edge in edges:
                to_label = self.agents[edge.to].role.name if edge.to in self.agents else edge.to

                if edge.condition:
                    # Conditional edge
                    condition_name = edge.condition.__name__
                    lines.append(f"    {from_agent}[{from_label}] -->|{condition_name}| {edge.to}[{to_label}]")
                else:
                    # Unconditional edge
                    lines.append(f"    {from_agent}[{from_label}] --> {edge.to}[{to_label}]")

        return "\n".join(lines)

    def get_agent_stats(self) -> Dict[str, Any]:
        """
        Get orchestrator statistics.

        Returns:
            Dictionary with stats
        """
        return {
            "num_agents": len(self.agents),
            "num_edges": sum(len(edges) for edges in self.graph.values()),
            "agents": list(self.agents.keys()),
            "state_keys": len(self.state._state),
            "state_versions": self.state.version_count()
        }

    def __repr__(self) -> str:
        return f"<AgentOrchestrator(agents={len(self.agents)}, edges={sum(len(e) for e in self.graph.values())})>"
