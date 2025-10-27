"""
Hierarchical collaboration pattern (leader-worker).

Implements a hierarchical structure where a leader agent delegates tasks
to worker agents and synthesizes their results.
"""

import asyncio
from typing import Dict, Any, List, Optional, Set
import logging

from ..base_agent import BaseCollaborativeAgent
from ..communication import MessageBus, MessageType, AgentMessage
from ..task_decomposition import Task, TaskDecomposer, DependencyGraph, DecompositionStrategy
from ..state import StateManager


class HierarchicalCollaboration:
    """
    Hierarchical collaboration pattern with leader-worker structure.

    The leader agent:
    - Receives high-level tasks
    - Decomposes tasks into subtasks
    - Delegates subtasks to worker agents
    - Synthesizes results from workers
    - Provides final output

    Worker agents:
    - Receive task assignments from leader
    - Execute assigned tasks independently
    - Report results back to leader
    - May collaborate with peer workers

    Example:
        >>> # Setup
        >>> state = StateManager(thread_id="hierarchical_001")
        >>> bus = MessageBus()
        >>>
        >>> # Create leader
        >>> leader = LeaderAgent(
        ...     role=AgentRole(name="leader", description="Coordinates workers"),
        ...     state_manager=state,
        ...     message_bus=bus
        ... )
        >>>
        >>> # Create workers
        >>> workers = [
        ...     WorkerAgent(
        ...         role=AgentRole(name=f"worker_{i}", description="Executes tasks"),
        ...         state_manager=state,
        ...         message_bus=bus
        ...     )
        ...     for i in range(3)
        ... ]
        >>>
        >>> # Setup collaboration
        >>> collaboration = HierarchicalCollaboration(leader, workers, bus)
        >>> await collaboration.initialize()
        >>>
        >>> # Execute task
        >>> result = await collaboration.execute({
        ...     "task": "Analyze market trends in tech sector"
        ... })
    """

    def __init__(self, leader: BaseCollaborativeAgent,
                 workers: List[BaseCollaborativeAgent],
                 message_bus: MessageBus):
        """
        Initialize hierarchical collaboration.

        Args:
            leader: Leader agent that coordinates work
            workers: List of worker agents
            message_bus: Message bus for communication
        """
        self.leader = leader
        self.workers = workers
        self.message_bus = message_bus
        self.task_decomposer = TaskDecomposer()
        self.logger = logging.getLogger("HierarchicalCollaboration")

        # Track worker assignments
        self.worker_tasks: Dict[str, Task] = {}
        self.available_workers: Set[str] = set()

    async def initialize(self) -> None:
        """Initialize all agents and communication."""
        # Initialize leader
        await self.leader.initialize()

        # Initialize workers
        for worker in self.workers:
            await worker.initialize()
            self.available_workers.add(worker.agent_id)

        self.logger.info(
            f"Hierarchical collaboration initialized: "
            f"1 leader, {len(self.workers)} workers"
        )

    async def shutdown(self) -> None:
        """Shutdown all agents."""
        await self.leader.shutdown()
        for worker in self.workers:
            await worker.shutdown()

    async def execute(self, task_description: Dict[str, Any],
                     strategy: DecompositionStrategy = DecompositionStrategy.PARALLEL) -> Dict[str, Any]:
        """
        Execute task using hierarchical collaboration.

        Args:
            task_description: High-level task to execute
            strategy: Task decomposition strategy

        Returns:
            Synthesized results from all workers
        """
        self.logger.info("Starting hierarchical task execution")

        # Phase 1: Leader decomposes task
        description = task_description.get("task", str(task_description))
        subtasks = await self.task_decomposer.decompose(
            description=description,
            strategy=strategy,
            context=task_description
        )

        self.logger.info(f"Task decomposed into {len(subtasks)} subtasks")

        # Phase 2: Leader assigns tasks to workers
        assignments = await self._assign_tasks(subtasks)

        # Phase 3: Workers execute tasks
        results = await self._execute_assigned_tasks(assignments)

        # Phase 4: Leader synthesizes results
        final_result = await self._synthesize_results(results, task_description)

        self.logger.info("Hierarchical task execution complete")

        return final_result

    async def _assign_tasks(self, tasks: List[Task]) -> Dict[str, Task]:
        """
        Assign tasks to available workers.

        Uses round-robin assignment respecting dependencies.

        Args:
            tasks: List of tasks to assign

        Returns:
            Dictionary mapping worker_id to assigned task
        """
        # Get execution order respecting dependencies
        execution_levels = self.task_decomposer.get_execution_order(tasks)

        assignments = {}
        worker_index = 0
        workers_list = list(self.available_workers)

        # Assign tasks level by level
        for level in execution_levels:
            for task in level:
                # Assign to next available worker (round-robin)
                worker_id = workers_list[worker_index % len(workers_list)]
                task.assigned_to = worker_id
                assignments[worker_id] = task

                self.logger.debug(f"Assigned {task.task_id} to {worker_id}")

                worker_index += 1

        return assignments

    async def _execute_assigned_tasks(self,
                                     assignments: Dict[str, Task]) -> Dict[str, Dict[str, Any]]:
        """
        Execute tasks assigned to workers.

        Args:
            assignments: Worker assignments

        Returns:
            Dictionary mapping worker_id to results
        """
        # Send task assignments to workers
        pending_tasks = []

        for worker_id, task in assignments.items():
            # Send task request to worker
            task_future = self.leader.send_task_request(
                recipient=worker_id,
                task_data={
                    "task_id": task.task_id,
                    "description": task.description,
                    "input_data": task.input_data
                },
                wait_for_response=True,
                timeout=60.0
            )
            pending_tasks.append((worker_id, task.task_id, task_future))

        # Wait for all workers to complete
        results = {}

        for worker_id, task_id, task_future in pending_tasks:
            try:
                response = await task_future
                results[worker_id] = {
                    "task_id": task_id,
                    "status": "completed",
                    "output": response.content if response else None
                }
                self.logger.info(f"Worker {worker_id} completed {task_id}")

            except Exception as e:
                self.logger.error(f"Worker {worker_id} failed on {task_id}: {e}")
                results[worker_id] = {
                    "task_id": task_id,
                    "status": "failed",
                    "error": str(e)
                }

        return results

    async def _synthesize_results(self, worker_results: Dict[str, Dict[str, Any]],
                                  original_task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize results from all workers.

        Args:
            worker_results: Results from each worker
            original_task: Original task description

        Returns:
            Synthesized final result
        """
        # Collect successful results
        successful_results = [
            result for result in worker_results.values()
            if result.get("status") == "completed"
        ]

        failed_results = [
            result for result in worker_results.values()
            if result.get("status") == "failed"
        ]

        # Create synthesis
        synthesis = {
            "task": original_task.get("task", ""),
            "total_workers": len(worker_results),
            "successful_workers": len(successful_results),
            "failed_workers": len(failed_results),
            "worker_outputs": [r.get("output") for r in successful_results],
            "status": "success" if not failed_results else "partial_success",
            "summary": self._create_synthesis_summary(successful_results)
        }

        if failed_results:
            synthesis["failures"] = [
                {"task": r["task_id"], "error": r.get("error")}
                for r in failed_results
            ]

        return synthesis

    def _create_synthesis_summary(self, results: List[Dict[str, Any]]) -> str:
        """Create summary from worker results."""
        if not results:
            return "No results to synthesize."

        summary_parts = [f"Synthesized results from {len(results)} workers:\n"]

        for i, result in enumerate(results, 1):
            output = result.get("output", {})
            if isinstance(output, dict):
                status = output.get("status", "unknown")
                summary_parts.append(f"{i}. Task {result['task_id']}: {status}")
            else:
                summary_parts.append(f"{i}. Task {result['task_id']}: completed")

        return "\n".join(summary_parts)


class LeaderAgent(BaseCollaborativeAgent):
    """
    Leader agent that coordinates worker agents.

    Extends BaseCollaborativeAgent with leadership capabilities:
    - Task decomposition
    - Worker management
    - Result synthesis
    - Progress monitoring
    """

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute as leader: decompose, delegate, synthesize.

        Args:
            input_data: Task to coordinate

        Returns:
            Synthesized results
        """
        self.logger.info("Leader coordinating task execution")

        # This would typically be called through HierarchicalCollaboration
        # But can also be used standalone

        task = input_data.get("task", "")

        # Broadcast task announcement to workers
        await self.broadcast_message(
            message_type=MessageType.BROADCAST,
            content={"action": "task_started", "task": task}
        )

        # Store task in state
        await self.write_state("leader_task", task)
        await self.write_state("leader_status", "coordinating")

        return {
            "status": "coordinating",
            "task": task
        }


class WorkerAgent(BaseCollaborativeAgent):
    """
    Worker agent that executes assigned tasks.

    Extends BaseCollaborativeAgent with worker capabilities:
    - Task execution
    - Progress reporting
    - Peer collaboration
    """

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute assigned task.

        Args:
            input_data: Task assignment

        Returns:
            Task results
        """
        task_id = input_data.get("task_id", "unknown")
        description = input_data.get("description", "")

        self.logger.info(f"Worker executing task: {task_id}")

        # Update status
        await self.write_state(f"worker_{self.agent_id}_status", "working")
        await self.write_state(f"worker_{self.agent_id}_task", task_id)

        # Execute task (simplified - would use actual processing)
        result = await self._process_task(input_data)

        # Update status
        await self.write_state(f"worker_{self.agent_id}_status", "completed")

        self.logger.info(f"Worker completed task: {task_id}")

        return result

    async def _process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process assigned task.

        Override this in specialized worker implementations.

        Args:
            task_data: Task parameters

        Returns:
            Processing results
        """
        # Simulate work
        await asyncio.sleep(0.1)

        return {
            "status": "success",
            "task_id": task_data.get("task_id"),
            "result": f"Processed: {task_data.get('description', '')}",
            "worker": self.agent_id
        }

    async def collaborate_with_peer(self, peer_id: str,
                                   collaboration_type: str,
                                   data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collaborate with peer worker.

        Args:
            peer_id: ID of peer worker
            collaboration_type: Type of collaboration needed
            data: Data to share

        Returns:
            Collaboration results
        """
        self.logger.info(f"Collaborating with peer {peer_id}")

        response = await self.send_query(
            recipient=peer_id,
            query={
                "action": "collaborate",
                "type": collaboration_type,
                "data": data
            },
            timeout=30.0
        )

        return response.content
