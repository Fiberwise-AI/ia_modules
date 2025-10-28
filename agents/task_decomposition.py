"""
Task decomposition and dependency management for multi-agent collaboration.

Provides strategies for breaking down complex tasks and managing dependencies.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Callable
from enum import Enum
import logging


class TaskStatus(Enum):
    """Status of a task in the workflow."""
    PENDING = "pending"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class DecompositionStrategy(Enum):
    """Strategies for decomposing tasks."""
    SEQUENTIAL = "sequential"  # Tasks must be done in order
    PARALLEL = "parallel"  # Tasks can be done concurrently
    HIERARCHICAL = "hierarchical"  # Tasks organized in tree structure
    PIPELINE = "pipeline"  # Tasks form a data processing pipeline
    DYNAMIC = "dynamic"  # Tasks determined at runtime


@dataclass
class Task:
    """
    A unit of work in a multi-agent workflow.

    Attributes:
        task_id: Unique task identifier
        description: Human-readable task description
        assigned_to: Agent ID assigned to execute this task
        status: Current task status
        input_data: Input data for task execution
        output_data: Results from task execution
        dependencies: Task IDs that must complete before this task
        dependents: Task IDs that depend on this task
        priority: Task priority (higher = more important)
        metadata: Additional task metadata
        error: Error message if task failed
    """
    task_id: str
    description: str
    assigned_to: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def is_ready(self, completed_tasks: Set[str]) -> bool:
        """
        Check if task is ready to execute.

        Args:
            completed_tasks: Set of completed task IDs

        Returns:
            True if all dependencies are satisfied
        """
        if self.status != TaskStatus.PENDING:
            return False
        return self.dependencies.issubset(completed_tasks)

    def is_blocked(self, failed_tasks: Set[str]) -> bool:
        """
        Check if task is blocked by failed dependencies.

        Args:
            failed_tasks: Set of failed task IDs

        Returns:
            True if any dependency failed
        """
        return bool(self.dependencies & failed_tasks)

    def mark_completed(self, output_data: Dict[str, Any]) -> None:
        """Mark task as completed with results."""
        self.status = TaskStatus.COMPLETED
        self.output_data = output_data

    def mark_failed(self, error: str) -> None:
        """Mark task as failed with error."""
        self.status = TaskStatus.FAILED
        self.error = error

    def __repr__(self) -> str:
        return (f"<Task(id={self.task_id}, "
                f"status={self.status.value}, "
                f"assigned_to={self.assigned_to})>")


class TaskDecomposer:
    """
    Decomposes complex tasks into manageable subtasks.

    Example:
        >>> decomposer = TaskDecomposer()
        >>>
        >>> # Decompose a complex task
        >>> tasks = await decomposer.decompose(
        ...     description="Analyze and summarize research papers",
        ...     strategy=DecompositionStrategy.SEQUENTIAL
        ... )
        >>>
        >>> # Get execution order
        >>> order = decomposer.get_execution_order(tasks)
    """

    def __init__(self):
        """Initialize task decomposer."""
        self.logger = logging.getLogger("TaskDecomposer")

    async def decompose(self, description: str,
                       strategy: DecompositionStrategy = DecompositionStrategy.SEQUENTIAL,
                       context: Optional[Dict[str, Any]] = None) -> List[Task]:
        """
        Decompose a complex task into subtasks.

        Args:
            description: High-level task description
            strategy: Decomposition strategy to use
            context: Additional context for decomposition

        Returns:
            List of subtasks with dependencies
        """
        self.logger.info(f"Decomposing task: {description[:100]}...")

        if strategy == DecompositionStrategy.SEQUENTIAL:
            return await self._decompose_sequential(description, context)
        elif strategy == DecompositionStrategy.PARALLEL:
            return await self._decompose_parallel(description, context)
        elif strategy == DecompositionStrategy.HIERARCHICAL:
            return await self._decompose_hierarchical(description, context)
        elif strategy == DecompositionStrategy.PIPELINE:
            return await self._decompose_pipeline(description, context)
        elif strategy == DecompositionStrategy.DYNAMIC:
            return await self._decompose_dynamic(description, context)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    async def _decompose_sequential(self, description: str,
                                   context: Optional[Dict[str, Any]]) -> List[Task]:
        """
        Decompose into sequential tasks.

        Each task depends on the previous one completing.
        """
        # Simplified decomposition - in real implementation would use LLM
        tasks = [
            Task(
                task_id="task_1",
                description="Understand requirements and gather information",
                priority=3
            ),
            Task(
                task_id="task_2",
                description="Process and analyze gathered information",
                dependencies={"task_1"},
                priority=2
            ),
            Task(
                task_id="task_3",
                description="Synthesize results and create output",
                dependencies={"task_2"},
                priority=1
            )
        ]

        # Set up dependent relationships
        tasks[0].dependents.add("task_2")
        tasks[1].dependents.add("task_3")

        self.logger.info(f"Created {len(tasks)} sequential tasks")
        return tasks

    async def _decompose_parallel(self, description: str,
                                 context: Optional[Dict[str, Any]]) -> List[Task]:
        """
        Decompose into parallel tasks.

        Tasks can be executed concurrently.
        """
        tasks = [
            Task(
                task_id="task_parallel_1",
                description="Research aspect A",
                priority=2
            ),
            Task(
                task_id="task_parallel_2",
                description="Research aspect B",
                priority=2
            ),
            Task(
                task_id="task_parallel_3",
                description="Research aspect C",
                priority=2
            ),
            Task(
                task_id="task_synthesis",
                description="Synthesize all research",
                dependencies={"task_parallel_1", "task_parallel_2", "task_parallel_3"},
                priority=1
            )
        ]

        # Set up dependent relationships
        for task in tasks[:3]:
            task.dependents.add("task_synthesis")

        self.logger.info(f"Created {len(tasks)} parallel tasks")
        return tasks

    async def _decompose_hierarchical(self, description: str,
                                     context: Optional[Dict[str, Any]]) -> List[Task]:
        """
        Decompose into hierarchical tree structure.

        High-level task -> subtasks -> sub-subtasks
        """
        tasks = [
            # Level 1: Main task
            Task(
                task_id="main_task",
                description="Complete main objective",
                dependencies={"subtask_1", "subtask_2"},
                priority=1
            ),
            # Level 2: Subtasks
            Task(
                task_id="subtask_1",
                description="Complete subtask 1",
                dependencies={"leaf_1_1", "leaf_1_2"},
                priority=2
            ),
            Task(
                task_id="subtask_2",
                description="Complete subtask 2",
                dependencies={"leaf_2_1"},
                priority=2
            ),
            # Level 3: Leaf tasks
            Task(
                task_id="leaf_1_1",
                description="Execute atomic task 1.1",
                priority=3
            ),
            Task(
                task_id="leaf_1_2",
                description="Execute atomic task 1.2",
                priority=3
            ),
            Task(
                task_id="leaf_2_1",
                description="Execute atomic task 2.1",
                priority=3
            ),
        ]

        # Set up dependent relationships
        tasks[3].dependents.add("subtask_1")  # leaf_1_1
        tasks[4].dependents.add("subtask_1")  # leaf_1_2
        tasks[5].dependents.add("subtask_2")  # leaf_2_1
        tasks[1].dependents.add("main_task")  # subtask_1
        tasks[2].dependents.add("main_task")  # subtask_2

        self.logger.info(f"Created {len(tasks)} hierarchical tasks")
        return tasks

    async def _decompose_pipeline(self, description: str,
                                 context: Optional[Dict[str, Any]]) -> List[Task]:
        """
        Decompose into pipeline stages.

        Data flows through stages sequentially.
        """
        tasks = [
            Task(
                task_id="stage_1_input",
                description="Stage 1: Input processing",
                priority=4
            ),
            Task(
                task_id="stage_2_transform",
                description="Stage 2: Data transformation",
                dependencies={"stage_1_input"},
                priority=3
            ),
            Task(
                task_id="stage_3_enrich",
                description="Stage 3: Data enrichment",
                dependencies={"stage_2_transform"},
                priority=2
            ),
            Task(
                task_id="stage_4_output",
                description="Stage 4: Output generation",
                dependencies={"stage_3_enrich"},
                priority=1
            )
        ]

        # Set up pipeline dependencies
        for i in range(len(tasks) - 1):
            tasks[i].dependents.add(tasks[i + 1].task_id)

        self.logger.info(f"Created {len(tasks)} pipeline stages")
        return tasks

    async def _decompose_dynamic(self, description: str,
                                context: Optional[Dict[str, Any]]) -> List[Task]:
        """
        Decompose dynamically based on context.

        Initial set of tasks that may spawn more tasks at runtime.
        """
        tasks = [
            Task(
                task_id="analyze_requirements",
                description="Analyze what needs to be done",
                priority=3,
                metadata={"dynamic": True}
            ),
            Task(
                task_id="plan_execution",
                description="Create execution plan",
                dependencies={"analyze_requirements"},
                priority=2,
                metadata={"dynamic": True}
            )
        ]

        tasks[0].dependents.add("plan_execution")

        self.logger.info(f"Created {len(tasks)} initial dynamic tasks")
        return tasks

    def get_execution_order(self, tasks: List[Task]) -> List[List[Task]]:
        """
        Get execution order respecting dependencies.

        Uses topological sort to determine execution order.
        Returns tasks grouped by execution level (tasks in same level can run in parallel).

        Args:
            tasks: List of tasks with dependencies

        Returns:
            List of task levels (each level can execute in parallel)

        Raises:
            ValueError: If circular dependency detected
        """
        # Build task map
        task_map = {task.task_id: task for task in tasks}

        # Track completed and in-progress tasks
        completed = set()
        levels = []

        # Keep processing until all tasks scheduled
        max_iterations = len(tasks) + 1
        iteration = 0

        while len(completed) < len(tasks):
            iteration += 1
            if iteration > max_iterations:
                raise ValueError("Circular dependency detected")

            # Find all ready tasks
            ready_tasks = [
                task for task in tasks
                if task.task_id not in completed and task.is_ready(completed)
            ]

            if not ready_tasks:
                # Check if there are blocked tasks
                remaining = [t for t in tasks if t.task_id not in completed]
                if remaining:
                    raise ValueError(
                        f"Deadlock detected. Remaining tasks: {[t.task_id for t in remaining]}"
                    )
                break

            # Add this level
            levels.append(sorted(ready_tasks, key=lambda t: t.priority, reverse=True))

            # Mark as completed for dependency purposes
            completed.update(task.task_id for task in ready_tasks)

        self.logger.info(f"Execution order: {len(levels)} levels")
        return levels

    def validate_dependencies(self, tasks: List[Task]) -> List[str]:
        """
        Validate task dependencies for correctness.

        Args:
            tasks: List of tasks to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        task_ids = {task.task_id for task in tasks}

        for task in tasks:
            # Check for missing dependencies
            missing_deps = task.dependencies - task_ids
            if missing_deps:
                errors.append(
                    f"Task {task.task_id} has missing dependencies: {missing_deps}"
                )

            # Check for self-dependency
            if task.task_id in task.dependencies:
                errors.append(f"Task {task.task_id} depends on itself")

        # Check for circular dependencies
        try:
            self.get_execution_order(tasks)
        except ValueError as e:
            errors.append(f"Dependency error: {e}")

        return errors


class DependencyGraph:
    """
    Manages task dependencies and execution flow.

    Example:
        >>> graph = DependencyGraph()
        >>> graph.add_tasks(tasks)
        >>>
        >>> # Get ready tasks
        >>> ready = graph.get_ready_tasks()
        >>>
        >>> # Mark task as completed
        >>> graph.mark_completed("task_1", {"result": "success"})
        >>>
        >>> # Check status
        >>> status = graph.get_status_summary()
    """

    def __init__(self):
        """Initialize dependency graph."""
        self.tasks: Dict[str, Task] = {}
        self.completed: Set[str] = set()
        self.failed: Set[str] = set()
        self.in_progress: Set[str] = set()
        self.logger = logging.getLogger("DependencyGraph")

    def add_task(self, task: Task) -> None:
        """
        Add a task to the graph.

        Args:
            task: Task to add
        """
        self.tasks[task.task_id] = task
        self.logger.debug(f"Added task: {task.task_id}")

    def add_tasks(self, tasks: List[Task]) -> None:
        """
        Add multiple tasks to the graph.

        Args:
            tasks: List of tasks to add
        """
        for task in tasks:
            self.add_task(task)

    def get_ready_tasks(self) -> List[Task]:
        """
        Get all tasks that are ready to execute.

        Returns:
            List of ready tasks sorted by priority
        """
        ready = [
            task for task in self.tasks.values()
            if task.is_ready(self.completed) and task.task_id not in self.in_progress
        ]
        return sorted(ready, key=lambda t: t.priority, reverse=True)

    def mark_in_progress(self, task_id: str) -> None:
        """
        Mark task as in progress.

        Args:
            task_id: ID of task that started
        """
        if task_id in self.tasks:
            self.tasks[task_id].status = TaskStatus.IN_PROGRESS
            self.in_progress.add(task_id)
            self.logger.info(f"Task {task_id} started")

    def mark_completed(self, task_id: str, output_data: Dict[str, Any]) -> None:
        """
        Mark task as completed.

        Args:
            task_id: ID of completed task
            output_data: Task results
        """
        if task_id in self.tasks:
            self.tasks[task_id].mark_completed(output_data)
            self.completed.add(task_id)
            self.in_progress.discard(task_id)
            self.logger.info(f"Task {task_id} completed")

    def mark_failed(self, task_id: str, error: str) -> None:
        """
        Mark task as failed.

        Args:
            task_id: ID of failed task
            error: Error description
        """
        if task_id in self.tasks:
            self.tasks[task_id].mark_failed(error)
            self.failed.add(task_id)
            self.in_progress.discard(task_id)

            # Mark dependent tasks as blocked
            self._mark_blocked(task_id)

            self.logger.error(f"Task {task_id} failed: {error}")

    def _mark_blocked(self, failed_task_id: str) -> None:
        """Mark all dependent tasks as blocked."""
        failed_task = self.tasks[failed_task_id]

        for dependent_id in failed_task.dependents:
            if dependent_id in self.tasks:
                dependent = self.tasks[dependent_id]
                if dependent.status == TaskStatus.PENDING:
                    dependent.status = TaskStatus.BLOCKED
                    self.logger.warning(f"Task {dependent_id} blocked by failure of {failed_task_id}")

    def get_status_summary(self) -> Dict[str, Any]:
        """
        Get summary of task statuses.

        Returns:
            Dictionary with status counts and lists
        """
        status_counts = {}
        for status in TaskStatus:
            count = sum(1 for t in self.tasks.values() if t.status == status)
            status_counts[status.value] = count

        return {
            "total_tasks": len(self.tasks),
            "completed": len(self.completed),
            "failed": len(self.failed),
            "in_progress": len(self.in_progress),
            "pending": status_counts[TaskStatus.PENDING.value],
            "blocked": status_counts[TaskStatus.BLOCKED.value],
            "status_counts": status_counts,
            "progress_percent": (len(self.completed) / len(self.tasks) * 100) if self.tasks else 0
        }

    def is_complete(self) -> bool:
        """
        Check if all tasks are complete.

        Returns:
            True if no more tasks to execute
        """
        return len(self.completed) + len(self.failed) == len(self.tasks)

    def has_failures(self) -> bool:
        """
        Check if any tasks have failed.

        Returns:
            True if at least one task failed
        """
        return len(self.failed) > 0

    def __repr__(self) -> str:
        return f"<DependencyGraph(tasks={len(self.tasks)}, completed={len(self.completed)})>"
