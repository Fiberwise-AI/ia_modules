"""
Agent Collaboration API endpoints.

This module provides REST API endpoints for demonstrating multi-agent collaboration,
including agent orchestration, collaboration patterns, and real-time agent communication.
"""

from fastapi import APIRouter, HTTPException, Depends, Request, WebSocket, WebSocketDisconnect
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import logging
import asyncio

logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response Models

class AgentRole(str, Enum):
    """Agent roles/specializations."""
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    CRITIC = "critic"
    PLANNER = "planner"
    EXECUTOR = "executor"
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"


class CollaborationPattern(str, Enum):
    """Patterns for agent collaboration."""
    SEQUENTIAL = "sequential"  # Agents work one after another
    PARALLEL = "parallel"  # Agents work simultaneously
    HIERARCHICAL = "hierarchical"  # Tree-like delegation
    DEBATE = "debate"  # Agents debate/critique each other
    CONSENSUS = "consensus"  # Agents reach consensus
    PIPELINE = "pipeline"  # Data flows through agents


class AgentStatus(str, Enum):
    """Status of an agent."""
    IDLE = "idle"
    THINKING = "thinking"
    WORKING = "working"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentDefinition(BaseModel):
    """Definition of an agent."""
    id: str = Field(..., description="Unique agent ID")
    name: str = Field(..., description="Agent name")
    role: AgentRole = Field(..., description="Agent role/specialization")
    description: str = Field(..., description="What this agent does")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    model: Optional[str] = Field(None, description="LLM model to use")
    temperature: float = Field(0.7, ge=0, le=2, description="Generation temperature")
    system_prompt: Optional[str] = Field(None, description="System prompt for agent")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "researcher_1",
                "name": "Research Agent",
                "role": "researcher",
                "description": "Conducts research and gathers information",
                "capabilities": ["web_search", "data_analysis"],
                "model": "gpt-4",
                "temperature": 0.7
            }
        }


class AgentMessage(BaseModel):
    """Message between agents."""
    from_agent: str = Field(..., description="Sender agent ID")
    to_agent: Optional[str] = Field(None, description="Recipient agent ID (None for broadcast)")
    content: str = Field(..., description="Message content")
    message_type: str = Field("message", description="Type of message")
    timestamp: float = Field(..., description="Message timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class AgentState(BaseModel):
    """Current state of an agent."""
    agent_id: str = Field(..., description="Agent ID")
    status: AgentStatus = Field(..., description="Current status")
    current_task: Optional[str] = Field(None, description="Current task description")
    progress: float = Field(0, ge=0, le=1, description="Task progress (0-1)")
    output: Optional[str] = Field(None, description="Current output")
    messages_sent: int = Field(0, description="Number of messages sent")
    messages_received: int = Field(0, description="Number of messages received")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional state data")


class OrchestrationRequest(BaseModel):
    """Request to orchestrate multiple agents."""
    task: str = Field(..., description="Main task to accomplish")
    agents: List[AgentDefinition] = Field(..., description="Agents to use")
    pattern: CollaborationPattern = Field(..., description="Collaboration pattern")
    max_iterations: int = Field(5, ge=1, le=20, description="Maximum collaboration iterations")
    timeout_seconds: float = Field(300, gt=0, description="Timeout in seconds")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")

    class Config:
        json_schema_extra = {
            "example": {
                "task": "Write a comprehensive article about AI",
                "agents": [
                    {
                        "id": "researcher",
                        "name": "Researcher",
                        "role": "researcher",
                        "description": "Gathers information",
                        "capabilities": ["research"]
                    },
                    {
                        "id": "writer",
                        "name": "Writer",
                        "role": "writer",
                        "description": "Writes content",
                        "capabilities": ["writing"]
                    }
                ],
                "pattern": "sequential",
                "max_iterations": 3
            }
        }


class OrchestrationResponse(BaseModel):
    """Response from agent orchestration."""
    success: bool = Field(..., description="Whether orchestration succeeded")
    task_id: str = Field(..., description="Unique task ID")
    final_output: str = Field(..., description="Final result from collaboration")
    iterations: int = Field(..., description="Number of iterations completed")
    agents_used: List[str] = Field(..., description="Agent IDs that participated")
    pattern_used: CollaborationPattern = Field(..., description="Pattern used")
    execution_time_ms: float = Field(..., description="Total execution time")
    agent_states: List[AgentState] = Field(..., description="Final state of all agents")
    message_history: List[AgentMessage] = Field(..., description="Complete message history")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class PatternInfo(BaseModel):
    """Information about a collaboration pattern."""
    name: CollaborationPattern = Field(..., description="Pattern name")
    description: str = Field(..., description="Pattern description")
    use_cases: List[str] = Field(..., description="When to use this pattern")
    strengths: List[str] = Field(..., description="Pattern strengths")
    weaknesses: List[str] = Field(..., description="Pattern weaknesses")
    example_tasks: List[str] = Field(..., description="Example tasks for this pattern")


class GetPatternsResponse(BaseModel):
    """Response with available collaboration patterns."""
    patterns: List[PatternInfo] = Field(..., description="Available patterns")
    total: int = Field(..., description="Total number of patterns")


class SpecialistInfo(BaseModel):
    """Information about a specialist agent."""
    role: AgentRole = Field(..., description="Agent role")
    name: str = Field(..., description="Role name")
    description: str = Field(..., description="What this specialist does")
    capabilities: List[str] = Field(..., description="Typical capabilities")
    best_for: List[str] = Field(..., description="Best use cases")
    example_tasks: List[str] = Field(..., description="Example tasks")


class GetSpecialistsResponse(BaseModel):
    """Response with available specialist agents."""
    specialists: List[SpecialistInfo] = Field(..., description="Available specialists")
    total: int = Field(..., description="Total number of specialists")


# In-memory storage for orchestrations
active_orchestrations: Dict[str, Dict[str, Any]] = {}
orchestration_counter = 0


# Pattern definitions
COLLABORATION_PATTERNS = {
    CollaborationPattern.SEQUENTIAL: PatternInfo(
        name=CollaborationPattern.SEQUENTIAL,
        description="Agents work one after another in a defined sequence",
        use_cases=[
            "Pipeline processing where output of one agent feeds into next",
            "Multi-stage refinement tasks",
            "Progressive transformation workflows"
        ],
        strengths=[
            "Clear flow of information",
            "Easy to debug",
            "Predictable execution order"
        ],
        weaknesses=[
            "Can be slower than parallel",
            "Bottlenecks if one agent is slow",
            "No concurrent processing"
        ],
        example_tasks=[
            "Research -> Analysis -> Writing -> Review",
            "Data collection -> Processing -> Visualization",
            "Planning -> Execution -> Validation"
        ]
    ),
    CollaborationPattern.PARALLEL: PatternInfo(
        name=CollaborationPattern.PARALLEL,
        description="Agents work simultaneously on different aspects",
        use_cases=[
            "Independent sub-tasks",
            "Diverse perspective gathering",
            "Fast parallel processing"
        ],
        strengths=[
            "Faster execution",
            "Diverse outputs",
            "Efficient resource usage"
        ],
        weaknesses=[
            "Requires result aggregation",
            "May produce conflicting outputs",
            "Coordination overhead"
        ],
        example_tasks=[
            "Multiple researchers gathering different sources",
            "Parallel analysis of different datasets",
            "Concurrent generation of alternatives"
        ]
    ),
    CollaborationPattern.HIERARCHICAL: PatternInfo(
        name=CollaborationPattern.HIERARCHICAL,
        description="Tree-like structure with coordinator delegating to sub-agents",
        use_cases=[
            "Complex tasks requiring decomposition",
            "Coordinated multi-agent systems",
            "Hierarchical decision making"
        ],
        strengths=[
            "Handles complexity well",
            "Clear responsibility hierarchy",
            "Scalable coordination"
        ],
        weaknesses=[
            "Coordinator can be bottleneck",
            "More communication overhead",
            "Requires good task decomposition"
        ],
        example_tasks=[
            "Project manager delegating to specialists",
            "Complex research with sub-topics",
            "Multi-level system design"
        ]
    ),
    CollaborationPattern.DEBATE: PatternInfo(
        name=CollaborationPattern.DEBATE,
        description="Agents debate and critique each other to improve quality",
        use_cases=[
            "Quality improvement through critique",
            "Exploring different perspectives",
            "Robust decision making"
        ],
        strengths=[
            "High quality outputs",
            "Multiple perspectives",
            "Error detection through critique"
        ],
        weaknesses=[
            "Can be slower",
            "May not converge",
            "Requires good moderation"
        ],
        example_tasks=[
            "Proposal review and improvement",
            "Design alternatives evaluation",
            "Strategic planning with multiple viewpoints"
        ]
    ),
    CollaborationPattern.CONSENSUS: PatternInfo(
        name=CollaborationPattern.CONSENSUS,
        description="Agents work towards reaching agreement",
        use_cases=[
            "Decision making requiring agreement",
            "Conflict resolution",
            "Collaborative problem solving"
        ],
        strengths=[
            "Balanced decisions",
            "Buy-in from all agents",
            "Conflict resolution"
        ],
        weaknesses=[
            "Can be slow to converge",
            "May result in compromises",
            "Requires consensus mechanism"
        ],
        example_tasks=[
            "Team decision on approach",
            "Resource allocation",
            "Priority setting"
        ]
    ),
    CollaborationPattern.PIPELINE: PatternInfo(
        name=CollaborationPattern.PIPELINE,
        description="Data flows through agents with transformations at each stage",
        use_cases=[
            "Data transformation pipelines",
            "Progressive refinement",
            "Multi-stage processing"
        ],
        strengths=[
            "Clean data flow",
            "Modular stages",
            "Easy to extend"
        ],
        weaknesses=[
            "Sequential bottlenecks",
            "Error propagation",
            "Stage dependencies"
        ],
        example_tasks=[
            "ETL processes",
            "Content generation and refinement",
            "Multi-stage analysis"
        ]
    )
}


# Specialist definitions
SPECIALIST_AGENTS = {
    AgentRole.RESEARCHER: SpecialistInfo(
        role=AgentRole.RESEARCHER,
        name="Research Specialist",
        description="Gathers information, conducts research, and finds relevant sources",
        capabilities=["web_search", "data_gathering", "source_verification"],
        best_for=[
            "Information gathering",
            "Fact finding",
            "Literature review"
        ],
        example_tasks=[
            "Research latest developments in AI",
            "Find relevant academic papers",
            "Gather market data"
        ]
    ),
    AgentRole.ANALYST: SpecialistInfo(
        role=AgentRole.ANALYST,
        name="Analysis Specialist",
        description="Analyzes data, identifies patterns, and draws insights",
        capabilities=["data_analysis", "pattern_recognition", "insight_generation"],
        best_for=[
            "Data analysis",
            "Trend identification",
            "Statistical analysis"
        ],
        example_tasks=[
            "Analyze sales trends",
            "Identify patterns in user behavior",
            "Evaluate performance metrics"
        ]
    ),
    AgentRole.WRITER: SpecialistInfo(
        role=AgentRole.WRITER,
        name="Writing Specialist",
        description="Creates written content, documentation, and narratives",
        capabilities=["content_creation", "documentation", "storytelling"],
        best_for=[
            "Content writing",
            "Documentation",
            "Narrative creation"
        ],
        example_tasks=[
            "Write article from research",
            "Create technical documentation",
            "Draft marketing copy"
        ]
    ),
    AgentRole.CRITIC: SpecialistInfo(
        role=AgentRole.CRITIC,
        name="Critique Specialist",
        description="Reviews and critiques work to improve quality",
        capabilities=["quality_review", "error_detection", "improvement_suggestions"],
        best_for=[
            "Quality assurance",
            "Code review",
            "Content improvement"
        ],
        example_tasks=[
            "Review written content",
            "Critique design decisions",
            "Evaluate code quality"
        ]
    ),
    AgentRole.PLANNER: SpecialistInfo(
        role=AgentRole.PLANNER,
        name="Planning Specialist",
        description="Creates plans, strategies, and roadmaps",
        capabilities=["strategic_planning", "task_decomposition", "roadmap_creation"],
        best_for=[
            "Project planning",
            "Strategy development",
            "Task breakdown"
        ],
        example_tasks=[
            "Create project plan",
            "Develop strategy",
            "Break down complex tasks"
        ]
    ),
    AgentRole.EXECUTOR: SpecialistInfo(
        role=AgentRole.EXECUTOR,
        name="Execution Specialist",
        description="Executes plans and implements solutions",
        capabilities=["task_execution", "implementation", "problem_solving"],
        best_for=[
            "Task execution",
            "Implementation",
            "Problem solving"
        ],
        example_tasks=[
            "Execute planned tasks",
            "Implement solutions",
            "Solve specific problems"
        ]
    ),
    AgentRole.COORDINATOR: SpecialistInfo(
        role=AgentRole.COORDINATOR,
        name="Coordination Specialist",
        description="Coordinates work between multiple agents",
        capabilities=["coordination", "delegation", "communication_management"],
        best_for=[
            "Multi-agent coordination",
            "Task delegation",
            "Workflow management"
        ],
        example_tasks=[
            "Coordinate team effort",
            "Delegate to specialists",
            "Manage workflows"
        ]
    )
}


# Dependency injection
def get_agent_service(request: Request):
    """Get agent service from app state."""
    # For now, return None. In production, return request.app.state.services.agent_service
    return None


# API Endpoints

@router.post("/orchestrate", response_model=OrchestrationResponse)
async def orchestrate_agents(
    request: OrchestrationRequest,
    service=Depends(get_agent_service)
) -> OrchestrationResponse:
    """
    Orchestrate multiple agents to accomplish a task.

    This endpoint coordinates multiple AI agents working together using
    various collaboration patterns to accomplish complex tasks.

    The orchestration process:
    1. Initializes all specified agents
    2. Applies the selected collaboration pattern
    3. Manages communication between agents
    4. Aggregates results
    5. Returns final output with complete history

    Example:
        ```python
        response = await client.post("/api/agents/orchestrate", json={
            "task": "Create a market analysis report",
            "agents": [
                {"id": "r1", "name": "Researcher", "role": "researcher"},
                {"id": "a1", "name": "Analyst", "role": "analyst"},
                {"id": "w1", "name": "Writer", "role": "writer"}
            ],
            "pattern": "sequential",
            "max_iterations": 3
        })
        ```
    """
    global orchestration_counter
    import time

    try:
        start_time = time.time()

        # Generate task ID
        orchestration_counter += 1
        task_id = f"task_{orchestration_counter}"

        logger.info(
            f"Starting orchestration {task_id}: {request.task} "
            f"with {len(request.agents)} agents using {request.pattern.value} pattern"
        )

        # In production, use ia_modules.agents.Orchestrator
        # For demo, create mock orchestration

        # Initialize agent states
        agent_states = []
        for agent in request.agents:
            agent_states.append(AgentState(
                agent_id=agent.id,
                status=AgentStatus.IDLE,
                current_task=None,
                progress=0.0,
                output=None,
                messages_sent=0,
                messages_received=0,
                metadata={"role": agent.role.value}
            ))

        # Simulate collaboration
        message_history = []
        iterations = 0

        # Pattern-specific execution
        if request.pattern == CollaborationPattern.SEQUENTIAL:
            # Sequential execution
            for i, agent in enumerate(request.agents):
                iterations += 1
                agent_state = agent_states[i]
                agent_state.status = AgentStatus.WORKING
                agent_state.current_task = request.task
                agent_state.progress = 0.5

                # Simulate work
                await asyncio.sleep(0.1)

                # Create output
                output = f"Output from {agent.name} ({agent.role.value}): Processed '{request.task}'"
                agent_state.output = output
                agent_state.status = AgentStatus.COMPLETED
                agent_state.progress = 1.0

                # Add message
                if i < len(request.agents) - 1:
                    next_agent = request.agents[i + 1]
                    message = AgentMessage(
                        from_agent=agent.id,
                        to_agent=next_agent.id,
                        content=output,
                        message_type="handoff",
                        timestamp=time.time(),
                        metadata={"iteration": iterations}
                    )
                    message_history.append(message)
                    agent_state.messages_sent += 1
                    agent_states[i + 1].messages_received += 1

        elif request.pattern == CollaborationPattern.PARALLEL:
            # Parallel execution
            iterations = 1
            for i, agent in enumerate(request.agents):
                agent_state = agent_states[i]
                agent_state.status = AgentStatus.WORKING
                agent_state.current_task = request.task
                agent_state.progress = 0.5

            # Simulate parallel work
            await asyncio.sleep(0.1)

            for i, agent in enumerate(request.agents):
                agent_state = agent_states[i]
                output = f"Parallel output from {agent.name}: {agent.role.value} perspective on '{request.task}'"
                agent_state.output = output
                agent_state.status = AgentStatus.COMPLETED
                agent_state.progress = 1.0

                # Broadcast message
                message = AgentMessage(
                    from_agent=agent.id,
                    to_agent=None,
                    content=output,
                    message_type="broadcast",
                    timestamp=time.time(),
                    metadata={"iteration": iterations}
                )
                message_history.append(message)
                agent_state.messages_sent += 1

        else:
            # Default execution for other patterns
            iterations = min(request.max_iterations, len(request.agents))
            for i, agent in enumerate(request.agents):
                agent_state = agent_states[i]
                agent_state.status = AgentStatus.COMPLETED
                agent_state.progress = 1.0
                agent_state.output = f"{agent.name} completed using {request.pattern.value} pattern"

        # Create final output
        final_output = f"Collaborative result from {len(request.agents)} agents using {request.pattern.value} pattern for: {request.task}\n\n"
        final_output += "\n".join([
            f"- {state.agent_id}: {state.output}"
            for state in agent_states if state.output
        ])

        execution_time_ms = (time.time() - start_time) * 1000

        # Store orchestration
        active_orchestrations[task_id] = {
            "task_id": task_id,
            "request": request.model_dump(),
            "response": {
                "final_output": final_output,
                "iterations": iterations,
                "agent_states": [s.model_dump() for s in agent_states],
                "message_history": [m.model_dump() for m in message_history]
            },
            "created_at": time.time()
        }

        logger.info(f"Orchestration {task_id} completed in {execution_time_ms:.2f}ms")

        return OrchestrationResponse(
            success=True,
            task_id=task_id,
            final_output=final_output,
            iterations=iterations,
            agents_used=[agent.id for agent in request.agents],
            pattern_used=request.pattern,
            execution_time_ms=execution_time_ms,
            agent_states=agent_states,
            message_history=message_history,
            metadata={
                "total_messages": len(message_history),
                "pattern": request.pattern.value
            }
        )

    except Exception as e:
        logger.error(f"Error orchestrating agents: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to orchestrate agents: {str(e)}"
        )


@router.get("/patterns", response_model=GetPatternsResponse)
async def get_collaboration_patterns() -> GetPatternsResponse:
    """
    Get available collaboration patterns.

    Returns information about all available patterns for agent collaboration,
    including their strengths, weaknesses, and best use cases.

    Example:
        ```python
        patterns = await client.get("/api/agents/patterns")
        for pattern in patterns.patterns:
            print(f"{pattern.name}: {pattern.description}")
        ```
    """
    try:
        patterns = list(COLLABORATION_PATTERNS.values())

        return GetPatternsResponse(
            patterns=patterns,
            total=len(patterns)
        )

    except Exception as e:
        logger.error(f"Error retrieving patterns: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve patterns: {str(e)}"
        )


@router.get("/specialists", response_model=GetSpecialistsResponse)
async def get_specialist_agents() -> GetSpecialistsResponse:
    """
    Get available specialist agents.

    Returns information about all available specialist agent roles,
    including their capabilities and best use cases.

    Example:
        ```python
        specialists = await client.get("/api/agents/specialists")
        for spec in specialists.specialists:
            print(f"{spec.name}: {spec.description}")
        ```
    """
    try:
        specialists = list(SPECIALIST_AGENTS.values())

        return GetSpecialistsResponse(
            specialists=specialists,
            total=len(specialists)
        )

    except Exception as e:
        logger.error(f"Error retrieving specialists: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve specialists: {str(e)}"
        )


@router.get("/orchestrations/{task_id}")
async def get_orchestration(
    task_id: str,
    service=Depends(get_agent_service)
):
    """
    Get details of a specific orchestration.

    Args:
        task_id: ID of the orchestration task

    Returns:
        Complete orchestration details

    Example:
        ```python
        details = await client.get("/api/agents/orchestrations/task_1")
        ```
    """
    try:
        if task_id not in active_orchestrations:
            raise HTTPException(
                status_code=404,
                detail=f"Orchestration '{task_id}' not found"
            )

        return active_orchestrations[task_id]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving orchestration: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve orchestration: {str(e)}"
        )


@router.websocket("/live")
async def agent_collaboration_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time agent collaboration updates.

    Provides live updates during agent orchestration including:
    - Agent status changes
    - Messages between agents
    - Progress updates
    - Final results

    Example:
        ```python
        async with websockets.connect("ws://localhost:5555/api/agents/live") as ws:
            # Send orchestration request
            await ws.send(json.dumps({
                "action": "start_orchestration",
                "task": "Research AI trends",
                "agents": [...]
            }))

            # Receive updates
            while True:
                update = json.loads(await ws.recv())
                print(f"Update: {update}")
        ```
    """
    await websocket.accept()
    logger.info("Agent collaboration WebSocket connected")

    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            action = data.get("action")

            if action == "start_orchestration":
                # Start orchestration and stream updates
                await websocket.send_json({
                    "type": "status",
                    "message": "Starting orchestration...",
                    "timestamp": datetime.utcnow().isoformat()
                })

                # Simulate agent updates
                for i in range(3):
                    await asyncio.sleep(0.5)
                    await websocket.send_json({
                        "type": "agent_update",
                        "agent_id": f"agent_{i}",
                        "status": "working",
                        "progress": 0.5,
                        "timestamp": datetime.utcnow().isoformat()
                    })

                await websocket.send_json({
                    "type": "complete",
                    "message": "Orchestration completed",
                    "timestamp": datetime.utcnow().isoformat()
                })

            elif action == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                })

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown action: {action}"
                })

    except WebSocketDisconnect:
        logger.info("Agent collaboration WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        await websocket.close()
