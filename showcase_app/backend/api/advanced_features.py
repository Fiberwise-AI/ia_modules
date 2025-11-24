"""
API endpoints for advanced AI features in showcase app.

Provides endpoints for:
- Constitutional AI
- Advanced Memory
- Multi-Modal Processing
- Agent Collaboration
- Prompt Optimization
- Advanced Tools
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio

# Import advanced features
from ia_modules.patterns import (
    ConstitutionalAIStep,
    ConstitutionalConfig,
    Principle,
    PrincipleCategory
)
from ia_modules.memory import (
    MemoryManager,
    MemoryConfig,
    MemoryType
)
from ia_modules.multimodal import (
    MultiModalProcessor,
    MultiModalConfig,
    MultiModalInput,
    ModalityType
)
from ia_modules.agents import (
    AgentOrchestrator,
    CollaborationConfig
)
from ia_modules.prompt_optimization import (
    PromptOptimizer,
    OptimizationConfig
)
from ia_modules.tools import (
    AdvancedToolExecutor,
    AdvancedToolConfig
)

router = APIRouter(prefix="/api/advanced", tags=["advanced-features"])

# Global instances (in production, use dependency injection)
memory_manager: Optional[MemoryManager] = None
tool_executor: Optional[AdvancedToolExecutor] = None


# ===== Data Models =====

class PrincipleCreate(BaseModel):
    """Model for creating a principle."""
    name: str
    description: str
    critique_prompt: str
    weight: float = 1.0
    category: str = "custom"
    min_score: float = 0.7


class ConstitutionalRequest(BaseModel):
    """Request for Constitutional AI execution."""
    prompt: str
    context: Dict[str, Any] = Field(default_factory=dict)
    principles: List[str] = Field(default_factory=list)  # Names of pre-built constitutions
    custom_principles: List[PrincipleCreate] = Field(default_factory=list)
    max_revisions: int = 3
    min_quality_score: float = 0.8


class MemoryCreate(BaseModel):
    """Model for creating a memory."""
    content: str
    memory_type: Optional[str] = None
    importance: float = 0.5
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentTaskRequest(BaseModel):
    """Request for agent collaboration."""
    task: str
    pattern: str = "hierarchical"
    max_rounds: int = 3
    agent_types: List[str] = Field(default_factory=lambda: ["research", "analysis", "synthesis"])


class OptimizationRequest(BaseModel):
    """Request for prompt optimization."""
    initial_prompt: str
    test_cases: List[Dict[str, Any]]
    strategy: str = "genetic"
    max_iterations: int = 50
    population_size: int = 20


class ToolExecutionRequest(BaseModel):
    """Request for tool execution."""
    goal: str
    context: Dict[str, Any] = Field(default_factory=dict)
    enable_planning: bool = True


class ToolChainCreate(BaseModel):
    """Model for creating a tool chain."""
    name: str
    tools: List[str]
    description: Optional[str] = None


# ===== Constitutional AI Endpoints =====

@router.post("/constitutional-ai/execute")
async def execute_constitutional_ai(request: ConstitutionalRequest):
    """
    Execute Constitutional AI with specified principles.

    Returns refined response with quality metrics.
    """
    try:
        # Load principles
        principles = []

        # Add pre-built constitutions
        if "harmless" in request.principles:
            from ia_modules.patterns.constitutions import harmless_principles
            principles.extend(harmless_principles)
        if "helpful" in request.principles:
            from ia_modules.patterns.constitutions import helpful_principles
            principles.extend(helpful_principles)
        if "honest" in request.principles:
            from ia_modules.patterns.constitutions import honest_principles
            principles.extend(honest_principles)

        # Add custom principles
        for p in request.custom_principles:
            principles.append(Principle(
                name=p.name,
                description=p.description,
                critique_prompt=p.critique_prompt,
                weight=p.weight,
                category=PrincipleCategory(p.category) if p.category != "custom" else PrincipleCategory.CUSTOM,
                min_score=p.min_score
            ))

        if not principles:
            raise HTTPException(400, "At least one principle is required")

        # Create configuration
        config = ConstitutionalConfig(
            principles=principles,
            max_revisions=request.max_revisions,
            min_quality_score=request.min_quality_score
        )

        # Execute
        step = ConstitutionalAIStep(
            name="api_execution",
            prompt=request.prompt,
            config=config
        )

        result = await step.execute(request.context)

        return {
            "success": True,
            "response": result["response"],
            "quality_score": result["quality_score"],
            "revisions": result["revisions"],
            "principles_passed": result["principles_passed"],
            "principles_failed": result["principles_failed"],
            "history": [
                {
                    "iteration": h.iteration,
                    "quality_score": h.quality_score,
                    "timestamp": h.timestamp
                }
                for h in result["history"]
            ]
        }

    except Exception as e:
        raise HTTPException(500, f"Constitutional AI execution failed: {str(e)}")


@router.get("/constitutional-ai/constitutions")
async def list_constitutions():
    """List available pre-built constitutions."""
    return {
        "constitutions": [
            {
                "name": "harmless",
                "description": "Safety and ethics principles",
                "principles_count": 4
            },
            {
                "name": "helpful",
                "description": "Utility and helpfulness principles",
                "principles_count": 5
            },
            {
                "name": "honest",
                "description": "Truthfulness and accuracy principles",
                "principles_count": 5
            }
        ]
    }


# ===== Memory Management Endpoints =====

def get_memory_manager() -> MemoryManager:
    """Get or create memory manager."""
    global memory_manager
    if memory_manager is None:
        config = MemoryConfig(
            semantic_enabled=True,
            episodic_enabled=True,
            working_memory_size=10,
            compression_threshold=50,
            enable_embeddings=False  # Set to True if embeddings available
        )
        memory_manager = MemoryManager(config)
    return memory_manager


@router.post("/memory/add")
async def add_memory(memory: MemoryCreate):
    """Add a memory to the system."""
    try:
        manager = get_memory_manager()

        metadata = memory.metadata.copy()
        if memory.memory_type:
            metadata["memory_type"] = memory.memory_type
        metadata["importance"] = memory.importance

        memory_id = await manager.add(memory.content, metadata=metadata)

        return {
            "success": True,
            "memory_id": memory_id,
            "message": "Memory added successfully"
        }

    except Exception as e:
        raise HTTPException(500, f"Failed to add memory: {str(e)}")


@router.get("/memory/retrieve")
async def retrieve_memories(query: str, k: int = 5, memory_type: Optional[str] = None):
    """Retrieve relevant memories."""
    try:
        manager = get_memory_manager()

        memory_types = None
        if memory_type:
            memory_types = [MemoryType(memory_type)]

        results = await manager.retrieve(query, k=k, memory_types=memory_types)

        return {
            "success": True,
            "count": len(results),
            "memories": [
                {
                    "id": m.id,
                    "content": m.content,
                    "memory_type": m.memory_type.value,
                    "importance": m.importance,
                    "timestamp": m.timestamp,
                    "access_count": m.access_count
                }
                for m in results
            ]
        }

    except Exception as e:
        raise HTTPException(500, f"Failed to retrieve memories: {str(e)}")


@router.get("/memory/stats")
async def get_memory_stats():
    """Get memory system statistics."""
    try:
        manager = get_memory_manager()
        stats = await manager.get_stats()

        return {
            "success": True,
            "stats": stats
        }

    except Exception as e:
        raise HTTPException(500, f"Failed to get stats: {str(e)}")


@router.post("/memory/compress")
async def compress_memories():
    """Manually trigger memory compression."""
    try:
        manager = get_memory_manager()
        compressed_count = await manager.compress()

        return {
            "success": True,
            "compressed_count": compressed_count,
            "message": f"Compressed {compressed_count} memories"
        }

    except Exception as e:
        raise HTTPException(500, f"Failed to compress: {str(e)}")


@router.delete("/memory/clear")
async def clear_memories(memory_type: Optional[str] = None):
    """Clear memories (optionally by type)."""
    try:
        manager = get_memory_manager()

        memory_types = None
        if memory_type:
            memory_types = [MemoryType(memory_type)]

        await manager.clear(memory_types=memory_types)

        return {
            "success": True,
            "message": "Memories cleared successfully"
        }

    except Exception as e:
        raise HTTPException(500, f"Failed to clear memories: {str(e)}")


# ===== Multi-Modal Endpoints =====

@router.post("/multimodal/process-image")
async def process_image(file: UploadFile = File(...), prompt: str = "Describe this image"):
    """Process uploaded image with vision model."""
    try:
        # Read image data
        image_data = await file.read()

        # Create processor
        config = MultiModalConfig(
            image_model="gpt-4-vision-preview",
            vision_provider="openai"
        )
        processor = MultiModalProcessor(config)

        # Process
        result = await processor.process_image(image_data, prompt)

        return {
            "success": True,
            "result": result,
            "filename": file.filename
        }

    except Exception as e:
        raise HTTPException(500, f"Image processing failed: {str(e)}")


@router.post("/multimodal/process-audio")
async def process_audio(file: UploadFile = File(...)):
    """Transcribe uploaded audio file."""
    try:
        # Read audio data
        audio_data = await file.read()

        # Create processor
        config = MultiModalConfig(audio_model="whisper-1")
        processor = MultiModalProcessor(config)

        # Transcribe
        result = await processor.process_audio(audio_data)

        return {
            "success": True,
            "transcription": result,
            "filename": file.filename
        }

    except Exception as e:
        raise HTTPException(500, f"Audio processing failed: {str(e)}")


@router.post("/multimodal/fuse")
async def fuse_modalities(text: Optional[str] = None, image_file: Optional[UploadFile] = File(None)):
    """Fuse multiple modalities."""
    try:
        inputs = []

        if text:
            inputs.append(MultiModalInput(
                content=text,
                modality=ModalityType.TEXT
            ))

        if image_file:
            image_data = await image_file.read()
            inputs.append(MultiModalInput(
                content=image_data,
                modality=ModalityType.IMAGE
            ))

        if not inputs:
            raise HTTPException(400, "At least one modality is required")

        # Create processor
        config = MultiModalConfig(enable_fusion=True)
        processor = MultiModalProcessor(config)

        # Process
        result = await processor.process(inputs)

        return {
            "success": True,
            "fused_result": result.result,
            "modality_count": len(inputs)
        }

    except Exception as e:
        raise HTTPException(500, f"Fusion failed: {str(e)}")


# ===== Agent Collaboration Endpoints =====

@router.post("/agents/orchestrate")
async def orchestrate_agents(request: AgentTaskRequest):
    """Execute multi-agent task with orchestration."""
    try:
        # Create configuration
        config = CollaborationConfig(
            pattern=request.pattern,
            max_rounds=request.max_rounds
        )

        # Create orchestrator
        orchestrator = AgentOrchestrator(config)

        # Register agents based on request
        from ia_modules.agents import ResearchAgent, AnalysisAgent, SynthesisAgent, CriticAgent

        if "research" in request.agent_types:
            orchestrator.register_agent(ResearchAgent(name="researcher"))
        if "analysis" in request.agent_types:
            orchestrator.register_agent(AnalysisAgent(name="analyst"))
        if "synthesis" in request.agent_types:
            orchestrator.register_agent(SynthesisAgent(name="synthesizer"))
        if "critic" in request.agent_types:
            orchestrator.register_agent(CriticAgent(name="critic"))

        # Execute
        result = await orchestrator.execute(task=request.task)

        return {
            "success": True,
            "result": result.get("result"),
            "agent_contributions": result.get("agent_contributions", {}),
            "rounds": result.get("rounds", 0),
            "pattern": request.pattern
        }

    except Exception as e:
        raise HTTPException(500, f"Agent orchestration failed: {str(e)}")


@router.get("/agents/patterns")
async def list_collaboration_patterns():
    """List available collaboration patterns."""
    return {
        "patterns": [
            {
                "name": "hierarchical",
                "description": "Leader-worker pattern with centralized coordination"
            },
            {
                "name": "peer_to_peer",
                "description": "Equal collaboration between agents"
            },
            {
                "name": "debate",
                "description": "Adversarial discussion for best solution"
            },
            {
                "name": "consensus",
                "description": "Agreement-based decision making"
            }
        ]
    }


@router.get("/agents/specialists")
async def list_specialist_agents():
    """List available specialist agent types."""
    return {
        "agents": [
            {
                "type": "research",
                "name": "Research Agent",
                "description": "Gathers and organizes information"
            },
            {
                "type": "analysis",
                "name": "Analysis Agent",
                "description": "Analyzes data and identifies patterns"
            },
            {
                "type": "synthesis",
                "name": "Synthesis Agent",
                "description": "Combines results into coherent output"
            },
            {
                "type": "critic",
                "name": "Critic Agent",
                "description": "Reviews and provides quality feedback"
            }
        ]
    }


# ===== Prompt Optimization Endpoints =====

# Store optimization jobs (in production, use database)
optimization_jobs: Dict[str, Dict] = {}


@router.post("/prompt-optimization/optimize")
async def optimize_prompt(request: OptimizationRequest):
    """Start prompt optimization job."""
    try:
        import uuid
        job_id = str(uuid.uuid4())

        # Store job info
        optimization_jobs[job_id] = {
            "status": "running",
            "progress": 0,
            "best_prompt": None,
            "best_score": 0
        }

        # Start optimization in background
        asyncio.create_task(run_optimization(job_id, request))

        return {
            "success": True,
            "job_id": job_id,
            "message": "Optimization started"
        }

    except Exception as e:
        raise HTTPException(500, f"Failed to start optimization: {str(e)}")


async def run_optimization(job_id: str, request: OptimizationRequest):
    """Background task for optimization."""
    try:
        config = OptimizationConfig(
            strategy=request.strategy,
            max_iterations=request.max_iterations,
            population_size=request.population_size
        )

        optimizer = PromptOptimizer(config)

        # Run optimization
        result = await optimizer.optimize(
            initial_prompt=request.initial_prompt,
            test_cases=request.test_cases
        )

        # Update job
        optimization_jobs[job_id] = {
            "status": "completed",
            "progress": 100,
            "best_prompt": result.template,
            "best_score": result.score
        }

    except Exception as e:
        optimization_jobs[job_id] = {
            "status": "failed",
            "error": str(e)
        }


@router.get("/prompt-optimization/status/{job_id}")
async def get_optimization_status(job_id: str):
    """Get optimization job status."""
    if job_id not in optimization_jobs:
        raise HTTPException(404, "Job not found")

    return {
        "success": True,
        "job_id": job_id,
        **optimization_jobs[job_id]
    }


@router.get("/prompt-optimization/strategies")
async def list_optimization_strategies():
    """List available optimization strategies."""
    return {
        "strategies": [
            {
                "name": "genetic",
                "description": "Genetic algorithm with mutation and crossover"
            },
            {
                "name": "reinforcement",
                "description": "Q-learning based optimization"
            },
            {
                "name": "ab_test",
                "description": "Statistical A/B testing"
            },
            {
                "name": "grid_search",
                "description": "Exhaustive parameter search"
            }
        ]
    }


# ===== Advanced Tools Endpoints =====

def get_tool_executor() -> AdvancedToolExecutor:
    """Get or create tool executor."""
    global tool_executor
    if tool_executor is None:
        config = AdvancedToolConfig(
            max_parallel=5,
            enable_planning=True,
            retry_attempts=3
        )
        tool_executor = AdvancedToolExecutor(config)

        # Register built-in tools
        register_builtin_tools(tool_executor)

    return tool_executor


def register_builtin_tools(executor: AdvancedToolExecutor):
    """Register built-in tools."""
    # This would register actual tools
    # For now, just a placeholder
    pass


@router.post("/tools/execute")
async def execute_tool(request: ToolExecutionRequest):
    """Execute tool with planning."""
    try:
        executor = get_tool_executor()

        result = await executor.execute(
            goal=request.goal,
            context=request.context
        )

        return {
            "success": True,
            "result": result.get("result"),
            "execution_plan": result.get("plan"),
            "tools_used": result.get("tools_used", [])
        }

    except Exception as e:
        raise HTTPException(500, f"Tool execution failed: {str(e)}")


@router.get("/tools/registry")
async def list_available_tools():
    """List all registered tools."""
    try:
        executor = get_tool_executor()
        tools = executor.registry.list_all()

        return {
            "success": True,
            "count": len(tools),
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "version": tool.version
                }
                for tool in tools
            ]
        }

    except Exception as e:
        raise HTTPException(500, f"Failed to list tools: {str(e)}")


# ===== WebSocket for Live Updates =====

@router.websocket("/ws/live-updates")
async def websocket_live_updates(websocket: WebSocket):
    """
    WebSocket endpoint for live updates from:
    - Agent collaboration
    - Memory updates
    - Tool execution
    - Optimization progress
    """
    await websocket.accept()

    try:
        while True:
            # Send periodic updates
            await asyncio.sleep(1)

            # Example update
            await websocket.send_json({
                "type": "status",
                "message": "System running",
                "timestamp": asyncio.get_event_loop().time()
            })

    except WebSocketDisconnect:
        print("Client disconnected")
