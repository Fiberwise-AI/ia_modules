"""
Pattern API Endpoints

Demonstrates agentic design patterns built from first principles.

These patterns show the evolution of AI capabilities:
- Basic LLM → Specialized Agent → Tool-Using Agent → Memory Agent → ReAct Agent

Each endpoint demonstrates a specific pattern that transforms 
text generation into intelligent, autonomous behavior.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import os
from services.pattern_service import PatternService
from services.llm_monitoring_service import LLMMonitoringService

router = APIRouter(prefix="/api/patterns", tags=["patterns"])
pattern_service = PatternService()
monitoring_service = LLMMonitoringService()


# ==================== LLM STATUS ENDPOINT ====================

@router.get("/llm/status")
async def get_llm_status():
    """
    Get LLM configuration status
    
    Returns which providers are configured and available
    """
    providers = []
    
    # Check OpenAI
    if os.getenv("OPENAI_API_KEY"):
        providers.append({
            "name": "openai",
            "status": "configured",
            "model": os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
        })
    else:
        providers.append({
            "name": "openai",
            "status": "not_configured",
            "setup_guide": "Set OPENAI_API_KEY in .env file"
        })
    
    # Check Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        providers.append({
            "name": "anthropic",
            "status": "configured",
            "model": os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        })
    else:
        providers.append({
            "name": "anthropic",
            "status": "not_configured",
            "setup_guide": "Set ANTHROPIC_API_KEY in .env file"
        })
    
    # Check Gemini
    if os.getenv("GEMINI_API_KEY"):
        providers.append({
            "name": "gemini",
            "status": "configured",
            "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        })
    else:
        providers.append({
            "name": "gemini",
            "status": "not_configured",
            "setup_guide": "Set GEMINI_API_KEY in .env file"
        })
    
    configured_count = sum(1 for p in providers if p["status"] == "configured")
    default_provider = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
    
    return {
        "configured": configured_count > 0,
        "configured_count": configured_count,
        "total_providers": len(providers),
        "default_provider": default_provider,
        "providers": providers,
        "message": "At least one LLM provider must be configured" if configured_count == 0 else f"{configured_count} provider(s) ready"
    }


@router.get("/llm/stats")
async def get_llm_stats():
    """
    Get LLM usage statistics
    
    Returns token usage, costs, and rate limit status
    """
    return monitoring_service.get_stats()


# ==================== REQUEST MODELS ====================

class ReflectionRequest(BaseModel):
    """Request for reflection pattern demo"""
    initial_output: str = Field(..., description="Initial agent output")
    criteria: Dict[str, str] = Field(..., description="Quality criteria")
    max_iterations: int = Field(default=3, ge=1, le=10)


class PlanningRequest(BaseModel):
    """Request for planning pattern demo"""
    goal: str = Field(..., description="Goal to achieve")
    constraints: Optional[Dict[str, Any]] = Field(default=None)


class ToolUseRequest(BaseModel):
    """Request for tool use pattern demo"""
    task: str = Field(..., description="Task to accomplish")
    available_tools: List[str] = Field(..., description="Available tool names")


class AgenticRAGRequest(BaseModel):
    """Request for agentic RAG pattern demo"""
    query: str = Field(..., description="Search query")
    max_refinements: int = Field(default=3, ge=1, le=5)


class MetacognitionRequest(BaseModel):
    """Request for metacognition pattern demo"""
    execution_trace: List[Dict[str, Any]] = Field(..., description="Execution history")
    performance_metrics: Dict[str, float] = Field(..., description="Performance metrics")


# ==================== ENDPOINTS ====================

@router.post("/reflection")
async def demonstrate_reflection(request: ReflectionRequest) -> Dict[str, Any]:
    """
    Demonstrate reflection pattern
    
    Shows how an agent can:
    - Self-critique its outputs
    - Iteratively improve through multiple refinements
    - Monitor quality scores
    - Apply improvements
    """
    try:
        result = await pattern_service.reflection_example(
            initial_output=request.initial_output,
            criteria=request.criteria,
            max_iterations=request.max_iterations
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/planning")
async def demonstrate_planning(request: PlanningRequest) -> Dict[str, Any]:
    """
    Demonstrate planning pattern
    
    Shows how an agent can:
    - Decompose high-level goals into subgoals
    - Create multi-step execution plans
    - Reason about dependencies
    - Define success criteria
    """
    try:
        result = await pattern_service.planning_example(
            goal=request.goal,
            constraints=request.constraints
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tool-use")
async def demonstrate_tool_use(request: ToolUseRequest) -> Dict[str, Any]:
    """
    Demonstrate tool use pattern
    
    Shows how an agent can:
    - Analyze task requirements
    - Select appropriate tools dynamically
    - Reason about tool selection
    - Plan tool usage sequence
    """
    try:
        result = await pattern_service.tool_use_example(
            task=request.task,
            available_tools=request.available_tools
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agentic-rag")
async def demonstrate_agentic_rag(request: AgenticRAGRequest) -> Dict[str, Any]:
    """
    Demonstrate agentic RAG pattern
    
    Shows how an agent can:
    - Retrieve documents based on queries
    - Evaluate document relevance
    - Refine queries iteratively
    - Improve retrieval quality
    """
    try:
        result = await pattern_service.agentic_rag_example(
            initial_query=request.query,
            max_refinements=request.max_refinements
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metacognition")
async def demonstrate_metacognition(request: MetacognitionRequest) -> Dict[str, Any]:
    """
    Demonstrate metacognition pattern
    
    Shows how an agent can:
    - Assess its own performance
    - Detect execution patterns
    - Identify issues proactively
    - Suggest strategy adjustments
    """
    try:
        result = await pattern_service.metacognition_example(
            execution_trace=request.execution_trace,
            performance_metrics=request.performance_metrics
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_patterns() -> Dict[str, Any]:
    """
    List all available agentic design patterns
    
    Returns information about each pattern including:
    - Pattern name and description
    - Core capability demonstrated
    - Evolution level in agent architecture
    - Real-world applications
    """
    return {
        "architecture": {
            "formula": "AI Agent = LLM + System Prompt + Tools + Memory + Reasoning Pattern",
            "components": {
                "LLM": "Brain - Language understanding and generation",
                "System Prompt": "Identity - Behavioral specialization",
                "Tools": "Hands - Ability to take action",
                "Memory": "State - Context retention across sessions",
                "Reasoning Pattern": "Strategy - Structured problem-solving approach"
            }
        },
        "evolution": [
            "1. Basic LLM → Simple text generation",
            "2. Specialized → System prompts shape behavior",
            "3. Tool-Using → Function calling enables action",
            "4. Memory Agent → Persistent state across sessions",
            "5. ReAct Agent → Reasoning + Acting iteratively",
            "6. Metacognitive → Self-monitoring and improvement"
        ],
        "patterns": [
            {
                "name": "Reflection",
                "level": "6. Metacognitive",
                "description": "Self-critique and iterative improvement",
                "capability": "Agents that evaluate and improve their own outputs",
                "use_cases": [
                    "Content generation quality improvement",
                    "Code review and refinement",
                    "Writing assistance with iterative editing"
                ],
                "key_concepts": [
                    "Self-evaluation loops",
                    "Quality scoring",
                    "Iterative refinement",
                    "Convergence to quality threshold"
                ]
            },
            {
                "name": "Planning",
                "level": "5. ReAct Agent",
                "description": "Multi-step goal decomposition",
                "capability": "Break complex goals into executable steps",
                "use_cases": [
                    "Project planning and management",
                    "Research workflow design",
                    "Complex task orchestration"
                ],
                "key_concepts": [
                    "Goal decomposition",
                    "Dependency management",
                    "Success criteria definition",
                    "Resource estimation"
                ]
            },
            {
                "name": "Tool Use",
                "level": "3. Tool-Using",
                "description": "Dynamic capability selection",
                "capability": "Analyze tasks and select appropriate tools",
                "use_cases": [
                    "Multi-tool workflows",
                    "Dynamic capability routing",
                    "Task-specific resource allocation"
                ],
                "key_concepts": [
                    "Requirement analysis",
                    "Tool selection reasoning",
                    "Capability matching",
                    "Execution planning"
                ]
            },
            {
                "name": "Agentic RAG",
                "level": "5. ReAct Agent",
                "description": "Query refinement and relevance evaluation",
                "capability": "Iteratively improve information retrieval",
                "use_cases": [
                    "Intelligent document search",
                    "Research assistance",
                    "Knowledge base exploration"
                ],
                "key_concepts": [
                    "Query refinement loops",
                    "Relevance evaluation",
                    "Corrective retrieval",
                    "Quality-driven iteration"
                ]
            },
            {
                "name": "Metacognition",
                "level": "6. Metacognitive",
                "description": "Self-monitoring and strategy adjustment",
                "capability": "Assess performance and adapt strategies",
                "use_cases": [
                    "Performance optimization",
                    "Error detection and recovery",
                    "Strategy evolution"
                ],
                "key_concepts": [
                    "Performance self-assessment",
                    "Pattern detection",
                    "Issue identification",
                    "Strategy adjustment"
                ]
            }
        ],
        "principles": [
            "1. LLMs are stateless - Context must be managed explicitly",
            "2. System prompts shape behavior - Same model, different roles",
            "3. Function calling enables agency - Tools transform generators into agents",
            "4. Memory is essential - Persistence required for meaningful behavior",
            "5. Reasoning patterns matter - Structured approaches outperform simple prompting",
            "6. Self-reflection improves quality - Iterative refinement beats one-shot generation"
        ]
    }
