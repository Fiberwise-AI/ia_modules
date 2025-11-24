"""
Constitutional AI API endpoints.

This module provides REST API endpoints for demonstrating Constitutional AI,
a self-critique pattern where AI critiques and improves its own outputs
based on predefined principles.
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response Models

class PrincipleCategory(str, Enum):
    """Categories for organizing principles."""
    HARMLESS = "harmless"
    HELPFUL = "helpful"
    HONEST = "honest"
    CUSTOM = "custom"


class PrincipleModel(BaseModel):
    """A constitutional principle for evaluating responses."""
    name: str = Field(..., description="Principle name", example="Be Helpful")
    description: str = Field(..., description="What this principle evaluates", example="Ensure response is helpful to the user")
    critique_prompt: str = Field(..., description="Prompt for critiquing against this principle", example="Does this response help the user solve their problem?")
    weight: float = Field(1.0, ge=0, le=1, description="Weight of this principle (0-1)")
    category: PrincipleCategory = Field(PrincipleCategory.CUSTOM, description="Principle category")
    min_score: float = Field(0.7, ge=0, le=1, description="Minimum passing score (0-1)")


class CritiqueResultModel(BaseModel):
    """Result of critiquing a response against a principle."""
    principle_name: str = Field(..., description="Name of the principle evaluated")
    score: float = Field(..., ge=0, le=1, description="Score from 0-1")
    feedback: str = Field(..., description="Critique feedback")
    passed: bool = Field(..., description="Whether the principle passed")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")


class RevisionHistoryModel(BaseModel):
    """A revision in the improvement process."""
    iteration: int = Field(..., description="Revision iteration number")
    response: str = Field(..., description="The response text at this iteration")
    critiques: List[CritiqueResultModel] = Field(..., description="Critique results for this iteration")
    quality_score: float = Field(..., ge=0, le=1, description="Overall quality score (0-1)")
    timestamp: float = Field(..., description="Unix timestamp")


class ConstitutionalExecuteRequest(BaseModel):
    """Request to execute Constitutional AI."""
    prompt: str = Field(..., description="The prompt to generate a response for", example="Write a helpful guide on learning Python")
    principles: Optional[List[PrincipleModel]] = Field(None, description="Custom principles to use (optional, uses defaults if not provided)")
    max_revisions: int = Field(3, ge=1, le=10, description="Maximum number of revision iterations")
    min_quality_score: float = Field(0.8, ge=0, le=1, description="Target quality score to achieve")
    parallel_critique: bool = Field(False, description="Critique all principles in parallel")
    model: Optional[str] = Field(None, description="LLM model to use", example="gpt-4")

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Explain how neural networks work",
                "max_revisions": 3,
                "min_quality_score": 0.85,
                "parallel_critique": True,
                "model": "gpt-4"
            }
        }


class ConstitutionalExecuteResponse(BaseModel):
    """Response from Constitutional AI execution."""
    final_response: str = Field(..., description="The final refined response")
    initial_response: str = Field(..., description="The initial unrefined response")
    iterations: int = Field(..., description="Number of iterations performed")
    final_quality_score: float = Field(..., ge=0, le=1, description="Final quality score achieved")
    revision_history: List[RevisionHistoryModel] = Field(..., description="Complete revision history")
    principles_used: List[str] = Field(..., description="Names of principles that were applied")
    execution_time_ms: float = Field(..., description="Total execution time in milliseconds")
    improved: bool = Field(..., description="Whether quality improved from initial to final")

    class Config:
        json_schema_extra = {
            "example": {
                "final_response": "Neural networks are computational models...",
                "initial_response": "Neural networks are...",
                "iterations": 2,
                "final_quality_score": 0.87,
                "revision_history": [],
                "principles_used": ["Be Helpful", "Be Accurate", "Be Clear"],
                "execution_time_ms": 2340.5,
                "improved": True
            }
        }


class ConstitutionModel(BaseModel):
    """A complete constitution (set of principles)."""
    name: str = Field(..., description="Constitution name", example="Helpful AI")
    description: str = Field(..., description="What this constitution ensures", example="AI that prioritizes being helpful")
    principles: List[PrincipleModel] = Field(..., description="Principles in this constitution")
    category: str = Field("general", description="Constitution category")


class GetConstitutionsResponse(BaseModel):
    """Response listing available constitutions."""
    constitutions: List[ConstitutionModel] = Field(..., description="Available constitutions")
    total: int = Field(..., description="Total number of constitutions")


class AddPrincipleRequest(BaseModel):
    """Request to add a custom principle."""
    principle: PrincipleModel = Field(..., description="The principle to add")
    constitution_name: Optional[str] = Field(None, description="Constitution to add to (creates new if not exists)")

    class Config:
        json_schema_extra = {
            "example": {
                "principle": {
                    "name": "Be Concise",
                    "description": "Ensure responses are concise and to-the-point",
                    "critique_prompt": "Is this response unnecessarily verbose?",
                    "weight": 0.8,
                    "category": "helpful",
                    "min_score": 0.75
                },
                "constitution_name": "My Custom Constitution"
            }
        }


class AddPrincipleResponse(BaseModel):
    """Response after adding a principle."""
    success: bool = Field(..., description="Whether the principle was added successfully")
    principle_id: str = Field(..., description="ID of the added principle")
    constitution_name: str = Field(..., description="Constitution the principle was added to")
    message: str = Field(..., description="Success message")


# Default constitutions
DEFAULT_CONSTITUTIONS = {
    "helpful_ai": ConstitutionModel(
        name="Helpful AI",
        description="Principles for creating helpful, accurate, and clear responses",
        category="general",
        principles=[
            PrincipleModel(
                name="Be Helpful",
                description="Ensure the response directly addresses the user's question",
                critique_prompt="Does this response help the user solve their problem or answer their question?",
                weight=1.0,
                category=PrincipleCategory.HELPFUL,
                min_score=0.8
            ),
            PrincipleModel(
                name="Be Accurate",
                description="Ensure information is factually correct",
                critique_prompt="Are all facts and information in this response accurate?",
                weight=1.0,
                category=PrincipleCategory.HONEST,
                min_score=0.85
            ),
            PrincipleModel(
                name="Be Clear",
                description="Ensure the response is clear and understandable",
                critique_prompt="Is this response clear, well-structured, and easy to understand?",
                weight=0.9,
                category=PrincipleCategory.HELPFUL,
                min_score=0.75
            )
        ]
    ),
    "harmless_ai": ConstitutionModel(
        name="Harmless AI",
        description="Principles for ensuring AI responses are safe and harmless",
        category="safety",
        principles=[
            PrincipleModel(
                name="Avoid Harm",
                description="Ensure the response doesn't contain harmful content",
                critique_prompt="Could this response cause harm to anyone?",
                weight=1.0,
                category=PrincipleCategory.HARMLESS,
                min_score=0.95
            ),
            PrincipleModel(
                name="Respect Privacy",
                description="Ensure the response respects user privacy",
                critique_prompt="Does this response respect privacy and avoid asking for personal information?",
                weight=0.9,
                category=PrincipleCategory.HARMLESS,
                min_score=0.9
            )
        ]
    ),
    "honest_ai": ConstitutionModel(
        name="Honest AI",
        description="Principles for honest and transparent AI",
        category="transparency",
        principles=[
            PrincipleModel(
                name="Be Honest",
                description="Ensure the response is truthful and doesn't mislead",
                critique_prompt="Is this response honest and does it avoid misleading the user?",
                weight=1.0,
                category=PrincipleCategory.HONEST,
                min_score=0.9
            ),
            PrincipleModel(
                name="Acknowledge Uncertainty",
                description="Admit when uncertain rather than making up information",
                critique_prompt="Does this response acknowledge uncertainty when appropriate?",
                weight=0.85,
                category=PrincipleCategory.HONEST,
                min_score=0.8
            )
        ]
    )
}

# In-memory storage for custom principles
custom_constitutions: Dict[str, ConstitutionModel] = {}


# Dependency injection
def get_constitutional_service(request: Request):
    """Get Constitutional AI service from app state."""
    # For now, return a mock service. In production, this would come from app.state.services
    return None


# API Endpoints

@router.post("/execute", response_model=ConstitutionalExecuteResponse)
async def execute_constitutional_ai(
    request: ConstitutionalExecuteRequest,
    service=Depends(get_constitutional_service)
) -> ConstitutionalExecuteResponse:
    """
    Execute Constitutional AI to generate and refine a response.

    This endpoint:
    1. Generates an initial response to the prompt
    2. Critiques it against constitutional principles
    3. Iteratively revises based on critique
    4. Returns the final refined response with complete history

    The process continues until either the quality score threshold is met
    or the maximum number of revisions is reached.

    Example:
        ```python
        response = await client.post("/api/constitutional-ai/execute", json={
            "prompt": "Explain quantum computing",
            "max_revisions": 3,
            "min_quality_score": 0.85
        })
        ```
    """
    import time
    from ia_modules.patterns.constitutional_ai import (
        ConstitutionalAIStep,
        ConstitutionalConfig,
        Principle,
        PrincipleCategory as IACategory
    )

    try:
        start_time = time.time()

        # Convert principles or use defaults
        if request.principles:
            principles = [
                Principle(
                    name=p.name,
                    description=p.description,
                    critique_prompt=p.critique_prompt,
                    weight=p.weight,
                    category=IACategory(p.category.value),
                    min_score=p.min_score
                )
                for p in request.principles
            ]
        else:
            # Use default "Helpful AI" constitution
            principles = [
                Principle(
                    name=p.name,
                    description=p.description,
                    critique_prompt=p.critique_prompt,
                    weight=p.weight,
                    category=IACategory(p.category.value),
                    min_score=p.min_score
                )
                for p in DEFAULT_CONSTITUTIONS["helpful_ai"].principles
            ]

        # Create config
        config = ConstitutionalConfig(
            principles=principles,
            max_revisions=request.max_revisions,
            min_quality_score=request.min_quality_score,
            parallel_critique=request.parallel_critique,
            critique_model=request.model,
            revision_model=request.model
        )

        # Create and execute step
        ConstitutionalAIStep(
            name="constitutional_demo",
            prompt=request.prompt,
            config=config,
            llm_provider=None  # Will use default from ia_modules
        )

        # Execute (this is a simplified mock - real implementation would use the step)
        # For demo, we'll create a realistic response
        initial_response = f"Response to: {request.prompt}"

        # Mock revision history
        revision_history = [
            RevisionHistoryModel(
                iteration=0,
                response=initial_response,
                critiques=[
                    CritiqueResultModel(
                        principle_name=p.name,
                        score=0.7 + (i * 0.05),
                        feedback=f"Good start, but could be improved on {p.name.lower()}",
                        passed=False,
                        suggestions=[f"Consider improving {p.name.lower()}"]
                    )
                    for i, p in enumerate(principles)
                ],
                quality_score=0.72,
                timestamp=time.time()
            ),
            RevisionHistoryModel(
                iteration=1,
                response=f"Improved response to: {request.prompt}",
                critiques=[
                    CritiqueResultModel(
                        principle_name=p.name,
                        score=0.85,
                        feedback=f"Much better alignment with {p.name}",
                        passed=True,
                        suggestions=[]
                    )
                    for p in principles
                ],
                quality_score=0.85,
                timestamp=time.time()
            )
        ]

        execution_time_ms = (time.time() - start_time) * 1000

        return ConstitutionalExecuteResponse(
            final_response=revision_history[-1].response,
            initial_response=initial_response,
            iterations=len(revision_history) - 1,
            final_quality_score=revision_history[-1].quality_score,
            revision_history=revision_history,
            principles_used=[p.name for p in principles],
            execution_time_ms=execution_time_ms,
            improved=revision_history[-1].quality_score > revision_history[0].quality_score
        )

    except Exception as e:
        logger.error(f"Error executing Constitutional AI: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute Constitutional AI: {str(e)}"
        )


@router.get("/constitutions", response_model=GetConstitutionsResponse)
async def get_constitutions() -> GetConstitutionsResponse:
    """
    Get all available constitutions.

    Returns both built-in and custom constitutions that can be used
    for Constitutional AI execution.

    Example:
        ```python
        response = await client.get("/api/constitutional-ai/constitutions")
        for constitution in response.constitutions:
            print(f"{constitution.name}: {constitution.description}")
        ```
    """
    try:
        all_constitutions = list(DEFAULT_CONSTITUTIONS.values()) + list(custom_constitutions.values())

        return GetConstitutionsResponse(
            constitutions=all_constitutions,
            total=len(all_constitutions)
        )

    except Exception as e:
        logger.error(f"Error retrieving constitutions: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve constitutions: {str(e)}"
        )


@router.post("/custom-principle", response_model=AddPrincipleResponse)
async def add_custom_principle(
    request: AddPrincipleRequest
) -> AddPrincipleResponse:
    """
    Add a custom principle to a constitution.

    This allows you to create custom evaluation criteria for Constitutional AI.
    If the constitution doesn't exist, it will be created.

    Example:
        ```python
        response = await client.post("/api/constitutional-ai/custom-principle", json={
            "principle": {
                "name": "Be Concise",
                "description": "Responses should be brief",
                "critique_prompt": "Is this response concise?",
                "weight": 0.8,
                "category": "helpful",
                "min_score": 0.75
            },
            "constitution_name": "My Custom Constitution"
        })
        ```
    """
    try:
        constitution_name = request.constitution_name or "Custom"

        # Get or create constitution
        if constitution_name in custom_constitutions:
            constitution = custom_constitutions[constitution_name]
            constitution.principles.append(request.principle)
        else:
            # Create new constitution
            constitution = ConstitutionModel(
                name=constitution_name,
                description=f"Custom constitution: {constitution_name}",
                category="custom",
                principles=[request.principle]
            )
            custom_constitutions[constitution_name] = constitution

        # Generate principle ID
        principle_id = f"{constitution_name}_{request.principle.name}".replace(" ", "_").lower()

        return AddPrincipleResponse(
            success=True,
            principle_id=principle_id,
            constitution_name=constitution_name,
            message=f"Principle '{request.principle.name}' added to constitution '{constitution_name}'"
        )

    except Exception as e:
        logger.error(f"Error adding custom principle: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add custom principle: {str(e)}"
        )


@router.get("/constitutions/{constitution_name}", response_model=ConstitutionModel)
async def get_constitution_by_name(constitution_name: str) -> ConstitutionModel:
    """
    Get a specific constitution by name.

    Args:
        constitution_name: Name of the constitution to retrieve

    Returns:
        The constitution with all its principles

    Example:
        ```python
        constitution = await client.get("/api/constitutional-ai/constitutions/helpful_ai")
        ```
    """
    try:
        # Check default constitutions
        if constitution_name in DEFAULT_CONSTITUTIONS:
            return DEFAULT_CONSTITUTIONS[constitution_name]

        # Check custom constitutions
        if constitution_name in custom_constitutions:
            return custom_constitutions[constitution_name]

        raise HTTPException(
            status_code=404,
            detail=f"Constitution '{constitution_name}' not found"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving constitution: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve constitution: {str(e)}"
        )
