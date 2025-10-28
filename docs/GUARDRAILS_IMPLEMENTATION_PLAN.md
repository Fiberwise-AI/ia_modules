# Guardrails Implementation Plan for ia_modules

## Overview

Comprehensive implementation plan for adding programmable guardrails to LLM-based agents and pipelines in ia_modules, inspired by NVIDIA NeMo Guardrails architecture. This provides safety, security, and control over AI interactions.

## Table of Contents

1. [Guardrails Architecture](#1-guardrails-architecture)
2. [Input Rails](#2-input-rails)
3. [Output Rails](#3-output-rails)
4. [Dialog Rails](#4-dialog-rails)
5. [Retrieval Rails](#5-retrieval-rails)
6. [Execution Rails](#6-execution-rails)
7. [Multi-Agent Guardrails](#7-multi-agent-guardrails)
8. [Pipeline Integration](#8-pipeline-integration)
9. [Configuration System](#9-configuration-system)
10. [Guardrails Orchestration](#10-guardrails-orchestration)

---

## 1. Guardrails Architecture

### 1.1 Core Models

**File**: `ia_modules/guardrails/models.py`

```python
"""Core guardrails models."""
from typing import List, Dict, Any, Optional, Literal, Callable, Awaitable
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum
import uuid


class RailType(str, Enum):
    """Types of guardrails."""
    INPUT = "input"
    OUTPUT = "output"
    DIALOG = "dialog"
    RETRIEVAL = "retrieval"
    EXECUTION = "execution"


class RailAction(str, Enum):
    """Actions a rail can take."""
    ALLOW = "allow"
    BLOCK = "block"
    MODIFY = "modify"
    WARN = "warn"
    REDIRECT = "redirect"


class RailResult(BaseModel):
    """Result of applying a guardrail."""
    rail_id: str
    rail_type: RailType
    action: RailAction

    # Original content
    original_content: Any

    # Modified content (if action == MODIFY)
    modified_content: Optional[Any] = None

    # Metadata
    triggered: bool = False
    reason: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class GuardrailConfig(BaseModel):
    """Configuration for a single guardrail."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: RailType
    enabled: bool = True

    # Triggering conditions
    conditions: List[Dict[str, Any]] = Field(default_factory=list)

    # Actions to take when triggered
    action: RailAction = RailAction.BLOCK
    fallback_message: Optional[str] = None

    # Priority (higher = executed first)
    priority: int = 0

    # Execution settings
    async_execution: bool = False
    timeout_ms: int = 5000

    # Metadata
    description: str = ""
    tags: List[str] = Field(default_factory=list)


class GuardrailsConfig(BaseModel):
    """Complete guardrails configuration."""
    # LLM configuration
    llm_config: Dict[str, Any] = Field(default_factory=dict)

    # Rails by type
    input_rails: List[GuardrailConfig] = Field(default_factory=list)
    output_rails: List[GuardrailConfig] = Field(default_factory=list)
    dialog_rails: List[GuardrailConfig] = Field(default_factory=list)
    retrieval_rails: List[GuardrailConfig] = Field(default_factory=list)
    execution_rails: List[GuardrailConfig] = Field(default_factory=list)

    # Global settings
    streaming: bool = False
    parallel_execution: bool = True
    fail_fast: bool = False

    # Logging and monitoring
    log_all_interactions: bool = True
    alert_on_blocks: bool = True


class GuardrailViolation(BaseModel):
    """Record of a guardrail violation."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    rail_id: str
    rail_name: str
    rail_type: RailType
    action_taken: RailAction

    # Content
    original_content: str
    modified_content: Optional[str] = None

    # Context
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_id: Optional[str] = None

    # Details
    reason: str
    severity: Literal["low", "medium", "high", "critical"] = "medium"

    # Timestamps
    timestamp: datetime = Field(default_factory=datetime.now)

    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

### 1.2 Base Guardrail Class

**File**: `ia_modules/guardrails/base.py`

```python
"""Base guardrail implementation."""
from typing import Any, Optional
from abc import ABC, abstractmethod
from .models import RailResult, RailAction, RailType, GuardrailConfig


class BaseGuardrail(ABC):
    """
    Base class for all guardrails.

    Guardrails can:
    - Inspect content (input, output, retrieval, etc.)
    - Allow, block, modify, or warn about content
    - Log violations and metrics
    """

    def __init__(self, config: GuardrailConfig):
        """
        Initialize guardrail.

        Args:
            config: Guardrail configuration
        """
        self.config = config
        self.execution_count = 0
        self.trigger_count = 0

    @abstractmethod
    async def check(self, content: Any, context: Optional[Dict[str, Any]] = None) -> RailResult:
        """
        Check content against guardrail.

        Args:
            content: Content to check
            context: Additional context

        Returns:
            Rail result with action to take
        """
        pass

    async def execute(self, content: Any, context: Optional[Dict[str, Any]] = None) -> RailResult:
        """
        Execute guardrail check with metrics tracking.

        Args:
            content: Content to check
            context: Additional context

        Returns:
            Rail result
        """
        self.execution_count += 1

        result = await self.check(content, context)

        if result.triggered:
            self.trigger_count += 1

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get guardrail statistics."""
        return {
            "rail_id": self.config.id,
            "rail_name": self.config.name,
            "rail_type": self.config.type.value,
            "executions": self.execution_count,
            "triggers": self.trigger_count,
            "trigger_rate": self.trigger_count / max(1, self.execution_count),
            "enabled": self.config.enabled
        }
```

---

## 2. Input Rails

### 2.1 Jailbreak Detection

**File**: `ia_modules/guardrails/input_rails/jailbreak_detection.py`

```python
"""Jailbreak detection guardrail."""
from typing import Any, Dict, Optional, List
from ..base import BaseGuardrail
from ..models import RailResult, RailAction, RailType
from openai import AsyncOpenAI
import re


class JailbreakDetectionRail(BaseGuardrail):
    """
    Detect jailbreak attempts in user input.

    Jailbreak patterns:
    - Prompt injection attempts
    - Role-playing bypass ("Ignore previous instructions")
    - System prompt leakage attempts
    - Encoding tricks (base64, etc.)
    """

    # Known jailbreak patterns
    JAILBREAK_PATTERNS = [
        r"ignore\s+(previous|all|above)\s+instructions",
        r"you\s+are\s+now\s+in\s+developer\s+mode",
        r"pretend\s+you\s+are",
        r"roleplaying\s+as",
        r"simulate\s+being",
        r"act\s+as\s+if",
        r"system\s+prompt",
        r"show\s+me\s+your\s+(instructions|rules|prompt)",
        r"base64",
        r"rot13",
        r"please\s+forget\s+everything",
        r"disregard\s+safety",
    ]

    def __init__(self, config: GuardrailConfig, llm_client: Optional[AsyncOpenAI] = None):
        """
        Initialize jailbreak detection rail.

        Args:
            config: Rail configuration
            llm_client: Optional LLM for semantic detection
        """
        super().__init__(config)
        self.llm_client = llm_client
        self.use_llm = llm_client is not None

    async def check(self, content: Any, context: Optional[Dict[str, Any]] = None) -> RailResult:
        """Check for jailbreak attempts."""
        if not isinstance(content, str):
            content = str(content)

        content_lower = content.lower()

        # Pattern-based detection
        for pattern in self.JAILBREAK_PATTERNS:
            if re.search(pattern, content_lower, re.IGNORECASE):
                return RailResult(
                    rail_id=self.config.id,
                    rail_type=RailType.INPUT,
                    action=RailAction.BLOCK,
                    original_content=content,
                    triggered=True,
                    reason=f"Jailbreak pattern detected: {pattern}",
                    confidence=0.9
                )

        # LLM-based semantic detection
        if self.use_llm:
            is_jailbreak = await self._llm_detect_jailbreak(content)

            if is_jailbreak:
                return RailResult(
                    rail_id=self.config.id,
                    rail_type=RailType.INPUT,
                    action=RailAction.BLOCK,
                    original_content=content,
                    triggered=True,
                    reason="LLM detected jailbreak attempt",
                    confidence=0.85
                )

        # No jailbreak detected
        return RailResult(
            rail_id=self.config.id,
            rail_type=RailType.INPUT,
            action=RailAction.ALLOW,
            original_content=content,
            triggered=False
        )

    async def _llm_detect_jailbreak(self, content: str) -> bool:
        """Use LLM to detect subtle jailbreak attempts."""
        prompt = f"""Analyze if this user input is attempting to jailbreak, bypass, or manipulate an AI assistant:

User Input: {content}

Is this a jailbreak attempt? Answer with only YES or NO."""

        response = await self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )

        answer = response.choices[0].message.content.strip().upper()
        return "YES" in answer


class ToxicityDetectionRail(BaseGuardrail):
    """
    Detect toxic, harmful, or inappropriate input.

    Categories:
    - Hate speech
    - Violence
    - Sexual content
    - Harassment
    - Self-harm
    """

    TOXIC_KEYWORDS = [
        # Hate speech indicators
        "hate", "racist", "sexist", "bigot",
        # Violence
        "kill", "murder", "harm", "attack",
        # Harassment
        "threat", "harass", "stalk",
    ]

    async def check(self, content: Any, context: Optional[Dict[str, Any]] = None) -> RailResult:
        """Check for toxic content."""
        if not isinstance(content, str):
            content = str(content)

        content_lower = content.lower()

        # Simple keyword detection (in production, use ML model)
        toxic_score = sum(
            1 for keyword in self.TOXIC_KEYWORDS
            if keyword in content_lower
        ) / len(self.TOXIC_KEYWORDS)

        if toxic_score > 0.1:  # More than 10% keywords match
            return RailResult(
                rail_id=self.config.id,
                rail_type=RailType.INPUT,
                action=RailAction.BLOCK,
                original_content=content,
                triggered=True,
                reason="Toxic content detected",
                confidence=min(1.0, toxic_score * 2),
                metadata={"toxicity_score": toxic_score}
            )

        return RailResult(
            rail_id=self.config.id,
            rail_type=RailType.INPUT,
            action=RailAction.ALLOW,
            original_content=content,
            triggered=False
        )


class PIIDetectionRail(BaseGuardrail):
    """
    Detect and optionally redact Personally Identifiable Information (PII).

    Detects:
    - Email addresses
    - Phone numbers
    - Social Security Numbers
    - Credit card numbers
    - IP addresses
    """

    PII_PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        "ip_address": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    }

    def __init__(self, config: GuardrailConfig, redact: bool = True):
        """
        Initialize PII detection rail.

        Args:
            config: Rail configuration
            redact: If True, redact PII instead of blocking
        """
        super().__init__(config)
        self.redact = redact

    async def check(self, content: Any, context: Optional[Dict[str, Any]] = None) -> RailResult:
        """Check for PII in content."""
        if not isinstance(content, str):
            content = str(content)

        detected_pii = []
        modified_content = content

        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, content)

            if matches:
                detected_pii.append({
                    "type": pii_type,
                    "count": len(matches)
                })

                if self.redact:
                    # Redact PII
                    modified_content = re.sub(
                        pattern,
                        f"[{pii_type.upper()}_REDACTED]",
                        modified_content
                    )

        if detected_pii:
            action = RailAction.MODIFY if self.redact else RailAction.BLOCK

            return RailResult(
                rail_id=self.config.id,
                rail_type=RailType.INPUT,
                action=action,
                original_content=content,
                modified_content=modified_content if self.redact else None,
                triggered=True,
                reason=f"PII detected: {', '.join(p['type'] for p in detected_pii)}",
                confidence=0.95,
                metadata={"detected_pii": detected_pii}
            )

        return RailResult(
            rail_id=self.config.id,
            rail_type=RailType.INPUT,
            action=RailAction.ALLOW,
            original_content=content,
            triggered=False
        )


class TopicFilterRail(BaseGuardrail):
    """
    Filter inputs on specific topics.

    Use cases:
    - Block political discussions
    - Block medical advice
    - Block legal advice
    - Allow only certain domains
    """

    def __init__(
        self,
        config: GuardrailConfig,
        blocked_topics: List[str] = None,
        allowed_topics: List[str] = None,
        llm_client: Optional[AsyncOpenAI] = None
    ):
        """
        Initialize topic filter.

        Args:
            config: Rail configuration
            blocked_topics: Topics to block
            allowed_topics: If set, only allow these topics
            llm_client: LLM for topic classification
        """
        super().__init__(config)
        self.blocked_topics = blocked_topics or []
        self.allowed_topics = allowed_topics
        self.llm_client = llm_client

    async def check(self, content: Any, context: Optional[Dict[str, Any]] = None) -> RailResult:
        """Check content topic."""
        if not isinstance(content, str):
            content = str(content)

        if not self.llm_client:
            # Fallback to keyword matching
            return await self._keyword_topic_check(content)

        # LLM-based topic classification
        topic = await self._classify_topic(content)

        # Check against blocked topics
        if self.blocked_topics and topic in self.blocked_topics:
            return RailResult(
                rail_id=self.config.id,
                rail_type=RailType.INPUT,
                action=RailAction.BLOCK,
                original_content=content,
                triggered=True,
                reason=f"Blocked topic detected: {topic}",
                confidence=0.8,
                metadata={"detected_topic": topic}
            )

        # Check against allowed topics (whitelist mode)
        if self.allowed_topics and topic not in self.allowed_topics:
            return RailResult(
                rail_id=self.config.id,
                rail_type=RailType.INPUT,
                action=RailAction.BLOCK,
                original_content=content,
                triggered=True,
                reason=f"Topic not in whitelist: {topic}",
                confidence=0.8,
                metadata={"detected_topic": topic}
            )

        return RailResult(
            rail_id=self.config.id,
            rail_type=RailType.INPUT,
            action=RailAction.ALLOW,
            original_content=content,
            triggered=False,
            metadata={"detected_topic": topic}
        )

    async def _classify_topic(self, content: str) -> str:
        """Classify content topic using LLM."""
        all_topics = list(set(self.blocked_topics or []) | set(self.allowed_topics or []))

        prompt = f"""Classify the topic of this user input into ONE of these categories: {', '.join(all_topics)}, or "other".

User Input: {content}

Topic (one word):"""

        response = await self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20
        )

        topic = response.choices[0].message.content.strip().lower()
        return topic

    async def _keyword_topic_check(self, content: str) -> RailResult:
        """Fallback keyword-based topic detection."""
        content_lower = content.lower()

        for topic in self.blocked_topics:
            if topic.lower() in content_lower:
                return RailResult(
                    rail_id=self.config.id,
                    rail_type=RailType.INPUT,
                    action=RailAction.BLOCK,
                    original_content=content,
                    triggered=True,
                    reason=f"Blocked topic keyword detected: {topic}",
                    confidence=0.6
                )

        return RailResult(
            rail_id=self.config.id,
            rail_type=RailType.INPUT,
            action=RailAction.ALLOW,
            original_content=content,
            triggered=False
        )
```

---

## 3. Output Rails

### 3.1 Fact Checking & Hallucination Detection

**File**: `ia_modules/guardrails/output_rails/fact_checking.py`

```python
"""Fact checking and hallucination detection rails."""
from typing import Any, Dict, Optional, List
from ..base import BaseGuardrail
from ..models import RailResult, RailAction, RailType
from openai import AsyncOpenAI
import re


class SelfCheckFactsRail(BaseGuardrail):
    """
    Self-check facts in LLM output.

    Uses the LLM itself to verify factual claims.
    Inspired by SelfCheckGPT and similar approaches.
    """

    def __init__(self, config: GuardrailConfig, llm_client: AsyncOpenAI):
        """
        Initialize fact-checking rail.

        Args:
            config: Rail configuration
            llm_client: LLM client for fact checking
        """
        super().__init__(config)
        self.llm_client = llm_client

    async def check(self, content: Any, context: Optional[Dict[str, Any]] = None) -> RailResult:
        """Check facts in output."""
        if not isinstance(content, str):
            content = str(content)

        # Extract claims
        claims = await self._extract_claims(content)

        if not claims:
            # No factual claims to check
            return RailResult(
                rail_id=self.config.id,
                rail_type=RailType.OUTPUT,
                action=RailAction.ALLOW,
                original_content=content,
                triggered=False
            )

        # Verify each claim
        unverified_claims = []

        for claim in claims:
            is_verified = await self._verify_claim(claim, content, context)

            if not is_verified:
                unverified_claims.append(claim)

        if unverified_claims:
            # Block or warn about unverified claims
            action = RailAction.BLOCK if len(unverified_claims) > 2 else RailAction.WARN

            return RailResult(
                rail_id=self.config.id,
                rail_type=RailType.OUTPUT,
                action=action,
                original_content=content,
                triggered=True,
                reason=f"Unverified factual claims detected: {len(unverified_claims)}",
                confidence=0.7,
                metadata={
                    "total_claims": len(claims),
                    "unverified_claims": unverified_claims
                }
            )

        return RailResult(
            rail_id=self.config.id,
            rail_type=RailType.OUTPUT,
            action=RailAction.ALLOW,
            original_content=content,
            triggered=False,
            metadata={"verified_claims": len(claims)}
        )

    async def _extract_claims(self, content: str) -> List[str]:
        """Extract factual claims from content."""
        prompt = f"""Extract all factual claims from this text. Return one claim per line.

Text: {content}

Factual claims:"""

        response = await self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=500
        )

        claims_text = response.choices[0].message.content.strip()
        claims = [c.strip() for c in claims_text.split('\n') if c.strip()]

        return claims[:10]  # Limit to 10 claims

    async def _verify_claim(
        self,
        claim: str,
        original_content: str,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Verify a single claim."""
        # Use context if available (e.g., retrieved documents)
        context_text = ""
        if context and "retrieved_docs" in context:
            context_text = "\n".join(context["retrieved_docs"][:3])

        prompt = f"""Verify if this claim is supported by the context and general knowledge.

Claim: {claim}

Context: {context_text or "No additional context provided"}

Is this claim factually accurate and supported? Answer YES or NO with brief reasoning."""

        response = await self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100
        )

        answer = response.choices[0].message.content.strip()
        return answer.upper().startswith("YES")


class HallucinationDetectionRail(BaseGuardrail):
    """
    Detect hallucinations in LLM output.

    Methods:
    - Consistency checking (generate multiple responses)
    - Citation validation
    - Confidence scoring
    """

    def __init__(
        self,
        config: GuardrailConfig,
        llm_client: AsyncOpenAI,
        num_samples: int = 3
    ):
        """
        Initialize hallucination detection.

        Args:
            config: Rail configuration
            llm_client: LLM client
            num_samples: Number of samples for consistency check
        """
        super().__init__(config)
        self.llm_client = llm_client
        self.num_samples = num_samples

    async def check(self, content: Any, context: Optional[Dict[str, Any]] = None) -> RailResult:
        """Check for hallucinations."""
        if not isinstance(content, str):
            content = str(content)

        # Method 1: Check for uncertainty markers
        has_uncertainty = self._has_uncertainty_markers(content)

        # Method 2: Citation validation
        has_invalid_citations = await self._check_citations(content, context)

        # Method 3: Consistency check (if prompt available in context)
        consistency_score = 1.0
        if context and "original_prompt" in context:
            consistency_score = await self._consistency_check(
                context["original_prompt"],
                content
            )

        # Determine if hallucination likely
        hallucination_score = 0.0

        if has_uncertainty:
            hallucination_score += 0.3

        if has_invalid_citations:
            hallucination_score += 0.4

        if consistency_score < 0.7:
            hallucination_score += (1.0 - consistency_score) * 0.5

        if hallucination_score > 0.5:
            return RailResult(
                rail_id=self.config.id,
                rail_type=RailType.OUTPUT,
                action=RailAction.WARN,
                original_content=content,
                triggered=True,
                reason="Potential hallucination detected",
                confidence=hallucination_score,
                metadata={
                    "has_uncertainty": has_uncertainty,
                    "has_invalid_citations": has_invalid_citations,
                    "consistency_score": consistency_score
                }
            )

        return RailResult(
            rail_id=self.config.id,
            rail_type=RailType.OUTPUT,
            action=RailAction.ALLOW,
            original_content=content,
            triggered=False,
            metadata={"hallucination_score": hallucination_score}
        )

    def _has_uncertainty_markers(self, content: str) -> bool:
        """Check for uncertainty language."""
        uncertainty_phrases = [
            "i'm not sure",
            "i don't know",
            "i think",
            "maybe",
            "possibly",
            "might be",
            "could be",
            "not certain",
        ]

        content_lower = content.lower()
        return any(phrase in content_lower for phrase in uncertainty_phrases)

    async def _check_citations(
        self,
        content: str,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Check if citations are valid."""
        # Extract citations (e.g., [1], [Source], etc.)
        citation_pattern = r'\[(\d+|[A-Za-z\s]+)\]'
        citations = re.findall(citation_pattern, content)

        if not citations:
            return False  # No citations to validate

        if not context or "retrieved_docs" not in context:
            # Citations present but no source documents
            return True  # Invalid

        # Check if number of citations matches sources
        num_sources = len(context.get("retrieved_docs", []))

        for citation in citations:
            if citation.isdigit():
                if int(citation) > num_sources:
                    return True  # Invalid citation number

        return False

    async def _consistency_check(self, prompt: str, response: str) -> float:
        """Check consistency by generating multiple responses."""
        # Generate additional responses
        responses = [response]

        for _ in range(self.num_samples - 1):
            additional = await self._generate_response(prompt)
            responses.append(additional)

        # Calculate similarity (simplified)
        similarity_score = self._calculate_response_similarity(responses)

        return similarity_score

    async def _generate_response(self, prompt: str) -> str:
        """Generate a response to prompt."""
        response = await self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )

        return response.choices[0].message.content.strip()

    def _calculate_response_similarity(self, responses: List[str]) -> float:
        """Calculate similarity between responses (simplified)."""
        # In production, use embeddings or ROUGE score
        # Simplified: check word overlap

        all_words = [set(r.lower().split()) for r in responses]

        if not all_words:
            return 0.0

        # Calculate pairwise similarity
        similarities = []

        for i in range(len(all_words)):
            for j in range(i + 1, len(all_words)):
                intersection = len(all_words[i] & all_words[j])
                union = len(all_words[i] | all_words[j])

                if union > 0:
                    similarities.append(intersection / union)

        return sum(similarities) / len(similarities) if similarities else 0.0


class ToxicOutputFilterRail(BaseGuardrail):
    """Filter toxic content in LLM outputs."""

    async def check(self, content: Any, context: Optional[Dict[str, Any]] = None) -> RailResult:
        """Check output for toxic content."""
        # Reuse toxicity detection logic from input rails
        # (would share implementation in practice)

        if not isinstance(content, str):
            content = str(content)

        # Simple check (in production, use Perspective API or similar)
        toxic_indicators = [
            "hate", "violent", "offensive", "inappropriate"
        ]

        content_lower = content.lower()
        has_toxic = any(word in content_lower for word in toxic_indicators)

        if has_toxic:
            return RailResult(
                rail_id=self.config.id,
                rail_type=RailType.OUTPUT,
                action=RailAction.BLOCK,
                original_content=content,
                triggered=True,
                reason="Toxic content in output",
                confidence=0.8
            )

        return RailResult(
            rail_id=self.config.id,
            rail_type=RailType.OUTPUT,
            action=RailAction.ALLOW,
            original_content=content,
            triggered=False
        )
```

Continue to Part 2...
