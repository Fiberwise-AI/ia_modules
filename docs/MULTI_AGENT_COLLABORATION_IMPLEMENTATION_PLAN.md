# Multi-Agent Collaboration Implementation Plan

## Overview

This document provides a comprehensive implementation plan for advanced multi-agent collaboration patterns in ia_modules. Multi-agent systems represent the cutting edge of AI development, enabling complex problem-solving through agent specialization, debate, collaboration, and emergent behaviors.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Agent Communication Protocol](#agent-communication-protocol)
3. [Debate Pattern](#debate-pattern)
4. [Delegation & Orchestration](#delegation--orchestration)
5. [Voting & Consensus](#voting--consensus)
6. [Swarm Intelligence](#swarm-intelligence)
7. [Agent Memory & Knowledge Sharing](#agent-memory--knowledge-sharing)
8. [Competitive Evolution](#competitive-evolution)
9. [Human-in-the-Loop Collaboration](#human-in-the-loop-collaboration)
10. [Pipeline Integration](#pipeline-integration)

---

## 1. Architecture Overview

### 1.1 Design Principles

- **Agent Autonomy**: Each agent operates independently with its own goals
- **Communication**: Structured message passing between agents
- **Specialization**: Agents have distinct capabilities and expertise
- **Emergence**: Complex behaviors emerge from simple agent interactions
- **Scalability**: Support for 2-1000+ agents in a system
- **Observability**: Full tracing of agent interactions and decisions

### 1.2 Component Architecture

```
ia_modules/
├── agents/
│   ├── __init__.py
│   ├── models.py              # Agent data models
│   ├── base.py                # Base agent class
│   ├── specialized/
│   │   ├── __init__.py
│   │   ├── researcher.py      # Research specialist
│   │   ├── analyst.py         # Data analyst
│   │   ├── critic.py          # Critical evaluator
│   │   ├── synthesizer.py     # Information synthesizer
│   │   └── executor.py        # Action executor
│   ├── collaboration/
│   │   ├── __init__.py
│   │   ├── debate.py          # Agent debate framework
│   │   ├── delegation.py      # Delegation patterns
│   │   ├── voting.py          # Voting mechanisms
│   │   └── swarm.py           # Swarm intelligence
│   ├── communication/
│   │   ├── __init__.py
│   │   ├── protocol.py        # Communication protocol
│   │   ├── message_bus.py     # Message routing
│   │   └── knowledge_base.py  # Shared knowledge
│   └── evolution/
│       ├── __init__.py
│       ├── genetic.py         # Genetic algorithms
│       └── tournament.py      # Competitive selection
└── tests/
    └── integration/
        └── test_multi_agent.py
```

---

## 2. Agent Communication Protocol

### 2.1 Message Models

**File**: `ia_modules/agents/models.py`

```python
"""Data models for multi-agent systems."""
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum
import uuid


class MessageType(str, Enum):
    """Types of agent messages."""
    PROPOSAL = "proposal"           # Agent proposes a solution
    CRITIQUE = "critique"           # Agent critiques another's work
    QUESTION = "question"           # Agent asks for information
    ANSWER = "answer"               # Agent provides information
    VOTE = "vote"                   # Agent casts a vote
    CONSENSUS = "consensus"         # Consensus announcement
    DELEGATION = "delegation"       # Task delegation
    RESULT = "result"               # Task result
    OBSERVATION = "observation"     # Agent shares observation


class AgentMessage(BaseModel):
    """Message exchanged between agents."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType
    sender_id: str = Field(..., description="Agent sending message")
    recipient_id: Optional[str] = Field(None, description="Specific recipient or None for broadcast")
    content: str = Field(..., description="Message content")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    in_reply_to: Optional[str] = Field(None, description="ID of message being replied to")

    class Config:
        json_schema_extra = {
            "example": {
                "type": "proposal",
                "sender_id": "agent_researcher_1",
                "recipient_id": None,
                "content": "Based on my analysis, I propose we use approach A because...",
                "metadata": {"confidence": 0.85, "reasoning_steps": 5}
            }
        }


class AgentCapability(str, Enum):
    """Agent capabilities/specializations."""
    RESEARCH = "research"           # Information gathering
    ANALYSIS = "analysis"           # Data analysis
    CRITIQUE = "critique"           # Critical evaluation
    SYNTHESIS = "synthesis"         # Information synthesis
    EXECUTION = "execution"         # Action execution
    PLANNING = "planning"           # Strategic planning
    CREATIVITY = "creativity"       # Creative ideation
    VERIFICATION = "verification"   # Fact checking


class AgentProfile(BaseModel):
    """Agent profile and characteristics."""
    id: str = Field(..., description="Unique agent identifier")
    name: str = Field(..., description="Agent name")
    role: str = Field(..., description="Agent role/specialty")
    capabilities: List[AgentCapability] = Field(default_factory=list)
    personality: Dict[str, float] = Field(
        default_factory=dict,
        description="Personality traits (0-1 scale)"
    )
    model: str = Field("gpt-4o", description="LLM model to use")
    temperature: float = Field(0.7, ge=0, le=2)
    system_prompt: str = Field(..., description="Agent's system prompt")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "agent_critic_1",
                "name": "Critical Analyst",
                "role": "critic",
                "capabilities": ["critique", "analysis"],
                "personality": {
                    "skepticism": 0.9,
                    "thoroughness": 0.85,
                    "creativity": 0.3
                },
                "model": "gpt-4o",
                "temperature": 0.3,
                "system_prompt": "You are a critical analyst. Your job is to find flaws and potential issues in proposed solutions."
            }
        }


class CollaborationSession(BaseModel):
    """Multi-agent collaboration session."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    pattern: Literal["debate", "delegation", "voting", "swarm"] = Field(...)
    agents: List[AgentProfile]
    goal: str = Field(..., description="Collaboration objective")
    context: Dict[str, Any] = Field(default_factory=dict)
    messages: List[AgentMessage] = Field(default_factory=list)
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: Literal["active", "completed", "failed"] = "active"
    result: Optional[Dict[str, Any]] = None


class DebateRound(BaseModel):
    """Single round in agent debate."""
    round_number: int
    proposals: List[AgentMessage] = Field(default_factory=list)
    critiques: List[AgentMessage] = Field(default_factory=list)
    rebuttals: List[AgentMessage] = Field(default_factory=list)
    votes: List[AgentMessage] = Field(default_factory=list)
    winner: Optional[str] = None


class VoteResult(BaseModel):
    """Voting results."""
    option_id: str
    votes: int
    vote_weight: float = Field(0.0, description="Weighted vote score")
    voters: List[str] = Field(default_factory=list)
    reasoning: List[str] = Field(default_factory=list)
```

### 2.2 Message Bus

**File**: `ia_modules/agents/communication/message_bus.py`

```python
"""Message bus for agent communication."""
from typing import List, Dict, Callable, Awaitable, Optional
import asyncio
from collections import defaultdict
from ..models import AgentMessage, MessageType


class MessageBus:
    """Asynchronous message bus for agent communication."""

    def __init__(self):
        """Initialize message bus."""
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._message_history: List[AgentMessage] = []
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._running = False

    def subscribe(
        self,
        agent_id: str,
        callback: Callable[[AgentMessage], Awaitable[None]]
    ) -> None:
        """
        Subscribe agent to messages.

        Args:
            agent_id: Agent identifier
            callback: Async callback to handle messages
        """
        self._subscribers[agent_id].append(callback)

    def unsubscribe(self, agent_id: str) -> None:
        """Unsubscribe agent from messages."""
        if agent_id in self._subscribers:
            del self._subscribers[agent_id]

    async def publish(self, message: AgentMessage) -> None:
        """
        Publish message to bus.

        Args:
            message: Message to publish
        """
        # Store in history
        self._message_history.append(message)

        # Add to queue
        await self._message_queue.put(message)

    async def _process_messages(self) -> None:
        """Process messages from queue (background task)."""
        while self._running:
            try:
                # Get message from queue
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0
                )

                # Deliver to recipients
                await self._deliver_message(message)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error processing message: {e}")

    async def _deliver_message(self, message: AgentMessage) -> None:
        """Deliver message to appropriate recipients."""
        if message.recipient_id:
            # Direct message to specific agent
            if message.recipient_id in self._subscribers:
                callbacks = self._subscribers[message.recipient_id]
                await asyncio.gather(
                    *[cb(message) for cb in callbacks],
                    return_exceptions=True
                )
        else:
            # Broadcast to all agents except sender
            for agent_id, callbacks in self._subscribers.items():
                if agent_id != message.sender_id:
                    await asyncio.gather(
                        *[cb(message) for cb in callbacks],
                        return_exceptions=True
                    )

    async def start(self) -> None:
        """Start message bus processing."""
        self._running = True
        asyncio.create_task(self._process_messages())

    async def stop(self) -> None:
        """Stop message bus processing."""
        self._running = False

    def get_history(
        self,
        agent_id: Optional[str] = None,
        message_type: Optional[MessageType] = None,
        limit: Optional[int] = None
    ) -> List[AgentMessage]:
        """
        Get message history with optional filtering.

        Args:
            agent_id: Filter by sender or recipient
            message_type: Filter by message type
            limit: Max messages to return

        Returns:
            List of messages
        """
        messages = self._message_history

        if agent_id:
            messages = [
                m for m in messages
                if m.sender_id == agent_id or m.recipient_id == agent_id
            ]

        if message_type:
            messages = [m for m in messages if m.type == message_type]

        if limit:
            messages = messages[-limit:]

        return messages
```

---

## 3. Debate Pattern

### 3.1 Agent Debate Framework

**File**: `ia_modules/agents/collaboration/debate.py`

```python
"""Agent debate framework for multi-perspective problem solving."""
from typing import List, Optional, Dict, Any
import asyncio
from openai import AsyncOpenAI
from ..models import (
    AgentProfile,
    AgentMessage,
    MessageType,
    CollaborationSession,
    DebateRound
)
from ..communication.message_bus import MessageBus


class AgentDebate:
    """
    Multi-agent debate system.

    Agents propose solutions, critique each other, and vote on the best approach.
    This pattern is excellent for:
    - Complex decision making
    - Exploring multiple perspectives
    - Reducing bias through diversity
    - Finding robust solutions
    """

    def __init__(
        self,
        agents: List[AgentProfile],
        llm_client: AsyncOpenAI,
        max_rounds: int = 3,
        consensus_threshold: float = 0.7
    ):
        """
        Initialize agent debate.

        Args:
            agents: List of participating agents
            llm_client: OpenAI client for LLM calls
            max_rounds: Maximum debate rounds
            consensus_threshold: Vote threshold for consensus
        """
        self.agents = agents
        self.llm_client = llm_client
        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold
        self.message_bus = MessageBus()
        self.session: Optional[CollaborationSession] = None

    async def run_debate(
        self,
        topic: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run multi-round debate.

        Args:
            topic: Debate topic/question
            context: Additional context for agents

        Returns:
            Debate results with winning proposal
        """
        # Initialize session
        self.session = CollaborationSession(
            name=f"Debate: {topic}",
            pattern="debate",
            agents=self.agents,
            goal=topic,
            context=context
        )

        # Start message bus
        await self.message_bus.start()

        # Subscribe agents to message bus
        for agent in self.agents:
            await self.message_bus.subscribe(
                agent.id,
                lambda msg, a=agent: self._handle_message(a, msg)
            )

        rounds: List[DebateRound] = []

        # Run debate rounds
        for round_num in range(1, self.max_rounds + 1):
            print(f"\n=== Round {round_num}/{self.max_rounds} ===\n")

            debate_round = DebateRound(round_number=round_num)

            # Phase 1: Proposals
            print(f"Phase 1: Agents propose solutions...")
            proposals = await self._proposal_phase(topic, context, rounds)
            debate_round.proposals = proposals

            # Phase 2: Critiques
            print(f"Phase 2: Agents critique proposals...")
            critiques = await self._critique_phase(proposals)
            debate_round.critiques = critiques

            # Phase 3: Rebuttals (optional)
            if round_num < self.max_rounds:
                print(f"Phase 3: Agents defend their proposals...")
                rebuttals = await self._rebuttal_phase(proposals, critiques)
                debate_round.rebuttals = rebuttals

            # Phase 4: Voting
            print(f"Phase 4: Agents vote on best proposal...")
            votes = await self._voting_phase(proposals, critiques)
            debate_round.votes = votes

            # Check for consensus
            winner = self._check_consensus(votes, proposals)
            if winner:
                debate_round.winner = winner
                rounds.append(debate_round)
                print(f"\n✅ Consensus reached! Winner: {winner}\n")
                break

            rounds.append(debate_round)

        # Stop message bus
        await self.message_bus.stop()

        # Compile results
        final_winner = rounds[-1].winner if rounds else None
        winning_proposal = None

        if final_winner:
            for proposal in rounds[-1].proposals:
                if proposal.sender_id == final_winner:
                    winning_proposal = proposal
                    break

        result = {
            "topic": topic,
            "rounds": len(rounds),
            "winner": final_winner,
            "winning_proposal": winning_proposal.content if winning_proposal else None,
            "all_proposals": [p.content for r in rounds for p in r.proposals],
            "debate_history": rounds
        }

        self.session.result = result
        self.session.status = "completed"

        return result

    async def _proposal_phase(
        self,
        topic: str,
        context: Dict[str, Any],
        previous_rounds: List[DebateRound]
    ) -> List[AgentMessage]:
        """Each agent proposes a solution."""
        proposals = []

        # Build context from previous rounds
        history_context = ""
        if previous_rounds:
            history_context = "\n\nPrevious discussion:\n"
            for round in previous_rounds:
                history_context += f"\nRound {round.round_number}:\n"
                for prop in round.proposals:
                    history_context += f"- {prop.sender_id}: {prop.content[:200]}...\n"

        # Get proposals from each agent concurrently
        tasks = []
        for agent in self.agents:
            task = self._get_agent_proposal(agent, topic, context, history_context)
            tasks.append(task)

        proposals = await asyncio.gather(*tasks)
        return proposals

    async def _get_agent_proposal(
        self,
        agent: AgentProfile,
        topic: str,
        context: Dict[str, Any],
        history: str
    ) -> AgentMessage:
        """Get proposal from single agent."""
        prompt = f"""Topic: {topic}

Context: {context}

{history}

As {agent.role}, propose your best solution or approach. Explain your reasoning clearly and concisely."""

        response = await self.llm_client.chat.completions.create(
            model=agent.model,
            messages=[
                {"role": "system", "content": agent.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=agent.temperature
        )

        proposal_content = response.choices[0].message.content

        return AgentMessage(
            type=MessageType.PROPOSAL,
            sender_id=agent.id,
            content=proposal_content,
            metadata={
                "round": len(self.message_bus.get_history()) + 1,
                "role": agent.role
            }
        )

    async def _critique_phase(
        self,
        proposals: List[AgentMessage]
    ) -> List[AgentMessage]:
        """Each agent critiques others' proposals."""
        critiques = []

        for agent in self.agents:
            # Agent doesn't critique their own proposal
            other_proposals = [p for p in proposals if p.sender_id != agent.id]

            if not other_proposals:
                continue

            # Build critique prompt
            proposals_text = "\n\n".join([
                f"Proposal by {p.sender_id}:\n{p.content}"
                for p in other_proposals
            ])

            prompt = f"""Review these proposals and provide constructive criticism:

{proposals_text}

Identify strengths and weaknesses. Be specific and fair."""

            response = await self.llm_client.chat.completions.create(
                model=agent.model,
                messages=[
                    {"role": "system", "content": agent.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=agent.temperature
            )

            critique = AgentMessage(
                type=MessageType.CRITIQUE,
                sender_id=agent.id,
                content=response.choices[0].message.content,
                metadata={"critiquing": [p.sender_id for p in other_proposals]}
            )

            critiques.append(critique)

        return critiques

    async def _rebuttal_phase(
        self,
        proposals: List[AgentMessage],
        critiques: List[AgentMessage]
    ) -> List[AgentMessage]:
        """Agents defend their proposals against critiques."""
        rebuttals = []

        for proposal in proposals:
            # Find critiques of this proposal
            relevant_critiques = [
                c for c in critiques
                if proposal.sender_id in c.metadata.get("critiquing", [])
            ]

            if not relevant_critiques:
                continue

            # Get agent profile
            agent = next(a for a in self.agents if a.id == proposal.sender_id)

            # Build rebuttal prompt
            critiques_text = "\n\n".join([
                f"Critique by {c.sender_id}:\n{c.content}"
                for c in relevant_critiques
            ])

            prompt = f"""Your proposal:\n{proposal.content}

Critiques received:\n{critiques_text}

Defend your proposal and address the critiques. You may revise your approach if needed."""

            response = await self.llm_client.chat.completions.create(
                model=agent.model,
                messages=[
                    {"role": "system", "content": agent.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=agent.temperature
            )

            rebuttal = AgentMessage(
                type=MessageType.ANSWER,
                sender_id=agent.id,
                content=response.choices[0].message.content,
                in_reply_to=proposal.id
            )

            rebuttals.append(rebuttal)

        return rebuttals

    async def _voting_phase(
        self,
        proposals: List[AgentMessage],
        critiques: List[AgentMessage]
    ) -> List[AgentMessage]:
        """Each agent votes on best proposal."""
        votes = []

        for agent in self.agents:
            # Can't vote for own proposal
            other_proposals = [p for p in proposals if p.sender_id != agent.id]

            if not other_proposals:
                continue

            # Build voting prompt
            proposals_text = "\n\n".join([
                f"{i+1}. Proposal by {p.sender_id}:\n{p.content}"
                for i, p in enumerate(other_proposals)
            ])

            prompt = f"""Vote for the best proposal:

{proposals_text}

Respond with ONLY the agent ID of your choice and a brief reason."""

            response = await self.llm_client.chat.completions.create(
                model=agent.model,
                messages=[
                    {"role": "system", "content": agent.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3  # Lower temperature for voting
            )

            vote_content = response.choices[0].message.content

            # Parse vote (extract agent ID)
            voted_for = None
            for proposal in other_proposals:
                if proposal.sender_id in vote_content:
                    voted_for = proposal.sender_id
                    break

            vote = AgentMessage(
                type=MessageType.VOTE,
                sender_id=agent.id,
                content=vote_content,
                metadata={"voted_for": voted_for}
            )

            votes.append(vote)

        return votes

    def _check_consensus(
        self,
        votes: List[AgentMessage],
        proposals: List[AgentMessage]
    ) -> Optional[str]:
        """Check if consensus reached."""
        if not votes:
            return None

        # Count votes
        vote_counts: Dict[str, int] = {}
        for vote in votes:
            voted_for = vote.metadata.get("voted_for")
            if voted_for:
                vote_counts[voted_for] = vote_counts.get(voted_for, 0) + 1

        if not vote_counts:
            return None

        # Find winner
        max_votes = max(vote_counts.values())
        total_votes = len(votes)

        # Check if threshold met
        if max_votes / total_votes >= self.consensus_threshold:
            return max(vote_counts, key=vote_counts.get)

        return None

    async def _handle_message(
        self,
        agent: AgentProfile,
        message: AgentMessage
    ) -> None:
        """Handle incoming message for agent."""
        # Agents can react to messages in real-time
        # (e.g., ask clarifying questions)
        pass
```

### 3.2 Debate Usage Example

```python
"""Example: Running an agent debate."""
import asyncio
from openai import AsyncOpenAI
from ia_modules.agents.collaboration.debate import AgentDebate
from ia_modules.agents.models import AgentProfile, AgentCapability


async def main():
    # Create diverse agent team
    agents = [
        AgentProfile(
            id="agent_optimist",
            name="Optimistic Innovator",
            role="innovator",
            capabilities=[AgentCapability.CREATIVITY, AgentCapability.PLANNING],
            personality={"optimism": 0.9, "risk_tolerance": 0.8},
            system_prompt="You are an optimistic innovator. You see opportunities and focus on potential benefits."
        ),
        AgentProfile(
            id="agent_skeptic",
            name="Critical Skeptic",
            role="skeptic",
            capabilities=[AgentCapability.CRITIQUE, AgentCapability.ANALYSIS],
            personality={"skepticism": 0.9, "risk_tolerance": 0.2},
            system_prompt="You are a critical skeptic. You identify risks and potential problems."
        ),
        AgentProfile(
            id="agent_pragmatist",
            name="Practical Pragmatist",
            role="pragmatist",
            capabilities=[AgentCapability.EXECUTION, AgentCapability.PLANNING],
            personality={"practicality": 0.9, "detail_orientation": 0.8},
            system_prompt="You are a practical pragmatist. You focus on feasibility and implementation details."
        ),
        AgentProfile(
            id="agent_analyst",
            name="Data Analyst",
            role="analyst",
            capabilities=[AgentCapability.ANALYSIS, AgentCapability.VERIFICATION],
            personality={"analytical": 0.9, "thoroughness": 0.9},
            system_prompt="You are a data analyst. You make decisions based on data and evidence."
        )
    ]

    # Initialize debate
    debate = AgentDebate(
        agents=agents,
        llm_client=AsyncOpenAI(api_key="sk-..."),
        max_rounds=3,
        consensus_threshold=0.6
    )

    # Run debate
    result = await debate.run_debate(
        topic="Should we adopt a microservices architecture for our AI pipeline system?",
        context={
            "current_state": "Monolithic application",
            "team_size": 15,
            "expected_growth": "3x in next year",
            "budget": "moderate"
        }
    )

    # Display results
    print(f"\n{'='*60}")
    print(f"DEBATE RESULTS")
    print(f"{'='*60}\n")
    print(f"Topic: {result['topic']}")
    print(f"Rounds: {result['rounds']}")
    print(f"Winner: {result['winner']}")
    print(f"\nWinning Proposal:\n{result['winning_proposal']}")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 4. Delegation & Orchestration

### 4.1 Master-Worker Pattern

**File**: `ia_modules/agents/collaboration/delegation.py`

```python
"""Agent delegation and orchestration patterns."""
from typing import List, Dict, Any, Optional
import asyncio
from openai import AsyncOpenAI
from ..models import AgentProfile, AgentMessage, MessageType, AgentCapability


class AgentOrchestrator:
    """
    Master agent that delegates tasks to specialist agents.

    Use cases:
    - Complex projects requiring multiple skills
    - Parallel task execution
    - Resource optimization
    """

    def __init__(
        self,
        master_agent: AgentProfile,
        specialist_agents: List[AgentProfile],
        llm_client: AsyncOpenAI
    ):
        """
        Initialize orchestrator.

        Args:
            master_agent: Orchestrator/coordinator agent
            specialist_agents: Specialist worker agents
            llm_client: OpenAI client
        """
        self.master = master_agent
        self.specialists = specialist_agents
        self.llm_client = llm_client

    async def orchestrate(
        self,
        task: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Orchestrate task execution across specialist agents.

        Args:
            task: Main task description
            context: Task context

        Returns:
            Orchestration results
        """
        # Phase 1: Master breaks down task
        print("Phase 1: Task decomposition...")
        subtasks = await self._decompose_task(task, context)

        # Phase 2: Assign subtasks to specialists
        print("Phase 2: Task assignment...")
        assignments = await self._assign_tasks(subtasks)

        # Phase 3: Execute subtasks in parallel
        print("Phase 3: Parallel execution...")
        results = await self._execute_subtasks(assignments)

        # Phase 4: Master synthesizes results
        print("Phase 4: Result synthesis...")
        final_result = await self._synthesize_results(task, results)

        return {
            "task": task,
            "subtasks": subtasks,
            "assignments": assignments,
            "subtask_results": results,
            "final_result": final_result
        }

    async def _decompose_task(
        self,
        task: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Master agent decomposes task into subtasks."""
        # List available specialist capabilities
        capabilities_text = "\n".join([
            f"- {agent.name} ({agent.role}): {', '.join(agent.capabilities)}"
            for agent in self.specialists
        ])

        prompt = f"""Task: {task}

Context: {context}

Available specialists:
{capabilities_text}

Break this task down into concrete subtasks that can be assigned to specialists. For each subtask, specify:
1. Subtask description
2. Required capability
3. Dependencies (which other subtasks must complete first)
4. Priority (high/medium/low)

Format as JSON array."""

        response = await self.llm_client.chat.completions.create(
            model=self.master.model,
            messages=[
                {"role": "system", "content": self.master.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        # Parse subtasks (simplified - would use JSON in production)
        import json
        try:
            subtasks = json.loads(response.choices[0].message.content)
        except:
            # Fallback parsing
            subtasks = [
                {
                    "description": task,
                    "capability": "analysis",
                    "dependencies": [],
                    "priority": "high"
                }
            ]

        return subtasks

    async def _assign_tasks(
        self,
        subtasks: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Assign subtasks to specialist agents."""
        assignments: Dict[str, List[Dict[str, Any]]] = {
            agent.id: [] for agent in self.specialists
        }

        for subtask in subtasks:
            required_cap = subtask.get("capability", "analysis")

            # Find specialists with required capability
            candidates = [
                agent for agent in self.specialists
                if required_cap in [c.value for c in agent.capabilities]
            ]

            if candidates:
                # Assign to least busy specialist
                assigned_agent = min(
                    candidates,
                    key=lambda a: len(assignments[a.id])
                )
                assignments[assigned_agent.id].append(subtask)
            else:
                # Assign to first specialist if no match
                assignments[self.specialists[0].id].append(subtask)

        return assignments

    async def _execute_subtasks(
        self,
        assignments: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[str]]:
        """Execute subtasks in parallel."""
        results: Dict[str, List[str]] = {}

        # Create tasks for each agent
        tasks = []
        for agent_id, subtasks in assignments.items():
            if subtasks:
                agent = next(a for a in self.specialists if a.id == agent_id)
                task = self._execute_agent_tasks(agent, subtasks)
                tasks.append((agent_id, task))

        # Execute in parallel
        completed = await asyncio.gather(
            *[task for _, task in tasks],
            return_exceptions=True
        )

        # Collect results
        for (agent_id, _), result in zip(tasks, completed):
            results[agent_id] = result if not isinstance(result, Exception) else []

        return results

    async def _execute_agent_tasks(
        self,
        agent: AgentProfile,
        subtasks: List[Dict[str, Any]]
    ) -> List[str]:
        """Execute all subtasks for a single agent."""
        results = []

        for subtask in subtasks:
            prompt = f"""Subtask: {subtask['description']}

Priority: {subtask.get('priority', 'medium')}

Complete this subtask to the best of your ability. Provide a detailed result."""

            response = await self.llm_client.chat.completions.create(
                model=agent.model,
                messages=[
                    {"role": "system", "content": agent.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=agent.temperature
            )

            results.append(response.choices[0].message.content)

        return results

    async def _synthesize_results(
        self,
        original_task: str,
        subtask_results: Dict[str, List[str]]
    ) -> str:
        """Master agent synthesizes all results."""
        # Compile all results
        results_text = ""
        for agent_id, results in subtask_results.items():
            agent = next(a for a in self.specialists if a.id == agent_id)
            results_text += f"\n\n{agent.name} results:\n"
            for i, result in enumerate(results, 1):
                results_text += f"{i}. {result}\n"

        prompt = f"""Original task: {original_task}

Specialist results:
{results_text}

Synthesize these results into a coherent, comprehensive final answer."""

        response = await self.llm_client.chat.completions.create(
            model=self.master.model,
            messages=[
                {"role": "system", "content": self.master.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        return response.choices[0].message.content
```

---

## 5. Voting & Consensus

### 5.1 Voting Mechanisms

**File**: `ia_modules/agents/collaboration/voting.py`

```python
"""Voting and consensus mechanisms for multi-agent systems."""
from typing import List, Dict, Any, Optional, Literal
from collections import Counter
from ..models import AgentProfile, AgentMessage, VoteResult


class VotingSystem:
    """Advanced voting mechanisms for agent consensus."""

    @staticmethod
    def simple_majority(votes: List[AgentMessage]) -> Optional[str]:
        """
        Simple majority voting.

        Returns:
            Winning option ID or None if tie
        """
        vote_counts = Counter(
            vote.metadata.get("voted_for")
            for vote in votes
            if vote.metadata.get("voted_for")
        )

        if not vote_counts:
            return None

        # Check for tie
        max_votes = max(vote_counts.values())
        winners = [k for k, v in vote_counts.items() if v == max_votes]

        return winners[0] if len(winners) == 1 else None

    @staticmethod
    def weighted_voting(
        votes: List[AgentMessage],
        agent_weights: Dict[str, float]
    ) -> Optional[str]:
        """
        Weighted voting based on agent expertise.

        Args:
            votes: Agent votes
            agent_weights: Weight for each agent (0-1)

        Returns:
            Winning option ID
        """
        weighted_scores: Dict[str, float] = {}

        for vote in votes:
            voted_for = vote.metadata.get("voted_for")
            if voted_for:
                weight = agent_weights.get(vote.sender_id, 1.0)
                weighted_scores[voted_for] = weighted_scores.get(voted_for, 0) + weight

        if not weighted_scores:
            return None

        return max(weighted_scores, key=weighted_scores.get)

    @staticmethod
    def ranked_choice(
        votes: List[AgentMessage],
        options: List[str]
    ) -> Optional[str]:
        """
        Ranked choice voting (instant runoff).

        Args:
            votes: Votes with ranked preferences in metadata
            options: All possible options

        Returns:
            Winning option after runoffs
        """
        # Implementation of instant runoff voting
        # Each agent ranks all options
        # Eliminate lowest-ranked until one has majority

        # Simplified implementation
        # (Full IRV would require multiple rounds)

        first_choice_counts = Counter()
        for vote in votes:
            rankings = vote.metadata.get("rankings", [])
            if rankings:
                first_choice_counts[rankings[0]] += 1

        total_votes = len(votes)
        for option, count in first_choice_counts.items():
            if count > total_votes / 2:
                return option

        # If no majority, would run elimination rounds
        # For simplicity, return plurality winner
        return first_choice_counts.most_common(1)[0][0] if first_choice_counts else None

    @staticmethod
    def quadratic_voting(
        votes: List[AgentMessage],
        vote_credits: Dict[str, int]
    ) -> VoteResult:
        """
        Quadratic voting - agents allocate credits to express preference strength.

        Args:
            votes: Votes with credit allocations
            vote_credits: Credits available per agent

        Returns:
            Vote result with scores
        """
        # Quadratic voting: cost = votes^2
        # Allows expressing intensity of preference

        option_scores: Dict[str, float] = {}
        option_voters: Dict[str, List[str]] = {}

        for vote in votes:
            voted_for = vote.metadata.get("voted_for")
            credits_spent = vote.metadata.get("credits", 1)

            if voted_for:
                # Quadratic cost
                actual_votes = credits_spent ** 0.5
                option_scores[voted_for] = option_scores.get(voted_for, 0) + actual_votes

                if voted_for not in option_voters:
                    option_voters[voted_for] = []
                option_voters[voted_for].append(vote.sender_id)

        if not option_scores:
            return None

        winner = max(option_scores, key=option_scores.get)

        return VoteResult(
            option_id=winner,
            votes=len(option_voters.get(winner, [])),
            vote_weight=option_scores[winner],
            voters=option_voters.get(winner, [])
        )


class ConsensusBuilder:
    """Build consensus through iterative refinement."""

    @staticmethod
    async def find_common_ground(
        proposals: List[str],
        synthesizer_agent: AgentProfile,
        llm_client: Any
    ) -> str:
        """
        Find common ground between proposals.

        Args:
            proposals: List of different proposals
            synthesizer_agent: Agent to synthesize common ground
            llm_client: LLM client

        Returns:
            Synthesized consensus proposal
        """
        proposals_text = "\n\n".join([
            f"Proposal {i+1}:\n{p}"
            for i, p in enumerate(proposals)
        ])

        prompt = f"""Find the common ground between these proposals:

{proposals_text}

Identify shared themes and create a unified proposal that incorporates the best ideas from all perspectives."""

        response = await llm_client.chat.completions.create(
            model=synthesizer_agent.model,
            messages=[
                {"role": "system", "content": synthesizer_agent.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        return response.choices[0].message.content
```

---

## 6. Swarm Intelligence

### 6.1 Particle Swarm Optimization

**File**: `ia_modules/agents/collaboration/swarm.py`

```python
"""Swarm intelligence patterns for distributed problem solving."""
from typing import List, Dict, Any, Callable, Optional
import asyncio
import random
from ..models import AgentProfile


class AgentSwarm:
    """
    Swarm intelligence for distributed optimization.

    Inspired by particle swarm optimization and ant colony optimization.
    Agents explore solution space and share discoveries with the swarm.
    """

    def __init__(
        self,
        num_agents: int,
        solution_space: Dict[str, Any],
        fitness_function: Callable[[Dict[str, Any]], float],
        llm_client: Any
    ):
        """
        Initialize agent swarm.

        Args:
            num_agents: Number of agents in swarm
            solution_space: Definition of solution space
            fitness_function: Function to evaluate solution quality
            llm_client: LLM client
        """
        self.num_agents = num_agents
        self.solution_space = solution_space
        self.fitness_function = fitness_function
        self.llm_client = llm_client

        # Swarm state
        self.global_best_solution: Optional[Dict[str, Any]] = None
        self.global_best_fitness: float = float('-inf')

        # Agent positions and velocities (in solution space)
        self.agents: List[Dict[str, Any]] = []

    async def optimize(
        self,
        max_iterations: int = 50,
        convergence_threshold: float = 0.001
    ) -> Dict[str, Any]:
        """
        Run swarm optimization.

        Args:
            max_iterations: Maximum iterations
            convergence_threshold: Convergence threshold

        Returns:
            Best solution found
        """
        # Initialize swarm
        await self._initialize_swarm()

        iteration = 0
        prev_best_fitness = float('-inf')

        while iteration < max_iterations:
            print(f"Iteration {iteration + 1}/{max_iterations}")

            # Update each agent
            tasks = [
                self._update_agent(agent)
                for agent in self.agents
            ]
            await asyncio.gather(*tasks)

            # Check convergence
            if abs(self.global_best_fitness - prev_best_fitness) < convergence_threshold:
                print(f"Converged after {iteration + 1} iterations")
                break

            prev_best_fitness = self.global_best_fitness
            iteration += 1

            print(f"  Best fitness: {self.global_best_fitness:.4f}")

        return {
            "best_solution": self.global_best_solution,
            "best_fitness": self.global_best_fitness,
            "iterations": iteration + 1
        }

    async def _initialize_swarm(self) -> None:
        """Initialize agent positions randomly in solution space."""
        for i in range(self.num_agents):
            # Random initial position
            position = self._random_solution()

            # Evaluate fitness
            fitness = await self.fitness_function(position)

            agent = {
                "id": f"swarm_agent_{i}",
                "position": position,
                "velocity": {},  # Initially zero
                "best_position": position.copy(),
                "best_fitness": fitness
            }

            self.agents.append(agent)

            # Update global best
            if fitness > self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_solution = position.copy()

    def _random_solution(self) -> Dict[str, Any]:
        """Generate random solution in solution space."""
        solution = {}

        for param, spec in self.solution_space.items():
            if spec["type"] == "continuous":
                value = random.uniform(spec["min"], spec["max"])
            elif spec["type"] == "discrete":
                value = random.choice(spec["options"])
            elif spec["type"] == "integer":
                value = random.randint(spec["min"], spec["max"])
            else:
                value = None

            solution[param] = value

        return solution

    async def _update_agent(self, agent: Dict[str, Any]) -> None:
        """Update agent position based on PSO rules."""
        # PSO update: velocity and position
        # v_new = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
        # x_new = x + v_new

        w = 0.7  # Inertia weight
        c1 = 1.5  # Cognitive weight
        c2 = 1.5  # Social weight

        new_position = {}

        for param in self.solution_space.keys():
            current = agent["position"][param]
            personal_best = agent["best_position"][param]
            global_best = self.global_best_solution[param]

            # Update velocity
            r1 = random.random()
            r2 = random.random()

            if isinstance(current, (int, float)):
                velocity = agent["velocity"].get(param, 0)

                new_velocity = (
                    w * velocity +
                    c1 * r1 * (personal_best - current) +
                    c2 * r2 * (global_best - current)
                )

                agent["velocity"][param] = new_velocity
                new_position[param] = current + new_velocity

                # Clamp to bounds
                spec = self.solution_space[param]
                if "min" in spec and "max" in spec:
                    new_position[param] = max(
                        spec["min"],
                        min(spec["max"], new_position[param])
                    )
            else:
                # For discrete parameters, probabilistic update
                if random.random() < 0.5:
                    new_position[param] = personal_best
                else:
                    new_position[param] = global_best

        # Evaluate new position
        fitness = await self.fitness_function(new_position)

        # Update agent's best
        if fitness > agent["best_fitness"]:
            agent["best_position"] = new_position.copy()
            agent["best_fitness"] = fitness

        # Update global best
        if fitness > self.global_best_fitness:
            self.global_best_fitness = fitness
            self.global_best_solution = new_position.copy()

        agent["position"] = new_position
```

---

## 7. Agent Memory & Knowledge Sharing

### 7.1 Shared Knowledge Base

**File**: `ia_modules/agents/communication/knowledge_base.py`

```python
"""Shared knowledge base for multi-agent systems."""
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio


class SharedKnowledgeBase:
    """
    Centralized knowledge repository for agent collaboration.

    Agents can:
    - Store discoveries
    - Query other agents' knowledge
    - Build on previous findings
    - Detect redundant work
    """

    def __init__(self):
        """Initialize knowledge base."""
        self._knowledge: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = asyncio.Lock()

    async def store(
        self,
        category: str,
        knowledge: Dict[str, Any],
        agent_id: str
    ) -> str:
        """
        Store knowledge in shared base.

        Args:
            category: Knowledge category
            knowledge: Knowledge content
            agent_id: Contributing agent

        Returns:
            Knowledge ID
        """
        async with self._lock:
            if category not in self._knowledge:
                self._knowledge[category] = []

            entry = {
                "id": f"{category}_{len(self._knowledge[category])}",
                "content": knowledge,
                "contributor": agent_id,
                "timestamp": datetime.now(),
                "confidence": knowledge.get("confidence", 1.0)
            }

            self._knowledge[category].append(entry)

            return entry["id"]

    async def query(
        self,
        category: str,
        filters: Optional[Dict[str, Any]] = None,
        min_confidence: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Query knowledge base.

        Args:
            category: Knowledge category
            filters: Optional filters
            min_confidence: Minimum confidence threshold

        Returns:
            Matching knowledge entries
        """
        async with self._lock:
            if category not in self._knowledge:
                return []

            results = self._knowledge[category]

            # Apply filters
            if filters:
                results = [
                    entry for entry in results
                    if all(
                        entry["content"].get(k) == v
                        for k, v in filters.items()
                    )
                ]

            # Filter by confidence
            results = [
                entry for entry in results
                if entry["confidence"] >= min_confidence
            ]

            return results

    async def get_agent_contributions(
        self,
        agent_id: str
    ) -> List[Dict[str, Any]]:
        """Get all knowledge contributed by specific agent."""
        contributions = []

        async with self._lock:
            for category, entries in self._knowledge.items():
                for entry in entries:
                    if entry["contributor"] == agent_id:
                        contributions.append({
                            "category": category,
                            **entry
                        })

        return contributions

    async def find_similar(
        self,
        query: Dict[str, Any],
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Find similar knowledge entries.

        Args:
            query: Query knowledge
            threshold: Similarity threshold

        Returns:
            Similar entries
        """
        # Simplified similarity - would use embeddings in production
        similar = []

        async with self._lock:
            for category, entries in self._knowledge.items():
                for entry in entries:
                    # Count matching fields
                    matches = sum(
                        1 for k, v in query.items()
                        if entry["content"].get(k) == v
                    )
                    similarity = matches / len(query) if query else 0

                    if similarity >= threshold:
                        similar.append({
                            "category": category,
                            "similarity": similarity,
                            **entry
                        })

        return sorted(similar, key=lambda x: x["similarity"], reverse=True)
```

---

## 8. Competitive Agent Evolution

### 8.1 Genetic Algorithm for Agent Improvement

**File**: `ia_modules/agents/evolution/genetic.py`

```python
"""Genetic algorithms for evolving agent capabilities."""
from typing import List, Dict, Any, Callable, Optional, Tuple
import asyncio
import random
import copy
from ..models import AgentProfile, AgentCapability


class AgentGenome(BaseModel):
    """Agent genetic representation."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_ids: List[str] = Field(default_factory=list)
    generation: int = 0

    # Evolvable parameters
    temperature: float = Field(0.7, ge=0, le=2)
    personality_traits: Dict[str, float] = Field(default_factory=dict)
    system_prompt_template: str = ""
    capabilities: List[AgentCapability] = Field(default_factory=list)
    model_preference: str = "gpt-4o"

    # Performance metrics
    fitness_score: float = 0.0
    task_success_rate: float = 0.0
    avg_response_quality: float = 0.0
    collaboration_score: float = 0.0

    class Config:
        json_schema_extra = {
            "example": {
                "temperature": 0.8,
                "personality_traits": {
                    "creativity": 0.9,
                    "analytical": 0.6,
                    "risk_tolerance": 0.7
                },
                "capabilities": ["research", "synthesis"],
                "fitness_score": 0.85
            }
        }


class GeneticAgentEvolution:
    """
    Evolve agent populations using genetic algorithms.

    Features:
    - Selection: Tournament, roulette wheel, rank-based
    - Crossover: Blend personality traits, combine capabilities
    - Mutation: Random variations in parameters
    - Elitism: Preserve best performers
    - Speciation: Maintain agent diversity
    """

    def __init__(
        self,
        population_size: int = 20,
        elite_ratio: float = 0.1,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.7
    ):
        """
        Initialize genetic evolution.

        Args:
            population_size: Number of agents in population
            elite_ratio: Ratio of top performers to preserve
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
        """
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.population: List[AgentGenome] = []
        self.generation = 0
        self.best_genome: Optional[AgentGenome] = None
        self.fitness_history: List[float] = []

    async def evolve(
        self,
        fitness_function: Callable[[AgentGenome], Awaitable[float]],
        num_generations: int = 50,
        convergence_threshold: float = 0.01
    ) -> AgentGenome:
        """
        Evolve agent population over multiple generations.

        Args:
            fitness_function: Async function to evaluate agent fitness
            num_generations: Number of generations to evolve
            convergence_threshold: Stop if improvement < threshold

        Returns:
            Best evolved agent genome
        """
        # Initialize random population
        if not self.population:
            await self._initialize_population()

        for gen in range(num_generations):
            self.generation = gen
            print(f"\n=== Generation {gen + 1}/{num_generations} ===")

            # Evaluate fitness for all agents
            await self._evaluate_population(fitness_function)

            # Track best agent
            current_best = max(self.population, key=lambda g: g.fitness_score)
            if not self.best_genome or current_best.fitness_score > self.best_genome.fitness_score:
                self.best_genome = copy.deepcopy(current_best)

            avg_fitness = sum(g.fitness_score for g in self.population) / len(self.population)
            self.fitness_history.append(avg_fitness)

            print(f"Best fitness: {self.best_genome.fitness_score:.4f}")
            print(f"Average fitness: {avg_fitness:.4f}")

            # Check convergence
            if len(self.fitness_history) >= 5:
                recent_improvement = abs(self.fitness_history[-1] - self.fitness_history[-5])
                if recent_improvement < convergence_threshold:
                    print(f"Converged after {gen + 1} generations")
                    break

            # Create next generation
            self.population = await self._create_next_generation()

        return self.best_genome

    async def _initialize_population(self) -> None:
        """Initialize random population of agents."""
        for i in range(self.population_size):
            genome = AgentGenome(
                id=f"agent_gen0_{i}",
                generation=0,
                temperature=random.uniform(0.1, 1.5),
                personality_traits={
                    "creativity": random.random(),
                    "analytical": random.random(),
                    "risk_tolerance": random.random(),
                    "thoroughness": random.random(),
                    "skepticism": random.random()
                },
                capabilities=random.sample(
                    list(AgentCapability),
                    k=random.randint(2, 4)
                ),
                system_prompt_template=self._generate_random_prompt()
            )
            self.population.append(genome)

    def _generate_random_prompt(self) -> str:
        """Generate random system prompt template."""
        templates = [
            "You are a {role} agent. Focus on {capability} and provide {style} responses.",
            "As a {role}, your specialty is {capability}. Approach tasks with {style}.",
            "You excel at {capability}. As a {role}, maintain a {style} perspective.",
        ]
        return random.choice(templates)

    async def _evaluate_population(
        self,
        fitness_function: Callable[[AgentGenome], Awaitable[float]]
    ) -> None:
        """Evaluate fitness for all agents in population."""
        tasks = [fitness_function(genome) for genome in self.population]
        fitness_scores = await asyncio.gather(*tasks, return_exceptions=True)

        for genome, score in zip(self.population, fitness_scores):
            if isinstance(score, Exception):
                genome.fitness_score = 0.0
            else:
                genome.fitness_score = score

    async def _create_next_generation(self) -> List[AgentGenome]:
        """Create next generation through selection, crossover, and mutation."""
        next_gen = []

        # Elitism: preserve top performers
        num_elites = max(1, int(self.population_size * self.elite_ratio))
        elites = sorted(self.population, key=lambda g: g.fitness_score, reverse=True)[:num_elites]
        next_gen.extend([copy.deepcopy(e) for e in elites])

        # Fill rest with offspring
        while len(next_gen) < self.population_size:
            # Selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()

            # Crossover
            if random.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = copy.deepcopy(parent1)

            # Mutation
            if random.random() < self.mutation_rate:
                child = self._mutate(child)

            child.generation = self.generation + 1
            child.id = f"agent_gen{child.generation}_{len(next_gen)}"
            next_gen.append(child)

        return next_gen

    def _tournament_selection(self, tournament_size: int = 3) -> AgentGenome:
        """Select agent using tournament selection."""
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda g: g.fitness_score)

    def _crossover(self, parent1: AgentGenome, parent2: AgentGenome) -> AgentGenome:
        """Crossover two parent genomes."""
        child = AgentGenome(
            parent_ids=[parent1.id, parent2.id],
            generation=max(parent1.generation, parent2.generation) + 1
        )

        # Blend temperature
        child.temperature = (parent1.temperature + parent2.temperature) / 2

        # Blend personality traits
        all_traits = set(parent1.personality_traits.keys()) | set(parent2.personality_traits.keys())
        child.personality_traits = {}
        for trait in all_traits:
            val1 = parent1.personality_traits.get(trait, 0.5)
            val2 = parent2.personality_traits.get(trait, 0.5)
            child.personality_traits[trait] = (val1 + val2) / 2

        # Combine capabilities (union, then randomly sample)
        combined_caps = list(set(parent1.capabilities) | set(parent2.capabilities))
        child.capabilities = random.sample(
            combined_caps,
            k=min(len(combined_caps), random.randint(2, 5))
        )

        # Inherit system prompt from fitter parent
        if parent1.fitness_score > parent2.fitness_score:
            child.system_prompt_template = parent1.system_prompt_template
            child.model_preference = parent1.model_preference
        else:
            child.system_prompt_template = parent2.system_prompt_template
            child.model_preference = parent2.model_preference

        return child

    def _mutate(self, genome: AgentGenome) -> AgentGenome:
        """Apply random mutations to genome."""
        mutation_type = random.choice([
            "temperature", "personality", "capabilities", "prompt"
        ])

        if mutation_type == "temperature":
            # Mutate temperature
            genome.temperature += random.gauss(0, 0.2)
            genome.temperature = max(0.0, min(2.0, genome.temperature))

        elif mutation_type == "personality":
            # Mutate one random personality trait
            if genome.personality_traits:
                trait = random.choice(list(genome.personality_traits.keys()))
                genome.personality_traits[trait] += random.gauss(0, 0.1)
                genome.personality_traits[trait] = max(0, min(1, genome.personality_traits[trait]))

        elif mutation_type == "capabilities":
            # Add or remove a capability
            if random.random() < 0.5 and len(genome.capabilities) < len(AgentCapability):
                # Add capability
                available = [c for c in AgentCapability if c not in genome.capabilities]
                if available:
                    genome.capabilities.append(random.choice(available))
            elif len(genome.capabilities) > 1:
                # Remove capability
                genome.capabilities.pop(random.randint(0, len(genome.capabilities) - 1))

        elif mutation_type == "prompt":
            # Mutate system prompt
            genome.system_prompt_template = self._generate_random_prompt()

        return genome

    def get_lineage(self, genome_id: str) -> List[str]:
        """Get evolutionary lineage of an agent."""
        lineage = [genome_id]

        genome = next((g for g in self.population if g.id == genome_id), None)
        if genome and genome.parent_ids:
            for parent_id in genome.parent_ids:
                lineage.extend(self.get_lineage(parent_id))

        return lineage


async def example_fitness_function(genome: AgentGenome) -> float:
    """
    Example fitness function for agent evaluation.

    In practice, this would run actual tasks and measure performance.
    """
    # Simulate task performance based on genome attributes
    score = 0.0

    # Reward balanced personality
    if genome.personality_traits:
        balance = 1 - abs(0.5 - (sum(genome.personality_traits.values()) / len(genome.personality_traits)))
        score += balance * 0.3

    # Reward diverse capabilities
    score += len(genome.capabilities) * 0.1

    # Reward moderate temperature
    temp_score = 1 - abs(0.7 - genome.temperature)
    score += temp_score * 0.2

    # Add some randomness (simulating task variance)
    score += random.uniform(0, 0.4)

    return min(1.0, score)
```

### 8.2 Tournament-Based Agent Competition

**File**: `ia_modules/agents/evolution/tournament.py`

```python
"""Tournament-based competitive agent selection."""
from typing import List, Dict, Any, Callable, Awaitable, Optional
import asyncio
from datetime import datetime
from pydantic import BaseModel, Field
from ..models import AgentProfile
from .genetic import AgentGenome


class TournamentMatch(BaseModel):
    """Single tournament match between agents."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent1_id: str
    agent2_id: str
    task: str
    winner_id: Optional[str] = None
    agent1_score: float = 0.0
    agent2_score: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TournamentBracket(BaseModel):
    """Tournament bracket structure."""
    round_number: int
    matches: List[TournamentMatch] = Field(default_factory=list)
    completed: bool = False


class AgentTournament:
    """
    Run competitive tournaments to select best-performing agents.

    Formats:
    - Single elimination
    - Double elimination
    - Round robin
    - Swiss system
    """

    def __init__(
        self,
        agents: List[AgentProfile],
        task_generator: Callable[[int], Awaitable[str]],
        judge_function: Callable[[str, str, str], Awaitable[Tuple[float, float]]]
    ):
        """
        Initialize tournament.

        Args:
            agents: Competing agents
            task_generator: Function to generate test tasks
            judge_function: Function to judge agent responses
        """
        self.agents = agents
        self.task_generator = task_generator
        self.judge_function = judge_function
        self.brackets: List[TournamentBracket] = []
        self.agent_stats: Dict[str, Dict[str, Any]] = {
            agent.id: {
                "wins": 0,
                "losses": 0,
                "total_score": 0.0,
                "matches_played": 0
            }
            for agent in agents
        }

    async def run_single_elimination(self) -> AgentProfile:
        """
        Run single elimination tournament.

        Returns:
            Winning agent
        """
        remaining_agents = self.agents.copy()
        round_num = 1

        while len(remaining_agents) > 1:
            print(f"\n=== Round {round_num} - {len(remaining_agents)} agents ===")

            bracket = TournamentBracket(round_number=round_num)

            # Pair agents randomly
            random.shuffle(remaining_agents)

            winners = []
            for i in range(0, len(remaining_agents) - 1, 2):
                agent1 = remaining_agents[i]
                agent2 = remaining_agents[i + 1]

                # Run match
                winner = await self._run_match(agent1, agent2, bracket, round_num)
                winners.append(winner)

            # Handle odd agent (gets bye)
            if len(remaining_agents) % 2 == 1:
                bye_agent = remaining_agents[-1]
                print(f"  {bye_agent.name} receives bye")
                winners.append(bye_agent)

            bracket.completed = True
            self.brackets.append(bracket)

            remaining_agents = winners
            round_num += 1

        champion = remaining_agents[0]
        print(f"\n🏆 CHAMPION: {champion.name} ({champion.id})")

        return champion

    async def run_round_robin(self) -> List[AgentProfile]:
        """
        Run round robin tournament (everyone plays everyone).

        Returns:
            Agents ranked by performance
        """
        print(f"\n=== Round Robin Tournament - {len(self.agents)} agents ===")

        bracket = TournamentBracket(round_number=1)

        # Generate all pairs
        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                await self._run_match(
                    self.agents[i],
                    self.agents[j],
                    bracket,
                    round_num=1
                )

        bracket.completed = True
        self.brackets.append(bracket)

        # Rank by wins and total score
        ranked = sorted(
            self.agents,
            key=lambda a: (
                self.agent_stats[a.id]["wins"],
                self.agent_stats[a.id]["total_score"]
            ),
            reverse=True
        )

        print("\n=== FINAL RANKINGS ===")
        for i, agent in enumerate(ranked, 1):
            stats = self.agent_stats[agent.id]
            print(f"{i}. {agent.name}: {stats['wins']} wins, {stats['total_score']:.2f} total score")

        return ranked

    async def _run_match(
        self,
        agent1: AgentProfile,
        agent2: AgentProfile,
        bracket: TournamentBracket,
        round_num: int
    ) -> AgentProfile:
        """Run a single match between two agents."""
        # Generate task
        task = await self.task_generator(round_num)

        print(f"  Match: {agent1.name} vs {agent2.name}")

        # Get responses from both agents
        response1 = await self._get_agent_response(agent1, task)
        response2 = await self._get_agent_response(agent2, task)

        # Judge responses
        score1, score2 = await self.judge_function(task, response1, response2)

        # Determine winner
        winner = agent1 if score1 > score2 else agent2

        # Record match
        match = TournamentMatch(
            agent1_id=agent1.id,
            agent2_id=agent2.id,
            task=task,
            winner_id=winner.id,
            agent1_score=score1,
            agent2_score=score2,
            metadata={
                "round": round_num,
                "response1_length": len(response1),
                "response2_length": len(response2)
            }
        )
        bracket.matches.append(match)

        # Update stats
        self.agent_stats[agent1.id]["matches_played"] += 1
        self.agent_stats[agent2.id]["matches_played"] += 1
        self.agent_stats[agent1.id]["total_score"] += score1
        self.agent_stats[agent2.id]["total_score"] += score2

        if score1 > score2:
            self.agent_stats[agent1.id]["wins"] += 1
            self.agent_stats[agent2.id]["losses"] += 1
        else:
            self.agent_stats[agent2.id]["wins"] += 1
            self.agent_stats[agent1.id]["losses"] += 1

        print(f"    Winner: {winner.name} ({score1:.2f} vs {score2:.2f})")

        return winner

    async def _get_agent_response(self, agent: AgentProfile, task: str) -> str:
        """Get agent's response to task."""
        # This would call the actual agent's LLM
        # Simplified for example
        from openai import AsyncOpenAI
        client = AsyncOpenAI()

        response = await client.chat.completions.create(
            model=agent.model,
            messages=[
                {"role": "system", "content": agent.system_prompt},
                {"role": "user", "content": task}
            ],
            temperature=agent.temperature
        )

        return response.choices[0].message.content

    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Get current tournament leaderboard."""
        leaderboard = []

        for agent in self.agents:
            stats = self.agent_stats[agent.id]
            win_rate = stats["wins"] / max(1, stats["matches_played"])
            avg_score = stats["total_score"] / max(1, stats["matches_played"])

            leaderboard.append({
                "agent_id": agent.id,
                "agent_name": agent.name,
                "wins": stats["wins"],
                "losses": stats["losses"],
                "win_rate": win_rate,
                "avg_score": avg_score,
                "total_score": stats["total_score"]
            })

        return sorted(leaderboard, key=lambda x: (x["win_rate"], x["avg_score"]), reverse=True)


class CoevolutionSystem:
    """
    Combined genetic evolution with tournament selection.

    Agents compete in tournaments, and top performers
    are selected for breeding the next generation.
    """

    def __init__(
        self,
        genetic_evolution: GeneticAgentEvolution,
        tournament_system: AgentTournament
    ):
        """
        Initialize coevolution system.

        Args:
            genetic_evolution: Genetic algorithm
            tournament_system: Tournament system
        """
        self.genetic = genetic_evolution
        self.tournament = tournament_system

    async def evolve_through_competition(
        self,
        num_generations: int = 20
    ) -> List[AgentProfile]:
        """
        Evolve agents through competitive tournaments.

        Args:
            num_generations: Number of evolution cycles

        Returns:
            Final evolved agent population
        """
        for gen in range(num_generations):
            print(f"\n{'='*60}")
            print(f"COEVOLUTION GENERATION {gen + 1}/{num_generations}")
            print(f"{'='*60}")

            # Convert genomes to agent profiles
            agents = [self._genome_to_agent(g) for g in self.genetic.population]
            self.tournament.agents = agents

            # Run tournament
            ranked_agents = await self.tournament.run_round_robin()

            # Use tournament results as fitness
            for i, agent in enumerate(ranked_agents):
                # Find corresponding genome
                genome = next(g for g in self.genetic.population if g.id == agent.id)

                # Fitness based on tournament rank
                genome.fitness_score = 1.0 - (i / len(ranked_agents))

                # Add tournament stats
                stats = self.tournament.agent_stats[agent.id]
                genome.task_success_rate = stats["wins"] / max(1, stats["matches_played"])
                genome.collaboration_score = stats["avg_score"] if "avg_score" in stats else 0

            # Evolve population
            self.genetic.population = await self.genetic._create_next_generation()

        # Return final best agents
        final_agents = [self._genome_to_agent(g) for g in self.genetic.population]
        return sorted(
            final_agents,
            key=lambda a: next(g.fitness_score for g in self.genetic.population if g.id == a.id),
            reverse=True
        )

    def _genome_to_agent(self, genome: AgentGenome) -> AgentProfile:
        """Convert genome to agent profile."""
        # Generate system prompt from template
        system_prompt = genome.system_prompt_template.format(
            role=genome.capabilities[0].value if genome.capabilities else "generalist",
            capability=", ".join(c.value for c in genome.capabilities),
            style="creative" if genome.personality_traits.get("creativity", 0.5) > 0.7 else "analytical"
        )

        return AgentProfile(
            id=genome.id,
            name=f"Agent {genome.id[-4:]}",
            role=genome.capabilities[0].value if genome.capabilities else "generalist",
            capabilities=genome.capabilities,
            personality=genome.personality_traits,
            model=genome.model_preference,
            temperature=genome.temperature,
            system_prompt=system_prompt
        )
```

### 8.3 Usage Example: Competitive Evolution

```python
"""Example: Evolving agents through competitive tournaments."""
import asyncio
from openai import AsyncOpenAI
from ia_modules.agents.evolution.genetic import GeneticAgentEvolution, example_fitness_function
from ia_modules.agents.evolution.tournament import AgentTournament, CoevolutionSystem
from ia_modules.agents.models import AgentProfile, AgentCapability


async def task_generator(round_num: int) -> str:
    """Generate increasingly difficult tasks."""
    tasks = [
        "Explain quantum entanglement in simple terms.",
        "Design a sustainable urban transportation system.",
        "Analyze the ethical implications of AI in healthcare.",
        "Propose a solution to reduce ocean plastic pollution."
    ]
    return tasks[min(round_num - 1, len(tasks) - 1)]


async def judge_function(task: str, response1: str, response2: str) -> tuple[float, float]:
    """Judge which response is better."""
    client = AsyncOpenAI()

    prompt = f"""Task: {task}

Response A: {response1}

Response B: {response2}

Judge both responses on:
1. Accuracy and correctness
2. Clarity and organization
3. Completeness
4. Creativity

Provide two scores (0-1) as: ScoreA ScoreB"""

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    # Parse scores (simplified)
    try:
        scores = response.choices[0].message.content.split()
        score1 = float(scores[0])
        score2 = float(scores[1])
    except:
        # Fallback: random scores
        import random
        score1 = random.random()
        score2 = random.random()

    return score1, score2


async def main():
    # Initialize genetic evolution
    genetic = GeneticAgentEvolution(
        population_size=16,
        elite_ratio=0.2,
        mutation_rate=0.3,
        crossover_rate=0.7
    )

    # Initialize tournament
    tournament = AgentTournament(
        agents=[],  # Will be filled by coevolution
        task_generator=task_generator,
        judge_function=judge_function
    )

    # Create coevolution system
    coevolution = CoevolutionSystem(
        genetic_evolution=genetic,
        tournament_system=tournament
    )

    # Evolve agents through competition
    print("Starting competitive coevolution...")
    final_agents = await coevolution.evolve_through_competition(
        num_generations=10
    )

    # Display results
    print(f"\n{'='*60}")
    print("FINAL EVOLVED AGENTS")
    print(f"{'='*60}\n")

    for i, agent in enumerate(final_agents[:5], 1):
        genome = next(g for g in genetic.population if g.id == agent.id)
        print(f"{i}. {agent.name}")
        print(f"   Fitness: {genome.fitness_score:.4f}")
        print(f"   Success Rate: {genome.task_success_rate:.2%}")
        print(f"   Capabilities: {', '.join(c.value for c in agent.capabilities)}")
        print(f"   Temperature: {agent.temperature:.2f}")
        print(f"   Generation: {genome.generation}\n")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 9. Human-in-the-Loop Collaboration

### 9.1 Human-Agent Collaboration Pattern

**File**: `ia_modules/agents/collaboration/human_agent.py`

```python
"""Human-in-the-loop collaboration with agents."""
from typing import List, Dict, Any, Optional, Callable, Awaitable
import asyncio
from datetime import datetime
from ..models import AgentProfile, AgentMessage, MessageType


class HumanAgentCollaboration:
    """
    Facilitate collaboration between human experts and AI agents.

    Patterns:
    - Human oversight: Agent proposes, human approves
    - Human guidance: Human steers agent direction
    - Human expertise injection: Human provides domain knowledge
    - Collaborative refinement: Iterative human-agent improvement
    """

    def __init__(
        self,
        agents: List[AgentProfile],
        human_input_callback: Callable[[str], Awaitable[str]],
        llm_client: Any
    ):
        """
        Initialize human-agent collaboration.

        Args:
            agents: AI agents
            human_input_callback: Async function to get human input
            llm_client: LLM client
        """
        self.agents = agents
        self.human_input = human_input_callback
        self.llm_client = llm_client
        self.conversation_history: List[Dict[str, Any]] = []

    async def collaborative_problem_solving(
        self,
        problem: str,
        max_iterations: int = 5
    ) -> Dict[str, Any]:
        """
        Solve problem collaboratively with human in the loop.

        Args:
            problem: Problem to solve
            max_iterations: Maximum iteration cycles

        Returns:
            Final solution with human approval
        """
        print(f"Problem: {problem}\n")

        current_solution = None

        for iteration in range(max_iterations):
            print(f"\n=== Iteration {iteration + 1}/{max_iterations} ===\n")

            # Phase 1: Agents propose solutions
            agent_proposals = await self._get_agent_proposals(problem, current_solution)

            # Phase 2: Present to human
            print("\nAgent Proposals:")
            for i, (agent, proposal) in enumerate(zip(self.agents, agent_proposals), 1):
                print(f"\n{i}. {agent.name}:")
                print(f"{proposal[:200]}...")

            # Phase 3: Get human feedback
            print("\n" + "="*60)
            human_feedback = await self.human_input(
                "Please provide feedback or select a proposal (or 'approve' to accept):"
            )

            self.conversation_history.append({
                "iteration": iteration + 1,
                "agent_proposals": agent_proposals,
                "human_feedback": human_feedback,
                "timestamp": datetime.now()
            })

            # Check for approval
            if "approve" in human_feedback.lower():
                print("\n✅ Solution approved by human!")
                current_solution = agent_proposals[0]
                break

            # Phase 4: Agents refine based on feedback
            current_solution = await self._refine_with_feedback(
                agent_proposals,
                human_feedback
            )

            print(f"\nRefined solution:\n{current_solution[:200]}...")

        return {
            "final_solution": current_solution,
            "iterations": len(self.conversation_history),
            "collaboration_history": self.conversation_history
        }

    async def _get_agent_proposals(
        self,
        problem: str,
        current_solution: Optional[str]
    ) -> List[str]:
        """Get proposals from all agents."""
        proposals = []

        context = f"Problem: {problem}"
        if current_solution:
            context += f"\n\nCurrent solution:\n{current_solution}"

        for agent in self.agents:
            prompt = f"""{context}

Propose your solution or improvement. Consider:
- Feasibility
- Effectiveness
- Potential issues
"""

            response = await self.llm_client.chat.completions.create(
                model=agent.model,
                messages=[
                    {"role": "system", "content": agent.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=agent.temperature
            )

            proposals.append(response.choices[0].message.content)

        return proposals

    async def _refine_with_feedback(
        self,
        proposals: List[str],
        feedback: str
    ) -> str:
        """Refine solutions based on human feedback."""
        # Use first agent to synthesize
        synthesizer = self.agents[0]

        proposals_text = "\n\n".join([
            f"Proposal {i+1}:\n{p}"
            for i, p in enumerate(proposals)
        ])

        prompt = f"""Proposals:
{proposals_text}

Human feedback: {feedback}

Synthesize an improved solution that addresses the human's feedback."""

        response = await self.llm_client.chat.completions.create(
            model=synthesizer.model,
            messages=[
                {"role": "system", "content": synthesizer.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        return response.choices[0].message.content
```

---

## 10. Pipeline Integration

### 10.1 Multi-Agent Pipeline Step

**File**: `ia_modules/agents/pipeline_integration.py`

```python
"""Integration of multi-agent systems with ia_modules pipelines."""
from typing import Dict, Any, List, Optional
from ia_modules.pipeline.core import Step
from .collaboration.debate import AgentDebate
from .collaboration.delegation import AgentOrchestrator
from .collaboration.voting import VotingSystem
from .models import AgentProfile, AgentCapability
from openai import AsyncOpenAI


class MultiAgentDebateStep(Step):
    """Pipeline step that runs agent debate."""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.api_key = config.get("openai_api_key")
        self.max_rounds = config.get("max_rounds", 3)
        self.num_agents = config.get("num_agents", 4)

    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run multi-agent debate on input topic."""
        topic = data.get("topic") or data.get("question")
        context = data.get("context", {})

        # Create diverse agent team
        agents = self._create_agent_team()

        # Initialize debate
        debate = AgentDebate(
            agents=agents,
            llm_client=AsyncOpenAI(api_key=self.api_key),
            max_rounds=self.max_rounds,
            consensus_threshold=0.6
        )

        # Run debate
        result = await debate.run_debate(topic, context)

        return {
            "debate_result": result,
            "winning_proposal": result["winning_proposal"],
            "winner_agent": result["winner"],
            "rounds_completed": result["rounds"],
            "all_perspectives": result["all_proposals"]
        }

    def _create_agent_team(self) -> List[AgentProfile]:
        """Create diverse team of agents."""
        return [
            AgentProfile(
                id="agent_optimist",
                name="Optimistic Innovator",
                role="innovator",
                capabilities=[AgentCapability.CREATIVITY, AgentCapability.PLANNING],
                system_prompt="You are optimistic and focus on opportunities."
            ),
            AgentProfile(
                id="agent_skeptic",
                name="Critical Skeptic",
                role="critic",
                capabilities=[AgentCapability.CRITIQUE, AgentCapability.ANALYSIS],
                system_prompt="You are skeptical and identify risks."
            ),
            AgentProfile(
                id="agent_analyst",
                name="Data Analyst",
                role="analyst",
                capabilities=[AgentCapability.ANALYSIS, AgentCapability.VERIFICATION],
                system_prompt="You analyze data and evidence."
            ),
            AgentProfile(
                id="agent_pragmatist",
                name="Practical Pragmatist",
                role="executor",
                capabilities=[AgentCapability.EXECUTION, AgentCapability.PLANNING],
                system_prompt="You focus on practical implementation."
            )
        ]


class MultiAgentOrchestratorStep(Step):
    """Pipeline step that orchestrates specialist agents."""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.api_key = config.get("openai_api_key")

    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate task across specialist agents."""
        task = data.get("task")
        context = data.get("context", {})

        # Create master and specialists
        master = AgentProfile(
            id="master_orchestrator",
            name="Master Orchestrator",
            role="orchestrator",
            capabilities=[AgentCapability.PLANNING],
            system_prompt="You coordinate and delegate tasks to specialists."
        )

        specialists = [
            AgentProfile(
                id="researcher",
                name="Research Specialist",
                role="researcher",
                capabilities=[AgentCapability.RESEARCH],
                system_prompt="You excel at gathering and organizing information."
            ),
            AgentProfile(
                id="analyst",
                name="Analysis Specialist",
                role="analyst",
                capabilities=[AgentCapability.ANALYSIS],
                system_prompt="You analyze data and draw insights."
            ),
            AgentProfile(
                id="synthesizer",
                name="Synthesis Specialist",
                role="synthesizer",
                capabilities=[AgentCapability.SYNTHESIS],
                system_prompt="You combine information into coherent narratives."
            )
        ]

        # Run orchestration
        orchestrator = AgentOrchestrator(
            master_agent=master,
            specialist_agents=specialists,
            llm_client=AsyncOpenAI(api_key=self.api_key)
        )

        result = await orchestrator.orchestrate(task, context)

        return {
            "orchestration_result": result,
            "final_answer": result["final_result"],
            "subtasks_completed": len(result["subtasks"]),
            "specialist_results": result["subtask_results"]
        }
```

**Example Pipeline JSON**:

```json
{
  "name": "Multi-Agent Analysis Pipeline",
  "description": "Use multiple AI agents to analyze complex topics",
  "steps": [
    {
      "id": "multi_agent_debate",
      "name": "Agent Debate",
      "step_class": "MultiAgentDebateStep",
      "module": "ia_modules.agents.pipeline_integration",
      "config": {
        "openai_api_key": "${OPENAI_API_KEY}",
        "max_rounds": 3,
        "num_agents": 4
      }
    },
    {
      "id": "orchestrated_research",
      "name": "Orchestrated Research",
      "step_class": "MultiAgentOrchestratorStep",
      "module": "ia_modules.agents.pipeline_integration",
      "config": {
        "openai_api_key": "${OPENAI_API_KEY}"
      }
    }
  ],
  "flow": {
    "start_at": "multi_agent_debate",
    "paths": [
      {
        "from": "multi_agent_debate",
        "to": "orchestrated_research",
        "condition": {"type": "always"}
      },
      {
        "from": "orchestrated_research",
        "to": "end_with_success",
        "condition": {"type": "always"}
      }
    ]
  }
}
```

---

## Summary

This implementation plan provides:

✅ **Agent Communication** - Message bus, protocols, structured messaging
✅ **Debate Pattern** - Multi-round debates with proposals, critiques, voting
✅ **Delegation** - Master-worker pattern with task decomposition
✅ **Voting Systems** - Multiple voting mechanisms (majority, weighted, ranked, quadratic)
✅ **Swarm Intelligence** - Particle swarm optimization for distributed problem solving
✅ **Shared Knowledge** - Centralized knowledge base for collaboration
✅ **Competitive Evolution** - Genetic algorithms + tournament selection for agent improvement
✅ **Human-in-the-Loop** - Collaborative problem solving with human oversight
✅ **Pipeline Integration** - Multi-agent steps compatible with graph_pipeline_runner.py
✅ **Type Safety** - Full Pydantic models throughout
✅ **Async/Await** - Non-blocking concurrent agent operations

### Competitive Evolution Features

🧬 **Genetic Algorithm**
- Selection: Tournament, roulette wheel, rank-based
- Crossover: Blend personality traits, combine capabilities
- Mutation: Random variations in temperature, prompts, capabilities
- Elitism: Preserve top performers across generations
- Fitness tracking: Multi-dimensional performance metrics

🏆 **Tournament System**
- Single elimination: Fast winner selection
- Round robin: Comprehensive performance evaluation
- Leaderboards: Track wins, losses, scores
- Match recording: Full tournament history

🔄 **Coevolution**
- Competition drives evolution
- Tournament results → fitness scores
- Best performers breed next generation
- Continuous improvement through competitive pressure

👥 **Human Collaboration**
- Oversight and approval workflows
- Expert guidance injection
- Iterative refinement
- Conversation history tracking

**This is cutting-edge, exciting AI infrastructure** - not dry operational stuff!

Multi-agent systems represent the future of AI, enabling:
- More robust decision making through diverse perspectives
- Complex problem decomposition and parallel solving
- Emergent intelligent behaviors
- Reduced single-agent biases and hallucinations
- **Self-improving agents through competitive evolution**
- **Human-AI collaborative intelligence**
