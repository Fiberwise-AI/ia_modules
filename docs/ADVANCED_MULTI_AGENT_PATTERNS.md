# Advanced Multi-Agent Patterns Implementation Plan

## Overview

This document extends the multi-agent collaboration capabilities with cutting-edge patterns discovered from 2024-2025 research, including MARL, advanced memory systems, temporal knowledge graphs, agent-to-agent protocols, and curriculum learning.

## Table of Contents

1. [Multi-Agent Reinforcement Learning (MARL)](#1-multi-agent-reinforcement-learning-marl)
2. [Advanced Memory Systems](#2-advanced-memory-systems)
3. [Temporal Knowledge Graphs](#3-temporal-knowledge-graphs)
4. [Agent-to-Agent (A2A) Protocol](#4-agent-to-agent-a2a-protocol)
5. [Curriculum Learning for Agents](#5-curriculum-learning-for-agents)
6. [Self-Organized Agent Networks](#6-self-organized-agent-networks)
7. [Dynamic Agent Teams](#7-dynamic-agent-teams)
8. [Agent Reflection & Introspection](#8-agent-reflection--introspection)
9. [Multi-Agent Planning](#9-multi-agent-planning)
10. [Social Learning & Imitation](#10-social-learning--imitation)

---

## 1. Multi-Agent Reinforcement Learning (MARL)

### 1.1 Cooperative MARL

**File**: `ia_modules/agents/marl/cooperative.py`

```python
"""Cooperative Multi-Agent Reinforcement Learning."""
from typing import List, Dict, Any, Tuple, Optional
import asyncio
import numpy as np
from pydantic import BaseModel, Field
from datetime import datetime


class AgentState(BaseModel):
    """State representation for a single agent."""
    agent_id: str
    position: List[float] = Field(default_factory=list)
    inventory: Dict[str, Any] = Field(default_factory=dict)
    health: float = 1.0
    energy: float = 1.0
    observations: List[Any] = Field(default_factory=list)


class SharedState(BaseModel):
    """Shared state visible to all agents."""
    global_resources: Dict[str, float] = Field(default_factory=dict)
    team_score: float = 0.0
    timestep: int = 0
    completed_objectives: List[str] = Field(default_factory=list)


class Action(BaseModel):
    """Agent action representation."""
    agent_id: str
    action_type: str  # "move", "gather", "build", "communicate", etc.
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class Reward(BaseModel):
    """Reward signal for agent."""
    agent_id: str
    individual_reward: float = 0.0
    team_reward: float = 0.0
    total_reward: float = 0.0
    reward_breakdown: Dict[str, float] = Field(default_factory=dict)


class CooperativeEnvironment:
    """
    Cooperative multi-agent environment.

    Agents share a common goal and receive team rewards.
    Implements credit assignment to determine individual contributions.
    """

    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        action_dim: int,
        reward_shaping: bool = True
    ):
        """
        Initialize cooperative environment.

        Args:
            num_agents: Number of cooperative agents
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            reward_shaping: Enable reward shaping for better learning
        """
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_shaping = reward_shaping

        self.agent_states: Dict[str, AgentState] = {}
        self.shared_state = SharedState()
        self.timestep = 0

    async def reset(self) -> Dict[str, AgentState]:
        """Reset environment to initial state."""
        self.agent_states = {
            f"agent_{i}": AgentState(
                agent_id=f"agent_{i}",
                position=[np.random.uniform(-1, 1) for _ in range(2)],
                health=1.0,
                energy=1.0
            )
            for i in range(self.num_agents)
        }

        self.shared_state = SharedState(
            global_resources={"food": 100.0, "materials": 100.0},
            team_score=0.0,
            timestep=0
        )

        self.timestep = 0
        return self.agent_states

    async def step(
        self,
        actions: Dict[str, Action]
    ) -> Tuple[Dict[str, AgentState], Dict[str, Reward], bool, Dict[str, Any]]:
        """
        Execute one timestep with joint actions.

        Args:
            actions: Dict mapping agent_id to Action

        Returns:
            Tuple of (next_states, rewards, done, info)
        """
        # Execute actions in environment
        next_states = await self._execute_actions(actions)

        # Calculate team reward
        team_reward = self._calculate_team_reward()

        # Credit assignment: distribute team reward to individuals
        individual_rewards = await self._credit_assignment(actions, team_reward)

        # Combine individual and team rewards
        rewards = {
            agent_id: Reward(
                agent_id=agent_id,
                individual_reward=individual_rewards[agent_id],
                team_reward=team_reward,
                total_reward=individual_rewards[agent_id] + team_reward
            )
            for agent_id in self.agent_states.keys()
        }

        # Check if episode is done
        done = self._check_done()

        # Additional info
        info = {
            "timestep": self.timestep,
            "team_score": self.shared_state.team_score,
            "global_resources": self.shared_state.global_resources
        }

        self.timestep += 1
        self.shared_state.timestep = self.timestep

        return next_states, rewards, done, info

    async def _execute_actions(
        self,
        actions: Dict[str, Action]
    ) -> Dict[str, AgentState]:
        """Execute all agent actions and update states."""
        for agent_id, action in actions.items():
            state = self.agent_states[agent_id]

            if action.action_type == "move":
                # Move agent
                direction = action.parameters.get("direction", [0, 0])
                state.position[0] += direction[0] * 0.1
                state.position[1] += direction[1] * 0.1
                state.energy -= 0.01

            elif action.action_type == "gather":
                # Gather resources
                resource = action.parameters.get("resource", "food")
                amount = 1.0
                self.shared_state.global_resources[resource] += amount
                state.energy -= 0.05

            elif action.action_type == "communicate":
                # Share information with team
                message = action.parameters.get("message", "")
                # Broadcast to other agents' observations
                for other_id, other_state in self.agent_states.items():
                    if other_id != agent_id:
                        other_state.observations.append({
                            "from": agent_id,
                            "message": message,
                            "timestep": self.timestep
                        })

        return self.agent_states

    def _calculate_team_reward(self) -> float:
        """Calculate reward based on team performance."""
        # Example: reward based on resource accumulation
        total_resources = sum(self.shared_state.global_resources.values())
        team_reward = total_resources / 100.0  # Normalized

        # Bonus for objectives
        team_reward += len(self.shared_state.completed_objectives) * 10.0

        return team_reward

    async def _credit_assignment(
        self,
        actions: Dict[str, Action],
        team_reward: float
    ) -> Dict[str, float]:
        """
        Assign credit to individual agents for team success.

        Uses counterfactual reasoning: "What would have happened
        if this agent didn't take this action?"
        """
        individual_rewards = {}

        for agent_id in self.agent_states.keys():
            # Simple credit assignment: equal share
            # More sophisticated: difference rewards, Shapley values
            individual_rewards[agent_id] = team_reward / self.num_agents

            # Add shaped reward based on individual contribution
            if self.reward_shaping:
                action = actions.get(agent_id)
                if action and action.action_type == "gather":
                    individual_rewards[agent_id] += 0.5  # Bonus for gathering
                elif action and action.action_type == "communicate":
                    individual_rewards[agent_id] += 0.2  # Bonus for coordination

        return individual_rewards

    def _check_done(self) -> bool:
        """Check if episode is complete."""
        # Episode ends after max timesteps or goal achieved
        if self.timestep >= 1000:
            return True

        if self.shared_state.team_score >= 100.0:
            return True

        # All agents out of energy
        if all(state.energy <= 0 for state in self.agent_states.values()):
            return True

        return False


class MARLPolicy(BaseModel):
    """Policy for MARL agent."""
    agent_id: str
    policy_type: str = "neural"  # "neural", "tabular", "rule-based"
    parameters: Dict[str, Any] = Field(default_factory=dict)

    # Learning hyperparameters
    learning_rate: float = 0.001
    discount_factor: float = 0.99
    exploration_rate: float = 0.1


class MARLTrainer:
    """Trainer for cooperative MARL agents."""

    def __init__(
        self,
        environment: CooperativeEnvironment,
        policies: List[MARLPolicy],
        llm_client: Any = None
    ):
        """
        Initialize MARL trainer.

        Args:
            environment: Cooperative environment
            policies: List of agent policies
            llm_client: Optional LLM for high-level reasoning
        """
        self.env = environment
        self.policies = {p.agent_id: p for p in policies}
        self.llm_client = llm_client

        # Training statistics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []

    async def train(
        self,
        num_episodes: int = 1000,
        max_steps: int = 500
    ) -> Dict[str, Any]:
        """
        Train agents using cooperative MARL.

        Args:
            num_episodes: Number of training episodes
            max_steps: Max steps per episode

        Returns:
            Training statistics
        """
        for episode in range(num_episodes):
            states = await self.env.reset()
            episode_reward = 0.0

            for step in range(max_steps):
                # Select actions for all agents
                actions = await self._select_actions(states)

                # Execute actions
                next_states, rewards, done, info = await self.env.step(actions)

                # Update policies
                await self._update_policies(states, actions, rewards, next_states)

                # Accumulate rewards
                total_step_reward = sum(r.total_reward for r in rewards.values())
                episode_reward += total_step_reward

                states = next_states

                if done:
                    break

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(step + 1)

            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")

        return {
            "total_episodes": num_episodes,
            "final_avg_reward": np.mean(self.episode_rewards[-100:]),
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths
        }

    async def _select_actions(
        self,
        states: Dict[str, AgentState]
    ) -> Dict[str, Action]:
        """Select actions for all agents using their policies."""
        actions = {}

        for agent_id, state in states.items():
            policy = self.policies[agent_id]

            # Epsilon-greedy exploration
            if np.random.random() < policy.exploration_rate:
                # Random action
                action_type = np.random.choice(["move", "gather", "communicate"])
            else:
                # Policy action (simplified - would use neural network)
                action_type = "gather" if state.energy > 0.5 else "move"

            actions[agent_id] = Action(
                agent_id=agent_id,
                action_type=action_type,
                parameters={
                    "direction": [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
                }
            )

        return actions

    async def _update_policies(
        self,
        states: Dict[str, AgentState],
        actions: Dict[str, Action],
        rewards: Dict[str, Reward],
        next_states: Dict[str, AgentState]
    ) -> None:
        """Update agent policies based on experience."""
        # This is simplified - would implement actual RL updates
        # (Q-learning, PPO, MADDPG, etc.)

        for agent_id in self.policies.keys():
            policy = self.policies[agent_id]

            # Decay exploration rate
            policy.exploration_rate *= 0.995
            policy.exploration_rate = max(0.01, policy.exploration_rate)
```

### 1.2 Competitive MARL

**File**: `ia_modules/agents/marl/competitive.py`

```python
"""Competitive Multi-Agent Reinforcement Learning."""
from typing import List, Dict, Any, Tuple
import numpy as np
from .cooperative import AgentState, Action, Reward, MARLPolicy


class CompetitiveEnvironment:
    """
    Competitive multi-agent environment (zero-sum games).

    Agents compete for limited resources or objectives.
    One agent's gain is another's loss.
    """

    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        action_dim: int,
        team_based: bool = False
    ):
        """
        Initialize competitive environment.

        Args:
            num_agents: Number of competing agents
            state_dim: State space dimension
            action_dim: Action space dimension
            team_based: If True, agents form competing teams
        """
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.team_based = team_based

        # Agent states and scores
        self.agent_states: Dict[str, AgentState] = {}
        self.agent_scores: Dict[str, float] = {}
        self.timestep = 0

    async def step(
        self,
        actions: Dict[str, Action]
    ) -> Tuple[Dict[str, AgentState], Dict[str, Reward], bool, Dict[str, Any]]:
        """Execute competitive actions."""
        # Execute actions
        next_states = await self._execute_competitive_actions(actions)

        # Calculate competitive rewards (zero-sum)
        rewards = self._calculate_competitive_rewards(actions)

        # Check for winner
        done = self._check_winner()

        info = {
            "timestep": self.timestep,
            "scores": self.agent_scores,
            "leader": max(self.agent_scores, key=self.agent_scores.get)
        }

        self.timestep += 1

        return next_states, rewards, done, info

    async def _execute_competitive_actions(
        self,
        actions: Dict[str, Action]
    ) -> Dict[str, AgentState]:
        """Execute actions with competition mechanics."""
        # Agents compete for resources
        for agent_id, action in actions.items():
            if action.action_type == "claim_resource":
                # Check if resource is contested
                contested_by = [
                    aid for aid, act in actions.items()
                    if act.action_type == "claim_resource" and
                    act.parameters.get("resource_id") == action.parameters.get("resource_id")
                ]

                if len(contested_by) == 1:
                    # Uncontested - agent gets full reward
                    self.agent_scores[agent_id] += 1.0
                else:
                    # Contested - split or strongest wins
                    self.agent_scores[agent_id] += 1.0 / len(contested_by)

        return self.agent_states

    def _calculate_competitive_rewards(
        self,
        actions: Dict[str, Action]
    ) -> Dict[str, Reward]:
        """Calculate zero-sum rewards."""
        rewards = {}

        # Zero-sum: total rewards sum to zero
        mean_score = np.mean(list(self.agent_scores.values()))

        for agent_id, score in self.agent_scores.items():
            # Relative reward (above or below average)
            reward_value = score - mean_score

            rewards[agent_id] = Reward(
                agent_id=agent_id,
                individual_reward=reward_value,
                team_reward=0.0,
                total_reward=reward_value
            )

        return rewards

    def _check_winner(self) -> bool:
        """Check if there's a winner."""
        if self.timestep >= 1000:
            return True

        # Winner reaches threshold
        if max(self.agent_scores.values()) >= 100.0:
            return True

        return False


class SelfPlayTrainer:
    """Self-play training for competitive agents."""

    def __init__(
        self,
        environment: CompetitiveEnvironment,
        policy: MARLPolicy
    ):
        """
        Initialize self-play trainer.

        Agents play against copies of themselves to improve.

        Args:
            environment: Competitive environment
            policy: Policy to train via self-play
        """
        self.env = environment
        self.policy = policy
        self.policy_versions: List[MARLPolicy] = [policy]

    async def train_self_play(
        self,
        num_iterations: int = 100,
        games_per_iteration: int = 10
    ) -> Dict[str, Any]:
        """
        Train via self-play.

        Args:
            num_iterations: Number of training iterations
            games_per_iteration: Games per iteration

        Returns:
            Training results
        """
        for iteration in range(num_iterations):
            # Play games against past versions
            for game in range(games_per_iteration):
                # Select opponent from past versions
                opponent_idx = np.random.randint(0, len(self.policy_versions))
                opponent = self.policy_versions[opponent_idx]

                # Play game
                await self._play_game(self.policy, opponent)

            # Save current policy version
            if iteration % 10 == 0:
                import copy
                self.policy_versions.append(copy.deepcopy(self.policy))

            print(f"Self-play iteration {iteration} complete")

        return {
            "iterations": num_iterations,
            "policy_versions": len(self.policy_versions)
        }

    async def _play_game(
        self,
        policy1: MARLPolicy,
        policy2: MARLPolicy
    ) -> Dict[str, Any]:
        """Play one game between two policies."""
        # Simplified game loop
        states = await self.env.reset()

        for step in range(100):
            # Both agents act
            actions = {
                "agent_0": await self._select_action(policy1, states["agent_0"]),
                "agent_1": await self._select_action(policy2, states["agent_1"])
            }

            next_states, rewards, done, info = await self.env.step(actions)

            states = next_states

            if done:
                break

        return info

    async def _select_action(
        self,
        policy: MARLPolicy,
        state: AgentState
    ) -> Action:
        """Select action using policy."""
        # Simplified action selection
        return Action(
            agent_id=state.agent_id,
            action_type="claim_resource",
            parameters={"resource_id": np.random.randint(0, 10)}
        )
```

---

## 2. Advanced Memory Systems

### 2.1 Episodic Memory

**File**: `ia_modules/agents/memory/episodic.py`

```python
"""Episodic memory system for agents."""
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class Episode(BaseModel):
    """Single episode in agent's memory."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    timestamp: datetime = Field(default_factory=datetime.now)

    # Episode content
    state: Dict[str, Any] = Field(default_factory=dict)
    action: Dict[str, Any] = Field(default_factory=dict)
    observation: Dict[str, Any] = Field(default_factory=dict)
    outcome: Dict[str, Any] = Field(default_factory=dict)

    # Episode metadata
    importance: float = 0.5  # 0-1 scale
    emotional_valence: float = 0.0  # -1 (negative) to +1 (positive)
    tags: List[str] = Field(default_factory=list)
    related_episodes: List[str] = Field(default_factory=list)

    # Retrieval metadata
    access_count: int = 0
    last_accessed: Optional[datetime] = None


class EpisodicMemory:
    """
    Episodic memory system storing specific events.

    Implements:
    - Temporal ordering
    - Importance weighting
    - Forgetting curve
    - Episodic retrieval by similarity
    """

    def __init__(
        self,
        agent_id: str,
        capacity: int = 10000,
        decay_rate: float = 0.99
    ):
        """
        Initialize episodic memory.

        Args:
            agent_id: Agent identifier
            capacity: Max episodes to store
            decay_rate: Memory decay rate over time
        """
        self.agent_id = agent_id
        self.capacity = capacity
        self.decay_rate = decay_rate

        self.episodes: List[Episode] = []
        self.episode_index: Dict[str, Episode] = {}

    async def store_episode(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        observation: Dict[str, Any],
        outcome: Dict[str, Any],
        importance: float = 0.5,
        tags: List[str] = None
    ) -> str:
        """
        Store new episode in memory.

        Args:
            state: Agent state when episode occurred
            action: Action taken
            observation: Observation received
            outcome: Outcome of action
            importance: Episode importance (0-1)
            tags: Tags for categorization

        Returns:
            Episode ID
        """
        episode = Episode(
            agent_id=self.agent_id,
            state=state,
            action=action,
            observation=observation,
            outcome=outcome,
            importance=importance,
            tags=tags or []
        )

        # Add to memory
        self.episodes.append(episode)
        self.episode_index[episode.id] = episode

        # Enforce capacity limit
        if len(self.episodes) > self.capacity:
            await self._forget_least_important()

        return episode.id

    async def retrieve_recent(
        self,
        n: int = 10,
        min_importance: float = 0.0
    ) -> List[Episode]:
        """
        Retrieve most recent episodes.

        Args:
            n: Number of episodes to retrieve
            min_importance: Minimum importance threshold

        Returns:
            List of recent episodes
        """
        # Filter by importance
        filtered = [
            ep for ep in self.episodes
            if ep.importance >= min_importance
        ]

        # Sort by recency
        sorted_episodes = sorted(
            filtered,
            key=lambda ep: ep.timestamp,
            reverse=True
        )

        # Update access metadata
        for ep in sorted_episodes[:n]:
            ep.access_count += 1
            ep.last_accessed = datetime.now()

        return sorted_episodes[:n]

    async def retrieve_by_tags(
        self,
        tags: List[str],
        n: int = 10
    ) -> List[Episode]:
        """
        Retrieve episodes matching tags.

        Args:
            tags: Tags to match
            n: Max episodes to return

        Returns:
            Matching episodes
        """
        matching = [
            ep for ep in self.episodes
            if any(tag in ep.tags for tag in tags)
        ]

        # Sort by importance and recency
        sorted_matching = sorted(
            matching,
            key=lambda ep: (ep.importance, ep.timestamp),
            reverse=True
        )

        return sorted_matching[:n]

    async def retrieve_similar(
        self,
        query_state: Dict[str, Any],
        n: int = 10
    ) -> List[Episode]:
        """
        Retrieve episodes similar to query state.

        Args:
            query_state: State to match against
            n: Number of episodes to return

        Returns:
            Similar episodes
        """
        # Calculate similarity for each episode
        similarities = []

        for ep in self.episodes:
            similarity = self._calculate_similarity(query_state, ep.state)
            similarities.append((similarity, ep))

        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)

        return [ep for _, ep in similarities[:n]]

    def _calculate_similarity(
        self,
        state1: Dict[str, Any],
        state2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two states."""
        # Simplified similarity - count matching keys
        common_keys = set(state1.keys()) & set(state2.keys())

        if not common_keys:
            return 0.0

        matches = sum(
            1 for key in common_keys
            if state1[key] == state2[key]
        )

        return matches / len(common_keys)

    async def _forget_least_important(self) -> None:
        """Remove least important episodes when capacity exceeded."""
        # Calculate forgetting score (importance * recency * access)
        scored_episodes = []

        now = datetime.now()

        for ep in self.episodes:
            recency = (now - ep.timestamp).total_seconds() / 86400  # Days
            recency_score = self.decay_rate ** recency

            access_score = min(1.0, ep.access_count / 10.0)

            forgetting_score = ep.importance * recency_score * (1 + access_score)

            scored_episodes.append((forgetting_score, ep))

        # Sort by score (lowest first)
        scored_episodes.sort(key=lambda x: x[0])

        # Remove lowest scored episode
        to_remove = scored_episodes[0][1]
        self.episodes.remove(to_remove)
        del self.episode_index[to_remove.id]

    async def consolidate_memories(self) -> List[str]:
        """
        Consolidate related episodes into higher-level memories.

        Returns:
            IDs of consolidated episodes
        """
        # Group similar episodes
        clusters = await self._cluster_episodes()

        consolidated_ids = []

        for cluster in clusters:
            if len(cluster) >= 3:
                # Create consolidated memory
                # Mark episodes as related
                episode_ids = [ep.id for ep in cluster]

                for ep in cluster:
                    ep.related_episodes.extend([
                        eid for eid in episode_ids if eid != ep.id
                    ])

                consolidated_ids.extend(episode_ids)

        return consolidated_ids

    async def _cluster_episodes(self) -> List[List[Episode]]:
        """Cluster similar episodes together."""
        # Simplified clustering
        clusters = []
        used = set()

        for i, ep1 in enumerate(self.episodes):
            if ep1.id in used:
                continue

            cluster = [ep1]
            used.add(ep1.id)

            for ep2 in self.episodes[i+1:]:
                if ep2.id in used:
                    continue

                similarity = self._calculate_similarity(ep1.state, ep2.state)

                if similarity > 0.7:
                    cluster.append(ep2)
                    used.add(ep2.id)

            if len(cluster) > 1:
                clusters.append(cluster)

        return clusters
```

### 2.2 Semantic Memory

**File**: `ia_modules/agents/memory/semantic.py`

```python
"""Semantic memory system for agents."""
from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class Concept(BaseModel):
    """Semantic concept representation."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    category: str

    # Concept properties
    attributes: Dict[str, Any] = Field(default_factory=dict)
    description: str = ""

    # Relations to other concepts
    is_a: List[str] = Field(default_factory=list)  # Inheritance
    part_of: List[str] = Field(default_factory=list)  # Composition
    related_to: List[str] = Field(default_factory=list)  # Association

    # Metadata
    confidence: float = 1.0
    source: str = "learned"  # "learned", "given", "inferred"
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class SemanticMemory:
    """
    Semantic memory for factual knowledge.

    Stores:
    - Concepts and their relationships
    - Facts and rules
    - Generalizations and abstractions
    """

    def __init__(self, agent_id: str):
        """Initialize semantic memory."""
        self.agent_id = agent_id
        self.concepts: Dict[str, Concept] = {}
        self.facts: List[Dict[str, Any]] = []

    async def add_concept(
        self,
        name: str,
        category: str,
        attributes: Dict[str, Any] = None,
        description: str = ""
    ) -> str:
        """
        Add new concept to semantic memory.

        Args:
            name: Concept name
            category: Concept category
            attributes: Concept attributes
            description: Text description

        Returns:
            Concept ID
        """
        concept = Concept(
            name=name,
            category=category,
            attributes=attributes or {},
            description=description
        )

        self.concepts[concept.id] = concept

        return concept.id

    async def add_relationship(
        self,
        concept1_id: str,
        relationship_type: str,
        concept2_id: str
    ) -> None:
        """
        Add relationship between concepts.

        Args:
            concept1_id: Source concept ID
            relationship_type: "is_a", "part_of", "related_to"
            concept2_id: Target concept ID
        """
        concept1 = self.concepts.get(concept1_id)

        if not concept1:
            raise ValueError(f"Concept {concept1_id} not found")

        if relationship_type == "is_a":
            concept1.is_a.append(concept2_id)
        elif relationship_type == "part_of":
            concept1.part_of.append(concept2_id)
        elif relationship_type == "related_to":
            concept1.related_to.append(concept2_id)

        concept1.updated_at = datetime.now()

    async def query_concepts(
        self,
        category: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> List[Concept]:
        """
        Query concepts by category or attributes.

        Args:
            category: Filter by category
            attributes: Filter by attributes

        Returns:
            Matching concepts
        """
        results = list(self.concepts.values())

        if category:
            results = [c for c in results if c.category == category]

        if attributes:
            results = [
                c for c in results
                if all(
                    c.attributes.get(k) == v
                    for k, v in attributes.items()
                )
            ]

        return results

    async def get_related_concepts(
        self,
        concept_id: str,
        max_depth: int = 2
    ) -> Set[str]:
        """
        Get all concepts related to given concept.

        Args:
            concept_id: Starting concept ID
            max_depth: Maximum traversal depth

        Returns:
            Set of related concept IDs
        """
        related = set()
        to_visit = [(concept_id, 0)]
        visited = set()

        while to_visit:
            current_id, depth = to_visit.pop(0)

            if current_id in visited or depth > max_depth:
                continue

            visited.add(current_id)
            related.add(current_id)

            concept = self.concepts.get(current_id)
            if not concept:
                continue

            # Add related concepts to queue
            for rel_id in concept.is_a + concept.part_of + concept.related_to:
                if rel_id not in visited:
                    to_visit.append((rel_id, depth + 1))

        related.discard(concept_id)  # Remove starting concept
        return related

    async def infer_relationships(self) -> int:
        """
        Infer new relationships from existing knowledge.

        Returns:
            Number of new relationships inferred
        """
        inferred_count = 0

        # Transitivity: if A is_a B and B is_a C, then A is_a C
        for concept in self.concepts.values():
            for parent_id in list(concept.is_a):
                parent = self.concepts.get(parent_id)
                if not parent:
                    continue

                for grandparent_id in parent.is_a:
                    if grandparent_id not in concept.is_a:
                        concept.is_a.append(grandparent_id)
                        inferred_count += 1

        return inferred_count
```

### 2.3 Procedural Memory

**File**: `ia_modules/agents/memory/procedural.py`

```python
"""Procedural memory for skills and procedures."""
from typing import List, Dict, Any, Optional, Callable, Awaitable
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class Skill(BaseModel):
    """Learned skill or procedure."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str = ""

    # Skill components
    preconditions: List[str] = Field(default_factory=list)
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    postconditions: List[str] = Field(default_factory=list)

    # Skill metadata
    proficiency: float = 0.0  # 0-1 scale
    success_rate: float = 0.0
    times_executed: int = 0
    last_executed: Optional[datetime] = None

    # Learning metadata
    learned_from: str = "experience"  # "experience", "instruction", "imitation"
    created_at: datetime = Field(default_factory=datetime.now)


class ProceduralMemory:
    """
    Procedural memory for skills and how-to knowledge.

    Stores:
    - Learned skills and procedures
    - Motor programs
    - Compiled knowledge
    """

    def __init__(self, agent_id: str):
        """Initialize procedural memory."""
        self.agent_id = agent_id
        self.skills: Dict[str, Skill] = {}

    async def add_skill(
        self,
        name: str,
        steps: List[Dict[str, Any]],
        preconditions: List[str] = None,
        postconditions: List[str] = None,
        description: str = ""
    ) -> str:
        """
        Add new skill to procedural memory.

        Args:
            name: Skill name
            steps: Procedure steps
            preconditions: Required preconditions
            postconditions: Expected postconditions
            description: Skill description

        Returns:
            Skill ID
        """
        skill = Skill(
            name=name,
            description=description,
            steps=steps,
            preconditions=preconditions or [],
            postconditions=postconditions or []
        )

        self.skills[skill.id] = skill

        return skill.id

    async def execute_skill(
        self,
        skill_id: str,
        context: Dict[str, Any],
        executor: Callable[[Dict[str, Any]], Awaitable[bool]]
    ) -> Dict[str, Any]:
        """
        Execute a learned skill.

        Args:
            skill_id: Skill to execute
            context: Execution context
            executor: Function to execute skill steps

        Returns:
            Execution results
        """
        skill = self.skills.get(skill_id)

        if not skill:
            raise ValueError(f"Skill {skill_id} not found")

        # Check preconditions
        if not await self._check_conditions(skill.preconditions, context):
            return {
                "success": False,
                "error": "Preconditions not met"
            }

        # Execute skill steps
        start_time = datetime.now()

        try:
            success = await executor({"skill": skill, "context": context})

            # Update skill statistics
            skill.times_executed += 1
            skill.last_executed = datetime.now()

            if success:
                skill.success_rate = (
                    (skill.success_rate * (skill.times_executed - 1) + 1.0) /
                    skill.times_executed
                )
                # Increase proficiency with practice
                skill.proficiency = min(1.0, skill.proficiency + 0.01)
            else:
                skill.success_rate = (
                    (skill.success_rate * (skill.times_executed - 1)) /
                    skill.times_executed
                )

            execution_time = (datetime.now() - start_time).total_seconds()

            return {
                "success": success,
                "skill_id": skill_id,
                "execution_time": execution_time,
                "proficiency": skill.proficiency,
                "success_rate": skill.success_rate
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _check_conditions(
        self,
        conditions: List[str],
        context: Dict[str, Any]
    ) -> bool:
        """Check if conditions are met in context."""
        # Simplified condition checking
        for condition in conditions:
            # Parse condition (simplified)
            if condition not in context:
                return False

        return True

    async def improve_skill(
        self,
        skill_id: str,
        feedback: Dict[str, Any]
    ) -> None:
        """
        Improve skill based on feedback.

        Args:
            skill_id: Skill to improve
            feedback: Performance feedback
        """
        skill = self.skills.get(skill_id)

        if not skill:
            return

        # Adjust skill based on feedback
        if feedback.get("success"):
            # Reinforce successful execution
            skill.proficiency = min(1.0, skill.proficiency + 0.05)
        else:
            # Learn from failure
            error_type = feedback.get("error_type")

            if error_type == "missing_step":
                # Add missing step
                new_step = feedback.get("suggested_step")
                if new_step:
                    skill.steps.append(new_step)

            elif error_type == "wrong_order":
                # Reorder steps based on feedback
                correct_order = feedback.get("correct_order")
                if correct_order:
                    skill.steps = [skill.steps[i] for i in correct_order]
```

---

## 3. Temporal Knowledge Graphs

### 3.1 Temporal Knowledge Graph Implementation

**File**: `ia_modules/agents/memory/temporal_kg.py`

```python
"""Temporal Knowledge Graph for agent memory."""
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import uuid


class TemporalNode(BaseModel):
    """Node in temporal knowledge graph."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str  # "entity", "event", "concept", "agent"
    label: str
    properties: Dict[str, Any] = Field(default_factory=dict)

    # Temporal validity
    valid_from: datetime = Field(default_factory=datetime.now)
    valid_to: Optional[datetime] = None

    # Bi-temporal tracking
    transaction_time: datetime = Field(default_factory=datetime.now)


class TemporalEdge(BaseModel):
    """Edge in temporal knowledge graph."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str
    target_id: str
    relationship: str
    properties: Dict[str, Any] = Field(default_factory=dict)

    # Temporal validity
    valid_from: datetime = Field(default_factory=datetime.now)
    valid_to: Optional[datetime] = None

    # Bi-temporal tracking
    transaction_time: datetime = Field(default_factory=datetime.now)

    # Edge metadata
    confidence: float = 1.0
    source: str = "observed"


class TemporalKnowledgeGraph:
    """
    Temporal Knowledge Graph for agent memory.

    Features:
    - Bi-temporal model (valid time + transaction time)
    - Time-aware querying
    - Knowledge evolution tracking
    - Episodic and semantic subgraphs
    """

    def __init__(self, agent_id: str):
        """Initialize temporal knowledge graph."""
        self.agent_id = agent_id
        self.nodes: Dict[str, TemporalNode] = {}
        self.edges: Dict[str, TemporalEdge] = {}

        # Indexes for fast retrieval
        self.node_by_type: Dict[str, List[str]] = {}
        self.edges_from: Dict[str, List[str]] = {}
        self.edges_to: Dict[str, List[str]] = {}

    async def add_node(
        self,
        node_type: str,
        label: str,
        properties: Dict[str, Any] = None,
        valid_from: Optional[datetime] = None
    ) -> str:
        """
        Add node to temporal knowledge graph.

        Args:
            node_type: Type of node
            label: Node label
            properties: Node properties
            valid_from: Validity start time

        Returns:
            Node ID
        """
        node = TemporalNode(
            type=node_type,
            label=label,
            properties=properties or {},
            valid_from=valid_from or datetime.now()
        )

        self.nodes[node.id] = node

        # Update indexes
        if node_type not in self.node_by_type:
            self.node_by_type[node_type] = []
        self.node_by_type[node_type].append(node.id)

        return node.id

    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        relationship: str,
        properties: Dict[str, Any] = None,
        confidence: float = 1.0,
        valid_from: Optional[datetime] = None
    ) -> str:
        """
        Add edge to temporal knowledge graph.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            relationship: Relationship type
            properties: Edge properties
            confidence: Confidence in relationship (0-1)
            valid_from: Validity start time

        Returns:
            Edge ID
        """
        edge = TemporalEdge(
            source_id=source_id,
            target_id=target_id,
            relationship=relationship,
            properties=properties or {},
            confidence=confidence,
            valid_from=valid_from or datetime.now()
        )

        self.edges[edge.id] = edge

        # Update indexes
        if source_id not in self.edges_from:
            self.edges_from[source_id] = []
        self.edges_from[source_id].append(edge.id)

        if target_id not in self.edges_to:
            self.edges_to[target_id] = []
        self.edges_to[target_id].append(edge.id)

        return edge.id

    async def query_at_time(
        self,
        query_time: datetime,
        node_type: Optional[str] = None
    ) -> List[TemporalNode]:
        """
        Query graph state at specific time.

        Args:
            query_time: Time to query
            node_type: Optional node type filter

        Returns:
            Nodes valid at query time
        """
        valid_nodes = []

        nodes_to_check = self.nodes.values()
        if node_type:
            node_ids = self.node_by_type.get(node_type, [])
            nodes_to_check = [self.nodes[nid] for nid in node_ids]

        for node in nodes_to_check:
            if self._is_valid_at(node, query_time):
                valid_nodes.append(node)

        return valid_nodes

    def _is_valid_at(
        self,
        temporal_obj: Any,
        query_time: datetime
    ) -> bool:
        """Check if node/edge is valid at given time."""
        if query_time < temporal_obj.valid_from:
            return False

        if temporal_obj.valid_to and query_time > temporal_obj.valid_to:
            return False

        return True

    async def get_relationships(
        self,
        node_id: str,
        relationship_type: Optional[str] = None,
        direction: str = "outgoing",
        at_time: Optional[datetime] = None
    ) -> List[Tuple[TemporalEdge, TemporalNode]]:
        """
        Get relationships for a node.

        Args:
            node_id: Node to get relationships for
            relationship_type: Optional relationship filter
            direction: "outgoing", "incoming", or "both"
            at_time: Optional time point for temporal query

        Returns:
            List of (edge, connected_node) tuples
        """
        results = []
        query_time = at_time or datetime.now()

        # Outgoing edges
        if direction in ["outgoing", "both"]:
            edge_ids = self.edges_from.get(node_id, [])

            for edge_id in edge_ids:
                edge = self.edges[edge_id]

                if not self._is_valid_at(edge, query_time):
                    continue

                if relationship_type and edge.relationship != relationship_type:
                    continue

                target_node = self.nodes[edge.target_id]

                if self._is_valid_at(target_node, query_time):
                    results.append((edge, target_node))

        # Incoming edges
        if direction in ["incoming", "both"]:
            edge_ids = self.edges_to.get(node_id, [])

            for edge_id in edge_ids:
                edge = self.edges[edge_id]

                if not self._is_valid_at(edge, query_time):
                    continue

                if relationship_type and edge.relationship != relationship_type:
                    continue

                source_node = self.nodes[edge.source_id]

                if self._is_valid_at(source_node, query_time):
                    results.append((edge, source_node))

        return results

    async def invalidate_node(
        self,
        node_id: str,
        valid_to: Optional[datetime] = None
    ) -> None:
        """
        Mark node as no longer valid.

        Args:
            node_id: Node to invalidate
            valid_to: Validity end time (default: now)
        """
        node = self.nodes.get(node_id)

        if node:
            node.valid_to = valid_to or datetime.now()

    async def query_temporal_patterns(
        self,
        start_time: datetime,
        end_time: datetime,
        pattern: List[Dict[str, Any]]
    ) -> List[List[TemporalNode]]:
        """
        Query for temporal patterns in the graph.

        Args:
            start_time: Pattern window start
            end_time: Pattern window end
            pattern: Pattern to match

        Returns:
            List of node sequences matching pattern
        """
        # Simplified pattern matching
        matches = []

        # Get all events in time window
        events = []
        for node in self.nodes.values():
            if node.type == "event":
                if start_time <= node.valid_from <= end_time:
                    events.append(node)

        # Sort by time
        events.sort(key=lambda n: n.valid_from)

        # Match pattern (simplified)
        # Real implementation would do proper pattern matching
        if len(events) >= len(pattern):
            matches.append(events[:len(pattern)])

        return matches

    async def get_evolution_history(
        self,
        entity_id: str
    ) -> List[Tuple[datetime, str, Any]]:
        """
        Get evolution history of an entity.

        Args:
            entity_id: Entity to track

        Returns:
            List of (time, property, value) changes
        """
        history = []

        node = self.nodes.get(entity_id)
        if not node:
            return history

        # In a real implementation, we'd track property changes over time
        # This is simplified - would need versioning system

        history.append((
            node.transaction_time,
            "created",
            node.properties
        ))

        return history
```

Continue to Part 2...
