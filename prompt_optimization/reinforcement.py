"""
Reinforcement learning optimizer for prompt selection.
"""

import asyncio
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .optimizer import OptimizationStrategy, PromptOptimizer, OptimizationResult
from .evaluators import PromptEvaluator


@dataclass
class RLConfig:
    """Configuration for reinforcement learning optimization."""

    learning_rate: float = 0.1
    discount_factor: float = 0.95
    epsilon: float = 0.1  # Exploration rate
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.01
    max_episodes: int = 100
    max_steps_per_episode: int = 10
    convergence_threshold: float = 0.001

    def validate(self):
        """Validate configuration parameters."""
        if not 0 <= self.learning_rate <= 1:
            raise ValueError("learning_rate must be between 0 and 1")
        if not 0 <= self.discount_factor <= 1:
            raise ValueError("discount_factor must be between 0 and 1")
        if not 0 <= self.epsilon <= 1:
            raise ValueError("epsilon must be between 0 and 1")
        if not 0 <= self.min_epsilon <= self.epsilon:
            raise ValueError("min_epsilon must be between 0 and epsilon")


class PromptAction:
    """Represents an action (modification) that can be applied to a prompt."""

    def __init__(self, name: str, apply_fn: Callable[[str], str]):
        """
        Initialize prompt action.

        Args:
            name: Action name
            apply_fn: Function to apply action to a prompt
        """
        self.name = name
        self.apply_fn = apply_fn

    def apply(self, prompt: str) -> str:
        """Apply this action to a prompt."""
        return self.apply_fn(prompt)

    def __repr__(self) -> str:
        return f"PromptAction({self.name})"

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other) -> bool:
        if not isinstance(other, PromptAction):
            return False
        return self.name == other.name


class RLOptimizer(PromptOptimizer):
    """
    Reinforcement learning optimizer using Q-learning.

    Learns optimal prompt modifications through trial and error,
    using Q-values to estimate the expected reward of actions.
    """

    def __init__(
        self,
        evaluator: PromptEvaluator,
        actions: Optional[List[PromptAction]] = None,
        config: Optional[RLConfig] = None,
        verbose: bool = False,
    ):
        """
        Initialize RL optimizer.

        Args:
            evaluator: Evaluator to score prompts
            actions: List of available prompt actions
            config: RL configuration
            verbose: Print progress
        """
        self.config = config or RLConfig()
        self.config.validate()

        super().__init__(
            evaluator,
            max_iterations=self.config.max_episodes,
            convergence_threshold=self.config.convergence_threshold,
            verbose=verbose,
        )

        self.actions = actions or self._default_actions()
        self.q_table: Dict[str, Dict[PromptAction, float]] = defaultdict(
            lambda: {action: 0.0 for action in self.actions}
        )

        self.current_epsilon = self.config.epsilon
        self.episode_rewards: List[float] = []

    def get_strategy(self) -> OptimizationStrategy:
        """Get the optimization strategy."""
        return OptimizationStrategy.REINFORCEMENT_LEARNING

    def _default_actions(self) -> List[PromptAction]:
        """
        Create default set of prompt modification actions.

        Returns:
            List of PromptAction objects
        """
        return [
            PromptAction(
                "add_please",
                lambda p: f"Please {p}" if not p.lower().startswith("please") else p
            ),
            PromptAction(
                "add_detail",
                lambda p: f"{p} Provide detailed information."
            ),
            PromptAction(
                "add_concise",
                lambda p: f"{p} Be concise."
            ),
            PromptAction(
                "add_step_by_step",
                lambda p: f"{p} Explain step by step."
            ),
            PromptAction(
                "add_examples",
                lambda p: f"{p} Include examples."
            ),
            PromptAction(
                "make_question",
                lambda p: f"{p.rstrip('.')}?" if not p.endswith("?") else p
            ),
            PromptAction(
                "add_context",
                lambda p: f"Given the context, {p.lower()}"
            ),
            PromptAction(
                "add_format",
                lambda p: f"{p} Format your response clearly."
            ),
        ]

    def _get_state_key(self, prompt: str) -> str:
        """
        Convert prompt to state key for Q-table.

        Uses a simplified representation to handle state space.

        Args:
            prompt: Prompt string

        Returns:
            State key
        """
        # Simple state representation based on prompt features
        length_category = "short" if len(prompt) < 50 else "medium" if len(prompt) < 150 else "long"
        has_question = "question" if "?" in prompt else "statement"
        has_please = "polite" if "please" in prompt.lower() else "direct"

        return f"{length_category}_{has_question}_{has_please}"

    def _select_action(self, state: str) -> PromptAction:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state

        Returns:
            Selected action
        """
        if random.random() < self.current_epsilon:
            # Exploration: random action
            return random.choice(self.actions)
        else:
            # Exploitation: best action
            q_values = self.q_table[state]
            max_q = max(q_values.values())
            best_actions = [a for a, q in q_values.items() if q == max_q]
            return random.choice(best_actions)

    def _update_q_value(
        self,
        state: str,
        action: PromptAction,
        reward: float,
        next_state: str,
    ):
        """
        Update Q-value using Q-learning update rule.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())

        # Q-learning update
        new_q = current_q + self.config.learning_rate * (
            reward + self.config.discount_factor * max_next_q - current_q
        )

        self.q_table[state][action] = new_q

    async def _run_episode(
        self,
        initial_prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Run one episode of RL optimization.

        Args:
            initial_prompt: Starting prompt
            context: Evaluation context

        Returns:
            Total episode reward
        """
        current_prompt = initial_prompt
        current_score = await self.evaluate_prompt(current_prompt, context)
        total_reward = 0.0

        for step in range(self.config.max_steps_per_episode):
            # Get state and select action
            state = self._get_state_key(current_prompt)
            action = self._select_action(state)

            # Apply action
            next_prompt = action.apply(current_prompt)
            next_score = await self.evaluate_prompt(next_prompt, context)

            # Calculate reward (improvement in score)
            reward = next_score - current_score
            total_reward += reward

            # Get next state
            next_state = self._get_state_key(next_prompt)

            # Update Q-value
            self._update_q_value(state, action, reward, next_state)

            # Track candidate
            self.track_candidate(
                next_prompt,
                next_score,
                {
                    "action": action.name,
                    "reward": reward,
                    "step": step,
                }
            )

            # Move to next state
            current_prompt = next_prompt
            current_score = next_score

        return total_reward

    def _decay_epsilon(self):
        """Decay exploration rate."""
        self.current_epsilon = max(
            self.config.min_epsilon,
            self.current_epsilon * self.config.epsilon_decay
        )

    async def optimize(
        self,
        initial_prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        """
        Optimize prompt using reinforcement learning.

        Args:
            initial_prompt: Starting prompt
            context: Additional context

        Returns:
            OptimizationResult
        """
        self.reset()
        self.current_epsilon = self.config.epsilon
        self.episode_rewards = []

        # Evaluate initial prompt
        initial_score = await self.evaluate_prompt(initial_prompt, context)
        self.track_candidate(initial_prompt, initial_score, {"type": "initial"})

        if self.verbose:
            print(f"Initial score: {initial_score:.4f}")

        # Run episodes
        for episode in range(self.config.max_episodes):
            self.current_iteration = episode + 1

            episode_reward = await self._run_episode(initial_prompt, context)
            self.episode_rewards.append(episode_reward)

            if self.verbose:
                print(
                    f"Episode {episode + 1}: "
                    f"reward = {episode_reward:.4f}, "
                    f"epsilon = {self.current_epsilon:.4f}, "
                    f"best_score = {self.best_candidate.score:.4f}"
                )

            # Decay exploration
            self._decay_epsilon()

            # Check convergence
            if self.has_converged(window_size=10):
                if self.verbose:
                    print(f"Converged at episode {self.current_iteration}")
                break

        result = self.get_optimization_result()
        result.metadata["q_table_size"] = len(self.q_table)
        result.metadata["final_epsilon"] = self.current_epsilon
        result.metadata["total_episodes"] = len(self.episode_rewards)
        result.metadata["average_reward"] = sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0

        return result

    def get_action_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics about learned action values.

        Returns:
            Dictionary mapping state keys to action statistics
        """
        stats = {}

        for state, q_values in self.q_table.items():
            stats[state] = {
                action.name: q_value
                for action, q_value in q_values.items()
            }

        return stats

    def get_best_action_per_state(self) -> Dict[str, str]:
        """
        Get the best action for each learned state.

        Returns:
            Dictionary mapping state keys to best action names
        """
        best_actions = {}

        for state, q_values in self.q_table.items():
            best_action = max(q_values.items(), key=lambda x: x[1])
            best_actions[state] = best_action[0].name

        return best_actions
