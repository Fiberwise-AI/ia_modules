"""
Debate collaboration pattern (adversarial).

Implements an adversarial structure where agents argue different perspectives
and refine their positions through structured debate.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import logging

from ..base_agent import BaseCollaborativeAgent
from ..communication import MessageBus, MessageType, AgentMessage
from ..state import StateManager
from ..core import AgentRole


class DebateRole(Enum):
    """Roles in a debate."""
    PROPONENT = "proponent"  # Argues for a position
    OPPONENT = "opponent"  # Argues against a position
    MODERATOR = "moderator"  # Facilitates debate
    JUDGE = "judge"  # Evaluates arguments


class DebateCollaboration:
    """
    Debate collaboration pattern with adversarial argumentation.

    Agents engage in structured debate:
    - Proponents argue for positions
    - Opponents argue against positions
    - Multiple rounds of argumentation
    - Moderator ensures fair process
    - Judge evaluates arguments
    - Final synthesis of best arguments

    This pattern is useful for:
    - Exploring multiple perspectives
    - Identifying weaknesses in reasoning
    - Stress-testing ideas
    - Critical analysis
    - Decision making under uncertainty

    Example:
        >>> # Setup
        >>> state = StateManager(thread_id="debate_001")
        >>> bus = MessageBus()
        >>>
        >>> # Create debate agents
        >>> proponent = DebateAgent(
        ...     role=AgentRole(name="proponent", description="Argues for position"),
        ...     state_manager=state,
        ...     message_bus=bus,
        ...     debate_role=DebateRole.PROPONENT
        ... )
        >>>
        >>> opponent = DebateAgent(
        ...     role=AgentRole(name="opponent", description="Argues against"),
        ...     state_manager=state,
        ...     message_bus=bus,
        ...     debate_role=DebateRole.OPPONENT
        ... )
        >>>
        >>> moderator = ModeratorAgent(
        ...     role=AgentRole(name="moderator", description="Moderates debate"),
        ...     state_manager=state,
        ...     message_bus=bus
        ... )
        >>>
        >>> # Setup debate
        >>> debate = DebateCollaboration(
        ...     proponents=[proponent],
        ...     opponents=[opponent],
        ...     moderator=moderator,
        ...     message_bus=bus,
        ...     state_manager=state
        ... )
        >>> await debate.initialize()
        >>>
        >>> # Run debate
        >>> result = await debate.execute({
        ...     "topic": "AI systems should be open source",
        ...     "rounds": 3
        ... })
    """

    def __init__(self,
                 proponents: List[BaseCollaborativeAgent],
                 opponents: List[BaseCollaborativeAgent],
                 moderator: BaseCollaborativeAgent,
                 message_bus: MessageBus,
                 state_manager: StateManager,
                 judge: Optional[BaseCollaborativeAgent] = None):
        """
        Initialize debate collaboration.

        Args:
            proponents: Agents arguing for position
            opponents: Agents arguing against position
            moderator: Agent moderating debate
            message_bus: Message bus for communication
            state_manager: Shared state manager
            judge: Optional judge to evaluate arguments
        """
        self.proponents = proponents
        self.opponents = opponents
        self.moderator = moderator
        self.judge = judge
        self.message_bus = message_bus
        self.state = state_manager
        self.logger = logging.getLogger("DebateCollaboration")

        # Track debate state
        self.debate_history: List[Dict[str, Any]] = []

    async def initialize(self) -> None:
        """Initialize all debate participants."""
        # Initialize moderator
        await self.moderator.initialize()

        # Initialize proponents
        for proponent in self.proponents:
            await proponent.initialize()

        # Initialize opponents
        for opponent in self.opponents:
            await opponent.initialize()

        # Initialize judge if present
        if self.judge:
            await self.judge.initialize()

        self.logger.info(
            f"Debate initialized: {len(self.proponents)} proponents, "
            f"{len(self.opponents)} opponents"
        )

    async def shutdown(self) -> None:
        """Shutdown all participants."""
        await self.moderator.shutdown()
        for agent in self.proponents + self.opponents:
            await agent.shutdown()
        if self.judge:
            await self.judge.shutdown()

    async def execute(self, debate_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute structured debate.

        Args:
            debate_config: Configuration including:
                - topic: Debate topic
                - rounds: Number of debate rounds
                - time_limit: Time limit per argument (optional)

        Returns:
            Debate results including all arguments and evaluation
        """
        topic = debate_config.get("topic", "")
        rounds = debate_config.get("rounds", 3)

        self.logger.info(f"Starting debate: {topic}")

        # Store debate topic
        await self.state.set("debate_topic", topic)
        await self.state.set("debate_rounds", rounds)

        # Phase 1: Opening statements
        opening_statements = await self._opening_statements(topic)

        # Phase 2: Debate rounds
        debate_rounds = []
        for round_num in range(1, rounds + 1):
            self.logger.info(f"Debate round {round_num}/{rounds}")
            round_result = await self._execute_round(round_num, topic)
            debate_rounds.append(round_result)

        # Phase 3: Closing statements
        closing_statements = await self._closing_statements(topic)

        # Phase 4: Evaluation (if judge present)
        evaluation = None
        if self.judge:
            evaluation = await self._evaluate_debate(
                opening_statements, debate_rounds, closing_statements
            )

        # Phase 5: Synthesis
        final_result = await self._synthesize_debate(
            topic, opening_statements, debate_rounds,
            closing_statements, evaluation
        )

        self.logger.info("Debate complete")

        return final_result

    async def _opening_statements(self, topic: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Collect opening statements from all participants.

        Args:
            topic: Debate topic

        Returns:
            Opening statements from proponents and opponents
        """
        self.logger.info("Collecting opening statements")

        statements = {
            "proponents": [],
            "opponents": []
        }

        # Get proponent statements
        for proponent in self.proponents:
            try:
                response = await proponent.send_task_request(
                    recipient=proponent.agent_id,
                    task_data={
                        "action": "opening_statement",
                        "topic": topic
                    },
                    wait_for_response=True,
                    timeout=20.0
                )

                if response:
                    statements["proponents"].append({
                        "agent": proponent.agent_id,
                        "statement": response.content
                    })

            except Exception as e:
                self.logger.error(f"Failed to get opening from {proponent.agent_id}: {e}")

        # Get opponent statements
        for opponent in self.opponents:
            try:
                response = await opponent.send_task_request(
                    recipient=opponent.agent_id,
                    task_data={
                        "action": "opening_statement",
                        "topic": topic
                    },
                    wait_for_response=True,
                    timeout=20.0
                )

                if response:
                    statements["opponents"].append({
                        "agent": opponent.agent_id,
                        "statement": response.content
                    })

            except Exception as e:
                self.logger.error(f"Failed to get opening from {opponent.agent_id}: {e}")

        return statements

    async def _execute_round(self, round_num: int, topic: str) -> Dict[str, Any]:
        """
        Execute one round of debate.

        Args:
            round_num: Current round number
            topic: Debate topic

        Returns:
            Round results including all arguments
        """
        # Get previous round's arguments for context
        previous_arguments = await self.state.get(f"round_{round_num - 1}_arguments", [])

        # Proponents argue first
        proponent_arguments = await self._collect_arguments(
            self.proponents, "proponent", round_num, topic, previous_arguments
        )

        # Opponents respond
        opponent_arguments = await self._collect_arguments(
            self.opponents, "opponent", round_num, topic,
            previous_arguments + proponent_arguments
        )

        # Store round results
        round_result = {
            "round": round_num,
            "proponent_arguments": proponent_arguments,
            "opponent_arguments": opponent_arguments
        }

        await self.state.set(
            f"round_{round_num}_arguments",
            proponent_arguments + opponent_arguments
        )

        return round_result

    async def _collect_arguments(self, agents: List[BaseCollaborativeAgent],
                                 side: str, round_num: int, topic: str,
                                 context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Collect arguments from one side.

        Args:
            agents: Agents to collect from
            side: Which side (proponent/opponent)
            round_num: Current round
            topic: Debate topic
            context: Previous arguments for context

        Returns:
            List of arguments
        """
        arguments = []

        for agent in agents:
            try:
                response = await agent.send_task_request(
                    recipient=agent.agent_id,
                    task_data={
                        "action": "argue",
                        "side": side,
                        "round": round_num,
                        "topic": topic,
                        "context": context
                    },
                    wait_for_response=True,
                    timeout=30.0
                )

                if response:
                    arguments.append({
                        "agent": agent.agent_id,
                        "side": side,
                        "round": round_num,
                        "argument": response.content
                    })

            except Exception as e:
                self.logger.error(f"Failed to get argument from {agent.agent_id}: {e}")

        return arguments

    async def _closing_statements(self, topic: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Collect closing statements from all participants.

        Args:
            topic: Debate topic

        Returns:
            Closing statements from both sides
        """
        self.logger.info("Collecting closing statements")

        statements = {
            "proponents": [],
            "opponents": []
        }

        # Get all debate history for context
        debate_history = await self.state.snapshot()

        # Get proponent closing statements
        for proponent in self.proponents:
            try:
                response = await proponent.send_task_request(
                    recipient=proponent.agent_id,
                    task_data={
                        "action": "closing_statement",
                        "topic": topic,
                        "debate_history": debate_history
                    },
                    wait_for_response=True,
                    timeout=20.0
                )

                if response:
                    statements["proponents"].append({
                        "agent": proponent.agent_id,
                        "statement": response.content
                    })

            except Exception as e:
                self.logger.error(f"Failed to get closing from {proponent.agent_id}: {e}")

        # Get opponent closing statements
        for opponent in self.opponents:
            try:
                response = await opponent.send_task_request(
                    recipient=opponent.agent_id,
                    task_data={
                        "action": "closing_statement",
                        "topic": topic,
                        "debate_history": debate_history
                    },
                    wait_for_response=True,
                    timeout=20.0
                )

                if response:
                    statements["opponents"].append({
                        "agent": opponent.agent_id,
                        "statement": response.content
                    })

            except Exception as e:
                self.logger.error(f"Failed to get closing from {opponent.agent_id}: {e}")

        return statements

    async def _evaluate_debate(self, opening: Dict[str, Any],
                              rounds: List[Dict[str, Any]],
                              closing: Dict[str, Any]) -> Dict[str, Any]:
        """
        Have judge evaluate the debate.

        Args:
            opening: Opening statements
            rounds: Debate rounds
            closing: Closing statements

        Returns:
            Judge's evaluation
        """
        if not self.judge:
            return {}

        self.logger.info("Judge evaluating debate")

        try:
            response = await self.judge.send_task_request(
                recipient=self.judge.agent_id,
                task_data={
                    "action": "evaluate_debate",
                    "opening_statements": opening,
                    "debate_rounds": rounds,
                    "closing_statements": closing
                },
                wait_for_response=True,
                timeout=30.0
            )

            if response:
                return response.content

        except Exception as e:
            self.logger.error(f"Judge evaluation failed: {e}")

        return {"evaluation": "evaluation_failed"}

    async def _synthesize_debate(self, topic: str,
                                opening: Dict[str, Any],
                                rounds: List[Dict[str, Any]],
                                closing: Dict[str, Any],
                                evaluation: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Synthesize debate results.

        Args:
            topic: Debate topic
            opening: Opening statements
            rounds: Debate rounds
            closing: Closing statements
            evaluation: Judge evaluation

        Returns:
            Synthesized debate results
        """
        # Count arguments
        total_arguments = sum(
            len(r.get("proponent_arguments", [])) + len(r.get("opponent_arguments", []))
            for r in rounds
        )

        # Extract key points from both sides
        proponent_points = self._extract_key_points(rounds, "proponent")
        opponent_points = self._extract_key_points(rounds, "opponent")

        synthesis = {
            "topic": topic,
            "total_rounds": len(rounds),
            "total_arguments": total_arguments,
            "proponent_key_points": proponent_points,
            "opponent_key_points": opponent_points,
            "opening_statements": opening,
            "debate_rounds": rounds,
            "closing_statements": closing,
            "evaluation": evaluation,
            "status": "completed"
        }

        return synthesis

    def _extract_key_points(self, rounds: List[Dict[str, Any]],
                           side: str) -> List[str]:
        """Extract key points from one side's arguments."""
        key_points = []

        for round_data in rounds:
            arguments_key = f"{side}_arguments"
            arguments = round_data.get(arguments_key, [])

            for arg in arguments:
                argument_content = arg.get("argument", {})
                if isinstance(argument_content, dict):
                    if "key_points" in argument_content:
                        key_points.extend(argument_content["key_points"])
                    elif "summary" in argument_content:
                        key_points.append(argument_content["summary"])

        return key_points[:10]  # Return top 10


class DebateAgent(BaseCollaborativeAgent):
    """
    Agent specialized for debate participation.

    Can argue for or against positions with evidence and reasoning.
    """

    def __init__(self, role: AgentRole, state_manager: StateManager,
                 message_bus: Optional[MessageBus] = None,
                 debate_role: DebateRole = DebateRole.PROPONENT):
        """
        Initialize debate agent.

        Args:
            role: Agent role
            state_manager: State manager
            message_bus: Message bus
            debate_role: Role in debate (proponent/opponent)
        """
        super().__init__(role, state_manager, message_bus)
        self.debate_role = debate_role

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute debate action.

        Args:
            input_data: Action and context

        Returns:
            Response based on action
        """
        action = input_data.get("action", "")

        if action == "opening_statement":
            return await self._make_opening_statement(input_data)
        elif action == "argue":
            return await self._make_argument(input_data)
        elif action == "closing_statement":
            return await self._make_closing_statement(input_data)
        else:
            return {"status": "unknown_action", "action": action}

    async def _make_opening_statement(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make opening statement."""
        topic = data.get("topic", "")

        # Would use LLM to generate actual statement
        statement = {
            "position": "for" if self.debate_role == DebateRole.PROPONENT else "against",
            "topic": topic,
            "statement": f"Opening statement {self.debate_role.value} the topic: {topic}",
            "key_points": [
                f"Key point 1 from {self.agent_id}",
                f"Key point 2 from {self.agent_id}"
            ]
        }

        return statement

    async def _make_argument(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make argument in debate round."""
        topic = data.get("topic", "")
        round_num = data.get("round", 1)
        context = data.get("context", [])

        # Analyze opponent's previous arguments
        opponent_points = [
            arg for arg in context
            if arg.get("side") != self.debate_role.value
        ]

        # Formulate counter-arguments and new points
        argument = {
            "position": self.debate_role.value,
            "round": round_num,
            "main_argument": f"Argument from {self.agent_id} in round {round_num}",
            "counter_arguments": [
                f"Counter to opponent point {i+1}"
                for i in range(min(2, len(opponent_points)))
            ],
            "supporting_evidence": [
                f"Evidence point {i+1} supporting {self.debate_role.value}"
                for i in range(2)
            ],
            "key_points": [
                f"Key argument {i+1} from {self.agent_id}"
                for i in range(3)
            ]
        }

        return argument

    async def _make_closing_statement(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make closing statement."""
        topic = data.get("topic", "")

        # Summarize position and strongest arguments
        statement = {
            "position": self.debate_role.value,
            "summary": f"Closing summary from {self.agent_id}",
            "strongest_arguments": [
                f"Strong point 1 from debate",
                f"Strong point 2 from debate"
            ],
            "conclusion": f"Final conclusion from {self.debate_role.value} side"
        }

        return statement


class ModeratorAgent(BaseCollaborativeAgent):
    """
    Agent that moderates debates.

    Ensures fair process and manages debate flow.
    """

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Moderate debate process.

        Args:
            input_data: Moderation task

        Returns:
            Moderation result
        """
        action = input_data.get("action", "moderate")

        if action == "moderate":
            return await self._moderate_round(input_data)
        else:
            return {"status": "unknown_action"}

    async def _moderate_round(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Moderate a debate round."""
        # Ensure fair time, check for rule violations, etc.
        return {
            "status": "round_moderated",
            "violations": [],
            "feedback": "Debate proceeding fairly"
        }
