"""
Agent communication infrastructure for collaboration patterns.

Provides message passing and event-driven communication between agents.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Set
from datetime import datetime
from enum import Enum
import logging
import uuid


class MessageType(Enum):
    """Types of messages that can be sent between agents."""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    QUERY = "query"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    STATUS_UPDATE = "status_update"
    ERROR = "error"
    VOTE = "vote"
    PROPOSAL = "proposal"
    CRITIQUE = "critique"
    AGREEMENT = "agreement"
    DISAGREEMENT = "disagreement"


@dataclass
class AgentMessage:
    """
    Message passed between agents in collaborative workflows.

    Attributes:
        message_id: Unique message identifier
        sender: Agent ID that sent the message
        recipient: Agent ID to receive message (None for broadcast)
        message_type: Type of message
        content: Message content/payload
        metadata: Additional message metadata
        timestamp: When message was created
        reply_to: ID of message this replies to
        priority: Message priority (higher = more urgent)
    """
    sender: str
    message_type: MessageType
    content: Any
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    recipient: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    reply_to: Optional[str] = None
    priority: int = 0

    def is_broadcast(self) -> bool:
        """Check if message is a broadcast."""
        return self.recipient is None or self.message_type == MessageType.BROADCAST

    def is_reply(self) -> bool:
        """Check if message is a reply to another message."""
        return self.reply_to is not None

    def create_reply(self, sender: str, content: Any,
                    message_type: Optional[MessageType] = None) -> "AgentMessage":
        """
        Create a reply to this message.

        Args:
            sender: Agent sending the reply
            content: Reply content
            message_type: Type of reply (defaults to RESPONSE)

        Returns:
            New message that replies to this one
        """
        return AgentMessage(
            sender=sender,
            recipient=self.sender,
            message_type=message_type or MessageType.RESPONSE,
            content=content,
            reply_to=self.message_id,
            priority=self.priority
        )

    def __repr__(self) -> str:
        return (f"<AgentMessage(id={self.message_id[:8]}, "
                f"type={self.message_type.value}, "
                f"from={self.sender}, "
                f"to={self.recipient or 'broadcast'})>")


class MessageBus:
    """
    Central message bus for agent communication.

    Handles message routing, delivery, and subscription management.
    Supports both direct messaging and broadcast patterns.

    Example:
        >>> bus = MessageBus()
        >>>
        >>> # Subscribe to messages
        >>> await bus.subscribe("agent1", agent1_handler)
        >>>
        >>> # Send direct message
        >>> message = AgentMessage(
        ...     sender="agent2",
        ...     recipient="agent1",
        ...     message_type=MessageType.TASK_REQUEST,
        ...     content={"task": "analyze_data"}
        ... )
        >>> await bus.send(message)
        >>>
        >>> # Broadcast to all agents
        >>> await bus.broadcast("coordinator", MessageType.STATUS_UPDATE,
        ...                     {"phase": "completed"})
    """

    def __init__(self):
        """Initialize message bus."""
        self.logger = logging.getLogger("MessageBus")

        # Agent subscriptions: agent_id -> handler function
        self._subscribers: Dict[str, Callable] = {}

        # Message queues: agent_id -> queue of messages
        self._queues: Dict[str, asyncio.Queue] = {}

        # Message history for debugging/replay
        self._message_history: List[AgentMessage] = []

        # Active agent IDs
        self._active_agents: Set[str] = set()

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

    async def subscribe(self, agent_id: str,
                       handler: Callable[[AgentMessage], Any]) -> None:
        """
        Subscribe an agent to receive messages.

        Args:
            agent_id: Unique agent identifier
            handler: Async function to handle incoming messages
                    Signature: async def handler(message: AgentMessage) -> Any
        """
        async with self._lock:
            self._subscribers[agent_id] = handler
            self._queues[agent_id] = asyncio.Queue()
            self._active_agents.add(agent_id)

        self.logger.info(f"Agent {agent_id} subscribed to message bus")

    async def unsubscribe(self, agent_id: str) -> None:
        """
        Unsubscribe an agent from receiving messages.

        Args:
            agent_id: Agent identifier to unsubscribe
        """
        async with self._lock:
            if agent_id in self._subscribers:
                del self._subscribers[agent_id]
            if agent_id in self._queues:
                del self._queues[agent_id]
            self._active_agents.discard(agent_id)

        self.logger.info(f"Agent {agent_id} unsubscribed from message bus")

    async def send(self, message: AgentMessage) -> bool:
        """
        Send a message to a specific agent or broadcast to all.

        Args:
            message: Message to send

        Returns:
            True if message was delivered, False otherwise
        """
        # Add to history
        self._message_history.append(message)

        if message.is_broadcast():
            # Broadcast to all agents except sender
            await self._broadcast_message(message)
            return True
        else:
            # Send to specific recipient
            return await self._deliver_message(message)

    async def _deliver_message(self, message: AgentMessage) -> bool:
        """
        Deliver message to specific recipient.

        Args:
            message: Message to deliver

        Returns:
            True if delivered, False if recipient not found
        """
        recipient = message.recipient

        if not recipient:
            self.logger.warning(f"Message {message.message_id} has no recipient")
            return False

        async with self._lock:
            if recipient not in self._queues:
                self.logger.warning(f"Recipient {recipient} not subscribed")
                return False

            queue = self._queues[recipient]

        # Put message in queue
        await queue.put(message)

        # Call handler if available
        if recipient in self._subscribers:
            try:
                handler = self._subscribers[recipient]
                # Don't await - let handler process asynchronously
                asyncio.create_task(self._invoke_handler(recipient, handler, message))
            except Exception as e:
                self.logger.error(f"Error invoking handler for {recipient}: {e}")

        self.logger.debug(f"Delivered message {message.message_id[:8]} to {recipient}")
        return True

    async def _broadcast_message(self, message: AgentMessage) -> None:
        """
        Broadcast message to all agents except sender.

        Args:
            message: Message to broadcast
        """
        sender = message.sender

        async with self._lock:
            recipients = [
                agent_id for agent_id in self._active_agents
                if agent_id != sender
            ]

        # Send to all recipients
        for recipient in recipients:
            # Create a copy for each recipient
            recipient_message = AgentMessage(
                sender=message.sender,
                recipient=recipient,
                message_type=message.message_type,
                content=message.content,
                metadata=message.metadata,
                timestamp=message.timestamp,
                reply_to=message.reply_to,
                priority=message.priority
            )
            await self._deliver_message(recipient_message)

        self.logger.debug(f"Broadcast message {message.message_id[:8]} to {len(recipients)} agents")

    async def _invoke_handler(self, agent_id: str, handler: Callable,
                             message: AgentMessage) -> None:
        """
        Invoke message handler with error handling.

        Args:
            agent_id: Agent receiving message
            handler: Handler function to invoke
            message: Message to handle
        """
        try:
            await handler(message)
        except Exception as e:
            self.logger.error(f"Handler error for {agent_id}: {e}", exc_info=True)

            # Send error message back to sender
            error_message = AgentMessage(
                sender=agent_id,
                recipient=message.sender,
                message_type=MessageType.ERROR,
                content={"error": str(e), "original_message_id": message.message_id},
                reply_to=message.message_id
            )
            await self.send(error_message)

    async def broadcast(self, sender: str, message_type: MessageType,
                       content: Any, **kwargs) -> None:
        """
        Convenience method to broadcast a message.

        Args:
            sender: Agent sending broadcast
            message_type: Type of message
            content: Message content
            **kwargs: Additional message parameters
        """
        message = AgentMessage(
            sender=sender,
            recipient=None,
            message_type=message_type,
            content=content,
            **kwargs
        )
        await self.send(message)

    async def receive(self, agent_id: str, timeout: Optional[float] = None) -> Optional[AgentMessage]:
        """
        Receive next message for an agent.

        Args:
            agent_id: Agent receiving message
            timeout: Max seconds to wait (None = wait forever)

        Returns:
            Next message or None if timeout
        """
        if agent_id not in self._queues:
            self.logger.warning(f"Agent {agent_id} not subscribed")
            return None

        queue = self._queues[agent_id]

        try:
            if timeout:
                message = await asyncio.wait_for(queue.get(), timeout=timeout)
            else:
                message = await queue.get()

            return message
        except asyncio.TimeoutError:
            return None

    async def receive_all(self, agent_id: str) -> List[AgentMessage]:
        """
        Receive all pending messages for an agent.

        Args:
            agent_id: Agent receiving messages

        Returns:
            List of pending messages
        """
        if agent_id not in self._queues:
            return []

        queue = self._queues[agent_id]
        messages = []

        while not queue.empty():
            try:
                message = queue.get_nowait()
                messages.append(message)
            except asyncio.QueueEmpty:
                break

        return messages

    def get_message_history(self, agent_id: Optional[str] = None,
                           limit: int = 100) -> List[AgentMessage]:
        """
        Get message history.

        Args:
            agent_id: Filter by agent (sender or recipient), None for all
            limit: Maximum messages to return

        Returns:
            List of messages (most recent first)
        """
        if agent_id:
            messages = [
                msg for msg in self._message_history
                if msg.sender == agent_id or msg.recipient == agent_id
            ]
        else:
            messages = self._message_history

        return list(reversed(messages[-limit:]))

    def get_active_agents(self) -> Set[str]:
        """
        Get set of active agent IDs.

        Returns:
            Set of agent IDs currently subscribed
        """
        return self._active_agents.copy()

    def clear_history(self) -> None:
        """Clear message history."""
        self._message_history.clear()
        self.logger.info("Message history cleared")

    async def wait_for_replies(self, message_id: str,
                               expected_count: int,
                               timeout: float = 30.0) -> List[AgentMessage]:
        """
        Wait for replies to a specific message.

        Args:
            message_id: ID of message to wait for replies to
            expected_count: Number of replies expected
            timeout: Max seconds to wait

        Returns:
            List of reply messages received

        Raises:
            asyncio.TimeoutError: If timeout reached before all replies received
        """
        replies = []
        deadline = asyncio.get_event_loop().time() + timeout

        while len(replies) < expected_count:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise asyncio.TimeoutError(
                    f"Timeout waiting for replies. Got {len(replies)}/{expected_count}"
                )

            # Check message history for new replies
            for msg in reversed(self._message_history):
                if msg.reply_to == message_id and msg not in replies:
                    replies.append(msg)

            if len(replies) >= expected_count:
                break

            # Wait a bit before checking again
            await asyncio.sleep(0.1)

        return replies[:expected_count]

    def __repr__(self) -> str:
        return (f"<MessageBus(agents={len(self._active_agents)}, "
                f"messages={len(self._message_history)})>")
