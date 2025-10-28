"""
Base agent for collaborative patterns.

Provides enhanced agent base class with message passing capabilities
for multi-agent collaboration patterns.
"""

import asyncio
from typing import Dict, Any, Optional, List, Callable
import logging

from .core import BaseAgent, AgentRole
from .state import StateManager
from .communication import MessageBus, AgentMessage, MessageType


class BaseCollaborativeAgent(BaseAgent):
    """
    Enhanced base agent with communication capabilities for collaboration.

    Extends BaseAgent with:
    - Message bus integration for agent-to-agent communication
    - Event handlers for different message types
    - Async message processing
    - Collaboration state management

    Example:
        >>> class MyCollaborativeAgent(BaseCollaborativeAgent):
        ...     async def execute(self, input_data):
        ...         # Send request to another agent
        ...         await self.send_message(
        ...             recipient="analyzer",
        ...             message_type=MessageType.TASK_REQUEST,
        ...             content={"data": input_data}
        ...         )
        ...
        ...         # Wait for response
        ...         response = await self.wait_for_message(
        ...             from_agent="analyzer",
        ...             timeout=30.0
        ...         )
        ...
        ...         return response.content
    """

    def __init__(self, role: AgentRole, state_manager: StateManager,
                 message_bus: Optional[MessageBus] = None):
        """
        Initialize collaborative agent.

        Args:
            role: Agent's role definition
            state_manager: Centralized state manager
            message_bus: Message bus for inter-agent communication
        """
        super().__init__(role, state_manager)

        self.message_bus = message_bus or MessageBus()
        self.agent_id = role.name

        # Message handlers: message_type -> handler function
        self._message_handlers: Dict[MessageType, Callable] = {}

        # Pending messages queue
        self._message_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()

        # Track sent messages awaiting replies
        self._pending_replies: Dict[str, asyncio.Future] = {}

        # Register default handlers
        self._register_default_handlers()

    async def initialize(self) -> None:
        """
        Initialize agent with message bus subscription.

        Should be called after agent creation to set up message handling.
        """
        await self.message_bus.subscribe(self.agent_id, self._handle_message)
        self.logger.info(f"Agent {self.agent_id} initialized and subscribed to message bus")

    async def shutdown(self) -> None:
        """
        Shutdown agent and clean up resources.
        """
        await self.message_bus.unsubscribe(self.agent_id)
        self.logger.info(f"Agent {self.agent_id} shutdown")

    def _register_default_handlers(self) -> None:
        """Register default message handlers."""
        self.register_message_handler(MessageType.TASK_REQUEST, self._handle_task_request)
        self.register_message_handler(MessageType.QUERY, self._handle_query)
        self.register_message_handler(MessageType.RESPONSE, self._handle_response)
        self.register_message_handler(MessageType.ERROR, self._handle_error)

    def register_message_handler(self, message_type: MessageType,
                                 handler: Callable[[AgentMessage], Any]) -> None:
        """
        Register a handler for a specific message type.

        Args:
            message_type: Type of message to handle
            handler: Async function to handle the message
                    Signature: async def handler(message: AgentMessage) -> Any
        """
        self._message_handlers[message_type] = handler
        self.logger.debug(f"Registered handler for {message_type.value}")

    async def _handle_message(self, message: AgentMessage) -> None:
        """
        Central message handler that routes to specific handlers.

        Args:
            message: Incoming message
        """
        self.logger.debug(f"Received {message.message_type.value} from {message.sender}")

        # Add to queue for polling
        await self._message_queue.put(message)

        # Route to specific handler
        handler = self._message_handlers.get(message.message_type)
        if handler:
            try:
                await handler(message)
            except Exception as e:
                self.logger.error(f"Error handling {message.message_type.value}: {e}", exc_info=True)

                # Send error response
                if message.sender:
                    await self.send_error(message.sender, str(e), message.message_id)
        else:
            self.logger.warning(f"No handler for message type: {message.message_type.value}")

    async def _handle_task_request(self, message: AgentMessage) -> None:
        """
        Handle task request from another agent.

        Override this to implement custom task handling.

        Args:
            message: Task request message
        """
        self.logger.info(f"Received task request from {message.sender}")

        try:
            # Execute the task
            result = await self.execute(message.content)

            # Send response
            await self.send_message(
                recipient=message.sender,
                message_type=MessageType.TASK_RESPONSE,
                content=result,
                reply_to=message.message_id
            )
        except Exception as e:
            await self.send_error(message.sender, str(e), message.message_id)

    async def _handle_query(self, message: AgentMessage) -> None:
        """
        Handle query from another agent.

        Override this to implement custom query handling.

        Args:
            message: Query message
        """
        self.logger.info(f"Received query from {message.sender}")

        # Default: echo query
        await self.send_message(
            recipient=message.sender,
            message_type=MessageType.RESPONSE,
            content={"status": "received", "query": message.content},
            reply_to=message.message_id
        )

    async def _handle_response(self, message: AgentMessage) -> None:
        """
        Handle response to a previous message.

        Args:
            message: Response message
        """
        # If this is a reply we're waiting for, resolve the future
        if message.reply_to and message.reply_to in self._pending_replies:
            future = self._pending_replies[message.reply_to]
            if not future.done():
                future.set_result(message)
            del self._pending_replies[message.reply_to]

    async def _handle_error(self, message: AgentMessage) -> None:
        """
        Handle error message from another agent.

        Args:
            message: Error message
        """
        self.logger.error(f"Error from {message.sender}: {message.content}")

        # If this is a reply we're waiting for, set exception
        if message.reply_to and message.reply_to in self._pending_replies:
            future = self._pending_replies[message.reply_to]
            if not future.done():
                error_content = message.content
                error_msg = error_content.get("error", "Unknown error") if isinstance(error_content, dict) else str(error_content)
                future.set_exception(Exception(error_msg))
            del self._pending_replies[message.reply_to]

    async def send_message(self, recipient: str, message_type: MessageType,
                          content: Any, reply_to: Optional[str] = None,
                          **kwargs) -> AgentMessage:
        """
        Send a message to another agent.

        Args:
            recipient: Target agent ID
            message_type: Type of message
            content: Message content
            reply_to: ID of message this replies to
            **kwargs: Additional message parameters

        Returns:
            The sent message
        """
        message = AgentMessage(
            sender=self.agent_id,
            recipient=recipient,
            message_type=message_type,
            content=content,
            reply_to=reply_to,
            **kwargs
        )

        await self.message_bus.send(message)
        self.logger.debug(f"Sent {message_type.value} to {recipient}")

        return message

    async def broadcast_message(self, message_type: MessageType,
                               content: Any, **kwargs) -> None:
        """
        Broadcast a message to all agents.

        Args:
            message_type: Type of message
            content: Message content
            **kwargs: Additional message parameters
        """
        await self.message_bus.broadcast(
            sender=self.agent_id,
            message_type=message_type,
            content=content,
            **kwargs
        )
        self.logger.debug(f"Broadcast {message_type.value}")

    async def send_task_request(self, recipient: str, task_data: Dict[str, Any],
                               wait_for_response: bool = True,
                               timeout: float = 30.0) -> Optional[AgentMessage]:
        """
        Send a task request to another agent.

        Args:
            recipient: Target agent ID
            task_data: Task parameters
            wait_for_response: Whether to wait for response
            timeout: Max seconds to wait for response

        Returns:
            Response message if wait_for_response=True, None otherwise
        """
        message = await self.send_message(
            recipient=recipient,
            message_type=MessageType.TASK_REQUEST,
            content=task_data
        )

        if wait_for_response:
            return await self.wait_for_reply(message.message_id, timeout=timeout)

        return None

    async def send_query(self, recipient: str, query: Any,
                        timeout: float = 30.0) -> AgentMessage:
        """
        Send a query to another agent and wait for response.

        Args:
            recipient: Target agent ID
            query: Query content
            timeout: Max seconds to wait for response

        Returns:
            Response message

        Raises:
            asyncio.TimeoutError: If no response within timeout
        """
        message = await self.send_message(
            recipient=recipient,
            message_type=MessageType.QUERY,
            content=query
        )

        return await self.wait_for_reply(message.message_id, timeout=timeout)

    async def send_error(self, recipient: str, error: str,
                        reply_to: Optional[str] = None) -> None:
        """
        Send an error message to another agent.

        Args:
            recipient: Target agent ID
            error: Error description
            reply_to: ID of message that caused error
        """
        await self.send_message(
            recipient=recipient,
            message_type=MessageType.ERROR,
            content={"error": error},
            reply_to=reply_to
        )

    async def wait_for_reply(self, message_id: str,
                            timeout: float = 30.0) -> AgentMessage:
        """
        Wait for a reply to a specific message.

        Args:
            message_id: ID of message to wait for reply to
            timeout: Max seconds to wait

        Returns:
            Reply message

        Raises:
            asyncio.TimeoutError: If no reply within timeout
            Exception: If error response received
        """
        # Create future for this reply
        future = asyncio.get_event_loop().create_future()
        self._pending_replies[message_id] = future

        try:
            # Wait for reply with timeout
            reply = await asyncio.wait_for(future, timeout=timeout)
            return reply
        except asyncio.TimeoutError:
            # Clean up pending reply
            if message_id in self._pending_replies:
                del self._pending_replies[message_id]
            raise

    async def wait_for_message(self, from_agent: Optional[str] = None,
                              message_type: Optional[MessageType] = None,
                              timeout: Optional[float] = None) -> Optional[AgentMessage]:
        """
        Wait for next message matching criteria.

        Args:
            from_agent: Filter by sender (None = any)
            message_type: Filter by type (None = any)
            timeout: Max seconds to wait (None = forever)

        Returns:
            Matching message or None if timeout
        """
        deadline = None
        if timeout:
            deadline = asyncio.get_event_loop().time() + timeout

        while True:
            # Check timeout
            if deadline and asyncio.get_event_loop().time() >= deadline:
                return None

            # Try to get message with remaining timeout
            remaining_timeout = None
            if deadline:
                remaining_timeout = max(0.1, deadline - asyncio.get_event_loop().time())

            try:
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=remaining_timeout
                )

                # Check if message matches criteria
                if from_agent and message.sender != from_agent:
                    # Put back and continue
                    await self._message_queue.put(message)
                    await asyncio.sleep(0.1)
                    continue

                if message_type and message.message_type != message_type:
                    # Put back and continue
                    await self._message_queue.put(message)
                    await asyncio.sleep(0.1)
                    continue

                return message

            except asyncio.TimeoutError:
                return None

    async def get_pending_messages(self) -> List[AgentMessage]:
        """
        Get all pending messages without waiting.

        Returns:
            List of pending messages
        """
        messages = []
        while not self._message_queue.empty():
            try:
                message = self._message_queue.get_nowait()
                messages.append(message)
            except asyncio.QueueEmpty:
                break
        return messages

    def has_pending_messages(self) -> bool:
        """
        Check if agent has pending messages.

        Returns:
            True if messages are pending
        """
        return not self._message_queue.empty()

    async def collaborate_with(self, other_agent_id: str,
                              task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collaborate with another agent on a task.

        Override this to implement custom collaboration logic.

        Args:
            other_agent_id: ID of agent to collaborate with
            task_data: Task parameters

        Returns:
            Collaboration results
        """
        # Default: send task request and return response
        response = await self.send_task_request(
            recipient=other_agent_id,
            task_data=task_data,
            wait_for_response=True
        )

        if response:
            return response.content
        else:
            return {"status": "no_response"}

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id={self.agent_id}, role={self.role.name})>"
