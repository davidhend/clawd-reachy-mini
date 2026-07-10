"""OpenClaw Gateway client for sending/receiving messages."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass
from typing import AsyncIterator, Awaitable, Callable

import websockets
from websockets.client import WebSocketClientProtocol

from clawd_reachy_mini.config import Config

logger = logging.getLogger(__name__)

# How long send_message waits for the agent's final reply. Runs that outlive
# this are NOT abandoned: send_message raises ReplyPending and the reply is
# delivered through the late-result callback when the run finishes.
REPLY_TIMEOUT_S = float(os.environ.get("OPENCLAW_REPLY_TIMEOUT", "120"))


class ReplyPending(TimeoutError):
    """The agent is still working; the reply will arrive via the late-result callback."""


@dataclass
class Message:
    """A message to/from OpenClaw."""

    content: str
    role: str = "user"  # "user" or "assistant"
    channel: str = "reachy-mini"
    session_id: str | None = None


class GatewayClient:
    """Client for communicating with OpenClaw Gateway."""

    def __init__(self, config: Config):
        self.config = config
        self._ws: WebSocketClientProtocol | None = None
        self._session_id: str = str(uuid.uuid4())
        self._connected = False
        self._authenticated = False
        self._auth_event: asyncio.Event | None = None
        self._register_event: asyncio.Event | None = None
        self._response_handlers: dict[str, asyncio.Future] = {}
        self._listener_task: asyncio.Task | None = None
        self._tool_dispatcher: Callable[[str, dict], dict] | None = None
        self._late_result_cb: Callable[[str], Awaitable[None]] | None = None

    def register_tool_dispatcher(self, dispatcher: Callable[[str, dict], dict]) -> None:
        """Register a sync callable that handles gateway-originated tool.request messages."""
        self._tool_dispatcher = dispatcher

    def register_late_result_callback(self, callback: Callable[[str], Awaitable[None]]) -> None:
        """Register a coroutine function that receives the final text of runs
        whose send_message call already timed out (see ReplyPending)."""
        self._late_result_cb = callback

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ws is not None

    async def connect(self) -> None:
        """Connect to the OpenClaw Gateway."""
        if self.is_connected:
            return

        headers = {}
        if self.config.gateway_token:
            headers["Authorization"] = f"Bearer {self.config.gateway_token}"

        try:
            self._ws = await websockets.connect(
                self.config.gateway_url,
                additional_headers=headers,
            )
            self._connected = True
            self._auth_event = asyncio.Event()
            self._register_event = asyncio.Event()
            self._listener_task = asyncio.create_task(self._listen())

            # Wait for authentication challenge to be handled
            try:
                await asyncio.wait_for(self._auth_event.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                # No challenge received, proceed without authentication
                logger.debug("No authentication challenge received, proceeding")
                self._authenticated = True

            # Wait for server to acknowledge the connect request
            try:
                await asyncio.wait_for(self._register_event.wait(), timeout=10.0)
                logger.debug("Connection established successfully")
            except asyncio.TimeoutError:
                logger.warning("No connect response received, connection may fail")

            logger.info(f"Connected to OpenClaw Gateway at {self.config.gateway_url}")

        except Exception as e:
            logger.error(f"Failed to connect to Gateway: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from the Gateway."""
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass

        if self._ws:
            await self._ws.close()
            self._ws = None
            self._connected = False

        logger.info("Disconnected from OpenClaw Gateway")

    async def send_message(self, text: str, image_path: str | None = None) -> str:
        """
        Send a message to OpenClaw and wait for response.

        Args:
            text: The user's message text
            image_path: Optional path to an image to include

        Returns:
            The assistant's response text
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to Gateway")

        message_id = str(uuid.uuid4())

        # Use OpenClaw protocol format: req/res with chat.send method
        request = {
            "type": "req",
            "id": message_id,
            "method": "chat.send",
            "params": {
                "message": text,
                "sessionKey": f"reachy-mini:{self._session_id}",
                "idempotencyKey": message_id,
            },
        }

        if image_path:
            # Include image as attachment
            request["params"]["attachments"] = [{"type": "image", "path": image_path}]

        # Create future for the initial response (runId)
        init_future: asyncio.Future[dict] = asyncio.Future()
        self._response_handlers[message_id] = init_future

        await self._send_raw(request)

        run_id = None
        run_still_pending = False
        try:
            # Wait for initial response with runId
            try:
                init_response = await asyncio.wait_for(init_future, timeout=30.0)
            except asyncio.TimeoutError:
                logger.error("Timeout waiting for Gateway to accept chat.send")
                raise
            run_id = init_response.get("runId")

            if not run_id:
                # Direct response without async run
                return init_response.get("text", init_response.get("content", str(init_response)))

            logger.debug(f"Chat run started: {run_id}")

            # Create handler dict to accumulate text and store future
            result_future: asyncio.Future[str] = asyncio.Future()
            self._response_handlers[run_id] = {
                "future": result_future,
                "text": "",
            }

            # Wait for the AI to complete
            try:
                return await asyncio.wait_for(result_future, timeout=REPLY_TIMEOUT_S)
            except asyncio.TimeoutError:
                if self._late_result_cb is None:
                    logger.error("Timeout waiting for Gateway response")
                    raise
                # Long-running task (e.g. "install updates on that host"). Keep
                # the run handler registered — _handle_event fires the late
                # callback when the run's completion event arrives.
                self._response_handlers[run_id]["late"] = True
                run_still_pending = True
                logger.warning(
                    f"Run {run_id} still going after {REPLY_TIMEOUT_S:.0f}s — "
                    "will announce the reply when it lands"
                )
                raise ReplyPending(run_id)
        finally:
            # OpenClaw uses the chat.send idempotencyKey as the runId, so
            # message_id and run_id are usually the SAME key — popping
            # message_id would destroy a still-pending run handler.
            if not (run_still_pending and message_id == run_id):
                self._response_handlers.pop(message_id, None)
            if run_id and run_id != message_id and not run_still_pending:
                self._response_handlers.pop(run_id, None)

    async def stream_message(
        self,
        text: str,
        on_chunk: Callable[[str], None] | None = None,
    ) -> AsyncIterator[str]:
        """
        Send a message and stream the response.

        Args:
            text: The user's message text
            on_chunk: Optional callback for each chunk

        Yields:
            Response text chunks
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to Gateway")

        message_id = str(uuid.uuid4())

        payload = {
            "type": "message.send",
            "id": message_id,
            "session_id": self._session_id,
            "channel": "reachy-mini",
            "content": text,
            "stream": True,
        }

        # Queue for streaming chunks
        chunk_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._response_handlers[message_id] = chunk_queue  # type: ignore

        await self._send_raw(payload)

        try:
            while True:
                chunk = await asyncio.wait_for(chunk_queue.get(), timeout=120.0)
                if chunk is None:  # End of stream
                    break
                if on_chunk:
                    on_chunk(chunk)
                yield chunk
        finally:
            self._response_handlers.pop(message_id, None)

    async def _send_raw(self, data: dict) -> None:
        """Send raw JSON to the Gateway."""
        if not self._ws:
            raise RuntimeError("WebSocket not connected")
        await self._ws.send(json.dumps(data))

    async def _listen(self) -> None:
        """Listen for incoming messages from the Gateway."""
        if not self._ws:
            return

        try:
            async for raw_message in self._ws:
                try:
                    data = json.loads(raw_message)
                    await self._handle_message(data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from Gateway: {raw_message}")
        except websockets.ConnectionClosed:
            logger.info("Gateway connection closed")
            self._connected = False
        except Exception as e:
            logger.error(f"Error in Gateway listener: {e}")
            self._connected = False

    async def _handle_message(self, data: dict) -> None:
        """Handle an incoming message from the Gateway."""
        msg_type = data.get("type", "")
        msg_id = data.get("reply_to") or data.get("id")

        if msg_type == "event":
            # Handle event wrapper messages
            event_name = data.get("event", "")
            await self._handle_event(event_name, data)
            return

        if msg_type == "res":
            # Handle response to our requests
            ok = data.get("ok", False)
            payload = data.get("payload", {})
            error = data.get("error")

            if ok:
                payload_type = payload.get("type", "")
                if payload_type == "hello-ok":
                    # Connect handshake successful
                    logger.info("Gateway handshake successful")
                    if self._register_event:
                        self._register_event.set()
                elif msg_id and msg_id in self._response_handlers:
                    handler = self._response_handlers[msg_id]
                    if isinstance(handler, asyncio.Future):
                        handler.set_result(payload)
                    elif isinstance(handler, asyncio.Queue):
                        await handler.put(payload)
            else:
                logger.error(f"Request failed: {error}")
                if msg_id and msg_id in self._response_handlers:
                    handler = self._response_handlers[msg_id]
                    if isinstance(handler, asyncio.Future):
                        handler.set_exception(RuntimeError(str(error)))
            return

        if msg_type == "message.response":
            # Complete response
            if msg_id and msg_id in self._response_handlers:
                handler = self._response_handlers[msg_id]
                if isinstance(handler, asyncio.Future):
                    handler.set_result(data.get("content", ""))
                elif isinstance(handler, asyncio.Queue):
                    await handler.put(data.get("content", ""))
                    await handler.put(None)  # Signal end

        elif msg_type == "message.chunk":
            # Streaming chunk
            if msg_id and msg_id in self._response_handlers:
                handler = self._response_handlers[msg_id]
                if isinstance(handler, asyncio.Queue):
                    await handler.put(data.get("content", ""))

        elif msg_type == "message.end":
            # End of stream
            if msg_id and msg_id in self._response_handlers:
                handler = self._response_handlers[msg_id]
                if isinstance(handler, asyncio.Queue):
                    await handler.put(None)

        elif msg_type == "tool.request":
            # Gateway requesting tool execution (e.g., move robot)
            await self._handle_tool_request(data)

        elif msg_type == "error":
            logger.error(f"Gateway error: {data.get('message', 'Unknown error')}")
            if msg_id and msg_id in self._response_handlers:
                handler = self._response_handlers[msg_id]
                if isinstance(handler, asyncio.Future):
                    handler.set_exception(RuntimeError(data.get("message", "Gateway error")))

    async def _handle_event(self, event_name: str, data: dict) -> None:
        """Handle event messages from the Gateway."""
        payload = data.get("payload", {})
        logger.debug(f"Received event: {event_name}, payload: {payload}")

        if event_name == "connect.challenge":
            # Respond to authentication challenge with proper connect request
            nonce = payload.get("nonce", "")
            ts = payload.get("ts", "")

            logger.debug(f"Handling connect.challenge: nonce={nonce}, ts={ts}")

            if not self.config.gateway_token:
                logger.warning(
                    "No gateway token configured - connection may be rejected. "
                    "Set --gateway-token or OPENCLAW_TOKEN environment variable."
                )

            # Send connect request per OpenClaw protocol
            # client.id must be one of the allowed constants
            connect_request = {
                "type": "req",
                "id": str(uuid.uuid4()),
                "method": "connect",
                "params": {
                    "minProtocol": 3,
                    "maxProtocol": 4,
                    "client": {
                        "id": "gateway-client",  # Must be an allowed client ID
                        "version": "0.1.0",
                        "platform": "python",
                        "mode": "backend",
                    },
                    "role": "operator",
                    # Scopes are granted per-connection from this list; without
                    # it the connection gets none and chat.send fails with
                    # "missing scope: operator.write".
                    "scopes": ["operator.read", "operator.write"],
                    "auth": {
                        "token": self.config.gateway_token or "",
                    },
                },
            }

            await self._send_raw(connect_request)
            logger.debug("Sent connect request")

            self._authenticated = True

            if self._auth_event:
                self._auth_event.set()

        elif event_name == "connect.accepted":
            logger.info("Connection accepted by Gateway")
            self._authenticated = True
            if self._auth_event:
                self._auth_event.set()
            if self._register_event:
                self._register_event.set()

        elif event_name == "connect.rejected":
            logger.error("Connection rejected by Gateway")
            self._authenticated = False
            if self._auth_event:
                self._auth_event.set()

        elif event_name == "agent":
            # Agent events contain the AI response
            run_id = payload.get("runId")
            stream_type = payload.get("stream")
            data = payload.get("data", {})

            if stream_type == "lifecycle":
                phase = data.get("phase")
                if phase == "end":
                    # Run completed - get final text from accumulated response
                    logger.debug(f"Agent run {run_id} completed")
                    # The final text should have been accumulated
                    if run_id and run_id in self._response_handlers:
                        handler = self._response_handlers.pop(run_id, None)
                        if isinstance(handler, dict):
                            # We stored accumulated text in a dict
                            final_text = handler.get("text", "")
                            future = handler.get("future")
                            if future and not future.done():
                                future.set_result(final_text)
                            elif handler.get("late"):
                                self._deliver_late_result(run_id, final_text)

            elif stream_type == "assistant":
                # Streaming text from assistant
                text = data.get("text", "")  # Accumulated text
                if run_id and run_id in self._response_handlers:
                    handler = self._response_handlers[run_id]
                    if isinstance(handler, dict):
                        handler["text"] = text  # Update accumulated text

        elif event_name == "chat":
            # Chat state updates
            run_id = payload.get("runId")
            state = payload.get("state")
            message = payload.get("message", {})

            if state == "complete":
                # Chat completed
                content = message.get("content", [])
                text = ""
                for item in content:
                    if item.get("type") == "text":
                        text = item.get("text", "")
                        break

                logger.debug(f"Chat complete for {run_id}: {text[:100] if text else '(empty)'}...")

                if run_id and run_id in self._response_handlers:
                    handler = self._response_handlers.pop(run_id)
                    if isinstance(handler, dict):
                        future = handler.get("future")
                        if future and not future.done():
                            future.set_result(text)
                        elif handler.get("late"):
                            self._deliver_late_result(run_id, text)
                    elif isinstance(handler, asyncio.Future) and not handler.done():
                        handler.set_result(text)

        else:
            logger.debug(f"Unhandled event: {event_name}")

    def _deliver_late_result(self, run_id: str, text: str) -> None:
        """Hand a post-timeout run result to the registered callback."""
        if not text.strip():
            logger.warning(f"Late run {run_id} finished with no text — nothing to announce")
            return
        if self._late_result_cb is not None:
            # The callback speaks (slow); don't block the websocket listener.
            asyncio.create_task(self._late_result_cb(text))

    async def _handle_tool_request(self, data: dict) -> None:
        """Handle tool execution requests from the Gateway."""
        tool_name = data.get("tool")
        tool_args = data.get("arguments", {}) or {}
        request_id = data.get("id")

        logger.info(f"Tool request: {tool_name}({tool_args})")

        if self._tool_dispatcher is None:
            result = {"status": "error", "message": "Tool handler not registered"}
        elif not isinstance(tool_name, str) or not tool_name:
            result = {"status": "error", "message": "Missing tool name"}
        else:
            try:
                # Dispatchers are sync (motion calls block); run off the event loop
                # so the websocket listener keeps draining messages.
                result = await asyncio.to_thread(
                    self._tool_dispatcher, tool_name, tool_args
                )
            except Exception as e:
                logger.exception("Tool dispatcher raised")
                result = {"status": "error", "message": str(e)}

        await self._send_raw({
            "type": "tool.response",
            "id": request_id,
            "result": result,
        })
