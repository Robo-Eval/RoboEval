"""
WebSocket client for TeleArms communication using JSON-RPC 2.0 protocol.

This module handles communication with the telearms-ws-server for:
- Instance registration
- Task lifecycle events (progress, completion, failure)
- Receiving commands from backend (start, stop, reset)
"""

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import websockets

logger = logging.getLogger(__name__)


class JsonRpcError(Exception):
    """JSON-RPC error response."""
    def __init__(self, code: int, message: str, data: Any = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"JSON-RPC Error {code}: {message}")


class InstanceStatus(Enum):
    """Simulation instance status."""
    AVAILABLE = "available"
    BUSY = "busy"
    IDLE = "idle"


@dataclass
class CommandHandler:
    """Handler for incoming commands from the server."""
    method: str
    callback: Callable[[Dict[str, Any]], Dict[str, Any]]


class TeleArmsWSClient:
    """
    WebSocket client for TeleArms WS Server.

    Uses JSON-RPC 2.0 protocol for bi-directional communication.
    """

    def __init__(
        self,
        ws_url: Optional[str] = None,
        api_key: Optional[str] = None,
        instance_id: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        on_connected: Optional[Callable[[], None]] = None,
        on_disconnected: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize the WebSocket client.

        Args:
            ws_url: WebSocket server URL (default: from WS_SERVER_URL env)
            api_key: API key for authentication (default: from TELEARMS_WS_API_KEY env)
            instance_id: Unique identifier for this simulation instance
            capabilities: List of capabilities this instance supports
            on_connected: Callback when connected and registered
            on_disconnected: Callback when disconnected
        """
        self.ws_url = ws_url or os.environ.get("WS_SERVER_URL", "ws://localhost:8080")
        # TELEARMS_WS_API_KEY for WS server authentication
        self.api_key = api_key or os.environ.get("TELEARMS_WS_API_KEY")
        self.instance_id = instance_id or os.environ.get("INSTANCE_ID", f"sim-{uuid.uuid4().hex[:8]}")
        self.capabilities = capabilities or ["simulation", "recording"]

        self.on_connected = on_connected
        self.on_disconnected = on_disconnected

        self._ws: Optional[Any] = None
        self._session_id: Optional[str] = None
        self._request_id = 0
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._command_handlers: Dict[str, CommandHandler] = {}
        self._current_task_id: Optional[str] = None

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._connected = asyncio.Event()
        self._last_not_connected_log = 0.0
        self._last_send_error_log = 0.0
        self._send_failure_count = 0
        self._first_send_failure_at = 0.0
        self._pending_notifications: List[Dict[str, Any]] = []
        self._pending_notifications_lock = threading.Lock()
        self._pending_notifications_max = 200
        self._reconnect_attempt = 0
        self._disconnected_since: Optional[float] = None
        self._health_log_interval_sec = int(os.environ.get("TELEARMS_WS_HEALTH_LOG_INTERVAL_SEC", "30"))

    def register_command_handler(self, method: str, callback: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """
        Register a handler for incoming commands from the server.

        Args:
            method: The JSON-RPC method name (e.g., "start", "stop", "reset")
            callback: Function that takes params dict and returns result dict
        """
        self._command_handlers[method] = CommandHandler(method=method, callback=callback)
        logger.debug(f"Registered command handler for: {method}")

    def start(self):
        """Start the WebSocket client in a background thread."""
        if self._running:
            logger.warning("Client already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the WebSocket client."""
        self._running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5.0)

    def wait_for_connection(self, timeout: float = 10.0) -> bool:
        """
        Wait for the client to connect and register.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if connected, False if timeout
        """
        if self._loop is None:
            return False

        future = asyncio.run_coroutine_threadsafe(
            asyncio.wait_for(self._connected.wait(), timeout),
            self._loop
        )
        try:
            future.result(timeout=timeout + 1)
            return True
        except (asyncio.TimeoutError, TimeoutError):
            return False

    # ==================== Notification Senders ====================

    def send_heartbeat(self, status: InstanceStatus = InstanceStatus.AVAILABLE):
        """Send a heartbeat notification."""
        self._send_notification("heartbeat", {
            "instance_id": self.instance_id,
            "status": status.value,
            "current_task_id": self._current_task_id,
        })

    def send_progress(
        self,
        task_id: str,
        progress: float,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Send task progress notification.

        Args:
            task_id: The task ID
            progress: Progress value between 0 and 1
            message: Optional progress message
            metadata: Optional additional metadata
        """
        params = {
            "task_id": task_id,
            "progress": progress,
        }
        if message:
            params["message"] = message
        if metadata:
            params["metadata"] = metadata

        self._send_notification("progress", params)

    def send_completed(
        self,
        task_id: str,
        success: bool = True,
        duration_seconds: Optional[float] = None,
        frames_recorded: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Send task completed notification.

        Args:
            task_id: The task ID
            success: Whether the task was successful
            duration_seconds: Task duration
            frames_recorded: Number of frames recorded
            metadata: Additional metadata
        """
        params = {
            "task_id": task_id,
            "success": success,
        }
        if duration_seconds is not None:
            params["duration_seconds"] = duration_seconds
        if frames_recorded is not None:
            params["frames_recorded"] = frames_recorded
        if metadata:
            params["metadata"] = metadata

        sent = self._send_notification("completed", params, queue_on_failure=True)
        self._current_task_id = None
        return sent

    def send_failed(
        self,
        task_id: str,
        error_code: str,
        error_message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Send task failed notification.

        Args:
            task_id: The task ID
            error_code: Error code
            error_message: Error description
            metadata: Additional metadata
        """
        params: Dict[str, Any] = {
            "task_id": task_id,
            "error_code": error_code,
            "error_message": error_message,
        }
        if metadata:
            params["metadata"] = metadata

        sent = self._send_notification("failed", params, queue_on_failure=True)
        self._current_task_id = None
        return sent

    def send_quit(self, task_id: str, reason: str = "user_disconnect", message: Optional[str] = None) -> bool:
        """
        Send task quit notification.

        Args:
            task_id: The task ID
            reason: Quit reason (user_disconnect, timeout, error, cancelled)
            message: Optional message
        """
        params = {
            "task_id": task_id,
            "reason": reason,
        }
        if message:
            params["message"] = message

        sent = self._send_notification("quit", params, queue_on_failure=True)
        self._current_task_id = None
        return sent

    def send_metrics(self, task_id: str, metrics: dict, is_final: bool = False) -> bool:
        """
        Send simulation metrics to backend.

        Args:
            task_id: The task ID
            metrics: Dictionary of simulation metrics
            is_final: Whether this is the final metrics report for the task
        """
        params = {
            "task_id": task_id,
            "metrics": metrics,
            "is_final": is_final,
            "timestamp": time.time(),
        }
        sent = self._send_notification("metrics", params, queue_on_failure=is_final)
        if sent:
            logger.debug(f"Sent metrics for task {task_id}: is_final={is_final}")
        elif is_final:
            logger.warning(f"Queued final metrics for retry: task={task_id}")
        return sent

    # ==================== Internal Methods ====================

    def _run_event_loop(self):
        """Run the asyncio event loop in a background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._connect_and_run())
        except Exception as e:
            logger.error(f"Event loop error: {e}")
        finally:
            self._loop.close()

    async def _connect_and_run(self):
        """Connect to the server and handle messages."""
        while self._running:
            try:
                # Enable WebSocket ping/pong to detect dead connections
                # ping_interval: Send ping every 20 seconds
                # ping_timeout: If no pong within 10 seconds, consider connection dead
                # close_timeout: Wait max 5 seconds when closing
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    self._ws = ws
                    self._disconnected_since = None
                    self._reconnect_attempt = 0
                    logger.info(f"Connected to {self.ws_url}")

                    # Start message handling loop FIRST (as background task)
                    # This is needed so we can receive the register response
                    message_task = asyncio.create_task(self._message_loop(ws))

                    # Now register (response will be handled by message_task)
                    await self._register()
                    await self._send_heartbeat_once()

                    # Start heartbeat task
                    heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                    health_task = asyncio.create_task(self._connection_health_loop())

                    try:
                        # Wait for message loop to complete (connection closed)
                        await message_task
                    finally:
                        heartbeat_task.cancel()
                        health_task.cancel()
                        message_task.cancel()

            except websockets.exceptions.ConnectionClosedError as e:
                logger.warning(f"Connection closed with error: code={e.code}, reason={e.reason}")
            except websockets.exceptions.ConnectionClosedOK as e:
                logger.info(f"Connection closed normally: code={e.code}, reason={e.reason}")
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"Connection closed: {e}")
            except Exception as e:
                logger.error(f"Connection error: {e}", exc_info=True)

            self._ws = None
            self._session_id = None
            self._connected.clear()
            self._reconnect_attempt += 1
            if self._disconnected_since is None:
                self._disconnected_since = time.monotonic()
            disconnected_for = time.monotonic() - self._disconnected_since
            logger.warning(
                f"WS disconnected: attempt={self._reconnect_attempt}, disconnected_for={disconnected_for:.1f}s"
            )

            if self.on_disconnected:
                self.on_disconnected()

            if self._running:
                logger.info("Reconnecting in 5 seconds...")
                await asyncio.sleep(5)

    async def _send_heartbeat_once(self):
        """Send a one-off heartbeat right after registration as connectivity self-check."""
        try:
            if not self._ws:
                return
            status = InstanceStatus.BUSY if self._current_task_id else InstanceStatus.AVAILABLE
            notification = {
                "jsonrpc": "2.0",
                "method": "heartbeat",
                "params": {
                    "instance_id": self.instance_id,
                    "status": status.value,
                    "current_task_id": self._current_task_id,
                },
            }
            await self._ws.send(json.dumps(notification))
            logger.info("Sent post-register heartbeat")
        except Exception as e:
            logger.warning(f"Failed post-register heartbeat: {e}")

    async def _connection_health_loop(self):
        """Periodic connectivity telemetry to make offline windows obvious in logs."""
        while True:
            await asyncio.sleep(self._health_log_interval_sec)
            logger.info(
                "WS health: connected=%s, session=%s, task=%s, queued_notifications=%d",
                self.is_connected,
                self._session_id,
                self._current_task_id,
                len(self._pending_notifications),
            )

    async def _message_loop(self, ws):
        """Handle incoming WebSocket messages."""
        async for message in ws:
            await self._handle_message(message)

    async def _register(self):
        """Register this instance with the server."""
        result = await self._send_request("register", {
            "api_key": self.api_key,
            "instance_id": self.instance_id,
            "capabilities": self.capabilities,
            "metadata": {
                "version": "1.0.0",
            }
        })

        if result.get("success"):
            self._session_id = result.get("session_id")
            logger.info(f"Registered as {self.instance_id}, session: {self._session_id}")
            self._connected.set()
            await self._flush_pending_notifications()

            if self.on_connected:
                self.on_connected()
        else:
            raise Exception(f"Registration failed: {result}")

    def _queue_notification(self, method: str, params: dict):
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }

        with self._pending_notifications_lock:
            self._pending_notifications.append(notification)
            if len(self._pending_notifications) > self._pending_notifications_max:
                dropped = len(self._pending_notifications) - self._pending_notifications_max
                self._pending_notifications = self._pending_notifications[-self._pending_notifications_max:]
                logger.warning(f"Dropped {dropped} oldest queued notifications")

    async def _flush_pending_notifications(self):
        with self._pending_notifications_lock:
            if not self._pending_notifications:
                return
            pending = list(self._pending_notifications)
            self._pending_notifications.clear()

        if not self._ws:
            with self._pending_notifications_lock:
                self._pending_notifications = pending + self._pending_notifications
            return

        logger.info(f"Flushing {len(pending)} queued notifications")
        failed_index = None
        for index, notification in enumerate(pending):
            try:
                await self._ws.send(json.dumps(notification))
            except Exception as e:
                failed_index = index
                logger.warning(
                    f"Failed while flushing queued notification {notification.get('method')}: {e}"
                )
                break

        if failed_index is not None:
            with self._pending_notifications_lock:
                self._pending_notifications = pending[failed_index:] + self._pending_notifications

    async def _heartbeat_loop(self):
        """Send periodic heartbeats.

        IMPORTANT: This runs inside the asyncio event loop, so we must use
        `await self._ws.send()` directly. Using the sync `_send_notification()`
        would deadlock because it calls `asyncio.run_coroutine_threadsafe()`
        + blocking `future.result()` on the same event loop thread.
        """
        while True:
            await asyncio.sleep(30)
            try:
                if not self._ws:
                    logger.warning("Cannot send heartbeat: not connected")
                    continue

                status = InstanceStatus.BUSY if self._current_task_id else InstanceStatus.AVAILABLE
                notification = {
                    "jsonrpc": "2.0",
                    "method": "heartbeat",
                    "params": {
                        "instance_id": self.instance_id,
                        "status": status.value,
                        "current_task_id": self._current_task_id,
                    },
                }
                await self._ws.send(json.dumps(notification))
                logger.debug("Sent heartbeat (async)")
                self._send_failure_count = 0
                self._first_send_failure_at = 0.0
            except Exception as e:
                logger.warning(f"Failed to send heartbeat: {e}")

    async def _handle_message(self, raw_message: str):
        """Handle an incoming WebSocket message."""
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON: {raw_message}")
            return

        # Check if this is a response to a pending request
        if "id" in message and ("result" in message or "error" in message):
            request_id = str(message["id"])
            if request_id in self._pending_requests:
                future = self._pending_requests.pop(request_id)
                if "error" in message:
                    future.set_exception(JsonRpcError(
                        message["error"]["code"],
                        message["error"]["message"],
                        message["error"].get("data")
                    ))
                else:
                    future.set_result(message.get("result", {}))
            return

        # This is a request or notification from the server
        if "method" in message:
            await self._handle_command(message)

    async def _handle_command(self, message: dict):
        """Handle an incoming command from the server."""
        method = message["method"]
        params = message.get("params", {})
        request_id = message.get("id")
        command_task_id = params.get("task_id")

        logger.info(f"Received command: {method}")

        # Find handler
        handler = self._command_handlers.get(method)

        if handler:
            try:
                # Run potentially blocking command handlers off the asyncio event loop.
                # Handlers like stop/start can do network + file I/O + SDK teardown,
                # which would otherwise stall ping/pong and trigger disconnects.
                result = await asyncio.to_thread(handler.callback, params)
                accepted = bool((result or {}).get("accepted", True))

                # Keep local task pointer aligned with command lifecycle only after handler result.
                # This prevents stale "busy" state when start is rejected/failed.
                if method == "start":
                    if accepted:
                        self._current_task_id = command_task_id
                    else:
                        logger.warning(
                            "Start command rejected; keeping current task unchanged. "
                            f"requested_task={command_task_id}, current_task={self._current_task_id}"
                        )
                elif method == "stop":
                    if accepted:
                        if not command_task_id or command_task_id == self._current_task_id:
                            self._current_task_id = None
                        else:
                            logger.warning(
                                "Stop command task mismatch; current task not cleared. "
                                f"stop_task={command_task_id}, current_task={self._current_task_id}"
                            )

                if request_id:
                    await self._send_response(request_id, result)
            except Exception as e:
                logger.error(f"Command handler error: {e}")
                if request_id:
                    await self._send_error_response(request_id, -32603, str(e))
        else:
            logger.warning(f"No handler for command: {method}")
            if request_id:
                # Send default acceptance response
                await self._send_response(request_id, {"accepted": True})

    async def _send_request(self, method: str, params: dict) -> dict:
        """Send a JSON-RPC request and wait for response."""
        if not self._ws:
            raise Exception("Not connected")
        if not self._loop:
            raise Exception("Event loop not available")

        self._request_id += 1
        request_id = str(self._request_id)

        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }

        loop = self._loop
        if loop is None:
            raise Exception("Event loop not available")
        future = loop.create_future()
        self._pending_requests[request_id] = future

        await self._ws.send(json.dumps(request))
        logger.debug(f"Sent request: {method}")

        try:
            return await asyncio.wait_for(future, timeout=10.0)
        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            raise Exception(f"Request timeout: {method}")

    def _send_notification(self, method: str, params: dict, queue_on_failure: bool = False) -> bool:
        """Send a JSON-RPC notification (no response expected)."""
        if not self._ws or not self._loop:
            now = time.monotonic()
            if now - self._last_not_connected_log > 10.0:
                logger.warning(f"Cannot send notification {method}: not connected")
                self._last_not_connected_log = now
            if queue_on_failure:
                self._queue_notification(method, params)
            return False

        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }

        send_timeout_seconds = 10.0
        # Check if send fails (connection dead)
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._ws.send(json.dumps(notification)),
                self._loop
            )
            future.result(timeout=send_timeout_seconds)
            logger.debug(f"Sent notification: {method}")
            self._send_failure_count = 0
            self._first_send_failure_at = 0.0
        except Exception as e:
            if method == "heartbeat" and self._ws and self._loop:
                try:
                    retry_future = asyncio.run_coroutine_threadsafe(
                        self._ws.send(json.dumps(notification)),
                        self._loop
                    )
                    retry_future.result(timeout=send_timeout_seconds)
                    logger.debug(f"Sent notification on retry: {method}")
                    self._send_failure_count = 0
                    self._first_send_failure_at = 0.0
                    return True
                except Exception as retry_error:
                    e = retry_error

            now = time.monotonic()
            if now - self._last_send_error_log > 10.0:
                logger.error(f"Failed to send notification {method}: {e}")
                self._last_send_error_log = now
            if queue_on_failure:
                self._queue_notification(method, params)
            if method == "heartbeat":
                if self._first_send_failure_at == 0.0 or now - self._first_send_failure_at > 60.0:
                    self._first_send_failure_at = now
                    self._send_failure_count = 1
                else:
                    self._send_failure_count += 1

                if self._send_failure_count < 3:
                    logger.warning(
                        f"Heartbeat send failed ({self._send_failure_count}/3); keeping connection open"
                    )
                    return False

                logger.error("Heartbeat send failed 3 times in 60s; closing connection")

            # Connection is dead or non-heartbeat send failed: trigger reconnect by closing
            if self._ws:
                asyncio.run_coroutine_threadsafe(self._ws.close(), self._loop)
            return False

        return True

    async def _send_response(self, request_id: str, result: dict):
        """Send a JSON-RPC response."""
        if not self._ws:
            return

        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result,
        }

        await self._ws.send(json.dumps(response))

    async def _send_error_response(self, request_id: str, code: int, message: str):
        """Send a JSON-RPC error response."""
        if not self._ws:
            return

        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message,
            }
        }

        await self._ws.send(json.dumps(response))

    @property
    def is_connected(self) -> bool:
        """Check if connected and registered."""
        return self._ws is not None and self._session_id is not None

    @property
    def session_id(self) -> Optional[str]:
        """Get the current session ID."""
        return self._session_id

    @property
    def current_task_id(self) -> Optional[str]:
        """Get the current task ID."""
        return self._current_task_id

