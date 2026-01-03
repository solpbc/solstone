# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Bidirectional bridge between Callosum message bus and WebSocket clients.

Receives Callosum events and broadcasts them to connected WebSocket clients.
Also provides emit() for route handlers to send events via the shared connection.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional

from flask_sock import Sock
from simple_websocket import ConnectionClosed

from think.callosum import CallosumConnection

from . import state

logger = logging.getLogger(__name__)

_WATCH_LOCK = threading.Lock()
_CALLOSUM_CONNECTION: Optional[CallosumConnection] = None
_WEBSOCKET_CLIENTS: List[object] = []


def _ensure_journal_env() -> None:
    if state.journal_root and not os.environ.get("JOURNAL_PATH"):
        os.environ["JOURNAL_PATH"] = state.journal_root


def _broadcast_to_websockets(event: dict) -> None:
    """Broadcast event to all connected WebSocket clients."""
    msg = json.dumps(event)
    for ws in list(_WEBSOCKET_CLIENTS):
        try:
            ws.send(msg)
        except ConnectionClosed:
            if ws in _WEBSOCKET_CLIENTS:
                _WEBSOCKET_CLIENTS.remove(ws)


def _broadcast_callosum_event(message: Dict[str, Any]) -> None:
    """Broadcast Callosum event to all connected clients."""
    try:
        _broadcast_to_websockets(message)
    except Exception:  # pragma: no cover - defensive against socket errors
        logger.exception("Failed to broadcast %s event", message.get("tract"))


def start_bridge() -> None:
    """Start listening for Callosum events and forwarding to WebSocket clients."""
    global _CALLOSUM_CONNECTION
    with _WATCH_LOCK:
        if _CALLOSUM_CONNECTION:
            return

        # Ensure JOURNAL_PATH is set
        _ensure_journal_env()

        # Create Callosum connection with callback
        try:
            _CALLOSUM_CONNECTION = CallosumConnection()
            _CALLOSUM_CONNECTION.start(callback=_broadcast_callosum_event)
            logger.info("Callosum bridge connected, forwarding all events to WebSocket")
        except Exception as e:
            logger.warning(f"Failed to start Callosum bridge: {e}")
            _CALLOSUM_CONNECTION = None


def stop_bridge() -> None:
    """Stop the Callosum bridge."""
    global _CALLOSUM_CONNECTION
    with _WATCH_LOCK:
        if _CALLOSUM_CONNECTION:
            _CALLOSUM_CONNECTION.stop()
            _CALLOSUM_CONNECTION = None
            logger.info("Callosum bridge stopped")


def emit(tract: str, event: str, **fields) -> bool:
    """Emit event via shared Callosum connection.

    Non-blocking: queues message for background thread to send.
    If disconnected, message is dropped (with debug logging).

    Args:
        tract: Event category/namespace
        event: Event type
        **fields: Additional event fields

    Returns:
        True if queued successfully, False if bridge not started or queue full
    """
    if _CALLOSUM_CONNECTION:
        return _CALLOSUM_CONNECTION.emit(tract, event, **fields)
    return False


def register_websocket(sock: Sock, path: str = "/ws/events") -> None:
    """Register WebSocket endpoint for event broadcasting.

    Args:
        sock: Flask-Sock instance
        path: WebSocket path (default: /ws/events)
    """

    @sock.route(path, endpoint="events_ws")
    def _handler(ws) -> None:
        _WEBSOCKET_CLIENTS.append(ws)
        try:
            while ws.connected:
                # Just keep connection alive, no need for subscriptions
                ws.receive(timeout=1)
        except ConnectionClosed:
            pass
        finally:
            if ws in _WEBSOCKET_CLIENTS:
                _WEBSOCKET_CLIENTS.remove(ws)
