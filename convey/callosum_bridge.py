"""Bridge between Callosum message bus and WebSocket clients.

Listens to all Callosum events (cortex, task, indexer, etc.) and broadcasts them
to connected WebSocket clients.
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


def _event_identifier(event: Dict[str, Any]) -> str:
    agent_id = event.get("agent_id") or ""
    event_type = event.get("event") or ""
    call_id = event.get("call_id") or ""
    tool = event.get("tool") or ""
    ts = event.get("ts") or ""
    # Ensure deterministic string so duplicates can be filtered client-side
    return f"{agent_id}:{event_type}:{call_id}:{tool}:{ts}"


def build_cortex_event_payload(
    event: Dict[str, Any], *, source: str = "cortex", view: str = "chat"
) -> Dict[str, Any]:
    payload = dict(event)
    payload.setdefault("view", view)
    payload["source"] = source
    payload["event_id"] = payload.get("event_id") or _event_identifier(payload)
    return payload


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
    tract = message.get("tract")

    # For cortex events, broadcast to all cortex views with enriched payload
    if tract == "cortex":
        for view in ["chat", "entities", "domains"]:
            payload = build_cortex_event_payload(message, view=view)
            try:
                _broadcast_to_websockets(payload)
            except Exception:  # pragma: no cover - defensive against socket errors
                logger.exception("Failed to broadcast Cortex event to view %s", view)
    else:
        # For all other tracts (task, indexer, etc.), broadcast as-is
        try:
            _broadcast_to_websockets(message)
        except Exception:  # pragma: no cover - defensive against socket errors
            logger.exception("Failed to broadcast %s event", tract)


def start_callosum_bridge() -> None:
    """Start listening for Callosum events and forwarding to WebSocket clients."""
    global _CALLOSUM_CONNECTION
    with _WATCH_LOCK:
        if _CALLOSUM_CONNECTION:
            return

        # Ensure JOURNAL_PATH is set
        _ensure_journal_env()

        # Create Callosum connection with callback
        try:
            _CALLOSUM_CONNECTION = CallosumConnection(callback=_broadcast_callosum_event)
            _CALLOSUM_CONNECTION.connect()
            logger.info("Callosum bridge connected, forwarding all events to WebSocket")
        except Exception as e:
            logger.warning(f"Failed to start Callosum bridge: {e}")
            _CALLOSUM_CONNECTION = None


def stop_callosum_bridge(timeout: float = 5.0) -> None:
    """Stop listening for Callosum events."""
    global _CALLOSUM_CONNECTION
    with _WATCH_LOCK:
        if _CALLOSUM_CONNECTION:
            _CALLOSUM_CONNECTION.close()
            _CALLOSUM_CONNECTION = None
            logger.info("Callosum bridge stopped")


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
