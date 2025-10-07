"""Flask utilities for Cortex agent interactions and event streaming."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Callable, Dict, List, Optional

import markdown  # type: ignore

from muse.cortex_client import cortex_watch

from . import state
from .push import push_server

# Note: The async infrastructure has been removed since cortex_client.py
# now provides synchronous functions directly

logger = logging.getLogger(__name__)

_WATCH_LOCK = threading.Lock()
_WATCH_THREAD: Optional[threading.Thread] = None
_WATCH_STOP: Optional[threading.Event] = None


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


def _broadcast_cortex_event(event: Dict[str, Any]) -> Optional[bool]:
    """Broadcast Cortex event to all connected clients with server-side HTML rendering."""
    # Add server-rendered HTML for finish and error events
    if event.get("event") == "finish":
        result_text = event.get("result", "")
        event["html"] = markdown.markdown(result_text, extensions=["extra"])
    elif event.get("event") == "error":
        # Format error message with emoji and code blocks
        error_msg = event.get("error", "Unknown error")
        trace = event.get("trace", "")
        error_text = (
            f"❌ **Error**: {error_msg}\n\n```\n{trace}\n```"
            if trace
            else f"❌ **Error**: {error_msg}"
        )
        event["html"] = markdown.markdown(error_text, extensions=["extra"])
        event["result"] = error_text

    # Broadcast to all views so clients can filter by agent_id
    for view in ["chat", "entities", "domains"]:
        payload = build_cortex_event_payload(event, view=view)
        try:
            push_server.push(payload)
        except Exception:  # pragma: no cover - defensive against socket errors
            logger.exception("Failed to broadcast Cortex event to view %s", view)
    return True


def _run_cortex_watcher() -> None:
    assert _WATCH_STOP is not None
    backoff = 1.0
    while not _WATCH_STOP.is_set():
        try:
            _ensure_journal_env()
            cortex_watch(_broadcast_cortex_event, stop_event=_WATCH_STOP)
            break
        except ValueError:
            logger.debug("Cortex watcher waiting for JOURNAL_PATH; retrying")
        except Exception:  # pragma: no cover - guard against unexpected failures
            logger.exception("Cortex watcher crashed; retrying")
        if _WATCH_STOP.wait(backoff):
            break
        backoff = min(backoff * 2, 30.0)


def start_cortex_event_watcher() -> None:
    global _WATCH_THREAD, _WATCH_STOP
    with _WATCH_LOCK:
        if _WATCH_THREAD and _WATCH_THREAD.is_alive():
            return
        _WATCH_STOP = threading.Event()
        _WATCH_THREAD = threading.Thread(
            target=_run_cortex_watcher, name="cortex-watch", daemon=True
        )
        _WATCH_THREAD.start()


def stop_cortex_event_watcher(timeout: float = 5.0) -> None:
    with _WATCH_LOCK:
        stop_event = _WATCH_STOP
        thread = _WATCH_THREAD
        if stop_event:
            stop_event.set()
    if thread:
        thread.join(timeout)
