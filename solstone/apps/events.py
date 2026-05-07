# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Server-side event handling framework for apps.

Apps can define event handlers in `events.py` that react to Callosum events.
Handlers are discovered at Convey startup and dispatched via a thread pool.

Usage in apps/my_app/events.py:

    from solstone.apps.events import on_event

    @on_event("observe", "observed")
    def handle_observation(ctx):
        day = ctx.msg.get("day")
        segment = ctx.msg.get("segment")
        # React to completed observation...

    @on_event("cortex", "finish")
    def handle_agent_done(ctx):
        # React to agent completion...

    @on_event("*", "*")  # Wildcard - all events
    def log_all(ctx):
        # Debug logging...

Handlers receive an EventContext with:
    - ctx.msg: The raw Callosum message dict
    - ctx.app: The app name that owns this handler
    - ctx.tract: Event tract (e.g., "observe")
    - ctx.event: Event type (e.g., "observed")

Handlers can access journal path via `from solstone.convey import state` then `state.journal_root`.
"""

from __future__ import annotations

import importlib
import logging
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Default timeout for handler execution (seconds)
DEFAULT_TIMEOUT = 30.0

# Default number of worker threads
DEFAULT_WORKERS = 4


@dataclass
class EventContext:
    """Context passed to event handlers."""

    msg: Dict[str, Any]
    app: str
    tract: str
    event: str


# Handler registry: (tract, event) -> [(app_name, handler_fn), ...]
_handlers: Dict[Tuple[str, str], List[Tuple[str, Callable[[EventContext], None]]]] = {}

# Thread pool for async dispatch
_executor: ThreadPoolExecutor | None = None

# Track which app is currently being imported (for decorator context)
_current_app: str | None = None


def on_event(tract: str, event: str) -> Callable:
    """Decorator to register a function as an event handler.

    Args:
        tract: Callosum tract to match (e.g., "observe", "cortex") or "*" for all
        event: Event type to match (e.g., "observed", "finish") or "*" for all

    Returns:
        Decorator function that registers the handler

    Example:
        @on_event("observe", "observed")
        def handle_observation(ctx: EventContext):
            print(f"Segment {ctx.msg['segment']} observed")
    """

    def decorator(fn: Callable[[EventContext], None]) -> Callable[[EventContext], None]:
        key = (tract, event)
        if key not in _handlers:
            _handlers[key] = []

        # Use current app context from discovery, or infer from module
        app_name = _current_app
        if app_name is None:
            # Fallback: extract from module name (apps.my_app.events -> my_app)
            module = fn.__module__
            if module.startswith("apps.") and ".events" in module:
                parts = module.split(".")
                if len(parts) >= 2:
                    app_name = parts[1]
            if app_name is None:
                app_name = "unknown"

        _handlers[key].append((app_name, fn))
        logger.debug(
            f"Registered handler {fn.__name__} for ({tract}, {event}) in app {app_name}"
        )
        return fn

    return decorator


def discover_handlers() -> int:
    """Discover and load event handlers from apps/*/events.py.

    This function scans the apps/ directory for events.py files and
    dynamically imports them, which triggers @on_event decorators.

    Returns:
        Number of apps with event handlers discovered

    Raises:
        No exceptions - errors are logged but don't prevent other apps from loading
    """
    global _current_app

    apps_dir = Path(__file__).parent

    if not apps_dir.exists():
        logger.debug("No apps/ directory found, skipping event handler discovery")
        return 0

    discovered_count = 0
    total_handlers = 0

    for app_dir in sorted(apps_dir.iterdir()):
        # Skip non-directories and private directories
        if not app_dir.is_dir() or app_dir.name.startswith("_"):
            continue

        events_file = app_dir / "events.py"
        if not events_file.exists():
            continue

        app_name = app_dir.name

        try:
            # Set context for decorator
            _current_app = app_name

            # Import triggers @on_event decorators
            module_name = f"solstone.apps.{app_name}.events"
            importlib.import_module(module_name)

            # Count handlers for this app
            app_handlers = sum(
                1
                for handlers in _handlers.values()
                for (app, _) in handlers
                if app == app_name
            )

            discovered_count += 1
            total_handlers += app_handlers
            logger.info(f"Loaded {app_handlers} event handler(s) from app: {app_name}")
        except Exception as e:
            # Gracefully handle errors - don't break server startup
            logger.error(
                f"Failed to load events from app '{app_name}': {e}", exc_info=True
            )
        finally:
            _current_app = None

    if discovered_count > 0:
        logger.info(
            f"Discovered {total_handlers} event handler(s) from {discovered_count} app(s)"
        )

    return discovered_count


def _get_handlers(
    msg: Dict[str, Any],
) -> List[Tuple[str, Callable[[EventContext], None]]]:
    """Get all handlers matching the given message.

    Matches exact (tract, event), plus wildcards:
    - ("*", "*") matches all events
    - (tract, "*") matches all events in a tract
    - ("*", event) matches event type across all tracts

    Args:
        msg: Callosum message with tract and event fields

    Returns:
        List of (app_name, handler_fn) tuples
    """
    tract = msg.get("tract", "")
    event = msg.get("event", "")

    handlers = []

    # Exact match
    handlers.extend(_handlers.get((tract, event), []))

    # Wildcard: all events in this tract
    if (tract, "*") in _handlers:
        handlers.extend(_handlers[(tract, "*")])

    # Wildcard: this event type in any tract
    if ("*", event) in _handlers:
        handlers.extend(_handlers[("*", event)])

    # Wildcard: all events
    if ("*", "*") in _handlers:
        handlers.extend(_handlers[("*", "*")])

    return handlers


def _run_handler(
    app_name: str,
    handler: Callable[[EventContext], None],
    ctx: EventContext,
) -> None:
    """Run a single handler with error handling.

    Args:
        app_name: Name of the app that owns this handler
        handler: The handler function to call
        ctx: Event context to pass to handler
    """
    try:
        handler(ctx)
    except Exception as e:
        logger.error(
            f"Event handler {handler.__name__} (app: {app_name}) failed: {e}",
            exc_info=True,
        )


def dispatch(msg: Dict[str, Any], timeout: float = DEFAULT_TIMEOUT) -> int:
    """Dispatch a Callosum message to matching handlers.

    Handlers are submitted to the thread pool and this function blocks until
    all handlers complete or timeout. This serializes event processing to
    ensure handlers finish before the next event is processed.

    Each handler is wrapped in error handling so failures don't affect other
    handlers or the caller.

    Args:
        msg: Callosum message dict with tract, event, and other fields
        timeout: Maximum seconds to wait per handler (default: 30)

    Returns:
        Number of handlers invoked
    """
    if _executor is None:
        logger.debug("Event dispatcher not started, skipping dispatch")
        return 0

    handlers = _get_handlers(msg)
    if not handlers:
        return 0

    tract = msg.get("tract", "")
    event = msg.get("event", "")

    futures: List[Tuple[str, str, Future]] = []

    for app_name, handler in handlers:
        ctx = EventContext(
            msg=msg,
            app=app_name,
            tract=tract,
            event=event,
        )
        future = _executor.submit(_run_handler, app_name, handler, ctx)
        futures.append((app_name, handler.__name__, future))

    # Wait for all handlers with timeout (serializes event processing)
    for app_name, handler_name, future in futures:
        try:
            future.result(timeout=timeout)
        except TimeoutError:
            logger.warning(
                f"Event handler {handler_name} (app: {app_name}) timed out after {timeout}s"
            )
        except Exception as e:
            # Should not happen since _run_handler catches exceptions
            logger.error(f"Unexpected error in handler {handler_name}: {e}")

    return len(handlers)


def start_dispatcher(workers: int = DEFAULT_WORKERS) -> None:
    """Start the event dispatcher thread pool.

    Args:
        workers: Number of worker threads (default: 4)
    """
    global _executor

    if _executor is not None:
        logger.debug("Event dispatcher already started")
        return

    _executor = ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="event_handler"
    )
    logger.info(f"Started event dispatcher with {workers} workers")


def stop_dispatcher() -> None:
    """Stop the event dispatcher thread pool gracefully."""
    global _executor

    if _executor is None:
        return

    logger.info("Stopping event dispatcher...")
    _executor.shutdown(wait=True, cancel_futures=False)
    _executor = None
    logger.info("Event dispatcher stopped")


def get_handler_count() -> int:
    """Get the total number of registered handlers.

    Returns:
        Total handler count across all apps and event patterns
    """
    return sum(len(handlers) for handlers in _handlers.values())


def clear_handlers() -> None:
    """Clear all registered handlers. Useful for testing."""
    _handlers.clear()
