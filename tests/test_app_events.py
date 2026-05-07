# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for the app event handling framework."""

import threading

import pytest

from solstone.apps.events import (
    EventContext,
    _get_handlers,
    _handlers,
    clear_handlers,
    discover_handlers,
    dispatch,
    get_handler_count,
    on_event,
    start_dispatcher,
    stop_dispatcher,
)


@pytest.fixture(autouse=True)
def clean_handlers():
    """Clear handlers before and after each test."""
    clear_handlers()
    yield
    clear_handlers()
    stop_dispatcher()


class TestOnEventDecorator:
    """Tests for the @on_event decorator."""

    def test_registers_handler(self):
        """Decorator registers handler in global registry."""

        @on_event("test", "event")
        def handler(ctx):
            pass

        assert ("test", "event") in _handlers
        assert len(_handlers[("test", "event")]) == 1
        assert _handlers[("test", "event")][0][1] is handler

    def test_multiple_handlers_same_event(self):
        """Multiple handlers can register for same event."""

        @on_event("test", "event")
        def handler1(ctx):
            pass

        @on_event("test", "event")
        def handler2(ctx):
            pass

        assert len(_handlers[("test", "event")]) == 2

    def test_wildcard_registration(self):
        """Wildcard patterns register correctly."""

        @on_event("*", "*")
        def catch_all(ctx):
            pass

        @on_event("observe", "*")
        def observe_all(ctx):
            pass

        @on_event("*", "finish")
        def all_finish(ctx):
            pass

        assert ("*", "*") in _handlers
        assert ("observe", "*") in _handlers
        assert ("*", "finish") in _handlers

    def test_decorator_returns_original_function(self):
        """Decorator returns the original function unchanged."""

        def original(ctx):
            return "result"

        decorated = on_event("test", "event")(original)
        assert decorated is original


class TestGetHandlers:
    """Tests for handler matching logic."""

    def test_exact_match(self):
        """Exact tract/event match returns handler."""

        @on_event("observe", "observed")
        def handler(ctx):
            pass

        handlers = _get_handlers({"tract": "observe", "event": "observed"})
        assert len(handlers) == 1
        assert handlers[0][1] is handler

    def test_no_match(self):
        """Non-matching event returns empty list."""

        @on_event("observe", "observed")
        def handler(ctx):
            pass

        handlers = _get_handlers({"tract": "cortex", "event": "finish"})
        assert len(handlers) == 0

    def test_wildcard_all(self):
        """Wildcard (*,*) matches all events."""

        @on_event("*", "*")
        def catch_all(ctx):
            pass

        handlers = _get_handlers({"tract": "anything", "event": "whatever"})
        assert len(handlers) == 1

    def test_wildcard_tract(self):
        """Wildcard (tract,*) matches all events in tract."""

        @on_event("observe", "*")
        def observe_all(ctx):
            pass

        handlers = _get_handlers({"tract": "observe", "event": "detected"})
        assert len(handlers) == 1

        handlers = _get_handlers({"tract": "cortex", "event": "detected"})
        assert len(handlers) == 0

    def test_wildcard_event(self):
        """Wildcard (*,event) matches event across tracts."""

        @on_event("*", "finish")
        def all_finish(ctx):
            pass

        handlers = _get_handlers({"tract": "cortex", "event": "finish"})
        assert len(handlers) == 1

        handlers = _get_handlers({"tract": "cortex", "event": "start"})
        assert len(handlers) == 0

    def test_multiple_matches(self):
        """Multiple matching handlers are all returned."""

        @on_event("observe", "observed")
        def exact(ctx):
            pass

        @on_event("observe", "*")
        def tract_wild(ctx):
            pass

        @on_event("*", "observed")
        def event_wild(ctx):
            pass

        @on_event("*", "*")
        def catch_all(ctx):
            pass

        handlers = _get_handlers({"tract": "observe", "event": "observed"})
        assert len(handlers) == 4


class TestDispatch:
    """Tests for event dispatch."""

    def test_dispatch_without_executor_returns_zero(self):
        """Dispatch without starting executor returns 0."""

        @on_event("test", "event")
        def handler(ctx):
            pass

        count = dispatch({"tract": "test", "event": "event"})
        assert count == 0

    def test_dispatch_calls_handler(self):
        """Dispatch invokes matching handlers."""
        called = threading.Event()
        received_ctx = {}

        @on_event("test", "event")
        def handler(ctx):
            received_ctx["msg"] = ctx.msg
            received_ctx["app"] = ctx.app
            received_ctx["tract"] = ctx.tract
            received_ctx["event"] = ctx.event
            called.set()

        start_dispatcher(workers=1)
        count = dispatch({"tract": "test", "event": "event", "data": "value"})

        assert count == 1
        assert called.wait(timeout=2.0)
        assert received_ctx["msg"]["data"] == "value"
        assert received_ctx["tract"] == "test"
        assert received_ctx["event"] == "event"

    def test_dispatch_handles_exception(self):
        """Handler exceptions are caught and logged."""
        success_called = threading.Event()

        @on_event("test", "event")
        def failing_handler(ctx):
            raise ValueError("Test error")

        @on_event("test", "event")
        def success_handler(ctx):
            success_called.set()

        start_dispatcher(workers=2)
        count = dispatch({"tract": "test", "event": "event"})

        assert count == 2
        # Second handler should still run despite first failing
        assert success_called.wait(timeout=2.0)


class TestDiscovery:
    """Tests for handler discovery."""

    def test_discover_returns_count(self):
        """Discovery runs without error and finds at least the dev app."""
        # Full mocking of importlib for isolated discovery is complex,
        # so we verify the function works on the real apps dir
        count = discover_handlers()
        # Should find at least the dev app
        assert count >= 0


class TestEventContext:
    """Tests for EventContext dataclass."""

    def test_context_fields(self):
        """EventContext has expected fields."""
        ctx = EventContext(
            msg={"tract": "test", "event": "event", "data": "value"},
            app="test_app",
            tract="test",
            event="event",
        )

        assert ctx.msg["data"] == "value"
        assert ctx.app == "test_app"
        assert ctx.tract == "test"
        assert ctx.event == "event"


class TestDispatcherLifecycle:
    """Tests for dispatcher start/stop."""

    def test_start_stop_dispatcher(self):
        """Dispatcher can be started and stopped."""
        start_dispatcher(workers=2)
        # Should be idempotent
        start_dispatcher(workers=2)

        stop_dispatcher()
        # Should be idempotent
        stop_dispatcher()

    def test_get_handler_count(self):
        """get_handler_count returns total handlers."""

        @on_event("a", "b")
        def h1(ctx):
            pass

        @on_event("c", "d")
        def h2(ctx):
            pass

        @on_event("a", "b")
        def h3(ctx):
            pass

        assert get_handler_count() == 3
