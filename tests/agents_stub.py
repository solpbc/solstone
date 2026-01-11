# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import sys
import types
from types import SimpleNamespace


def install_agents_stub():
    """Install a dummy `agents` package with the pieces used by `muse.providers.openai`."""
    agents_stub = types.ModuleType("agents")
    agents_mcp_stub = types.ModuleType("agents.mcp")
    agents_mcp_server_stub = types.ModuleType("agents.mcp.server")
    agents_items_stub = types.ModuleType("agents.items")
    agents_run_stub = types.ModuleType("agents.run")
    agents_model_settings_stub = types.ModuleType("agents.model_settings")

    last_kwargs: dict[str, dict] = {}

    class DummyAgent:
        def __init__(self, *_, **kwargs):
            last_kwargs.update(kwargs)

    class StreamResult:
        def __init__(self, events=None):
            self.events = events or []
            self.final_output = "ok"

        async def stream_events(self):
            for event in self.events:
                yield event

    class DummyRunner:
        called = False
        events_to_stream = []

        @staticmethod
        def run_streamed(
            agent, input, session=None, run_config=None, max_turns=None
        ):  # noqa: D417
            DummyRunner.called = True
            return StreamResult(DummyRunner.events_to_stream)

    class DummySession:
        """Minimal async session used by SQLiteSession."""

        def __init__(self, *_, conversation_id=None, session_id=None, **__):
            self._items: list[dict] = []
            # Support both conversation_id and session_id parameters
            self.session_id = session_id or conversation_id or "test-session"
            self.conversation_id = self.session_id
            self._session_id = self.session_id

        async def get_items(self, limit=None):
            if limit is None:
                return list(self._items)
            return list(self._items)[-limit:]

        async def add_items(self, items):
            if not isinstance(items, list):
                raise TypeError("items must be a list")
            self._items.extend(items)

        async def pop_item(self):
            if self._items:
                return self._items.pop()
            return None

    class DummyConversationSession(DummySession):
        pass

    agents_stub.Agent = DummyAgent
    agents_stub.Runner = DummyRunner
    agents_stub.Session = DummySession
    agents_stub.SQLiteSession = DummySession
    agents_stub.OpenAIConversationsSession = DummyConversationSession
    agents_stub.TResponseInputItem = dict
    agents_stub.set_default_openai_key = lambda key: None
    agents_run_stub.RunConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    agents_model_settings_stub.ModelSettings = lambda **kwargs: SimpleNamespace(
        **kwargs
    )
    agents_stub.RunConfig = agents_run_stub.RunConfig
    agents_stub.ModelSettings = agents_model_settings_stub.ModelSettings

    class DummyMCPServer:
        def __init__(self, *_, **kwargs):
            last_kwargs["mcp_server"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

    agents_mcp_server_stub.MCPServerStreamableHttp = DummyMCPServer

    agents_items_stub.MessageOutputItem = SimpleNamespace
    agents_items_stub.ToolCallItem = SimpleNamespace
    agents_items_stub.ToolCallOutputItem = SimpleNamespace
    agents_items_stub.ReasoningItem = SimpleNamespace

    for key in list(sys.modules.keys()):
        if key == "agents" or key.startswith("agents."):
            sys.modules.pop(key)
    sys.modules["agents"] = agents_stub
    sys.modules["agents.mcp"] = agents_mcp_stub
    sys.modules["agents.mcp.server"] = agents_mcp_server_stub
    sys.modules["agents.items"] = agents_items_stub
    sys.modules["agents.run"] = agents_run_stub
    sys.modules["agents.model_settings"] = agents_model_settings_stub
    sys.modules.pop("muse.providers.openai", None)

    return last_kwargs, DummyRunner
