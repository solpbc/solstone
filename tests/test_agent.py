import importlib
import sys
import types
from types import SimpleNamespace


def test_agent_run(monkeypatch):
    agents_stub = types.ModuleType("openai_agents")

    class DummyAgent:
        def __init__(self, *a, **k):
            pass

        def run(self, prompt):
            return "done"

    def decorator(fn=None, **_):
        if fn is None:
            return lambda f: f
        return fn

    agents_stub.Agent = DummyAgent
    agents_stub.function_tool = decorator
    agents_stub.ResponsesAPIConfig = lambda **k: SimpleNamespace()
    sys.modules["openai_agents"] = agents_stub

    mod = importlib.import_module("think.agent")

    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("JOURNAL_PATH", "/tmp")

    monkeypatch.setattr(mod, "tool_search_ponder", lambda query: "ponder")
    monkeypatch.setattr(mod, "tool_search_occurrences", lambda query: "occ")
    monkeypatch.setattr(mod, "tool_read_markdown", lambda d, f: "md")

    monkeypatch.setattr(mod, "Agent", DummyAgent)
    monkeypatch.setattr(mod, "ResponsesAPIConfig", lambda **k: SimpleNamespace())

    agent = mod.build_agent("gpt-4", 100)
    result = agent.run("Hello")
    assert result == "done"
