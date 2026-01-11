# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import importlib
import sys
from types import SimpleNamespace


def _setup_genai(monkeypatch):
    import types

    google_mod = types.ModuleType("google")
    genai_mod = types.ModuleType("google.genai")

    class DummyModels:
        def generate_content(self, *a, **k):
            DummyModels.kwargs = {"args": a, "kwargs": k}
            return SimpleNamespace(text="plan")

    class DummyClient:
        def __init__(self, *a, **k):
            self.models = SimpleNamespace(
                generate_content=DummyModels().generate_content
            )

    genai_mod.Client = DummyClient
    genai_mod.types = types.SimpleNamespace(
        GenerateContentConfig=lambda **k: SimpleNamespace(**k),
        ThinkingConfig=lambda **k: SimpleNamespace(**k),
    )
    google_mod.genai = genai_mod
    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)


def test_generate_plan(monkeypatch):
    sys.modules.pop("think.planner", None)
    mod = importlib.import_module("think.planner")

    # Mock generate to return "plan"
    def mock_generate(**kwargs):
        return "plan"

    monkeypatch.setattr("think.planner.generate", mock_generate)
    result = mod.generate_plan("do something")
    assert result == "plan"


def test_planner_main(tmp_path, monkeypatch, capsys):
    sys.modules.pop("think.planner", None)
    mod = importlib.import_module("think.planner")
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setattr(mod, "generate_plan", lambda *a, **k: "ok")
    task = tmp_path / "t.txt"
    task.write_text("hi")
    monkeypatch.setattr("sys.argv", ["think-planner", str(task)])
    mod.main()
    out = capsys.readouterr().out.strip()
    assert out == "ok"


def test_load_prompt_with_mcp_tools(monkeypatch):
    """Test that _load_prompt includes MCP tools when available."""
    sys.modules.pop("think.planner", None)

    # Import and patch
    mod = importlib.import_module("think.planner")

    async def fake_get_mcp_tools():
        return "\n## Available Tools\n\n**test_tool**: A test tool for testing"

    monkeypatch.setattr(mod, "_get_mcp_tools", fake_get_mcp_tools)

    # Test the function
    prompt = mod._load_prompt()

    # Check that tools section was added
    assert "## Available Tools" in prompt
    assert "**test_tool**: A test tool for testing" in prompt


def test_load_prompt_without_mcp_tools(monkeypatch):
    """Test that _load_prompt works when MCP tools are not available."""
    sys.modules.pop("think.planner", None)

    # Import and patch
    mod = importlib.import_module("think.planner")

    async def unavailable_tools():
        raise RuntimeError("MCP not available")

    monkeypatch.setattr(mod, "_get_mcp_tools", unavailable_tools)

    # Test the function
    prompt = mod._load_prompt()

    # Check that it still returns the base prompt without tools
    assert "You are a strategic research planner" in prompt
    assert "## Available Tools" not in prompt
