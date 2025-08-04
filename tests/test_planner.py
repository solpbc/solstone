import importlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock


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
    _setup_genai(monkeypatch)
    sys.modules.pop("think.planner", None)
    mod = importlib.import_module("think.planner")
    result = mod.generate_plan("do something", api_key="x")
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
    
    # Mock MCP client and tools
    mock_tool = Mock()
    mock_tool.name = "test_tool"
    mock_tool.description = "A test tool for testing"
    
    mock_client = AsyncMock()
    mock_client.list_tools = AsyncMock(return_value=[mock_tool])
    
    # Mock create_mcp_client to return our mock
    def mock_create_mcp_client():
        return mock_client
    
    # Import and patch
    mod = importlib.import_module("think.planner")
    monkeypatch.setattr(mod, "create_mcp_client", mock_create_mcp_client)
    
    # Test the function
    prompt = mod._load_prompt()
    
    # Check that tools section was added
    assert "## Available Tools" in prompt
    assert "**test_tool**: A test tool for testing" in prompt


def test_load_prompt_without_mcp_tools(monkeypatch):
    """Test that _load_prompt works when MCP tools are not available."""
    sys.modules.pop("think.planner", None)
    
    # Mock create_mcp_client to raise an exception
    def mock_create_mcp_client():
        raise RuntimeError("MCP not available")
    
    # Import and patch
    mod = importlib.import_module("think.planner")
    monkeypatch.setattr(mod, "create_mcp_client", mock_create_mcp_client)
    
    # Test the function
    prompt = mod._load_prompt()
    
    # Check that it still returns the base prompt without tools
    assert "You are a strategic task planner" in prompt
    assert "## Available Tools" not in prompt
