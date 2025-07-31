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
    _setup_genai(monkeypatch)
    mod = importlib.import_module("think.planner")
    result = mod.generate_plan("do something", api_key="x")
    assert result == "plan"


def test_planner_main(tmp_path, monkeypatch, capsys):
    mod = importlib.import_module("think.planner")
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setattr(mod, "generate_plan", lambda *a, **k: "ok")
    task = tmp_path / "t.txt"
    task.write_text("hi")
    monkeypatch.setattr("sys.argv", ["think-planner", str(task)])
    mod.main()
    out = capsys.readouterr().out.strip()
    assert out == "ok"
