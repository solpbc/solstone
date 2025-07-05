import importlib
from types import SimpleNamespace


def test_generate_top_summary(monkeypatch):
    utils = importlib.import_module("dream.utils")

    class DummyModels:
        def generate_content(self, **kwargs):
            return SimpleNamespace(text="combined")

    class DummyClient:
        def __init__(self, *a, **k):
            self.models = DummyModels()

    monkeypatch.setattr(utils.genai, "Client", lambda *a, **k: DummyClient())
    info = {"descriptions": {"20240101": "A"}}
    result = utils.generate_top_summary(info, "key")
    assert result == "combined"
