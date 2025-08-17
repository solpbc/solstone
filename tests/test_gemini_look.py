import importlib

from PIL import Image


class DummyClient:
    class Models:
        def generate_content(self, **kwargs):
            class R:
                text = '{"ok": true}'

            return R()

    def __init__(self):
        self.models = self.Models()


def test_gemini_describe_region(tmp_path, monkeypatch):
    mod = importlib.import_module("see.gemini_look")
    monkeypatch.setattr(mod, "_gemini_client", DummyClient())
    monkeypatch.setattr(mod, "_system_instruction", "sys")
    img = Image.new("RGB", (10, 10), "white")
    box = [0, 0, 5, 5]
    entities = tmp_path / "e.md"
    entities.write_text("")
    result = mod.gemini_describe_region(img, box, entities_text=str(entities))
    assert result["result"]["ok"]
