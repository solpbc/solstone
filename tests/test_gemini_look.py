import importlib

from PIL import Image


def test_gemini_describe_region(tmp_path, monkeypatch):
    mod = importlib.import_module("see.gemini_look")

    # Mock the gemini_generate function
    def mock_gemini_generate(**kwargs):
        return '{"ok": true}'

    monkeypatch.setattr("see.gemini_look.gemini_generate", mock_gemini_generate)
    monkeypatch.setattr(mod, "_system_instruction", "sys")

    img = Image.new("RGB", (10, 10), "white")
    box = [0, 0, 5, 5]
    entities = tmp_path / "e.md"
    entities.write_text("")
    result = mod.gemini_describe_region(img, box, entities_text=str(entities))
    assert result["result"]["ok"]
