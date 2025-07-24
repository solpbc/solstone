import importlib
from types import SimpleNamespace

import pytest


def test_number_lines_and_segments():
    mod = importlib.import_module("think.detect_transcript")
    numbered, lines = mod.number_lines("a\nb\nc\nd")
    assert numbered == "1: a\n2: b\n3: c\n4: d"
    assert lines == ["a", "b", "c", "d"]

    nums = mod.parse_line_numbers("[2,4]", len(lines))
    segments = mod.segments_from_lines(lines, nums)
    assert segments == ["a", "b\nc", "d"]


def test_parse_line_numbers_invalid():
    mod = importlib.import_module("think.detect_transcript")
    with pytest.raises(ValueError):
        mod.parse_line_numbers("not json", 3)
    with pytest.raises(ValueError):
        mod.parse_line_numbers("[]", 3)
    with pytest.raises(ValueError):
        mod.parse_line_numbers("[1,1]", 3)
    with pytest.raises(ValueError):
        mod.parse_line_numbers("[0]", 3)
    with pytest.raises(ValueError):
        mod.parse_line_numbers("[5]", 3)


def test_detect_transcript_segment(monkeypatch):
    mod = importlib.import_module("think.detect_transcript")

    class DummyClient:
        class Models:
            def generate_content(self, **kwargs):
                return SimpleNamespace(text="[3]")

        def __init__(self):
            self.models = self.Models()

    monkeypatch.setattr(mod.genai, "Client", lambda *a, **k: DummyClient())
    monkeypatch.setattr(
        mod,
        "types",
        SimpleNamespace(
            GenerateContentConfig=lambda **k: None,
            ThinkingConfig=lambda **k: None,
        ),
    )
    result = mod.detect_transcript_segment("a\nb\nc\nd", api_key="x")
    assert result == ["a\nb", "c\nd"]
