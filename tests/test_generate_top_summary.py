import importlib


def test_generate_top_summary(monkeypatch):
    utils = importlib.import_module("dream.utils")

    # Mock the gemini_generate function
    def mock_gemini_generate(**kwargs):
        return "combined"

    monkeypatch.setattr("dream.utils.gemini_generate", mock_gemini_generate)
    info = {"descriptions": {"20240101": "A"}}
    result = utils.generate_top_summary(info, "key")
    assert result == "combined"
