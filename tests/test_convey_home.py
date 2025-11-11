import importlib
import json


def test_home_template_renders(tmp_path):
    convey = importlib.import_module("convey")
    convey.journal_root = str(tmp_path)
    with convey.app.test_request_context("/"):
        html = convey.home()
    assert "Journal Dashboard" in html


def test_stats_api(tmp_path):
    convey = importlib.import_module("convey")
    stats = {
        "days": {"20240101": {"activity": 1, "audio_sessions": 1}},
        "totals": {"audio_sessions": 1},
        "total_audio_duration": 3600,
        "total_screen_duration": 7200,
        "token_totals_by_model": {
            "gemini-2.5-flash": {
                "input_tokens": 1000,
                "output_tokens": 500,
                "total_tokens": 1500,
            }
        },
    }
    (tmp_path / "stats.json").write_text(json.dumps(stats))
    convey.journal_root = str(tmp_path)
    with convey.app.test_request_context("/api/stats"):
        resp = convey.stats_data()
    assert resp.json["stats"] == stats
    assert "flow" in resp.json["topics"]
