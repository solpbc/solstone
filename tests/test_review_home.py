import importlib
import json


def test_home_template_renders(tmp_path):
    review = importlib.import_module("dream")
    review.journal_root = str(tmp_path)
    with review.app.test_request_context("/"):
        html = review.home()
    assert "Journal Dashboard" in html


def test_stats_api(tmp_path):
    review = importlib.import_module("dream")
    stats = {"days": {"20240101": {"activity": 1}}}
    (tmp_path / "stats.json").write_text(json.dumps(stats))
    summary = tmp_path / "summary.md"
    summary.write_text("# Hello\nWorld")
    review.journal_root = str(tmp_path)
    with review.app.test_request_context("/api/stats"):
        resp = review.stats_data()
    assert resp.json["stats"] == stats
    assert "<h1>Hello</h1>" in resp.json["summary_html"]
    assert "day" in resp.json["topics"]
