import importlib
import json


def test_agents_list(monkeypatch, tmp_path):
    review = importlib.import_module("dream")
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "100000.jsonl").write_text(
        json.dumps({"event": "start", "prompt": "hi", "persona": "p", "model": "m"})
    )
    review.journal_root = str(tmp_path)
    monkeypatch.setattr("time.time", lambda: 160)
    with review.app.test_request_context("/agents/api/list"):
        resp = review.agents_list()
    assert resp.json == [
        {
            "id": "100000",
            "start": 100.0,
            "since": "1 minute ago",
            "model": "m",
            "persona": "p",
            "prompt": "hi",
        }
    ]
