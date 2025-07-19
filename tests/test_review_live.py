import importlib


def test_live_page_renders():
    review = importlib.import_module("dream")
    with review.app.test_request_context("/live"):
        html = review.live_page()
    assert "Join" in html


def test_live_join(monkeypatch):
    review = importlib.import_module("dream")
    called = {}

    def fake_start(ws_url, callback, use_speaker=True):
        called["ws"] = ws_url
        called["started"] = True
        return type("T", (), {"is_alive": lambda self: True})()

    monkeypatch.setattr("dream.views.live.start_thread", fake_start)
    with review.app.test_request_context("/live/api/join", method="POST", json={}):
        resp = review.live_join()
    assert resp.json == {"status": "started"}
    assert called.get("started")
