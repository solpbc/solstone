import importlib


def test_live_page_renders():
    review = importlib.import_module("dream")
    with review.app.test_request_context("/live"):
        html = review.live_page()
    assert "Join" in html


def test_live_join_leave(monkeypatch):
    review = importlib.import_module("dream")
    called = {}

    class FakeThread:
        def __init__(self):
            self.alive = True
            self.stop_called = False

        def is_alive(self):
            return self.alive

    fake_thread = FakeThread()

    def fake_start(ws_url, callback, use_speaker=True):
        called["ws"] = ws_url
        called["started"] = True
        return fake_thread

    def fake_stop(thread):
        if thread is fake_thread:
            fake_thread.alive = False
            called["stopped"] = True

    monkeypatch.setattr("hear.live.start_thread", fake_start)
    monkeypatch.setattr("hear.live.stop_thread", fake_stop)
    with review.app.test_request_context("/live/api/join", method="POST", json={}):
        resp = review.live_join()
    assert resp.json == {"status": "started"}
    assert called.get("started")

    with review.app.test_request_context("/live/api/leave", method="POST", json={}):
        resp = review.live_leave()
    assert resp.json["status"] == "stopped"
    assert called.get("stopped")
