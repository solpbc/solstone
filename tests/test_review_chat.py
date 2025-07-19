import importlib


def test_chat_page_renders(tmp_path):
    review = importlib.import_module("dream")
    review.journal_root = str(tmp_path)
    with review.app.test_request_context("/chat"):
        html = review.chat_page()
    assert "Chat" in html


def test_send_message_no_key(monkeypatch):
    review = importlib.import_module("dream")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with review.app.test_request_context(
        "/chat/api/send", method="POST", json={"message": "hi"}
    ):
        resp = review.send_message()
    assert resp[1] == 500


def test_send_message_success(monkeypatch):
    review = importlib.import_module("dream")
    monkeypatch.setenv("GOOGLE_API_KEY", "x")

    def fake_gemini(msg, attach, key):
        review.state.chat_history.append({"role": "user", "text": msg})
        review.state.chat_history.append({"role": "bot", "text": "pong"})
        return "pong"

    monkeypatch.setattr("dream.views.chat.ask_gemini", fake_gemini)

    with review.app.test_request_context(
        "/chat/api/send", method="POST", json={"message": "hi"}
    ):
        resp = review.send_message()
    assert resp.json == {"text": "pong"}
    assert review.state.chat_history[-2:] == [
        {"role": "user", "text": "hi"},
        {"role": "bot", "text": "pong"},
    ]


def test_history_and_clear(monkeypatch):
    review = importlib.import_module("dream")
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    review.state.chat_history = [
        {"role": "user", "text": "u"},
        {"role": "bot", "text": "b"},
    ]

    with review.app.test_request_context("/chat/api/history"):
        resp = review.chat_history()
    assert resp.json == {"history": review.state.chat_history}

    with review.app.test_request_context("/chat/api/clear", method="POST"):
        resp = review.clear_history()
    assert resp.json == {"ok": True}
    assert review.state.chat_history == []
