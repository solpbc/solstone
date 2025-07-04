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
    with review.app.test_request_context("/chat/api/send", method="POST", json={"message": "hi"}):
        resp = review.send_message()
    assert resp[1] == 500
