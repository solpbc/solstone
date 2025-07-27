import asyncio
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
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with review.app.test_request_context(
        "/chat/api/send", method="POST", json={"message": "hi", "backend": "google"}
    ):
        resp = asyncio.run(review.send_message())
    assert resp[1] == 500


def test_send_message_success(monkeypatch):
    review = importlib.import_module("dream")
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "x")

    class DummyAgent:
        def __init__(self):
            self.history = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def run(self, prompt):
            self.history.append({"role": "user", "content": prompt})
            self.history.append({"role": "assistant", "content": "pong"})
            return "pong"

    monkeypatch.setattr("dream.views.chat.GoogleAgent", DummyAgent)
    monkeypatch.setattr("dream.views.chat.OpenAIAgent", DummyAgent)

    with review.app.test_request_context(
        "/chat/api/send", method="POST", json={"message": "hi", "backend": "google"}
    ):
        resp = asyncio.run(review.send_message())
    assert resp.json == {"text": "pong"}
    assert review.state.chat_agent.history[-2:] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "pong"},
    ]


def test_send_message_openai(monkeypatch):
    review = importlib.import_module("dream")
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "x")

    class DummyAgent:
        def __init__(self):
            self.history = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def run(self, prompt):
            self.history.append({"role": "user", "content": prompt})
            self.history.append({"role": "assistant", "content": "pong"})
            return "pong"

    monkeypatch.setattr("dream.views.chat.GoogleAgent", DummyAgent)
    monkeypatch.setattr("dream.views.chat.OpenAIAgent", DummyAgent)

    with review.app.test_request_context(
        "/chat/api/send", method="POST", json={"message": "hi", "backend": "openai"}
    ):
        resp = asyncio.run(review.send_message())
    assert resp.json == {"text": "pong"}
    assert review.state.chat_backend == "openai"


def test_history_and_clear(monkeypatch):
    review = importlib.import_module("dream")
    monkeypatch.setenv("GOOGLE_API_KEY", "x")

    class DummyAgent:
        def __init__(self):
            self.history = [
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "b"},
            ]

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def run(self, prompt):
            self.history.append({"role": "user", "content": prompt})
            self.history.append({"role": "assistant", "content": "pong"})
            return "pong"

    review.state.chat_agent = DummyAgent()

    with review.app.test_request_context("/chat/api/history"):
        resp = review.chat_history()
    assert resp.json == {
        "history": [
            {"role": "user", "text": "u"},
            {"role": "assistant", "text": "b"},
        ]
    }

    with review.app.test_request_context("/chat/api/clear", method="POST"):
        resp = asyncio.run(review.clear_history())
    assert resp.json == {"ok": True}
    assert review.state.chat_agent is None
