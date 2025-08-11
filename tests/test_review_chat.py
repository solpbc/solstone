import importlib
import time


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
        resp = review.send_message()
    assert resp.status_code == 500
    assert resp.json == {"error": "GOOGLE_API_KEY not set"}


def test_send_message_success(monkeypatch):
    review = importlib.import_module("dream")
    monkeypatch.setenv("GOOGLE_API_KEY", "x")

    def dummy_ask_agent_via_cortex(prompt, attachments, backend):
        dummy_ask_agent_via_cortex.called = (prompt, attachments, backend)
        return "pong"

    monkeypatch.setattr(
        "dream.views.chat.ask_agent_via_cortex", dummy_ask_agent_via_cortex
    )

    with review.app.test_request_context(
        "/chat/api/send", method="POST", json={"message": "hi", "backend": "google"}
    ):
        resp = review.send_message()
    assert resp.json == {"text": "pong", "html": "<p>pong</p>"}
    assert dummy_ask_agent_via_cortex.called == ("hi", [], "google")


def test_send_message_openai(monkeypatch):
    review = importlib.import_module("dream")
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "x")

    called = {}

    def dummy_ask_agent_via_cortex(prompt, attachments, backend):
        called["backend"] = backend
        return "pong"

    monkeypatch.setattr(
        "dream.views.chat.ask_agent_via_cortex", dummy_ask_agent_via_cortex
    )

    with review.app.test_request_context(
        "/chat/api/send", method="POST", json={"message": "hi", "backend": "openai"}
    ):
        resp = review.send_message()
    assert resp.json["text"] == "pong"
    assert called["backend"] == "openai"


def test_send_message_anthropic(monkeypatch):
    review = importlib.import_module("dream")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")

    called = {}

    def dummy_ask_agent_via_cortex(prompt, attachments, backend):
        called["backend"] = backend
        return "pong"

    monkeypatch.setattr(
        "dream.views.chat.ask_agent_via_cortex", dummy_ask_agent_via_cortex
    )

    with review.app.test_request_context(
        "/chat/api/send", method="POST", json={"message": "hi", "backend": "anthropic"}
    ):
        resp = review.send_message()
    assert resp.json["text"] == "pong"
    assert called["backend"] == "anthropic"


def test_history_and_clear(monkeypatch):
    review = importlib.import_module("dream")

    with review.app.test_request_context("/chat/api/history"):
        resp = review.chat_history()
    assert resp.json == {"history": []}

    with review.app.test_request_context("/chat/api/clear", method="POST"):
        resp = review.clear_history()
    assert resp.json == {"ok": True}


def test_tool_event_pushed(monkeypatch):
    review = importlib.import_module("dream")
    monkeypatch.setenv("GOOGLE_API_KEY", "x")

    events = []

    monkeypatch.setattr("dream.views.chat.push_server.push", lambda e: events.append(e))

    # Mock cortex client to simulate tool events
    class MockCortexClient:
        def __init__(self):
            self.event_callback = None

        def set_event_callback(self, callback):
            self.event_callback = callback

        def spawn_agent(self, prompt, backend, persona):
            # Simulate tool event
            self.event_callback(
                {
                    "event": "tool_start",
                    "ts": int(time.time() * 1000),
                    "tool": "search_events",
                    "args": {"query": prompt},
                }
            )

    def dummy_ask_agent_via_cortex(prompt, attachments, backend):
        # Simulate the cortex interaction flow
        client = MockCortexClient()
        import dream.views.chat

        client.set_event_callback(dream.views.chat._handle_cortex_event)
        client.spawn_agent(prompt, backend, "default")
        return "pong"

    monkeypatch.setattr(
        "dream.views.chat.ask_agent_via_cortex", dummy_ask_agent_via_cortex
    )

    with review.app.test_request_context(
        "/chat/api/send", method="POST", json={"message": "hi", "backend": "google"}
    ):
        resp = review.send_message()

    assert resp.json["text"] == "pong"
    assert events[0]["view"] == "chat"
    assert events[0]["tool"] == "search_events"
    assert isinstance(events[0]["ts"], int)
