import importlib
import os


def test_chat_page_renders(tmp_path):
    review = importlib.import_module("convey")
    review.journal_root = str(tmp_path)
    with review.app.test_request_context("/chat"):
        html = review.chat_page()
    assert "Chat" in html


def test_send_message_no_key(monkeypatch, tmp_path):
    review = importlib.import_module("convey")
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with review.app.test_request_context(
        "/chat/api/send", method="POST", json={"message": "hi", "backend": "google"}
    ):
        resp = review.send_message()
    assert resp.status_code == 500
    assert resp.json == {"error": "GOOGLE_API_KEY not set"}


def test_send_message_success(monkeypatch, tmp_path):
    review = importlib.import_module("convey")
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setenv("GOOGLE_API_KEY", "x")

    # Create the agents directory that cortex_request expects
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    def dummy_cortex_request(prompt, persona="default", backend="openai", config=None):
        dummy_cortex_request.called = (prompt, persona, backend, config)
        # Return a fake agent file path
        agent_file = agents_dir / "test_agent_123.jsonl"
        agent_file.touch()
        return str(agent_file)

    monkeypatch.setattr("muse.cortex_client.cortex_request", dummy_cortex_request)

    with review.app.test_request_context(
        "/chat/api/send", method="POST", json={"message": "hi", "backend": "google"}
    ):
        resp = review.send_message()
    assert resp.json == {"agent_id": "test_agent_123"}
    assert dummy_cortex_request.called[0] == "hi"
    assert dummy_cortex_request.called[2] == "google"


def test_send_message_openai(monkeypatch, tmp_path):
    review = importlib.import_module("convey")
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "x")

    # Create the agents directory that cortex_request expects
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    called = {}

    def dummy_cortex_request(prompt, persona="default", backend="openai", config=None):
        called["backend"] = backend
        called["persona"] = persona
        called["config"] = config
        # Return a fake agent file path
        agent_file = agents_dir / "test_agent_456.jsonl"
        agent_file.touch()
        return str(agent_file)

    monkeypatch.setattr("muse.cortex_client.cortex_request", dummy_cortex_request)

    with review.app.test_request_context(
        "/chat/api/send", method="POST", json={"message": "hi", "backend": "openai"}
    ):
        resp = review.send_message()
    assert resp.json["agent_id"] == "test_agent_456"
    assert called["backend"] == "openai"


def test_send_message_anthropic(monkeypatch, tmp_path):
    review = importlib.import_module("convey")
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")

    # Create the agents directory that cortex_request expects
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    called = {}

    def dummy_cortex_request(prompt, persona="default", backend="openai", config=None):
        called["backend"] = backend
        called["persona"] = persona
        called["config"] = config
        # Return a fake agent file path
        agent_file = agents_dir / "test_agent_789.jsonl"
        agent_file.touch()
        return str(agent_file)

    monkeypatch.setattr("muse.cortex_client.cortex_request", dummy_cortex_request)

    with review.app.test_request_context(
        "/chat/api/send", method="POST", json={"message": "hi", "backend": "anthropic"}
    ):
        resp = review.send_message()
    assert resp.json["agent_id"] == "test_agent_789"
    assert called["backend"] == "anthropic"


def test_history_and_clear(monkeypatch):
    review = importlib.import_module("convey")

    with review.app.test_request_context("/chat/api/history"):
        resp = review.chat_history()
    assert resp.json == {"history": []}

    with review.app.test_request_context("/chat/api/clear", method="POST"):
        resp = review.clear_history()
    assert resp.json == {"ok": True}
