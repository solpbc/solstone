# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import asyncio
import importlib
import json
import sys
import types
from types import SimpleNamespace

from muse.models import GEMINI_FLASH
from tests.agents_stub import install_agents_stub


async def run_main(mod, argv, stdin_data=None):
    sys.argv = argv
    if stdin_data:
        import io

        sys.stdin = io.StringIO(stdin_data)
    await mod.main_async()


def _setup_genai_stub(monkeypatch):
    google_mod = types.ModuleType("google")
    genai_mod = types.ModuleType("google.genai")

    class DummyChat:
        def __init__(self, model, history=None, config=None):
            self.model = model
            self.history = list(history or [])
            self.config = config

        def get_history(self):
            return list(self.history)

        def record_history(self, content):
            self.history.append(content)

        async def send_message(self, message, config=None):
            DummyChat.kwargs = {
                "message": message,
                "config": config,
                "model": self.model,
            }
            return SimpleNamespace(text="ok")

    class DummyChats:
        def create(self, *, model, config=None, history=None):
            return DummyChat(model, history=history, config=config)

    class DummyClient:
        def __init__(self, *a, **k):
            self.chats = DummyChats()
            self.aio = SimpleNamespace(chats=DummyChats())

    genai_mod.Client = DummyClient
    genai_mod.types = types.SimpleNamespace(
        GenerateContentConfig=lambda **k: SimpleNamespace(**k),
        ToolConfig=lambda **k: SimpleNamespace(**k),
        FunctionCallingConfig=lambda **k: SimpleNamespace(**k),
        ThinkingConfig=lambda **k: SimpleNamespace(**k),
        Content=lambda **k: SimpleNamespace(**k),
        Part=lambda **k: SimpleNamespace(**k),
    )
    google_mod.genai = genai_mod
    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)


def test_google_main(monkeypatch, tmp_path, capsys):
    _setup_genai_stub(monkeypatch)
    install_agents_stub()
    sys.modules.pop("muse.providers.google", None)
    importlib.reload(importlib.import_module("muse.providers.google"))
    mod = importlib.reload(importlib.import_module("muse.agents"))

    journal = tmp_path / "journal"
    journal.mkdir()

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setenv("GOOGLE_API_KEY", "x")

    ndjson_input = json.dumps(
        {
            "prompt": "hello",
            "provider": "google",
            "disable_mcp": True,
        }
    )
    asyncio.run(run_main(mod, ["muse-agents"], stdin_data=ndjson_input))

    out_lines = capsys.readouterr().out.strip().splitlines()
    events = [json.loads(line) for line in out_lines]
    assert events[0]["event"] == "start"
    assert isinstance(events[0]["ts"], int)
    assert events[0]["prompt"] == "hello"
    assert events[0]["persona"] == "default"
    assert events[0]["model"] == GEMINI_FLASH
    assert events[-1]["event"] == "finish"
    assert isinstance(events[-1]["ts"], int)
    assert events[-1]["result"] == "ok"

    # Journal logging is now handled by cortex, not by agents directly
    # So we don't check for journal files here


def test_google_outfile(monkeypatch, tmp_path, capsys):
    _setup_genai_stub(monkeypatch)
    install_agents_stub()
    sys.modules.pop("muse.providers.google", None)
    importlib.reload(importlib.import_module("muse.providers.google"))
    mod = importlib.reload(importlib.import_module("muse.agents"))

    journal = tmp_path / "journal"
    journal.mkdir()
    # out_file = tmp_path / "out.txt"

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setenv("GOOGLE_API_KEY", "x")

    ndjson_input = json.dumps(
        {
            "prompt": "hello",
            "provider": "google",
            "disable_mcp": True,
        }
    )
    asyncio.run(run_main(mod, ["muse-agents"], stdin_data=ndjson_input))

    # Output file functionality was removed in NDJSON-only mode
    # Check stdout instead
    out_lines = capsys.readouterr().out.strip().splitlines()
    events = [json.loads(line) for line in out_lines]
    assert events[0]["event"] == "start"
    assert isinstance(events[0]["ts"], int)
    assert events[0]["prompt"] == "hello"
    assert events[0]["persona"] == "default"
    assert events[0]["model"] == GEMINI_FLASH
    assert events[-1]["event"] == "finish"
    assert isinstance(events[-1]["ts"], int)
    assert events[-1]["result"] == "ok"

    # Journal logging is now handled by cortex, not by agents directly
    # So we don't check for journal files here


def test_google_outfile_error(monkeypatch, tmp_path, capsys):
    _setup_genai_stub(monkeypatch)
    install_agents_stub()

    class ErrorClient:
        async def __aenter__(self):
            raise RuntimeError("boom")

        async def __aexit__(self, exc_type, exc, tb):
            pass

    monkeypatch.setattr(
        "think.utils.create_mcp_client", lambda _url=None: ErrorClient()
    )

    sys.modules.pop("muse.providers.google", None)
    importlib.reload(importlib.import_module("muse.providers.google"))
    mod = importlib.reload(importlib.import_module("muse.agents"))

    journal = tmp_path / "journal"
    journal.mkdir()
    # out_file = tmp_path / "out.txt"

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setenv("GOOGLE_API_KEY", "x")

    ndjson_input = json.dumps(
        {
            "prompt": "hello",
            "provider": "google",
            "mcp_server_url": "http://localhost:5175/mcp",
        }
    )
    asyncio.run(run_main(mod, ["muse-agents"], stdin_data=ndjson_input))

    # Check stdout for error event
    out_lines = capsys.readouterr().out.strip().splitlines()
    events = [json.loads(line) for line in out_lines]
    assert events[-1]["event"] == "error"
    assert isinstance(events[-1]["ts"], int)
    assert events[-1]["error"] == "boom"
    assert "trace" in events[-1]

    # Journal logging is now handled by cortex, not by agents directly
    # So we don't check for journal files here
