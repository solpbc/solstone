# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import asyncio
import importlib
import json
import sys

from muse.models import GEMINI_FLASH
from tests.agents_stub import install_agents_stub
from tests.conftest import setup_google_genai_stub


async def run_main(mod, argv, stdin_data=None):
    sys.argv = argv
    if stdin_data:
        import io

        sys.stdin = io.StringIO(stdin_data)
    await mod.main_async()


def test_google_main(monkeypatch, tmp_path, capsys):
    setup_google_genai_stub(monkeypatch, with_thinking=False)
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
            "model": GEMINI_FLASH,
            "disable_mcp": True,
        }
    )
    asyncio.run(run_main(mod, ["sol agents"], stdin_data=ndjson_input))

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


def test_google_mcp_error(monkeypatch, tmp_path, capsys):
    setup_google_genai_stub(monkeypatch, with_thinking=False)
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

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setenv("GOOGLE_API_KEY", "x")

    ndjson_input = json.dumps(
        {
            "prompt": "hello",
            "provider": "google",
            "model": GEMINI_FLASH,
            "mcp_server_url": "http://localhost:6270/mcp",
        }
    )
    asyncio.run(run_main(mod, ["sol agents"], stdin_data=ndjson_input))

    # Check stdout for error event
    out_lines = capsys.readouterr().out.strip().splitlines()
    events = [json.loads(line) for line in out_lines]
    assert events[-1]["event"] == "error"
    assert isinstance(events[-1]["ts"], int)
    assert events[-1]["error"] == "boom"
    assert "trace" in events[-1]
