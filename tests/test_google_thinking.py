# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import asyncio
import importlib
import json
import sys

from tests.agents_stub import install_agents_stub
from tests.conftest import setup_google_genai_stub
from think.models import GEMINI_FLASH


async def run_main(mod, argv, stdin_data=None):
    sys.argv = argv
    if stdin_data:
        import io

        sys.stdin = io.StringIO(stdin_data)
    await mod.main_async()


def test_google_thinking_events(monkeypatch, tmp_path, capsys):
    setup_google_genai_stub(monkeypatch, with_thinking=True)
    install_agents_stub()

    sys.modules.pop("think.providers.google", None)
    importlib.reload(importlib.import_module("think.providers.google"))
    mod = importlib.reload(importlib.import_module("think.agents"))

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

    # Check that we have start, thinking, and finish events
    assert events[0]["event"] == "start"
    assert isinstance(events[0]["ts"], int)
    assert events[0]["prompt"] == "hello"

    # Look for thinking event
    thinking_events = [e for e in events if e["event"] == "thinking"]
    assert len(thinking_events) == 1
    assert thinking_events[0]["summary"] == "I need to analyze this step by step."
    assert thinking_events[0]["model"] == GEMINI_FLASH
    assert isinstance(thinking_events[0]["ts"], int)

    assert events[-1]["event"] == "finish"
    assert events[-1]["result"] == "ok"
