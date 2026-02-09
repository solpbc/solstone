# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import asyncio
import importlib
import json
import sys
from unittest.mock import AsyncMock

from tests.conftest import setup_google_genai_stub
from tests.test_google import make_mock_process
from think.models import GEMINI_FLASH


async def run_main(mod, argv, stdin_data=None):
    sys.argv = argv
    if stdin_data:
        import io

        sys.stdin = io.StringIO(stdin_data)
    await mod.main_async()


def test_google_thinking_events(monkeypatch, tmp_path, capsys):
    setup_google_genai_stub(monkeypatch, with_thinking=True)

    sys.modules.pop("think.providers.google", None)
    importlib.reload(importlib.import_module("think.providers.google"))
    mod = importlib.reload(importlib.import_module("think.agents"))

    journal = tmp_path / "journal"
    journal.mkdir()

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setattr(
        "think.providers.cli.shutil.which",
        lambda name: "/usr/bin/gemini" if name == "gemini" else None,
    )

    # Simulate CLI output: thinking text before a tool call triggers thinking event
    stdout_lines = [
        json.dumps(
            {
                "type": "init",
                "timestamp": 100,
                "session_id": "sess-think",
                "model": "gemini-2.5-flash",
            }
        ),
        json.dumps(
            {
                "type": "message",
                "role": "assistant",
                "delta": True,
                "content": "I need to analyze this step by step.",
            }
        ),
        json.dumps(
            {
                "type": "tool_use",
                "timestamp": 150,
                "tool_name": "search_insights",
                "tool_id": "t1",
                "parameters": {"query": "hello"},
            }
        ),
        json.dumps(
            {
                "type": "tool_result",
                "timestamp": 200,
                "tool_id": "t1",
                "status": "success",
                "output": "result data",
            }
        ),
        json.dumps(
            {
                "type": "message",
                "role": "assistant",
                "delta": True,
                "content": "ok",
            }
        ),
        json.dumps(
            {
                "type": "result",
                "timestamp": 300,
                "status": "success",
                "stats": {
                    "total_tokens": 20,
                    "input_tokens": 10,
                    "output_tokens": 10,
                },
            }
        ),
    ]
    process = make_mock_process(stdout_lines)
    monkeypatch.setattr(
        "think.providers.cli.asyncio.create_subprocess_exec",
        AsyncMock(return_value=process),
    )

    ndjson_input = json.dumps(
        {
            "prompt": "hello",
            "provider": "google",
            "model": GEMINI_FLASH,
            "tools": ["search_insights"],
        }
    )
    asyncio.run(run_main(mod, ["sol agents"], stdin_data=ndjson_input))

    out_lines = capsys.readouterr().out.strip().splitlines()
    events = [json.loads(line) for line in out_lines]

    # Check that we have start, thinking, and finish events
    assert events[0]["event"] == "start"
    assert isinstance(events[0]["ts"], int)
    assert "hello" in events[0]["prompt"]

    # Look for thinking event (flushed when tool_use arrives)
    thinking_events = [e for e in events if e["event"] == "thinking"]
    assert len(thinking_events) == 1
    assert thinking_events[0]["summary"] == "I need to analyze this step by step."
    assert thinking_events[0]["model"] == GEMINI_FLASH
    assert isinstance(thinking_events[0]["ts"], int)

    assert events[-1]["event"] == "finish"
    assert events[-1]["result"] == "ok"
