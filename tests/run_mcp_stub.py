import os
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if "google" not in sys.modules:
    google_mod = types.ModuleType("google")
    genai_mod = types.ModuleType("google.genai")

    class DummyClient:
        def __init__(self, *a, **k):
            pass

    genai_mod.Client = DummyClient
    genai_mod.types = types.SimpleNamespace(GenerateContentConfig=lambda **k: None)
    google_mod.genai = genai_mod
    sys.modules["google"] = google_mod
    sys.modules["google.genai"] = genai_mod

from think import mcp_tools  # noqa: E402

CALLS_FILE = os.environ.get("CALLS_FILE")


def log_call(entry):
    if CALLS_FILE:
        with open(CALLS_FILE, "a", encoding="utf-8") as f:
            f.write(entry + "\n")


# Stub the indexer functions


def stub_search_summaries(query, limit=5, offset=0, *, topic=None):
    log_call(f"topics:{query}:{limit}:{offset}")
    return 1, [{"text": "hello", "metadata": {"day": "20240101", "topic": "foo.md"}}]


def stub_search_transcripts(query, limit=5, offset=0, day=None):
    log_call(f"raws:{query}:{day}:{limit}:{offset}")
    return 1, [
        {
            "text": "occurred",
            "metadata": {"day": "20240101", "time": "123000", "type": "audio"},
        }
    ]


def stub_search_events(
    query,
    limit=5,
    offset=0,
    *,
    day=None,
    start=None,
    end=None,
    topic=None,
):
    log_call(f"events:{query}:{day}:{topic}:{start}:{end}:{limit}:{offset}")
    return 1, [
        {
            "text": "Standup",
            "metadata": {
                "day": "20240101",
                "topic": "meetings",
                "start": "09:00",
                "end": "09:30",
            },
            "occurrence": {"title": "Standup", "summary": "Daily sync"},
            "score": 0.1,
            "id": "20240101/topics/meetings.json:0",
        }
    ]


mcp_tools.search_summaries_impl = stub_search_summaries
mcp_tools.search_transcripts_impl = stub_search_transcripts
mcp_tools.search_events_impl = stub_search_events

if __name__ == "__main__":
    mcp_tools.mcp.run()
