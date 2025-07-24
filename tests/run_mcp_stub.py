import os

from think import mcp_server

CALLS_FILE = os.environ.get("CALLS_FILE")


def log_call(entry):
    if CALLS_FILE:
        with open(CALLS_FILE, "a", encoding="utf-8") as f:
            f.write(entry + "\n")


# Stub the indexer functions


def stub_search_topics(query, limit=5, offset=0):
    log_call(f"topics:{query}:{limit}:{offset}")
    return 1, [{"text": "hello", "metadata": {"day": "20240101", "topic": "foo.md"}}]


def stub_search_raws(query, limit=5, offset=0, day=None):
    log_call(f"raws:{query}:{day}:{limit}:{offset}")
    return 1, [
        {
            "text": "occurred",
            "metadata": {"day": "20240101", "time": "123000", "type": "audio"},
        }
    ]


def stub_search_occurrences(
    query,
    n_results=5,
    *,
    day=None,
    start=None,
    end=None,
    topic=None,
):
    log_call(f"events:{query}:{day}:{topic}:{start}:{end}:{n_results}")
    return [
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


mcp_server.search_topics_impl = stub_search_topics
mcp_server.search_raws_impl = stub_search_raws
mcp_server.search_occurrences_impl = stub_search_occurrences

if __name__ == "__main__":
    mcp_server.mcp.run()
