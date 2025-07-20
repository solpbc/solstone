import os

from think import mcp_server

CALLS_FILE = os.environ.get("CALLS_FILE")


def log_call(entry):
    if CALLS_FILE:
        with open(CALLS_FILE, "a", encoding="utf-8") as f:
            f.write(entry + "\n")


# Stub the indexer functions


def stub_search_topics(journal, query, limit=5, offset=0):
    log_call(f"topics:{query}:{limit}:{offset}")
    return 1, [{"text": "hello", "metadata": {"day": "20240101", "topic": "foo.md"}}]


def stub_search_occurrences(journal, query, limit):
    log_call(f"occurrences:{query}:{limit}")
    return [{"text": "occurred", "metadata": {"day": "20240101", "type": "note"}}]


mcp_server.search_topics_impl = stub_search_topics
mcp_server.search_occurrences_impl = stub_search_occurrences

if __name__ == "__main__":
    mcp = mcp_server.create_server(os.environ["JOURNAL_PATH"])
    mcp.run("stdio")
