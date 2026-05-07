# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc


def test_append_chat_event_indexes_without_rescan(tmp_path, monkeypatch):
    from solstone.convey.chat_stream import append_chat_event
    from solstone.think.indexer.journal import search_journal

    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))

    total, results = search_journal("nebula phrase X42", stream="chat")
    assert total == 0
    assert results == []

    append_chat_event(
        "owner_message",
        text="unique nebula phrase X42",
        app="sol",
        path="/app/sol",
        facet="work",
    )

    total, results = search_journal("nebula phrase X42", stream="chat")
    assert total == 1
    assert any("nebula phrase x42" in result["text"].lower() for result in results)
    assert {result["metadata"]["stream"] for result in results} == {"chat"}

    append_chat_event(
        "owner_message",
        text="second unique aurora signal Y99",
        app="sol",
        path="/app/sol",
        facet="work",
    )

    total, results = search_journal("aurora signal Y99", stream="chat")
    assert total == 1
    assert any("aurora signal y99" in result["text"].lower() for result in results)
    assert {result["metadata"]["stream"] for result in results} == {"chat"}

    total, results = search_journal("nebula phrase X42", stream="chat")
    assert total == 1
    assert any("nebula phrase x42" in result["text"].lower() for result in results)
