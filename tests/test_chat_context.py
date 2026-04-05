# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from talent.chat_context import pre_process


def test_chat_context_appends_conversation_memory(monkeypatch, tmp_path):
    """Conversation memory is appended when recent exchanges exist."""
    from think.conversation import record_exchange
    from think.utils import now_ms

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

    record_exchange(
        ts=now_ms(),
        facet="work",
        user_message="hello",
        agent_response="hi there!",
        talent="unified",
    )

    result = pre_process({"user_instruction": "Base instruction.", "facet": "work"})

    assert result is not None
    assert "## Recent Conversation" in result["user_instruction"]
    assert "hello" in result["user_instruction"]
    assert "hi there!" in result["user_instruction"]


def test_chat_context_no_memory(monkeypatch, tmp_path):
    """Other sections still append when no conversation history exists."""
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

    result = pre_process({"user_instruction": "Base instruction."})

    assert result is not None
    assert "## Recent Conversation" not in result["user_instruction"]


def test_chat_context_always_appends_location_context(monkeypatch):
    """Location context is always appended."""
    monkeypatch.setattr("think.conversation.build_memory_context", lambda **kw: "")

    result = pre_process({"user_instruction": "Base instruction."})

    assert result is not None
    assert "## Location Context" in result["user_instruction"]


def test_chat_context_always_appends_system_health(monkeypatch):
    """System health guidance is always appended."""
    monkeypatch.setattr("think.conversation.build_memory_context", lambda **kw: "")

    result = pre_process({"user_instruction": "Base instruction."})

    assert result is not None
    assert "## System Health" in result["user_instruction"]


def test_chat_context_always_appends_behavioral_defaults(monkeypatch):
    """Behavioral defaults are always appended."""
    monkeypatch.setattr("think.conversation.build_memory_context", lambda **kw: "")

    result = pre_process({"user_instruction": "Base instruction."})

    assert result is not None
    assert "## Behavioral Defaults" in result["user_instruction"]


def test_chat_context_onboarding_observing(monkeypatch):
    """Observing onboarding state appends observation guidance."""
    monkeypatch.setattr("think.conversation.build_memory_context", lambda **kw: "")
    monkeypatch.setattr(
        "think.awareness.get_onboarding", lambda: {"status": "observing"}
    )

    result = pre_process({"user_instruction": "Base instruction."})

    assert result is not None
    assert "## Onboarding Observation Context" in result["user_instruction"]


def test_chat_context_onboarding_ready(monkeypatch):
    """Ready onboarding state appends recommendation guidance."""
    monkeypatch.setattr("think.conversation.build_memory_context", lambda **kw: "")
    monkeypatch.setattr("think.awareness.get_onboarding", lambda: {"status": "ready"})

    result = pre_process({"user_instruction": "Base instruction."})

    assert result is not None
    assert "## Onboarding Observation Complete" in result["user_instruction"]


def test_chat_context_import_awareness_injected(monkeypatch):
    """Import awareness is appended when onboarding is complete and empty."""
    monkeypatch.setattr("think.conversation.build_memory_context", lambda **kw: "")
    monkeypatch.setattr(
        "think.awareness.get_onboarding", lambda: {"status": "complete"}
    )
    monkeypatch.setattr("think.awareness.get_imports", lambda: {"has_imported": False})

    result = pre_process({"user_instruction": "Base instruction."})

    assert result is not None
    assert "## Import Awareness" in result["user_instruction"]


def test_chat_context_import_done_no_nudge(monkeypatch):
    """Import awareness is omitted once imports exist."""
    monkeypatch.setattr("think.conversation.build_memory_context", lambda **kw: "")
    monkeypatch.setattr(
        "think.awareness.get_onboarding", lambda: {"status": "complete"}
    )
    monkeypatch.setattr("think.awareness.get_imports", lambda: {"has_imported": True})

    result = pre_process({"user_instruction": "Base instruction."})

    assert result is not None
    assert "## Import Awareness" not in result["user_instruction"]


def test_chat_context_naming_awareness_default(monkeypatch):
    """Naming awareness is appended when the default agent name is still active."""
    monkeypatch.setattr("think.conversation.build_memory_context", lambda **kw: "")
    monkeypatch.setattr(
        "think.awareness.get_onboarding", lambda: {"status": "complete"}
    )
    monkeypatch.setattr("think.awareness.get_imports", lambda: {"has_imported": True})
    monkeypatch.setattr("think.utils.get_config", lambda: {"agent": {"name": "sol"}})

    result = pre_process({"user_instruction": "Base instruction."})

    assert result is not None
    assert "## Naming Awareness" in result["user_instruction"]


def test_chat_context_naming_awareness_chosen(monkeypatch):
    """Naming awareness is omitted once a custom agent name is chosen."""
    monkeypatch.setattr("think.conversation.build_memory_context", lambda **kw: "")
    monkeypatch.setattr(
        "think.awareness.get_onboarding", lambda: {"status": "complete"}
    )
    monkeypatch.setattr("think.awareness.get_imports", lambda: {"has_imported": True})
    monkeypatch.setattr("think.utils.get_config", lambda: {"agent": {"name": "aria"}})

    result = pre_process({"user_instruction": "Base instruction."})

    assert result is not None
    assert "## Naming Awareness" not in result["user_instruction"]


def test_chat_context_awareness_error_graceful(monkeypatch):
    """Awareness failures do not prevent the base sections from appending."""
    monkeypatch.setattr("think.conversation.build_memory_context", lambda **kw: "")

    def _raise() -> dict:
        raise RuntimeError("boom")

    monkeypatch.setattr("think.awareness.get_onboarding", _raise)

    result = pre_process({"user_instruction": "Base instruction."})

    assert result is not None
    assert "## Location Context" in result["user_instruction"]


def test_chat_context_routines_injected(monkeypatch):
    """Active routines section is appended when routines exist."""
    monkeypatch.setattr("think.conversation.build_memory_context", lambda **kw: "")
    monkeypatch.setattr(
        "think.routines.get_routine_state",
        lambda: [
            {
                "name": "Morning Briefing",
                "cadence": "0 9 * * *",
                "last_run": None,
                "enabled": True,
                "paused_until": None,
                "output_summary": None,
            }
        ],
    )

    result = pre_process({"user_instruction": "Base instruction."})

    assert result is not None
    assert "## Active Routines" in result["user_instruction"]
    assert "Morning Briefing" in result["user_instruction"]


def test_chat_context_routines_omitted_when_empty(monkeypatch):
    """Active routines section is omitted when no routines configured."""
    monkeypatch.setattr("think.conversation.build_memory_context", lambda **kw: "")
    monkeypatch.setattr("think.routines.get_routine_state", lambda: [])

    result = pre_process({"user_instruction": "Base instruction."})

    assert result is not None
    assert "## Active Routines" not in result["user_instruction"]


def test_chat_context_routines_error_graceful(monkeypatch):
    """Routine state failures do not prevent other sections from appending."""
    monkeypatch.setattr("think.conversation.build_memory_context", lambda **kw: "")
    monkeypatch.setattr(
        "think.routines.get_routine_state",
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    result = pre_process({"user_instruction": "Base instruction."})

    assert result is not None
    assert "## Active Routines" not in result["user_instruction"]
    assert "## Location Context" in result["user_instruction"]
