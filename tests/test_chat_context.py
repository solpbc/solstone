# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import json
from datetime import datetime

from muse.chat_context import pre_process


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
        muse="unified",
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


def test_chat_context_routine_section(monkeypatch, tmp_path):
    """Routine outputs appear in chat context when recent."""
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    monkeypatch.setattr("think.conversation.build_memory_context", lambda **kw: "")

    routines_dir = tmp_path / "routines"
    routines_dir.mkdir()
    routine_id = "test-routine-123"
    config = {
        routine_id: {
            "id": routine_id,
            "name": "Morning Briefing",
            "cadence": "0 8 * * *",
            "enabled": True,
            "last_run": datetime.now().isoformat(),
        }
    }
    (routines_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    output_dir = routines_dir / routine_id
    output_dir.mkdir()
    today = datetime.now().strftime("%Y%m%d")
    (output_dir / f"{today}.md").write_text(
        "Your day looks clear with one meeting at 2pm.",
        encoding="utf-8",
    )

    result = pre_process({"user_instruction": "Base instruction."})

    assert result is not None
    assert "## Recent Routine Outputs" in result["user_instruction"]
    assert "Morning Briefing" in result["user_instruction"]
