# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import importlib.util
from pathlib import Path

TEMPLATE_VAR_KEYS = {
    "recent_conversation",
    "active_routines",
    "routine_suggestion",
}


def _load_chat_context_module():
    path = Path(__file__).resolve().parents[1] / "talent" / "chat_context.py"
    spec = importlib.util.spec_from_file_location("test_chat_context_local", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_template_vars_result(result):
    assert isinstance(result, dict)
    assert "template_vars" in result
    assert "user_instruction" not in result
    assert set(result["template_vars"]) == TEMPLATE_VAR_KEYS
    return result["template_vars"]


def _read_chat_md() -> str:
    chat_md = Path(__file__).resolve().parents[1] / "talent" / "chat.md"
    return chat_md.read_text(encoding="utf-8")


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

    result = _load_chat_context_module().pre_process(
        {"user_instruction": "Base instruction.", "facet": "work"}
    )

    template_vars = _assert_template_vars_result(result)
    assert "## Recent Conversation" in template_vars["recent_conversation"]
    assert "hello" in template_vars["recent_conversation"]
    assert "hi there!" in template_vars["recent_conversation"]


def test_chat_context_no_memory(monkeypatch, tmp_path):
    """Recent conversation is empty when no conversation history exists."""
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

    result = _load_chat_context_module().pre_process(
        {"user_instruction": "Base instruction."}
    )

    template_vars = _assert_template_vars_result(result)
    assert template_vars["recent_conversation"] == ""


def test_chat_md_contains_location_context():
    """Location context lives in the static chat prompt."""
    chat_md = _read_chat_md()
    assert "## Location Context" in chat_md


def test_chat_md_contains_system_health():
    """System health guidance lives in the static chat prompt."""
    chat_md = _read_chat_md()
    assert "## System Health" in chat_md


def test_chat_md_contains_behavioral_defaults():
    """Behavioral defaults live in the static chat prompt."""
    chat_md = _read_chat_md()
    assert "## Behavioral Defaults" in chat_md


def test_chat_md_contains_static_import_guidance():
    """Import guidance lives in the static chat prompt."""
    chat_md = _read_chat_md()
    assert "## Import Awareness" in chat_md


def test_chat_md_contains_static_naming_guidance():
    """Naming guidance lives in the static chat prompt."""
    chat_md = _read_chat_md()
    assert "## Naming Awareness" in chat_md


def test_chat_context_awareness_error_graceful(monkeypatch):
    """Awareness failures still return the full template var shape."""
    monkeypatch.setattr("think.conversation.build_memory_context", lambda **kw: "")
    monkeypatch.setattr("think.routines.get_routine_state", lambda: [])
    monkeypatch.setattr(
        "think.routines.get_config", lambda: {"_meta": {"suggestions": {}}}
    )
    monkeypatch.setattr(
        "think.utils.get_config",
        lambda: {"agent": {"name": "aria", "name_status": "default"}},
    )
    monkeypatch.setattr("think.utils.get_journal", lambda: "/nonexistent")

    result = _load_chat_context_module().pre_process(
        {"user_instruction": "Base instruction."}
    )

    template_vars = _assert_template_vars_result(result)
    assert all(template_vars[key] == "" for key in TEMPLATE_VAR_KEYS)


def test_chat_context_does_not_return_sol_awareness(monkeypatch):
    """sol_awareness is no longer part of the chat pre-hook output."""
    monkeypatch.setattr("think.conversation.build_memory_context", lambda **kw: "")
    monkeypatch.setattr("think.routines.get_routine_state", lambda: [])
    monkeypatch.setattr(
        "think.routines.get_config", lambda: {"_meta": {"suggestions": {}}}
    )
    monkeypatch.setattr(
        "think.utils.get_config",
        lambda: {"agent": {"name": "aria", "name_status": "default"}},
    )

    result = _load_chat_context_module().pre_process(
        {"user_instruction": "Base instruction."}
    )

    assert "sol_awareness" not in result["template_vars"]


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

    result = _load_chat_context_module().pre_process(
        {"user_instruction": "Base instruction."}
    )

    template_vars = _assert_template_vars_result(result)
    assert "## Active Routines" in template_vars["active_routines"]
    assert "Morning Briefing" in template_vars["active_routines"]


def test_chat_context_routines_omitted_when_empty(monkeypatch):
    """Active routines section is omitted when no routines configured."""
    monkeypatch.setattr("think.conversation.build_memory_context", lambda **kw: "")
    monkeypatch.setattr("think.routines.get_routine_state", lambda: [])

    result = _load_chat_context_module().pre_process(
        {"user_instruction": "Base instruction."}
    )

    template_vars = _assert_template_vars_result(result)
    assert template_vars["active_routines"] == ""


def test_chat_context_routines_error_graceful(monkeypatch):
    """Routine state failures still return the full template var shape."""
    monkeypatch.setattr("think.conversation.build_memory_context", lambda **kw: "")
    monkeypatch.setattr(
        "think.routines.get_routine_state",
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        "think.routines.get_config", lambda: {"_meta": {"suggestions": {}}}
    )
    monkeypatch.setattr(
        "think.utils.get_config",
        lambda: {"agent": {"name": "aria", "name_status": "default"}},
    )

    result = _load_chat_context_module().pre_process(
        {"user_instruction": "Base instruction."}
    )

    template_vars = _assert_template_vars_result(result)
    assert template_vars["active_routines"] == ""
    assert set(template_vars) == TEMPLATE_VAR_KEYS
