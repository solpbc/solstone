# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import importlib.util
import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from convey.chat_stream import append_chat_event
from think.identity import ensure_identity_directory

TEMPLATE_VAR_KEYS = {
    "digest_contents",
    "identity_self",
    "identity_agency",
    "active_talents",
    "trigger_context",
    "location",
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


def _write_journal_config(journal: Path, data: dict) -> None:
    config_dir = journal / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "journal.json").write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )


def _ts(hour: int, minute: int, second: int = 0) -> int:
    return int(datetime(2026, 4, 20, hour, minute, second).timestamp() * 1000)


def test_chat_context_injects_digest_tail_trigger_location_and_routine_state(
    monkeypatch, tmp_path
):
    journal = tmp_path / "journal"
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(journal))
    (journal / "identity").mkdir(parents=True, exist_ok=True)
    (journal / "identity" / "digest.md").write_text(
        "Digest notes for today.",
        encoding="utf-8",
    )
    _write_journal_config(
        journal,
        {
            "identity": {"preferred": "Alice"},
            "agent": {"name": "Sol-agent", "name_status": "custom"},
        },
    )

    owner_ts = _ts(9, 0)
    append_chat_event(
        "owner_message",
        ts=owner_ts,
        text="Please brief me for my meeting",
        app="home",
        path="/app/home",
        facet="work",
    )
    append_chat_event(
        "sol_message",
        ts=_ts(9, 1),
        use_id="use-chat-1",
        text="I can help with that.",
        notes="Responded directly.",
        requested_target=None,
        requested_task=None,
    )
    append_chat_event(
        "talent_spawned",
        ts=_ts(9, 2),
        use_id="use-exec-1",
        name="exec",
        task="Prepare the meeting brief",
        started_at=_ts(9, 2),
    )

    routines_config = {
        "_meta": {
            "suggestions_enabled": True,
            "suggestions": {
                "meeting-prep": {
                    "trigger_count": 3,
                    "first_trigger": "2026-04-01",
                    "last_trigger": "2026-04-19",
                    "trigger_data": {},
                    "response": None,
                    "suggested": False,
                }
            },
        }
    }
    monkeypatch.setattr(
        "think.routines.get_routine_state",
        lambda: [
            {
                "name": "Morning Briefing",
                "cadence": "0 9 * * *",
                "last_run": None,
                "enabled": True,
                "paused_until": None,
                "output_summary": "Shared the top priorities.",
            }
        ],
    )
    monkeypatch.setattr("think.routines.get_config", lambda: deepcopy(routines_config))
    monkeypatch.setattr("think.routines.save_config", lambda config: None)

    result = _load_chat_context_module().pre_process(
        {
            "prompt": "Please brief me for my meeting",
            "facet": "work",
            "day": "20260420",
            "trigger_kind": "owner_message",
            "trigger_payload": {
                "text": "Please brief me for my meeting",
                "app": "home",
                "path": "/app/home",
                "facet": "work",
                "ts": owner_ts,
            },
        }
    )

    template_vars = _assert_template_vars_result(result)
    assert template_vars["digest_contents"] == "Digest notes for today."
    assert result["messages"] == [
        {"role": "user", "content": "Please brief me for my meeting"},
        {"role": "assistant", "content": "I can help with that."},
    ]
    assert all("exec spawned" not in msg["content"] for msg in result["messages"])
    assert "## Active Talents" in template_vars["active_talents"]
    assert "Prepare the meeting brief" in template_vars["active_talents"]
    assert "## Trigger Context" in template_vars["trigger_context"]
    assert "Type: owner_message" in template_vars["trigger_context"]
    assert "Please brief me for my meeting" in template_vars["trigger_context"]
    assert "## Location" in template_vars["location"]
    assert "/app/home" in template_vars["location"]
    assert "work" in template_vars["location"]
    assert "## Active Routines" in template_vars["active_routines"]
    assert "Morning Briefing" in template_vars["active_routines"]
    assert "Routine Suggestion Eligible" in template_vars["routine_suggestion"]
    assert "meeting-prep" in template_vars["routine_suggestion"]


def test_chat_context_routine_suggestion_only_counts_owner_messages(
    monkeypatch, tmp_path
):
    journal = tmp_path / "journal"
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(journal))

    routines_config = {"_meta": {"suggestions_enabled": True, "suggestions": {}}}
    save_calls: list[dict] = []
    monkeypatch.setattr("think.routines.get_routine_state", lambda: [])
    monkeypatch.setattr("think.routines.get_config", lambda: routines_config)
    monkeypatch.setattr(
        "think.routines.save_config",
        lambda config: save_calls.append(deepcopy(config)),
    )

    module = _load_chat_context_module()

    module.pre_process(
        {
            "prompt": "What is on my calendar today?",
            "trigger_kind": "talent_finished",
            "trigger_payload": {
                "name": "exec",
                "summary": "Collected the latest meeting prep notes.",
            },
        }
    )

    assert routines_config["_meta"]["suggestions"] == {}
    assert save_calls == []

    module.pre_process(
        {
            "prompt": "What is on my calendar today?",
            "trigger_kind": "owner_message",
            "trigger_payload": {
                "text": "What is on my calendar today?",
                "ts": _ts(10, 0),
            },
        }
    )

    suggestion = routines_config["_meta"]["suggestions"]["morning-briefing"]
    assert suggestion["trigger_count"] == 1
    assert len(save_calls) == 1


def test_chat_context_talent_finished_appends_internal_followup_message(
    monkeypatch, tmp_path
):
    journal = tmp_path / "journal"
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(journal))

    append_chat_event(
        "owner_message",
        ts=_ts(10, 0),
        text="What happened?",
        app="home",
        path="/app/home",
        facet="work",
    )
    append_chat_event(
        "sol_message",
        ts=_ts(10, 1),
        use_id="use-chat-2",
        text="Looking into it.",
        notes="Acknowledged request.",
        requested_target=None,
        requested_task=None,
    )
    append_chat_event(
        "talent_finished",
        ts=_ts(10, 2),
        use_id="use-exec-2",
        name="exec",
        summary="Found the latest notes.",
    )

    monkeypatch.setattr("think.routines.get_routine_state", lambda: [])
    monkeypatch.setattr(
        "think.routines.get_config",
        lambda: {"_meta": {"suggestions_enabled": False, "suggestions": {}}},
    )
    monkeypatch.setattr("think.routines.save_config", lambda config: None)

    result = _load_chat_context_module().pre_process(
        {
            "day": "20260420",
            "trigger_kind": "talent_finished",
            "trigger_payload": {
                "name": "exec",
                "summary": "Found the latest notes.",
            },
        }
    )

    _assert_template_vars_result(result)
    assert result["messages"] == [
        {"role": "user", "content": "What happened?"},
        {"role": "assistant", "content": "Looking into it."},
        {
            "role": "user",
            "content": (
                "[internal follow-up: talent exec finished. Use this result "
                "to answer the owner's pending request with a short summary. "
                "Result: Found the latest notes.]"
            ),
        },
    ]


def test_chat_context_includes_identity_grounding(monkeypatch, tmp_path):
    journal = tmp_path / "journal"
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(journal))
    _write_journal_config(journal, {})
    ensure_identity_directory()

    digest_seed = (journal / "identity" / "digest.md").read_text(encoding="utf-8")
    assert digest_seed == "not yet generated\n"

    monkeypatch.setattr("think.routines.get_routine_state", lambda: [])
    monkeypatch.setattr(
        "think.routines.get_config",
        lambda: {"_meta": {"suggestions_enabled": False, "suggestions": {}}},
    )
    monkeypatch.setattr("think.routines.save_config", lambda config: None)

    result = _load_chat_context_module().pre_process({"day": "20260420"})

    template_vars = _assert_template_vars_result(result)
    assert template_vars["identity_self"]
    assert template_vars["identity_agency"]
    assert template_vars["identity_self"] != digest_seed.strip()
    assert template_vars["identity_agency"] != digest_seed.strip()


def test_chat_context_preserves_save_routines_config_side_effect(monkeypatch, tmp_path):
    journal = tmp_path / "journal"
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(journal))

    routines_config = {"_meta": {"suggestions_enabled": True, "suggestions": {}}}
    save_calls: list[dict] = []
    monkeypatch.setattr("think.routines.get_routine_state", lambda: [])
    monkeypatch.setattr("think.routines.get_config", lambda: routines_config)
    monkeypatch.setattr(
        "think.routines.save_config",
        lambda config: save_calls.append(deepcopy(config)),
    )

    _load_chat_context_module().pre_process(
        {
            "prompt": "What is on my calendar today?",
            "trigger_kind": "owner_message",
            "trigger_payload": {
                "text": "What is on my calendar today?",
                "ts": _ts(11, 0),
            },
        }
    )

    assert len(save_calls) == 1
    saved = save_calls[0]
    assert saved["_meta"]["suggestions"]["morning-briefing"]["trigger_count"] == 1
    assert saved["_meta"]["suggestions"]["morning-briefing"]["first_trigger"]


def test_chat_context_routines_omitted_when_empty(monkeypatch, tmp_path):
    journal = tmp_path / "journal"
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(journal))
    monkeypatch.setattr("think.routines.get_routine_state", lambda: [])
    monkeypatch.setattr(
        "think.routines.get_config",
        lambda: {"_meta": {"suggestions_enabled": False, "suggestions": {}}},
    )
    monkeypatch.setattr("think.routines.save_config", lambda config: None)

    result = _load_chat_context_module().pre_process({"day": "20260420"})

    template_vars = _assert_template_vars_result(result)
    assert template_vars["active_routines"] == ""
    assert template_vars["active_talents"] == ""
    assert "messages" not in result


def test_chat_context_enrichment_errors_are_graceful(monkeypatch, tmp_path):
    journal = tmp_path / "journal"
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(journal))

    module = _load_chat_context_module()

    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(module, "_load_digest_contents", _boom)
    monkeypatch.setattr(module, "read_chat_tail", _boom)
    monkeypatch.setattr(module, "reduce_chat_state", _boom)
    monkeypatch.setattr("think.routines.get_routine_state", _boom)
    monkeypatch.setattr("think.routines.get_config", _boom)
    monkeypatch.setattr("think.routines.save_config", lambda config: None)

    result = module.pre_process(
        {
            "prompt": "What is on my calendar today?",
            "trigger_kind": "owner_message",
            "trigger_payload": {
                "text": "What is on my calendar today?",
                "path": "/app/home",
                "ts": _ts(12, 0),
            },
        }
    )

    template_vars = _assert_template_vars_result(result)
    assert template_vars["digest_contents"] == ""
    assert template_vars["active_talents"] == ""
    assert template_vars["active_routines"] == ""
    assert template_vars["routine_suggestion"] == ""
    assert "Type: owner_message" in template_vars["trigger_context"]
    assert "/app/home" in template_vars["location"]
    assert "messages" not in result


def test_chat_context_drops_legacy_memory_imports(monkeypatch):
    monkeypatch.setattr("think.routines.get_routine_state", lambda: [])
    monkeypatch.setattr(
        "think.routines.get_config",
        lambda: {"_meta": {"suggestions_enabled": False, "suggestions": {}}},
    )
    monkeypatch.setattr("think.routines.save_config", lambda config: None)

    legacy_module = "think" + ".con" + "versation"
    legacy_memory = "conversation_" + "memory"
    source = (
        Path(__file__).resolve().parents[1] / "talent" / "chat_context.py"
    ).read_text(encoding="utf-8")
    assert legacy_module not in source
    assert legacy_memory not in source

    sys.modules.pop(legacy_module, None)
    _load_chat_context_module()

    assert legacy_module not in sys.modules


def test_chat_prompt_includes_meta_question_inline_rule():
    prompt_path = Path(__file__).resolve().parents[1] / "talent" / "chat.md"
    prompt_text = prompt_path.read_text(encoding="utf-8")

    assert (
        "Questions about your role, capabilities, limits, current context, naming, "
        "or system status stay inline."
    ) in prompt_text
