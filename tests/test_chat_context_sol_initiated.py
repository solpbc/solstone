# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from solstone.convey.sol_initiated.copy import (
    CATEGORIES,
    KIND_SOL_CHAT_REQUEST,
    TRIGGER_LABEL_SOL_INITIATED,
)
from solstone.talent import chat_context


def _patch_routines(monkeypatch) -> None:
    monkeypatch.setattr("solstone.think.routines.get_routine_state", lambda: [])
    monkeypatch.setattr(
        "solstone.think.routines.get_config",
        lambda: {"_meta": {"suggestions_enabled": False, "suggestions": {}}},
    )
    monkeypatch.setattr("solstone.think.routines.save_config", lambda config: None)


def test_pre_hook_sets_template_vars_for_sol_request(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    _patch_routines(monkeypatch)
    since_ts = 1_775_000_000_000

    result = chat_context.pre_process(
        {
            "day": "20260331",
            "trigger": {
                "type": KIND_SOL_CHAT_REQUEST,
                "summary": "Notice this",
                "message": "Here is why.",
                "category": CATEGORIES[0],
                "since_ts": since_ts,
                "trigger_talent": "reflection",
                "request_id": "req",
            },
        }
    )

    template_vars = result["template_vars"]
    assert template_vars["trigger_kind"] == TRIGGER_LABEL_SOL_INITIATED
    assert template_vars["summary"] == "Notice this"
    assert template_vars["message"] == "Here is why."
    assert template_vars["category"] == CATEGORIES[0]
    assert template_vars["since_ts"] == since_ts
    assert template_vars["trigger_talent"] == "reflection"
    assert f"Type: {TRIGGER_LABEL_SOL_INITIATED}" in template_vars["trigger_context"]
    assert "Summary: Notice this" in template_vars["trigger_context"]
    assert "Since ts: 1775000000000" in template_vars["trigger_context"]


def test_existing_trigger_kind_labels_remain(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    _patch_routines(monkeypatch)

    owner = chat_context.pre_process(
        {
            "prompt": "hello",
            "trigger": {"type": "owner_message", "message": "hello"},
        }
    )
    finished = chat_context.pre_process(
        {
            "trigger": {
                "type": "talent_finished",
                "name": "exec",
                "summary": "done",
            }
        }
    )

    assert owner["template_vars"]["trigger_kind"] == "owner_message"
    assert finished["template_vars"]["trigger_kind"] == "talent_finished"
