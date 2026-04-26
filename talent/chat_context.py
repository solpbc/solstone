# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Pre-hook: provide template vars for chat prompt context."""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

from convey.chat_stream import read_chat_tail, reduce_chat_state
from talent._routine_context import (
    _TEMPLATE_TRIGGERS as TEMPLATE_TRIGGERS,
)
from talent._routine_context import render_active_routines, render_routine_suggestion
from think.utils import get_journal

logger = logging.getLogger(__name__)


def _count_triggers(msg: str, facet: str | None, config: dict) -> bool:
    """Count trigger signals in the user's message. Returns True if config was mutated."""
    lower = msg.lower()
    today = date.today().isoformat()
    meta = config.setdefault("_meta", {})
    suggestions = meta.setdefault("suggestions", {})
    changed = False

    for template, info in TEMPLATE_TRIGGERS.items():
        if not any(p in lower for p in info["patterns"]):
            continue

        if template == "domain-watch":
            if not facet:
                continue
            entry = suggestions.setdefault(
                template,
                {
                    "trigger_count": 0,
                    "first_trigger": None,
                    "last_trigger": None,
                    "trigger_data": {},
                    "response": None,
                    "suggested": False,
                },
            )
            topics = entry.setdefault("trigger_data", {}).setdefault("topics", {})
            dates = topics.setdefault(facet, [])
            if today not in dates:
                dates.append(today)
                entry["trigger_count"] = len(dates)
                entry["first_trigger"] = entry["first_trigger"] or min(dates)
                entry["last_trigger"] = max(dates)
                changed = True
        else:
            entry = suggestions.setdefault(
                template,
                {
                    "trigger_count": 0,
                    "first_trigger": None,
                    "last_trigger": None,
                    "trigger_data": {},
                    "response": None,
                    "suggested": False,
                },
            )
            entry["trigger_count"] = entry.get("trigger_count", 0) + 1
            entry["first_trigger"] = entry.get("first_trigger") or today
            entry["last_trigger"] = today
            changed = True

    return changed


def pre_process(context: dict) -> dict:
    """Build chat-context template vars for the chat talent prompt."""
    from think.routines import get_config as get_routines_config
    from think.routines import save_config as save_routines_config

    facet = context.get("facet")
    trigger_kind, trigger_payload = _normalize_trigger(context)
    day = _resolve_day(context, trigger_payload)
    template_vars = {
        "digest_contents": "",
        "identity_self": "",
        "identity_agency": "",
        "active_talents": "",
        "trigger_context": "",
        "location": "",
        "active_routines": "",
        "routine_suggestion": "",
    }
    result = {"template_vars": template_vars}

    try:
        template_vars["digest_contents"] = _load_digest_contents()
    except Exception:
        logger.debug("Digest enrichment failed", exc_info=True)

    try:
        template_vars["identity_self"] = _load_identity_contents("self.md")
        template_vars["identity_agency"] = _load_identity_contents("agency.md")
    except Exception:
        logger.debug("Identity enrichment failed", exc_info=True)

    try:
        tail = read_chat_tail(day, limit=20)
        messages: list[dict[str, str]] = []
        for event in tail:
            if event["kind"] == "owner_message":
                messages.append({"role": "user", "content": event["text"]})
            elif event["kind"] == "sol_message":
                messages.append({"role": "assistant", "content": event["text"]})

        if trigger_kind == "talent_finished":
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "[internal follow-up: talent "
                        f"{trigger_payload['name']} finished. This is a "
                        "report-back turn, not a dispatch turn. Do not "
                        "request another talent for this task. Use the "
                        "result below to answer the owner's pending request "
                        f"with a short summary. Result: {trigger_payload['summary']}]"
                    ),
                }
            )
        elif trigger_kind == "talent_errored":
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "[internal follow-up: talent "
                        f"{trigger_payload['name']} errored. This is a "
                        "stop-and-report turn, not a dispatch turn. Do "
                        "not retry this task or request another talent for "
                        "it. Stop here and report the failure to the owner "
                        "directly using the reason below. Reason: "
                        f"{trigger_payload['reason']}]"
                    ),
                }
            )

        if messages:
            result["messages"] = messages
    except Exception:
        logger.debug("Chat tail enrichment failed", exc_info=True)

    try:
        state = reduce_chat_state(day)
        template_vars["active_talents"] = _render_active_talents(
            state.get("active_talents", [])
        )
    except Exception:
        logger.debug("Active talent enrichment failed", exc_info=True)

    template_vars["trigger_context"] = _render_trigger_context(
        trigger_kind, trigger_payload, context
    )
    template_vars["location"] = _render_location(trigger_payload, context)

    try:
        template_vars["active_routines"] = render_active_routines()
    except Exception:
        logger.debug("Routine state enrichment failed", exc_info=True)

    try:
        prompt = context.get("prompt", "")
        if trigger_kind == "owner_message" and prompt:
            routines_config = get_routines_config()
            if _count_triggers(prompt, facet, routines_config):
                save_routines_config(routines_config)
    except Exception:
        logger.debug("Routine trigger counting failed", exc_info=True)

    try:
        template_vars["routine_suggestion"] = render_routine_suggestion()
    except Exception:
        logger.debug("Routine suggestion eligibility check failed", exc_info=True)

    return result


def _load_digest_contents() -> str:
    digest_path = Path(get_journal()) / "identity" / "digest.md"
    if not digest_path.exists():
        return ""
    return digest_path.read_text(encoding="utf-8").strip()


def _load_identity_contents(file_name: str) -> str:
    identity_path = Path(get_journal()) / "identity" / file_name
    if not identity_path.exists():
        return ""
    return identity_path.read_text(encoding="utf-8").strip()


def _normalize_trigger(context: dict) -> tuple[str | None, dict[str, Any]]:
    trigger_info = context.get("trigger")
    kind = None
    payload: dict[str, Any] = {}

    if isinstance(trigger_info, dict):
        kind = trigger_info.get("kind")
        raw_payload = trigger_info.get("payload")
        if isinstance(raw_payload, dict):
            payload.update(raw_payload)

    if not kind:
        kind = context.get("trigger_kind")

    raw_payload = context.get("trigger_payload")
    if isinstance(raw_payload, dict):
        payload.update(raw_payload)

    location = context.get("location")
    if isinstance(location, dict):
        if "app" not in payload and location.get("app"):
            payload["app"] = location["app"]
        if "path" not in payload and location.get("path"):
            payload["path"] = location["path"]
        if "facet" not in payload and location.get("facet"):
            payload["facet"] = location["facet"]

    if "facet" not in payload and context.get("facet"):
        payload["facet"] = context["facet"]
    if "app" not in payload and context.get("app"):
        payload["app"] = context["app"]
    if "path" not in payload and context.get("ui_path"):
        payload["path"] = context["ui_path"]
    if "ts" not in payload and isinstance(context.get("trigger_ts"), int):
        payload["ts"] = context["trigger_ts"]

    if not kind and context.get("prompt"):
        kind = "owner_message"
    if kind == "owner_message" and "text" not in payload and context.get("prompt"):
        payload["text"] = context["prompt"]

    return kind, payload


def _resolve_day(context: dict, trigger_payload: dict[str, Any]) -> str:
    day = context.get("day")
    if isinstance(day, str) and len(day) == 8 and day.isdigit():
        return day

    ts_value = trigger_payload.get("ts")
    if isinstance(ts_value, int):
        return datetime.fromtimestamp(ts_value / 1000).strftime("%Y%m%d")

    return date.today().strftime("%Y%m%d")


def _render_active_talents(active_talents: list[dict[str, Any]]) -> str:
    if not active_talents:
        return ""

    lines = ["## Active Talents\n"]
    for talent in active_talents:
        started_at = _format_started_at(talent.get("started_at"))
        line = f"- **{talent.get('name', 'exec')}** — {talent.get('task', '')}"
        if started_at:
            line += f" (started {started_at})"
        lines.append(line)
    return "\n".join(lines)


def _format_started_at(value: Any) -> str:
    if not isinstance(value, int):
        return ""
    return datetime.fromtimestamp(value / 1000).strftime("%Y-%m-%d %H:%M")


def _render_trigger_context(
    trigger_kind: str | None,
    payload: dict[str, Any],
    context: dict[str, Any],
) -> str:
    if not trigger_kind:
        return ""

    lines = ["## Trigger Context\n", f"- Type: {trigger_kind}"]
    if trigger_kind == "owner_message":
        text = str(payload.get("text") or context.get("prompt") or "").strip()
        if text:
            lines.append(f"- Message: {text}")
    elif trigger_kind == "talent_finished":
        if payload.get("name"):
            lines.append(f"- Talent: {payload['name']}")
        lines.append("- Mode: report_back_only")
        lines.append(
            "- Instruction: Answer the owner directly; do not dispatch or "
            "redispatch a talent for this trigger."
        )
        if payload.get("summary"):
            lines.append(f"- Summary: {payload['summary']}")
    elif trigger_kind == "talent_errored":
        if payload.get("name"):
            lines.append(f"- Talent: {payload['name']}")
        lines.append("- Mode: report_back_only")
        lines.append(
            "- Instruction: Answer the owner directly; report the failure to "
            "the owner and stop; do not retry, dispatch, or redispatch a "
            "talent for this trigger."
        )
        if payload.get("reason"):
            lines.append(f"- Reason: {payload['reason']}")
    elif trigger_kind == "synthetic-max-active":
        if payload.get("reason"):
            lines.append(f"- Reason: {payload['reason']}")
    else:
        if payload:
            for key, value in payload.items():
                lines.append(f"- {key}: {value}")

    return "\n".join(lines)


def _render_location(payload: dict[str, Any], context: dict[str, Any]) -> str:
    app = payload.get("app") or context.get("app")
    path = payload.get("path") or context.get("ui_path")
    facet = payload.get("facet") or context.get("facet")

    if not any((app, path, facet)):
        return ""

    lines = ["## Location\n"]
    if app:
        lines.append(f"- App: {app}")
    if path:
        lines.append(f"- Path: {path}")
    if facet:
        lines.append(f"- Facet: {facet}")
    return "\n".join(lines)
