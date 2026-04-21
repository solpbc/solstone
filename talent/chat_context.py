# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Pre-hook: provide template vars for chat prompt context."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from convey.chat_stream import read_chat_tail, reduce_chat_state
from think.utils import get_config, get_journal

logger = logging.getLogger(__name__)


TEMPLATE_TRIGGERS = {
    "morning-briefing": {
        "patterns": [
            "calendar",
            "schedule",
            "agenda",
            "what do i have today",
            "what's on my calendar",
            "whats on my calendar",
            "what's happening today",
            "whats happening today",
        ],
        "threshold": 3,
        "description": "asked about your calendar or schedule",
    },
    "weekly-review": {
        "patterns": [
            "this week",
            "last week",
            "past few days",
            "how did my week",
            "what happened this week",
            "how was my week",
        ],
        "threshold": 3,
        "description": "asked for week-scale synthesis",
    },
    "domain-watch": {
        "patterns": [
            "track",
            "watch",
            "keep an eye on",
            "follow",
            "across days",
            "over time",
            "lately",
            "trend",
            "trends",
        ],
        "threshold": 3,
        "description": "revisited the same topic across multiple days",
    },
    "relationship-pulse": {
        "patterns": [
            "who haven't i",
            "who havent i",
            "relationship",
            "when did i last talk to",
            "catch up with",
        ],
        "threshold": 2,
        "description": "asked about relationships",
    },
    "commitment-audit": {
        "patterns": [
            "follow up",
            "follow-up",
            "commitment",
            "dropped",
            "overdue",
            "what did i promise",
            "pending",
        ],
        "threshold": 2,
        "description": "asked about commitments or follow-ups",
    },
    "meeting-prep": {
        "patterns": [
            "brief me",
            "who am i meeting",
            "meeting with",
            "prepare me for",
            "prep for my meeting",
            "prep me for",
            "meeting prep",
        ],
        "threshold": 3,
        "description": "asked for meeting briefings",
    },
}


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


def _get_eligible_suggestion(
    routines_config: dict, journal_config: dict
) -> dict | None:
    """Evaluate 5-gate chain and return the best eligible suggestion, or None."""
    meta = routines_config.get("_meta", {})

    if not meta.get("suggestions_enabled", True):
        return None

    name_status = journal_config.get("agent", {}).get("name_status", "default")
    if name_status == "default":
        return None

    last_date_str = meta.get("last_suggestion_date")
    if last_date_str:
        try:
            last_date = date.fromisoformat(last_date_str)
            if (date.today() - last_date) < timedelta(days=7):
                return None
        except ValueError:
            pass

    suggestions = meta.get("suggestions", {})
    active_templates = {
        value.get("template")
        for value in routines_config.values()
        if isinstance(value, dict) and value.get("id")
    }

    candidates = []

    for template_name, entry in suggestions.items():
        if template_name in active_templates:
            continue
        if entry.get("response") == "declined":
            continue

        info = TEMPLATE_TRIGGERS.get(template_name)
        if info and entry.get("trigger_count", 0) >= info["threshold"]:
            candidates.append(
                {
                    "template_name": template_name,
                    "trigger_count": entry["trigger_count"],
                    "first_trigger": entry.get("first_trigger"),
                    "pattern_description": info["description"],
                }
            )

    if "monthly-patterns" not in active_templates:
        mp_entry = suggestions.get("monthly-patterns", {})
        if mp_entry.get("response") != "declined":
            try:
                from think.utils import day_dirs

                days = day_dirs()
                if days:
                    earliest = min(days.keys())
                    earliest_date = date(
                        int(earliest[:4]),
                        int(earliest[4:6]),
                        int(earliest[6:8]),
                    )
                    if (date.today() - earliest_date) >= timedelta(days=30):
                        candidates.append(
                            {
                                "template_name": "monthly-patterns",
                                "trigger_count": 0,
                                "first_trigger": (
                                    f"{earliest[:4]}-{earliest[4:6]}-{earliest[6:8]}"
                                ),
                                "pattern_description": (
                                    "your journal has 30+ days of history"
                                ),
                            }
                        )
            except Exception:
                pass

    if not candidates:
        return None

    candidates.sort(key=lambda candidate: candidate["trigger_count"], reverse=True)
    return candidates[0]


def pre_process(context: dict) -> dict:
    """Build chat-context template vars for the chat talent prompt."""
    from think.routines import get_config as get_routines_config
    from think.routines import get_routine_state
    from think.routines import save_config as save_routines_config

    facet = context.get("facet")
    trigger_kind, trigger_payload = _normalize_trigger(context)
    day = _resolve_day(context, trigger_payload)
    template_vars = {
        "digest_contents": "",
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
                        f"[talent {trigger_payload['name']} finished: "
                        f"{trigger_payload['summary']}]"
                    ),
                }
            )
        elif trigger_kind == "talent_errored":
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"[talent {trigger_payload['name']} errored: "
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
        routines = get_routine_state()
        if routines:
            lines = ["## Active Routines\n"]
            for routine in routines:
                status = "on" if routine["enabled"] else "paused"
                if routine.get("paused_until"):
                    status = f"paused until {routine['paused_until']}"
                line = f"- **{routine['name']}** ({routine['cadence']}) — {status}"
                if routine.get("output_summary"):
                    line += f" | recent: {routine['output_summary']}"
                lines.append(line)
            template_vars["active_routines"] = "\n".join(lines)
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
        routines_config = get_routines_config()
        suggestion = _get_eligible_suggestion(routines_config, get_config())
        if suggestion:
            if suggestion["trigger_count"] == 0:
                pattern_line = (
                    f"Pattern: {suggestion['pattern_description']} "
                    f"since {suggestion['first_trigger']}."
                )
            else:
                pattern_line = (
                    f"Pattern: You've {suggestion['pattern_description']} "
                    f"{suggestion['trigger_count']} times since "
                    f"{suggestion['first_trigger']}."
                )
            template_vars["routine_suggestion"] = (
                "## Routine Suggestion Eligible\n\n"
                f"Template: {suggestion['template_name']}\n"
                f"{pattern_line}\n"
                f"Trigger count: {suggestion['trigger_count']}\n"
                f"First seen: {suggestion['first_trigger']}\n\n"
                "### Etiquette\n"
                "- Mention this ONCE, naturally, at the end of your response\n"
                '- Frame as observation: "I\'ve noticed you often... — would a routine help?"\n'
                "- If $name declines or ignores, do not bring it up again this conversation\n"
                "- After suggesting, run: `sol call routines suggest-respond "
                f"{suggestion['template_name']} --accepted` or `--declined`"
            )
    except Exception:
        logger.debug("Routine suggestion eligibility check failed", exc_info=True)

    return result


def _load_digest_contents() -> str:
    digest_path = Path(get_journal()) / "identity" / "digest.md"
    if not digest_path.exists():
        return ""
    return digest_path.read_text(encoding="utf-8").strip()


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
        if payload.get("summary"):
            lines.append(f"- Summary: {payload['summary']}")
    elif trigger_kind == "talent_errored":
        if payload.get("name"):
            lines.append(f"- Talent: {payload['name']}")
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
