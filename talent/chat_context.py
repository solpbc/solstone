# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Pre-hook: provide template vars for chat prompt context.

Replaces conversation_memory as the unified talent's pre-hook.
Builds dynamic chat context as template vars for the identity-first
prompt while preserving routine trigger side effects and awareness
guidance.

Loaded via hook config: {"hook": {"pre": "chat_context"}}
"""

import logging
from datetime import date, timedelta

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
    """Build chat-context template vars for the unified talent prompt."""
    from think.conversation import build_memory_context
    from think.utils import get_config

    facet = context.get("facet")
    template_vars = {
        "recent_conversation": "",
        "active_routines": "",
        "routine_suggestion": "",
        "sol_awareness": "",
    }

    try:
        memory_context = build_memory_context(facet=facet, recent_limit=10)
        if memory_context:
            template_vars["recent_conversation"] = (
                f"## Recent Conversation\n\n{memory_context}"
            )
    except Exception:
        logger.debug("Conversation memory enrichment failed", exc_info=True)

    try:
        from think.routines import get_routine_state

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
        from think.routines import get_config as get_routines_config
        from think.routines import save_config as save_routines_config

        prompt = context.get("prompt", "")
        if prompt:
            routines_config = get_routines_config()
            if _count_triggers(prompt, facet, routines_config):
                save_routines_config(routines_config)
    except Exception:
        logger.debug("Routine trigger counting failed", exc_info=True)

    try:
        from think.routines import get_config as get_routines_config

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
            hint = (
                "## Routine Suggestion Eligible\n\n"
                f"Template: {suggestion['template_name']}\n"
                f"{pattern_line}\n"
                f"Trigger count: {suggestion['trigger_count']}\n"
                f"First seen: {suggestion['first_trigger']}\n\n"
                "### Etiquette\n"
                "- Mention this ONCE, naturally, at the end of your response\n"
                "- Frame as observation: \"I've noticed you often... — would a "
                'routine help?"\n'
                "- If $name declines or ignores, do not bring it up again this "
                "conversation\n"
                "- After suggesting, run: `sol call routines suggest-respond "
                f"{suggestion['template_name']} --accepted` or `--declined`"
            )
            template_vars["routine_suggestion"] = hint
    except Exception:
        logger.debug("Routine suggestion eligibility check failed", exc_info=True)

    try:
        from pathlib import Path

        from think.utils import get_journal

        awareness_path = Path(get_journal()) / "sol" / "awareness.md"
        if awareness_path.exists():
            content = awareness_path.read_text(encoding="utf-8")
            # Cold-start gating: don't inject placeholder content
            if content.strip() != "not yet updated":
                template_vars["sol_awareness"] = f"## Awareness\n\n{content}"
    except Exception:
        logger.debug("Awareness context loading failed", exc_info=True)

    return {"template_vars": template_vars}
