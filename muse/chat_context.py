# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Pre-hook: inject chat-bar context into sol's user instruction.

Replaces conversation_memory as the unified muse's pre-hook.
Appends conversation memory, location/health context instructions,
awareness-conditional guidance, and behavioral defaults to the
identity-first prompt.

Loaded via hook config: {"hook": {"pre": "chat_context"}}
"""

import logging
from datetime import date, timedelta

logger = logging.getLogger(__name__)

# --- Awareness-conditional instruction blocks ---

ONBOARDING_OBSERVATION_TEXT = """## Onboarding Observation Context

The user is in Path A onboarding observation. If they ask "what have you noticed?" or similar, read recent observations from the awareness log and summarize progress encouragingly. You are quietly watching how they work, learning their patterns.
""".strip()

ONBOARDING_READY_TEXT = """## Onboarding Observation Complete

Path A observation is complete — recommendations are ready. Proactively suggest reviewing: "I've finished observing and have suggestions for organizing your journal. Want to take a look?" If they agree, read observations, synthesize recommendations, and walk through setup in-place.
""".strip()

IMPORT_AWARENESS_TEXT = """## Import Awareness

Onboarding is complete but no content has been imported yet. If the user's message touches on their journal or what you can do, weave a single soft mention of importing into your response. Available sources: Calendar, ChatGPT, Claude, Gemini, Notes, Kindle. Do not repeat if already nudged.
""".strip()

NAMING_AWARENESS_TEXT = """## Naming Awareness

The journal is still using its default name. When the moment feels right — after enough shared history — you may offer to suggest a name, or let the user choose one. Check naming readiness before offering. Only do this once per session.
""".strip()


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


def pre_process(context: dict) -> dict | None:
    """Append chat-context instructions to the unified muse prompt."""
    from think.awareness import get_imports, get_onboarding
    from think.conversation import build_memory_context
    from think.utils import get_config

    user_instruction = context.get("user_instruction", "")
    facet = context.get("facet")

    sections: list[str] = []

    try:
        memory_context = build_memory_context(facet=facet, recent_limit=10)
        if memory_context:
            sections.append(f"## Recent Conversation\n\n{memory_context}")
    except Exception:
        logger.debug("Conversation memory enrichment failed", exc_info=True)

    sections.append(
        """## Location Context

You receive context about the user's current app, URL path, and active facet. Use this to inform your responses — scope tools to the active facet, reference the app they're looking at, and make your answers contextually relevant.
""".strip()
    )

    sections.append(
        """## System Health

When the context includes a `System health:` line, there is an active attention item:

- **"what needs my attention?"** — Report the system health item. Be concise.
- **Agent errors:** Explain which agents failed. Suggest checking logs.
- **Capture offline:** Suggest checking that the observer service is running.
- **Import complete:** Describe what was imported, offer to explore or import more.

When no `System health:` line is present, everything is fine.
""".strip()
    )

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
            sections.append("\n".join(lines))
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
            sections.append(hint)
    except Exception:
        logger.debug("Routine suggestion eligibility check failed", exc_info=True)

    try:
        onboarding = get_onboarding()
        onboarding_status = onboarding.get("status", "")

        if onboarding_status == "observing":
            sections.append(ONBOARDING_OBSERVATION_TEXT)
        elif onboarding_status == "ready":
            sections.append(ONBOARDING_READY_TEXT)
        elif onboarding_status in ("complete", "skipped"):
            imports = get_imports()
            if not imports.get("has_imported"):
                sections.append(IMPORT_AWARENESS_TEXT)

            config = get_config()
            agent_name = config.get("agent", {}).get("name", "sol")
            if agent_name == "sol":
                sections.append(NAMING_AWARENESS_TEXT)
    except Exception:
        logger.debug("Awareness context enrichment failed", exc_info=True)

    sections.append(
        """## Behavioral Defaults

- SOL_DAY and SOL_FACET environment variables are already set — tools use them as defaults when --day/--facet are omitted. You can often omit these flags.
- If searching reveals sensitive or personal content, handle with care and focus on what was specifically asked.
- When a tool call returns an error, note briefly what was unavailable and move on. Do not retry or debug. Work with whatever data you successfully retrieved.
""".strip()
    )

    if sections:
        modified = user_instruction + "\n\n" + "\n\n".join(sections)
        return {"user_instruction": modified}
    return None
