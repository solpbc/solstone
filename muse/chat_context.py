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
from datetime import datetime
from pathlib import Path

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

ROUTINES_GUIDANCE = """## Recent Routine Outputs

{routine_summaries}

**How to reference routines in conversation:**
- When a routine is relevant to the owner's question, cite it by name: "Your Morning Briefing from earlier noted..."
- Surface routine findings as a natural "by the way" when contextually relevant — not forced
- Do not reference routines during deep, focused conversations unless the owner asks
- If the owner asks about a routine, offer to show the full output
""".strip()


def _extract_chat_summary(path: Path) -> str:
    """Extract a chat-oriented summary from a routine output markdown file."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""

    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                lines = lines[i + 1 :]
                break

    summary_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        summary_lines.append(stripped)
        if len(summary_lines) == 2:
            break

    summary = " ".join(summary_lines)
    if len(summary) > 150:
        return summary[:149] + "…"
    return summary


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
        from think.routines import get_config as get_routines_config
        from think.utils import get_journal

        routines_config = get_routines_config()
        journal = Path(get_journal())
        now = datetime.now()
        routine_lines = []
        for value in routines_config.values():
            if not isinstance(value, dict):
                continue
            if not value.get("enabled"):
                continue
            last_run = value.get("last_run")
            if not last_run:
                continue
            try:
                last_run_dt = datetime.fromisoformat(
                    last_run.replace("Z", "+00:00")
                ).replace(tzinfo=None)
            except (ValueError, AttributeError):
                continue
            if (now - last_run_dt).total_seconds() > 86400:
                continue
            routine_id = value.get("id", "")
            name = value.get("name", routine_id)
            output_dir = journal / "routines" / routine_id
            summary = ""
            if output_dir.exists():
                outputs = sorted(
                    output_dir.glob("*.md"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if outputs:
                    summary = _extract_chat_summary(outputs[0])
            if summary:
                routine_lines.append(f"- **{name}**: {summary}")
        if routine_lines:
            summaries_text = "\n".join(routine_lines)
            sections.append(ROUTINES_GUIDANCE.format(routine_summaries=summaries_text))
    except Exception:
        logger.debug("Routine context enrichment failed", exc_info=True)

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
