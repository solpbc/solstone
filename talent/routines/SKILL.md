---
name: routines
description: >
  Set up recurring routines — daily briefings, weekly reviews, domain
  watches, commitment audits, meeting prep — or custom automations.
  Create from templates with sensible defaults or from custom
  instructions, adjust timing and scope, pause/resume, delete, run
  immediately to test, respond to routine suggestions, and read past
  output.
  TRIGGER: routine, routines, schedule, recurring, automate, daily brief,
  morning briefing, weekly review, domain watch, meeting prep,
  commitment audit, routine output, pause routine, run immediately,
  routine suggestions, sol call routines create, sol call routines list,
  sol call routines edit, sol call routines run, sol call routines output,
  sol call routines suggest-respond, sol call routines suggest-state.
---

# Routines CLI Skill

Manage recurring routines. Invoke via Bash: `sol call routines <command> [args...]`. Never expose cron syntax, UUIDs, or CLI internals to the owner.

## Template guidance

| Template | When to propose | Default timing | What to ask about |
|----------|----------------|----------------|-------------------|
| `morning-briefing` | Wants a daily digest, morning summary, or "what's on my plate today" | Every morning at 7am | Which facets to include |
| `weekly-review` | Wants a weekly recap, reflection, or "how did my week go" | Friday evening | Which facets to cover, preferred day/time |
| `domain-watch` | Wants to track a topic, project, or area over time | Monday morning | Which domains/topics to watch, which facets |
| `relationship-pulse` | Wants to stay on top of key relationships or "who haven't I talked to" | Monday morning | Which facets, which relationships matter most |
| `commitment-audit` | Wants to catch dropped commitments, overdue items, or stale follow-ups | Monday morning | Which facets to audit |
| `monthly-patterns` | Wants a monthly retrospective or trend analysis | First of the month, morning | Which facets, what patterns matter |
| `meeting-prep` | Wants briefings before meetings — "prep me before each meeting" | 30 minutes before each calendar event | Which facets to draw context from |

Meeting-prep is event-triggered, not clock-scheduled. Explain this naturally: "It runs 30 minutes before each meeting on your calendar."

## Command reference

| Intent | Command |
|--------|---------|
| Create from template | `sol call routines create --template {template} --timezone {tz}` (add `--facets`, `--cadence` if overridden) |
| Create custom | `sol call routines create --name "{name}" --instruction "{instruction}" --cadence "{cron}" --timezone {tz}` (add `--facets` if specified) |
| List all | `sol call routines list` |
| Show templates | `sol call routines templates` |
| Pause | `sol call routines edit {name} --enabled false` |
| Resume | `sol call routines edit {name} --enabled true` |
| Pause until date | `sol call routines edit {name} --enabled false --resume-date {YYYY-MM-DD}` |
| Change cadence | `sol call routines edit {name} --cadence "{cron}"` |
| Change facets | `sol call routines edit {name} --facets "{comma-separated}"` |
| Change instruction | `sol call routines edit {name} --instruction "{new instruction}"` |
| Delete | `sol call routines delete {name}` |
| Run immediately | `sol call routines run {name}` |
| Read output | `sol call routines output {name}` (add `--date YYYY-MM-DD` for a specific day) |
| Toggle suggestions | `sol call routines suggestions --enable` or `sol call routines suggestions --disable` |
| Record response to a suggestion | `sol call routines suggest-respond {template} --accepted` or `--declined` |
| Show suggestion state | `sol call routines suggest-state` |

Use the routine's name for identification, never UUIDs.

## Responding to suggestions

When the system surfaces a routine suggestion and the owner accepts or declines it, record their response so the suggestion engine doesn't re-surface the same template prematurely:

```bash
sol call routines suggest-respond morning-briefing --accepted
sol call routines suggest-respond weekly-review --declined
```

Exactly one of `--accepted` or `--declined` is required. `suggest-state` prints the full JSON state for all templates if you need to inspect what the engine already knows.

## Gotchas

- **Timezone must be an IANA name.** `--timezone America/Denver` works; `--timezone MDT` does not. The CLI rejects the latter with a terse error.
- **Suggestion responses are idempotent within a day.** Calling `suggest-respond` twice in the same day overwrites the previous response. Don't loop on it.
