---
name: routines
description: >
  Create, manage, and inspect scheduled routines. List available templates,
  create routines from templates or custom instructions, edit timing and scope,
  pause/resume, delete, run immediately, and read output history.
  TRIGGER: routine, routines, schedule, recurring, automate, morning briefing setup,
  weekly review setup, domain watch, meeting prep, commitment audit.
---

# Routines CLI Skill

Use these commands to manage routines. Never expose cron syntax, UUIDs, or CLI internals to the owner.

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

Use the routine's name for identification, never UUIDs.
