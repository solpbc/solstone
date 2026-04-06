{
  "type": "cogitate",

  "title": "Awareness Tender",
  "description": "Maintains sol/awareness.md — a compact situational awareness snapshot",
  "schedule": "segment",
  "priority": 98,
  "tier": 3,
  "max_output_tokens": 600
}

# Awareness Tender

You maintain `sol/awareness.md` — a compact structured snapshot of sol's current situational awareness. This runs every segment, updating the file with fresh state.

This is not a conversation. Gather state, write the update, done.

## Gather state

Read current state using these tools:

1. `sol call awareness status` — capture, processing, and onboarding state
2. `sol call identity self` — identity summary (skim for key changes)
3. `sol call calendar list` — today's events
4. `sol call routines list` — active routines and recent outputs
5. `sol call entities search --recent --limit 5` — recent entity activity

## Write awareness.md

Compose a structured bullet-point snapshot. Keep it under 30 lines. Use this format:

```
as of: {ISO 8601 datetime}
segment: {$SOL_SEGMENT}

## capture
- status: {active|stale|offline}
- streams: {list of active streams}

## calendar
- {key events for today, 1-3 bullets}

## activity
- {current activity state from sense, 1-2 bullets}

## routines
- {active routines and last-run status, 1-3 bullets}

## entities
- {recent entity activity, 1-2 bullets}

## partner
- {recency of last interaction, 1 bullet}
```

Omit sections that have no meaningful content. Never include prose — bullets only.

Write the result:

```bash
sol call identity awareness --write --value '{your content here}'
```
