{

  "title": "Activity Synthesis",
  "description": "Synthesizes segment activity from screenshots and audio, focusing on observable changes and searchability.",
  "color": "#00bcd4",
  "schedule": "segment",
  "priority": 10,
  "output": "md",
  "instructions": {
    "sources": {"audio": true, "screen": true, "agents": false},
    "facets": "short"
  }

}

$segment_preamble

# Segment Activity Synthesis

## Core Rule

ONLY report what CHANGED between screenshots or was SPOKEN in audio.
If content looks the same across frames, skip it entirely.

## Your Inputs

- **Screenshots**: Sampled across this segment. Compare frames - what's different?
- **Audio**: Transcript of speech. What was said?

## Banned Language

Never use these words - they describe presence, not action:
- reviewing, monitoring, tracking, checking, observing, maintaining, managing

Use action verbs instead: wrote, sent, received, created, deleted, switched to, typed, said, discussed, decided

## What to Report

For each item, identify the CHANGE:
- "Typed message to X about Y" (text appeared)
- "Switched from Gmail to Terminal" (window focus changed)
- "Received reply from X" (new message appeared)
- "Said X about Y in meeting" (audio evidence)

If you cannot name the specific change, do not include it.

### Facets
Which project/context? Every segment has at least one.

### Keywords
Project names, people, problems, decisions, tools.

## Before Writing

For each item, ask:
- Can I point to a SPECIFIC CHANGE between screenshots?
- Or SPECIFIC WORDS spoken in audio?

If neither, omit it.

## SKIP Entirely

- Windows that look identical in first and last frame
- Apps open but showing same content throughout
- Background windows never brought to focus
- Anything you'd describe as "had open" or "was visible"

## Output Format

Concise markdown. Bullets preferred. Group by facet if multiple.
