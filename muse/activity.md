{
  "type": "generate",
  "disabled": true,

  "title": "Activity Synthesis",
  "description": "Synthesizes segment activity from content, focusing on observable changes and searchability.",
  "color": "#00bcd4",
  "schedule": "segment",
  "priority": 10,
  "output": "md",
  "instructions": {
    "sources": {"transcripts": true, "percepts": true, "agents": false},
    "facets": true
  }

}

$segment_preamble

# Segment Activity Synthesis

Report the key activities, discussions, and actions observable in this content.

$import_guidance

## Banned Language

Never use these words — they describe presence, not action:
- reviewing, monitoring, tracking, checking, observing, maintaining, managing

Use action verbs instead: wrote, sent, received, created, deleted, switched to, typed, said, discussed, decided, asked, proposed, resolved

## What to Report

For each item, identify what happened — the specific action, change, or exchange.

### Facets
Which project/context? Every segment has at least one.

### Keywords
Project names, people, problems, decisions, tools.

## Before Writing

For each item, ask: can I point to a SPECIFIC action, exchange, or change in the content? If not, omit it.

## Output Format

Concise markdown. Bullets preferred. Group by facet if multiple.
