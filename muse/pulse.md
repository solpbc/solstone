{
  "type": "cogitate",

  "title": "Pulse",
  "description": "Living narrative of the owner's day — updated each segment",
  "schedule": "segment",
  "priority": 99,
  "tier": 3,
  "max_output_tokens": 1000,
  "instructions": {"system": "journal", "facets": true, "now": true}

}

# Pulse

You are generating the owner's Pulse — a living narrative that captures the shape
of their day so far. This runs every segment, building on the previous pulse.

This is not a conversation. Gather context, write the pulse, done.

## Gather context

Read current state using these tools:

1. `sol call sol pulse` — previous pulse (may not exist yet; that's fine)
2. `sol call sol self` — who the owner is
3. `sol call calendar list` — today's events
4. `sol call todos list` — pending action items
5. `sol call entities search --recent` — recent entity activity
6. `sol call awareness status` — system health (brief check)

## Write the pulse

Compose a short, natural narrative (3-8 sentences) describing the shape of the
owner's day so far. Lead with what matters most right now. Mention upcoming events,
active work, and anything that shifted since the last pulse.

After the narrative, include a `## needs you` section — a ranked list of 3-7
action items the owner should notice. Format as markdown bullet points:

````
## needs you
- Most urgent item
- Second priority
- Third item
````

Draw needs-you items from: pending todos, upcoming calendar events needing prep,
entity follow-ups, and anything the narrative highlights as important.

## Write output

Write the complete pulse (YAML frontmatter + narrative + needs-you section) via:

```bash
cat <<'EOF' | sol call sol pulse --write
---
updated: 2026-03-22T14:35:00
segment: 143022_300
source: pulse-cogitate
---

[Your narrative here]

## needs you
- Item 1
- Item 2
EOF
```

The `updated` field must be an ISO 8601 datetime (no timezone). The `segment`
field is the current segment key from $SOL_SEGMENT.

Then append a log entry to `sol/pulse-log.jsonl` (same directory as pulse.md):

```bash
JOURNAL=$(sol config env | head -1)
echo '{"ts": 1742680500, "segment": "143022_300", "narrative": "...", "needs_you": ["Item 1", "Item 2"]}' >> "$JOURNAL/sol/pulse-log.jsonl"
```

Use the current epoch timestamp for `ts`. Keep the narrative value brief (first
sentence or two). The needs_you array should match the items from the ## needs you
section.

## Guidelines

- Be concise. The owner sees this on their landing page.
- Don't repeat the same narrative if nothing changed — note stability.
- Don't include greetings or meta-commentary about being an AI.
- If the day is just starting and there's little data, say so briefly.
