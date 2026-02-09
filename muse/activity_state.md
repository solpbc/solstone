{
  "type": "generate",

  "title": "Activity State",
  "description": "Detects configured activities present in segment and tracks state across segments",
  "color": "#00897b",
  "schedule": "segment",
  "priority": 95,
  "multi_facet": true,
  "output": "json",
  "hook": {"pre": "activity_state", "post": "activity_state"},
  "tier": 3,
  "thinking_budget": 4096,
  "max_output_tokens": 2048,
  "instructions": {
    "sources": {"audio": true, "screen": true, "agents": false},
    "facets": true
  }

}

Detect which of the facet's configured activities are present in this segment.

## Input

You receive:
1. **Facet Activities** - The list of activities configured for this facet. Only detect these.
2. **Previous State** - What activities were active/ended in the prior segment (if available).
3. **Current Segment Content** - Audio transcript and screen activity from this recording window.

## Task

Analyze the current segment and determine the state of each detected activity:
- **continuing** - Was active in the previous segment AND you see evidence it is still happening
- **new** - Just started in this segment (or same type restarted — e.g., one meeting ended, another began)
- **ended** - Was active in the previous segment but stopped during this segment

## Output Format

Return a JSON array of activity objects:

```json
[
  {"activity": "meeting", "state": "continuing", "description": "Design review with UX team, now discussing navigation", "level": "high", "active_entities": ["Sarah Chen", "UX Team"]},
  {"activity": "messaging", "state": "new", "description": "Slack thread about deployment", "level": "low", "active_entities": ["DevOps"]},
  {"activity": "email", "state": "ended", "description": "Replied to deployment notification from ops team"}
]
```

### Field Definitions

- `activity`: Activity ID from the configured list
- `state`: One of `"continuing"`, `"new"`, or `"ended"`
- `description`: Brief description of what this activity involves (update as context evolves)
- `level`: Engagement level — `"high"` (primary focus), `"medium"` (secondary), `"low"` (background). Only for continuing/new, omit for ended.
- `active_entities`: Names of people, companies, projects, or tools that were noticeably active in this segment and associated with this activity. Only include entities with clear evidence of involvement (speaking, mentioned, visible on screen). Omit for ended.

## Rules

1. **Only detect configured activities** — Ignore activity that doesn't match the facet's list
2. **Active vs. visible** — Only report an activity if the user is actively interacting with it during this segment. An application merely visible on screen but unchanged is NOT active. Look for evidence of interaction: typing, clicking, new content, spoken discussion.
3. **Report endings** — If an activity listed as **active** in the Previous State is no longer happening, report it as `"ended"`. Only report endings for activities that were active — do not re-report activities that already ended previously.
4. **Same-type transitions** — If a meeting ends and a different meeting starts, report both: the old one as `"ended"` and the new one as `"new"`
5. **Update descriptions** — As activities continue, refine the description with new context
6. **Empty is valid** — `[]` is correct when no activities are detected

## Examples

**New activity starts:**
```json
[{"activity": "coding", "state": "new", "description": "Implementing user auth flow", "level": "high", "active_entities": ["Claude Code", "VS Code"]}]
```

**Activity continues from previous:**
```json
[{"activity": "meeting", "state": "continuing", "description": "Sprint planning - now discussing blockers", "level": "high", "active_entities": ["Alice Johnson", "Bob Smith"]}]
```

**One meeting ends, another starts:**
```json
[
  {"activity": "meeting", "state": "ended", "description": "Sprint planning completed"},
  {"activity": "meeting", "state": "new", "description": "1:1 with manager", "level": "high", "active_entities": ["Manager Name"]}
]
```

**No activities detected:**
```json
[]
```

Return ONLY the JSON array, no other text.
