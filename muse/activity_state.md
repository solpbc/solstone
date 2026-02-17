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
  "max_output_tokens": 3072,
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
- **continuing** - Was active in the previous segment AND the same type of activity is still happening. **This is the default** when the same activity type appears in both previous state and current segment. Changing focus within the same activity (different files, different subtask, different topic) is still "continuing" — update the description instead.
- **new** - This activity type was NOT active in the previous segment, or there was a clear session boundary (e.g., one meeting ended and a distinctly different meeting with different people began). Shifting focus within the same activity type is NOT "new."
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
2. **Prefer continuing** — If the same activity type was active in Previous State and you see evidence of that type still happening, always use `"continuing"`. Switching files, changing subtasks, or shifting focus within the same activity type is NOT a new activity — it is a continuation with an updated description. Only use `"new"` when the activity type was not previously active, or there is a clear session boundary (e.g., one meeting ended and a different meeting with different participants began).
3. **Active vs. visible** — Only report an activity if the user is actively engaged with it. Look for evidence: typing, clicking, new content, spoken discussion, or focused reading/review. An application merely visible on screen but unchanged and not being read is NOT active.
4. **Report endings** — If an activity listed as **active** in the Previous State is no longer happening, report it as `"ended"`. Only report endings for activities that were active — do not re-report activities that already ended previously.
5. **Same-type transitions** — These are rare and apply mainly to meetings/calls: if one meeting ends and a clearly different meeting begins (different participants, different topic), report the old as `"ended"` and the new as `"new"`. For other activity types like coding, browsing, or reading, a change in focus is just a continuation — do NOT split.
6. **Update descriptions** — As activities continue, evolve the description to reflect current focus. The description changing is normal and expected for continuing activities.
7. **Empty is valid** — `[]` is correct when no activities are detected

## Examples

**Activity continues — same type, different focus (most common):**

Previous state had: `coding [high]: Implementing user auth flow`
```json
[{"activity": "coding", "state": "continuing", "description": "Writing tests for the auth flow and fixing a bug in token refresh", "level": "high", "active_entities": ["Claude Code", "VS Code"]}]
```

**Activity continues — same type, different project:**

Previous state had: `coding [high]: Working on backend API endpoints`
```json
[{"activity": "coding", "state": "continuing", "description": "Switched to updating the CLI tool's help text and argument parsing", "level": "high", "active_entities": ["VS Code", "terminal"]}]
```

**New activity starts (no previous state for this type):**
```json
[{"activity": "messaging", "state": "new", "description": "Slack thread about deployment", "level": "low", "active_entities": ["DevOps"]}]
```

**Meeting transition (rare same-type split — different participants):**
```json
[
  {"activity": "meeting", "state": "ended", "description": "Sprint planning completed"},
  {"activity": "meeting", "state": "new", "description": "1:1 with manager about performance review", "level": "high", "active_entities": ["Manager Name"]}
]
```

**No activities detected:**
```json
[]
```

Return ONLY the JSON array, no other text.
