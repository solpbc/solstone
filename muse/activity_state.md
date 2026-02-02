{

  "title": "Activity State",
  "description": "Detects configured activities present in segment and tracks state across segments",
  "color": "#00897b",
  "schedule": "segment",
  "priority": 95,
  "multi_facet": true,
  "output": "json",
  "hook": {"pre": "activity_state"},
  "tier": 3,
  "thinking_budget": 2048,
  "max_output_tokens": 512,
  "instructions": {
    "sources": {"audio": true, "screen": true, "agents": false},
    "facets": "none"
  }

}

Detect which of the facet's configured activities are present in this segment.

## Input

You receive:
1. **Facet Activities** - The list of activities configured for this facet. Only detect these.
2. **Previous State** - What activities were active/ended in the prior segment (if available).
3. **Current Segment Content** - Audio transcript and screen activity from this recording window.

## Task

Analyze the current segment and determine:
- Which configured activities are happening at the END of this segment (active)
- Which previously active activities ended DURING this segment (ended)

Consider continuity: if an activity was active in the previous segment and you see evidence of it continuing, keep tracking it with the same `since` value and update the description if context has evolved.

## Output Format

Return a JSON object with two arrays:

```json
{
  "active": [
    {
      "activity": "meeting",
      "since": "143000_300",
      "description": "Design review with UX team discussing navigation patterns",
      "level": "high"
    }
  ],
  "ended": [
    {
      "activity": "email",
      "since": "140000_300",
      "description": "Replied to deployment notification from ops team"
    }
  ]
}
```

### Field Definitions

**active** - Activities ongoing at segment end:
- `activity`: Activity ID from the configured list
- `since`: Segment key when this activity instance started (copy from previous if continuing, use current segment if new)
- `description`: Brief description of what this activity involves (update as context evolves)
- `level`: Engagement level - "high" (primary focus), "medium" (secondary), "low" (background)

**ended** - Activities that stopped during this segment:
- `activity`: Activity ID
- `since`: When this instance started (for duration tracking)
- `description`: Final summary of what the activity was

## Rules

1. **Only detect configured activities** - Ignore activity that doesn't match the facet's list
2. **One instance per type** - If a meeting ends and another starts, the first goes to `ended`, the new one to `active`
3. **Preserve `since`** - For continuing activities, keep the original start segment
4. **Update descriptions** - As activities continue, refine the description with new context
5. **Empty is valid** - `{"active": [], "ended": []}` is correct when no activities detected

## Examples

**New activity starts:**
```json
{
  "active": [{"activity": "coding", "since": "143500_300", "description": "Implementing user auth flow", "level": "high"}],
  "ended": []
}
```

**Activity continues from previous:**
```json
{
  "active": [{"activity": "meeting", "since": "140000_300", "description": "Sprint planning - now discussing blockers", "level": "high"}],
  "ended": []
}
```

**One activity ends, another starts (same type):**
```json
{
  "active": [{"activity": "meeting", "since": "144500_300", "description": "1:1 with manager", "level": "high"}],
  "ended": [{"activity": "meeting", "since": "140000_300", "description": "Sprint planning completed"}]
}
```

**Multiple concurrent activities:**
```json
{
  "active": [
    {"activity": "meeting", "since": "143000_300", "description": "Team standup", "level": "high"},
    {"activity": "messaging", "since": "143000_300", "description": "Slack thread about deployment", "level": "low"}
  ],
  "ended": []
}
```

**No activities detected:**
```json
{"active": [], "ended": []}
```

Return ONLY the JSON object, no other text.
