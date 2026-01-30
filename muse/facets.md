{

  "title": "Facet Classification",
  "description": "Classifies segment activity into relevant facets based on other segment outputs",
  "color": "#7c4dff",
  "schedule": "segment",
  "priority": 90,
  "tier": 3,
  "thinking_budget": 1024,
  "max_output_tokens": 512,
  "output": "json",
  "instructions": {
    "system": "journal",
    "facets": "short",
    "sources": {"audio": false, "screen": false, "agents": true}
  }

}

Classify the activity in this recording segment into relevant facets.

## Input

You receive segment output summaries (activity synthesis, screen record, entities) from the current recording window.

## Output Format

Return a JSON array. Each object has:
- `facet`: facet ID (the slug in parentheses, like "work" or "personal")
- `activity`: 1-sentence description of what was observed for this facet
- `level`: engagement level - "high" (primary focus), "medium" (significant), or "low" (brief/peripheral)

Only include facets with clear evidence of activity. Omit facets with no connection to observed content.

## Examples

Single facet, focused work:
```json
[{"facet": "work", "activity": "Code review and authentication module development", "level": "high"}]
```

Multiple facets:
```json
[
  {"facet": "work", "activity": "Team standup meeting discussing sprint progress", "level": "high"},
  {"facet": "personal", "activity": "Brief personal email check", "level": "low"}
]
```

No clear facet match:
```json
[]
```

Return ONLY the JSON array, no other text.
