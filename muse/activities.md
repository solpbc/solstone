{
  "type": "generate",

  "title": "Activity Records",
  "description": "Detects completed activities across all facets and writes records with synthesized descriptions",
  "color": "#00695c",
  "schedule": "segment",
  "priority": 96,
  "output": "json",
  "hook": {"pre": "activities", "post": "activities"},
  "tier": 3,
  "thinking_budget": 4096,
  "max_output_tokens": 2048,
  "instructions": {
    "sources": {"audio": false, "screen": false, "agents": false},
    "facets": false
  }

}

You are given a set of completed activities organized by facet. Each activity has been tracked across multiple recording segments and includes per-segment descriptions and engagement levels.

## Task

For each activity, synthesize all the per-segment descriptions into a single cohesive description that captures the full arc of the activity from start to finish. Merge details, don't just concatenate — produce a readable narrative summary.

## Output Format

Return a JSON object keyed by facet name, where each value is an array of activity objects:

```json
{
  "facet_name": [
    {
      "id": "activity_id",
      "description": "Unified description of the full activity span..."
    }
  ]
}
```

Rules:
- Keep descriptions concise but complete (1-3 sentences)
- Capture the key details: what was done, with what tools/people, and any notable outcomes
- If an activity was brief (1-2 segments), a short description is fine
- Preserve important entity names and specific details from the segment descriptions
- Only include facets and activities provided in the input — do not invent new ones

Return ONLY the JSON object, no other text.
