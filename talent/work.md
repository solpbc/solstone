{
  "type": "generate",
  "title": "Work Story",
  "description": "Writes a structured narrative span row for coding, browsing, and reading activities.",
  "color": "#6d4c41",
  "schedule": "activity",
  "activities": ["coding", "browsing", "reading"],
  "priority": 10,
  "output": "json",
  "hook": {"post": "spans"},
  "load": {
    "transcripts": true,
    "percepts": true,
    "talents": false
  }
}

$facets

$activity_context

$activity_preamble

# Work Story

Write JSON only. No markdown fences. No prose outside the JSON object.

Summarize what this person accomplished, investigated, or worked through during
the activity. Participation and entity extraction already happened upstream.
Use that context; do not re-extract people or entities into new structures.

Return exactly these three fields:
- `body`: string narrative prose about the work performed and what changed.
- `topics`: array of 3-8 short string tags.
- `confidence`: float from 0.0 to 1.0.

Body requirements:
- Write one tight paragraph in chronological order.
- Emphasize concrete progress, investigation, blockers, and outcomes.
- Prefer the actual work performed over UI description.
- If evidence is partial, describe the most defensible story and keep the
  confidence honest.

Output a single JSON object with only `body`, `topics`, and `confidence`.
