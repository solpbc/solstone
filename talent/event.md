{
  "type": "generate",
  "title": "Event Story",
  "description": "Writes a structured narrative span row for appointment, event, travel, errand, celebration, deadline, and reminder activities.",
  "color": "#ff7043",
  "schedule": "activity",
  "activities": ["appointment", "event", "travel", "errand", "celebration", "deadline", "reminder"],
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

# Event Story

Write JSON only. No markdown fences. No prose outside the JSON object.

Summarize what happened in this event, appointment, errand, travel block, or
deadline-related activity. Participation and entity extraction already happened
upstream. Use that context; do not re-extract people or entities into new
structures.

Return exactly these three fields:
- `body`: string narrative prose describing what happened and any outcome.
- `topics`: array of 3-8 short string tags.
- `confidence`: float from 0.0 to 1.0.

Body requirements:
- Write one tight paragraph in chronological order.
- Capture the event context, notable actions, and any decision or outcome.
- Prefer what actually occurred over generic labels from the activity type.
- If evidence is thin, keep the narrative modest and confidence honest.

Output a single JSON object with only `body`, `topics`, and `confidence`.
