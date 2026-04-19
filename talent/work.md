{
  "type": "generate",
  "title": "Work Story",
  "description": "Generates a work story, topics, and structured commitments, closures, and decisions to merge onto the activity record.",
  "color": "#6d4c41",
  "schedule": "activity",
  "activities": ["coding", "browsing", "reading"],
  "priority": 20,
  "output": "json",
  "hook": {"post": "story"},
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

Return exactly this six-field JSON object:
- `body`: string narrative prose about the work performed and what changed.
- `topics`: array of short string tags; use `[]` when there are no durable topics worth preserving.
- `confidence`: float from 0.0 to 1.0.
- `commitments`: array of objects with required string fields `owner`, `action`, `counterparty`, `when`, `context`.
  Example: `{"owner":"Avery","action":"post the benchmark results","counterparty":"Priya","when":"after lunch","context":"Avery said the new retry benchmark would be shared once the run completed."}`
- `closures`: array of objects with required string fields `owner`, `action`, `counterparty`, `resolution`, `context`. `resolution` must be one of `sent`, `done`, `signed`, `dropped`, `deferred`.
  Example: `{"owner":"Avery","action":"follow-up PR","counterparty":"Priya","resolution":"done","context":"Avery noted the cleanup PR was merged during this work block."}`
- `decisions`: array of objects with required string fields `owner`, `action`, `context`.
  Example: `{"owner":"Avery","action":"switch the retry path to queue-backed backoff","context":"The work session concluded that queue-backed backoff was simpler than the timer-based branch."}`

Return `[]` if you do not observe a clear commitment / closure / decision. Better to omit than invent.

Body requirements:
- Write one tight paragraph in chronological order.
- Emphasize concrete progress, investigation, blockers, and outcomes.
- Prefer the actual work performed over UI description.
- If evidence is partial, describe the most defensible story and keep the
  confidence honest.

Output a single JSON object with all six required fields: `body`, `topics`, `confidence`, `commitments`, `closures`, and `decisions`.
