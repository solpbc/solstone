{
  "type": "generate",
  "title": "Conversation Story",
  "description": "Writes a structured narrative span row for meeting, call, messaging, and email activities.",
  "color": "#00796b",
  "schedule": "activity",
  "activities": ["meeting", "call", "messaging", "email"],
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

# Conversation Story

Write JSON only. No markdown fences. No prose outside the JSON object.

Summarize this conversation as one coherent narrative for the full activity.
Participation and entity extraction already happened upstream. Reuse that context;
do not re-extract people or entities into new structures.

Return exactly these three fields:
- `body`: string narrative prose covering what was discussed, what moved, and any commitments.
- `topics`: array of 3-8 short string tags.
- `confidence`: float from 0.0 to 1.0.

Body requirements:
- Write one tight paragraph in chronological order.
- Include 1-3 short verbatim quotes inline only when they sharpen a decision,
  commitment, or disagreement.
- Focus on the actual exchange, not generic meeting boilerplate.
- If the activity mixes channels, unify them into one narrative rather than
  listing separate threads.

Output a single JSON object with only `body`, `topics`, and `confidence`.
