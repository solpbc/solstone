{
  "type": "generate",
  "title": "Event Story",
  "description": "Generates an event story, topics, and structured commitments, closures, and decisions to merge onto the activity record.",
  "color": "#ff7043",
  "schedule": "activity",
  "activities": ["appointment", "event", "travel", "errand", "celebration", "deadline", "reminder"],
  "priority": 20,
  "output": "json",
  "schema": "story.schema.json",
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

# Event Story

Write JSON only. No markdown fences. No prose outside the JSON object.

Summarize what happened in this event, appointment, errand, travel block, or
deadline-related activity. Participation and entity extraction already happened
upstream. Use that context; do not re-extract people or entities into new
structures.

Return exactly this six-field JSON object:
- `body`: string narrative prose describing what happened and any outcome.
- `topics`: array of short string tags; use `[]` when there are no durable topics worth preserving.
- `confidence`: float from 0.0 to 1.0.
- `commitments`: array of objects with required string fields `owner`, `action`, `counterparty`, `when`, `context`.
  Example: `{"owner":"Jordan","action":"send the updated itinerary","counterparty":"Taylor","when":"tonight","context":"Jordan said the revised travel plan would be sent after the delay was confirmed."}`
- `closures`: array of objects with required string fields `owner`, `action`, `counterparty`, `resolution`, `context`. `resolution` must be one of `sent`, `done`, `signed`, `dropped`, `deferred`.
  Example: `{"owner":"Jordan","action":"hotel confirmation","counterparty":"Taylor","resolution":"signed","context":"Jordan completed and signed the hotel check-in form during the event."}`
- `decisions`: array of objects with required string fields `owner`, `action`, `context`.
  Example: `{"owner":"Travel group","action":"take the shuttle instead of renting a car","context":"After the delay, the group agreed the shuttle was the fastest remaining option."}`

Return `[]` if you do not observe a clear commitment / closure / decision. Better to omit than invent.

Body requirements:
- Write one tight paragraph in chronological order.
- Capture the event context, notable actions, and any decision or outcome.
- Prefer what actually occurred over generic labels from the activity type.
- If evidence is thin, keep the narrative modest and confidence honest.

Output a single JSON object with all six required fields: `body`, `topics`, `confidence`, `commitments`, `closures`, and `decisions`.
