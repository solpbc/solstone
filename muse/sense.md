{
  "type": "generate",

  "title": "Segment Sense",
  "description": "Unified segment understanding — density, content type, entities, facets, speakers, and routing recommendations in a single pass",
  "color": "#ff6f00",
  "schedule": "segment",
  "priority": 10,
  "tier": 3,
  "thinking_budget": 4096,
  "max_output_tokens": 3072,
  "output": "json",
  "instructions": {
    "sources": {"transcripts": true, "percepts": true, "agents": false},
    "facets": true
  }

}

$segment_preamble

# Segment Sense

Analyze this recording segment and produce a single structured assessment covering density, content type, activity, entities, facets, speakers, and processing recommendations.

## Task

Read the transcript and screen data. Produce a JSON object with ALL of the following fields.

## Output Schema

```json
{
  "density": "active|low_change|idle",
  "content_type": "meeting|coding|browsing|email|messaging|reading|idle|mixed",
  "activity_summary": "1-3 sentence description of what happened",
  "entities": [
    {"type": "Person|Company|Project|Tool", "name": "Full Name", "context": "Why this entity matters in this segment"}
  ],
  "facets": [
    {"facet": "facet_id", "activity": "1-sentence description for this facet", "level": "high|medium|low"}
  ],
  "meeting_detected": false,
  "speakers": [],
  "recommend": {
    "screen_record": false,
    "speaker_attribution": false,
    "pulse_update": false
  }
}
```

## Field-by-Field Instructions

### density
Classify based on content volume:
- **active**: Meaningful transcript content (>10 lines or >100 words) OR meaningful screen changes (>5 distinct frames with different visual descriptions)
- **low_change**: Some content but minimal change — fewer than 10 transcript lines AND fewer than 5 distinct screen states. Something is happening but it's repetitive or minimal.
- **idle**: Near-zero content — fewer than 3 transcript lines AND fewer than 3 distinct screen frames. Static screen, silence, or system noise only.

### content_type
The dominant activity type observed:
- **meeting**: Multi-person discussion with turn-taking (video call, in-person meeting, phone call)
- **coding**: Writing or editing code, using a terminal, IDE, or code review tool
- **browsing**: Web browsing, reading articles, searching
- **email**: Reading or composing email
- **messaging**: Chat applications (Slack, Teams, Discord, iMessage)
- **reading**: Focused reading of documents, PDFs, books
- **idle**: No meaningful activity
- **mixed**: Multiple distinct activity types with no clear dominant one

### activity_summary
Describe what $preferred did during this segment using action verbs. Be specific — name the tools, people, projects, and actions. Ban passive words: never use "reviewing", "monitoring", "tracking", "checking", "observing", "maintaining", "managing." Use instead: wrote, sent, discussed, created, switched to, typed, said, decided, asked, proposed.

### entities
Extract named entities. Four types only:
- **Person**: Individual people by name. Prefer full names. Consolidate variants ("JB" + "John Borthwick" → one entity "John Borthwick"). Skip ambiguous first-name-only references.
- **Company**: Businesses and organizations.
- **Project**: Named projects, products, or codebases.
- **Tool**: Software applications and services.

Skip URLs, domains, filenames, paths. Each entity needs type, name, and context (brief description of the entity's role in this segment).

### facets
Classify into the owner's configured facets. Only include facets with clear evidence of activity. For each:
- `facet`: The facet ID slug
- `activity`: 1-sentence description of what was observed for this facet
- `level`: "high" (primary focus), "medium" (significant), "low" (brief/peripheral)

### meeting_detected
`true` if any of these conditions are met:
- Screen shows a video conferencing app (Zoom, Meet, Teams, Webex) with participant panels
- Audio shows multiple speakers with conversational turn-taking
- Meeting-style patterns: greetings, introductions, agenda items, discussion, decisions

`false` otherwise. Podcasts, streaming content, and recorded media do NOT count.

### speakers
If `meeting_detected` is true, extract participant names from:
1. Visible participant list/panel on screen
2. Names spoken in conversation — direct address ("Thanks, Sarah"), mentions ("John was saying...")
3. Self-introductions ("Hi, I'm Alex from...")

Prefer complete canonical forms (full names when identifiable). Do NOT include the journal owner's name. Return `[]` if no meeting or no names identified.

### recommend
Processing recommendations for downstream agents:
- **screen_record**: `true` if density is "active" AND there is meaningful screen content worth documenting (not just a static/repetitive screen)
- **speaker_attribution**: `true` if `meeting_detected` is true AND there are multiple speakers to attribute
- **pulse_update**: `true` if this segment represents a meaningful change in activity — new activity started, activity ended, significant context shift, or noteworthy event occurred. `false` for continuation of the same activity with no notable change.

## Rules

1. Every field is required. Never omit a field.
2. `entities` and `speakers` may be empty arrays `[]`.
3. `facets` may be empty array `[]` if no configured facets match.
4. Be precise with density — misclassifying active segments as idle is the worst error.
5. For content_type, choose the single best match. Use "mixed" sparingly — only when there are truly multiple equal activities.
6. Activity summary must describe observable actions, not inferred states.

Return ONLY the JSON object, no other text or explanation.
