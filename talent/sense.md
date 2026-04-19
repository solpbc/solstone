{
  "type": "generate",

  "title": "Segment Sense",
  "description": "Unified segment understanding — density, content type, entities, facets, speakers, and routing recommendations in a single pass",
  "color": "#ff6f00",
  "schedule": "segment",
  "priority": 5,
  "tier": 3,
  "thinking_budget": 4096,
  "max_output_tokens": 4096,
  "output": "json",
  "load": {"transcripts": true, "percepts": true, "talents": false}
}

$facets

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
    {"type": "Person|Company|Project|Tool", "name": "Full Name", "role": "attendee|mentioned", "source": "voice|speaker_label|transcript|screen|other", "context": "Why this entity matters in this segment"}
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
  },
  "emotional_register": "high_energy|tense|focused|collaborative|flat|celebratory|strained|neutral"
}
```

## Field-by-Field Instructions

### density
Classify based on whether meaningful human activity occurred:
- **active**: ANY of these: transcript has >5 lines or >50 words, screen shows the user interacting with content (browsing pages, typing, reading articles, using applications, scrolling), or screen descriptions mention different pages/views/applications. **Default to active if there is any user-directed activity, even if the screen looks similar across frames.** Web browsing and document reading ARE active.
- **low_change**: Minimal new content AND no user interaction — same static screen unchanged across all frames, fewer than 5 transcript words, no scrolling or navigation evident.
- **idle**: Near-zero content — fewer than 3 transcript lines AND fewer than 3 distinct screen frames. Static screen with no user activity, silence, or system noise only.

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
Extract ALL named entities mentioned in the content. Be thorough — extract every entity you can identify, not just the most prominent ones. Four types only:
- **Person**: Individual people by name. Prefer full names. Consolidate variants ("JB" + "John Borthwick" → one entity "John Borthwick"). Skip ambiguous first-name-only references. Include historical figures, authors, scientists, politicians — anyone mentioned by name.
- **Company**: Businesses and organizations. Include companies, government agencies (NASA, NOAA), universities, media outlets.
- **Project**: Named projects, products, or codebases. Include missions (OSIRIS-REx), initiatives, specific product models.
- **Tool**: Software applications and services. Include websites (Fox News, Wikipedia, Amazon), browser extensions, developer tools, hardware products mentioned by name.

**For screen content specifically:** Extract entities from visible text in screen descriptions — article headlines, page titles, product names, people mentioned in articles, organizations referenced. If the user is browsing a website about the Renaissance, extract the specific historical figures, art movements, and institutions mentioned.

Skip URLs, domains, filenames, paths. Each entity needs type, name, and context (brief description of the entity's role in this segment).

#### role
- **attendee**: The entity was directly participating in the live interaction during this segment. Use only for people who were actively present in the meeting or call.
- **mentioned**: The entity was referenced, quoted, shown on screen, or otherwise relevant, but was not directly participating.

Contamination guard: tool or product names visible on screen must be `source: screen` and `role: mentioned`, never `attendee`. Video-conference app names such as Google Meet or Zoom are platform/tool entities, not attendees. `role: attendee` requires `meeting_detected: true` for this same segment; when `meeting_detected: false`, every Person must be `role: mentioned` even if they spoke, were quoted, or were referenced in the transcript.

#### source
- **voice**: Use when the entity is identified from spoken audio content.
- **speaker_label**: Use when the entity comes from an explicit speaker/participant label in meeting UI or transcript metadata.
- **transcript**: Use when the entity appears in transcript text but not as an actively speaking participant signal.
- **screen**: Use when the entity is visible in screen content such as UI, documents, headlines, or app chrome.
- **other**: Use only when the entity is grounded in another clear signal that does not fit the categories above.

### facets
Classify into the owner's configured facets. Only include facets with clear, direct evidence of activity. Be precise — assign exactly ONE primary facet in most cases. Only add a second facet if there is genuinely distinct secondary activity. For each:
- `facet`: The facet ID slug — MUST be one of the configured facets listed in the input
- `activity`: 1-sentence description of what was observed for this facet
- `level`: "high" (primary focus), "medium" (significant), "low" (brief/peripheral)

**Facet assignment rules:**
- "meetings" — ONLY for live synchronous group meetings where $preferred is a participant. NOT for listening to lectures, podcasts, press conferences, or recorded media.
- "learning" — Educational content: lectures, tutorials, articles, research, Wikipedia, podcasts. Includes listening to press conferences or recorded briefings.
- "technical-work" — Hands-on technical tasks: coding, using developer tools, configuring software, installing extensions, product research for work purposes, technical document writing, online tool usage, price comparison shopping.

### meeting_detected
`true` ONLY if you can identify distinct, named participants in a live multi-person interaction:
- Screen shows a video conferencing app with participant panels showing names
- Audio has multiple distinct speakers who can be identified by name (from introductions, direct address, or context)
- The interaction is live/synchronous — NOT a recording, podcast, lecture, news conference, or media playback

`false` for: podcasts, press conferences, recorded interviews, solo narration, streaming content, lectures, or any audio where the speakers are media personalities rather than meeting participants. Even if multiple people are speaking, if they are NOT in a meeting with $preferred, set this to `false`.

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

### emotional_register
The observable emotional tone of the segment based on conversation tone, speech patterns, and behavioral signals — not inferred feelings. Choose the single best match:
- **high_energy**: Fast-paced, enthusiastic, productive momentum
- **tense**: Conflict, disagreement, pressure, frustration evident in tone or content
- **focused**: Quiet concentration, deep work, minimal interruption
- **collaborative**: Engaged multi-person work, building on each other's ideas
- **flat**: Low energy, going through motions, no strong signal either way
- **celebratory**: Wins acknowledged, positive outcomes, shared excitement
- **strained**: Fatigue, overload, pushing through difficulty
- **neutral**: No clear emotional register observable — use this as the default when the segment doesn't carry detectable emotional tone

## Rules

1. Every field is required. Never omit a field.
2. `entities` and `speakers` may be empty arrays `[]`.
3. `facets` may be empty array `[]` if no configured facets match.
4. Be precise with density — misclassifying active segments as idle is the worst error.
5. For content_type, choose the single best match. Use "mixed" sparingly — only when there are truly multiple equal activities.
6. Activity summary must describe observable actions, not inferred states.

Return ONLY the JSON object, no other text or explanation.
