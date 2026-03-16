{
  "type": "generate",

  "title": "Meeting Speakers",
  "description": "Detects meetings in the segment and extracts participant names from screen and conversation.",
  "schedule": "segment",
  "priority": 10,
  "output": "json",
  "color": "#e64a19",
  "exclude_streams": ["import.*"],
  "instructions": {
    "sources": {"transcripts": "required", "percepts": true, "agents": false}
  }

}

$segment_preamble

# Speaker Name Extraction

## Objective

Identify meetings or multi-person discussions in this segment and extract the names of participants.

## Detection

A meeting/discussion is detected when:
- Screen shows a video conferencing app (Zoom, Meet, Teams, Webex) with participant panels
- Audio shows multiple speakers with conversational turn-taking
- Meeting-style patterns: greetings, agenda items, discussion, decisions

## Name Sources

Extract participant names from:
1. Visible participant list/panel on screen
2. Names spoken in conversation - direct address ("Thanks, Sarah"), mentions ("John was saying...")
3. Self-introductions ("Hi, I'm Alex from...")

## Name Preferences

When naming speakers, prefer complete canonical forms:
- Use full names when identifiable: "Mitch Baumgartner" rather than just "Mitch"
- If you can determine both first and last name from context, include both
- Use consistent naming: the same person should have the same name form across outputs
- If only a first name is confidently known, use that rather than guessing the surname

## What to Ignore

- Podcasts, videos, or streaming content being watched
- Music or background audio
- The journal owner's name (they are always present)

## Output Format

Return a JSON array of speaker name strings. Include only names you can identify with reasonable confidence.

Examples:
- Meeting with identified speakers: `["Alice Chen", "Bob Smith", "Carol"]`
- Meeting but no names identified: `[]`
- No meeting detected: `[]`

Return ONLY the JSON array, no other text or explanation.
