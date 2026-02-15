{
  "type": "generate",

  "title": "Meeting Notes",
  "description": "Produces detailed meeting notes for each meeting activity, including participants, topics discussed, action items, and presentation details.",
  "occurrences": "Each meeting should generate an occurrence with start and end times, list of participants and a concise summary. If slides are present, mention them in the details field.",
  "hook": {"post": "occurrence"},
  "color": "#e83e8c",
  "schedule": "activity",
  "activities": ["meeting"],
  "priority": 10,
  "output": "md",
  "instructions": {
    "sources": {"audio": true, "screen": false, "agents": {"screen": true}},
    "facets": true,
    "activity": true
  }

}

$activity_preamble

# Meeting Notes

## Objective

Produce detailed meeting notes for this $activity_type activity. The activity system has already identified this as a meeting and provided the time range, description, and known participants above. Your job is to enrich that with thorough notes from the transcript.

Use the Activity Context and Activity State Per Segment sections above to understand the meeting's scope and participants.

## Extract Meeting Details

Prioritize the audio transcript as the primary source of truth:

1. **Participants**
   - Analyze the audio transcript for names of individuals speaking or referred to by name.
   - Use screen activity to supplement: meeting software participant lists, chat names, etc.
   - Consolidate names that overlap due to transcription errors.
   - Include $name as a default participant.

2. **Topics Discussed**
   - Synthesize the conversation into a concise summary of key subjects.
   - Note entities mentioned: people, teams, projects, technologies, companies, organizations.

3. **Meeting Brief**
   - Create a short one-liner title describing the meeting for list context.

4. **Slides Presented**
   - Note whether presentation slides were visible on screen (PowerPoint, Google Slides, Keynote, slide transitions, structured presentation content).
   - If slides were shown, provide a short summary of slide content and themes.

5. **Key Outcomes**
   - Decisions made during the meeting.
   - Action items or follow-ups assigned to specific people.
   - Open questions left unresolved.

## Exclusions

- Content from concurrent activities unrelated to this meeting.
- Duplicates of the same topic; merge overlapping discussion threads.

## Output Format

Produce a friendly Markdown document with:

- **Brief** – one-liner meeting title
- **Time:** $segment_start–$segment_end
- **Participants** – list of names involved
- **Topics Discussed** – concise summary of key subjects and entities mentioned
- **Slides Presented** – yes/no, with short description if yes
- **Key Outcomes** – decisions, action items, and open questions

Conclude with a brief summary (<= 100 words) of the meeting's significance and any immediate next steps.

If the transcript does not contain substantive meeting content (e.g., a false positive from the activity detector), output only a brief sentence explaining why (e.g., "No substantive meeting content was found in this activity's transcript.").
