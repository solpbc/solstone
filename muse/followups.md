{
  "type": "generate",

  "title": "Follow-Up Items",
  "description": "Detects promised tasks, commitments, and reminders for future action within each activity. Outputs a concise Markdown list of follow-ups with context.",
  "occurrences": "Whenever a future task or commitment is mentioned, create an occurrence with the expected action and deadline if known. Note who requested it and whether it is work or personal.",
  "hook": {"post": "occurrence"},
  "color": "#ffc107",
  "schedule": "activity",
  "activities": ["*"],
  "priority": 10,
  "output": "md",
  "instructions": {
    "sources": {"audio": true, "screen": false, "agents": {"screen": true}},
    "facets": true,
    "activity": true
  }

}

$activity_preamble

# Follow-up Identification

## Objective

Review this activity's transcript to find instances where an important future action is implied, requested, or promised, covering both professional and personal contexts.

Use the Activity Context and Activity State Per Segment sections above to understand what this activity involves and to focus on relevant content in the transcript.

## Approach

1. **Sequential Review**
   - Read the transcript chronologically, one block at a time.
   - Look for statements or screen cues indicating outstanding tasks, open questions, or commitments to reconnect later.

2. **Recognize Follow-up Triggers**
   - Phrases such as "I'll do that tomorrow," "Let's talk later," or "Need to check".
   - References to documents, links, or resources to revisit.
   - Unresolved issues or decisions deferred to a future date.
   - Personal reminders like errands, appointments, or messages to send.

3. **Assess Importance**
   - Note who initiated the follow-up and any deadlines or urgency mentioned.
   - Determine if the context is work-related or personal.
   - Prioritize items that appear critical for ongoing projects or relationships.

## Exclusions

- Content from concurrent activities unrelated to this $activity_type activity.
- Pure speculation or hypothetical scenarios without a concrete commitment.
- Duplicates of the same follow-up; merge overlapping mentions.

## Output Format

Produce a concise Markdown list capturing each follow-up with the following fields in individual blocks with a short title:

- **Time:** HH:MM:SS–HH:MM:SS (tight span)
- **Context** – short description of the discussion or screen activity.
- **Action Needed** – what follow-up is expected.
- **Work/Personal** – classify the nature of the task.
- **Confidence:** 0.0–1.0 calibration

Conclude with a brief summary (<= 100 words) highlighting the most significant follow-ups.

If no follow-ups are found in this activity, output only a brief sentence explaining why (e.g., "No follow-up items were identified during this $activity_type activity.").
