{
  "type": "generate",

  "title": "Messaging Summary",
  "description": "Extracts contacts, channels, apps, and message content from completed messaging and email activities.",
  "color": "#78909c",
  "schedule": "activity",
  "activities": ["messaging", "email"],
  "priority": 10,
  "output": "md",
  "load": {"transcripts": true, "percepts": false, "talents": {"screen": true}}
}

$facets

$activity_context

$activity_preamble

# Messaging & Email Analysis

## Objective

Produce a detailed summary of this $activity_type activity. The activity system has already identified this as a messaging or email session and provided the time range, description, and known participants above. Your job is to enrich that with thorough extraction from the transcript.

Use the Activity Context and Activity State Per Segment sections above to understand the session's scope and participants.

## Extract Communication Details

Prioritize the audio transcript and screen activity together:

1. **Application & Account**
   - Identify which messaging or email app is in use (Gmail, Slack, Messages, Discord, Teams, Sococo, etc.).
   - Determine the account if deducible from screen content.

2. **Contacts & Channels**
   - Capture names of people or channels actively involved.
   - Note whether each contact was the sender, recipient, or part of a group thread.
   - Include $name as a default participant.

3. **Actions**
   - Distinguish whether $name was reading, composing, replying, or sending.
   - If multiple exchanges occur with different participants, capture them individually.
   - If multiple exchanges occur with the same participants, summarize them together.

4. **Message Content**
   - When message text is visible on screen, summarize what was read or written, noting all important entities.
   - If only partial content is visible, summarize the visible portion and note it is incomplete.

5. **Context Classification**
   - Assess whether each interaction is work-related or personal based on participants, content, and application.

## Exclusions

- Content from concurrent activities unrelated to this $activity_type session.
- Brief notifications or popups unless they show sustained interaction.
- Duplicates of the same exchange; merge overlapping mentions.

## Output Format

Produce a friendly Markdown document with each messaging interaction in its own section with a short title:

- **Time:** $segment_start–$segment_end (or tighter span if identifiable)
- **App** – the messaging or email application used
- **Contacts** – people or channels involved
- **Context** – work or personal classification
- **Action** – reading, composing, replying, or sending
- **Summary** – short recap of visible message contents

List interactions in chronological order.

Conclude with a brief summary (<= 100 words) of who was communicated with, via which apps, and what topics were discussed.

If the transcript does not contain substantive messaging content (e.g., a false positive from the activity detector), output only a brief sentence explaining why (e.g., "No substantive messaging content was found in this activity's transcript.").
