{

  "title": "Messaging Activity",
  "description": "Tracks use of email and chat applications across the day. Each interaction is summarized with participants, app used and visible message content.",
  "occurrences": "Create an occurrence for every message read or sent. Include the time block, app name, contacts involved and whether $preferred was reading or replying. Summaries should capture any visible text.",
  "hook": {"post": "occurrence"},
  "color": "#78909c",
  "schedule": "daily",
  "output": "md",
  "instructions": {
    "sources": {"audio": true, "screen": false, "agents": {"screen": true}},
    "facets": "short"
  }

}

$daily_preamble

# Workday Messaging and Email Analysis

**Input:** A markdown file containing a chronologically ordered transcript of a workday for $name. The transcript is organized by recording segments, each combining audio and screen activity for that time period.

**Objective:** Identify every instance where messaging or email applications are used, determine who $name is communicating with, summarize any visible message content, and record the time block when it occurs.

**Instructions:**

1. **Scan Each Time Block**
   - Read the transcript sequentially, one recording segment at a time.
   - Pay close attention to screen descriptions that show messaging or email windows (Gmail, Slack, Messages, Sococo, etc.).
   - Note all visual cues mentioning sending, reading, or replying to messages.

2. **Identify the Application and Contacts**
   - Determine which messaging or email app is in use and for what account (if it can be deduced).
   - Capture the names of contacts or channels actively corresponding with if visible on screen.
   - Assess whether the communication context is work-related or personal based on participants, content, and application used.

3. **Capture Message Actions**
   - Distinguish whether reading, composing, or replying.
   - If multiple messages are exchanged within the same time block with different participants, capture them individually.
   - If multiple messages are exchanged within the same time block with the same participants, summarize them together.

4. **Summarize Visible Content**
   - When message text is shown on screen, capture an accurate summary of what was read or written, noting all important entities visible.
   - If only partial content is visible, summarise the portion that can be seen and note that it is incomplete.

5. **Record the Time Block**
   - Use the timestamp at the start of each recording segment as the `timeBlock` for any messaging activity found within that segment.
   - If the interaction spans multiple segments, list each segment where it remains visible or active.

6. **Output Format**
   - Produce a nicely formatted Markdown document with the following information for each messaging interaction in its own section with a short title:
     * Time Block – the time window of the recording segment.
     * App – the messaging or email application used.
     * Contacts – people or channels involved.
     * Context – work or personal classification.
     * Action – reading, composing, replying, or sending.
     * Summary – short recap of visible message contents.
   - List the interactions in chronological order.

**Key Considerations:**
- Include email exchanges (inbox review, reading, replying, composing new emails) as well as Slack or other chat threads.
- Brief notifications or popups may be noted, but focus on sustained interactions where message content is visible or clear engagement.
- Be thorough yet avoid duplication—only include an interaction once per time block unless distinct messages are exchanged.
- Multitasking is common, so messaging windows may appear while other work continues; capture them whenever they are being interacted with or updating.

The final report should help quickly see who was communicated with, via which apps, and what topics were discussed throughout the day.
