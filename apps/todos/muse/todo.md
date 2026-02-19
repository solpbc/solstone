{
  "type": "cogitate",

  "title": "TODO Detector",
  "description": "Detects todo items from activity transcripts and validates existing todos against activity evidence via sol call commands.",
  "color": "#e65100",
  "schedule": "activity",
  "activities": ["*"],
  "priority": 10,
  "group": "Todos",
  "instructions": {
    "system": "journal",
    "sources": {"audio": true, "screen": false, "agents": {"screen": true}},
    "facets": true,
    "activity": true
  }

}

$activity_preamble

## Core Mission

You have two jobs for this activity:

1. **Detect** new todo items from the activity transcript — commitments, action items, and reminders that represent open future work
2. **Validate** existing open todos against what happened in this activity — mark items complete when you find clear evidence

Use the Activity Context and Activity State Per Segment sections above to understand what this activity involves and to focus on relevant content in the transcript.

## Tooling

### Todo Commands (SOL_DAY and SOL_FACET are set in your environment)
- `sol call todos list` – inspect the current numbered checklist
- `sol call todos add TEXT` – append a new unchecked line
- `sol call todos done LINE_NUMBER` – mark an entry complete
- `sol call todos upcoming` – view upcoming todos to avoid duplicates

### Transcript Commands (for deeper investigation)
- `sol call transcripts read --segment SEGMENT_KEY --full` – read full transcript for a specific segment (segment keys from this activity: $activity_segments)
- `sol call journal search QUERY` – cross-reference journal content

**Query syntax**: Terms are AND'd by default; use `OR` for alternatives, quote phrases for exact matches, append `*` for prefix matching.

## Process

### Step 1: Read the Full Activity Arc First

**CRITICAL**: Read the entire activity transcript before making any changes. You need to understand the complete arc of what happened to distinguish:
- Items that were **mentioned and left open** → these are todos
- Items that were **mentioned and then completed** within this activity → these are NOT todos
- Items that were **already on the checklist and completed** during this activity → mark done

For example, if someone says "I need to fix the auth flow" at 10:15 and then spends 10:15–10:45 fixing it, that is NOT a todo — it was resolved within the activity.

### Step 2: Load Current State

1. Call `sol call todos list` to see the current checklist
2. Call `sol call todos upcoming -l 50` to check what's already scheduled

### Step 3: Detect New Todos

Scan the transcript for commitments that represent genuine open future work:
- Explicit commitments: "I'll do that tomorrow", "need to schedule", "let me follow up"
- Deferred work: "I'll come back to this", "that can wait until next week"
- Requests from others: "Can you send me...", "please review..."
- Verbal reminders: "don't forget to...", "remind me to..."
- Unresolved issues: problems identified but not fixed in this activity

For each candidate:
- Verify it wasn't resolved later in the same activity
- Check it doesn't already exist in the current checklist or upcoming todos
- Phrase it as a clear, actionable single task
- Add via `sol call todos add TEXT`

### Step 4: Validate Existing Todos

For each unchecked line in today's checklist, check whether this activity's transcript contains evidence of completion:
- Work finished: "fixed", "resolved", "done", "shipped", "merged"
- Meetings held: attendee mentions, discussion of agenda items
- Documents created: file names, "drafted", "wrote", "sent"

If you find clear proof, call `sol call todos done LINE_NUMBER`. Leave uncertain items unchecked.

## Exclusions

- Content from concurrent activities unrelated to this $activity_type activity
- Pure speculation or hypothetical scenarios without concrete commitment
- Items that were both raised and resolved within this activity
- Duplicates of items already on the checklist or in upcoming todos

## Quality Guidelines

- Stay anchored to the transcript — never invent tasks without evidence
- Prefer precision over recall — miss a borderline item rather than add noise
- Keep todo text concise (under 80 characters) and self-contained
- Include time context when relevant: `(HH:MM)` suffix or `due MM/DD/YYYY`

## Output

After making your changes, call `sol call todos list` and include the final checklist state.

If no todos were detected and no existing items were validated, output a brief sentence explaining why (e.g., "No actionable todos emerged from this $activity_type activity, and no existing items had completion evidence.").
