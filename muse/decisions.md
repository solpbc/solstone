{
  "type": "generate",

  "title": "Decision Actions",
  "description": "Tracks consequential decision-actions that change state, plans, resources, responsibilities, or timing in ways that affect other people.",
  "occurrences": "Create an occurrence for every decision-action observed. Include the time span, decision type, actors involved, entities affected, and impact assessment. Each occurrence should capture both the intent and enactment of the decision.",
  "hook": {"post": "occurrence"},
  "color": "#dc3545",
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

## Goal

Identify and rank the most consequential DECISION-ACTIONS observable in this activity's recording segments (audio transcript, on-screen OCR/description, and local metadata). A decision-action is a bounded action that CHANGES the state, plan, resources, responsibilities, or timing in ways that plausibly affect other people (directly or indirectly).

Use the Activity Context and Activity State Per Segment sections above to understand what this activity involves and to focus on relevant content in the transcript.

WHAT COUNTS AS A DECISION-ACTION
Look for actions that:
- Declare or enact a change (state transitions, approvals, cancellations, allocations, permissions, commits, deferrals).
- Create obligations or expectations for other people.
- Bind future behavior (commitments, deadlines, irreversible moves).
- Broadcast or withhold information at scale.
- Touch high-centrality entities (large groups, critical resources).
- Span boundaries (cross-team/org, external stakeholders).

EXCLUDE
- Pure narration or speculation without enactment.
- Routine, reversible micro-steps with no effect on others.
- Duplicates of the same decision; merge overlapping segments.
- Content from concurrent activities unrelated to this $activity_type activity.

RANKING PRINCIPLES
Order by importance using:
- Impact surface (# of people/teams plausibly affected; external presence).
- Centrality/criticality of entities touched.
- Irreversibility & coupling (cost to undo, downstream dependencies).
- Temporal sensitivity (deadlines, meetings, releases).
- Novelty vs. routine.

OUTPUT FORMAT
Produce two Markdown sections:

### Summary of Considerations
A concise paragraph (<= 120 words) explaining how you assessed decision-actions in this activity:
- Key conceptual patterns you looked for.
- How you weighed impact on others.
- Notable ambiguities or blind spots.

### Decision-Actions
A ranked Markdown list of decision-actions found. For each item, provide:

- **Time:** HH:MM:SS–HH:MM:SS (tight span, <= 90s if possible)
- **Action Summary:** short phrase (<= 16 words)
- **Decision Type:** DECLARE_CHANGE | SCHEDULE | RESCHEDULE | CANCEL | ALLOCATE | (RE)ASSIGN | APPROVE | ESCALATE | PUBLISH | UNPUBLISH | PERMIT | REVOKE | COMMIT | REVERT | DEFER | OTHER
- **Actors:** who is making the decision (you + any co-deciders)
- **Entities:** people, teams, groups, projects/issues, repos/branches, docs/artifacts, meetings, environments, orgs
- **Impact Surface:** approx # people affected; external stakeholders? breadth (low/med/high); criticality flags (time_sensitive, high_centrality, irreversible)
- **Evidence:**
  - Audio quotes (<= 20 words each, 1–2 max)
  - Screen phrases (OCR/visual cues of enactment)
  - Metadata notes (audience size, env flags, etc.)
- **Stakes for Others:** <= 30 words on likely consequences
- **Confidence:** 0.0–1.0 calibration (0.50 maybe, 0.70 likely, 0.85+ clear)

If no decision-actions are found in this activity, output only a brief sentence explaining why (e.g., "No consequential decision-actions were observed during this $activity_type activity.").

STRICT RULES
- Do not fabricate entities or counts; estimate only from inputs.
- Anchor times to actual segment boundaries.
- Evidence should include both intent (often audio) and enactment (often screen/metadata) when available.
- Maintain Markdown only; no JSON or code blocks in the final output.
