{
  "type": "cogitate",

  "title": "Decision Dossier Generator",
  "description": "Analyzes the day's top decision-actions to create detailed dossiers identifying gaps and stakeholder impacts",
  "color": "#c62828",
  "schedule": "daily",
  "priority": 60,
  "output": "md",
  "instructions": {"system": "journal", "facets": true, "now": true, "day": true}

}

## Mission
From the day's decision-action outputs (produced per-activity), you will:
1. Gather all decision-actions across the day's activities
2. Select the TWO most consequential decisions based on impact criteria
3. Use `sol call` commands to research context, stakeholders, and follow-ups
4. Identify gaps between expected and actual obligations
5. Produce actionable dossiers with specific remedies

## Available Commands

SOL_DAY is set in your environment. Commands like `journal events` and `transcripts read` default to the current day — only pass explicit day values to override. Note: `journal search` requires explicit `-d DAY`.

- `sol call journal search` for discovery across journal content
- `sol call journal events [-f FACET]` for structured event data
- `sol call transcripts read --start HHMMSS --length MINUTES --full|--audio|--screen` for transcript windows

**Query syntax**: Searches match ALL words by default; use `OR` between words to match ANY (e.g., `apple OR orange`), quote phrases for exact matches (e.g., `"project meeting"`), and append `*` for prefix matching (e.g., `debug*`).

## Inputs
- The analysis day in YYYYMMDD format (provided as context)
- Current date/time (provided as context)

## PHASE 0: Load the Day's Decisions

**CRITICAL FIRST STEP**: Before any analysis, gather all decision outputs from the day's activities:

```
sol call journal search "decision" -d $day_YYYYMMDD -t decisions
```

Decision-actions are now produced per-activity, so this search may return multiple result sets (one per activity that had decisions). Collect ALL decision-actions across all activities into a single working list. If no decision outputs exist for this day, stop and report this clearly.

## PHASE 1: Decision Selection

From all decision-actions gathered across activities, select the TWO most consequential based on:
- Number of people affected (breadth of impact)
- Criticality and irreversibility
- Time sensitivity and deadline pressures
- External stakeholder involvement
- Dependencies on critical systems or processes

## PHASE 2: Research Protocol (for each selected decision)

### Step 1: Immediate Context Discovery
Use these tools in sequence:

1. **Find the decision moment:**
   - `sol call journal search "decision keywords here" -d $day_YYYYMMDD -t audio -n 10`
   - Goal: Pinpoint exact time of decision (HH:MM:SS)

2. **Get full context:**
   - `sol call transcripts read --start HHMMSS --length 30 --full`
   - Goal: Extract 30 minutes of raw activity around decision time

### Step 2: Stakeholder & Dependency Mapping

1. **Identify all entities:**
   - `sol call journal search "entity names from decision" -d $day_YYYYMMDD`
   - Goal: Find all people, teams, projects mentioned

2. **Map meeting participants:**
   - `sol call journal events` or `sol call journal search "[keywords]" -d $day_YYYYMMDD -t event`
   - `sol call journal search "[keywords]" -t news -f work -d $day_YYYYMMDD` for public announcements
   - Goal: Identify who needs to know about this decision

### Step 3: Historical Precedent Mining (30-day lookback)

1. **Find similar past decisions:**
   - `sol call journal search "decision type AND key entities" -n 20`
   - Goal: Discover patterns in how similar decisions were handled

2. **Check commitment history:**
   - `sol call journal search "entity decision approve cancel" -t audio -n 15`
   - Goal: Identify typical follow-up patterns

### Step 4: Forward Impact Assessment (2-6 hours post-decision)

1. **Check for communications:**
   - `sol call journal search "[keywords]" -d $day_YYYYMMDD -t audio -n 10`
   - `sol call journal search "[keywords]" -t news -d $day_YYYYMMDD` for decision announcements
   - Goal: Find follow-up notifications or discussions

2. **Review meetings:**
   - `sol call journal search "[keywords]" -d $day_YYYYMMDD -t event`
   - Goal: See if decision was discussed

3. **Check messaging:**
   - `sol call journal search "message OR notification" -d $day_YYYYMMDD`
   - Goal: Verify notifications were sent

### Step 5: Gap Detection

1. **Search for problems:**
   - `sol call journal search "rollback revert issue problem" -d $day_YYYYMMDD`
   - Goal: Identify emerging issues

2. **Verify updates:**
   - `sol call journal search "document update change commit" -d $day_YYYYMMDD -t audio`
   - Goal: Confirm tracking artifacts were updated

OBLIGATION CATEGORIES (derive expectations from decision type + precedents; adapt to what the data shows)
- Communication: informing affected people/groups at an appropriate breadth and channel; acknowledging external stakeholders when present.
- Traceability: references to a tracking artifact or review process; linkage between the decision and the canonical record.
- Plan-of-Record: updates to sources-of-truth (documents, schedules, tickets) so others can rely on current state.
- Coordination: rescheduling, assignment, or next-step instructions when the decision creates work for others.
- Mitigation & Reversibility: contingency, rollback plan, or risk acknowledgement for sensitive/irreversible changes.

SCORING DIMENSIONS (qualitative, then calibrate a single confidence)
- Others Affected: low / medium / high (breadth of impact).
- Severity: low / medium / high (cost, sensitivity, coupling).
- Reversibility: short-window / recoverable / painful.

OUTPUT FORMAT (Markdown only; no JSON, no code blocks)

## Selected Decisions

Brief statement of which 2 decisions you selected and why they are the most consequential.

## Decision Dossiers

For each of your TWO selected decisions, create a detailed dossier:

### [#N] Decision: <short action summary>
**When:** HH:MM:SS–HH:MM:SS
**Type:** <decision type>
**Why it matters (one-liner):** <succinct statement of potential impact on others>

**Stakeholders & Dependencies**
- Direct stakeholders: <bulleted names/roles/groups>
- Indirect stakeholders: <bulleted names/roles/groups>
- Dependencies (artifacts/services/meetings/docs): <bulleted list>
- Breadth & sensitivity: <low/med/high breadth; flags like external_involved, high_centrality, time_sensitive>

**Context & Precedents (recent history)**
- Prior related decisions: <brief bullets with dates and how they resolved>
- Usual follow-ups observed in similar past cases: <brief bullets>
- Noted commitments that may apply: <brief bullets with who/what/when>

**Observed Follow-ups (forward window)**
- Communications observed: <bullets with times and audience scope>
- Traceability updates: <bullets with times and artifact names if present>
- Coordination/mitigation actions: <bullets, note absence if notable>

**Evidence Timeline (minimal, high-signal)**
- [HH:MM:SS] Audio: "<≤20 words>"
- [HH:MM:SS] Screen/OCR: <short phrase>
- [HH:MM:SS] Metadata: <concise note such as audience size/externality/environment>

**Possible Side-Effects & Gaps**
- <Gap/Side-effect #1>: <why this matters to others>; **obligation category:** <communication|traceability|plan|coordination|mitigation>
- <Gap/Side-effect #2>: <…>
(Include 2–5 items, only if supported by evidence or strong precedent.)

**Assessment & Recommendation**
- Others Affected: <low|medium|high> · Severity: <low|medium|high> · Reversibility: <short-window|recoverable|painful> · Confidence: <0.00–1.00>
- Suggested next step(s): <clear, minimal actions to close gaps or reduce side-effects>

## Insights & Blind Spots

Brief paragraph on:
- Key patterns observed across both decisions
- Data gaps or missing windows that limited analysis
- Systemic issues that may require attention

## EXECUTION NOTES

1. **Tool Usage**: Use the exact `sol call` commands provided. Do not invent command names or parameters.
2. **Evidence**: Only cite what you find via tools. Never fabricate entities or counts.
3. **Precision**: When uncertain, mark confidence levels clearly.
4. **Focus**: Analyze only TWO decisions deeply rather than all of them superficially.
5. **Actionability**: Every gap identified should have a specific remedy.
6. **Critical Gaps**: If you discover a critical gap that needs immediate attention, clearly highlight it in your output with a **CRITICAL GAP** heading.
