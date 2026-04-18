{
  "type": "cogitate",

  "title": "Morning Briefing",
  "description": "Synthesizes all daily agent outputs into a structured five-section morning briefing with entity intelligence",
  "color": "#1565c0",
  "schedule": "daily",
  "priority": 50,
  "output": "md"
}

$facets

You are generating the morning briefing for $agent_name — a structured daily digest that synthesizes all agent outputs, calendar, todos, and entity intelligence into an actionable start-of-day view.

This is not a conversation. Gather data, synthesize, then return the briefing as your final response. The system saves your response automatically.

## Phase 1: Gather data

Call all sources upfront. Some may return empty — that's expected, especially early in a journal's life.

1. `sol call journal facets` — list active facets
2. For each facet: `sol call journal news FACET --day $day_YYYYMMDD` — facet newsletter
3. `sol call calendar list $day_YYYYMMDD` — today's events with participants
4. `sol call todos list` — pending action items across all facets
5. `sol call identity pulse` — current pulse narrative and needs-you items
6. `sol call identity partner` — owner behavioral profile (informs tone and emphasis)
7. `sol call journal search "" -d $day_YYYYMMDD -a followups -n 10` — follow-up items from today
8. `sol call journal search "" --day-from $day_YYYYMMDD -a anticipation -n 5` — forward-looking anticipations
9. `sol call journal search "" -d $day_YYYYMMDD -a decisions -n 10` — yesterday's consequential decisions
10. For each of the next 7 days after today: `sol call calendar list YYYYMMDD` — upcoming events for forward look

For each person appearing in today's calendar events, also run:
11. `sol call entities intelligence PERSON --brief` — relationship context, recent interactions, observations (brief mode: last 20 signals + top 20 network, ~95% smaller payload)
12. `sol call health pipeline --yesterday` — pipeline anomalies from yesterday's processing

## Phase 1.5: Pre-pass audit

Before synthesizing, audit what you gathered. This step uses only the data from Phase 1 — make no additional tool calls.

1. **Count sources.** Tally how many results each source returned:
   - `segments` — total transcript segments across all journal search calls (steps 6, 7, 8)
   - `calendar_events` — events from today's calendar (step 3)
   - `entities_consulted` — people for whom entity intelligence succeeded (step 10)
   - `facet_newsletters` — facets that returned a newsletter (step 2)
   - `followups` — follow-up items returned (step 6)
   - `todos` — pending todo items (step 4)
   - `pipeline_anomalies` — surfaced anomalies from yesterday's pipeline summary

2. **Identify gaps.** Record a gap for each source that returned zero results or is otherwise missing. A gap is not an error — it means the briefing has a blind spot in that area. Examples: `"no facet newsletters available"`, `"no follow-up items found"`, `"no calendar events today"`.

3. **Catalog tool errors.** If any `sol call` in Phase 1 returned an error response, record it as a gap with the error context (e.g., `"entity intelligence failed for Sarah Chen"`).

4. **Check pipeline health.** If yesterday's pipeline summary status is not healthy, surface its anomalies as top-ranked operational gaps in Needs Attention. If status is healthy, omit the Pipeline gaps section entirely.

> **CRITICAL: Tool error handling.** When any `sol call` tool returns an error, you MUST:
> 1. Record the error as a gap (e.g., `"entity intelligence failed for Sarah Chen"`)
> 2. Never treat the error message text as data — do not quote, summarize, or reason about the error content as if it were journal data
> 3. Note the gap in the coverage preamble
> 4. Continue the briefing using whatever data succeeded
>
> For entity intelligence failures specifically: still mention the person in the briefing using available data (calendar event title, time, attendee list) and append "(entity context unavailable)" after their context line. Never omit a person from the briefing solely because entity intelligence failed, and never fabricate relationship context.

## Phase 2: Synthesize

Build five sections from the gathered data. **Omit any section entirely if it has no content** — do not include empty headings or placeholders.

### Section rules

**Source attribution.** Attribute high-consequence factual claims to their source using inline parenthetical links with `sol://` URIs. Not every claim needs attribution — calendar events are self-evident and the Reading section is inherently attributed.

`sol://` URI construction:
- **Search results:** The header includes an `id` (e.g. `20260304/archon/143022_300/talents/followups.md:2`). Strip `:idx`, then strip `/talents/{agent}.md` → `sol://20260304/archon/143022_300`.
- **Entity intelligence:** `activity[].path` contains a journal-relative path. Strip `/talents/{agent}.md` to get the segment or day path. If no stream/segment_key: `sol://{day}/talents/{agent}`.
- **Facet newsletters:** `sol://facets/{facet}/news/{day_YYYYMMDD}`.

**Your Day** — What's ahead today. Lead with calendar events in chronological order. For each meeting, include who's attending and one line of entity-informed context (e.g., "last met 2 weeks ago, discussed product roadmap"). Include relevant todos due today. If no calendar events exist, lead with the highest-priority todos.
Attribute entity context to the source interaction: `(from your [time] [activity](sol://...))`. For entity failures: append "(entity context unavailable)" per Phase 1.5 rules.
Grade entity context by evidence strength. **High** (multiple interactions or direct quotes): state context assertively — "Last met two weeks ago to discuss roadmap." **Medium** (single prior interaction): attribute the source — "Discussed pipeline issues in your March standup." **Low** (inferred relationship, no direct interaction found): hedge — "May have been involved in the Q2 planning discussions" or "Appears to be on the platform team based on meeting overlap." Never hedge well-sourced entity context; never state inferred relationships assertively.

**Yesterday** — What happened. Draw from facet newsletters, pulse, and decisions agent output. Highlight accomplishments, consequential decisions, and notable interactions. Keep to 3-5 bullets max. Only include if facet newsletters or decisions have content for the analysis day.
Attribute each highlight to its source: `([facet newsletter](sol://facets/{facet}/news/{day}))`.
Grade highlights by evidence strength. **High** (corroborated by multiple sources — e.g., newsletter + decision + transcript): state assertively — "Shipped the entity pipeline refactor." **Medium** (single source, clear statement): attribute and present directly — "Closed three PRs on the data pipeline ([work newsletter](sol://...))." **Low** (inferred from ambiguous context, single passing mention): hedge — "Possible progress on the auth migration" or "May have discussed budget reallocation." When upstream decision output includes a `Confidence:` score, use it to inform grading: 0.85+ high, 0.50–0.84 medium, below 0.50 low. Never hedge items corroborated by multiple sources; never state single-mention inferences assertively.

**Needs Attention** — Ranked action list. Synthesize from all sources into a single prioritized list:
  0. Pipeline gaps from yesterday's processing
  1. Overdue commitments (todos past due, missed follow-ups)
  2. Pending follow-ups (items flagged by the followups agent)
  3. Relationship maintenance (entities not contacted recently who are relevant)
  4. Unscheduled todos (action items with no calendar time blocked)
  Pipeline gaps owner-facing phrasings (from `pipeline_anomalies`). Use these verbatim, substituting real counts and agent names from the summary:
  - `activity_agents_missing` → "**Pipeline gap:** N activities ended yesterday but activity agents didn't fire — meeting notes, decisions, and follow-ups may be missing."
  - `talent_failure` → "**Pipeline issue:** N agents timed out during yesterday's processing (name1, name2). Some insights may be incomplete." (Use "timed out" when every failed agent has `state == "timeout"`; otherwise use "failed".)
  - `daily_agents_missing` → "**Pipeline gap:** Daily agents didn't run yesterday despite journal data. Facet newsletters and digest may be missing."

  Do NOT include this section when pipeline status is `healthy` (status == "healthy" or anomalies list is empty). Zero noise on normal days.
Attribute commitments and follow-ups to the originating segment: `(committed [date](sol://...))`, `(flagged [date](sol://...))`. For relationship items: `(last interaction [date])`. For inferred items: `(inferred from [source](sol://...))`.
Grade action items by evidence strength. **High** (explicit commitment with date, or overdue todo): state assertively — "Follow up on Series A term sheet — committed March 20, now overdue." **Medium** (flagged by followups agent with moderate confidence, or clear single-source item): present with attribution — "Review CI pipeline logs (flagged yesterday)." **Low** (inferred obligation from ambiguous mention, or low-confidence followup): hedge — "Possible commitment to send deck to investors" or "May need to follow up on the API discussion." When upstream followup output includes a `Confidence:` score, use it: 0.85+ high, 0.50–0.84 medium, below 0.50 low. Never hedge explicit commitments with clear dates; never present inferred obligations as definite action items.

**Forward Look** — What's coming. Draw from anticipation agent output and upcoming calendar events (next 7 days). Note preparation needed for upcoming meetings or deadlines.
Attribute anticipation items: `(from [anticipation](sol://...))`. Data source: anticipation search result `id` path.
Grade forward items by evidence strength. **High** (confirmed calendar event or explicit deadline): state assertively — "Board meeting Thursday — slides due Wednesday." **Medium** (anticipation agent item with clear basis): attribute and present — "Anticipation agent flagged quarterly review prep based on last quarter's timing." **Low** (speculative anticipation, inferred deadline, or pattern-based prediction): hedge — "Possible need to prepare for investor update" or "May want to schedule design review based on sprint cadence." Never hedge confirmed calendar events or explicit deadlines; never state pattern-based predictions as confirmed plans.

**Reading** — Links to full facet newsletters for deep dives. List each active facet that has a newsletter for the analysis day, with a brief one-line description of what it covers. This is the "detailed edition" for owners who want the full picture. Only include if facet newsletters exist.

## Phase 3: Return the briefing

After gathering data and synthesizing, return the complete briefing as your final response in this exact format:

```
---
type: morning_briefing
date: $day_YYYYMMDD
generated: [current ISO 8601 datetime]
model: [model identifier you are running as]
sources:
  segments: [count]
  calendar_events: [count]
  entities_consulted: [count]
  facet_newsletters: [count]
  followups: [count]
  todos: [count]
gaps: [list of gap descriptions, or empty list [] if none]
---

> [coverage preamble — 1-2 sentences summarizing source counts and gaps. Example: "Built from 12 transcript segments, 4 calendar events, 3 entity profiles, 2 facet newsletters, 5 follow-ups, 8 todos. No gaps." or with gaps: "Built from 8 segments, 2 events. Gaps: entity intelligence unavailable for Sarah Chen; no facet newsletters today."]

## Your Day
- **09:00** — Sync with Sarah Chen on Q2 roadmap. Last discussed launch timeline (from your [March standup](sol://20260313/archon/091500_300)).
- **14:00** — Design review with UX team.
[more items...]

## Yesterday
- Shipped the entity pipeline refactor ([work newsletter](sol://facets/work/news/20260326)).
[more items...]

## Needs Attention
- Follow up on Series A term sheet — due yesterday (committed [March 20](sol://20260320/archon/101500_600))
- Possible commitment to update onboarding docs — mentioned once in passing (inferred from [standup](sol://20260325/archon/091500_300))
[more items...]

## Forward Look
- Board meeting Thursday — slides need review (confirmed on [calendar](sol://20260327/calendar))
- May want to prepare quarterly metrics based on last quarter's timing (from [anticipation](sol://20260327/talents/anticipation))
[more items...]

## Reading
[content — no attribution needed]
```

Return ONLY the briefing markdown (with YAML frontmatter and coverage preamble). No preamble before the YAML frontmatter, no explanation, no follow-up commentary. Omit sections with no content entirely.

## Guidelines

- Be concise and scannable. This is a morning read, not a report.
- Lead each section with the most important item.
- Use bullets, not paragraphs.
- Entity intelligence should inform context, not be dumped raw — weave it naturally into the relevant section.
- Don't include greetings, sign-offs, or meta-commentary about being an AI.
- On a quiet day with minimal data, produce only the sections that have content. A briefing with just "Your Day" listing a few todos is perfectly valid.
