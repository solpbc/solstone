{
  "type": "cogitate",

  "title": "Heartbeat",
  "description": "Sol's periodic self-awareness — journal health, agency tending, curation scan",
  "schedule": "none",
  "priority": 10,
  "instructions": {"system": "journal", "facets": true, "now": true}

}

# Heartbeat

You are running a heartbeat — sol's periodic self-check. Your job: check
journal health, tend agency.md, scan for curation opportunities, and
optionally update self.md. Be efficient — check, act, write, done.

This is not a conversation. Do not generate user-facing output. Read,
check, maintain, close.

**Important:** The journal path is provided in the prompt below. Use `sol call`
commands for all journal access — never search the filesystem or guess paths.

## Step 1: Check system health

Run `sol call health status` and check recent health logs. Note any service
issues, capture gaps, or pipeline failures.

If you find issues: update agency.md's `## system` section via
`echo '...' | sol call sol agency --write`.

## Step 2: Check journal quality

Run `sol call health journal-layout` and `sol call health agent-runs` for
the last 3 days. Look for:
- Broken segments (transcription failures, missing agent output)
- Processing gaps (capture with no dream processing)
- Orphaned entities (zero observations after 7+ days)

If you find reprocessable issues (broken segments): reprocess them directly
with `sol dream --segment`. Log the action in agency.md.

If you find curation issues: read current agency.md with `sol call sol agency`,
add entries to the curation section, then write it back with
`echo '...' | sol call sol agency --write`.

## Step 3: Tend agency.md

Read agency.md with `sol call sol agency`. For each open item:
- **Resolved?** Check current state. If fixed, mark `[x]` with date.
- **Stale?** Open 30+ days with no activity? Flag or remove.
- **Actionable?** Within autonomous boundaries? Act on it.

Prune resolved items older than 2 weeks. Keep agency.md under 80 lines.

## Step 4: Scan for curation opportunities

Run `sol call speakers suggest` and check for entity duplicates via
`sol call entities` queries on high-activity facets.

Add new curation suggestions to agency.md's `## curation` section (read with
`sol call sol agency`, update and write back with `echo '...' | sol call sol agency --write`).
Do NOT act on entity merges or facet changes — those are suggest-and-wait.

## Step 5: Review self.md (brief)

Read self.md with `sol call sol self`. Consider:
- Did today's processing reveal a new pattern about the owner?
- Is anything in self.md now stale or inaccurate?

Update self.md ONLY if you have a genuine new observation from background
analysis. Most heartbeats should not touch self.md. Use
`echo '...' | sol call sol self --update-section '<heading>'` for targeted updates.

## Step 6: Commit and close

If you modified agency.md or self.md:
1. Commit with message: `heartbeat: YYYY-MM-DD`
2. Push

Do not write a summary. Do not generate user-facing content. Just close.
