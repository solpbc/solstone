---
name: speakers
description: >
  Manage speaker voiceprints in observed media. Detect and confirm the owner
  voice, bootstrap voiceprints, attribute segments, backfill attribution,
  discover unknown recurring speakers, identify clusters, merge name
  variants, and curate the speaker library over time. Use when the owner
  asks about voices in observed media, wants to identify speakers, or
  manage voice recognition.
  TRIGGER: speaker, voice, who was talking, identify speaker, owner voice,
  unknown voice, merge speakers, voice recognition, speaker curation,
  voiceprint, attribute segment, backfill, sol call speakers status,
  sol call speakers detect, sol call speakers confirm-owner,
  sol call speakers reject-owner, sol call speakers suggest,
  sol call speakers identify, sol call speakers merge-names,
  sol call speakers bootstrap, sol call speakers backfill,
  sol call speakers discover.
---

# Speakers CLI Skill

Manage speaker voiceprints and owner identification in observed media. Invoke via Bash: `sol call speakers <command> [args...]`.

**Writer polarity**: `bootstrap`, `resolve-names`, `attribute-segment`, `backfill`, and `seed-from-imports` preview by default. Pass `--commit` to persist. Without `--commit` they print a report and return — no data is modified.

Common pattern:

```bash
sol call speakers <command> [args...]
```

**Typical workflow**: status → detect → confirm-owner (auto-backfills) → suggest → identify/merge-names

## status

```bash
sol call speakers status [SECTION]
```

Speaker ID subsystem dashboard (embeddings, owner, speakers, clusters, imports, attribution). Returns JSON.

- `SECTION`: optional section name to filter (e.g., `owner`).

Behavior notes:

- Without a section, returns the full dashboard.
- Use `sol call speakers status owner` to check just the owner centroid state.

Examples:

```bash
sol call speakers status
sol call speakers status owner
```

## suggest

```bash
sol call speakers suggest [--limit N]
```

Actionable curation opportunities: unknown recurring voices, name variants, low-confidence segments. Returns JSON.

- `--limit`: max suggestions to return.

Behavior notes:

- Run after think processing completes, or when the owner is engaging with transcripts or observed media.
- Surface suggestions one at a time conversationally — don't stack them.

Example:

```bash
sol call speakers suggest --limit 5
```

## detect

```bash
sol call speakers detect [--json]
```

Run owner voice candidate detection. Returns the candidate plus sample segments.

Behavior notes:

- Only attempt when there are 50+ segments with embeddings across 3+ streams (check `owner-ready` first).
- If fewer segments exist, wait — don't mention speaker ID proactively until there's enough data.

Example:

```bash
sol call speakers detect
sol call speakers detect --json
```

## confirm-owner

```bash
sol call speakers confirm-owner [--backfill/--no-backfill] [--json]
```

Save the detected owner centroid after the owner confirms. By default, runs attribution backfill across all existing segments immediately after saving.

- `--backfill/--no-backfill`: default `--backfill`. Pass `--no-backfill` to defer the bulk pass.
- `--json`: emit full result as JSON.

Behavior notes:

- Only run after presenting the candidate to the owner and getting explicit confirmation.
- The default backfill run can process a lot of segments; if that matters (battery, time), pass `--no-backfill` and run `backfill` later.

Example:

```bash
sol call speakers confirm-owner
sol call speakers confirm-owner --no-backfill --json
```

## reject-owner

```bash
sol call speakers reject-owner
```

Discard the candidate if the owner says "that's not me." Enters a 14-day cooldown before detection will re-surface.

Behavior notes:

- Don't re-ask about a rejected owner candidate until cooldown clears.

## owner-ready

```bash
sol call speakers owner-ready
```

Report whether owner voice detection should be surfaced right now (enough embeddings, enough streams, not in cooldown).

Example:

```bash
sol call speakers owner-ready
```

## identify

```bash
sol call speakers identify <cluster_id> <name> [--entity-id ID]
```

Name an unknown speaker cluster after the owner provides the name.

- `cluster_id`: cluster identifier from `suggest` output (positional argument).
- `name`: speaker name to assign (positional argument).
- `--entity-id`: optional entity ID to link the speaker to an existing entity.

Example:

```bash
sol call speakers identify abc123 "Alicia Chen"
sol call speakers identify abc123 "Alicia Chen" --entity-id alicia_chen
```

## merge-names

```bash
sol call speakers merge-names <alias> <canonical>
```

Merge a name variant into the canonical entity.

- `alias`: the variant name to merge away (positional argument).
- `canonical`: the canonical name to keep (positional argument).

Behavior notes:

- Use when `suggest` surfaces a name variant (e.g., "Mitch" and "Mitch Baumgartner" sound identical).
- Ask the owner for confirmation before merging.

Example:

```bash
sol call speakers merge-names "Mitch" "Mitch Baumgartner"
```

## Library maintenance

The following commands maintain the voiceprint library over time. All of them are preview-by-default unless noted — pass `--commit` to persist.

### bootstrap

```bash
sol call speakers bootstrap [--commit] [--json]
```

Scan the full journal for single-speaker segments and save voiceprints for every non-owner speaker in those segments. Uses the owner centroid for subtraction.

### resolve-names

```bash
sol call speakers resolve-names [--commit] [--json]
```

Compare voiceprint centroids across all entities. Pairs with cosine similarity > 0.90 are flagged as the same person; unambiguous variants (short name is first word of full name) are auto-merged by adding the short name as an aka on the canonical entity. Ambiguous matches are reported but not auto-merged.

### attribute-segment

```bash
sol call speakers attribute-segment DAY STREAM SEGMENT [--commit] [--save/--no-save] [--accumulate/--no-accumulate] [--json]
```

Run speaker attribution (Layers 1-3) on a single segment.

- `--save/--no-save`: default `--save`. Write `speaker_labels.json`.
- `--accumulate/--no-accumulate`: default `--accumulate`. Run voiceprint accumulation.
- `--save` / `--accumulate` only take effect when `--commit` is passed.

### backfill

```bash
sol call speakers backfill [--commit] [--json]
```

Run attribution across every segment in the journal. Skips segments that already have labels.

### discover

```bash
sol call speakers discover [--json]
```

Find recurring unknown speakers across the journal. `suggest` is the curated surface; `discover` is the raw list — reach for it when debugging or doing manual triage.

### link-import

```bash
sol call speakers link-import NAME --entity-id ID
```

Link an imported-media participant name to a journal entity id.

### seed-from-imports

```bash
sol call speakers seed-from-imports [--commit] [--json]
```

Bootstrap voiceprints from imported-media tracks where the participant roster is known.

## Gotchas

- **Writers preview by default.** `bootstrap`, `resolve-names`, `attribute-segment`, `backfill`, and `seed-from-imports` do nothing without `--commit`. A successful-looking report is not a successful write. When testing, run without `--commit` first to see the plan; then re-run with `--commit` to persist.
- **`confirm-owner` backfills by default.** It runs attribution across the entire journal immediately after saving the centroid. Pass `--no-backfill` if you want to defer.
- **Owner rejection cools down for 14 days.** `reject-owner` blocks detection from re-surfacing until the cooldown clears.
- **`discover` is the raw feed; `suggest` is the curated one.** Don't surface `discover` output to the owner unfiltered — use `suggest`.

## Owner Detection

Check `speakers owner-ready` (or look at `speakers status owner`). If the owner centroid doesn't exist:

- If readiness passes (50+ segments with embeddings across 3+ streams): good time to try. Run `speakers detect`.
- If not: wait. Don't mention speaker ID proactively until there's enough data.

When you have a candidate, present it naturally: "I've been listening to your journal across your different devices and I think I can recognize your voice. Here are a few moments — does this sound right?" Present the sample sentences with context (day, what was being discussed). Don't play audio — show text and context.

- If the owner confirms: run `speakers confirm-owner` (backfill runs automatically).
- If the owner rejects: run `speakers reject-owner`. A 14-day cooldown starts; don't re-ask until it clears.

## Speaker Curation

Run `speakers suggest` after think processing completes, or when the owner is engaging with transcripts or observed media. Surface suggestions conversationally based on type:

- **Unknown recurring voice:** "I keep hearing a voice in your [day/context] observed media. They said things like '[sample text]'. Do you know who that is?" If the owner names them, run `speakers identify <cluster_id> <name>`.
- **Name variant:** "I noticed 'Mitch' and 'Mitch Baumgartner' sound identical in your observed media. Should I merge them?" If yes, run `speakers merge-names <alias> <canonical>`.
- **Low confidence review:** "There are a few speakers in this conversation I'm not sure about. Want to take a quick look?"

**Don't stack suggestions.** Surface one at a time. Wait for the owner to respond before presenting another. Speaker curation should feel like a natural aside, not a checklist.

## When NOT to Act

- Don't proactively surface speaker ID during unrelated conversations. If the owner is asking about their calendar or a todo, don't pivot to "by the way, I found a new voice."
- Don't surface low-confidence suggestions. If a cluster has only a few embeddings, wait for it to grow.
- Don't re-ask about a rejected owner candidate within the same week.
