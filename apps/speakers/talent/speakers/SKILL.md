---
name: speakers
description: >
  Manage the speaker identification subsystem. Check speaker status, detect
  the owner voice, identify unknown speakers, merge name variants, and curate
  the speaker library over time. Use when the owner asks about voices in
  recordings, wants to identify speakers, or manage voice recognition.
  TRIGGER: speaker, voice, who was talking, identify speaker, owner voice,
  unknown voice, merge speakers, voice recognition, speaker curation.
---

# Speakers CLI Skill

Use these commands to manage the speaker identification subsystem from the terminal.

**Environment defaults**: When `SOL_DAY` is set, commands that take a DAY argument will use it automatically. Same for `SOL_FACET` where FACET is required.

Common pattern:

```bash
sol call speakers <command> [args...]
```

**Typical workflow**: status → status owner → suggest → identify/merge-names

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

## owner detect

```bash
sol call speakers owner detect [--force]
```

Run owner voice detection. Returns candidate with samples.

- `--force`: re-run even if a candidate already exists.

Behavior notes:

- Only attempt when there are 50+ segments with embeddings across 3+ streams.
- If fewer segments exist, wait — don't mention speaker ID proactively until there's enough data.

Example:

```bash
sol call speakers owner detect
sol call speakers owner detect --force
```

## owner confirm

```bash
sol call speakers owner confirm
```

Save detected owner centroid after owner confirms.

Behavior notes:

- Only run after presenting the candidate to the owner and receiving explicit confirmation.
- After confirmation, the system can start identifying other voices.

## owner reject

```bash
sol call speakers owner reject
```

Discard candidate if owner says "that's not me."

Behavior notes:

- Wait for more data before trying detection again.
- Don't re-ask about a rejected owner candidate within the same week.

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

## Owner Detection

Check `speakers status owner`. If the owner centroid doesn't exist:

- If there are 50+ segments with embeddings across 3+ streams: good time to try. Run `speakers owner detect`.
- If fewer: wait. Don't mention speaker ID proactively until there's enough data.

When you have a candidate, present it naturally: "I've been listening to your journal across your different devices and I think I can recognize your voice. Here are a few moments — does this sound right?" Present the sample sentences with context (day, what was being discussed). Don't play audio — show text and context.

- If the owner confirms: run `speakers owner confirm`.
- If the owner rejects: run `speakers owner reject`. Wait for more data before trying again.

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
