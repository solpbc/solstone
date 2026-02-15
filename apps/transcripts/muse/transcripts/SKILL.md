---
name: transcripts
description: Browse and read transcript content using sol call transcripts commands. Inspect audio recordings, screen captures, and agent summaries. Check recording coverage, list segments, read transcript text with source filtering, or review monthly statistics.
---

# Transcripts CLI Skill

Use these commands to inspect transcript availability and content from the terminal.

**Environment defaults**: When `SOL_DAY` is set, commands that take a DAY argument will use it automatically. `SOL_SEGMENT` and `SOL_STREAM` provide defaults for `--segment` and `--stream` options.

Common pattern:

```bash
sol call transcripts <command> [args...]
```

**Typical workflow**: `scan` a day for recording windows → `segments` to get segment keys → `read` to retrieve transcript text.

## scan

```bash
sol call transcripts scan [DAY]
```

Show audio and screen coverage ranges.

- `DAY`: day in `YYYYMMDD` (default: `SOL_DAY` env).

Use this first to confirm what recording windows exist before running detailed reads.

Example:

```bash
sol call transcripts scan 20260115
```

## segments

```bash
sol call transcripts segments [DAY]
```

List recording segments and their source types.

- `DAY`: day in `YYYYMMDD` (default: `SOL_DAY` env).

Behavior notes:

- Segment keys use `HHMMSS_LEN` format (for example `091500_300`).
- Each segment shows start/end time and available types (`audio`, `screen`, or both).

Use this when you want a specific segment key for targeted reads.

Example:

```bash
sol call transcripts segments 20260115
```

## read

```bash
sol call transcripts read [DAY] [--start HHMMSS --length MINUTES] [--segment KEY] [--stream NAME] [--full] [--raw] [--audio] [--screen] [--agents]
```

Read transcript content for a day, time range, or segment.

- `DAY`: day in `YYYYMMDD` (default: `SOL_DAY` env).

Read modes (mutually exclusive):

1. `--start HHMMSS --length MINUTES`: range mode. Both flags are required together; end time is auto-computed.
2. `--segment KEY`: segment mode using key from `segments`.
3. No time flags: whole-day mode.

Source flags:

- `--full`: audio + screen + agents.
- `--raw`: audio + screen only (no agents).
- `--audio`: audio only.
- `--screen`: screen only.
- `--agents`: agent outputs only.
- Default with no source flags: audio + agents (no screen).

Rules:

- Do not combine `--segment` with `--start/--length`.
- Do not combine `--full` or `--raw` with individual source flags.

Source type meanings:

- **Audio**: spoken-word transcripts from microphone/system audio.
- **Screen**: frame-level screen activity transcription.
- **Agents**: AI-generated summaries and insights.

Examples:

```bash
sol call transcripts read 20260115
sol call transcripts read 20260115 --start 090000 --length 30 --raw
sol call transcripts read 20260115 --segment 091500_300 --full
sol call transcripts read 20260115 --audio
```

## stats

```bash
sol call transcripts stats MONTH
```

Show daily transcript coverage counts for a month.

- `MONTH`: required month in `YYYYMM`.

Use this to understand recording density and coverage patterns across a month.

Example:

```bash
sol call transcripts stats 202601
```
