---
name: activities
description: >
  Manage completed activity records organized by facet and day. List, inspect,
  create, update, mute, and unmute activity spans, including synthetic records
  created from the CLI. Use when the owner mentions activities, work sessions,
  completed spans, muted activity records, or wants to review or edit an
  activity entry.
  TRIGGER: activity, activities, work session, completed span, mute activity,
  unmute activity, activity record, meeting attendees, participation,
  sol call activities list, sol call activities create,
  sol call activities update, sol call activities mute,
  sol call activities unmute, sol call activities get.
---

# Activities CLI Skill

Manage completed activity records. Invoke via Bash: `sol call activities <command> [args...]`.

**Environment defaults**: When `SOL_DAY` is set, commands that take `DAY` use it automatically. Same for `SOL_FACET` where `FACET` is accepted.

Common pattern:

```bash
sol call activities <command> [args...]
```

## list

```bash
sol call activities list [-d DAY | --from DAY --to DAY] [-f FACET] [-a ACTIVITY] [--entity ENTITY] [--source SOURCE] [--all] [--json]
```

List activity records for one day or an inclusive day range.

- `-d, --day`: single day in `YYYYMMDD` (default: `SOL_DAY` env).
- `--from`, `--to`: inclusive day range in `YYYYMMDD`.
- `-f, --facet`: optional facet filter. Omit to include all facets.
- `-a, --activity`: optional activity-type filter.
- `--entity`: optional active-entity filter.
- `--source`: optional record-source filter (`anticipated`, `user`, or `cogitate`). Omit to include all sources.
- `--all`: include hidden activity records.
- `--json`: emit raw JSON instead of formatted text.

Examples:

```bash
sol call activities list -d 20260115 -f work
sol call activities list --from 20260101 --to 20260107 -f work --activity coding
sol call activities list -d 20260115 --entity "Alicia Chen" --json
```

## get

```bash
sol call activities get SPAN_ID [-f FACET] [-d DAY] [--json]
```

Fetch one activity record by id.

- `SPAN_ID`: activity record id.
- `-f, --facet`: facet name (default: `SOL_FACET` env).
- `-d, --day`: day in `YYYYMMDD` (default: `SOL_DAY` env).
- `--json`: emit the raw record.

Example:

```bash
sol call activities get coding_090000_300 -f work -d 20260115
```

## create

```bash
sol call activities create [-f FACET] [-d DAY] [--since-segment SEGMENT] [--source user|cogitate] [--json]
```

Create a new activity record from a JSON object on stdin.

- `-f, --facet`: facet name (default: `SOL_FACET` env).
- `-d, --day`: day in `YYYYMMDD` (default: `SOL_DAY` env).
- `--since-segment`: optional segment key to anchor a real activity span.
- `--source`: record source label (`user` by default).
- `--json`: emit the created record.

Stdin JSON requirements:

- Required: `title`, `activity`
- Optional: `description`, `details`, `participation`

`participation` is an array of participant objects. Each entry must include:

- `name`: non-empty string.
- `role`: `"attendee"` or `"mentioned"`.
- `source`: one of `"voice"`, `"speaker_label"`, `"transcript"`, `"screen"`, `"other"`.
- `confidence`: number (typically 0.0-1.0).
- `context`: string describing where the detection came from.

Names are resolved against journal entities after validation; any caller-supplied `entity_id` is ignored.

Examples:

```bash
echo '{"title":"Deep work","activity":"coding"}' | sol call activities create -f work
echo '{"title":"Session review","activity":"coding","details":"Retrospective notes"}' | sol call activities create -f work --since-segment 090000_300 --source cogitate --json
echo '{
  "title":"Sync with Alicia",
  "activity":"meeting",
  "participation":[
    {"name":"Alicia Chen","role":"attendee","source":"voice","confidence":0.95,"context":"voice detected start of segment"},
    {"name":"You","role":"attendee","source":"screen","confidence":1.0,"context":"screen share"}
  ]
}' | sol call activities create -f work
```

## update

```bash
sol call activities update SPAN_ID [-f FACET] [-d DAY] [--note TEXT] [--json]
```

Update one activity record from a JSON patch on stdin.

- `SPAN_ID`: activity record id.
- `-f, --facet`: facet name (default: `SOL_FACET` env).
- `-d, --day`: day in `YYYYMMDD` (default: `SOL_DAY` env).
- `--note`: optional edit note stored in the record history.
- `--json`: emit the updated record.

Stdin JSON may include only `title`, `description`, and `details`.

Example:

```bash
echo '{"title":"Focused coding","details":"Closed out CLI edge cases"}' | sol call activities update coding_090000_300 -f work --note "tightened summary"
```

## mute

```bash
sol call activities mute SPAN_ID [-f FACET] [-d DAY] [--reason TEXT] [--json]
```

Hide an activity record without deleting it.

- `SPAN_ID`: activity record id.
- `-f, --facet`: facet name (default: `SOL_FACET` env).
- `-d, --day`: day in `YYYYMMDD` (default: `SOL_DAY` env).
- `--reason`: optional note stored in the edit history.
- `--json`: emit the updated record.

Example:

```bash
sol call activities mute coding_090000_300 -f work --reason "synthetic duplicate"
```

## unmute

```bash
sol call activities unmute SPAN_ID [-f FACET] [-d DAY] [--reason TEXT] [--json]
```

Restore a muted activity record.

- `SPAN_ID`: activity record id.
- `-f, --facet`: facet name (default: `SOL_FACET` env).
- `-d, --day`: day in `YYYYMMDD` (default: `SOL_DAY` env).
- `--reason`: optional note stored in the edit history.
- `--json`: emit the updated record.

Example:

```bash
sol call activities unmute coding_090000_300 -f work --reason "keep owner-authored record"
```
