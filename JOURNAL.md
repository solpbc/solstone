# Sunstone Journal Guide

This document describes the layout of a **journal** directory where all audio, screen and analysis artifacts are stored. Each dated `YYYYMMDD` folder is referred to as a **day**.

## Top level files

- `entities.md` – top list of entities gathered across days. Used by several tools.
- `entity_review.log` – operations performed in the web UI are appended here.
- `task_log.txt` – optional log of utility runs in `[epoch]\tmessage` format.
- `domains/` – domain-specific organization folders described below.
- `YYYYMMDD/` – individual day folders described below.

## Domain folders

The `domains/` directory provides a way to organize journal content by scope or focus area. Each domain represents a cohesive grouping of related activities, projects, or areas of interest.

### Domain structure

Each domain is organized as `domains/<domain>/` where `<domain>` is a descriptive name. Each domain folder contains:

- `domain.json` – metadata file with domain title and description.
- `entities.md` – entities specific to this domain.
- `matters/` – domain-specific sub-projects (to be defined).

### Domain metadata

The `domain.json` file contains basic information about the domain:

```json
{
  "title": "Machine Learning Research",
  "description": "AI/ML research projects, experiments, and related activities",
  "color": "#4f46e5",
  "emoji": "🧠"
}
```

Optional fields:
- `color` – hex color code for the domain card background in the web UI
- `emoji` – emoji icon displayed in the top-left of the domain card

### Domain entities

The `entities.md` file follows the same format as the top-level entities file but contains only entities relevant to this specific domain. This allows for more targeted entity tracking within focused areas of work.

## Day folder contents

Audio capture tools write FLAC files and transcripts:

- `HHMMSS_raw.flac` – temporary mixed audio; removed after processing.
- `heard/HHMMSS_audio.flac` – final clipped audio segment moved after transcription.
- `HHMMSS_audio.json` – transcript JSON produced by `gemini-transcribe`.

Screen capture utilities produce per-source diff files. After `screen-describe`
moves the image and its bounding box into a `seen/` directory, the Gemini
description remains in the day folder:

- `HHMMSS_<source>_N_diff.png` – screenshot of the changed region, moved to
  `seen/` once processed, contains a box_2d metadata field for the changed area.
- `HHMMSS_<source>_N_diff.json` – Gemini description of the diff.

`reduce-screen` summarises these diffs into five‑minute chunks:

- `HHMMSS_screen.md` – Markdown summary for that interval.

- Post‑processing commands may generate additional analysis files, for example:

- `topics/flow.md` – high level summary of the day.
- `topics/knowledge_graph.md` – knowledge graph / network summary.
- `topics/meetings.md` – meeting list used by the calendar web UI.
- `entities.md` – daily entity rollup produced by `entity-roll`.
- `task_log.txt` – log of tasks for that day in `[epoch]\tmessage` format.

### Crumbs

Most generated files are accompanied by a `.crumb` file capturing dependencies and model information. See `CRUMBS.md` for the format. Example: `20250610/topics/flow.md.crumb`.

## Occurrence JSON

Several `think/topics` prompts extract time based events from the day's
transcripts—meetings, messages, follow ups, file activity and more.  To index
these consistently the results can be normalised into an **occurrence** container
stored as `occurrences.json` inside each day folder.

```json
{
  "day": "YYYYMMDD",
  "occurrences": [
    {
      "type": "meeting",
      "source": "topics/meetings.md",
      "start": "09:00:00",
      "end": "09:30:00",
      "title": "Team stand-up",
      "summary": "Status update with the engineering team",
      "work": true,
      "participants": ["Jeremie Miller", "Alice", "Bob"],
      "details": {...}
    }
  ]
}
```

### Common fields

- **type** – the kind of occurrence such as `meeting`, `message`, `file`, `followup`, `documentation`, `research`, `media`, etc.
- **source** - the file the occurence was extracted from.
- **start** and **end** – HH:MM:SS timestamps containing the occurence.
- **title** and **summary** – short text for display and search.
- **work** – boolean, work vs. personal classification when known.
- **participants** – optional list of people or entities involved.
- **details** – free-form string of other occurrence specific information.

Each topic analysis can map its findings into this structure allowing the
indexer to collect and search occurrences across all days.

