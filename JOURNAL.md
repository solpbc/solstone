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
- `<timestamp>/` – individual matter directories for domain-specific sub-projects and focused topics.

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

### Domain matters

Matters represent specific scoped topics, sub-projects, or focused areas of work within a domain. Each matter is stored as a directory within the domain using a timestamp-based ID system.

#### Matter file structure

Each matter is organized as `domains/<domain>/<timestamp>/` where the timestamp serves as the matter ID. Within each matter directory:

- `matter.json` – matter metadata including title, description, and other properties
- `matter.jsonl` – chronological log of matter-related activities in JSON Lines format
- `attachments/` – directory containing files and their metadata
- `objectives/` – directory containing objectives with metadata and activity logs

The timestamp follows the same format used for agents and tasks, ensuring unique identification and chronological ordering.

#### Matter metadata format

The `matter.json` file contains the matter's core information:

```json
{
  "title": "API Performance Optimization",
  "description": "Investigating and implementing improvements to reduce API response times",
  "created": "2025-01-15T10:30:00Z",
  "status": "active",
  "priority": "high",
  "tags": ["performance", "backend", "optimization"]
}
```

Required fields:
- `title` – concise name for the matter
- `description` – detailed explanation of the matter's scope and purpose

Optional fields:
- `created` – ISO 8601 timestamp of matter creation
- `status` – current state (e.g., "active", "completed", "paused", "cancelled")
- `priority` – importance level (e.g., "low", "medium", "high", "critical")
- `tags` – array of relevant keywords for categorization and search

#### Matter activity log

The `matter.jsonl` file maintains a chronological record of all matter-related activities in JSON Lines format. The specific format and fields will be defined separately.

#### Matter attachments

The `attachments/` directory contains files relevant to the matter along with their metadata. Each attachment consists of:

- `<filename>.<extension>` – the actual file (document, image, code, etc.)
- `<filename>.json` – metadata describing the attachment

The metadata file format:

```json
{
  "title": "API Documentation",
  "description": "Complete API reference documentation for the performance optimization work",
  "created": "2025-01-15T14:30:00Z",
  "modified": "2025-01-15T14:30:00Z",
  "size": 2048576,
  "mime_type": "application/pdf",
  "tags": ["documentation", "api", "reference"]
}
```

Required fields:
- `title` – human-readable name for the attachment
- `description` – detailed explanation of the attachment's content and relevance

Optional fields:
- `created` – ISO 8601 timestamp when attachment was added
- `modified` – ISO 8601 timestamp when attachment was last modified
- `size` – file size in bytes
- `mime_type` – MIME type of the attached file
- `tags` – array of keywords for categorization

#### Matter objectives

The `objectives/` directory contains specific goals and sub-tasks related to the matter. Each objective is organized as `<timestamp>/` containing:

- `<timestamp>.json` – objective metadata
- `<timestamp>.jsonl` – chronological log of objective-related activities

The objective metadata format:

```json
{
  "title": "Reduce API response time by 50%",
  "description": "Implement caching and database query optimization to achieve sub-200ms response times for all endpoints",
  "created": "2025-01-15T10:45:00Z",
  "status": "in_progress",
  "priority": "high",
  "target_date": "2025-02-01T00:00:00Z",
  "completion_criteria": [
    "All API endpoints respond in under 200ms",
    "Database query optimization completed",
    "Redis caching layer implemented"
  ]
}
```

Required fields:
- `title` – concise objective statement
- `description` – detailed explanation of what needs to be accomplished

Optional fields:
- `created` – ISO 8601 timestamp of objective creation
- `status` – current state (e.g., "pending", "in_progress", "completed", "blocked")
- `priority` – importance level (e.g., "low", "medium", "high", "critical")
- `target_date` – ISO 8601 timestamp for when objective should be completed
- `completion_criteria` – array of specific measurable outcomes that define success

This structure allows matters to serve as comprehensive project management units with full activity history, relevant files, and structured goal tracking.

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

