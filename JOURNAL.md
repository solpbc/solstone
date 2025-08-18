# Sunstone Journal Guide

This document describes the layout of a **journal** directory where all audio, screen and analysis artifacts are stored. Each dated `YYYYMMDD` folder is referred to as a **day**.

## Top level files

- `entities.md` – top list of entities gathered across days. Used by several tools.
- `entity_review.log` – operations performed in the web UI are appended here.
- `task_log.txt` – optional log of utility runs in `[epoch]\tmessage` format.
- `domains/` – domain-specific organization folders described below.
- `inbox/` – asynchronous messaging system for agent communications described below.
- `YYYYMMDD/` – individual day folders described below.

## Domain folders

The `domains/` directory provides a way to organize journal content by scope or focus area. Each domain represents a cohesive grouping of related activities, projects, or areas of interest.

### Domain structure

Each domain is organized as `domains/<domain>/` where `<domain>` is a descriptive short unique name. Each domain folder contains:

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

### Domain Entities

The `entities.md` file follows the same format as the top-level entities file but contains only entities relevant to this specific domain. This allows for more targeted entity tracking within focused areas of work.

### Domain Matters

Matters represent specific scoped topics, sub-projects, or focused areas of work within a domain. Each matter is stored as a directory within the domain using an incrementing integer ID system.

#### Matter Folders Organization

Each active matter is organized as `domains/<domain>/matter_X/` where X is an incrementing integer that serves as the matter ID. Archived matters get moved to the `<domain>/archived/matter_X/` location.

The incrementing integer format ensures unique identification and allows for easy ordering by creation sequence, but both active and archived matters must be checked to get the next integer when adding a new matter.

#### Matter directory:

- `matter.json` – matter metadata including title, description, and other properties
- `activity_log.jsonl` – chronological log of matter-related activities in JSON Lines format
- `attachments/` – directory containing files and their metadata
- `objective_<name>/` – individual objective directories containing OBJECTIVE.md and optional OUTCOME.md
- the directory created/modified timestamps are the source when the matter was first created or last modified
- `.git` - each matter's files and changes are managed via git

#### Matter metadata format

The `matter.json` file contains the matter's core information:

```json
{
  "title": "API Performance Optimization",
  "description": "Investigating and implementing improvements to reduce API response times",
  "status": "active",
  "priority": "high",
}
```

Required fields:
- `title` – concise name for the matter
- `description` – detailed explanation of the matter's scope and purpose

Optional fields:
- `status` – current state (e.g., "active", "archived")
- `priority` – importance level (e.g., "low", "medium", "high")

#### Matter activity log

The `activity_log.jsonl` file maintains a chronological record of all matter-related activities in JSON Lines format. Each line is a JSON object with the following fields plus any other fields relevant to that type:

```json
{
  "timestamp": 1755450767962,  // epoch ms
  "type": "update",                      // Event type
  "message": "Updated matter status",    // Human-readable description of the activity

}
```

#### Matter attachments

The `attachments/` directory contains files relevant to the matter along with their metadata. Each attachment consists of:

- `<filename>.<extension>` – the actual file (document, image, code, etc.)
- `<filename>.<extension>.json` – metadata describing the attachment
- The .json file created/modified timestamps represent those values for the attachment relative to the matter

The metadata file format:

```json
{
  "title": "API Documentation",
  "description": "Complete API reference documentation for the performance optimization work",
  "mime_type": "application/pdf",
}
```

Required fields:
- `title` – human-readable name for the attachment
- `description` – detailed explanation of the attachment's content and relevance
- `mime_type` – MIME type of the attached file

#### Matter objectives

Objectives are specific goals and sub-tasks related to the matter. Each objective is organized as `objective_<name>/` where `<name>` is a unique alphanumeric identifier with underscores for separation. Each objective directory contains:

- `OBJECTIVE.md` – markdown file describing the objective, its requirements, and approach
- `OUTCOME.md` – markdown file describing the results and completion details (present only when objective is completed)

The objective name serves as the unique identifier and should be descriptive yet concise (e.g., `objective_ui_implementation`, `objective_database_optimization`, `objective_api_testing`).

Example objective structure:

```
objective_performance_optimization/
├── OBJECTIVE.md
└── OUTCOME.md    # Only present when completed
```

The presence of `OUTCOME.md` indicates objective completion. Directory timestamps (created/modified) provide temporal tracking without requiring separate metadata files.

## Inbox

The `inbox/` directory provides an asynchronous messaging system where agents and automated processes can leave messages for user review. Messages are organized in active and archived subdirectories.

### Inbox structure

The inbox is organized as follows:

- `inbox/active/` – directory containing unread and active messages
- `inbox/archived/` – directory containing archived messages
- `inbox/activity_log.jsonl` – chronological log of inbox activities

### Message files

Each message is stored as a single JSON file named `msg_<timestamp>.json` where `<timestamp>` is epoch milliseconds (e.g., `msg_1755450767962.json`).

Message files can exist in either:
- `inbox/active/msg_<timestamp>.json` – for active/unread messages
- `inbox/archived/msg_<timestamp>.json` – for archived messages

### Message format

Each message JSON file contains:

```json
{
  "id": "msg_1755450767962",
  "timestamp": 1755450767962,
  "from": {
    "type": "agent",
    "id": "research_agent"
  },
  "body": "Message content in plain text or markdown format",
  "status": "unread",
  "context": {
    "domain": "ml_research",
    "matter": "matter_1",
    "day": "20250117"
  }
}
```

Required fields:
- `id` – unique message identifier matching the filename
- `timestamp` – epoch milliseconds when the message was created
- `from` – sender information with `type` (agent/system/domain) and `id`
- `body` – message content as text or markdown
- `status` – message state (unread/read/archived)

Optional fields:
- `context` – reference to related journal entities (domain, matter, day)

### Inbox activity log

The `inbox/activity_log.jsonl` file tracks all inbox operations in JSON Lines format:

```json
{"timestamp": 1755450767962, "action": "received", "message_id": "msg_1755450767962", "from": "research_agent"}
{"timestamp": 1755450768000, "action": "read", "message_id": "msg_1755450767962"}
{"timestamp": 1755450769000, "action": "archived", "message_id": "msg_1755450767962"}
```

Common actions include:
- `received` – new message created
- `read` – message marked as read
- `archived` – message moved to archive
- `deleted` – message removed

## Day folder contents

Audio capture tools write FLAC files and transcripts:

- `HHMMSS_raw.flac` – mixed audio file, moved to `heard/` after transcription.
- `heard/HHMMSS_raw.flac` – audio files moved here after processing.
- `HHMMSS_audio.json` – transcript JSON produced by transcription.

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

