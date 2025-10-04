# Sunstone Journal Guide

This document describes the layout of a **journal** directory where all audio, screen and analysis artifacts are stored. Each dated `YYYYMMDD` folder is referred to as a **day**.

## Top level files

- `entities.md` – top list of entities gathered across days. Used by several tools.
- `entity_review.log` – operations performed in the web UI are appended here.
- `task_log.txt` – optional log of utility runs in `[epoch]\tmessage` format.
- `config/journal.json` – user configuration for the journal (optional, see below).
- `domains/` – domain-specific organization folders described below.
- `inbox/` – asynchronous messaging system for agent communications described below.
- `YYYYMMDD/` – individual day folders described below.

## User configuration

The optional `config/journal.json` file allows customization of journal processing and presentation based on user preferences. This file should be created at the journal root and contains personal settings that affect how the system processes and interprets journal data.

### Identity configuration

The `identity` block contains information about the journal owner that helps tools correctly identify the user in transcripts, meetings, and other captured content:

```json
{
  "identity": {
    "name": "Jeremie Miller",
    "preferred": "Jer",
    "pronouns": {
      "subject": "he",
      "object": "him",
      "possessive": "his",
      "reflexive": "himself"
    },
    "aliases": ["Jer", "jeremie"],
    "email_addresses": ["jer@example.com"],
    "timezone": "America/Los_Angeles",
    "entity": "Jeremie Miller (Jer)"
  }
}
```

Fields:
- `name` (string) – Full legal or formal name of the journal owner
- `preferred` (string) – Preferred name or nickname to be used when addressing the user
- `pronouns` (object) – Structured pronoun set for template usage with fields:
  - `subject` – Subject pronoun (e.g., "he", "she", "they")
  - `object` – Object pronoun (e.g., "him", "her", "them")
  - `possessive` – Possessive adjective (e.g., "his", "her", "their")
  - `reflexive` – Reflexive pronoun (e.g., "himself", "herself", "themselves")
- `aliases` (array of strings) – Alternative names, nicknames, or usernames that may appear in transcripts
- `email_addresses` (array of strings) – Email addresses associated with the user for participant detection
- `timezone` (string) – IANA timezone identifier (e.g., "America/New_York", "Europe/London") for timestamp interpretation
- `entity` (string) – Links to a Person entity from the top-level entities.md file for entity tracking and analysis. Auto-matches by name prefix if not set.

**Entity Integration:**
When an entity is selected in the configuration UI, the entity's description is automatically loaded into the "Short Bio" field for editing. Any changes to the short bio are saved both to the identity configuration and to the entity itself in entities.md. This ensures the journal owner's biographical information stays synchronized between the identity config and the entity system, allowing agents and analysis tools to access consistent contextual information about the user.

This configuration helps meeting extraction identify the user as a participant, enables personalized agent interactions, and ensures timestamps are interpreted correctly across the journal.

### Convey configuration

The `convey` block contains settings for the web application:

```json
{
  "convey": {
    "password": "your-password-here"
  }
}
```

Fields:
- `password` (string) – Password for accessing the convey web application. When set, users must authenticate before accessing the journal interface.

#### Template usage examples

The structured pronoun format enables proper pronoun usage in generated text and agent responses:

```python
# In templates or generated text:
f"{identity.pronouns.subject} joined the meeting"  # "he joined the meeting"
f"I spoke with {identity.pronouns.object}"         # "I spoke with him"
f"That is {identity.pronouns.possessive} desk"     # "That is his desk"
f"{identity.pronouns.subject} did it {identity.pronouns.reflexive}"  # "he did it himself"
```

## Domain folders

The `domains/` directory provides a way to organize journal content by scope or focus area. Each domain represents a cohesive grouping of related activities, projects, or areas of interest.

### Domain structure

Each domain is organized as `domains/<domain>/` where `<domain>` is a descriptive short unique name. When referencing domains in the system, use hashtags (e.g., `#personal` for the "Personal Life" domain, `#ml_research` for "Machine Learning Research"). Each domain folder contains:

- `domain.json` – metadata file with domain title and description.
- `entities.md` – entities specific to this domain.
- `news/` – daily news and updates relevant to the domain (optional).
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

### Domain News

The `news/` directory provides a chronological record of news, updates, and external developments relevant to the domain. This allows tracking of industry news, research updates, regulatory changes, or any external information that impacts the domain's focus area.

#### News organization

News files are organized by date as `news/YYYYMMDD.md` where each file contains the day's relevant news items. Only create files for days that have news to record—sparse population is expected.

#### News file format

Each `YYYYMMDD.md` file is a markdown document with a consistent structure:

```markdown
# 2025-01-18 News - Machine Learning Research

## OpenAI Announces New Model Architecture
**Source:** techcrunch.com | **Time:** 09:15
Summary of the announcement and its relevance to current research projects...

## Paper: "Efficient Attention Mechanisms in Transformers"
**Source:** arxiv.org | **Time:** 14:30
Key findings from the paper and potential applications...

## Google Research Updates Dataset License Terms
**Source:** blog.google | **Time:** 16:45
Changes to dataset licensing that may affect ongoing experiments...
```

#### News entry structure

Each news entry should include:
- **Title** – concise headline as a level 2 heading
- **Source** – origin of the news (website, journal, etc.)
- **Time** – optional time of publication or discovery (HH:MM format)
- **Summary** – brief description focusing on relevance to the domain
- **Impact** – optional notes on how this affects domain work

#### News metadata

Optionally, a `news.json` file can be maintained at the root of the news directory to track metadata:

```json
{
  "last_updated": "2025-01-18",
  "sources": ["arxiv.org", "techcrunch.com", "nature.com"],
  "auto_fetch": false,
  "keywords": ["transformer", "attention", "llm", "research"]
}
```

This allows for future automation of news gathering while maintaining manual curation quality.

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

## todos/today.md Format

Each day folder stores a simple markdown checklist at `todos/today.md`. The
file is a flat list—no sections or headers—so the tools can treat every line as
a single actionable entry.

```markdown
- [ ] Draft standup update
- [ ] Review PR #1234 for indexing tweaks
- [x] Morning planning session notes
- [ ] ~~Cancel meeting with vendor~~
```

### Format Specification

**Line structure:**

```
- [checkbox] optional-context
```

**Components:**
- `- [ ]` – Uncompleted task checkbox
- `- [x]` – Completed task checkbox (lower- or upper-case `x` accepted)
- `optional-context` – Free-form markdown content; include timestamps,
  annotations, or a single `#domain` tag for cross-referencing (e.g., `Sync
  with design @ 14:00`, `File weekly review #think`).
- `~~text~~` – Wrap any portion of the line to mark cancellation while keeping
  the original wording visible.

**Rules:**
- Every checklist line becomes the source of truth for agent tools; external
  callers provide numbered views on demand rather than storing numbering in the
  file.
- Append new todos at the end of the file to maintain stable numbering
  semantics for concurrent tooling.
- Keep completed items in place by switching the checkbox to `[x]`.
- Use consistent phrasing so guard checks (which compare the full line) remain
  reliable.

This minimalist structure keeps manual editing simple while enabling automated
tools to manage tasks deterministically.

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
