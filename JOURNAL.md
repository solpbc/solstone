# Sunstone Journal Guide

This document describes the layout of a **journal** directory where all audio, screen and analysis artifacts are stored. Each dated `YYYYMMDD` folder is referred to as a **day**, and within each day captured content is organized into **periods** (timestamped duration folders). Each period folder uses the format `HHMMSS_LEN/` where `HHMMSS` is the start time and `LEN` is the duration in seconds. This folder name serves as the **period key**, uniquely identifying the period within a given day.

## Top level files

- `task_log.txt` – optional log of utility runs in `[epoch]\tmessage` format.
- `config/journal.json` – user configuration for the journal (optional, see below).
- `facets/` – facet-specific organization folders described below.
- `inbox/` – asynchronous messaging system for agent communications described below.
- `tokens/` – token usage logs from AI model calls, organized by day (see below).
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

## Facet folders

The `facets/` directory provides a way to organize journal content by scope or focus area. Each facet represents a cohesive grouping of related activities, projects, or areas of interest.

### Facet structure

Each facet is organized as `facets/<facet>/` where `<facet>` is a descriptive short unique name. When referencing facets in the system, use hashtags (e.g., `#personal` for the "Personal Life" facet, `#ml_research` for "Machine Learning Research"). Each facet folder contains:

- `facet.json` – metadata file with facet title and description.
- `entities.jsonl` – entities specific to this facet in JSONL format.
- `news/` – daily news and updates relevant to the facet (optional).

### Facet metadata

The `facet.json` file contains basic information about the facet:

```json
{
  "title": "Machine Learning Research",
  "description": "AI/ML research projects, experiments, and related activities",
  "color": "#4f46e5",
  "emoji": "🧠"
}
```

Optional fields:
- `color` – hex color code for the facet card background in the web UI
- `emoji` – emoji icon displayed in the top-left of the facet card

### Facet Entities

Entities in Sunstone use a two-state system: **detected** (daily discoveries) and **attached** (promoted/persistent). This agent-driven architecture automatically identifies entities from journal content while allowing manual curation.

#### Entity Storage Structure

```
facets/{facet}/
  ├── entities.jsonl              # Attached entities (persistent)
  └── entities/YYYYMMDD.jsonl     # Daily detected entities
```

#### Attached Entities

The `entities.jsonl` file contains manually promoted entities that are persistently associated with the facet. These entities are loaded into agent context and appear in the facet UI as starred items.

Format example (JSONL - one JSON object per line):
```jsonl
{"type": "Person", "name": "Alice Johnson", "description": "Lead engineer on the API project", "aka": ["Ali", "AJ"]}
{"type": "Company", "name": "TechCorp", "description": "Primary client for consulting work", "tier": "enterprise", "aka": ["TC", "TechCo"]}
{"type": "Project", "name": "API Optimization", "description": "Performance improvement initiative", "status": "active", "priority": "high"}
{"type": "Tool", "name": "PostgreSQL", "description": "Database system used in production", "version": "16.0", "aka": ["Postgres", "PG"]}
```

Entity types are flexible and user-defined. Common examples: `Person`, `Company`, `Project`, `Tool`, `Location`, `Event`. Type names must be alphanumeric with spaces, minimum 3 characters.

Each entity is a JSON object with required fields (`type`, `name`, `description`) and optional custom fields for extensibility (e.g., `status`, `priority`, `tags`, `contact`, etc.). Custom fields are preserved throughout the system.

**Standard optional field:**
- `aka` (array of strings) – Alternative names, nicknames, or acronyms for the entity. Used in audio transcription to improve entity recognition.

#### Detected Entities

Daily entity detection files (`entities/YYYYMMDD.jsonl`) contain entities automatically discovered by agents from:
- Journal transcripts and screen captures
- Knowledge graphs and summaries
- News feeds and external content

Detected entities accumulate historical context over time. Entities appearing in multiple daily detections can be promoted to attached status through the web UI or MCP tools.

Format matches attached entities (JSONL):
```jsonl
{"type": "Person", "name": "Charlie Brown", "description": "Mentioned in standup meeting"}
{"type": "Tool", "name": "React", "description": "Used in UI development work"}
```

#### Entity Lifecycle

1. **Detection**: Daily agents scan journal content and record entities in `entities/YYYYMMDD.jsonl`
2. **Aggregation**: Review agent tracks detection frequency across recent days
3. **Promotion**: Entities with 3+ detections are auto-promoted to attached, or users manually promote via UI
4. **Persistence**: Attached entities in `entities.jsonl` remain until manually removed

#### Cross-Facet Behavior

The same entity name can exist in multiple facets with independent descriptions. Agents receive entity context from all facets, with alphabetically-first facet winning for name conflicts during aggregation.

### Facet News

The `news/` directory provides a chronological record of news, updates, and external developments relevant to the facet. This allows tracking of industry news, research updates, regulatory changes, or any external information that impacts the facet's focus area.

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
- **Summary** – brief description focusing on relevance to the facet
- **Impact** – optional notes on how this affects facet work

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

## Facet-Scoped Todos

Todos are organized by facet in `facets/{facet}/todos/{day}.md` where each file stores a simple markdown checklist. Todos belong to a specific facet (e.g., "personal", "work", "research") and are completely separated by scope.

**File path pattern:**
```
facets/personal/todos/20250110.md
facets/work/todos/20250110.md
facets/research/todos/20250112.md
```

Each file is a flat list—no sections or headers—so the tools can treat every line as a single actionable entry.

```markdown
- [ ] Draft standup update
- [ ] Review PR #1234 for indexing tweaks (14:30)
- [x] Morning planning session notes
- [ ] ~~Cancel meeting with vendor~~
```

### Format Specification

**Line structure:**

```
- [checkbox] task description with optional time annotation
```

**Components:**
- `- [ ]` – Uncompleted task checkbox
- `- [x]` – Completed task checkbox (lower- or upper-case `x` accepted)
- `task description` – Free-form markdown content describing the task
- `(HH:MM)` – Optional time annotation for scheduled work (e.g., `(14:30)`)
- `~~text~~` – Wrap any portion of the line to mark cancellation while keeping the original wording visible

**Facet context:**
- Facet is determined by the file location, not inline tags
- Each facet has its own independent todo list for each day
- Work todos (`facets/work/todos/`) are completely separate from personal todos (`facets/personal/todos/`)
- No `#facet` tags are needed in the content since the facet context comes from the file path

**Rules:**
- Every checklist line becomes the source of truth for agent tools; external callers provide numbered views on demand rather than storing numbering in the file
- Append new todos at the end of the file to maintain stable numbering semantics for concurrent tooling
- Keep completed items in place by switching the checkbox to `[x]`
- Use consistent phrasing so guard checks (which compare the full line) remain reliable

**MCP Tool Access:**
All todo operations require both `day` and `facet` parameters:
- `todo_list(day, facet)` – view numbered checklist for a specific facet
- `todo_add(day, facet, line_number, text)` – add new todo
- `todo_done(day, facet, line_number, guard)` – mark complete
- `todo_remove(day, facet, line_number, guard)` – remove entry
- `todo_upcoming(limit, facet=None)` – view upcoming todos (optionally filtered by facet)

This facet-scoped structure provides true separation of concerns while keeping manual editing simple and enabling automated tools to manage tasks deterministically.

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
    "facet": "ml_research",
    "day": "20250117"
  }
}
```

Required fields:
- `id` – unique message identifier matching the filename
- `timestamp` – epoch milliseconds when the message was created
- `from` – sender information with `type` (agent/system/facet) and `id`
- `body` – message content as text or markdown
- `status` – message state (unread/read/archived)

Optional fields:
- `context` – reference to related journal entities (facet, day)

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

## Token Usage

The `tokens/` directory tracks token usage from all AI model calls across the system. Usage data is organized by day as `tokens/YYYYMMDD.jsonl` where each file contains JSON Lines entries for that day's API calls.

### Token log format

Each line in a token log file is a JSON object with the following structure:

```json
{
  "timestamp": 1736812345.678,
  "model": "gemini-2.5-flash",
  "context": "agent.default.20250113_143022",
  "usage": {
    "input_tokens": 1500,
    "output_tokens": 500,
    "total_tokens": 2000,
    "cached_tokens": 800,
    "reasoning_tokens": 200
  }
}
```

Required fields:
- `timestamp` – Unix timestamp (seconds with fractional milliseconds)
- `model` – Model identifier (e.g., "gemini-2.5-flash", "gpt-5", "claude-sonnet-4-5")
- `context` – Calling context (e.g., "agent.persona.agent_id" or "module.function:line")
- `usage` – Token counts dictionary with normalized field names

Usage fields (all optional depending on model capabilities):
- `input_tokens` – Tokens in the prompt/input
- `output_tokens` – Tokens in the response/output
- `total_tokens` – Total tokens consumed
- `cached_tokens` – Tokens served from cache (reduces cost)
- `reasoning_tokens` – Tokens used for extended thinking/reasoning
- `requests` – Number of API requests made (for batch operations)

The logging system normalizes provider-specific formats (OpenAI, Gemini, Anthropic) into this unified schema for consistent cost tracking and analysis across all models.

## Day folder contents

Within each day, captured content is organized into **periods** (timestamped duration folders). The folder name is the **period key**, which uniquely identifies the period within the day and follows this format:

- `HHMMSS_LEN/` – Start time and duration in seconds (e.g., `143022_300/` for a 5-minute period starting at 14:30:22)

Audio capture tools write FLAC files and transcripts:

- `HHMMSS_LEN_*.flac` – audio files in day root (e.g., `143022_300_audio.flac`), moved to period after transcription.
- `HHMMSS_LEN/*.flac` – audio files moved here after processing, preserving descriptive suffix (e.g., `audio.flac`, `mic.flac`).
- `HHMMSS_LEN/audio.jsonl` – transcript JSONL produced by transcription.

Note: The descriptive portion after the period (e.g., `_audio`, `_recording`) is preserved when files are moved into period directories. Processing tools match files by extension only, ignoring the descriptive suffix.

### Audio transcript output

The transcript file (`*_audio.jsonl`) contains a metadata line followed by one JSON object per transcript segment.

Example transcript file:

```jsonl
{"raw": "audio.flac", "topics": ["authentication", "testing", "planning"], "setting": "workplace"}
{"start": "00:00:01", "source": "mic", "speaker": 1, "text": "So we need to finalize the authentication module today.", "description": "professional tone"}
{"start": "00:00:15", "source": "sys", "speaker": "Alice", "text": "I agree. [clears throat] Let's make sure we have proper unit tests.", "description": "thoughtful, slightly hesitant"}
```

**Metadata line (first line):**
- `raw` – path to processed audio file (required)
- `topics` – array of conversation topics extracted by the model (optional)
- `setting` – environment or context description, e.g., "workplace", "personal", "educational" (optional)
- `imported` – object with import metadata for external files (optional):
  - `id` – unique import identifier
  - `facet` – facet name for entity extraction
  - `setting` – contextual setting description

**Transcript segments (subsequent lines):**
- `start` – timestamp in HH:MM:SS format (required)
- `text` – transcribed text with inline vocalizations in brackets like "[laughs]", "[sigh]" (required)
- `source` – audio source: "mic" or "sys" (optional)
- `speaker` – speaker identifier, numeric or string (optional)
- `description` – audio-impaired style description of tone, emotion, vocal quality (optional)

Screen capture produces screencast videos with multi-monitor metadata:

- `HHMMSS_LEN_*.webm` – screencast video files in day root (e.g., `143022_300_screen.webm`), moved to period after analysis.
- `HHMMSS_LEN/*.webm` – video files moved here after analysis, preserving descriptive suffix (e.g., `screen.webm`, `monitor1.webm`).
- `HHMMSS_LEN/screen.jsonl` – vision analysis results in JSON Lines format.
- `HHMMSS_LEN/screen.md` – human-readable markdown summary of the video.

Note: Like audio files, the descriptive portion is preserved when files are moved into period directories.

### Screencast video format

Videos contain monitor layout information in their metadata title field using the format:
```
DP-3:center,1920,0,5360,1440 HDMI-4:right,5360,219,7280,1299
```

Each monitor entry: `<monitor_name>:<position>,<x1>,<y1>,<x2>,<y2>` where coordinates define the monitor's bounding box in the combined virtual screen space.

### Vision analysis output

The analysis file (`*_screen.jsonl`) contains one JSON object per qualified frame. Frames qualify when they contain a changed region of at least 400×400 pixels, detected using block-based SSIM comparison.

Example frame record:

```json
{
  "frame_id": 123,
  "timestamp": 45.67,
  "monitor": "DP-3",
  "monitor_position": "center",
  "box_2d": [100, 200, 500, 600],
  "requests": [
    {"type": "describe_json", "model": "gemini-2.0-flash-lite", "duration": 0.5}
  ],
  "analysis": {
    "visual_description": "A terminal window showing command output with green text on dark background.",
    "visible": "terminal"
  }
}
```

**Common fields:**
- `frame_id` – sequential frame number in the video
- `timestamp` – time in seconds from video start
- `monitor` – monitor identifier from video metadata
- `monitor_position` – optional monitor position (e.g., "center", "left", "right")
- `box_2d` – bounding box of changed region `[y_min, x_min, y_max, x_max]` relative to monitor
- `requests` – list of vision API requests made for this frame
- `analysis` – categorization and visual description from initial analysis

**Optional fields (conditional processing):**
- `extracted_text` – present when frame contains messaging, browsing, reading, or productivity content
- `meeting_analysis` – present when frame contains video conferencing, includes participant detection and bounding boxes
- `error` – present when processing failed after retries

The vision analysis uses multi-stage conditional processing:
1. Initial categorization determines content type (terminal, code, messaging, meeting, browsing, reading, media, gaming, productivity)
2. Text extraction triggered for categories: messaging, browsing, reading, productivity
3. Meeting analysis triggered for meeting category, provides full-screen participant detection with entity recognition

### Summary generation

After all frames are processed, a markdown summary (`*_screen.md`) is generated from the analysis file. The summary provides a chronological narrative of the screencast, organizing frames by timestamp and including visual descriptions, extracted text, and meeting analysis where applicable.

Post‑processing commands may generate additional analysis files, for example:

- `topics/flow.md` – high level summary of the day.
- `topics/knowledge_graph.md` – knowledge graph / network summary.
- `topics/meetings.md` – meeting list used by the calendar web UI.
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
      "facet": "work",
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
- **facet** – facet name the occurrence is associated with (e.g., "work", "personal", "ml_research").
- **work** – boolean, work vs. personal classification when known.
- **participants** – optional list of people or entities involved.
- **details** – free-form string of other occurrence specific information.

Each topic analysis can map its findings into this structure allowing the
indexer to collect and search occurrences across all days.
