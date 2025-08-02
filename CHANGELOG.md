# Changelog

Guide for updating:
- Never remove or insert above this section
- The changelog is in chronological order (oldest first)
- Use a system tool to check the current date in the Mountain time zone
- Look to see if today's date has already been added as a section at the bottom of this file, start a new section if not
- Always append new entries to the existing list for today's section at the bottom

## 2025-07-25

- CHANGELOG.md created to start tracking agentic coding updates
- Renamed `think/detect.py` and `think/detect.txt` to `think/created_detect.py` and
  `think/created_detect.txt`, updating imports accordingly
- Renamed `think/border_detect.py` to `think/detect_border.py` and
  `think/created_detect.py` to `think/detect_created.py`, updating function names
  and references
- `think/detect_transcript.py` and `think/detect_transcript.txt` for splitting
  transcripts into 5 minute segments
- Renamed `detect_transcript` function and prompt to
  `detect_transcript_segment`/`detect_transcript_segment.txt`.
- `detect_transcript_json` utility and `detect_transcript_json.txt` for
  converting transcript text segments into the JSON format used by
  `hear.transcribe`.
- `think/importer.py` can now process `.txt` and `.pdf` transcript files.
- Dream import view saves uploads to `importer/`, detects timestamps via
  `detect_created`, launches importer tasks, and shows importer task log.
- Detected creation time via `detect_created` and splits transcripts with
  `detect_transcript_segment`, converting each chunk to JSON with
  `detect_transcript_json`.
- JSON segments are written as `HHMMSS_imported_audio.json` files incremented by
  five minutes per chunk.
- search_* functions no longer require JOURNAL_PATH argument; get_index infers from environment and tests updated
- think-agent supports `-o/--out` to write the final result or error to a file
- Added initial Agents view accessible at `/agents` with a rocket icon and a textarea form.
- Agents page now lists previous runs from `<journal>/agents` showing start time, model, persona and prompt.
- Search page now uses `#q=` fragments for shareable queries and auto-runs them.
- Query strings support `day:YYYYMMDD` and `topic:<topic>` filters parsed client side.
- Search APIs accept `day` and `topic` parameters and filter results accordingly.

## 2025-07-26

- Task list view truncates long descriptions with ellipsis and prevents wrapping
- Integrated task and event WebSocket endpoints into the main Flask app
- Search results allow filtering by clicking dates or topics which insert
  `day:` and `topic:` filters into the query and auto-run the search
- Raw search results for a specific day are ordered by time ascending rather
  than relevance
- Search view gains a "Transcripts" tab to query raw data by day while ignoring
  `topic:` filters
- Topic indexing renamed to "summaries" (old `search_topics` alias removed)
- Removed compatibility aliases `search_topic_api`, `search_occurrence_api`, and
  `search_raw_api`; tests now call `search_summaries_api`, `search_events_api`,
  and `search_transcripts_api` directly
- Indexer split into `summaries.sqlite`, `events.sqlite`, and per-day
  `transcripts.sqlite` files
- `think-indexer` CLI now requires `--index` and supports `--reset`
- Admin reindex task runs all three indexes sequentially
- `think-agent` provides an `AgentSession` helper for stateful runs with event callbacks.
- The standalone `think.agents` module was removed and its functionality merged into `think.agent`.
- `think-agent` CLI now uses `AgentSession` for a cleaner implementation.
- Added `think.genai` providing a Gemini-powered agent CLI mirroring `think.agent`.
- Updated `think.agent` and `think.genai` to rely on built-in in-memory session management.
- Cleaned up whitespace in detection utilities to satisfy linters.
- Unified `AgentSession` APIs for `think.agent` and `think.genai`.
- `dream` chat view now works with either agent implementation.
- Documentation expanded with a common interface section.

## 2025-07-27

- Dream chat backend now stores the shared `AgentSession` instance instead of
  manually managing chat history.
- Added `think.google` module replacing `think.genai`.
- Added `think.events` module standardizing event emission across backends.
- Introduced `BaseAgentSession` ensuring common API for agent sessions.
- OpenAI logic moved to `think.openai`; `think.agent` now re-exports it.
- Merged `agent.py`, `agent_session.py` and `events.py` into new `agents.py`.
- Dream chat view now lets you pick Google or OpenAI as the backend and clears
  history when changed.
- Removed re-exports from `think.agents`; imports must use `think.openai` or
  `think.google` directly.
- Dream APIs return JSON error bodies and the client shows errors in a modal.
- Chat view now streams agent tool events via WebSocket and displays
  small search cards linking to the relevant query.
- `think.ponder` skips occurrence generation when the topic metadata
  includes `"skip_occurrences": true`.
- `think.utils.get_raw_file` returns the relative raw path, mime type and
  metadata for a transcript.
- `think.mcp_server.get_media` resource returns the raw FLAC or PNG
  referenced by a transcript JSON using `think.utils.get_raw_file`.
- Agent instructions moved to `think/agents/default.txt` and loaded via
  `agent_instructions(persona)` with optional personas.
- Agent sessions are automatically logged under `JOURNAL_PATH/agents` as JSONL files.

## 2025-07-28

- Agent start events now log the active persona and model.
- Added `think.anthropic` CLI using the Anthropic Claude SDK.
- New ``think-agents`` CLI consolidates ``think-agent`` and ``think-claude``.
- Backend modules no longer expose their own command line interfaces.
- Agent event JSON messages now include a ``ts`` epoch timestamp.
- `think.ponder` now skips occurrence generation when the topic metadata
  has `"occurrences": false`.
- Dream chat view now supports the Anthropic backend and UI shows a "Claude"
  option.
- Fixed `think.google` to use Gemini's asynchronous chat API instead of running
  synchronous calls in a thread.
- Added constants for OpenAI models `GPT_4O`, `GPT_4_TURBO`, `GPT_O3` and `GPT_O4_MINI`.
- `think.openai` now defaults to `GPT_O4_MINI`.
- Updated tests for the new default model.
- Bumped MCP client timeout for OpenAI agent to avoid stdio cancel scope errors.
- Fixed Anthropic agent sessions to reconnect MCP client if the connection closes.
- OpenAI chat backend now starts and stops its MCP server for each request,
  fixing cancel scope errors when switching sessions.

## 2025-07-29

- Added `entities` index to `think.indexer` with FTS search of `entities.md` files.
- Refactored entity indexing to use `entities` and `entity_appearances` tables,
  tracking first/last seen days and top descriptions.

## 2025-07-30

- Enhanced dream chat view with modern UI improvements including gradient user messages,
  animated message appearances, and improved input styling with rounded corners
- Added server-side markdown rendering for bot messages using Python markdown library
- Implemented typing indicator animation that displays while waiting for agent responses
- Improved backend selector with emoji indicators for each AI provider (Google, OpenAI, Claude)

## 2025-07-31

- Added `think.planner` module with CLI and `generate_plan` function for creating agent plans using Gemini Pro.
- New `think-planner` command exposed via project scripts.
- Added `get_resource` tool in `think.mcp_server` to fetch ``journal://`` resources through a tool call.
- Fixed `get_resource` to return `Image`/`Audio` wrappers instead of base64-encoded blobs.
