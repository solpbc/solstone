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
- `think.mcp_tools.get_media` resource returns the raw FLAC or PNG
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
- Added `get_resource` tool in `think.mcp_tools` to fetch ``journal://`` resources through a tool call.
- Fixed `get_resource` to return `Image`/`Audio` wrappers instead of base64-encoded blobs.

## 2025-08-02

- Renamed `think.mcp_server` to `think.mcp_tools` and updated project references.
- `think.supervisor` now launches an HTTP MCP server to expose Sunstone tools.

## 2025-08-08

- Updated OpenAI agent hooks to align with latest Agents SDK lifecycle events.

## 2025-08-09

- Added `cluster_scan` to `think.cluster` to report 15-minute intervals with
  audio and screen transcripts and exposed it via the package API.
- Added calendar transcript viewer that uses `think.cluster_scan` and `cluster_range` to browse and load transcripts for a day.

## 2025-08-10

- Streamlined domain creation by auto-generating handles from titles and hiding the description field.
- Removed descriptions from the domains list view.

## 2025-08-12

- Added "source" column to think/indexer transcripts to track origin of text (e.g., 'mic', 'sys', 'monitor_1')
- Audio transcripts now extract source from the JSON "source" field alongside "text"
- Screen diff transcripts extract source from filename pattern (e.g., '123456_monitor_1_diff.json' → 'monitor_1')
- Added optional --source filter to transcript search for filtering results by source
- Search results now display source in brackets (e.g., [mic], [sys], [monitor_1])

## 2025-08-16

- Standardised agent event fields across OpenAI, Gemini and Claude backends
- Documented Cortex WebSocket API and event types in new `CORTEX.md`

## 2025-09-05

- Migrated `think/supervisor.py` to use `CortexClient` for spawning scheduled agents instead of direct subprocess calls
- Centralized all agent spawning through the Cortex service for proper event tracking and management
- Updated documentation to clarify Cortex as the central agent manager
- Removed references to direct `think-agents` invocation from documentation
- Removed redundant `test_openai_backend_multi_turn` test that duplicated basic functionality after simplification
- Added comprehensive integration test suite for Cortex service (`tests/integration/test_cortex.py`) with streaming event verification
- Added date range filtering support to `think.indexer.transcripts.search_transcripts()` with `start_date` and `end_date` parameters
- Updated MCP tools `search_transcripts` to support date range queries alongside single day searches
- Made `day` parameter optional in MCP `search_transcripts` tool to allow searching across all days or date ranges

## 2025-09-07

- Enhanced Cortex service to clean up stale active agent files on startup, marking them as failed due to unexpected shutdown

## 2025-09-08

- Added `cortex_request()` function to `think/cortex_client.py` for creating Cortex agent request files
- Enhanced `think/cortex.py` to validate request files and handle errors more gracefully before spawning agents
- Refactored `think/cortex.py` to use `cortex_request()` for handoff functionality, applying DRY principles
- Added `cortex_watch()` function to `think/cortex_client.py` for blocking file watch of agent events with efficient tailing using watchfiles
- Added watchfiles dependency for monitoring file system changes in Cortex agent directory
- Replaced `tests/test_cortex_client.py` with comprehensive tests for new cortex_client functions
- Added `run_agent()` function to `think/cortex_client.py` that combines request creation and event watching for synchronous agent execution
- Simplified `dream/cortex_utils.py` to use the new synchronous `run_agent()` function, removing unnecessary async wrapper infrastructure
- Enhanced `cortex_watch()` with FileState dataclass to properly handle atomic file renames by tracking inodes
- Added partial line buffering to `cortex_watch()` to handle incomplete JSON lines without newlines
- Added error protection to `cortex_watch()` callback execution, treating exceptions as stop signals
- Added `cortex_agents()` function to `think/cortex_client.py` for listing agents with pagination and filtering support
- Refactored dream app views to use `cortex_agents()` and `cortex_request()` directly instead of wrapper classes
- Added comprehensive tests for `cortex_agents()` function covering pagination, filtering, and edge cases
- Removed unnecessary global client wrapper functions and cleanup code from dream app
- Fixed Cortex service to properly reap zombie processes by using `wait()` instead of `poll()` and breaking on finish/error events
- Fixed Dream agents view JavaScript filtering by non-existent `is_live` field, changed to use `status === 'running'`
- Enhanced `cortex_agents()` function to extract prompt, model (from backend), and runtime_seconds fields for frontend display
- Fixed timestamp display showing negative seconds by converting milliseconds to seconds before passing to `time_since()`
- Updated `think/supervisor.py` to use new `cortex_request()` function instead of deprecated `CortexClient` class
- Converted `spawn_scheduled_agents()` from async to synchronous, removing unnecessary asyncio usage
- Updated `think/README.md` documentation to reflect new cortex_client function-based API
- Fixed test mocks in `test_supervisor_schedule.py` to match the new cortex_client interface
- Refactored `tests/test_cortex_client.py` to remove old TestCortexClient class and align tests with the refactored cortex_client module
- Updated `tests/test_agents_view.py` to work with refactored cortex_client functions instead of removed SyncCortexClient class
- Updated `tests/test_review_chat.py` to set JOURNAL_PATH environment variable for proper cortex_client operation
- Refactored `tests/integration/test_cortex.py` to use new cortex_client functional API instead of CortexClient class
- Fixed 8 failing tests in `tests/test_cortex.py` to align with refactored Cortex service that now passes full request objects instead of individual parameters
- Updated `tests/test_todo_generation.py` to use new `cortex_request()` API and properly set JOURNAL_PATH environment variable

## 2025-09-09

- Fixed test cleanup issue in `tests/integration/test_cortex.py` by using daemon threads and simplifying error handling test to avoid indefinite blocking in watchfiles
- Updated integration tests to use correct model constants from `think/models.py` for consistency
- Fixed calendar day view tabs to always fit within viewport by using flexbox instead of scrolling
- Added title attributes to calendar tabs to show full names as tooltips on hover
- Added `save` field support to Cortex agent requests for saving results to journal day directories
- Updated `cortex_request()` in `think/cortex_client.py` to accept optional `save` parameter
- Modified `think/cortex.py` to save agent results to `<journal>/<current_day>/<filename>` when save field is present
- Updated `think/supervisor.py` to pass save field from agent metadata for scheduled agents
- Enhanced `CORTEX.md` documentation to describe the new save field and Agent Result Saving feature
- Added comprehensive tests for save functionality including error handling in `tests/test_cortex.py`
- Enhanced Cortex save functionality to support optional `day` parameter for saving to specific journal day directories
- Updated `_save_agent_result()` in `think/cortex.py` to use `think.utils.day_path()` for proper YYYYMMDD validation
- Updated `CORTEX.md` documentation to describe the optional day parameter for Agent Result Saving
- Added comprehensive test coverage for day parameter including validation and error handling in `tests/test_cortex.py`
- Enhanced `day_path()` utility in `think/utils.py` to accept optional day parameter (defaults to today), auto-create directories, and return Path objects
- Simplified 9 call sites across the codebase by removing redundant date calculations, Path() conversions, and mkdir() calls
- Updated callers in `think/cortex.py`, `think/cluster.py`, `dream/views/calendar.py`, `think/ponder.py`, `think/entity_roll.py`, `see/reduce.py`, and `hear/split.py`
- Fixed test mocks in `tests/test_cortex.py` to patch `think.utils.datetime` instead of `think.cortex.datetime` for proper date mocking
- Refactored 5 additional locations to use the improved `day_path()` utility for better DRY compliance
- Updated `see/scan.py` `recent_audio_activity()` to use `day_path()` instead of manual path construction
- Updated `see/describe.py` `repair_day()` and `start()` methods to use `day_path()` for day directory creation
- Updated `hear/transcribe.py` `start()` method to use `day_path()` for monitoring loop
- Updated `hear/capture.py` `save_flac()` method to use `day_path()` for audio file saving
