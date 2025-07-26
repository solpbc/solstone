# Changelog

Guide for updating: always append new entries to the existing list for the same day, but start a new day section at the top if the date has changed in the Mountain time zone.

## 2025-07-26
- Task list view truncates long descriptions with ellipsis and prevents wrapping
- Integrated task and event WebSocket endpoints into the main Flask app
- Search results allow filtering by clicking dates or topics which insert
  `day:` and `topic:` filters into the query and auto-run the search
- Search view gains a "Transcripts" tab to query raw data by day while ignoring
  `topic:` filters

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
- Search page now uses `#q=` fragments for shareable queries and auto-runs them.
- Query strings support `day:YYYYMMDD` and `topic:<topic>` filters parsed client side.
- Search APIs accept `day` and `topic` parameters and filter results accordingly.
