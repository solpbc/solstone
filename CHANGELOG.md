# Changelog

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

## 2025-07-26

- `think/importer.py` can now process `.txt` and `.pdf` transcript files.
- Detected creation time via `detect_created` and splits transcripts with
  `detect_transcript_segment`, converting each chunk to JSON with
  `detect_transcript_json`.
- JSON segments are written as `HHMMSS_imported_audio.json` files incremented by
  five minutes per chunk.
