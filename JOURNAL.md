# Sunstone Journal Guide

This document describes the layout of a **journal** directory where all audio, screen and analysis artifacts are stored. Each dated `YYYYMMDD` folder is referred to as a **day**.

## Top level files

- `entities.md` – top list of entities gathered across days. Used by several tools.
- `indexer.json` – cache file created by `dream.entity_review` to speed up indexing.
- `voice_sample.flac` – optional reference voice sample loaded by `gemini-transcribe`.
- `entity_review.log` – operations performed in the web UI are appended here.
- `YYYYMMDD/` – individual day folders described below.

## Day folder contents

Audio capture tools write FLAC files and transcripts:

- `HHMMSS_raw.flac` – temporary mixed audio; removed after processing.
- `HHMMSS_audio.flac` – final clipped audio segment.
- `HHMMSS_audio.json` – transcript JSON produced by `gemini-transcribe`.

Screen capture utilities produce per monitor diff files:

- `HHMMSS_monitor_N_diff.png` – screenshot of changed region.
- `HHMMSS_monitor_N_diff_box.json` – bounding box for the change.
- `HHMMSS_monitor_N_diff.json` – Gemini description of the diff.

`reduce-screen` summarises these diffs into five‑minute chunks:

- `HHMMSS_screen.md` – Markdown summary for that interval.

Post‑processing commands may generate additional analysis files:

- `ponder_day.md` or `ponder_day.json` – high level summary of the day.
- `ponder_kg.md` – knowledge graph / network summary.
- `ponder_meetings.json` – meeting list used by the calendar web UI.
- `entities.md` – daily entity rollup produced by `entity-roll`.

### Crumbs

Most generated files are accompanied by a `.crumb` file capturing dependencies and model information. See `CRUMBS.md` for the format. Example: `20250610/ponder_day.md.crumb`.

