# timeline-app graduation seed

This directory holds the original timeline-app prototype source. It is **input material** for two hop lodes that build the proper `apps/timeline/` Convey app and Cortex-scheduled talents.

The `_seed/` directory is removed after both lodes ship and the integration test passes. Do not import from these files; do not depend on them at runtime. Read them to understand the behavior the new code must reproduce.

## what's here

| file | role |
|------|------|
| `prototype.html` | Single-file vanilla-JS demo app — five zoom levels (year → month → day → hour → 5-min), inline CSS + inline JS. The shape the new `apps/timeline/workspace.html` + `static/timeline.{js,css}` must reproduce. |
| `data-mock.js` | Baseline mock data used as fallback when the API is unreachable. Same shape the new routes return. |
| `timeline_server.py` | Stdlib Python HTTP server with four JSON endpoints (`/api/index`, `/api/day/<YYYYMMDD>`, `/api/segment/<day>/<stream>/<seg>`, plus static). The new `apps/timeline/routes.py` reproduces these endpoint shapes inside Convey — see scope for the one shape change (`/api/index` slims + new `/api/month/<YYYYMM>`). |
| `timeline_generate.py` | Per-segment LLM summarizer — input `<seg>/talents/activity.md`, output `<seg>/timeline.json`. The behavior the new `talent/segment_summary.{md,py}` reproduces under Cortex (schedule=segment). |
| `timeline_rollup.py` | Shared rollup library — `pick_top_events_async()` calling Gemini to pick top-N events from a candidate list. Used by both day and master rollup. Re-implement inline in the talent hooks or as a small shared module. |
| `timeline_rollup_day.py` | Per-day rollup — walks `<seg>/timeline.json` files, picks top 4 per hour + top 4 across the day. Output `chronicle/<day>/timeline.json`. The behavior the new `talent/day_rollup.{md,py}` reproduces (schedule=daily, priority lower than master_rollup). |
| `timeline_rollup_master.py` | Master rollup — walks `chronicle/<day>/timeline.json` files, picks top 4 per month, composes year-top from month-tops, writes `<journal>/timeline.json`. The behavior the new `talent/master_rollup.{md,py}` reproduces (schedule=daily, priority higher than day_rollup). |

## paths in these files

The seed Python files hard-code paths under `/data/solstone/...` and `/home/jer/projects/extro/...` from when they ran outside the repo on the maintainer's host. The new code in `apps/timeline/` must use the same path-resolution conventions as other apps and talents in solstone — `state.journal_root`, `solstone.think.utils.day_path()`, `solstone.think.utils.iter_segments()`, etc. — not hardcoded absolute paths.

## models

- `timeline_generate.py` uses `gemini-3.1-flash-lite` (literal). Production. Use as-is.
- `timeline_rollup*.py` use `gemini-3-flash-preview` (literal). Matches solstone's `GEMINI_FLASH` in `think/models.py`. Use as-is.

## what becomes of this directory

Both lodes leave `_seed/` untouched. Maintainer removes the entire `_seed/` directory after both lodes ship and pass integration testing on the journal.
