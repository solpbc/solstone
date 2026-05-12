#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc
"""
Per-day rollup: load every segment timeline.json under chronicle/<day>/,
group by hour, pick top-4 per hour (concurrent), then top-4 across all the
hour picks for the day. Writes chronicle/<day>/timeline.json.

Output shape:
{
  "day": "20260409",
  "model": "gemini-3-flash-preview",
  "generated_at": <unix-seconds>,
  "segment_count": 21,
  "hour_count": 5,                     # hours that had any events
  "day_top": [{"title","description"}, ...up to 4],
  "hours": {
    "06": {
      "segment_count": 3,
      "picks": [{"title","description"}, ...up to 4],
      "rationale": "..."
    },
    ...
  }
}

Usage:
    timeline_rollup_day.py 20260409                   # one day
    timeline_rollup_day.py 20260408 20260411          # inclusive range
    timeline_rollup_day.py 20260409 --dry-run         # show work
    timeline_rollup_day.py 20260409 --force           # overwrite existing
    timeline_rollup_day.py 20260409 --top 4 --jobs 5

Run with /data/solstone/.venv/bin/python so think.* imports resolve.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

SOLSTONE_REPO = Path("/home/jer/projects/solstone")
sys.path.insert(0, str(SOLSTONE_REPO))
sys.path.insert(0, "/data/solstone/scratch")  # for timeline_rollup import


def bootstrap_solstone(journal_path: Path) -> None:
    os.environ.setdefault("SOLSTONE_JOURNAL", str(journal_path))
    from think.utils import get_config
    for k, v in get_config().get("env", {}).items():
        os.environ[k] = str(v)


SEGMENT_RE = re.compile(r"^(\d{2})(\d{2})(\d{2})_\d{1,6}$")  # captures HH MM SS


def parse_day(s: str) -> str:
    if not re.fullmatch(r"\d{8}", s):
        sys.exit(f"day must be YYYYMMDD, got {s!r}")
    return s


def day_range(start: str, end: str) -> list[str]:
    d0 = datetime.strptime(start, "%Y%m%d").date()
    d1 = datetime.strptime(end, "%Y%m%d").date()
    if d1 < d0:
        sys.exit(f"end {end} before start {start}")
    out = []
    cur = d0
    while cur <= d1:
        out.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return out


def origin_for_segment(seg_dir: Path) -> str:
    """Composite ID '<day>[/<stream>]/<seg>' relative to the chronicle root."""
    parts = seg_dir.parts
    try:
        ci = parts.index("chronicle")
    except ValueError:
        return seg_dir.name
    return "/".join(parts[ci + 1 :])


def load_day_segments(journal: Path, day: str) -> list[dict]:
    """
    Return [{"segment", "hour", "title", "description", "origin",
             "timeline_path"}, ...] sorted by segment start time. Walks any
    layout (chronicle/<day>/<seg>/, chronicle/<day>/<stream>/<seg>/, etc).
    `origin` is computed from the segment dir path so it works on files
    written before timeline_generate.py started embedding the field.
    """
    day_dir = journal / "chronicle" / day
    if not day_dir.is_dir():
        return []
    rows: list[dict] = []
    seen_segs: set[Path] = set()
    for tl in day_dir.rglob("timeline.json"):
        seg = None
        for ancestor in tl.parents:
            if SEGMENT_RE.match(ancestor.name):
                seg = ancestor
                break
        if not seg or seg in seen_segs:
            continue
        seen_segs.add(seg)
        m = SEGMENT_RE.match(seg.name)
        hh = m.group(1)
        try:
            data = json.loads(tl.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        title = data.get("title", "").strip()
        desc = data.get("description", "").strip()
        if not (title or desc):
            continue
        rows.append({
            "segment": seg.name,
            "hour": hh,
            "title": title,
            "description": desc,
            "origin": data.get("origin") or origin_for_segment(seg),
            "timeline_path": tl,
        })
    rows.sort(key=lambda r: r["segment"])
    return rows


def group_by_hour(segments: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for s in segments:
        out.setdefault(s["hour"], []).append(s)
    return out


def show_dry_run(day: str, segments: list[dict]) -> None:
    by_hour = group_by_hour(segments)
    print(f"\n== {day} ==  segments: {len(segments)}  hours: {len(by_hour)}")
    for hh in sorted(by_hour):
        rows = by_hour[hh]
        print(f"  {hh}h  ({len(rows)} segs)")
        for r in rows[:6]:
            print(f"      {r['segment']:20s}  {r['title']}  -  {r['description'][:70]}")
        if len(rows) > 6:
            print(f"      ... +{len(rows)-6} more")


async def rollup_day(
    journal: Path,
    day: str,
    top: int,
    jobs: int,
    dry_run: bool,
    force: bool,
) -> dict | None:
    out_path = journal / "chronicle" / day / "timeline.json"
    if out_path.exists() and not force and not dry_run:
        print(f"  [skip] {day}: timeline.json already exists (use --force to overwrite)")
        return None

    segments = load_day_segments(journal, day)
    if not segments:
        print(f"  [empty] {day}: no segment timeline.json found")
        return None

    by_hour = group_by_hour(segments)

    if dry_run:
        show_dry_run(day, segments)
        return None

    # Phase 1: per-hour rollups in parallel.
    # Each event carries `origin` so picks (which dereference the input
    # verbatim by index) preserve it through both rollup passes.
    from timeline_rollup import pick_top_events_batch
    jobs_in = []
    for hh in sorted(by_hour):
        events = [
            {"title": r["title"], "description": r["description"], "origin": r["origin"]}
            for r in by_hour[hh]
        ]
        jobs_in.append({"key": hh, "events": events})

    t0 = time.time()
    print(f"  rolling up {day}: {len(segments)} segs across {len(by_hour)} hours, "
          f"top={top}, jobs={jobs}")
    hour_results = await pick_top_events_batch(
        jobs=jobs_in, n=top, scope_label="hour", max_concurrent=jobs,
    )
    t_hr = time.time() - t0

    hours_out: dict[str, dict] = {}
    hour_picks_flat: list[dict] = []
    for rec in hour_results:
        hh = rec["key"]
        result = rec["result"]
        if "error" in result:
            print(f"    [hour-err {hh}h] {result['error'][:120]}")
            hours_out[hh] = {"segment_count": len(rec["events"]), "picks": [],
                             "rationale": "", "error": result["error"]}
            continue
        hours_out[hh] = {
            "segment_count": len(rec["events"]),
            "picks": result["picks"],
            "rationale": result["rationale"],
        }
        hour_picks_flat.extend(result["picks"])

    # Phase 2: day-level rollup across all hour picks. Wrapped in try/except
    # so a single backend cancel (we see ~0.15% 499 CANCELLED on long runs)
    # logs and skips the day's write rather than halting the whole loop —
    # the next idempotent run will pick this day back up.
    if not hour_picks_flat:
        day_top, day_rationale = [], "no hour picks available"
    else:
        from timeline_rollup import pick_top_events_async
        if len(hour_picks_flat) <= top:
            day_top = hour_picks_flat
            day_rationale = "fewer than N hour-picks; returning all"
        else:
            try:
                day_result = await pick_top_events_async(
                    events=hour_picks_flat, n=top, scope_label="day",
                )
                day_top = day_result["picks"]
                day_rationale = day_result["rationale"]
            except Exception as e:
                t_total = time.time() - t0
                print(f"  [day-err {day}] day-level rollup failed in "
                      f"{t_total:.1f}s: {str(e)[:160]}")
                print(f"  [day-err {day}] no timeline.json written; "
                      f"re-run will retry this day")
                return None
    t_total = time.time() - t0

    payload = {
        "day": day,
        "model": "gemini-3-flash-preview",
        "generated_at": int(time.time()),
        "segment_count": len(segments),
        "hour_count": len(hours_out),
        "day_top": day_top,
        "day_rationale": day_rationale,
        "hours": hours_out,
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"  [ok {day}] hour-rollup {t_hr:.1f}s  total {t_total:.1f}s  "
          f"→ {out_path}")
    print(f"      day_top: {[p['title'] for p in day_top]}")
    return payload


async def run(days: list[str], journal: Path, top: int, jobs: int,
              dry_run: bool, force: bool) -> int:
    for day in days:
        await rollup_day(journal, day, top, jobs, dry_run, force)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("days", nargs="+", help="YYYYMMDD; one or two for an inclusive range")
    ap.add_argument("--journal", default="/data/solstone/journal", type=Path)
    ap.add_argument("--top", type=int, default=4, help="top-N per hour and per day")
    ap.add_argument("--jobs", type=int, default=5, help="parallel hour-rollup calls")
    ap.add_argument("--dry-run", action="store_true",
                    help="list segments grouped by hour, don't call gemini")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing day-level timeline.json")
    args = ap.parse_args()

    if len(args.days) == 1:
        days = [parse_day(args.days[0])]
    elif len(args.days) == 2:
        days = day_range(parse_day(args.days[0]), parse_day(args.days[1]))
    else:
        sys.exit("pass one day or two for inclusive range")

    bootstrap_solstone(args.journal)

    return asyncio.run(run(days, args.journal, args.top, args.jobs,
                           args.dry_run, args.force))


if __name__ == "__main__":
    sys.exit(main())
