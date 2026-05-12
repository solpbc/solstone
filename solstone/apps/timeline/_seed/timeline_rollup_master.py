#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc
"""
Master timeline rollup: days → months → year, all in one big JSON blob.

Walks every chronicle/<day>/timeline.json (the per-day rollups), groups days
by YYYY-MM, calls gemini-3-flash-preview to pick top-N day_top events per
month, then picks the headline of each month for the year row. Composes
everything into a single `journal/timeline.json` master file the prototype
can fetch in one round-trip.

Output shape:
{
  "generated_at": <unix>,
  "model": "gemini-3-flash-preview",
  "top_n": 4,
  "year_top": [
    {"month": "202507", "title", "description", "origin"}, ...one per month
  ],
  "months": {
    "202507": {
      "month_top": [{title, description, origin}, ...up to top_n],
      "month_rationale": "...",
      "day_count": 23,
      "days": {
        "20250725": <full day-level timeline.json content>,
        ...
      }
    },
    ...
  }
}

Usage:
    timeline_rollup_master.py                          # full year, all months
    timeline_rollup_master.py --dry-run                # show grouping, no calls
    timeline_rollup_master.py --force                  # overwrite existing
    timeline_rollup_master.py --top 4 --jobs 5         # tunables
    timeline_rollup_master.py --months 202604,202605   # subset (debug)

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
from pathlib import Path

SOLSTONE_REPO = Path("/home/jer/projects/solstone")
sys.path.insert(0, str(SOLSTONE_REPO))
sys.path.insert(0, "/data/solstone/scratch")  # for timeline_rollup import


def bootstrap_solstone(journal_path: Path) -> None:
    os.environ.setdefault("SOLSTONE_JOURNAL", str(journal_path))
    from think.utils import get_config
    for k, v in get_config().get("env", {}).items():
        os.environ[k] = str(v)


DAY_RE = re.compile(r"^\d{8}$")
MODEL = "gemini-3-flash-preview"


def load_day_rollups(journal: Path) -> dict[str, dict]:
    """Map day (YYYYMMDD) → parsed day-level timeline.json (rollup)."""
    chronicle = journal / "chronicle"
    out = {}
    for day_dir in sorted(chronicle.iterdir()):
        if not (day_dir.is_dir() and DAY_RE.match(day_dir.name)):
            continue
        tl = day_dir / "timeline.json"
        if not tl.is_file():
            continue
        try:
            data = json.loads(tl.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not data.get("day_top"):
            continue
        out[day_dir.name] = data
    return out


def group_by_month(day_rollups: dict[str, dict]) -> dict[str, list[str]]:
    """Map YYYYMM → sorted list of YYYYMMDD strings present that month."""
    by_month: dict[str, list[str]] = {}
    for day in sorted(day_rollups):
        ym = day[:6]
        by_month.setdefault(ym, []).append(day)
    return by_month


def show_dry_run(day_rollups: dict[str, dict], by_month: dict[str, list[str]]) -> None:
    print(f"\n== dry run ==")
    print(f"days with day-level timeline.json: {len(day_rollups)}")
    print(f"months covered                   : {len(by_month)}")
    print()
    for ym in sorted(by_month):
        days = by_month[ym]
        cands = sum(len(day_rollups[d].get("day_top") or []) for d in days)
        print(f"  {ym}  {len(days):3d} days, {cands:4d} day_top candidates")
        for d in days[:3]:
            top0 = day_rollups[d]["day_top"][0]
            print(f"      {d}  {top0.get('title','')}")
        if len(days) > 3:
            print(f"      ... +{len(days)-3} more")
    print()


async def rollup_master(
    journal: Path,
    top: int,
    jobs: int,
    dry_run: bool,
    force: bool,
    months_filter: set[str] | None,
) -> int:
    out_path = journal / "timeline.json"
    if out_path.exists() and not force and not dry_run:
        sys.exit(f"{out_path} exists; pass --force to overwrite")

    day_rollups = load_day_rollups(journal)
    if not day_rollups:
        sys.exit(f"no day-level timeline.json found under {journal}/chronicle/*/")

    by_month = group_by_month(day_rollups)
    if months_filter:
        by_month = {ym: ds for ym, ds in by_month.items() if ym in months_filter}
        if not by_month:
            sys.exit(f"no overlap between --months {sorted(months_filter)} and journal")

    if dry_run:
        show_dry_run(day_rollups, by_month)
        return 0

    # Build the per-month rollup jobs. For each month: candidate set is the
    # union of every day_top in that month (each event already carries
    # title/description/origin from earlier passes — picks dereference verbatim).
    from timeline_rollup import pick_top_events_batch

    jobs_in = []
    for ym in sorted(by_month):
        events = []
        for d in by_month[ym]:
            for ev in (day_rollups[d].get("day_top") or []):
                events.append({
                    "title": ev.get("title", ""),
                    "description": ev.get("description", ""),
                    "origin": ev.get("origin", ""),
                })
        jobs_in.append({"key": ym, "events": events})

    print(f"rolling up {len(jobs_in)} month(s) with model={MODEL} top={top} jobs={jobs}")
    t0 = time.time()
    month_results = await pick_top_events_batch(
        jobs=jobs_in, n=top, scope_label="month", max_concurrent=jobs,
    )
    t_total = time.time() - t0
    print(f"month-rollup done in {t_total:.1f}s")

    # Compose master output.
    months_out: dict[str, dict] = {}
    year_top: list[dict] = []
    for rec in month_results:
        ym = rec["key"]
        result = rec["result"]
        days = by_month[ym]

        if "error" in result:
            print(f"  [month-err {ym}] {result['error'][:120]}")
            month_top = []
            month_rationale = f"ERROR: {result['error'][:200]}"
        else:
            month_top = result["picks"]
            month_rationale = result["rationale"]

        months_out[ym] = {
            "month_top": month_top,
            "month_rationale": month_rationale,
            "day_count": len(days),
            "days": {d: day_rollups[d] for d in days},
        }

        # Year row: take month_top[0] as the month's headline.
        if month_top:
            head = month_top[0]
            year_top.append({
                "month": ym,
                "title": head.get("title", ""),
                "description": head.get("description", ""),
                "origin": head.get("origin", ""),
            })

    payload = {
        "generated_at": int(time.time()),
        "model": MODEL,
        "top_n": top,
        "year_top": year_top,
        "months": months_out,
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    size_kb = out_path.stat().st_size / 1024
    print(f"\n[ok] wrote {out_path} ({size_kb:.1f} KB)")
    print(f"  months: {len(months_out)}")
    print(f"  year_top: {len(year_top)}")
    print(f"  year_top headlines:")
    for ev in year_top:
        print(f"    {ev['month']}  {ev['title']:25s}  ({ev.get('origin','')})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--journal", default="/data/solstone/journal", type=Path)
    ap.add_argument("--top", type=int, default=4, help="top-N per month")
    ap.add_argument("--jobs", type=int, default=5, help="parallel rollup calls")
    ap.add_argument("--dry-run", action="store_true",
                    help="list months + day_top candidates; don't call gemini")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing journal/timeline.json")
    ap.add_argument("--months", default="",
                    help="optional comma-sep YYYYMM filter (e.g., 202604,202605)")
    args = ap.parse_args()

    months_filter = (
        {s.strip() for s in args.months.split(",") if s.strip()}
        if args.months else None
    )

    bootstrap_solstone(args.journal)
    return asyncio.run(rollup_master(
        args.journal, args.top, args.jobs, args.dry_run, args.force, months_filter,
    ))


if __name__ == "__main__":
    sys.exit(main())
