#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc
"""
Per-segment timeline.json generator (scratch, NOT shipped).

For each segment in the given day(s) that has a `talents/activity.md` summary
and does NOT already have a `timeline.json`, ask gemini-3.1-flash-lite-preview
for a small `{title, description}` JSON capturing the most noteworthy thing in
that segment. Writes the JSON to `<segment>/timeline.json`.

Concurrency: think.batch.Batch with max_concurrent=10. Schema-constrained
output via the google provider's structured-output support. Uses solstone's
own auth (loads journal.json env block into process env).

Usage:
    timeline_generate.py 20260409                # one day
    timeline_generate.py 20260409 20260410       # range (inclusive)
    timeline_generate.py 20260409 --dry-run      # show work, don't call gemini
    timeline_generate.py 20260409 --force        # overwrite existing timeline.json
    timeline_generate.py 20260409 --jobs 5
    timeline_generate.py 20260409 --journal /data/solstone/journal

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
from datetime import date, datetime, timedelta
from pathlib import Path

# Wire up solstone imports + load API keys from journal.json env block.
SOLSTONE_REPO = Path("/home/jer/projects/solstone")
sys.path.insert(0, str(SOLSTONE_REPO))


def bootstrap_solstone(journal_path: Path) -> None:
    os.environ.setdefault("SOLSTONE_JOURNAL", str(journal_path))
    from think.utils import get_config
    for k, v in get_config().get("env", {}).items():
        os.environ[k] = str(v)


SEGMENT_RE = re.compile(r"^\d{6}_\d{1,6}$")  # HHMMSS_LEN (LEN can be up to 6 digits for long captures)


def origin_for_segment(seg_dir):
    """Composite ID '<day>[/<stream>]/<seg>' relative to the chronicle root."""
    parts = seg_dir.parts
    try:
        ci = parts.index("chronicle")
    except ValueError:
        return seg_dir.name
    return "/".join(parts[ci + 1 :])

MODEL = "gemini-3.1-flash-lite"

TIMELINE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": (
                "MAX 3 words, MAX 22 characters total, headline case. "
                "Common shorthand encouraged: Dev, Env, Config, UI, App, "
                "Repo, PR, Bug, Cli, Doc, Auth, K8s, KDE. "
                "Examples: 'Sprint Planning', 'Roadmap Lock', 'Display Reset', "
                "'KDE Config', 'Dev Env Debug'."
            ),
        },
        "description": {
            "type": "string",
            "description": (
                "ONE sentence, MAX 10 words, MAX 60 characters, third person, "
                "present-tense, verb-led. Examples: "
                "'Maps the app surface and names rebuild priorities.' "
                "'Restarts display manager to recover the desktop session.' "
                "'Fixes panel placement and terminal artifacts in Plasma.' "
                "No first-person. No times. No segment IDs."
            ),
        },
    },
    "required": ["title", "description"],
}

SYSTEM_INSTRUCTION = (
    "Pick the SINGLE MOST IMPORTANT EVENT from this ~5-minute slice of a "
    "personal life-journal and name it. The output is one cell in a multi-scale "
    "timeline UI — each cell shows a 2-line title and a 3-line description, so "
    "brevity matters more than completeness.\n"
    "\n"
    "An EVENT is a discrete thing that happened: a decision made, a problem "
    "solved, a message sent, a system change applied, a person met, a file "
    "shipped, a milestone reached. NOT a topic area, NOT a feeling, NOT a "
    "generic activity descriptor.\n"
    "\n"
    "Anti-patterns to avoid:\n"
    "  BAD: 'Coding Session' (topic, not event)\n"
    "  BAD: 'Working on KDE' (activity descriptor)\n"
    "  BAD: 'System Maintenance' (generic)\n"
    "  GOOD: 'GDM Service Restart' (specific event)\n"
    "  GOOD: 'Trademark Filed' (discrete action)\n"
    "  GOOD: 'Crash Diagnosed' (concrete result)\n"
    "\n"
    "If multiple noteworthy events occurred, pick the one with the highest "
    "consequence — a decision over a routine action, a fix over an "
    "investigation, a shipped artifact over a draft.\n"
    "\n"
    "FIELD RULES (hard caps):\n"
    "- title: max 3 words, max 22 characters, headline case. Name the EVENT "
    "as a noun phrase or past-tense action. Shorthand is encouraged: Dev, "
    "Env, Config, UI, App, Repo, PR, Bug, Cli, Doc, Auth, K8s, KDE, GDM, "
    "Wallet, Plasma, GNOME. Drop articles. Prefer specific over generic "
    "('KDE Panel Fix' beats 'System Config'). Examples: 'Display Reset', "
    "'Trademark Filed', 'Dev Env Debug', 'Sprint Planned', 'Crash Triage'.\n"
    "- description: max 10 words, max 60 characters, ONE sentence, third "
    "person, present tense, verb-led, describing what happened in/around "
    "the event. Examples: 'Restarts display manager to recover desktop "
    "session.' (51c) 'Files trademark application with specimen images.' "
    "(49c) 'Identifies GNOME DBus dependency causing the crash.' (52c). "
    "No first-person ('I', 'me'). No times ('17:02'). No segment IDs.\n"
    "\n"
    "If the input is empty or trivial, still return a plausible compact "
    "{title, description} for whatever did happen."
)


def parse_day(s: str) -> str:
    if not re.fullmatch(r"\d{8}", s):
        sys.exit(f"day must be YYYYMMDD, got {s!r}")
    return s


def day_range(start: str, end: str) -> list[str]:
    d0 = datetime.strptime(start, "%Y%m%d").date()
    d1 = datetime.strptime(end, "%Y%m%d").date()
    if d1 < d0:
        sys.exit(f"end date {end} before start date {start}")
    out = []
    cur = d0
    while cur <= d1:
        out.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return out


def find_segments(journal: Path, day: str):
    """Yield (segment_dir, activity_md_path, timeline_json_path).

    Handles three on-disk layouts seen in the journal:
      - chronicle/<day>/<segment>/activity.md                 (oldest)
      - chronicle/<day>/<stream>/<segment>/activity.md        (mid)
      - chronicle/<day>/<stream>/<segment>/talents/activity.md (current)
    The "segment dir" is the closest ancestor matching HHMMSS_LEN.
    timeline.json always lands at <segment_dir>/timeline.json.
    """
    day_dir = journal / "chronicle" / day
    if not day_dir.is_dir():
        return
    seen: set[Path] = set()
    for activity in sorted(day_dir.rglob("activity.md")):
        seg = None
        for ancestor in activity.parents:
            if SEGMENT_RE.match(ancestor.name):
                seg = ancestor
                break
        if not seg or seg in seen:
            continue
        seen.add(seg)
        yield seg, activity, seg / "timeline.json"


def build_user_prompt(activity_text: str, segment_path: Path) -> str:
    # The segment_path is included for the model only as silent context; the
    # prompt explicitly forbids using time/segment information in the output.
    rel = segment_path.relative_to(segment_path.parents[3])  # chronicle/<day>/<stream>/<seg>
    return (
        f"Segment: {rel}\n\n"
        f"Activity summary for this slice:\n---\n{activity_text.strip()}\n---\n\n"
        "Return JSON {title, description} per the system instruction."
    )


def list_work(journal: Path, days: list[str], force: bool):
    rows = []
    for day in days:
        for seg, activity, timeline in find_segments(journal, day):
            already = timeline.is_file()
            rows.append({
                "day": day,
                "stream": seg.parent.name,
                "segment": seg.name,
                "segment_path": seg,
                "activity_path": activity,
                "timeline_path": timeline,
                "already_processed": already,
                "to_run": (not already) or force,
            })
    return rows


def print_dry_run(rows):
    by_day = {}
    for r in rows:
        by_day.setdefault(r["day"], []).append(r)
    total_segs = len(rows)
    total_already = sum(1 for r in rows if r["already_processed"])
    total_run = sum(1 for r in rows if r["to_run"])
    print(f"\nDry run: {total_segs} segments with activity.md across "
          f"{len(by_day)} day(s)")
    print(f"  already processed (timeline.json present): {total_already}")
    print(f"  to run                                  : {total_run}\n")
    for day, day_rows in by_day.items():
        print(f"== {day} ({len(day_rows)} segments) ==")
        for r in day_rows:
            mark = "OK " if r["already_processed"] else "..."
            run = "[run]" if r["to_run"] and not r["already_processed"] else (
                "[force]" if r["to_run"] and r["already_processed"] else "[skip]"
            )
            print(f"  {mark} {run:8s} {r['stream']}/{r['segment']}")
        print()


async def run_batch(rows, max_concurrent: int):
    from think.batch import Batch

    work = [r for r in rows if r["to_run"]]
    if not work:
        print("Nothing to do (all already processed; pass --force to overwrite).")
        return

    print(f"Submitting {len(work)} request(s) at concurrency={max_concurrent} "
          f"using model={MODEL} ...")
    batch = Batch(max_concurrent=max_concurrent)

    for r in work:
        try:
            text = r["activity_path"].read_text(encoding="utf-8")
        except OSError as e:
            r["read_error"] = str(e)
            continue

        req = batch.create(
            contents=build_user_prompt(text, r["segment_path"]),
            context="timeline.scratch.summarize",
            model=MODEL,
            system_instruction=SYSTEM_INSTRUCTION,
            json_output=True,
            json_schema=TIMELINE_SCHEMA,
            temperature=0.4,
            max_output_tokens=512,
            timeout_s=60.0,
        )
        req.row = r
        batch.add(req)

    started = time.time()
    done = 0
    failures = 0
    async for req in batch.drain_batch():
        r = req.row
        if req.error:
            failures += 1
            print(f"  [err  {req.duration:5.2f}s] {r['stream']}/{r['segment']}: "
                  f"{req.error[:120]}")
            continue
        # Parse + write
        try:
            payload = json.loads(req.response)
        except (json.JSONDecodeError, TypeError) as e:
            failures += 1
            print(f"  [parse {req.duration:5.2f}s] {r['stream']}/{r['segment']}: "
                  f"{e} | response={req.response[:120]!r}")
            continue
        title = payload.get("title", "")
        desc = payload.get("description", "")
        # Origin: stable composite ID for this segment, lossless through any
        # rollup. Format: "<day>/<stream>/<seg>" (or "<day>/<seg>" for
        # old-layout segments without a stream subdir).
        origin = origin_for_segment(r["segment_path"])
        out = {"title": title, "description": desc, "origin": origin,
               "model": MODEL, "generated_at": int(time.time())}
        try:
            r["timeline_path"].write_text(
                json.dumps(out, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        except OSError as e:
            failures += 1
            print(f"  [write {req.duration:5.2f}s] {r['stream']}/{r['segment']}: {e}")
            continue
        done += 1
        print(f"  [ok   {req.duration:5.2f}s] {r['stream']}/{r['segment']}: "
              f"{title} — {desc[:60]}")

    elapsed = time.time() - started
    print(f"\nDone: wrote {done}, failed {failures}, in {elapsed:.1f}s "
          f"({len(work)/elapsed:.2f} req/s).")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("days", nargs="+", help="YYYYMMDD; one day or two for an inclusive range")
    ap.add_argument("--journal", default="/data/solstone/journal", type=Path)
    ap.add_argument("--jobs", default=10, type=int, help="max concurrent gemini calls")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be processed, don't call gemini")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing timeline.json files")
    args = ap.parse_args()

    if len(args.days) == 1:
        days = [parse_day(args.days[0])]
    elif len(args.days) == 2:
        days = day_range(parse_day(args.days[0]), parse_day(args.days[1]))
    else:
        sys.exit("pass one day, or start + end (inclusive)")

    bootstrap_solstone(args.journal)

    rows = list_work(args.journal, days, force=args.force)
    if not rows:
        print(f"No segments with talents/activity.md found in {days}")
        return 0

    print_dry_run(rows)
    if args.dry_run:
        return 0

    asyncio.run(run_batch(rows, max_concurrent=args.jobs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
