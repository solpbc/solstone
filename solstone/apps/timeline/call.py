# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI commands for timeline rollups."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import typer

from solstone.apps.timeline.rollup import (
    MODEL,
    pick_top_events_async,
    pick_top_events_batch,
)
from solstone.think.utils import get_journal, get_owner_timezone, require_solstone

logger = logging.getLogger(__name__)

app = typer.Typer(help="Timeline rollup tools.")

SEGMENT_RE = re.compile(r"^(\d{2})(\d{2})(\d{2})_\d{1,6}$")
DAY_RE = re.compile(r"^\d{8}$")


@app.callback()
def _require_up() -> None:
    require_solstone()


def _default_day() -> str:
    return (datetime.now(get_owner_timezone()) - timedelta(days=1)).strftime("%Y%m%d")


def _parse_day(value: str) -> str:
    if not DAY_RE.fullmatch(value):
        raise typer.BadParameter("day must be YYYYMMDD")
    return value


def _atomic_write_json(
    path: Path, payload: dict[str, Any], *, indent: int | None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_name = handle.name
            handle.write(json.dumps(payload, ensure_ascii=False, indent=indent) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        if tmp_name:
            Path(tmp_name).unlink(missing_ok=True)
        raise


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
        if tl.parent == day_dir:
            continue
        seg = None
        for ancestor in tl.parents:
            if SEGMENT_RE.match(ancestor.name):
                seg = ancestor
                break
        if not seg or seg in seen_segs:
            continue
        seen_segs.add(seg)
        m = SEGMENT_RE.match(seg.name)
        if not m:
            continue
        hh = m.group(1)
        try:
            data = json.loads(tl.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        title = data.get("title", "").strip()
        desc = data.get("description", "").strip()
        if not (title or desc):
            continue
        rows.append(
            {
                "segment": seg.name,
                "hour": hh,
                "title": title,
                "description": desc,
                "origin": data.get("origin") or origin_for_segment(seg),
                "timeline_path": tl,
            }
        )
    rows.sort(key=lambda r: r["segment"])
    return rows


def group_by_hour(segments: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for segment in segments:
        out.setdefault(segment["hour"], []).append(segment)
    return out


def _show_day_dry_run(day: str, segments: list[dict]) -> None:
    by_hour = group_by_hour(segments)
    typer.echo(f"\n== {day} ==  segments: {len(segments)}  hours: {len(by_hour)}")
    for hh in sorted(by_hour):
        rows = by_hour[hh]
        typer.echo(f"  {hh}h  ({len(rows)} segs)")
        for row in rows[:6]:
            typer.echo(
                f"      {row['segment']:20s}  {row['title']}  -  "
                f"{row['description'][:70]}"
            )
        if len(rows) > 6:
            typer.echo(f"      ... +{len(rows) - 6} more")


async def _rollup_day(
    journal: Path,
    day: str,
    top: int,
    jobs: int,
    dry_run: bool,
    force: bool,
) -> dict | None:
    out_path = journal / "chronicle" / day / "timeline.json"
    if out_path.exists() and not force and not dry_run:
        typer.echo(
            f"  [skip] {day}: timeline.json already exists (use --force to overwrite)"
        )
        return None

    segments = load_day_segments(journal, day)
    if not segments:
        typer.echo(f"  [empty] {day}: no segment timeline.json found")
        return None

    by_hour = group_by_hour(segments)

    if dry_run:
        _show_day_dry_run(day, segments)
        return None

    jobs_in = []
    for hh in sorted(by_hour):
        events = [
            {
                "title": row["title"],
                "description": row["description"],
                "origin": row["origin"],
            }
            for row in by_hour[hh]
        ]
        jobs_in.append({"key": hh, "events": events})

    t0 = time.time()
    typer.echo(
        f"  rolling up {day}: {len(segments)} segs across {len(by_hour)} hours, "
        f"top={top}, jobs={jobs}"
    )
    hour_results = await pick_top_events_batch(
        jobs=jobs_in,
        n=top,
        scope_label="hour",
        max_concurrent=jobs,
    )
    t_hr = time.time() - t0

    hours_out: dict[str, dict] = {}
    hour_picks_flat: list[dict] = []
    for rec in hour_results:
        hh = rec["key"]
        result = rec["result"]
        if "error" in result:
            logger.warning(
                "timeline day %s hour %s rollup failed: %s", day, hh, result["error"]
            )
            typer.echo(f"    [hour-err {hh}h] {result['error'][:120]}")
            hours_out[hh] = {
                "segment_count": len(rec["events"]),
                "picks": [],
                "rationale": "",
                "error": result["error"],
            }
            continue
        hours_out[hh] = {
            "segment_count": len(rec["events"]),
            "picks": result["picks"],
            "rationale": result["rationale"],
        }
        hour_picks_flat.extend(result["picks"])

    if not hour_picks_flat:
        day_top, day_rationale = [], "no hour picks available"
    elif len(hour_picks_flat) <= top:
        day_top = hour_picks_flat
        day_rationale = "fewer than N hour-picks; returning all"
    else:
        try:
            day_result = await pick_top_events_async(
                events=hour_picks_flat,
                n=top,
                scope_label="day",
            )
        except Exception as exc:
            t_total = time.time() - t0
            logger.warning(
                "timeline day %s day-level rollup failed in %.1fs: %s",
                day,
                t_total,
                exc,
            )
            typer.echo(
                f"  [day-err {day}] day-level rollup failed in "
                f"{t_total:.1f}s: {str(exc)[:160]}"
            )
            typer.echo(
                f"  [day-err {day}] no timeline.json written; re-run will retry this day"
            )
            return None
        day_top = day_result["picks"]
        day_rationale = day_result["rationale"]

    t_total = time.time() - t0
    payload = {
        "day": day,
        "model": MODEL,
        "generated_at": int(time.time()),
        "segment_count": len(segments),
        "hour_count": len(hours_out),
        "day_top": day_top,
        "day_rationale": day_rationale,
        "hours": hours_out,
    }

    _atomic_write_json(out_path, payload, indent=2)
    typer.echo(
        f"  [ok {day}] hour-rollup {t_hr:.1f}s  total {t_total:.1f}s  → {out_path}"
    )
    typer.echo(f"      day_top: {[p['title'] for p in day_top]}")
    return payload


def load_day_rollups(journal: Path) -> dict[str, dict]:
    """Map day (YYYYMMDD) → parsed day-level timeline.json (rollup)."""
    chronicle = journal / "chronicle"
    if not chronicle.is_dir():
        return {}
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


def _show_master_dry_run(
    day_rollups: dict[str, dict], by_month: dict[str, list[str]]
) -> None:
    typer.echo("\n== dry run ==")
    typer.echo(f"days with day-level timeline.json: {len(day_rollups)}")
    typer.echo(f"months covered                   : {len(by_month)}")
    typer.echo()
    for ym in sorted(by_month):
        days = by_month[ym]
        cands = sum(len(day_rollups[d].get("day_top") or []) for d in days)
        typer.echo(f"  {ym}  {len(days):3d} days, {cands:4d} day_top candidates")
        for day in days[:3]:
            top0 = day_rollups[day]["day_top"][0]
            typer.echo(f"      {day}  {top0.get('title', '')}")
        if len(days) > 3:
            typer.echo(f"      ... +{len(days) - 3} more")
    typer.echo()


async def _rollup_master(
    journal: Path,
    top: int,
    jobs: int,
    dry_run: bool,
    force: bool,
    months_filter: set[str] | None,
) -> int:
    out_path = journal / "timeline.json"
    if out_path.exists() and not force and not dry_run:
        typer.echo(f"  [skip] {out_path}: already exists (use --force to overwrite)")
        return 0

    day_rollups = load_day_rollups(journal)
    if not day_rollups:
        typer.echo(
            f"  [empty] no day-level timeline.json found under {journal}/chronicle/*/"
        )
        return 0

    by_month = group_by_month(day_rollups)
    if months_filter:
        by_month = {ym: ds for ym, ds in by_month.items() if ym in months_filter}
        if not by_month:
            typer.echo(
                f"  [empty] no overlap between --months {sorted(months_filter)} and journal"
            )
            return 0

    if dry_run:
        _show_master_dry_run(day_rollups, by_month)
        return 0

    jobs_in = []
    for ym in sorted(by_month):
        events = []
        for day in by_month[ym]:
            for ev in day_rollups[day].get("day_top") or []:
                events.append(
                    {
                        "title": ev.get("title", ""),
                        "description": ev.get("description", ""),
                        "origin": ev.get("origin", ""),
                    }
                )
        if events:
            jobs_in.append({"key": ym, "events": events})

    if not jobs_in:
        typer.echo("  [empty] no month candidates found")
        return 0

    typer.echo(
        f"rolling up {len(jobs_in)} month(s) with model={MODEL} top={top} jobs={jobs}"
    )
    t0 = time.time()
    try:
        month_results = await pick_top_events_batch(
            jobs=jobs_in,
            n=top,
            scope_label="month",
            max_concurrent=jobs,
        )
    except Exception as exc:
        logger.warning("timeline month-level rollup failed: %s", exc)
        month_results = [
            {"key": job["key"], "events": job["events"], "result": {"error": str(exc)}}
            for job in jobs_in
        ]
    t_total = time.time() - t0
    typer.echo(f"month-rollup done in {t_total:.1f}s")

    months_out: dict[str, dict] = {}
    year_top: list[dict] = []
    for rec in month_results:
        ym = rec["key"]
        result = rec["result"]
        days = by_month[ym]

        if "error" in result:
            logger.warning("timeline month %s rollup failed: %s", ym, result["error"])
            typer.echo(f"  [month-err {ym}] {result['error'][:120]}")
            month_top = []
            month_rationale = f"ERROR: {result['error'][:200]}"
        else:
            month_top = result["picks"]
            month_rationale = result["rationale"]

        months_out[ym] = {
            "month_top": month_top,
            "month_rationale": month_rationale,
            "day_count": len(days),
            "days": {day: day_rollups[day] for day in days},
        }

        if month_top:
            head = month_top[0]
            year_top.append(
                {
                    "month": ym,
                    "title": head.get("title", ""),
                    "description": head.get("description", ""),
                    "origin": head.get("origin", ""),
                }
            )

    payload = {
        "generated_at": int(time.time()),
        "model": MODEL,
        "top_n": top,
        "year_top": year_top,
        "months": months_out,
    }

    _atomic_write_json(out_path, payload, indent=2)
    size_kb = out_path.stat().st_size / 1024
    typer.echo(f"\n[ok] wrote {out_path} ({size_kb:.1f} KB)")
    typer.echo(f"  months: {len(months_out)}")
    typer.echo(f"  year_top: {len(year_top)}")
    typer.echo("  year_top headlines:")
    for ev in year_top:
        typer.echo(f"    {ev['month']}  {ev['title']:25s}  ({ev.get('origin', '')})")
    return 0


@app.command("rollup-day")
def rollup_day(
    day: str | None = typer.Argument(
        None, metavar="DAY", help="Day to roll up (YYYYMMDD)."
    ),
    top: int = typer.Option(4, "--top", help="Top-N per hour and per day."),
    force: bool = typer.Option(
        False, "--force", help="Overwrite existing day timeline."
    ),
    jobs: int = typer.Option(5, "--jobs", help="Parallel hour-rollup calls."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="List work without calling Gemini."
    ),
) -> None:
    """Roll segment timelines up into one day timeline."""
    resolved_day = _parse_day(day or _default_day())
    asyncio.run(
        _rollup_day(
            journal=Path(get_journal()),
            day=resolved_day,
            top=top,
            jobs=jobs,
            dry_run=dry_run,
            force=force,
        )
    )


@app.command("rollup-master")
def rollup_master(
    top: int = typer.Option(4, "--top", help="Top-N per month."),
    force: bool = typer.Option(
        False, "--force", help="Overwrite existing master timeline."
    ),
    jobs: int = typer.Option(5, "--jobs", help="Parallel month-rollup calls."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="List work without calling Gemini."
    ),
    months: str = typer.Option(
        "",
        "--months",
        help="Optional comma-separated YYYYMM filter, empty means all.",
    ),
) -> None:
    """Roll day timelines up into the journal master timeline."""
    months_filter = {item.strip() for item in months.split(",") if item.strip()} or None
    if months_filter:
        invalid = sorted(
            item for item in months_filter if not re.fullmatch(r"\d{6}", item)
        )
        if invalid:
            raise typer.BadParameter(
                f"--months must be comma-separated YYYYMM values: {invalid}"
            )
    asyncio.run(
        _rollup_master(
            journal=Path(get_journal()),
            top=top,
            jobs=jobs,
            dry_run=dry_run,
            force=force,
            months_filter=months_filter,
        )
    )
