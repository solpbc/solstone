#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc
"""
Tiny stdlib HTTP server for the timeline-app prototype.

Replaces `python3 -m http.server` for the prototype workspace. Serves:

  GET /<static>                              prototype.html, data-mock.js, etc.
  GET /api/index                             year + 12 most-recent real months
  GET /api/day/<YYYYMMDD>                    rollup + per-hour 5-min bucket avail
  GET /api/segment/<day>/<stream>/<seg>      audio + screen jsonl, parsed

Reads from /data/solstone/journal/ directly. Caches the master timeline.json
(mtime-invalidated) and the most recent N segment requests in memory.

Usage:
    /data/solstone/.venv/bin/python /data/solstone/scratch/timeline_server.py [--port 8000]
    python3 /data/solstone/scratch/timeline_server.py [--port 8000]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict, OrderedDict
from datetime import date
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

JOURNAL = Path("/data/solstone/journal")
MASTER = JOURNAL / "timeline.json"
CURATED = JOURNAL / "timeline.curated.json"
PROTOTYPE = Path("/home/jer/projects/extro/vpe/workspace/timeline-app")

DAY_RE = re.compile(r"^\d{8}$")
SEG_RE = re.compile(r"^(\d{2})(\d{2})(\d{2})_\d{1,6}$")  # captures HH MM SS

# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------

_master_cache: dict | None = None
_master_mtime: float = 0.0

_curated_cache: dict | None = None
_curated_mtime: float = 0.0


def load_master() -> dict:
    global _master_cache, _master_mtime
    if not MASTER.exists():
        return {}
    mtime = MASTER.stat().st_mtime
    if _master_cache is None or mtime > _master_mtime:
        _master_cache = json.loads(MASTER.read_text(encoding="utf-8"))
        _master_mtime = mtime
    return _master_cache


def load_curated() -> dict:
    """Load the demo-day curated overlay if present.

    Returns {} if the file doesn't exist (behavior is unchanged for un-curated
    deploys). mtime-cached alongside the master so changes hot-reload.
    """
    global _curated_cache, _curated_mtime
    if not CURATED.exists():
        return {}
    mtime = CURATED.stat().st_mtime
    if _curated_cache is None or mtime > _curated_mtime:
        try:
            _curated_cache = json.loads(CURATED.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            _curated_cache = {}
        _curated_mtime = mtime
    return _curated_cache or {}


_seg_cache: "OrderedDict[str, dict]" = OrderedDict()
_SEG_CACHE_MAX = 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def recent_12_months(today: date | None = None) -> list[str]:
    """12 YYYYMM strings ending at today's calendar month, oldest first."""
    if today is None:
        today = date.today()
    out: list[str] = []
    cur_y, cur_m = today.year, today.month
    for delta in range(11, -1, -1):
        y, m = cur_y, cur_m - delta
        while m <= 0:
            m += 12
            y -= 1
        out.append(f"{y:04d}{m:02d}")
    return out


def days_in_month(ym: str) -> int:
    y = int(ym[:4])
    m = int(ym[4:6])
    nxt = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
    return (nxt - date(y, m, 1)).days


def first_weekday(ym: str) -> int:
    """Return Python weekday (Mon=0..Sun=6) of YYYYMM-01."""
    return date(int(ym[:4]), int(ym[4:6]), 1).weekday()


# ---------------------------------------------------------------------------
# Endpoint builders
# ---------------------------------------------------------------------------

def build_index() -> dict:
    """12 most-recent calendar months + per-month metadata + year_top filtered.

    If a curated overlay exists at /data/solstone/journal/timeline.curated.json,
    its `year_top_overrides` (keyed by ym) replace matching entries in year_top
    and its `month_top_overrides` (keyed by ym) replace per-month month_top.
    """
    master = load_master()
    curated = load_curated()
    yt_over = curated.get("year_top_overrides") or {}
    mt_over = curated.get("month_top_overrides") or {}

    months_data = master.get("months", {})
    yms = recent_12_months()
    ym_set = set(yms)

    months_meta: list[dict] = []
    for ym in yms:
        m = months_data.get(ym, {})
        days_with_data = sorted((m.get("days") or {}).keys())
        month_top = mt_over.get(ym) if ym in mt_over else m.get("month_top", [])
        months_meta.append({
            "ym": ym,
            "year": int(ym[:4]),
            "month_num": int(ym[4:6]),
            "days_in_month": days_in_month(ym),
            "first_weekday": first_weekday(ym),
            "month_top": month_top,
            "month_rationale": m.get("month_rationale", ""),
            "day_count": m.get("day_count", 0),
            "days_with_data": days_with_data,
        })

    # Compose year_top: start from master, replace matches by ym, append overrides
    # for months that didn't appear in master, then filter to the visible window.
    # An override value of None means "strike" — drop that month from year_top.
    base = {e.get("month"): e for e in master.get("year_top", [])}
    for ym, override in yt_over.items():
        if override is None:
            base.pop(ym, None)
        else:
            base[ym] = override
    year_top = [base[ym] for ym in sorted(base) if ym in ym_set]

    return {
        "generated_at": master.get("generated_at"),
        "model": master.get("model"),
        "year_top": year_top,
        "months": months_meta,
    }


def build_day(yyyymmdd: str) -> dict:
    """Day-level rollup + per-hour 5-minute bucket availability.

    If a curated overlay exists, its `day_top_overrides[yyyymmdd]` replaces the
    day's day_top so month/year-view picks navigate to a matching headline.
    """
    master = load_master()
    curated = load_curated()
    dt_over = curated.get("day_top_overrides") or {}
    ym = yyyymmdd[:6]
    months_data = master.get("months", {})
    day_data = (months_data.get(ym, {}).get("days", {}) or {}).get(yyyymmdd, {})
    day_top = dt_over.get(yyyymmdd) if yyyymmdd in dt_over else day_data.get("day_top", [])

    # Walk chronicle/<day>/.../<seg> to learn what's actually on disk.
    day_dir = JOURNAL / "chronicle" / yyyymmdd
    # buckets[hh][bucket_minute] = list of segment-meta dicts
    buckets: dict[int, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    if day_dir.is_dir():
        for path in day_dir.rglob("*"):
            if not path.is_dir():
                continue
            m = SEG_RE.match(path.name)
            if not m:
                continue
            hh = int(m.group(1))
            mm = int(m.group(2))
            bucket = (mm // 5) * 5
            rel_parts = path.relative_to(day_dir).parts
            stream = rel_parts[0] if len(rel_parts) > 1 else ""
            origin = (
                f"{yyyymmdd}/{stream}/{path.name}" if stream else f"{yyyymmdd}/{path.name}"
            )
            has_audio = (path / "audio.jsonl").is_file()
            has_screen = bool(list(path.glob("*screen.jsonl")))
            buckets[hh][bucket].append({
                "origin": origin,
                "stream": stream,
                "has_audio": has_audio,
                "has_screen": has_screen,
            })

    # "Best" = both audio+screen > screen only > audio only > metadata only
    def rank(seg: dict) -> int:
        if seg["has_audio"] and seg["has_screen"]:
            return 0
        if seg["has_screen"]:
            return 1
        if seg["has_audio"]:
            return 2
        return 3

    hours_avail: dict[str, dict] = {}
    for hh in range(24):
        bucket_list = []
        hour_buckets = buckets.get(hh, {})
        for minute in range(0, 60, 5):
            segs = hour_buckets.get(minute, [])
            if segs:
                segs.sort(key=rank)
                best = segs[0]
                bucket_list.append({
                    "minute": minute,
                    "best_origin": best["origin"],
                    "has_audio": best["has_audio"],
                    "has_screen": best["has_screen"],
                    "segment_count": len(segs),
                })
            else:
                bucket_list.append({
                    "minute": minute,
                    "best_origin": None,
                    "has_audio": False,
                    "has_screen": False,
                    "segment_count": 0,
                })
        if any(b["best_origin"] for b in bucket_list):
            hours_avail[f"{hh:02d}"] = {"buckets": bucket_list}

    return {
        "day": yyyymmdd,
        "day_top": day_top,
        "day_rationale": day_data.get("day_rationale", ""),
        "hours": day_data.get("hours", {}),
        "hours_avail": hours_avail,
    }


def load_segment(yyyymmdd: str, stream: str, seg: str) -> dict:
    """Parsed audio.jsonl + screen.jsonl for one segment."""
    key = f"{yyyymmdd}/{stream}/{seg}"
    if key in _seg_cache:
        _seg_cache.move_to_end(key)
        return _seg_cache[key]

    if stream:
        seg_dir = JOURNAL / "chronicle" / yyyymmdd / stream / seg
    else:
        seg_dir = JOURNAL / "chronicle" / yyyymmdd / seg

    out: dict = {
        "day": yyyymmdd,
        "stream": stream,
        "segment": seg,
        "audio": None,
        "screen": None,
    }

    audio_file = seg_dir / "audio.jsonl"
    if audio_file.is_file():
        try:
            text = audio_file.read_text(encoding="utf-8")
            lines = [l for l in text.splitlines() if l.strip()]
            if lines:
                header = json.loads(lines[0])
                items = [json.loads(l) for l in lines[1:]]
                out["audio"] = {"header": header, "lines": items}
        except (OSError, json.JSONDecodeError):
            pass

    screen_files = sorted(seg_dir.glob("*screen.jsonl"))
    if screen_files:
        try:
            text = screen_files[0].read_text(encoding="utf-8")
            lines = [l for l in text.splitlines() if l.strip()]
            if lines:
                header = json.loads(lines[0])
                items = [json.loads(l) for l in lines[1:]]
                out["screen"] = {
                    "header": header,
                    "frames": items,
                    "filename": screen_files[0].name,
                }
        except (OSError, json.JSONDecodeError):
            pass

    if not seg_dir.is_dir():
        out["error"] = f"segment dir not found: {seg_dir}"

    _seg_cache[key] = out
    if len(_seg_cache) > _SEG_CACHE_MAX:
        _seg_cache.popitem(last=False)
    return out


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PROTOTYPE), **kwargs)

    def log_message(self, fmt, *args):  # quieter logging
        sys.stderr.write(f"  {self.command} {self.path} → {args[1] if len(args) > 1 else '?'}\n")

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        try:
            if path == "/api/index":
                return self._json(build_index())
            if path.startswith("/api/day/"):
                ymd = path[len("/api/day/"):]
                if not DAY_RE.fullmatch(ymd):
                    return self._error(400, f"bad day {ymd!r}")
                return self._json(build_day(ymd))
            if path.startswith("/api/segment/"):
                rest = path[len("/api/segment/"):].rstrip("/")
                parts = rest.split("/")
                if len(parts) == 3:
                    return self._json(load_segment(*parts))
                if len(parts) == 2:
                    return self._json(load_segment(parts[0], "", parts[1]))
                return self._error(400, f"bad segment path: {rest!r}")
        except Exception as e:
            return self._error(500, f"{type(e).__name__}: {e}")
        return super().do_GET()

    def _json(self, payload: dict, code: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code: int, msg: str) -> None:
        body = json.dumps({"error": msg}).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    if not PROTOTYPE.is_dir():
        sys.exit(f"prototype dir not found: {PROTOTYPE}")
    if not JOURNAL.is_dir():
        sys.exit(f"journal dir not found: {JOURNAL}")

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"\n  timeline server: http://{args.host}:{args.port}/")
    print(f"  static : {PROTOTYPE}")
    print(f"  journal: {JOURNAL}")
    print(f"  endpoints:")
    print(f"    /api/index")
    print(f"    /api/day/<YYYYMMDD>")
    print(f"    /api/segment/<day>/<stream>/<seg>")
    print()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped.")
        srv.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
