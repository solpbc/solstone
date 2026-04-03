#!/usr/bin/env python3
"""
Sense A/B Test Harness

Runs the unified Sense agent on field journal segments and compares output
against the existing multi-agent pipeline baseline.

Usage:
    python harness.py [--journal PATH] [--model MODEL] [--max-segments N]
                      [--segment DAY/STREAM/SEGMENT] [--output-dir PATH]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

from compare import (
    compare_activity_summary,
    compare_density,
    compare_entities,
    compare_facets,
    compare_meeting_detection,
    compare_speakers,
)
from state_machine import ActivityStateMachine, compare_state_machine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_JOURNAL = Path("/home/jer/projects/field_journal/journal")
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results"
MANIFEST_PATH = Path("/home/jer/projects/field_journal/manifest.json")
SENSE_MD_PATH = Path("/home/jer/projects/solstone/muse/sense.md")
CONFIGURED_FACETS = ["meetings", "learning"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_manifest() -> list[dict]:
    """Load segment manifest from field journal."""
    with open(MANIFEST_PATH) as f:
        data = json.load(f)
    return data.get("segments", [])


def load_sense_instruction() -> str:
    """
    Load Sense system prompt from sense.md.
    Strip the JSON frontmatter — everything before and including the first
    blank line after the closing `}` on its own line.
    """
    text = SENSE_MD_PATH.read_text()
    lines = text.split("\n")

    # Find the closing `}` on its own line (the frontmatter end)
    end_idx = 0
    for i, line in enumerate(lines):
        if line.strip() == "}":
            end_idx = i
            break

    # Skip past the closing `}` and any immediately following blank line
    start = end_idx + 1
    while start < len(lines) and lines[start].strip() == "":
        start += 1

    return "\n".join(lines[start:]).strip()


def read_audio_transcript(segment_path: Path) -> str | None:
    """Read and concatenate transcript lines from audio.jsonl."""
    audio_file = segment_path / "audio.jsonl"
    if not audio_file.exists():
        return None

    lines = []
    with open(audio_file) as f:
        for i, raw_line in enumerate(f):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                entry = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            # First line is metadata (has "raw" or "backend" key), skip it
            if i == 0 and ("raw" in entry or "backend" in entry):
                continue
            text = entry.get("text", "")
            start = entry.get("start", "")
            if text:
                lines.append(f"[{start}] {text}" if start else text)

    return "\n".join(lines) if lines else None


def read_screen_descriptions(segment_path: Path) -> list[str] | None:
    """Read unique visual descriptions from screen.jsonl."""
    screen_file = segment_path / "screen.jsonl"
    if not screen_file.exists():
        return None

    descriptions = []
    seen = set()

    with open(screen_file) as f:
        for i, raw_line in enumerate(f):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                entry = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            # First line is metadata (has "raw" key), skip it
            if i == 0 and "raw" in entry and "analysis" not in entry:
                continue
            analysis = entry.get("analysis", {})
            desc = analysis.get("visual_description", "")
            if desc and desc not in seen:
                seen.add(desc)
                descriptions.append(desc)

    return descriptions if descriptions else None


def read_baseline_activity(segment_path: Path) -> str | None:
    """Read baseline activity.md."""
    p = segment_path / "agents" / "activity.md"
    if p.exists():
        return p.read_text().strip()
    return None


def read_baseline_entities(segment_path: Path) -> list[dict]:
    """Read baseline entities.jsonl."""
    p = segment_path / "agents" / "entities.jsonl"
    if not p.exists():
        return []
    entities = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entities.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entities


def read_baseline_speakers(segment_path: Path) -> list[str] | None:
    """Read baseline speakers.json. Returns None if file doesn't exist."""
    p = segment_path / "agents" / "speakers.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        return None


def read_baseline_facets(segment_path: Path) -> list[dict]:
    """Read baseline facets.json."""
    p = segment_path / "agents" / "facets.json"
    if not p.exists():
        return []
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        return []


def read_baseline_activity_state(segment_path: Path) -> list[dict]:
    """
    Read baseline activity_state.json from facet subdirectories.
    Checks each known facet subdir under agents/.
    """
    agents_dir = segment_path / "agents"
    if not agents_dir.exists():
        return []

    all_states = []
    for subdir in agents_dir.iterdir():
        if subdir.is_dir():
            state_file = subdir / "activity_state.json"
            if state_file.exists():
                try:
                    with open(state_file) as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        all_states.extend(data)
                    elif isinstance(data, dict):
                        all_states.append(data)
                except (json.JSONDecodeError, ValueError):
                    continue
    return all_states


def segment_time_range(segment_key: str) -> tuple[str, str]:
    """
    Parse segment key like '091500_420' into start and end time strings.
    Returns (start, end) as "HH:MM:SS" strings.
    """
    parts = segment_key.split("_")
    if len(parts) != 2:
        return (segment_key, segment_key)

    time_str = parts[0]
    duration = int(parts[1])

    h = int(time_str[0:2])
    m = int(time_str[2:4])
    s = int(time_str[4:6])

    start = f"{h:02d}:{m:02d}:{s:02d}"

    total_seconds = h * 3600 + m * 60 + s + duration
    eh = total_seconds // 3600
    em = (total_seconds % 3600) // 60
    es = total_seconds % 60
    end = f"{eh:02d}:{em:02d}:{es:02d}"

    return (start, end)


def compose_user_message(day: str, segment_key: str,
                         transcript: str | None,
                         screen_descriptions: list[str] | None) -> str:
    """Assemble the user message for the Sense prompt."""
    start, end = segment_time_range(segment_key)
    parts = [f"Analyzing segment from {day} covering {start} to {end}."]

    if transcript:
        parts.append(f"\n## Transcript\n\n{transcript}")

    if screen_descriptions:
        parts.append(f"\n## Screen Activity\n\n" + "\n".join(screen_descriptions))

    parts.append(
        f"\n## Configured Facets\n\n"
        f"- meetings\n"
        f"- learning"
    )

    return "\n".join(parts)


def call_sense(client: OpenAI, model: str, system_prompt: str,
               user_message: str) -> tuple[dict | None, dict]:
    """
    Call the Sense agent via OpenAI API.
    Returns (parsed_response, metadata) where metadata includes tokens and latency.
    """
    t0 = time.monotonic()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
    except Exception as e:
        elapsed = time.monotonic() - t0
        return None, {
            "error": str(e),
            "latency_seconds": round(elapsed, 3),
            "input_tokens": 0,
            "output_tokens": 0,
        }

    elapsed = time.monotonic() - t0

    usage = response.usage
    meta = {
        "latency_seconds": round(elapsed, 3),
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
        "model": response.model,
    }

    content = response.choices[0].message.content if response.choices else ""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        meta["parse_error"] = str(e)
        meta["raw_response"] = content[:2000]
        parsed = None

    return parsed, meta


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------

def run_segment(client: OpenAI, model: str, system_prompt: str,
                segment_info: dict, journal_root: Path,
                state_machine: ActivityStateMachine,
                prev_segment_key: str | None) -> dict:
    """Run Sense on a single segment and compare against baseline."""
    day = segment_info["day"]
    stream = segment_info["stream"]
    segment_key = segment_info["segment"]

    segment_path = journal_root / day / stream / segment_key
    segment_id = f"{day}/{stream}/{segment_key}"

    print(f"  [{segment_id}] ", end="", flush=True)

    if not segment_path.exists():
        print("SKIP (path missing)")
        return {"segment": segment_id, "status": "skipped", "reason": "path_missing"}

    # Read inputs
    transcript = read_audio_transcript(segment_path)
    screen_descs = read_screen_descriptions(segment_path)

    if not transcript and not screen_descs:
        print("SKIP (no input data)")
        return {"segment": segment_id, "status": "skipped", "reason": "no_input"}

    # Read baselines
    baseline_activity = read_baseline_activity(segment_path)
    baseline_entities = read_baseline_entities(segment_path)
    baseline_speakers = read_baseline_speakers(segment_path)
    baseline_facets = read_baseline_facets(segment_path)
    baseline_states = read_baseline_activity_state(segment_path)

    # Compose and send
    user_msg = compose_user_message(day, segment_key, transcript, screen_descs)
    sense_output, api_meta = call_sense(client, model, system_prompt, user_msg)

    if sense_output is None:
        print(f"FAIL ({api_meta.get('error', api_meta.get('parse_error', 'unknown'))})")
        return {
            "segment": segment_id,
            "status": "error",
            "api": api_meta,
        }

    # Run state machine
    sm_changes = state_machine.update(sense_output, segment_key, day, prev_segment_key)
    sm_current = state_machine.get_current_state()

    # Compare all fields
    comparisons = {}

    # Density — all field journal segments have full agent outputs = "active"
    comparisons["density"] = compare_density(
        sense_output.get("density", ""),
        "active"
    )

    # Entities
    comparisons["entities"] = compare_entities(
        sense_output.get("entities", []),
        baseline_entities
    )

    # Speakers
    comparisons["speakers"] = compare_speakers(
        sense_output.get("speakers", []),
        baseline_speakers or []
    )

    # Facets
    comparisons["facets"] = compare_facets(
        sense_output.get("facets", []),
        baseline_facets
    )

    # Activity summary
    comparisons["activity_summary"] = compare_activity_summary(
        sense_output.get("activity_summary", ""),
        baseline_activity or ""
    )

    # Meeting detection
    comparisons["meeting_detection"] = compare_meeting_detection(
        sense_output.get("meeting_detected", False),
        baseline_speakers
    )

    # State machine comparison
    comparisons["state_machine"] = compare_state_machine(sm_current, baseline_states)

    # Score summary
    score = _compute_score(comparisons)

    print(f"OK (score={score:.2f}, "
          f"in={api_meta['input_tokens']}, out={api_meta['output_tokens']}, "
          f"{api_meta['latency_seconds']}s)")

    return {
        "segment": segment_id,
        "status": "ok",
        "day": day,
        "stream": stream,
        "segment_key": segment_key,
        "api": api_meta,
        "sense_output": sense_output,
        "comparisons": comparisons,
        "score": score,
    }


def _compute_score(comparisons: dict) -> float:
    """
    Compute a weighted quality score from comparisons.
    Returns 0.0-1.0.
    """
    weights = {
        "density": 0.10,
        "entities": 0.25,
        "speakers": 0.10,
        "facets": 0.20,
        "activity_summary": 0.15,
        "meeting_detection": 0.10,
        "state_machine": 0.10,
    }

    scores = {}

    # Density: binary match
    scores["density"] = 1.0 if comparisons.get("density", {}).get("match") else 0.0

    # Entities: F1
    scores["entities"] = comparisons.get("entities", {}).get("f1", 0.0)

    # Speakers: overlap (Jaccard)
    scores["speakers"] = comparisons.get("speakers", {}).get("overlap", 0.0)

    # Facets: combination of facet match + level closeness
    fc = comparisons.get("facets", {})
    facet_score = 0.0
    if fc.get("facet_match"):
        facet_score += 0.5
    elif fc.get("common_facets", 0) > 0:
        total = fc["common_facets"] + len(fc.get("sense_only_facets", [])) + len(fc.get("baseline_only_facets", []))
        facet_score += 0.5 * (fc["common_facets"] / total) if total > 0 else 0.0
    if fc.get("level_close"):
        facet_score += 0.5
    elif fc.get("common_facets", 0) > 0:
        facet_score += 0.5 * (fc.get("level_close_count", 0) / fc["common_facets"])
    scores["facets"] = facet_score

    # Activity summary: keyword overlap
    scores["activity_summary"] = comparisons.get("activity_summary", {}).get("keyword_overlap", 0.0)

    # Meeting detection: binary match
    scores["meeting_detection"] = 1.0 if comparisons.get("meeting_detection", {}).get("match") else 0.0

    # State machine: activity match + level match ratio
    sm = comparisons.get("state_machine", {})
    sm_score = 0.5 if sm.get("activity_match") else 0.0
    if sm.get("level_match_total", 0) > 0:
        sm_score += 0.5 * (sm.get("level_match_count", 0) / sm["level_match_total"])
    elif sm.get("activity_match"):
        sm_score += 0.5
    scores["state_machine"] = sm_score

    total = sum(scores[k] * weights[k] for k in weights)
    return round(total, 4)


def generate_summary(results: list[dict]) -> str:
    """Generate a human-readable summary report."""
    ok_results = [r for r in results if r.get("status") == "ok"]
    skipped = [r for r in results if r.get("status") == "skipped"]
    errors = [r for r in results if r.get("status") == "error"]

    lines = [
        "# Sense A/B Test Results",
        f"",
        f"**Segments:** {len(results)} total, {len(ok_results)} completed, "
        f"{len(skipped)} skipped, {len(errors)} errors",
        "",
    ]

    if not ok_results:
        lines.append("No completed results to summarize.")
        return "\n".join(lines)

    # Aggregate scores
    scores = [r["score"] for r in ok_results]
    avg_score = sum(scores) / len(scores)
    lines.append(f"**Overall Score:** {avg_score:.4f} (avg across {len(ok_results)} segments)")
    lines.append("")

    # Token usage
    total_input = sum(r["api"]["input_tokens"] for r in ok_results)
    total_output = sum(r["api"]["output_tokens"] for r in ok_results)
    avg_latency = sum(r["api"]["latency_seconds"] for r in ok_results) / len(ok_results)
    lines.append("## Token Usage")
    lines.append(f"- Input tokens: {total_input:,} total, {total_input // len(ok_results):,} avg/segment")
    lines.append(f"- Output tokens: {total_output:,} total, {total_output // len(ok_results):,} avg/segment")
    lines.append(f"- Latency: {avg_latency:.2f}s avg/segment")
    lines.append("")

    # Per-comparison averages
    comparison_keys = ["density", "entities", "speakers", "facets",
                       "activity_summary", "meeting_detection", "state_machine"]
    lines.append("## Comparison Breakdown")
    lines.append("")

    # Density match rate
    density_matches = sum(1 for r in ok_results
                          if r["comparisons"]["density"]["match"])
    lines.append(f"### Density")
    lines.append(f"- Match rate: {density_matches}/{len(ok_results)} "
                 f"({density_matches / len(ok_results):.1%})")
    lines.append("")

    # Entity F1
    entity_f1s = [r["comparisons"]["entities"]["f1"] for r in ok_results]
    entity_precisions = [r["comparisons"]["entities"]["precision"] for r in ok_results]
    entity_recalls = [r["comparisons"]["entities"]["recall"] for r in ok_results]
    lines.append(f"### Entities")
    lines.append(f"- Avg F1: {sum(entity_f1s) / len(entity_f1s):.4f}")
    lines.append(f"- Avg Precision: {sum(entity_precisions) / len(entity_precisions):.4f}")
    lines.append(f"- Avg Recall: {sum(entity_recalls) / len(entity_recalls):.4f}")
    lines.append("")

    # Speakers
    speaker_overlaps = [r["comparisons"]["speakers"]["overlap"] for r in ok_results]
    meeting_segments = [r for r in ok_results if r["comparisons"]["meeting_detection"]["baseline"]]
    lines.append(f"### Speakers")
    lines.append(f"- Avg overlap (Jaccard): {sum(speaker_overlaps) / len(speaker_overlaps):.4f}")
    lines.append(f"- Meeting segments (baseline): {len(meeting_segments)}")
    lines.append("")

    # Facets
    facet_matches = sum(1 for r in ok_results
                        if r["comparisons"]["facets"]["facet_match"])
    level_close_matches = sum(1 for r in ok_results
                              if r["comparisons"]["facets"]["level_close"])
    lines.append(f"### Facets")
    lines.append(f"- Facet ID match rate: {facet_matches}/{len(ok_results)} "
                 f"({facet_matches / len(ok_results):.1%})")
    lines.append(f"- Level within +/-1 tier: {level_close_matches}/{len(ok_results)} "
                 f"({level_close_matches / len(ok_results):.1%})")
    lines.append("")

    # Activity summary
    keyword_overlaps = [r["comparisons"]["activity_summary"]["keyword_overlap"]
                        for r in ok_results]
    lines.append(f"### Activity Summary")
    lines.append(f"- Avg keyword overlap (Jaccard): "
                 f"{sum(keyword_overlaps) / len(keyword_overlaps):.4f}")
    lines.append("")

    # Meeting detection
    meeting_matches = sum(1 for r in ok_results
                          if r["comparisons"]["meeting_detection"]["match"])
    lines.append(f"### Meeting Detection")
    lines.append(f"- Match rate: {meeting_matches}/{len(ok_results)} "
                 f"({meeting_matches / len(ok_results):.1%})")
    lines.append("")

    # Per-segment scores table
    lines.append("## Per-Segment Scores")
    lines.append("")
    lines.append("| Segment | Score | Density | Entity F1 | Speaker | Facet | Meeting |")
    lines.append("|---------|-------|---------|-----------|---------|-------|---------|")
    for r in ok_results:
        c = r["comparisons"]
        lines.append(
            f"| {r['segment']} "
            f"| {r['score']:.3f} "
            f"| {'Y' if c['density']['match'] else 'N'} "
            f"| {c['entities']['f1']:.3f} "
            f"| {c['speakers']['overlap']:.3f} "
            f"| {'Y' if c['facets']['facet_match'] else 'N'} "
            f"| {'Y' if c['meeting_detection']['match'] else 'N'} |"
        )
    lines.append("")

    if errors:
        lines.append("## Errors")
        for r in errors:
            lines.append(f"- {r['segment']}: {r.get('api', {}).get('error', 'unknown')}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Sense A/B test harness — compare unified Sense agent against multi-agent baseline"
    )
    parser.add_argument("--journal", type=Path, default=DEFAULT_JOURNAL,
                        help=f"Path to field journal root (default: {DEFAULT_JOURNAL})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"OpenAI model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--max-segments", type=int, default=None,
                        help="Max segments to process (default: all)")
    parser.add_argument("--segment", type=str, default=None,
                        help="Run a single segment: DAY/STREAM/SEGMENT (e.g. 20260201/field.audio/091500_420)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Output directory for results (default: {DEFAULT_OUTPUT})")

    args = parser.parse_args()

    # Validate
    if not args.journal.exists():
        print(f"Error: journal path not found: {args.journal}")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    # Load manifest and system prompt
    print("Loading manifest and Sense instruction...")
    manifest = load_manifest()
    system_prompt = load_sense_instruction()
    print(f"  {len(manifest)} segments in manifest")
    print(f"  Sense instruction: {len(system_prompt)} chars")

    # Filter segments
    if args.segment:
        parts = args.segment.split("/")
        if len(parts) != 3:
            print(f"Error: --segment must be DAY/STREAM/SEGMENT, got: {args.segment}")
            sys.exit(1)
        target_day, target_stream, target_seg = parts
        manifest = [s for s in manifest
                    if s["day"] == target_day
                    and s["stream"] == target_stream
                    and s["segment"] == target_seg]
        if not manifest:
            print(f"Error: segment not found in manifest: {args.segment}")
            sys.exit(1)

    if args.max_segments:
        manifest = manifest[:args.max_segments]

    print(f"  Running {len(manifest)} segments with model {args.model}")
    print()

    # Setup
    client = OpenAI(api_key=api_key)
    state_machine = ActivityStateMachine()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    prev_segment_key = None

    for i, seg_info in enumerate(manifest):
        print(f"[{i + 1}/{len(manifest)}]", end="")
        result = run_segment(
            client, args.model, system_prompt,
            seg_info, args.journal, state_machine, prev_segment_key
        )
        results.append(result)
        prev_segment_key = seg_info["segment"]

    print()
    print("=" * 60)

    # Write JSONL results
    results_file = args.output_dir / "results.jsonl"
    with open(results_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"Results written to {results_file}")

    # Write summary report
    summary = generate_summary(results)
    summary_file = args.output_dir / "summary.md"
    summary_file.write_text(summary)
    print(f"Summary written to {summary_file}")
    print()
    print(summary)


if __name__ == "__main__":
    main()
