# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI interface for speaker voiceprint management.

Provides:
    sol call speakers status [section]
    sol call speakers bootstrap [--dry-run] [--json]
    sol call speakers resolve-names [--dry-run] [--json]
    sol call speakers attribute-segment <day> <stream> <segment> [--json]
    sol call speakers backfill [--dry-run] [--json]
    sol call speakers discover [--json]
    sol call speakers identify <cluster-id> <name> [--entity-id ID]
    sol call speakers merge-names <alias> <canonical>
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="speakers",
    help="Speaker voiceprint management.",
    no_args_is_help=True,
)


@app.command("status")
def status(
    section: str | None = typer.Argument(
        None,
        help=(
            "Section to show (embeddings, owner, speakers, clusters, imports, "
            "attribution). Omit for all."
        ),
    ),
) -> None:
    """Show speaker subsystem status as JSON."""
    import json as json_mod

    from apps.speakers.status import get_speakers_status

    result = get_speakers_status(section=section)
    typer.echo(json_mod.dumps(result, indent=2, default=str))


@app.command("bootstrap")
def bootstrap(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be saved without saving."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output full result as JSON."
    ),
) -> None:
    """Bootstrap voiceprints from single-speaker segments.

    Scans the full journal for segments where speakers.json lists exactly
    one speaker. In those segments, all non-owner embeddings belong to that
    speaker. Saves them as voiceprints using the owner centroid for
    owner subtraction.
    """
    from apps.speakers.bootstrap import bootstrap_voiceprints

    if dry_run and not json_output:
        typer.echo("DRY RUN — no voiceprints will be saved\n")

    if not json_output:
        typer.echo("Bootstrapping voiceprints from single-speaker segments...")
    stats = bootstrap_voiceprints(dry_run=dry_run)

    if "error" in stats:
        typer.echo(f"Error: {stats['error']}", err=True)
        raise typer.Exit(1)
    if json_output:
        import json as json_mod

        typer.echo(json_mod.dumps(stats, indent=2, default=str))
        return

    typer.echo(f"\nSegments scanned: {stats['segments_scanned']}")
    typer.echo(f"Single-speaker segments: {stats['single_speaker_segments']}")
    typer.echo(f"Unique speakers: {len(stats['speakers_found'])}")
    typer.echo(f"Entities created: {stats['entities_created']}")
    typer.echo(f"Embeddings saved: {stats['embeddings_saved']}")
    typer.echo(f"Embeddings skipped (owner): {stats['embeddings_skipped_owner']}")
    typer.echo(
        f"Embeddings skipped (duplicate): {stats['embeddings_skipped_duplicate']}"
    )

    if stats["speakers_found"]:
        typer.echo("\nTop speakers by embedding count:")
        sorted_speakers = sorted(
            stats["speakers_found"].items(), key=lambda x: x[1], reverse=True
        )
        for name, count in sorted_speakers[:15]:
            typer.echo(f"  {name}: {count}")
        if len(sorted_speakers) > 15:
            typer.echo(f"  ... and {len(sorted_speakers) - 15} more")

    if stats["errors"]:
        typer.echo(f"\nErrors ({len(stats['errors'])}):", err=True)
        for err in stats["errors"]:
            typer.echo(f"  {err}", err=True)


@app.command("resolve-names")
def resolve_names(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show merges without applying them."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output full result as JSON."
    ),
) -> None:
    """Resolve speaker name variants using voiceprint similarity.

    Compares voiceprint centroids between all entities. Pairs with cosine
    similarity > 0.90 are flagged as the same person. Unambiguous variants
    (short name is first word of full name) are auto-merged by adding the
    short name as an aka on the canonical entity.
    """
    from apps.speakers.bootstrap import resolve_name_variants

    if dry_run and not json_output:
        typer.echo("DRY RUN — no merges will be applied\n")

    if not json_output:
        typer.echo("Resolving speaker name variants...")
    stats = resolve_name_variants(dry_run=dry_run)

    if json_output:
        import json as json_mod

        typer.echo(json_mod.dumps(stats, indent=2, default=str))
        return

    typer.echo(f"\nEntities with voiceprints: {stats['entities_with_voiceprints']}")
    typer.echo(f"Pairs compared: {stats['pairs_compared']}")
    typer.echo(f"High-similarity pairs: {len(stats['matches_found'])}")

    if stats["auto_merged"]:
        typer.echo(f"\nAuto-merged ({len(stats['auto_merged'])}):")
        for merge in stats["auto_merged"]:
            typer.echo(
                f"  {merge['alias']} -> {merge['canonical']} ({merge['similarity']})"
            )

    if stats["ambiguous"]:
        typer.echo(f"\nAmbiguous ({len(stats['ambiguous'])}):")
        for amb in stats["ambiguous"]:
            candidates = ", ".join(
                f"{c['name']} ({c['similarity']})" for c in amb["candidates"]
            )
            typer.echo(f"  {amb['name']}: {candidates}")

    if stats["errors"]:
        typer.echo(f"\nErrors ({len(stats['errors'])}):", err=True)
        for err in stats["errors"]:
            typer.echo(f"  {err}", err=True)


@app.command("attribute-segment")
def attribute_segment_cmd(
    day: str = typer.Argument(..., help="Day in YYYYMMDD format."),
    stream: str = typer.Argument(..., help="Stream name."),
    segment: str = typer.Argument(..., help="Segment key (HHMMSS_LEN)."),
    save: bool = typer.Option(
        True, "--save/--no-save", help="Write speaker_labels.json."
    ),
    accumulate: bool = typer.Option(
        True,
        "--accumulate/--no-accumulate",
        help="Run voiceprint accumulation.",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output full result as JSON."
    ),
) -> None:
    """Run speaker attribution (Layers 1-3) on a single segment.

    Classifies each sentence using owner detection, structural heuristics,
    and acoustic voiceprint matching.  Optionally writes speaker_labels.json
    and accumulates high-confidence voiceprints.
    """
    import json as json_mod

    from apps.speakers.attribution import (
        accumulate_voiceprints,
        attribute_segment,
        save_speaker_labels,
    )
    from think.utils import segment_path

    result = attribute_segment(day, stream, segment)

    if result.get("error"):
        typer.echo(f"Error: {result['error']}", err=True)
        raise typer.Exit(1)

    labels = result.get("labels", [])
    unmatched = result.get("unmatched", [])
    source = result.get("source")
    metadata = result.get("metadata", {})

    if json_output:
        typer.echo(json_mod.dumps(result, indent=2))
    else:
        resolved = sum(1 for lab in labels if lab["speaker"] is not None)
        typer.echo(f"Sentences: {len(labels)}")
        typer.echo(f"Resolved:  {resolved}")
        typer.echo(f"Unmatched: {len(unmatched)}")

        methods: dict[str, int] = {}
        for lab in labels:
            m = lab.get("method") or "unmatched"
            methods[m] = methods.get(m, 0) + 1
        typer.echo("\nBy method:")
        for method, count in sorted(methods.items()):
            typer.echo(f"  {method}: {count}")

    if save:
        seg_dir = segment_path(day, segment, stream)
        out_path = save_speaker_labels(seg_dir, labels, metadata)
        if not json_output:
            typer.echo(f"\nWrote: {out_path}")

    if accumulate and source:
        saved = accumulate_voiceprints(day, stream, segment, labels, source)
        if saved and not json_output:
            typer.echo("\nAccumulated voiceprints:")
            for eid, count in saved.items():
                typer.echo(f"  {eid}: {count} embeddings")


@app.command("backfill")
def backfill(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Enumerate segments without processing."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output full result as JSON."
    ),
) -> None:
    """Run speaker attribution across all segments with embeddings.

    Processes segments oldest-first for progressive voiceprint building.
    Skips segments that already have speaker_labels.json (safe to re-run).
    """
    import time

    from apps.speakers.attribution import backfill_segments

    if dry_run and not json_output:
        typer.echo("DRY RUN — no labels will be written\n")

    if not json_output:
        typer.echo("Scanning journal for segments with embeddings...")

    start = time.monotonic()
    last_day = ""

    def on_progress(
        processed: int, total: int, day: str, stream: str, seg_key: str
    ) -> None:
        nonlocal last_day
        if day != last_day:
            typer.echo(f"\n  {day} ", nl=False)
            last_day = day
        typer.echo(".", nl=False)
        if processed % 100 == 0 or processed == total:
            typer.echo(f" [{processed}/{total}]", nl=False)

    stats = backfill_segments(
        dry_run=dry_run,
        progress_callback=None if dry_run or json_output else on_progress,
    )

    elapsed = time.monotonic() - start

    if json_output:
        import json as json_mod

        typer.echo(json_mod.dumps(stats, indent=2, default=str))
        return

    typer.echo("\n")
    typer.echo(f"Total segments scanned:    {stats['total_segments']}")
    typer.echo(f"With embeddings:           {stats['total_eligible']}")
    typer.echo(f"Without embeddings:        {stats['skipped_no_embed']}")
    typer.echo(f"Already labeled (skipped): {stats['already_labeled']}")
    typer.echo(f"Processed this run:        {stats['processed']}")
    typer.echo(f"Elapsed:                   {elapsed:.1f}s")

    speakers = stats.get("speakers_seen", {})
    if speakers:
        typer.echo(f"\nSpeakers identified ({len(speakers)}):")
        sorted_speakers = sorted(speakers.items(), key=lambda x: x[1], reverse=True)
        for eid, count in sorted_speakers[:20]:
            typer.echo(f"  {eid}: {count} attributions")
        if len(sorted_speakers) > 20:
            typer.echo(f"  ... and {len(sorted_speakers) - 20} more")

    if stats["errors"]:
        typer.echo(f"\nErrors ({len(stats['errors'])}):", err=True)
        for err in stats["errors"][:10]:
            typer.echo(f"  {err}", err=True)
        if len(stats["errors"]) > 10:
            typer.echo(f"  ... and {len(stats['errors']) - 10} more", err=True)


@app.command()
def discover(
    json_output: bool = typer.Option(
        False, "--json", help="Output full result as JSON."
    ),
) -> None:
    """Discover recurring unknown speakers across segments."""
    import json as json_mod

    from apps.speakers.discovery import discover_unknown_speakers

    result = discover_unknown_speakers()
    if json_output:
        typer.echo(json_mod.dumps(result, indent=2, default=str))
        return
    clusters = result.get("clusters", [])

    if not clusters:
        typer.echo("No recurring unknown speakers found.")
        raise typer.Exit()

    typer.echo(f"Found {len(clusters)} unknown speaker cluster(s):\n")
    for cluster in clusters:
        typer.echo(
            f"  Cluster {cluster['cluster_id']}: "
            f"{cluster['size']} samples across {cluster['segment_count']} segments"
        )
        for sample in cluster.get("samples", []):
            text_preview = (sample.get("text") or "")[:60]
            typer.echo(
                f"    - {sample['day']}/{sample['stream']}/{sample['segment_key']} "
                f"sid={sample['sentence_id']}: {text_preview}"
            )
        typer.echo()


@app.command()
def identify(
    cluster_id: int = typer.Argument(..., help="Cluster ID from discovery output."),
    name: str = typer.Argument(..., help="Speaker name to assign."),
    entity_id: str | None = typer.Option(
        None, "--entity-id", help="Link to existing entity ID instead of name matching."
    ),
) -> None:
    """Identify a discovered unknown speaker cluster."""
    import json

    from apps.speakers.discovery import identify_cluster

    result = identify_cluster(cluster_id, name, entity_id=entity_id)
    output = json.dumps(result, indent=2, default=str)
    if "error" in result:
        typer.echo(output, err=True)
        raise typer.Exit(1)
    typer.echo(output)


@app.command("merge-names")
def merge_names_cmd(
    alias: str = typer.Argument(..., help="Alias/variant speaker name to merge from."),
    canonical: str = typer.Argument(..., help="Canonical speaker name to merge into."),
) -> None:
    """Merge a speaker name variant into a canonical entity."""
    import json

    from apps.speakers.bootstrap import merge_names

    result = merge_names(alias, canonical)
    output = json.dumps(result, indent=2, default=str)
    if "error" in result:
        typer.echo(output, err=True)
        raise typer.Exit(1)
    typer.echo(output)
