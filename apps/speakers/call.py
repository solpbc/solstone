# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI interface for speaker voiceprint management.

Provides:
    sol call speakers bootstrap [--dry-run]
    sol call speakers resolve-names [--dry-run]
    sol call speakers attribute-segment <day> <stream> <segment>
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="speakers",
    help="Speaker voiceprint management.",
    no_args_is_help=True,
)


@app.command("bootstrap")
def bootstrap(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be saved without saving."
    ),
) -> None:
    """Bootstrap voiceprints from single-speaker segments.

    Scans the full journal for segments where speakers.json lists exactly
    one speaker. In those segments, all non-owner embeddings belong to that
    speaker. Saves them as voiceprints using the owner centroid for
    owner subtraction.
    """
    from apps.speakers.bootstrap import bootstrap_voiceprints

    if dry_run:
        typer.echo("DRY RUN — no voiceprints will be saved\n")

    typer.echo("Bootstrapping voiceprints from single-speaker segments...")
    stats = bootstrap_voiceprints(dry_run=dry_run)

    if "error" in stats:
        typer.echo(f"Error: {stats['error']}", err=True)
        raise typer.Exit(1)

    typer.echo(f"\nSegments scanned: {stats['segments_scanned']}")
    typer.echo(f"Single-speaker segments: {stats['single_speaker_segments']}")
    typer.echo(f"Unique speakers: {len(stats['speakers_found'])}")
    typer.echo(f"Entities created: {stats['entities_created']}")
    typer.echo(f"Embeddings saved: {stats['embeddings_saved']}")
    typer.echo(f"Embeddings skipped (owner): {stats['embeddings_skipped_owner']}")
    typer.echo(f"Embeddings skipped (duplicate): {stats['embeddings_skipped_duplicate']}")

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
) -> None:
    """Resolve speaker name variants using voiceprint similarity.

    Compares voiceprint centroids between all entities. Pairs with cosine
    similarity > 0.90 are flagged as the same person. Unambiguous variants
    (short name is first word of full name) are auto-merged by adding the
    short name as an aka on the canonical entity.
    """
    from apps.speakers.bootstrap import resolve_name_variants

    if dry_run:
        typer.echo("DRY RUN — no merges will be applied\n")

    typer.echo("Resolving speaker name variants...")
    stats = resolve_name_variants(dry_run=dry_run)

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
