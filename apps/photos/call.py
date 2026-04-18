# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import sys

import typer

from think.utils import require_solstone

app = typer.Typer(
    name="photos",
    help="Photo intelligence from macOS Photos library.",
    no_args_is_help=True,
)


@app.callback()
def _require_up() -> None:
    require_solstone()


@app.command("sync")
def sync(
    library: str | None = typer.Option(
        None,
        "--library",
        help="Path to Photos.sqlite. Default: ~/Pictures/Photos Library.photoslibrary/database/Photos.sqlite",
    ),
) -> None:
    """Sync face clusters from macOS Photos to entity signals."""
    if sys.platform != "darwin":
        typer.echo("This command requires macOS (Photos library is macOS-only).")
        raise typer.Exit(1)

    import logging
    from pathlib import Path

    from apps.photos.reader import read_face_clusters
    from think.entities.matching import build_name_resolution_map
    from think.indexer.journal import (
        _insert_signal_row,
        _load_index_entity_dicts,
        get_journal_index,
    )

    logger = logging.getLogger(__name__)

    if library is None:
        library = str(
            Path.home()
            / "Pictures"
            / "Photos Library.photoslibrary"
            / "database"
            / "Photos.sqlite"
        )

    if not Path(library).exists():
        typer.echo(f"Photos database not found: {library}")
        raise typer.Exit(1)

    try:
        clusters = read_face_clusters(library)
    except Exception as e:
        typer.echo(f"Error reading Photos database: {e}")
        raise typer.Exit(1)

    typer.echo(f"Found {len(clusters)} named face clusters.")
    if not clusters:
        return

    conn, _ = get_journal_index()
    try:
        entity_dicts = _load_index_entity_dicts(conn)
        face_names = [cluster["name"] for cluster in clusters]
        name_map = build_name_resolution_map(face_names, entity_dicts)

        matched = {name for name, entity_id in name_map.items() if entity_id}
        typer.echo(f"Matched {len(matched)} to entities.")

        if not matched:
            return

        conn.execute(
            "DELETE FROM entity_signals WHERE signal_type='photo_cooccurrence'"
        )

        signal_count = 0
        for cluster in clusters:
            if cluster["name"] not in matched:
                continue
            for day in cluster["days"]:
                _insert_signal_row(
                    conn,
                    {
                        "signal_type": "photo_cooccurrence",
                        "entity_name": cluster["name"],
                        "entity_type": None,
                        "target_name": None,
                        "relationship_type": None,
                        "day": day,
                        "facet": None,
                        "event_title": None,
                        "event_type": None,
                        "path": f"photos/{cluster['person_pk']}/{day}",
                    },
                )
                signal_count += 1

        conn.commit()
        typer.echo(f"Created {signal_count} photo signals.")
        logger.info(
            "Photo sync: %d clusters, %d matched, %d signals",
            len(clusters),
            len(matched),
            signal_count,
        )
    finally:
        conn.close()
