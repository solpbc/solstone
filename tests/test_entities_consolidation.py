# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import json
import logging
from datetime import datetime
from pathlib import Path

from typer.testing import CliRunner

from think.call import call_app
from think.entities.consolidation import consolidate_detected_entities
from think.entities.journal import save_journal_entity

runner = CliRunner()


def _seed_detection(
    journal_path: Path,
    detection: dict[str, str],
    *,
    day: str | None = None,
    segment: str = "120000_300",
) -> Path:
    day = day or datetime.now().strftime("%Y%m%d")
    talents_dir = journal_path / "chronicle" / day / "default" / segment / "talents"
    talents_dir.mkdir(parents=True, exist_ok=True)
    path = talents_dir / "entities.jsonl"
    path.write_text(json.dumps(detection) + "\n", encoding="utf-8")
    return path


def _entity_file_count(journal_path: Path) -> int:
    return sum(1 for _ in (journal_path / "entities").glob("*/entity.json"))


def test_consolidate_writes_new_entity(journal_copy):
    journal_path = Path(journal_copy)
    _seed_detection(
        journal_path,
        {
            "name": "Zephyr Quartz Index",
            "type": "Project",
            "description": "Unique regression seed",
        },
    )

    written = consolidate_detected_entities(str(journal_path), full=True)

    assert written == 1
    entity_path = journal_path / "entities" / "zephyr_quartz_index" / "entity.json"
    assert entity_path.exists()
    entity = json.loads(entity_path.read_text(encoding="utf-8"))
    assert entity == {
        "id": "zephyr_quartz_index",
        "name": "Zephyr Quartz Index",
        "type": "Project",
        "source": "detected",
        "created_at": entity["created_at"],
        "updated_at": entity["updated_at"],
        "description": "Unique regression seed",
    }


def test_consolidate_skips_fuzzy_match(journal_copy, caplog):
    journal_path = Path(journal_copy)
    before_count = _entity_file_count(journal_path)
    save_journal_entity(
        {"id": "jeremie_miller", "name": "Jeremie Miller", "type": "Person"}
    )
    _seed_detection(
        journal_path,
        {
            "name": "Jeremee Miler",
            "type": "Person",
            "description": "Should fuzzy-match and skip",
        },
    )

    with caplog.at_level(logging.INFO):
        written = consolidate_detected_entities(str(journal_path), full=True)

    assert written == 0
    assert _entity_file_count(journal_path) == before_count + 1
    assert (
        "consolidate: 1 detections, 1 matched-skipped (1 fuzzy, 0 exact), 0 new entities"
        in caplog.text
    )


def test_consolidate_creates_for_unrelated_name(journal_copy):
    journal_path = Path(journal_copy)
    save_journal_entity(
        {"id": "jeremie_miller", "name": "Jeremie Miller", "type": "Person"}
    )
    _seed_detection(
        journal_path,
        {
            "name": "Quillon Vastworth",
            "type": "Person",
            "description": "Unique unrelated entity",
        },
    )

    written = consolidate_detected_entities(str(journal_path), full=True)

    assert written == 1
    assert (journal_path / "entities" / "quillon_vastworth" / "entity.json").exists()


def test_consolidate_is_idempotent(journal_copy):
    journal_path = Path(journal_copy)
    _seed_detection(
        journal_path,
        {
            "name": "Zephyr Quartz Index",
            "type": "Project",
            "description": "Unique regression seed",
        },
    )

    first = consolidate_detected_entities(str(journal_path), full=True)
    second = consolidate_detected_entities(str(journal_path), full=True)

    assert first == 1
    assert second == 0


def test_cli_consolidate_dispatches(journal_copy):
    result = runner.invoke(call_app, ["entities", "consolidate", "--full"])

    assert result.exit_code == 0
    assert "Wrote" in result.stdout
