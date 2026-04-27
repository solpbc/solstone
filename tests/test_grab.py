# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for the sol grab CLI."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "sol_grab"
FIXTURE_JOURNAL = FIXTURE_ROOT / "journal"


def _expected(name: str) -> dict:
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def _invoke_grab(monkeypatch, capsys, *argv: str):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(FIXTURE_JOURNAL))
    monkeypatch.setenv("SOL_SKIP_SUPERVISOR_CHECK", "1")
    monkeypatch.setattr("sys.argv", ["sol grab", *argv])

    from observe.grab import main

    exit_code = 0
    exit_message = ""
    try:
        main()
    except SystemExit as exc:
        if isinstance(exc.code, int):
            exit_code = exc.code
        elif exc.code is None:
            exit_code = 0
        else:
            exit_code = 1
            exit_message = str(exc.code)
    captured = capsys.readouterr()
    return exit_code, exit_message, captured.out, captured.err


def _normalize_saved_paths(actual: dict, expected: dict) -> dict:
    normalized = json.loads(json.dumps(actual))
    for actual_item, expected_item in zip(
        normalized["data"]["saved"], expected["data"]["saved"], strict=True
    ):
        actual_item["path"] = expected_item["path"]
    return normalized


def test_grab_level_0_json_matches_fixture(monkeypatch, capsys):
    code, message, out, err = _invoke_grab(monkeypatch, capsys, "--json")
    assert code == 0
    assert message == ""
    assert err == ""
    assert json.loads(out) == _expected("level_0.json")


def test_grab_level_0_human_lists_days_with_counts(monkeypatch, capsys):
    code, message, out, err = _invoke_grab(monkeypatch, capsys)
    assert code == 0
    assert message == ""
    assert err == ""
    assert "day" in out
    assert "20240102" in out
    assert "20240103" in out


def test_grab_level_1_json_matches_fixture(monkeypatch, capsys):
    code, message, out, err = _invoke_grab(monkeypatch, capsys, "--json", "20240102")
    assert code == 0
    assert message == ""
    assert err == ""
    assert json.loads(out) == _expected("level_1.json")


def test_grab_missing_day_errors(monkeypatch, capsys):
    code, message, out, err = _invoke_grab(monkeypatch, capsys, "20990101")
    assert code == 1
    assert out == ""
    assert err == ""
    assert message == "day 20990101 not found"


def test_grab_level_2_json_matches_fixture(monkeypatch, capsys):
    code, message, out, err = _invoke_grab(
        monkeypatch, capsys, "--json", "20240103", "default"
    )
    assert code == 0
    assert message == ""
    assert err == ""
    assert json.loads(out) == _expected("level_2.json")


def test_grab_missing_stream_errors(monkeypatch, capsys):
    code, message, out, err = _invoke_grab(monkeypatch, capsys, "20240102", "missing")
    assert code == 1
    assert out == ""
    assert err == ""
    assert message == "stream missing not found in 20240102"


def test_grab_level_3_json_matches_fixture(monkeypatch, capsys):
    code, message, out, err = _invoke_grab(
        monkeypatch, capsys, "--json", "20240103", "default", "110000_300"
    )
    assert code == 0
    assert message == ""
    assert err == ""
    assert json.loads(out) == _expected("level_3.json")


def test_grab_level_3_lists_named_monitors(monkeypatch, capsys):
    code, message, out, err = _invoke_grab(
        monkeypatch, capsys, "--json", "20240103", "default", "100000_300"
    )
    payload = json.loads(out)
    assert code == 0
    assert message == ""
    assert err == ""
    assert [screen["screen"] for screen in payload["data"]["screens"]] == [
        "left_DP-1",
        "right_HDMI-1",
    ]
    assert payload["data"]["screens"][0]["position"] == "left"
    assert payload["data"]["screens"][1]["connector"] == "HDMI-1"


def test_grab_missing_segment_errors(monkeypatch, capsys):
    code, message, out, err = _invoke_grab(
        monkeypatch, capsys, "20240103", "default", "999999_300"
    )
    assert code == 1
    assert out == ""
    assert err == ""
    assert message == "segment 999999_300 not found in 20240103/default"


def test_grab_level_4_json_matches_fixture(monkeypatch, capsys):
    code, message, out, err = _invoke_grab(
        monkeypatch, capsys, "--json", "20240102", "default", "233000_300", "screen"
    )
    assert code == 0
    assert message == ""
    assert err == ""
    assert json.loads(out) == _expected("level_4.json")


def test_grab_level_4_human_includes_error_notes(monkeypatch, capsys):
    code, message, out, err = _invoke_grab(
        monkeypatch, capsys, "20240102", "default", "233000_300", "screen"
    )
    assert code == 0
    assert message == ""
    assert err == ""
    assert "frame_id" in out
    assert "error: Vision request timed out while describing frame 18." in out


def test_grab_level_4_legacy_schema_reports_zero_frames(monkeypatch, capsys):
    code, message, out, err = _invoke_grab(
        monkeypatch, capsys, "20240101", "default", "123456_300", "screen"
    )
    assert code == 0
    assert message == ""
    assert err == ""
    assert out.strip() == "0 frames analyzed: file uses pre-frame_id schema"


def test_grab_level_4_captured_but_not_analyzed_errors(monkeypatch, capsys):
    code, message, out, err = _invoke_grab(
        monkeypatch, capsys, "20240103", "default", "110000_300", "screen"
    )
    assert code == 1
    assert out == ""
    assert err == ""
    assert message == "screen screen in 110000_300 is captured but not analyzed"


def test_grab_missing_screen_errors(monkeypatch, capsys):
    code, message, out, err = _invoke_grab(
        monkeypatch, capsys, "20240102", "default", "233000_300", "missing_screen"
    )
    assert code == 1
    assert out == ""
    assert err == ""
    assert message == "screen missing_screen not found in 20240102/default/233000_300"


def test_grab_level_5a_json_matches_fixture(monkeypatch, capsys):
    code, message, out, err = _invoke_grab(
        monkeypatch,
        capsys,
        "--json",
        "20240102",
        "default",
        "233000_300",
        "screen",
        "7",
    )
    assert code == 0
    assert message == ""
    assert err == ""
    assert json.loads(out) == _expected("level_5a.json")


def test_grab_level_5a_human_shows_frame_metadata(monkeypatch, capsys):
    code, message, out, err = _invoke_grab(
        monkeypatch, capsys, "20240102", "default", "233000_300", "screen", "7"
    )
    assert code == 0
    assert message == ""
    assert err == ""
    assert "Screen: screen" in out
    assert '"frame_id": 7' in out


def test_grab_level_5a_legacy_schema_errors(monkeypatch, capsys):
    code, message, out, err = _invoke_grab(
        monkeypatch, capsys, "20240101", "default", "123456_300", "screen", "1"
    )
    assert code == 1
    assert out == ""
    assert err == ""
    assert (
        message
        == "screen file uses pre-frame_id schema; frame selection is unavailable"
    )


def test_grab_missing_frame_id_errors(monkeypatch, capsys):
    code, message, out, err = _invoke_grab(
        monkeypatch, capsys, "20240102", "default", "233000_300", "screen", "999"
    )
    assert code == 1
    assert out == ""
    assert err == ""
    assert message == "frame id 999 not found in screen for 233000_300"


def test_grab_level_5b_json_matches_fixture_and_writes_png(
    monkeypatch, capsys, tmp_path
):
    out_path = tmp_path / "frame.png"
    code, message, out, err = _invoke_grab(
        monkeypatch,
        capsys,
        "--json",
        "--out",
        str(out_path),
        "20240102",
        "default",
        "233000_300",
        "screen",
        "7",
    )
    actual = json.loads(out)
    expected = _expected("level_5b.json")
    assert code == 0
    assert message == ""
    assert err == ""
    assert _normalize_saved_paths(actual, expected) == expected
    assert out_path.is_file()
    with Image.open(out_path) as image:
        assert image.size == (64, 48)


def test_grab_level_5b_refuses_overwrite_without_force(monkeypatch, capsys, tmp_path):
    out_path = tmp_path / "frame.png"
    out_path.write_bytes(b"existing")
    code, message, out, err = _invoke_grab(
        monkeypatch,
        capsys,
        "--out",
        str(out_path),
        "20240102",
        "default",
        "233000_300",
        "screen",
        "7",
    )
    assert code == 1
    assert out == ""
    assert err == ""
    assert "output path exists" in message


def test_grab_level_5b_force_replaces_existing_file(monkeypatch, capsys, tmp_path):
    out_path = tmp_path / "frame.png"
    out_path.write_bytes(b"existing")
    code, message, out, err = _invoke_grab(
        monkeypatch,
        capsys,
        "--force",
        "--out",
        str(out_path),
        "20240102",
        "default",
        "233000_300",
        "screen",
        "7",
    )
    assert code == 0
    assert message == ""
    assert err == ""
    assert out_path.is_file()
    with Image.open(out_path) as image:
        assert image.size == (64, 48)


def test_grab_level_5b_unknown_suffix_is_argparse_error(monkeypatch, capsys, tmp_path):
    code, message, out, err = _invoke_grab(
        monkeypatch,
        capsys,
        "--out",
        str(tmp_path / "frame.gif"),
        "20240102",
        "default",
        "233000_300",
        "screen",
        "7",
    )
    assert code == 2
    assert message == ""
    assert out == ""
    assert "--out must end in .png, .jpg, .jpeg, or .webp" in err


def test_grab_level_5c_json_matches_fixture_and_writes_numbered_files(
    monkeypatch, capsys, tmp_path
):
    out_path = tmp_path / "frame.png"
    code, message, out, err = _invoke_grab(
        monkeypatch,
        capsys,
        "--json",
        "--out",
        str(out_path),
        "20240102",
        "default",
        "233000_300",
        "screen",
        "7,12,23",
    )
    actual = json.loads(out)
    expected = _expected("level_5c.json")
    assert code == 0
    assert message == ""
    assert err == ""
    assert _normalize_saved_paths(actual, expected) == expected
    for frame_id in (7, 12, 23):
        saved = tmp_path / f"frame_{frame_id}.png"
        assert saved.is_file()
        with Image.open(saved) as image:
            assert image.size == (64, 48)


def test_grab_level_5c_conflict_scan_happens_before_decode(
    monkeypatch, capsys, tmp_path
):
    out_path = tmp_path / "frame.png"
    (tmp_path / "frame_12.png").write_bytes(b"existing")
    decode_mock = Mock(side_effect=AssertionError("decode should not run"))
    monkeypatch.setattr("observe.grab.decode_frames", decode_mock)
    code, message, out, err = _invoke_grab(
        monkeypatch,
        capsys,
        "--out",
        str(out_path),
        "20240102",
        "default",
        "233000_300",
        "screen",
        "7,12,23",
    )
    assert code == 1
    assert out == ""
    assert err == ""
    assert "output path exists" in message
    decode_mock.assert_not_called()


def test_grab_level_5c_decode_failure_writes_no_files(monkeypatch, capsys, tmp_path):
    out_path = tmp_path / "frame.png"
    monkeypatch.setattr(
        "observe.grab.decode_frames", Mock(side_effect=RuntimeError("decode blew up"))
    )
    code, message, out, err = _invoke_grab(
        monkeypatch,
        capsys,
        "--out",
        str(out_path),
        "20240102",
        "default",
        "233000_300",
        "screen",
        "7,12,23",
    )
    assert code == 1
    assert out == ""
    assert err == ""
    assert message == "decode blew up"
    assert list(tmp_path.iterdir()) == []


def test_grab_level_5c_requires_out_for_multiple_frame_ids(monkeypatch, capsys):
    code, message, out, err = _invoke_grab(
        monkeypatch, capsys, "20240102", "default", "233000_300", "screen", "7,12,23"
    )
    assert code == 2
    assert message == ""
    assert out == ""
    assert "multiple frame ids require --out" in err


def test_grab_rejects_more_than_five_positionals(monkeypatch, capsys):
    code, message, out, err = _invoke_grab(
        monkeypatch, capsys, "a", "b", "c", "d", "e", "f"
    )
    assert code == 2
    assert message == ""
    assert out == ""
    assert "at most 5 positional tokens" in err


def test_grab_force_requires_out(monkeypatch, capsys):
    code, message, out, err = _invoke_grab(monkeypatch, capsys, "--force")
    assert code == 2
    assert message == ""
    assert out == ""
    assert "--force requires --out" in err


def test_grab_out_requires_level_5(monkeypatch, capsys, tmp_path):
    code, message, out, err = _invoke_grab(
        monkeypatch, capsys, "--out", str(tmp_path / "frame.png"), "20240102"
    )
    assert code == 2
    assert message == ""
    assert out == ""
    assert "--out requires day stream segment screen and frame-id" in err


@pytest.mark.parametrize("token", ["0", "-1", "abc", "7,7", "1,,2"])
def test_grab_frame_id_token_rejects_invalid_values(token):
    from observe.grab import parse_frame_id_token

    with pytest.raises(ValueError):
        parse_frame_id_token(token)


def test_grab_frame_id_token_sorts_batch_ids():
    from observe.grab import parse_frame_id_token

    assert parse_frame_id_token("23,7,12") == [7, 12, 23]
