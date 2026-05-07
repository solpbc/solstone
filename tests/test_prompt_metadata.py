# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for prompt metadata parsing failures."""

import json

import pytest

import solstone.think.prompts as prompts
import solstone.think.talent as talent
from solstone.think.prompts import PromptMetadataError, _load_prompt_metadata


def test_load_prompt_metadata_returns_expected_fields(tmp_path):
    md_path = tmp_path / "valid.md"
    md_path.write_text(
        '{\n  "title": "Valid Prompt",\n  "schedule": "daily"\n}\n\nBody text\n',
        encoding="utf-8",
    )

    info = _load_prompt_metadata(md_path)

    assert info["path"] == str(md_path)
    assert info["mtime"] == int(md_path.stat().st_mtime)
    assert info["title"] == "Valid Prompt"
    assert info["schedule"] == "daily"
    assert info["color"] == "#6c757d"


def test_load_prompt_metadata_raises_prompt_metadata_error_for_bad_json(tmp_path):
    md_path = tmp_path / "invalid.md"
    md_path.write_text(
        '{\n  "title": "Invalid Prompt",\n  "disabled": True\n}\n\nBody text\n',
        encoding="utf-8",
    )

    with pytest.raises(PromptMetadataError) as excinfo:
        _load_prompt_metadata(md_path)

    exc = excinfo.value
    assert exc.path == md_path
    assert str(md_path) in str(exc)
    assert isinstance(exc.__cause__, json.JSONDecodeError)


def test_load_raw_templates_raises_prompt_metadata_error_for_bad_template(
    tmp_path, monkeypatch
):
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    bad_template = templates_dir / "broken.md"
    bad_template.write_text(
        '{\n  "title": "Broken Template",\n  "disabled": True\n}\n\nBody text\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(prompts, "TEMPLATES_DIR", templates_dir)
    monkeypatch.setattr(prompts, "_templates_cache", None)

    with pytest.raises(PromptMetadataError) as excinfo:
        prompts._load_raw_templates()

    exc = excinfo.value
    assert exc.path == bad_template
    assert str(bad_template) in str(exc)
    assert isinstance(exc.__cause__, json.JSONDecodeError)


def test_get_talent_configs_propagates_prompt_metadata_error(tmp_path, monkeypatch):
    talent_dir = tmp_path / "talent"
    talent_dir.mkdir()
    apps_dir = tmp_path / "apps"
    apps_dir.mkdir()
    broken_prompt = talent_dir / "broken.md"
    broken_prompt.write_text(
        '{\n  "title": "Broken Prompt",\n  "disabled": True\n}\n\nBody text\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(talent, "TALENT_DIR", talent_dir)
    monkeypatch.setattr(talent, "APPS_DIR", apps_dir)

    with pytest.raises(PromptMetadataError) as excinfo:
        talent.get_talent_configs(include_disabled=True)

    exc = excinfo.value
    assert exc.path == broken_prompt
    assert str(broken_prompt) in str(exc)
    assert isinstance(exc.__cause__, json.JSONDecodeError)
