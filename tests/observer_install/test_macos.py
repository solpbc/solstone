# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

from solstone.observe.observer_install.macos import REDIRECT_TEXT, MacosDriver


def test_macos_redirect(args_factory, capsys):
    args = args_factory(platform="macos")

    assert MacosDriver().run(args) == 0

    assert capsys.readouterr().out.strip() == REDIRECT_TEXT


def test_macos_dry_run_redirect(args_factory, capsys, observer_install_env):
    args = args_factory(platform="macos", dry_run=True)

    assert MacosDriver().run(args) == 0

    output = capsys.readouterr().out
    assert output.startswith("Dry-run: would direct you to download solstone-macos:")
    assert REDIRECT_TEXT in output
    assert not (observer_install_env.home / ".local" / "share" / "solstone").exists()
