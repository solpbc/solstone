# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Regression tests for link workspace modal visibility."""

from __future__ import annotations

import re


def test_workspace_modals_are_hidden_by_attribute_and_css(link_env):
    env = link_env()
    response = env.client.get("/app/link/")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert 'id="link-pair-modal"' in body
    assert re.search(r'<div id="link-pair-modal"[^>]{0,200}\bhidden\b', body)
    assert 'id="link-unpair-modal"' in body
    assert re.search(r'<div id="link-unpair-modal"[^>]{0,200}\bhidden\b', body)
    assert ".link-modal[hidden]" in body
