# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from unittest.mock import Mock, patch

import pytest

from solstone.think import providers
from solstone.think.providers import anthropic, get_provider_models, google, openai


def test_get_provider_models_anthropic():
    model = Mock()
    model.model_dump.return_value = {"id": "claude-sonnet-4-20250514", "type": "model"}

    client = Mock()
    client.models.list.return_value = [model]

    with patch(
        "solstone.think.providers.anthropic._get_anthropic_client", return_value=client
    ):
        result = get_provider_models("anthropic")

    assert result == [{"id": "claude-sonnet-4-20250514", "type": "model"}]


def test_get_provider_models_openai():
    model = Mock()
    model.model_dump.return_value = {"id": "gpt-4o", "object": "model"}

    client = Mock()
    client.models.list.return_value = [model]

    with patch(
        "solstone.think.providers.openai._get_openai_client", return_value=client
    ):
        result = get_provider_models("openai")

    assert result == [{"id": "gpt-4o", "object": "model"}]


def test_get_provider_models_google():
    model = Mock()
    model.model_dump.return_value = {
        "name": "models/gemini-2.0-flash",
        "display_name": "Gemini 2.0 Flash",
    }

    client = Mock()
    client.models.list.return_value = [model]

    with patch(
        "solstone.think.providers.google.get_or_create_client", return_value=client
    ):
        result = get_provider_models("google")

    assert result == [
        {"name": "models/gemini-2.0-flash", "display_name": "Gemini 2.0 Flash"}
    ]


def test_get_provider_models_unknown():
    with pytest.raises(ValueError):
        get_provider_models("bogus")


def test_list_models_in_provider_all():
    assert "list_models" in anthropic.__all__
    assert "list_models" in openai.__all__
    assert "list_models" in google.__all__
    assert "get_provider_models" in providers.__all__
