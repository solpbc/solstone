# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Integration tests for Ollama provider with a live local Ollama instance."""

import asyncio

import pytest

from think.models import OLLAMA_LITE

# Use the smallest available model for fast integration tests
_TEST_MODEL = OLLAMA_LITE


def _ollama_reachable() -> bool:
    """Check if the local Ollama instance is reachable."""
    try:
        from think.providers.ollama import validate_key

        return validate_key("")["valid"]
    except Exception:
        return False


# Skip all tests in this module if Ollama is not running
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _ollama_reachable(),
        reason="Local Ollama instance not reachable",
    ),
]


class TestOllamaGenerate:
    def test_basic_generation(self):
        from think.providers.ollama import run_generate

        result = run_generate(
            "What is 2 + 2? Reply with just the number.",
            model=_TEST_MODEL,
            max_output_tokens=64,
            thinking_budget=0,
        )

        assert result["text"]
        assert "4" in result["text"]
        assert result["usage"] is not None
        assert result["usage"]["input_tokens"] > 0
        assert result["usage"]["output_tokens"] > 0
        assert result["finish_reason"] == "stop"
        # With think=False via native API, thinking should be absent
        assert result["thinking"] is None

    def test_system_instruction(self):
        from think.providers.ollama import run_generate

        result = run_generate(
            "What color is the sky?",
            model=_TEST_MODEL,
            max_output_tokens=64,
            system_instruction="Always respond in exactly one word.",
            thinking_budget=0,
        )

        assert result["text"]

    def test_json_output(self):
        from think.providers.ollama import run_generate

        result = run_generate(
            'Return a JSON object with key "answer" and value 42.',
            model=_TEST_MODEL,
            max_output_tokens=256,
            json_output=True,
            thinking_budget=0,
        )

        # The response should contain JSON content. Small models may wrap
        # it in markdown fences, so we check for the key rather than strict
        # parsing. (JSON validation is handled centrally by think/models.py.)
        assert result["text"]
        assert "answer" in result["text"]

    def test_thinking_enabled(self):
        from think.providers.ollama import run_generate

        result = run_generate(
            "What is 15 * 17?",
            model=_TEST_MODEL,
            max_output_tokens=512,
            thinking_budget=4096,
        )

        assert result["text"]
        assert result["usage"] is not None
        # With think=True, thinking content should be present
        assert result["thinking"] is not None
        assert len(result["thinking"]) > 0
        assert result["thinking"][0]["summary"]

    def test_thinking_disabled_no_reasoning(self):
        """Verify that think=False actually suppresses reasoning on the native API."""
        from think.providers.ollama import run_generate

        result = run_generate(
            "What is 2 + 2? Reply with just the number.",
            model=_TEST_MODEL,
            max_output_tokens=64,
            thinking_budget=0,
        )

        assert result["text"]
        assert result["thinking"] is None


class TestOllamaAgenerate:
    def test_async_generation(self):
        from think.providers.ollama import run_agenerate

        result = asyncio.run(
            run_agenerate(
                "What is 3 + 5? Reply with just the number.",
                model=_TEST_MODEL,
                max_output_tokens=64,
                thinking_budget=0,
            )
        )

        assert result["text"]
        assert "8" in result["text"]
        assert result["usage"] is not None


class TestOllamaListModels:
    def test_list_models(self):
        from think.providers.ollama import list_models

        models = list_models()
        assert isinstance(models, list)
        assert len(models) > 0
        # Native API returns models with "name" field
        assert "name" in models[0]


class TestOllamaValidateKey:
    def test_reachable(self):
        from think.providers.ollama import validate_key

        result = validate_key("")
        assert result["valid"] is True
