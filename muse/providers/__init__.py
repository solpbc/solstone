# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""AI provider backends for muse.

This package contains provider-specific implementations for LLM generation
and agent execution. Each provider module exposes:

- generate(): Sync text generation
- agenerate(): Async text generation
- run_agent(): Agent execution with MCP tools

Available providers:
- google: Google Gemini models
- openai: OpenAI GPT models
- anthropic: Anthropic Claude models
"""
