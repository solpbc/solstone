# Muse Module

AI agent system and MCP tooling for solstone.

## Commands

| Command | Purpose |
|---------|---------|
| `sol cortex` | Agent orchestration service |
| `sol mcp` | MCP tool server (runs inside Cortex) |
| `sol agents` | Direct agent invocation (testing only) |

## Architecture

```
Cortex (orchestrator)
   ├── Callosum connection (events)
   ├── MCP HTTP server (tools)
   └── Agent subprocess management
          ↓
   Providers (openai, google, anthropic)
```

## Providers

| Provider | Module | Features |
|----------|--------|----------|
| OpenAI | `muse/providers/openai.py` | GPT models via Agents SDK |
| Google | `muse/providers/google.py` | Gemini models |
| Anthropic | `muse/providers/anthropic.py` | Claude via Anthropic SDK |

Providers implement `generate()`, `agenerate()`, and `run_agent()` functions. See [PROVIDERS.md](PROVIDERS.md) for implementation details.

## Key Components

- **cortex.py** - Central agent manager, file watcher, event distribution
- **cortex_client.py** - Client functions: `cortex_request()`, `cortex_agents()`
- **mcp.py** - FastMCP server with journal search tools
- **agents.py** - CLI entry point and shared event types
- **models.py** - Unified `generate()`/`agenerate()` API, provider routing, token logging
- **batch.py** - `Batch` class for concurrent LLM requests with dynamic queuing

## Agent Personas

System prompts in `muse/agents/*.txt` with metadata in matching `.json` files. Apps can add custom agents in `apps/{app}/agents/`.

JSON metadata supports `title`, `provider`, `model`, `tools`, `schedule`, `priority`, `multi_facet`, and `instructions` keys. See [APPS.md](APPS.md#instructions-configuration) for the `instructions` schema that controls system prompts, facet context, and source filtering.

## Documentation

- [PROVIDERS.md](PROVIDERS.md) - Provider implementation guide
- [CORTEX.md](CORTEX.md) - Full API, event schemas, request format
- [CALLOSUM.md](CALLOSUM.md) - Message bus protocol
- [THINK.md](THINK.md) - Cortex usage examples
