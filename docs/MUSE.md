# Muse Module

AI agent system and MCP tooling for solstone.

## Commands

| Command | Purpose |
|---------|---------|
| `muse-cortex` | Agent orchestration service |
| `muse-mcp-tools` | MCP tool server (runs inside Cortex) |
| `muse-agents` | Direct agent invocation (testing only) |

## Architecture

```
Cortex (orchestrator)
   ├── Callosum connection (events)
   ├── MCP HTTP server (tools)
   └── Agent subprocess management
          ↓
   Providers (openai, google, anthropic, claude)
```

## Providers

| Provider | Module | Features |
|----------|--------|----------|
| OpenAI | `muse/providers/openai.py` | GPT models via Agents SDK |
| Google | `muse/providers/google.py` | Gemini models |
| Anthropic | `muse/providers/anthropic.py` | Claude via Anthropic SDK |
| Claude | `muse/claude.py` | Claude Code SDK with filesystem tools |

Each provider implements `run_agent(config, on_event)` for agent execution with MCP tools and event streaming.

## Key Components

- **cortex.py** - Central agent manager, file watcher, event distribution
- **cortex_client.py** - Client functions: `cortex_request()`, `cortex_agents()`
- **mcp.py** - FastMCP server with journal search tools
- **agents.py** - CLI entry point and shared event types
- **models.py** - Provider routing, model constants, token logging
- **batch.py** - Async batch processing for LLM requests

## Agent Personas

System prompts in `muse/agents/*.txt` with metadata in matching `.json` files. Apps can add custom agents in `apps/{app}/agents/`.

## Documentation

- [CORTEX.md](CORTEX.md) - Full API, event schemas, request format
- [CALLOSUM.md](CALLOSUM.md) - Message bus protocol
- [THINK.md](THINK.md) - Cortex usage examples
