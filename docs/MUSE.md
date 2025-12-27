# Muse Module

AI agent system and MCP tooling for Sunstone.

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
   Backend runners (openai, google, anthropic, claude)
```

## Agent Backends

| Backend | Module | Features |
|---------|--------|----------|
| OpenAI | `muse/openai.py` | GPT models via Agents SDK |
| Google | `muse/google.py` | Gemini models |
| Anthropic | `muse/anthropic.py` | Claude via Anthropic SDK |
| Claude | `muse/claude.py` | Claude Code SDK with filesystem tools |

All backends implement `AgentSession` with `run()` and `add_history()` methods.

## Key Components

- **cortex.py** - Central agent manager, file watcher, event distribution
- **cortex_client.py** - Client functions: `cortex_request()`, `cortex_agents()`
- **mcp.py** - FastMCP server with journal search tools
- **agents.py** - CLI and shared `AgentSession` interface

## Agent Personas

System prompts in `muse/agents/*.txt` with metadata in matching `.json` files. Apps can add custom agents in `apps/{app}/agents/`.

## Documentation

- [CORTEX.md](CORTEX.md) - Full API, event schemas, request format
- [CALLOSUM.md](CALLOSUM.md) - Message bus protocol
- [THINK.md](THINK.md) - Cortex usage examples
