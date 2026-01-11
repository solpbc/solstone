# Provider Implementation Guide

Guide for implementing new AI providers in the muse module.

For a high-level overview of the muse module, see [MUSE.md](MUSE.md).

## Required Exports

Each provider module in `muse/providers/` must export three functions:

| Function | Purpose |
|----------|---------|
| `generate()` | Synchronous text generation |
| `agenerate()` | Asynchronous text generation |
| `run_agent()` | Agentic execution with MCP tool support |

See `muse/providers/__init__.py` for the canonical export list and `muse/providers/google.py` as a reference implementation.

Each provider module must also define `__all__` exporting these three functions.

## API Key Handling

Providers load API keys from environment variables using a consistent pattern:

**Naming convention:** `{PROVIDER}_API_KEY` (e.g., `GOOGLE_API_KEY`, `OPENAI_API_KEY`)

**Implementation pattern:**
```python
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("MYPROVIDER_API_KEY")
if not api_key:
    raise ValueError("MYPROVIDER_API_KEY not found in environment")
```

**Client caching:** Providers typically cache client instances as module-level singletons to enable connection reuse:
```python
_client = None

def _get_client():
    global _client
    if _client is None:
        load_dotenv()
        api_key = os.getenv("MYPROVIDER_API_KEY")
        if not api_key:
            raise ValueError("MYPROVIDER_API_KEY not found in environment")
        _client = MyProviderClient(api_key=api_key)
    return _client
```

**Settings app integration:** Add new API keys to `apps/settings/routes.py` in the `PROVIDER_API_KEYS` dict and corresponding UI fields in `apps/settings/workspace.html` for user configuration.

## generate() / agenerate()

These functions handle direct LLM text generation. The unified API in `muse/models.py` routes requests to provider-specific implementations.

**Key responsibilities:**
- Accept the common parameter set (see `muse/models.py` `generate()` signature)
- Return the response text as a string
- Call `log_token_usage()` after successful generation
- Handle provider-specific response parsing and validation

**Important notes:**
- The `model` parameter arrives already resolved - providers don't do routing
- The `context` parameter is for token logging attribution only
- Absorb unknown kwargs via `**kwargs` to maintain forward compatibility
- Provider-specific features (e.g., `cached_content` for Google) are passed through kwargs

**Handling optional features:**

Some parameters are provider-specific:
- `thinking_budget`: Supported by Google (via ThinkingConfig) and Anthropic (certain models)
- `json_output`: Google uses `response_mime_type`, Anthropic adds system instruction
- `cached_content`: Google-only content caching

Providers should gracefully ignore unsupported parameters.

## run_agent()

Handles agentic execution with MCP tool integration.

```python
async def run_agent(
    config: Dict[str, Any],
    on_event: Optional[Callable[[dict], None]] = None,
) -> str:
```

**Config dict fields** (see `muse/agents.py` `main_async()` for routing logic):
- `prompt`: User's input (required)
- `model`: Model identifier
- `max_tokens`: Output token limit
- `instruction`: System instruction
- `extra_context`: Additional context prepended as first user message
- `mcp_server_url`: URL for MCP tool server
- `disable_mcp`: Skip MCP tool integration
- `tools`: Optional list of allowed tool names
- `agent_id`, `persona`: Identity for logging and tool calls
- `continue_from`: Agent ID for conversation continuation

**Event emission:**

Providers must emit events via the `on_event` callback. See `muse/agents.py` for TypedDict definitions:

| Event | When |
|-------|------|
| `StartEvent` | Agent run begins |
| `ToolStartEvent` | Tool invocation starts |
| `ToolEndEvent` | Tool invocation completes |
| `ThinkingEvent` | Reasoning/thinking content available |
| `FinishEvent` | Agent run completes successfully |
| `ErrorEvent` | Error occurs |

Use `JSONEventCallback` from `muse/agents.py` to wrap the callback and auto-add timestamps.

**Error handling pattern:**

All providers must follow this pattern to prevent duplicate error reporting:
```python
try:
    # ... agent logic ...
except Exception as exc:
    callback.emit({
        "event": "error",
        "error": str(exc),
        "trace": traceback.format_exc(),
    })
    setattr(exc, "_evented", True)  # Prevents duplicate reporting
    raise
```

**MCP tool integration:**

Use `create_mcp_client()` from `think/utils.py` to connect to the MCP server:
```python
from think.utils import create_mcp_client

async with create_mcp_client(config["mcp_server_url"]) as mcp:
    # mcp.session provides call_tool(), list_tools(), etc.
```

**Conversation continuation:**

When `continue_from` is provided, load conversation history using:
```python
from muse.agents import parse_agent_events_to_turns

turns = parse_agent_events_to_turns(config["continue_from"])
# Returns [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
```

## Token Logging

All generation calls must log token usage for cost tracking.

```python
from muse.models import log_token_usage

log_token_usage(model=model, usage=response, context=context)
```

**Key points:**
- Call after successful generation
- `usage` can be provider-specific format or response object - `log_token_usage()` normalizes it
- `context` enables attribution to specific features/operations
- See `muse/models.py` `log_token_usage()` for supported formats

## Context & Routing

Context strings determine provider and model selection. Providers receive already-resolved models, but understanding the system helps:

**Context naming convention:** `{module}.{feature}[.{operation}]`
- Examples: `observe.enrich`, `insight.meetings`, `app.chat.title`

**Resolution** (handled by `muse/models.py` `resolve_provider()`):
1. Exact match in journal.json `providers.contexts`
2. Glob pattern match (fnmatch) with specificity ranking
3. Built-in `CONTEXT_DEFAULTS` in `muse/models.py`
4. Default provider/tier from config

Providers don't implement routing - they receive the resolved model.

## Configuration

Provider configuration lives in `journal.json` under the `providers` key.

**Structure:**
```
providers:
  default:
    provider: <provider-name>
    tier: <1|2|3>
  contexts:
    <context-pattern>:
      provider: <provider-name>
      model: <explicit-model>  # OR
      tier: <1|2|3>            # tier-based resolution
  models:
    <provider-name>:
      "<tier>": "<model-override>"
```

**Tier system:**
- 1 = PRO (most capable)
- 2 = FLASH (balanced)
- 3 = LITE (fast/cheap)

See `fixtures/journal/config/journal.json` for a complete example and `muse/models.py` `PROVIDER_DEFAULTS` for tier-to-model mappings.

## Testing

**Required test coverage:**

1. **Unit tests** in `tests/test_<provider>.py`:
   - Mock API responses
   - Test parameter handling
   - Test error cases

2. **Integration tests** in `tests/integration/test_<provider>_backend.py`:
   - Live API calls (require API keys)
   - End-to-end generation
   - Token usage verification

See existing test files for patterns:
- `tests/test_google.py`, `tests/test_openai.py`, `tests/test_anthropic.py`
- `tests/integration/test_google_backend.py`, etc.

Run integration tests with: `make test-integration`

## Checklist for New Providers

1. Create `muse/providers/<name>.py` with `__all__` exports
2. Implement `generate()`, `agenerate()`, `run_agent()`
3. Add model constants to `muse/models.py` (e.g., `MYPROVIDER_PRO`)
4. Add provider to `PROVIDER_DEFAULTS` in `muse/models.py`
5. Add routing cases in `muse/models.py`:
   - `generate()` function (around line 622)
   - `agenerate()` function (around line 700)
6. Add routing case in `muse/agents.py` `main_async()`
7. Add API key to `apps/settings/routes.py` `PROVIDER_API_KEYS`
8. Add API key UI field in `apps/settings/workspace.html`
9. Create unit tests in `tests/test_<name>.py`
10. Create integration tests in `tests/integration/test_<name>_backend.py`
11. Update `muse/providers/__init__.py` docstring
