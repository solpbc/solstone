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

**Settings app integration:** Add your provider to `PROVIDER_METADATA` in `muse/providers/__init__.py` with `label` and `env_key` fields. The settings UI dynamically builds provider dropdowns from the registry. Add corresponding API key UI fields in `apps/settings/workspace.html` for user configuration.

## generate() / agenerate()

These functions handle direct LLM text generation. The unified API in `muse/models.py` routes requests to provider-specific implementations.

**Function signature:**
```python
def generate(
    contents: Union[str, List[Any]],
    model: str,
    temperature: float = 0.3,
    max_output_tokens: int = 8192 * 2,
    system_instruction: Optional[str] = None,
    json_output: bool = False,
    thinking_budget: Optional[int] = None,
    timeout_s: Optional[float] = None,
    context: Optional[str] = None,
    **kwargs: Any,
) -> str:
```

The `agenerate()` function has the same signature but is `async`.

**Parameter details:**

| Parameter | Notes |
|-----------|-------|
| `contents` | String, list of strings, or list with mixed content. For vision-capable providers (currently Google only), can include PIL Image objects. Other providers stringify non-text content. |
| `model` | Already resolved by routing - providers don't need to handle model selection. |
| `max_output_tokens` | Response token limit. Note: Google internally adds `thinking_budget` to this for total budget calculation. |
| `system_instruction` | System prompt. Providers handle this per their API (separate field, prepended message, etc.). |
| `json_output` | Request JSON response. Google uses `response_mime_type`, Anthropic/OpenAI use response format or system instruction. |
| `thinking_budget` | Token budget for reasoning/thinking. Only Google and Anthropic (certain models) support this - others should silently ignore. |
| `timeout_s` | Request timeout in seconds. Convert to provider's expected format (e.g., Google uses milliseconds internally). |
| `context` | For token logging attribution only - not used in generation. |
| `**kwargs` | Absorb unknown kwargs for forward compatibility. Provider-specific options (e.g., `cached_content` for Google) pass through here. |

**Key responsibilities:**
- Accept the common parameter set shown above
- Return the response text as a string
- Call `log_token_usage()` after successful generation
- Handle provider-specific response parsing and validation

**Important:** Providers should gracefully ignore unsupported parameters rather than raising errors.

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

**Finish event format:**

The `finish` event must include the result text and should include usage for token tracking:
```python
callback.emit({
    "event": "finish",
    "result": final_text,
    "usage": usage_dict,  # Same format as token logging
    "ts": int(time.time() * 1000),
})
```

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

log_token_usage(model=model, usage=usage_dict, context=context)
```

**Usage dict format:**

Providers should build a usage dict from their response. The normalized format is:
```python
usage_dict = {
    "input_tokens": 1500,      # Required
    "output_tokens": 500,      # Required
    "total_tokens": 2000,      # Required
    "cached_tokens": 800,      # Optional: cache hits
    "reasoning_tokens": 200,   # Optional: thinking/reasoning tokens
}
```

Provider-specific extraction examples:
```python
# OpenAI / OpenAI-compatible
usage_dict = {
    "input_tokens": response.usage.prompt_tokens,
    "output_tokens": response.usage.completion_tokens,
    "total_tokens": response.usage.total_tokens,
}

# Anthropic
usage_dict = {
    "input_tokens": response.usage.input_tokens,
    "output_tokens": response.usage.output_tokens,
    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
}

# Google (can pass response object directly)
log_token_usage(model=model, usage=response, context=context)
```

**Key points:**
- Call after successful generation
- `log_token_usage()` normalizes various formats automatically
- `context` enables attribution to specific features/operations

## Context & Routing

Context strings determine provider and model selection. Providers receive already-resolved models, but understanding the system helps:

**Context naming convention:** `{module}.{feature}[.{operation}]`

**Dynamic discovery:** Categories and agents can express their own tier/label/group in their JSON configs:
- Categories: `observe/categories/*.json` - add `tier`, `label`, `group` fields
- System agents: `muse/agents/*.json` - add `tier`, `label`, `group` fields
- App agents: `apps/*/agents/*.json` - add `tier`, `label`, `group` fields

These are discovered at runtime and merged with static defaults. Use `get_context_registry()` to get the complete context map including discovered entries.

See `CONTEXT_DEFAULTS` in `muse/models.py` for static context patterns (non-discoverable contexts like `observe.detect.*`, `insight.*`).

**Resolution** (handled by `muse/models.py` `resolve_provider()`):
1. Exact match in journal.json `providers.contexts`
2. Glob pattern match (fnmatch) with specificity ranking
3. Dynamic context registry (static defaults + discovered categories/agents)
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

## Batch Processing

The `Batch` class in `muse/batch.py` automatically works with all providers via the unified `agenerate()` API. No provider-specific batch implementation is needed - just ensure your `agenerate()` works correctly.

## OpenAI-Compatible Providers

For providers with OpenAI-compatible APIs (e.g., DigitalOcean, Azure OpenAI, local LLMs), you can leverage the OpenAI SDK with a custom base URL:

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("MYPROVIDER_API_KEY"),
    base_url="https://api.myprovider.com/v1",
)
```

This allows reusing much of the OpenAI provider's patterns for request/response handling.

## Checklist for New Providers

**Core implementation:**
1. Create `muse/providers/<name>.py` with `__all__ = ["generate", "agenerate", "run_agent"]`
2. Implement `generate()`, `agenerate()`, `run_agent()` following signatures above

**Model constants** in `muse/models.py`:
3. Add model constants using the pattern `{PROVIDER}_{TIER}` (e.g., `DO_LLAMA_70B`, `DO_MISTRAL_NEMO`)
   - Existing examples: `GEMINI_FLASH`, `GPT_5`, `CLAUDE_SONNET_4`
4. Add provider tier mappings to `PROVIDER_DEFAULTS` dict
5. Update `get_model_provider()` to detect your models by prefix (critical for cost tracking)

**Routing:**
6. Add `elif provider == "<name>"` cases in `muse/models.py`:
   - `generate()` function (around line 622)
   - `agenerate()` function (around line 700)
7. Add routing case in `muse/agents.py` `main_async()` (around line 331)

**Settings UI:**
8. Add provider to `PROVIDER_METADATA` in `muse/providers/__init__.py` with `label` and `env_key`
9. Add API key UI field in `apps/settings/workspace.html`

**Testing:**
10. Create unit tests in `tests/test_<name>.py`
11. Create integration tests in `tests/integration/test_<name>_backend.py`
12. Add test contexts to `fixtures/journal/config/journal.json`

**Documentation:**
13. Update `muse/providers/__init__.py` docstring
14. Update `docs/MUSE.md` providers table
15. Update `docs/CORTEX.md` valid provider values
