# Provider Implementation Guide

Guide for implementing new AI providers in the think module.

For a high-level overview of the think module, see [THINK.md](THINK.md).

## Required Exports

Each provider module in `think/providers/` must export three functions:

| Function | Purpose |
|----------|---------|
| `run_generate()` | Synchronous text generation, returns `GenerateResult` |
| `run_agenerate()` | Asynchronous text generation, returns `GenerateResult` |
| `run_tools()` | Tool-calling execution with MCP integration |

See `think/providers/__init__.py` for the canonical export list and `think/providers/google.py` as a reference implementation.

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

**Settings app integration:** Add your provider to `PROVIDER_METADATA` in `think/providers/__init__.py` with `label` and `env_key` fields. The settings UI dynamically builds provider dropdowns from the registry. Add corresponding API key UI fields in `apps/settings/workspace.html` for user configuration.

## run_generate() / run_agenerate()

These functions handle direct LLM text generation. The unified API in `think/models.py` routes requests to provider-specific implementations and handles token logging and JSON validation centrally.

**Function signature:**
```python
from think.providers.shared import GenerateResult

def run_generate(
    contents: Union[str, List[Any]],
    model: str,
    temperature: float = 0.3,
    max_output_tokens: int = 8192 * 2,
    system_instruction: Optional[str] = None,
    json_output: bool = False,
    thinking_budget: Optional[int] = None,
    timeout_s: Optional[float] = None,
    **kwargs: Any,
) -> GenerateResult:
```

The `run_agenerate()` function has the same signature but is `async`.

**Return type - GenerateResult:**
```python
class GenerateResult(TypedDict, total=False):
    text: Required[str]           # Response text
    usage: Optional[dict]         # Normalized usage dict
    finish_reason: Optional[str]  # Normalized: "stop", "max_tokens", etc.
    thinking: Optional[list]      # List of thinking block dicts
```

**Parameter details:**

| Parameter | Notes |
|-----------|-------|
| `contents` | String, list of strings, or list with mixed content. For vision-capable providers (currently Google only), can include PIL Image objects. Other providers stringify non-text content. |
| `model` | Already resolved by routing - providers don't need to handle model selection. |
| `max_output_tokens` | Response token limit. Note: Google internally adds `thinking_budget` to this for total budget calculation. |
| `system_instruction` | System prompt. Providers handle this per their API (separate field, prepended message, etc.). |
| `json_output` | Request JSON response. Google uses `response_mime_type`, Anthropic/OpenAI use response format or system instruction. |
| `thinking_budget` | Token budget for reasoning/thinking. Must be `> 0` to enable; `None` or `0` means no thinking. Only Google and Anthropic support this - OpenAI ignores it (uses fixed "medium" reasoning effort). Note: `run_tools()` always enables thinking regardless of this parameter. |
| `timeout_s` | Request timeout in seconds. Convert to provider's expected format (e.g., Google uses milliseconds internally). |
| `**kwargs` | Absorb unknown kwargs for forward compatibility. Provider-specific options (e.g., `cached_content` for Google) pass through here. |

**Key responsibilities:**
- Accept the common parameter set shown above
- Return `GenerateResult` with text, usage, finish_reason, and thinking
- Normalize `finish_reason` to standard values: `"stop"`, `"max_tokens"`, `"safety"`, etc.
- Handle provider-specific response parsing

**Note:** Token logging and JSON validation are handled by the wrapper in `think/models.py`, not by providers.

**Important:** Providers should gracefully ignore unsupported parameters rather than raising errors.

## run_tools()

Handles tool-calling execution with MCP integration.

```python
async def run_tools(
    config: Dict[str, Any],
    on_event: Optional[Callable[[dict], None]] = None,
) -> str:
```

**Config dict fields** (see `think/agents.py` `main_async()` for routing logic):
- `prompt`: User's input (required)
- `model`: Model identifier
- `max_tokens`: Output token limit
- `system_instruction`: System instruction (journal.md for agents)
- `extra_context`: Runtime context (facets, insights list, datetime) as first user message
- `user_instruction`: Agent-specific prompt as second user message
- `mcp_server_url`: URL for MCP tool server
- `disable_mcp`: Skip MCP tool integration
- `tools`: Optional list of allowed tool names
- `agent_id`, `name`: Identity for logging and tool calls
- `continue_from`: Agent ID for conversation continuation

**Event emission:**

Providers must emit events via the `on_event` callback. See `think/agents.py` for TypedDict definitions:

| Event | When |
|-------|------|
| `StartEvent` | Agent run begins |
| `ToolStartEvent` | Tool invocation starts |
| `ToolEndEvent` | Tool invocation completes |
| `ThinkingEvent` | Reasoning/thinking content available |
| `FinishEvent` | Agent run completes successfully |
| `ErrorEvent` | Error occurs |

Use `JSONEventCallback` from `think/agents.py` to wrap the callback and auto-add timestamps.

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
from think.agents import parse_agent_events_to_turns

turns = parse_agent_events_to_turns(config["continue_from"])
# Returns [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
```

## Token Logging

Token logging is handled centrally by the wrapper in `think/models.py`. Providers return usage data in their `GenerateResult`, and the wrapper calls `log_token_usage()`.

**Usage dict format:**

Providers should build a normalized usage dict from their response:
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

# Google
usage_dict = {
    "input_tokens": response.usage_metadata.prompt_token_count,
    "output_tokens": response.usage_metadata.candidates_token_count,
    "total_tokens": response.usage_metadata.total_token_count,
}
```

**Key points:**
- Return usage in `GenerateResult["usage"]` - wrapper handles logging
- For `run_tools()`, include usage in the `finish` event

## Context & Routing

Context strings determine provider and model selection. Providers receive already-resolved models, but understanding the system helps:

**Context naming convention:**
- Muse configs (agents/generators): `muse.{source}.{name}` where source is `system` or app name
  - System: `muse.system.meetings`, `muse.system.default`
  - App: `muse.entities.observer`, `muse.chat.helper`
- Other contexts: `{module}.{feature}[.{operation}]`
  - Examples: `observe.describe.frame`, `app.chat.title`

**Dynamic discovery:** All context metadata (tier/label/group) is defined in prompt .md files via YAML frontmatter:
- Prompt files: Listed in `PROMPT_PATHS` in `think/models.py` - add `context`, `tier`, `label`, `group` fields
- Categories: `observe/categories/*.md` - add `tier`, `label`, `group` fields
- System muse: `muse/*.md` - add `tier`, `label`, `group` fields in frontmatter
- App muse: `apps/*/muse/*.md` - add `tier`, `label`, `group` fields in frontmatter

All contexts are discovered at runtime. Use `get_context_registry()` to get the complete context map.

**Resolution** (handled by `think/models.py` `resolve_provider()`):
1. Exact match in journal.json `providers.contexts`
2. Glob pattern match (fnmatch) with specificity ranking
3. Dynamic context registry (discovered prompts, categories, muse configs)
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

See `fixtures/journal/config/journal.json` for a complete example and `think/models.py` `PROVIDER_DEFAULTS` for tier-to-model mappings.

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

The `Batch` class in `think/batch.py` automatically works with all providers via the unified `agenerate()` API in `think/models.py`. No provider-specific batch implementation is needed - just ensure your `run_agenerate()` works correctly.

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
1. Create `think/providers/<name>.py` with `__all__ = ["run_generate", "run_agenerate", "run_tools"]`
2. Implement `run_generate()`, `run_agenerate()`, `run_tools()` following signatures above
3. Import `GenerateResult` from `think.providers.shared` and return it from generate functions

**Model constants** in `think/models.py`:
4. Add model constants using the pattern `{PROVIDER}_{TIER}` (e.g., `DO_LLAMA_70B`, `DO_MISTRAL_NEMO`)
   - Existing examples: `GEMINI_FLASH`, `GPT_5`, `CLAUDE_SONNET_4`
5. Add provider tier mappings to `PROVIDER_DEFAULTS` dict
6. Update `get_model_provider()` to detect your models by prefix (critical for cost tracking)

**Registry:**
7. Add provider to `PROVIDER_REGISTRY` in `think/providers/__init__.py`
8. Add routing case in `think/agents.py` `main_async()` (around line 331)

**Settings UI:**
9. Add provider to `PROVIDER_METADATA` in `think/providers/__init__.py` with `label` and `env_key`
10. Add API key UI field in `apps/settings/workspace.html`

**Testing:**
11. Create unit tests in `tests/test_<name>.py`
12. Create integration tests in `tests/integration/test_<name>_backend.py`
13. Add test contexts to `fixtures/journal/config/journal.json`

**Documentation:**
14. Update `think/providers/__init__.py` docstring
15. Update `docs/THINK.md` providers table
16. Update `docs/CORTEX.md` valid provider values
