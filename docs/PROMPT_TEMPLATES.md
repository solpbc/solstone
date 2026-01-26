# Prompt Template System

This document describes solstone's template variable system for personalizing prompts used in generators and agents. Templates enable dynamic substitution of user identity, contextual information, and reusable prompt fragments.

## Overview

Prompts are stored as `.md` files with optional JSON frontmatter for metadata. The prompt content is loaded via `load_prompt()` from `think/utils.py`, which uses Python's `string.Template` with `safe_substitute`. This means:

- Variables use `$name` or `${name}` syntax
- Undefined variables are left as-is (no errors)
- Use `$$` to escape a literal dollar sign

The system supports three categories of variables with the following precedence (highest to lowest):

1. **Context variables** - Passed by callers at runtime
2. **Identity variables** - From journal configuration
3. **Template variables** - From reusable template files

## File Format

Prompt files use JSON frontmatter with `{` and `}` as delimiters (braces on their own lines):

```markdown
{
  "title": "Activity Synthesis",
  "color": "#00bcd4",
  "schedule": "segment"
}

$segment_preamble

# Segment Activity Synthesis

Your prompt content here...
```

Files without metadata can omit the frontmatter entirely - just write the prompt content directly.

## Variable Categories

### Identity Variables

Identity variables come from the `identity` block in `config/journal.json`. These are available in all prompts automatically.

**Common variables:**
- `$name` - Full name
- `$preferred` - Preferred name or nickname
- `$bio` - Self-description
- `$timezone` - IANA timezone identifier

**Pronoun variables** (flattened from nested structure):
- `$pronouns_subject` - e.g., "he", "she", "they"
- `$pronouns_object` - e.g., "him", "her", "them"
- `$pronouns_possessive` - e.g., "his", "her", "their"
- `$pronouns_reflexive` - e.g., "himself", "herself", "themselves"

**Uppercase-first versions** are automatically generated for all identity variables:
- `$Name`, `$Preferred`, `$Bio`
- `$Pronouns_subject`, `$Pronouns_possessive`, etc.

The flattening logic converts nested objects using underscore separators. For example, `identity.pronouns.subject` becomes `$pronouns_subject`.

**References:**
- Identity configuration: [JOURNAL.md](JOURNAL.md) (identity section)
- Flattening implementation: `think/utils.py` → `_flatten_identity_to_template_vars()`

### Template Variables

Template variables come from `.md` files in the `think/templates/` directory. Each file's stem becomes a variable name containing its contents.

**Current templates:**
- `$daily_preamble` - Preamble for full-day output analysis
- `$segment_preamble` - Preamble for single-segment analysis

Templates can themselves use identity and context variables, enabling composable prompt construction. For example, `daily_preamble.md` uses `$preferred` and `$date`.

**Pattern:** To add a new template variable, create `think/templates/mytemplate.md` and it becomes available as `$mytemplate` in all prompts.

**Reference:** `think/templates/` directory

### Context Variables

Context variables are passed at runtime by the code calling `load_prompt()`. These are use-case specific and not globally available.

**Common generator context:**
- `$day` - Day in YYYYMMDD format
- `$date` - Human-readable date (e.g., "Friday, January 24, 2026")
- `$segment` - Segment key (e.g., "143022_300")
- `$segment_start` - Formatted start time (e.g., "2:30 PM")
- `$segment_end` - Formatted end time (e.g., "2:35 PM")

Context variables also get automatic uppercase-first versions (`$Day`, `$Date`, etc.).

**References:**
- Generator context building: `think/generate.py` (search for `prompt_context`)
- Other callers: `observe/extract.py`, `observe/enrich.py`

## Usage Patterns

### For Generators

Generator prompts typically compose a shared preamble with topic-specific instructions:

```markdown
{
  "title": "My Generator",
  "color": "#4caf50",
  "schedule": "segment"
}

$segment_preamble

# Segment Activity Synthesis

Your specific instructions here...
```

The `$segment_preamble` or `$daily_preamble` template provides standardized context about what's being analyzed, while the rest of the prompt defines the specific analysis task.

**Optional model configuration:** Add `max_output_tokens` (response length limit) and `thinking_budget` (model thinking token budget) to override provider defaults.

**Reference:** `muse/*.md` for examples (files with `schedule` field but no `tools` field)

### For Agents

Agent prompts are `.md` files with configuration in frontmatter:

```markdown
{
  "title": "My Agent",
  "tier": 2,
  "tools": "journal"
}

You are a helpful assistant...
```

Agent prompts are split into two parts:

1. **System instruction** - `think/journal.md` (shared across all agents, cacheable)
2. **User instruction** - Agent-specific `.md` file (e.g., `muse/default.md`)

The system instruction establishes the journal partnership context. The user instruction defines the agent's specific role and capabilities.

**Optional model configuration:** Add `max_output_tokens` (response length limit) and `thinking_budget` (model thinking token budget) to override provider defaults. Note: OpenAI uses fixed reasoning and ignores `thinking_budget`.

**Reference:** `think/utils.py` → `get_agent()` for agent configuration loading

### The load_prompt() Function

```python
load_prompt(
    name: str,                      # Prompt filename (without .md)
    base_dir: Path | None = None,   # Directory containing prompt
    include_journal: bool = False,  # Prepend journal.md content
    context: dict | None = None,    # Runtime context variables
) -> PromptContent
```

Returns a `PromptContent` named tuple with `text` (substituted content), `path` (source file), and `metadata` (frontmatter dict).

**Reference:** `think/utils.py` → `load_prompt()`

## Adding New Variables

### Identity Variables

Edit `config/journal.json` to add or modify identity fields. Nested objects are automatically flattened with underscore separators.

### Template Variables

Create a new `.md` file in `think/templates/`. The filename stem becomes the variable name.

### Context Variables

Pass via the `context` parameter when calling `load_prompt()`:

```python
load_prompt("myprompt", context={"custom_var": "value"})
```

## Reference Index

| Category | Authoritative Source |
|----------|---------------------|
| Identity config schema | [JOURNAL.md](JOURNAL.md) (identity section) |
| Identity flattening | `think/utils.py` (`_flatten_identity_to_template_vars`) |
| Template loading | `think/utils.py` (`_load_templates`) |
| Core load function | `think/utils.py` (`load_prompt`) |
| Template files | `think/templates/*.md` |
| Test coverage | `tests/test_template_substitution.py` |
| Generator prompts | `muse/*.md` (files with `schedule` field but no `tools`) |
| Agent prompts | `muse/*.md` (files with `tools` field) |
