# Screen Description Categories

This directory contains category definitions for vision analysis of screencast frames.

## Adding a New Category

Each category requires a `.md` file with metadata in JSON frontmatter. The file can optionally include extraction prompt content.

### 1. `<category>.md` (required)

Defines the category with JSON frontmatter and optional extraction prompt:

```markdown
{
  "description": "One-line description for categorization prompt",
  "output": "markdown"
}

Optional extraction prompt content goes here...
```

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `description` | Yes | - | Single-line description used in the categorization prompt |
| `output` | No | `"markdown"` | Response format for extraction: `"json"` or `"markdown"` |

Model selection is handled via the providers configuration in `journal.json`. Each category uses the context pattern `observe.describe.<category>` for routing. See [JOURNAL.md](JOURNAL.md) for details on configuring providers per context.

Categories with prompt content after the frontmatter are "extractable" - they can receive detailed content extraction after initial categorization. The prompt is sent to the model for analysis and should instruct the model to:
- Analyze the screenshot for this specific category
- Return content in the format specified by `output` (markdown or JSON)

### 2. `<category>.py` (optional)

Custom formatter for rich markdown output. If not provided, default formatting applies:
- Markdown content: displayed with category header
- JSON content: displayed in a code block

To add a custom formatter, create a `format` function:

```python
def format(content: Any, context: dict) -> str:
    """Format category content to markdown.

    Args:
        content: The category content (str for markdown, dict for JSON)
        context: Dict with:
            - frame: Full frame dict from JSONL
            - file_path: Path to JSONL file
            - timestamp_str: Formatted time like "14:30:22"

    Returns:
        Formatted markdown string (empty string to skip)
    """
    # Your formatting logic here
    return "**Header:**\n\nFormatted content..."
```

## How It Works

1. `observe/describe.py` discovers all `.md` files and builds the categorization prompt dynamically
2. **Phase 1 (Categorization)**: All frames get initial category analysis (primary/secondary)
3. **Phase 2 (Selection)**: AI or fallback logic selects which frames get detailed extraction (configurable via `describe.max_extractions`)
4. **Phase 3 (Extraction)**: Selected frames with extractable categories (those with extraction prompts in their `.md` files) get detailed content extraction
5. Results are stored in JSONL with `enhanced: true/false` indicating extraction status
6. `observe/screen.py` formats JSONL to markdown, using custom formatters when available
