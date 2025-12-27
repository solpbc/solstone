# Screen Description Categories

This directory contains category definitions for vision analysis of screencast frames.

## Adding a New Category

Each category requires a `.json` file with metadata, and optionally a `.txt` prompt file for follow-up analysis.

### 1. `<category>.json` (required)

Defines the category and its behavior:

```json
{
  "description": "One-line description for categorization prompt",
  "followup": true,
  "output": "markdown",
  "iq": "lite"
}
```

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `description` | Yes | - | Single-line description used in the categorization prompt |
| `followup` | No | `false` | Whether to run follow-up analysis for this category |
| `output` | No | `"markdown"` | Response format: `"json"` or `"markdown"` |
| `iq` | No | `"lite"` | Model tier: `"lite"`, `"flash"`, or `"pro"` |

### 2. `<category>.txt` (required if `followup: true`)

The vision prompt template sent to the model for detailed analysis. Should instruct the model to:
- Analyze the screenshot for this specific category
- Return content in the format specified by `output` (markdown or JSON)

### 3. `<category>.py` (optional)

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

1. `observe/describe.py` discovers all `.json` files and builds the categorization prompt dynamically
2. Initial categorization identifies primary/secondary categories from the screenshot
3. For categories with `followup: true`, a follow-up request extracts detailed content using the `.txt` prompt
4. Results are stored in JSONL under the category name (e.g., `"meeting": {...}`)
5. `observe/screen.py` formats JSONL to markdown, using custom formatters when available
