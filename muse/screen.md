{

  "title": "Screen Record",
  "description": "Creates a detailed documentary record of screen activity. Focuses on the 'what' - chronological account with preserved details, excerpts, and entities.",
  "color": "#9c27b0",
  "schedule": "segment",
  "output": "md"

}

$segment_preamble

# Segment Screen Record

## Objective

Create a **detailed documentary record** of what occurred on screen during this segment. Your job is to produce a comprehensive, factual account that preserves important details for future reference and review.

## Input Structure

The segment data includes:

### Audio Transcript
Spoken content captured during the segment - use this to understand context for screen actions.

### Screen Activity
Frame-by-frame analyses with:
- **Timestamp**: Wall-clock time (HH:MM:SS)
- **Monitor**: Which display (when multiple monitors present)
- **Category**: Activity type (terminal, code, messaging, meeting, browsing, reading, media, productivity)
- **Visual description**: What was visible on screen
- **Extracted text**: OCR content (commands, code, messages, documents)
- **Meeting analysis**: Participants, topics, shared content when meetings detected

## Guidelines

### Chronological Narrative
- Structure by time periods or major activity shifts
- Include approximate timestamps for transitions
- Weave multiple monitors into a coherent timeline

### Preserve Details
- Include key commands and their outputs
- Note specific files, functions, or code sections edited
- Quote relevant message excerpts
- List documentation topics reviewed
- Capture URLs, file paths, error messages

### Meeting Documentation
- List all participants detected
- Summarize discussion topics
- Note any shared screens, slides, or documents

### Entity Extraction
Capture all significant entities encountered:
- People names
- Project/repository names
- Company/product names
- Tool names
- File paths and URLs

End with a **## Entities** section listing all significant entities.

## Output Format

Provide a comprehensive markdown report:

- **Headers** (##, ###) to organize by time or activity
- **Past tense** narrative style
- **Code blocks** for commands, code, and technical output
- **Blockquotes** for message excerpts
- **Lists** for entity sections

## What NOT to Include

- Interpretation of intent or underlying goals
- Progress state analysis (blocked, exploring, etc.)
- Facet associations
- Idle activities (games, screensavers)

The output should serve as a **detailed activity log** - someone reviewing it should know exactly what happened on screen without watching the recording.
