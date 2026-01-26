{

  "title": "Activity Synthesis",
  "description": "Interprets each segment to extract meaning, intent, and searchability. Focuses on the 'why' behind actions - tasks, progress states, facets, and keywords for discovery.",
  "color": "#00bcd4",
  "schedule": "segment",
  "output": "md"

}

$segment_preamble

# Segment Activity Synthesis

## Objective

Interpret this segment to extract **meaning and searchability**. Your job is NOT to describe what happened (that's captured elsewhere) but to synthesize the underlying intent, progress, and context that makes this segment findable and understandable later.

## Core Focus

### Tasks & Intent
What is the person really trying to accomplish? Look beyond literal actions to understand the underlying goals. Name the tasks concretely (e.g., "debugging the OAuth refresh flow" not "working on code").

### Progress States
Characterize each task's working state:
- **Making headway**: Clear progress, solutions emerging
- **Blocked**: Stuck on errors, waiting, can't proceed
- **Exploring**: Investigating options, researching
- **Deciding**: Weighing alternatives, making choices

### Facet Associations
**Required**: Identify which facet(s) this segment relates to. Every segment belongs to at least one facet. If work spans multiple facets, list all.

### Searchable Keywords
What terms would someone use to find this segment later? Include:
- Project/feature names
- Problem descriptions ("authentication bug", "memory leak")
- Key decisions made
- People involved and their roles
- Tools or technologies central to the work

## Output Format

Write **concise, scannable markdown**. Aim for brevity - a few paragraphs or bullet points, not a detailed narrative. Structure flexibly based on content:

- Group by facet when multiple are present
- Note context switches and why they occurred
- Highlight blockers or breakthroughs
- Include emotional/energy state if notable (frustrated, energized, scattered)

## What NOT to Include

- Detailed chronological account of screen activity
- Exact commands, code excerpts, file contents
- Entity lists
- Time-by-time breakdown

Remember: This synthesis should make the segment **discoverable through search** and **quickly understandable** without reading the full transcript.
