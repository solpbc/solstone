{

  "title": "Occurrence Extraction",
  "description": "Extracts structured occurrence events from insight summaries.",
  "color": "#37474f"

}

# Occurrence JSON Conversion

## Objective

Extract events from a Markdown summary generated from daily transcripts and convert them into structured JSON occurrences.

## Instructions
1. **Extract every distinct event** mentioned in the summary - no matter how brief
2. **Be comprehensive** - capture meetings, messages, file activities, follow-ups, documentation work, research, media consumption, etc.
3. **Preserve timing information** when available in the source
4. **Infer missing details** reasonably when context provides clues
5. **Assign facets** - for every occurrence, always choose the best matching facet from the Available Facets context. Use the facet name/ID (e.g., `facet_name`) based on entities mentioned, subject matter, or context. This field is required.
6. **Return only valid JSON** - no commentary, explanations, or wrapper objects
7. **Handle empty sources** - if the source indicates no events occurred (e.g., "No meetings detected"), return an empty array: `[]`

## Occurrence Fields
- **type** – the kind of occurrence such as `meeting`, `message`, `file`, `followup`, `documentation`, `research`, `media`, etc.
- **start** and **end** – HH:MM:SS timestamps containing the occurrence (use "00:00:00" if unknown)
- **title** – short descriptive title for display
- **summary** – concise one-sentence recap of what happened
- **work** – boolean classification: `true` for work-related, `false` for personal or not work related
- **participants** – optional list of people or entities involved (empty array if none)
- **facet** – required facet identifier; use the facet name/ID from Available Facets context that best matches the occurrence based on entities, subject matter, or context
- **details** – free-form string capturing all additional context, specifics, outcomes, or other relevant information from the original document that is not covered already by the other fields

## Output Format
Return a JSON array of occurrences only. Each occurrence must include all required fields.

## Example
[
    {
        "type": "meeting",
        "start": "09:00:00",
        "end": "09:30:00",
        "title": "Team stand-up",
        "summary": "Daily status update with the engineering team discussing sprint progress and blockers.",
        "work": true,
        "participants": ["$name", "Alice", "Bob"],
        "facet": "work_project_alpha",
        "details": "Alice reported database optimization complete ahead of schedule. Bob mentioned UI testing delays due to missing design assets. Scheduled follow-up meeting for authentication module review. Sprint velocity tracking discussed."
    },
    {
        "type": "message",
        "start": "14:22:00",
        "end": "14:22:00",
        "title": "Slack message to design team",
        "summary": "Requested updated mockups for the login flow redesign.",
        "work": true,
        "participants": ["$name", "Design Team"],
        "facet": "work_project_alpha",
        "details": "Specifically asked for mobile responsive versions and accessibility considerations. Mentioned deadline of end of week."
    },
    {
        "type": "research",
        "start": "15:30:00",
        "end": "16:15:00",
        "title": "Authentication framework comparison",
        "summary": "Researched OAuth 2.0 vs Auth0 implementation options for the new user system.",
        "work": true,
        "participants": [],
        "facet": "work_project_alpha",
        "details": "Compared security features, pricing models, and integration complexity. Created comparison spreadsheet with pros/cons. Leaning toward Auth0 for faster implementation."
    }
]
