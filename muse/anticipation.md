{

  "title": "Anticipation Extraction",
  "description": "Extracts structured anticipation events (future scheduled items) from insight summaries.",
  "color": "#4527a0"

}

# Anticipation JSON Conversion

## Objective

Extract future scheduled events from a Markdown summary and convert them into structured JSON anticipations. These are events that have not yet occurred but are planned or scheduled for future dates.

## Instructions
1. **Extract every distinct future event** mentioned in the summary - meetings, deadlines, appointments, etc.
2. **Be comprehensive** - capture calendar events, scheduled meetings, deadlines, personal appointments, recurring events, travel, etc.
3. **Preserve date and timing information** - always extract the scheduled date, and time if known
4. **Handle uncertainty gracefully** - use `null` for start/end times when not specified
5. **Assign facets** - for every anticipation, choose the best matching facet from the Available Facets context. This field is required.
6. **Return only valid JSON** - no commentary, explanations, or wrapper objects
7. **Handle empty sources** - if the source indicates no future events, return an empty array: `[]`

## Anticipation Fields
- **type** – the kind of event such as `meeting`, `deadline`, `appointment`, `event`, `travel`, `reminder`, etc.
- **date** – ISO date (YYYY-MM-DD) when the event is scheduled to occur. Required.
- **start** and **end** – HH:MM:SS timestamps for the event time, or `null` if time is unknown/TBD
- **title** – short descriptive title for display
- **summary** – concise one-sentence description of what is planned
- **work** – boolean classification: `true` for work-related, `false` for personal
- **participants** – optional list of people or entities involved (empty array if none)
- **facet** – required facet identifier; use the facet name/ID from Available Facets context
- **details** – free-form string capturing location, agenda, preparation notes, or other context

## Handling Time Uncertainty
- If exact time is known (e.g., "1:00 PM"): use `"start": "13:00:00"`
- If time is vague (e.g., "Morning", "Afternoon"): use `null` and mention in details
- If time is TBD or not specified: use `null`
- If it's an all-day event: use `null` for both start and end

## Handling Recurring Events
For recurring events (e.g., "every Wednesday"), extract the next upcoming instance with its specific date. Mention the recurrence pattern in details.

## Output Format
Return a JSON array of anticipations only. Each anticipation must include all required fields.

## Example
[
    {
        "type": "meeting",
        "date": "2025-11-05",
        "start": "13:00:00",
        "end": "14:30:00",
        "title": "Meeting with Center for the Blind",
        "summary": "Introductory meeting to discuss potential collaboration on AI-powered tools.",
        "work": true,
        "participants": ["Jack Anderson", "Center for the Blind Representatives"],
        "facet": "blind_center",
        "details": "Virtual meeting. Topics include grant writing and building an AI brain from their documents."
    },
    {
        "type": "deadline",
        "date": "2025-11-10",
        "start": null,
        "end": null,
        "title": "Airport System Deployment Target",
        "summary": "Target date for deploying the cleaned-up UI and new restart scripts.",
        "work": true,
        "participants": ["Mitch Baumgartner"],
        "facet": "aviation_networks",
        "details": "Software deployment. No specific time set."
    },
    {
        "type": "event",
        "date": "2025-10-27",
        "start": null,
        "end": null,
        "title": "FMDS Partner Workshop",
        "summary": "Multi-day in-person workshop with partners running through October 30.",
        "work": true,
        "participants": ["L3Harris", "GDIT", "Frequentis", "uAvionix"],
        "facet": "fmds",
        "details": "In-person, security clearance required. Day 1: Win strategy and demo dry run. Day 2: Solution definition. Day 3: Prototype planning. Runs Oct 27-30."
    },
    {
        "type": "appointment",
        "date": "2025-11-15",
        "start": null,
        "end": null,
        "title": "Gymnastics Meet",
        "summary": "First gymnastics meet of the season.",
        "work": false,
        "participants": ["Blade"],
        "facet": "family",
        "details": "Location: Marshaltown Community Center."
    }
]
