{

  "description": "Spreadsheets, slides, document editors, calendars, task and issue tracking tools, other workplace desktop or web apps and professional tools",
  "output": "markdown",
  "extraction": "Extract when different application or service is shown (e.g. ChatGPT vs Calendar vs Docs)",
  "importance": "high"

}

# Productivity App Text Extraction

Extract text from this productivity screenshot (spreadsheets, slides, calendars, task managers, issue trackers, project management tools).

## Header

`# [App Name - Document/View Title]`

## Content Focus

Extract all visible data with appropriate structure:

- **Tables/Spreadsheets**: Use markdown tables, include headers
- **Calendars**: Include event times and titles (`**9:00 AM** - Team Standup`)
- **Tasks/Issues**: Include title, status, assignee, and due date if visible
- **Slides**: Use `##` for slide titles, bullets for content

## Quality

- Preserve data relationships and hierarchy
- Include key metadata (dates, statuses, assignees)
- Mark unclear text with `[unclear]`
- Mark cut-off text with `...`

Return ONLY the formatted markdown.
