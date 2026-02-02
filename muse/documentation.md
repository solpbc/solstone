{

  "title": "Documentation Moments",
  "description": "Finds when important knowledge is shared in the transcript and suggests what should be written down. Output is a Markdown list of documentation opportunities with time ranges and destinations.",
  "occurrences": "Record an occurrence whenever a new procedure, decision or troubleshooting step is described. Capture the related file or tool and where the documentation should live such as wiki or README.",
  "hook": {"post": "occurrence"},
  "color": "#007bff",
  "schedule": "daily",
  "priority": 10,
  "output": "md",
  "instructions": {
    "sources": {"audio": true, "screen": false, "agents": {"screen": true}},
    "facets": true
  }

}

$daily_preamble

# Workday Documentation Opportunity Analysis

## Objective

Help $preferred capture important knowledge from each workday by analyzing the combined audio and screen transcripts to pinpoint moments when valuable information is shared that should be recorded for future reference.

## Evaluation Goals

1. **Identify Documentable Moments**
   - Look for explanations of workflows, configuration steps, or troubleshooting techniques.
   - Note when key decisions or design rationales are discussed.
   - Capture tips, commands, or best practices mentioned verbally or shown on screen.

2. **Detect Missing Documentation**
   - Flag situations where someone says they'll document something later, but no record exists.
   - Notice instructions or clarifications that appear important yet aren't written down elsewhere.
   - Example: A teammate walking through setup steps during a call that aren't in any README.

3. **Opportunities for Reuse**
   - Identify explanations or procedures that would benefit others if added to docs.
   - Highlight repeated questions or confusion that better documentation could resolve.

## Summarizing Documentation Tasks

Create a friendly markdown document output, for each opportunity add a section with a short title and containing:

- **Time Range**: When the discussion or explanation occurred.
- **Context**: What problem or task was being covered.
- **Key Details to Capture**: The information that should be documented.
- **Suggested Destination**: Where to add it (README, wiki, code comment, etc.).

Be concise but thorough—capture anything that would save time later if clearly documented.
