{
  "type": "generate",

  "title": "Document Analysis",
  "description": "Extracts structured intelligence from imported documents",
  "color": "#5c6bc0",
  "schedule": "segment",
  "priority": 10,
  "hook": {"pre": "documents"},
  "thinking_budget": 8192,
  "max_output_tokens": 8192,
  "output": "md",
  "load": {"transcripts": true, "percepts": false, "talents": false}

}

$segment_preamble

Analyze the imported document and extract structured intelligence from it.

Use only what the document explicitly states; never infer or assume anything not written. Use exact names, titles, and terms as they appear in the document. Distinguish primary appointees from successor or contingent appointees. When paraphrasing a legal or formal role, preserve the formal term parenthetically, for example: the person managing the estate (Personal Representative). If a section has no relevant content, write "Not specified in this document" and still include the section.

Write exactly these sections as markdown H2 headings:

## Overview
Document type, title, date, parties, and purpose in 2-3 sentences.

## Parties and Roles
Every named person or entity and their role in the document.

## Key Provisions
Substantive terms, obligations, rights, distributions, and powers.

## Assets and Property
Any assets, accounts, property, real estate, or valuables referenced.

## Conditions and Triggers
Conditions that activate provisions, including death, incapacity, dates, or other triggering events.

## Important Dates
Execution date, effective dates, deadlines, review dates, and other material dates.

## Summary
Plain-language summary suitable for quick reference.
