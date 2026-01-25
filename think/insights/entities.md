{

  "title": "Entity Extraction",
  "description": "Extracts people, companies, projects, and tools from segment content",
  "frequency": "segment",
  "hook": "entities",
  "thinking_budget": 4096,
  "max_output_tokens": 1024,
  "output": "md",
  "instructions": {
    "system": "journal",
    "facets": "none",
    "sources": {"audio": true, "screen": true, "insights": false}
  }

}

$segment_insight

Extract named entities and descriptions from the given segment transcription document.

Focus on only these four types of entities:
- Person: individual people by name
- Company: businesses and organizations
- Project: named projects, products, or codebases
- Tool: software applications and services

Skip entities that don't fit one of these four types.
Skip any url, domain, filename, or path.

Output as a markdown list. Each line has three parts separated by colon and dash:
* Type: Entity Name - Description

Example:
* Person: Alice Smith - Mentioned in Slack discussing the project timeline
* Tool: Grafana - Visible on screen showing metrics dashboards
