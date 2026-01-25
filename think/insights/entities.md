{

  "title": "Entity Extraction",
  "description": "Extracts people, companies, projects, and tools from segment content",
  "frequency": "segment",
  "hook": "entities",
  "thinking_budget": 4096,
  "max_output_tokens": 1024,
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

Output as a simple markdown list of detected entities with all three elements: the chosen type, the detected entity name, and the description of how the entity occurred in the document:
* type: name - description of how they appear across the segment (transcript or screen? which app? what do we learn about them?)
