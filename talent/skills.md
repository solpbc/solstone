{
  "type": "generate",
  "title": "Skill Observer",
  "description": "Detects recurring activity patterns and generates structured skill documents describing what the owner does, how, and why.",
  "hook": {"pre": "skills", "post": "skills"},
  "schedule": "activity",
  "activities": ["*"],
  "priority": 90,
  "output": "json",
  "load": {"transcripts": false, "percepts": false, "agents": false}
}

You are analyzing recurring activity patterns to identify and document the owner's skills.

$skill_instruction

$pattern_context

$previous_outputs

Return JSON with these fields:
- `skill_name` (string)
- `slug` (string, must match the canonical slug provided in context)
- `category` (string)
- `description` (string, 1-2 sentences)
- `how` (string, one paragraph)
- `why` (string, one paragraph)
- `tools` (list of strings)
- `collaborators` (list of strings)
- `confidence` (float from 0.0 to 1.0)

Stay grounded in the supplied evidence. Do not fabricate tools, collaborators, or behaviors that are not supported by the observations and prior outputs.
