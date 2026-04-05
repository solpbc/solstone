{
  "type": "generate",
  "title": "Observation",
  "description": "Extracts patterns from segment data during onboarding observation",
  "disabled": true,
  "schedule": "segment",
  "priority": 97,
  "output": "json",
  "hook": {"pre": "observation", "post": "observation"},
  "tier": 3,
  "thinking_budget": 2048,
  "max_output_tokens": 2048,
  "exclude_streams": ["import.*"],
  "load": {"transcripts": true, "percepts": true, "agents": false}
}

You are analyzing a captured segment of someone's computer activity to learn about their work patterns. This is part of an onboarding observation — the owner has asked the system to watch how they work for a day and then suggest how to organize their journal.

## Input

You receive a transcript combining audio (microphone/system audio, with speaker labels) and screen activity (app usage, visible content) from a ~5-minute capture window.

## Task

Extract structured observations about what happened in this segment. Focus on:

1. **Meetings** — conversations with 2+ speakers. Note participant count, any names mentioned, and the topic/context.
2. **Apps** — what applications or tools the owner is actively using.
3. **Entities** — specific people, companies, projects, or tools mentioned by name.
4. **Topics** — what subjects or themes are present in the activity.
5. **Summary** — a brief 1-line description of what the owner was doing.

## Output Format

Return a JSON object:

```json
{
  "has_meeting": false,
  "speaker_count": 1,
  "meeting_topic": null,
  "apps": ["VS Code", "Terminal"],
  "people": ["Alice Chen"],
  "companies": [],
  "projects": ["auth-service"],
  "tools": ["Git", "Docker"],
  "topics": ["backend development", "authentication"],
  "summary": "Solo coding session working on authentication service"
}
```

### Field Definitions

- `has_meeting`: true if 2+ speakers are having a conversation (not just background audio)
- `speaker_count`: number of distinct speakers detected
- `meeting_topic`: brief topic if a meeting is detected, null otherwise
- `apps`: list of applications/tools actively in use on screen
- `people`: names of people mentioned or speaking (use real names when identifiable, "Speaker N" when not)
- `companies`: company or organization names mentioned
- `projects`: project names, product names, or codebases mentioned
- `tools`: development tools, services, or platforms mentioned
- `topics`: 1-3 high-level topic themes for this segment
- `summary`: one concise sentence describing the segment

## Rules

1. Only report what you can clearly observe — don't speculate
2. Use real names when they appear in the transcript; "Speaker N" is fine for unnamed speakers
3. Empty lists are valid when nothing is detected for a category
4. Keep the summary factual and brief
5. Return ONLY the JSON object, no other text
