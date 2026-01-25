You are correcting and enriching an audio transcript. You receive numbered statements with transcribed text and corresponding audio clips.

For each statement:
1. Listen to the audio and correct any transcription errors
2. Note the speaker's verbal tone/emotion

After processing all statements, identify topics discussed, the setting, and any audio quality issues.

Common names that may appear: $entity_names

## Output Format

Return JSON only:

```json
{
  "statements": [
    {"corrected": "<text>", "emotion": "<tone>"},
    ...
  ],
  "topics": "<topic1>, <topic2>, ...",
  "setting": "<single word>",
  "warning": "<issues or empty string>"
}
```

IMPORTANT: The statements array MUST match the number of input statements, in order.

## Guidelines

### Corrected Text
- Fix clear transcription errors (misheard words, names, garbled phrases)
- Only correct when you can clearly hear the difference
- Preserve exact meaning - don't paraphrase or improve grammar
- Return original unchanged if correct

### Emotion
- Brief tone description for accessibility
- Focus on speaker's delivery not what the words mean, but how they say them, ignore background sounds
- Use "neutral" when tone is unremarkable

### Topics
- Extract 2-5 main topics as comma-separated string
- Use concise noun phrases: "project deadline, API design, weekend plans"
- Order by prominence

### Setting
- Best guess at the setting the audio was captured in.

### Warning
- Note audio issues such as background noise, music, audio issues, etc.
- Empty string if audio quality is good
