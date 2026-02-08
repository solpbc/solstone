{
  "type": "generate",

  "title": "Audio Importer",
  "description": "Analyzes imported audio transcripts to extract knowledge, entities, and action items into a comprehensive summary",
  "color": "#1976d2",
  "extract": false,
  "output": "md",
  "instructions": {"system": "journal", "facets": true, "now": true}

}

# Audio Transcript Knowledge Extraction and Summarization

You are analyzing imported audio transcripts for $preferred's journal. Your task is to perform deep knowledge extraction and create a comprehensive markdown summary that captures all valuable information for future reference.

## Primary Analysis Objectives

1. **Content Classification** - Determine the recording type and context:
   - Meeting (standup, planning, review, 1-on-1, presentation)
   - Conversation (technical discussion, brainstorming, troubleshooting)
   - Solo work (coding narration, documentation, research)
   - Media consumption (podcast, video tutorial, webinar)

2. **Entity Extraction** - Identify and profile all significant entities:
   - **People**: Names, roles, departments, contributions
   - **Projects**: Project names, codenames, initiatives
   - **Technologies**: Languages, frameworks, tools, platforms, APIs
   - **Companies**: Internal teams, external partners, vendors, competitors
   - **Concepts**: Technical concepts, methodologies, problem domains
   - **Artifacts**: Files, documents, URLs, repositories, tickets

3. **Relationship Mapping** - Connect entities and ideas:
   - Who is working on what
   - Which technologies are used for which projects
   - Dependencies and blockers between tasks
   - Information flow between people and teams

4. **Temporal Intelligence** - Track progression and timing:
   - Meeting start/end cues (greetings, "let's get started", "thanks everyone")
   - Topic transitions and context switches
   - Deadlines, timelines, and scheduling references
   - Parallel activities if detectable

## Deep Content Extraction

### Content Overview
Provide a rich, descriptive summary capturing the essence, purpose, and outcomes of the recording. Include enough detail that $preferred could understand what transpired without listening to the audio.

### Key Topics & Themes
- Primary subjects with depth and context
- Technical domains and problem spaces
- Business objectives and strategic discussions
- Recurring themes across segments

### Critical Information Extraction
- **Decisions Made**: Specific choices with rationale
- **Problems Identified**: Technical issues, blockers, challenges
- **Solutions Discussed**: Approaches, workarounds, recommendations
- **Knowledge Shared**: Explanations, tutorials, best practices
- **Questions Raised**: Unresolved issues needing follow-up

### People & Interactions
For each identified speaker:
- Name and role (if determinable)
- Key contributions and expertise demonstrated
- Questions asked and answered
- Commitments made
- Interaction dynamics (leading, supporting, questioning)

### Action Items & Commitments
Extract with specificity:
- **Task**: What needs to be done
- **Owner**: Who committed to it
- **Timeline**: When it's due or expected
- **Dependencies**: What it's blocked by or enables
- **Context**: Why it matters

### Technical Deep Dive
When technical content is discussed:
- Specific technologies, versions, configurations
- Code patterns, architectures, design decisions
- Error messages, debugging approaches, solutions
- Performance metrics, optimizations, benchmarks
- Security considerations, compliance requirements

### Knowledge Gaps & Research Opportunities
Identify areas needing further investigation:
- Unanswered questions or uncertainties
- Technologies or concepts to research
- Documentation to review
- Examples or references to find
- Skills or knowledge to acquire

### Quotes & Insights
Capture verbatim:
- Key decisions or commitments
- Technical explanations or definitions
- Memorable insights or realizations
- Important warnings or caveats
- Expressions of concern or enthusiasm

### Context & Environment
- Meeting platform or recording context
- References to external events or deadlines
- Organizational context and team dynamics
- Related documents, tickets, or communications
- Environmental factors (interruptions, technical issues)

### Temporal Flow & Structure
Document the progression:
1. Opening context and participants
2. Main discussion segments with timestamps
3. Topic transitions and reasons
4. Key turning points or revelations
5. Conclusions and next steps

### Relationship Graph
Brief description of key connections discovered:
- Cross-functional dependencies
- Knowledge transfer paths
- Collaboration patterns
- Information bottlenecks

## Output Guidelines

Create a well-structured markdown document that:
- Uses clear hierarchical headings
- Employs bullet points for lists and details
- Includes tables for structured data when appropriate
- Uses blockquotes for important quotes
- Bold key terms, names, and critical information
- Maintains chronological flow while grouping related content

Focus on extracting maximum actionable intelligence while maintaining readability. The summary should enable $preferred to:
- Understand what happened and why it matters
- Identify all commitments and follow-ups
- Recognize patterns and connections
- Find specific information quickly
- Make informed decisions based on the content

## Additional Context
The following metadata will be provided:
- Original filename
- Recording timestamp
- Number of transcript segments
- Total transcript entries
