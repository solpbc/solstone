{
  "type": "generate",

  "title": "Innovation Opportunities",
  "description": "Scans conversations and tasks for sparks of new ideas, problem statements and potential ventures. Outputs a list of the most promising opportunities with context and suggested next steps.",
  "occurrences": "Whenever a novel idea or pain point is raised, record an occurrence describing the opportunity and any proposed solution. Include who mentioned it and classify the potential impact.",
  "hook": {"post": "occurrence"},
  "color": "#20c997",
  "schedule": "daily",
  "priority": 10,
  "disabled": true,
  "output": "md",
  "instructions": {
    "sources": {"audio": true, "screen": false, "agents": {"screen": true}},
    "facets": true
  }

}

$daily_preamble

# Workday Innovation Opportunity Discovery

## Objective

Analyze the full workday transcript to identify embryonic ideas, innovation sparks, and potential business opportunities that emerge throughout the day. The transcript combines audio conversations and screen activity organized by recording segments.

## Discovery Framework

### 1. Problem-Solution Signals
- **Unmet Needs**: Complaints, frustrations, or workarounds that suggest market opportunities
- **"Wouldn't it be nice if..."**: Wishful thinking statements that reveal innovation potential
- **Manual Processes**: Repetitive tasks or workflows that could be automated or productized
- **Tool Gaps**: Missing features or capabilities in existing tools that prompt creative solutions

### 2. Creative Sparks & Ideation
- **Spontaneous Ideas**: Off-hand comments about potential solutions or new approaches
- **Cross-Facet Connections**: Moments when concepts from different fields intersect unexpectedly
- **"What if" Explorations**: Hypothetical scenarios or thought experiments mentioned
- **Adjacent Possibilities**: Ideas that build on existing work but point to new directions

### 3. Market & Business Signals
- **Customer Pain Points**: Direct or indirect mentions of user/customer struggles
- **Industry Trends**: Discussions about emerging technologies or market shifts
- **Competitive Gaps**: References to things competitors aren't doing well
- **Partnership Opportunities**: Potential collaborations or integrations mentioned

### 4. Technical Innovation Seeds
- **Novel Implementations**: Unique approaches to solving technical problems
- **Architecture Ideas**: New ways of structuring systems or data flows
- **Feature Concepts**: Functionality that doesn't exist but was imagined or desired
- **Integration Opportunities**: Connecting previously unconnected systems or data

### 5. Strategic Opportunities
- **Pivot Potential**: Discussions that suggest alternative business directions
- **Market Expansion**: Ideas for reaching new user segments or use cases
- **Platform Possibilities**: Concepts that could become foundational for other innovations
- **Ecosystem Plays**: Opportunities to create or tap into larger systems

## Analysis Approach

1. **Sequential Discovery**
   - Read chronologically, staying alert for innovation signals
   - Note both explicit ideas and implicit opportunities
   - Consider context - some of the best ideas emerge from frustration or constraints

2. **Pattern Recognition**
   - Look for recurring themes across different conversations or tasks
   - Identify problems mentioned multiple times or by different people
   - Notice when similar solutions are independently suggested

3. **Opportunity Assessment**
   - Gauge the potential impact and feasibility of each opportunity
   - Consider whether it's a quick experiment or longer-term venture
   - Note any existing momentum or interest from others

## Output Format

Create a friendly markdown document with individual sections for each opportunity, using a catchy short title, containing:

- **Time Range**: When the opportunity was identified
- **Context**: The discussion or activity that sparked the idea
- **Opportunity Description**: What the innovation or business idea entails
- **Why It Matters**: The problem it solves or value it creates
- **Next Steps**: Immediate experiments or explorations to validate the concept
- **Innovation Type**: Quick win / Feature enhancement / New product / Platform / Business model

Focus on the top 10-15 most promising opportunities. Conclude with:

### Innovation Summary
- **Most Exciting Opportunity**: The single idea with highest potential
- **Quick Experiments**: 3-5 ideas that could be tested within a week
- **Strategic Ventures**: 2-3 concepts worth deeper exploration
- **Cross-Cutting Themes**: Patterns that suggest broader innovation directions

## Special Considerations

- Pay attention to moments of excitement or energy in conversations
- Note when multiple people express interest in the same concept
- Look for intersections between personal interests and professional capabilities
- Consider both B2B and B2C opportunities
- Include both technical innovations and business model innovations
- Don't dismiss "small" ideas - they often grow into bigger opportunities
- Watch for moments when existing solutions are criticized or found lacking

Remember: The goal is to surface nascent opportunities that might otherwise be forgotten, helping transform daily insights into potential ventures or innovations.
