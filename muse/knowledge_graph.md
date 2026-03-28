{
  "type": "generate",

  "title": "Knowledge Graph",
  "description": "Extracts people, projects, tools and other entities from the transcript and maps how they relate. Produces a Markdown report plus narrative describing network hubs and bridges discovered during the day.",
  "occurrences": "For each entity interaction or relationship mentioned, create an occurrence describing the connection. Include start and end times when the relationship is visible, and capture the type of link such as works-on or discusses-with.",
  "hook": {"post": "occurrence"},
  "color": "#6f42c1",
  "schedule": "daily",
  "priority": 10,
  "output": "md",
  "instructions": {
    "sources": {"transcripts": true, "percepts": false, "agents": {"screen": true}},
    "facets": true
  }

}

$daily_preamble

# Comprehensive Workday Knowledge Graph and Network Analysis from Transcripts

**Input:** A markdown file containing a chronologically ordered transcript of a workday for $name. The transcript is organized by recording segments, each combining information from audio recordings and screen activity for that time period.

**Objective:** Generate a comprehensive knowledge graph and perform a network analysis based on the provided workday transcripts.

**Instructions:**

1.  **Entity Extraction and Profiling:**
    * Identify all distinct entities. Categorize entities using appropriate types such as: `Person`, `Organization` (companies, groups, teams), `Project`, `Task`, `Concept` (abstract ideas, features, problem domains), `Tool` (software, applications, physical tools), `Location`, `Event`, and `Topic` (general subjects of discussion), or other contextually relevant categories.
    * For each identified entity, provide:
        * `Entity Name`: The canonical name of the entity.
        * `Entity Type`: Its category from the list above.
        * `First Appearance`: Time.
        * `Total Engagement`: Approximately how many times was it mentioned throughout the day.
        * `Context`:  Provide a concise (1-2 sentence) summary of its role or context in how it was used, referencing key actions or discussions from both audio and screen data if distinguishable.
    * **Concept Quality Filter:** Distinguish `Concept` from `Topic`. A Concept must be a genuinely reusable idea — a mental model, framework, strategic insight, or principle — valuable to recall in a different context. "Discussed the migration timeline" is a Topic (use `Topic` type). "Conway's Law applies to their API design" is a Concept (use `Concept` type). When extracting Concepts, capture WHY the concept matters in the `Context` field, not just that it was mentioned.

2.  **Relationship Mapping:**
    * Identify and map all significant connections between entities.
    * For each connection, define:
        * `Source Name`
        * `Target Name`
        * `Relationship Type`: Use descriptive labels (e.g., `works-on`, `discusses-with`, `uses-tool-for-project`, `reports-to`, `blocked-by`, `enables`, `references-concept`, `mentions-organization`). Be prepared to infer novel relationship types if the provided examples are insufficient, and briefly justify any novel types.

3.  **Network Analysis and Insights:**
    * Based on the extracted entities and relationships, identify and describe:
        * `Hub Entities`: Entities with the highest number of diverse connections, acting as central points in the workday. List the top 3-5.
        * `Bridge Entities`: Entities that uniquely connect disparate clusters of entities or topics that would otherwise be disconnected.
        * `Orphan Idea`: Concepts, tasks, or topics mentioned but not substantially connected to other entities or followed up upon.
        * `Temporal Flows`: Describe how attention and activity move through the network over time. For example, "Morning focused on Project X involving Person A and Tool Y, shifting to Concept Z discussions with Person B in the afternoon."

4.  **Output Format:**
    * **Part 1: Friendly Markdown Report:**
        * A list of all entities with their profiles (from step 1).
        * A list of all relationships (from step 2).
    * **Part 2: Narrative Analysis (Markdown):**
        * A summary of the key findings from the Network Analysis (step 3).
        * A qualitative description of what a visual network diagram of this day would highlight. Include specific examples of the 2-3 most interesting or unexpected connections discovered, explaining why they are noteworthy (e.g., "An interesting connection is Person A using Tool Z, typically associated with Project Q, for an ad-hoc task related to Concept R. This suggests a novel application or workaround.").

**Key Considerations:**
* Synthesize information from all transcript content within each chunk.
* Disambiguate entities by consolidating name variants to a single canonical entity using the most complete name when the same entity is referenced by different names within the transcript (e.g., "John", "John D.", "John Doe" → use "John Doe" throughout).
* When first-name-only references are ambiguous, note the ambiguity in the entity's Context field rather than guessing identity.
* Cross-reference names against attendees and participants mentioned earlier in the transcript for spelling corrections and consistent naming.
* Infer implicit relationships where explicit statements are lacking but context strongly suggests a connection.
* Focus on the most relevant and significant entities and relationships to avoid an overly noisy graph.
* For live capture, $preferred often multi-tasks — e.g., joined on a team zoom in the background while working on an unrelated task — so different content streams may not always align.
* Take time to consider all of the nuance of the interactions from the day, deeply think through how best to prioritize the most important aspects and understandings, formulate the best approach for each step of the analysis.
