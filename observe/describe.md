You have one job: identify the primary foreground and (if present) secondary app categories in this desktop screenshot, and return ONLY this JSON:

{
  "visual_description":"<1–2 sentences describing what is visible>",
  "primary": "<largest and most visible app category>",
  "secondary": "<second most visible app category or 'none'>",
  "overlap": <boolean, does the primary overlap or cover the secondary, or is it fully standalone and separate>
}

Rules:
- For visual_description summarize the **overall desktop view** in **1–2 sentences** for a visually impaired user, focus on layout, window arrangement, and types of content.
- For the most visible primary foreground app choose the best category from the list below.
- Set "secondary" to "none" and "overlap" to true if the primary effectively fills the screen or no distinct second category/window is visible.
- Set overlap to true if the primary app overlaps, covers, clips, or obscures the secondary in any way.
- Only set a category for secondary if it is very visible and occupies more than 30% of the screen.

Categories (choose one):
$categories
