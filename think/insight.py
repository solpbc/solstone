import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from think.cluster import cluster, cluster_period
from think.crumbs import CrumbBuilder
from think.models import GEMINI_FLASH, GEMINI_PRO, gemini_generate
from think.utils import (
    PromptNotFoundError,
    day_log,
    day_path,
    get_insight_topic,
    get_insights,
    load_prompt,
    setup_cli,
)

COMMON_SYSTEM_INSTRUCTION = "You are an expert productivity analyst tasked with analyzing a full workday transcript containing both audio conversations and screen activity data, segmented into 5-minute chunks. You will be given the transcripts and then following that you will have a detailed user request for how to process them.  Please follow those instructions carefully. Take time to consider all of the nuance of the interactions from the day, deeply think through how best to prioritize the most important aspects and understandings, formulate the best approach for each step of the analysis."


def _insight_keys() -> list[str]:
    """Return available insight keys."""
    return sorted(get_insights().keys())


def _output_paths(
    day_dir: os.PathLike[str], key: str, segment: str | None = None
) -> tuple[Path, Path]:
    """Return markdown and JSON output paths for insight ``key`` in ``day_dir``.

    Args:
        day_dir: Day directory path (YYYYMMDD)
        key: Insight key (e.g., "activity" or "chat:sentiment")
        segment: Optional segment key (HHMMSS_LEN)

    Returns:
        (md_path, json_path) tuple
        - Daily: YYYYMMDD/insights/{topic}.md (where topic = get_insight_topic(key))
        - Segment: YYYYMMDD/{segment}/{topic}.md
    """
    day = Path(day_dir)
    topic = get_insight_topic(key)

    if segment:
        # Segment insights go directly in segment directory
        segment_dir = day / segment
        return segment_dir / f"{topic}.md", segment_dir / f"{topic}.json"
    else:
        # Daily insights go in insights/ subdirectory
        insights_dir = day / "insights"
        return insights_dir / f"{topic}.md", insights_dir / f"{topic}.json"


def scan_day(day: str) -> dict[str, list[str]]:
    """Return lists of processed and pending insight markdown files."""
    day_dir = day_path(day)
    processed: list[str] = []
    pending: list[str] = []
    for key in _insight_keys():
        md_path, _ = _output_paths(day_dir, key)
        if md_path.exists():
            processed.append(os.path.join("insights", md_path.name))
        else:
            pending.append(os.path.join("insights", md_path.name))
    return {"processed": sorted(processed), "repairable": sorted(pending)}


def count_tokens(markdown: str, prompt: str, api_key: str, model: str) -> None:
    client = genai.Client(api_key=api_key)

    total_tokens = client.models.count_tokens(
        model=model,
        contents=[markdown],
    )
    print(f"Token count: {total_tokens}")


def _get_or_create_cache(
    client: genai.Client, model: str, display_name: str, transcript: str
) -> str | None:
    """Return cache name for ``display_name`` or None if content too small.

    Creates cache with ``transcript`` and :data:`COMMON_SYSTEM_INSTRUCTION` if needed.
    Returns None if content is below estimated 2048 token minimum (~10k chars).

    The cache contains the system instruction + transcript which are identical
    for all topics on the same day, so display_name should be day-based only."""

    MIN_CACHE_CHARS = 10000  # Heuristic: ~4 chars/token → 2048 tokens ≈ 8k-10k chars

    # Check existing caches first
    for c in client.caches.list():
        if c.model == model and c.display_name == display_name:
            return c.name

    # Skip cache creation for small content
    if len(transcript) < MIN_CACHE_CHARS:
        return None

    cache = client.caches.create(
        model=model,
        config=types.CreateCachedContentConfig(
            display_name=display_name,
            system_instruction=COMMON_SYSTEM_INSTRUCTION,
            contents=[transcript],
            ttl="1800s",  # 30 minutes to accommodate multiple topic analyses
        ),
    )
    return cache.name


def send_markdown(
    markdown: str,
    prompt: str,
    api_key: str,
    model: str,
    cache_display_name: str | None = None,
) -> str:
    client = genai.Client(api_key=api_key) if cache_display_name else None

    if cache_display_name:
        cache_name = _get_or_create_cache(client, model, cache_display_name, markdown)

        if cache_name:
            # Cache created successfully, use it
            contents: list[str] = [prompt]
            return gemini_generate(
                contents=contents,
                model=model,
                temperature=0.3,
                max_output_tokens=8192 * 6,
                thinking_budget=8192 * 3,
                cached_content=cache_name,
                client=client,
            )
        else:
            # Content too small for caching, proceed without cache
            contents = [markdown, prompt]
            return gemini_generate(
                contents=contents,
                model=model,
                temperature=0.3,
                max_output_tokens=8192 * 6,
                thinking_budget=8192 * 3,
                system_instruction=COMMON_SYSTEM_INSTRUCTION,
            )
    else:
        contents = [markdown, prompt]
        return gemini_generate(
            contents=contents,
            model=model,
            temperature=0.3,
            max_output_tokens=8192 * 6,
            thinking_budget=8192 * 3,
            system_instruction=COMMON_SYSTEM_INSTRUCTION,
        )


def send_extraction(
    markdown: str,
    prompt: str,
    api_key: str,
    model: str,
    extra_instructions: str | None = None,
) -> list:
    """Extract structured JSON events from markdown summary.

    Used for both occurrences (past events) and anticipations (future events).

    Parameters
    ----------
    markdown:
        Markdown summary to extract events from.
    prompt:
        System instruction guiding the extraction.
    api_key:
        Google API key for authentication.
    model:
        Gemini model name.
    extra_instructions:
        Optional additional instructions prepended to ``markdown``.

    Returns
    -------
    list
        Array of extracted event objects.
    """
    contents = [markdown]
    if extra_instructions:
        contents.insert(0, extra_instructions)

    response_text = gemini_generate(
        contents=contents,
        model=model,
        temperature=0.3,
        max_output_tokens=8192 * 6,
        thinking_budget=8192 * 3,
        system_instruction=prompt,
        json_output=True,
    )

    try:
        events = json.loads(response_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response: {e}: {response_text[:100]}")

    if not isinstance(events, list):
        raise ValueError(f"Response is not an array: {response_text[:100]}")

    return events


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send a day's clustered Markdown to Gemini for analysis."
    )
    parser.add_argument(
        "day",
        help="Day in YYYYMMDD format",
    )
    parser.add_argument(
        "-f",
        "--topic",
        "--prompt",
        dest="topic",
        required=True,
        help="Insight key (e.g., 'activity', 'chat:sentiment') or path to .txt file",
    )
    parser.add_argument(
        "-p",
        "--pro",
        action="store_true",
        help="Use the gemini 2.5 pro model",
    )
    parser.add_argument(
        "-c",
        "--count",
        action="store_true",
        help="Count tokens only and exit",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it already exists",
    )
    parser.add_argument(
        "--segment",
        help="Segment key in HHMMSS_LEN format (processes only this segment within the day)",
    )
    args = setup_cli(parser)

    # Resolve insight key or path to metadata
    all_insights = get_insights()
    topic_arg = args.topic

    # Check if it's a known insight key first
    if topic_arg in all_insights:
        insight_key = topic_arg
        insight_meta = all_insights[insight_key]
        insight_path = Path(insight_meta["path"])
    elif Path(topic_arg).exists():
        # Fall back to treating it as a file path (backwards compat)
        insight_path = Path(topic_arg)
        # Try to find matching key by path
        insight_key = insight_path.stem
        for key, meta in all_insights.items():
            if meta.get("path") == str(insight_path):
                insight_key = key
                break
        insight_meta = all_insights.get(insight_key, {})
    else:
        parser.error(
            f"Insight not found: {topic_arg}. "
            f"Available: {', '.join(sorted(all_insights.keys()))}"
        )

    extra_occ = insight_meta.get("occurrences")
    skip_occ = extra_occ is False
    do_anticipations = insight_meta.get("anticipations") is True
    success = False

    # Choose clustering function based on mode
    if args.segment:
        markdown, file_count = cluster_period(args.day, args.segment)
    else:
        markdown, file_count = cluster(args.day)
    day_dir = str(day_path(args.day))

    try:

        load_dotenv()
        if args.verbose:
            print("Verbose mode enabled")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            parser.error("GOOGLE_API_KEY not found in environment")

        try:
            insight_prompt = load_prompt(
                insight_path.stem, base_dir=insight_path.parent, include_journal=True
            )
        except PromptNotFoundError:
            parser.error(f"Insight file not found: {insight_path}")

        prompt = insight_prompt.text

        model = GEMINI_PRO if args.pro else GEMINI_FLASH
        day = args.day
        size_kb = len(markdown.encode("utf-8")) / 1024

        print(
            f"Topic: {insight_key} | Model: {model} | Day: {day} | Files: {file_count} | Size: {size_kb:.1f}KB"
        )

        if args.count:
            count_tokens(markdown, prompt, api_key, model)
            return

        md_path, json_path = _output_paths(day_dir, insight_key, segment=args.segment)
        # Use cache key scoped to day or segment
        if args.segment:
            cache_display_name = f"{day}_{args.segment}"
        else:
            cache_display_name = f"{day}"

        # Check if markdown file already exists
        md_exists = md_path.exists() and md_path.stat().st_size > 0

        if md_exists and not args.force:
            print(f"Markdown file already exists: {md_path}. Loading existing content.")
            with open(md_path, "r") as f:
                result = f.read()
        elif md_exists and args.force:
            print("Markdown file exists but --force specified. Regenerating.")
            result = send_markdown(
                markdown,
                prompt,
                api_key,
                model,
                cache_display_name=cache_display_name,
            )
        else:
            result = send_markdown(
                markdown,
                prompt,
                api_key,
                model,
                cache_display_name=cache_display_name,
            )

        # Check if we got a valid response
        if result is None:
            print("Error: No text content in response")
            return

        # Only write markdown if it was newly generated
        if not md_exists or args.force:
            os.makedirs(md_path.parent, exist_ok=True)
            with open(md_path, "w") as f:
                f.write(result)
            print(f"Results saved to: {md_path}")

            crumb_builder = (
                CrumbBuilder()
                .add_file(str(insight_prompt.path))
                .add_glob(os.path.join(day_dir, "*/audio.jsonl"))
                .add_glob(os.path.join(day_dir, "*/*_audio.jsonl"))  # Split audio
                .add_glob(os.path.join(day_dir, "*/screen.md"))
                .add_model(model)
            )
            crumb_path = crumb_builder.commit(str(md_path))
            print(f"Crumb saved to: {crumb_path}")

        if skip_occ and not do_anticipations:
            print('"occurrences" set to false; skipping JSON generation')
            success = True
            return

        # Determine which prompt to use: anticipations or occurrences
        if do_anticipations:
            prompt_name = "anticipate"
            array_key = "anticipations"
        else:
            prompt_name = "summarize"
            array_key = "occurrences"

        # Load the appropriate extraction prompt
        try:
            extraction_prompt_content = load_prompt(
                prompt_name, base_dir=Path(__file__).parent, include_journal=True
            )
        except PromptNotFoundError as exc:
            print(exc)
            return

        extraction_prompt = extraction_prompt_content.text

        json_output_path = json_path
        json_exists = json_output_path.exists() and json_output_path.stat().st_size > 0

        if json_exists and not args.force:
            print(
                f"JSON file already exists: {json_output_path}. Use --force to overwrite."
            )
            return
        elif json_exists and args.force:
            print("JSON file exists but --force specified. Regenerating.")

        try:
            # Load facet summaries and combine with topic-specific instructions
            from think.facets import facet_summaries

            facets_context = facet_summaries(detailed_entities=True)

            # Combine facet summaries with topic-specific instructions
            if extra_occ and not do_anticipations:
                combined_instructions = f"{facets_context}\n\n{extra_occ}"
            else:
                combined_instructions = facets_context

            events = send_extraction(
                result,
                extraction_prompt,
                api_key,
                model,
                extra_instructions=combined_instructions,
            )
        except ValueError as e:
            print(f"Error: {e}")
            return

        # Build output JSON with appropriate array key
        if args.segment:
            full_obj = {
                "day": day,
                "segment": args.segment,
                array_key: events,
            }
        else:
            full_obj = {"day": day, array_key: events}

        json_result = json.dumps(full_obj, indent=2)

        os.makedirs(json_output_path.parent, exist_ok=True)
        with open(json_output_path, "w") as f:
            f.write(json_result)

        print(f"Results saved to: {json_output_path}")

        json_crumb_builder = (
            CrumbBuilder()
            .add_file(str(extraction_prompt_content.path))
            .add_file(md_path)
            .add_model(model)
        )
        json_crumb_path = json_crumb_builder.commit(str(json_output_path))
        print(f"Crumb saved to: {json_crumb_path}")
        success = True

    finally:
        msg = f"insight {insight_key} {'ok' if success else 'failed'}"
        if args.force:
            msg += " --force"
        day_log(args.day, msg)


if __name__ == "__main__":
    main()
