import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from think.cluster import cluster
from think.crumbs import CrumbBuilder
from think.models import GEMINI_FLASH, GEMINI_PRO, gemini_generate
from think.utils import (
    PromptNotFoundError,
    day_log,
    day_path,
    get_topics,
    load_prompt,
    setup_cli,
)

COMMON_SYSTEM_INSTRUCTION = "You are an expert productivity analyst tasked with analyzing a full workday transcript containing both audio conversations and screen activity data, segmented into 5-minute chunks. You will be given the transcripts and then following that you will have a detailed user request for how to process them.  Please follow those instructions carefully. Take time to consider all of the nuance of the interactions from the day, deeply think through how best to prioritize the most important aspects and understandings, formulate the best approach for each step of the analysis."


def _topic_basenames() -> list[str]:
    """Return available topic basenames under :data:`TOPICS`."""
    return sorted(get_topics().keys())


def _output_paths(day_dir: os.PathLike[str], basename: str) -> tuple[Path, Path]:
    """Return markdown and JSON output paths for ``basename`` in ``day_dir``."""
    day = Path(day_dir)
    topic_dir = day / "topics"
    return topic_dir / f"{basename}.md", topic_dir / f"{basename}.json"


def scan_day(day: str) -> dict[str, list[str]]:
    """Return lists of processed and pending summary markdown files."""
    day_dir = day_path(day)
    summarized: list[str] = []
    unsummarized: list[str] = []
    for base in _topic_basenames():
        md_path, _ = _output_paths(day_dir, base)
        if md_path.exists():
            summarized.append(os.path.join("topics", md_path.name))
        else:
            unsummarized.append(os.path.join("topics", md_path.name))
    return {"processed": sorted(summarized), "repairable": sorted(unsummarized)}


def count_tokens(markdown: str, prompt: str, api_key: str, model: str) -> None:
    client = genai.Client(api_key=api_key)

    total_tokens = client.models.count_tokens(
        model=model,
        contents=[markdown],
    )
    print(f"Token count: {total_tokens}")


def _get_or_create_cache(
    client: genai.Client, model: str, display_name: str, transcript: str
) -> str:
    """Return cache name for ``display_name`` creating it with ``transcript`` and
    :data:`COMMON_SYSTEM_INSTRUCTION` if needed.

    The cache contains the system instruction + transcript which are identical
    for all topics on the same day, so display_name should be day-based only."""

    for c in client.caches.list():
        if c.model == model and c.display_name == display_name:
            return c.name

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
        contents = [markdown, prompt]
        return gemini_generate(
            contents=contents,
            model=model,
            temperature=0.3,
            max_output_tokens=8192 * 6,
            thinking_budget=8192 * 3,
            system_instruction=COMMON_SYSTEM_INSTRUCTION,
        )


def send_occurrence(
    markdown: str,
    prompt: str,
    api_key: str,
    model: str,
    extra_instructions: str | None = None,
) -> object:
    """Send markdown to generate occurrence data and return parsed JSON.

    Parameters
    ----------
    markdown:
        Markdown summary to convert into occurrences.
    prompt:
        System instruction guiding the model.
    api_key:
        Google API key for authentication.
    model:
        Gemini model name.
    extra_instructions:
        Optional additional instructions prepended to ``markdown``.
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
        occurrences = json.loads(response_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response: {e}: {response_text[:100]}")

    if not isinstance(occurrences, list):
        raise ValueError(f"Response is not an array: {response_text[:100]}")

    return occurrences


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
        help="Topic file to use (required)",
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
    args = setup_cli(parser)

    markdown, file_count = cluster(args.day)
    day_dir = str(day_path(args.day))
    topic_basename = Path(args.topic).stem
    topic_meta = get_topics().get(topic_basename, {})
    extra_occ = topic_meta.get("occurrences")
    skip_occ = extra_occ is False
    success = False

    try:

        load_dotenv()
        if args.verbose:
            print("Verbose mode enabled")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            parser.error("GOOGLE_API_KEY not found in environment")

        topic_path = Path(args.topic)
        try:
            topic_prompt = load_prompt(
                topic_path.stem, base_dir=topic_path.parent, include_journal=True
            )
        except PromptNotFoundError:
            parser.error(f"Topic file not found: {topic_path}")

        prompt = topic_prompt.text

        model = GEMINI_PRO if args.pro else GEMINI_FLASH
        day = args.day
        size_kb = len(markdown.encode("utf-8")) / 1024

        print(
            f"Topic: {args.topic} | Model: {model} | Day: {day} | Files: {file_count} | Size: {size_kb:.1f}KB"
        )

        if args.count:
            count_tokens(markdown, prompt, api_key, model)
            return

        md_path, json_path = _output_paths(day_dir, topic_basename)
        # Use day-only cache key so all topics share the same cached transcript
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
                .add_file(str(topic_prompt.path))
                .add_glob(os.path.join(day_dir, "*_audio.jsonl"))
                .add_glob(os.path.join(day_dir, "*_screen.md"))
                .add_model(model)
            )
            crumb_path = crumb_builder.commit(str(md_path))
            print(f"Crumb saved to: {crumb_path}")

        if skip_occ:
            print('"occurrences" set to false; skipping occurrence generation')
            success = True
            return

        # Create a corresponding occurrence JSON from the markdown summary
        try:
            occ_prompt_content = load_prompt(
                "summarize", base_dir=Path(__file__).parent, include_journal=True
            )
        except PromptNotFoundError as exc:
            print(exc)
            return

        occ_prompt = occ_prompt_content.text.replace("DAY", day)

        occ_output_path = json_path
        json_exists = occ_output_path.exists() and occ_output_path.stat().st_size > 0

        if json_exists and not args.force:
            print(
                f"JSON file already exists: {occ_output_path}. Use --force to overwrite."
            )
            return
        elif json_exists and args.force:
            print("JSON file exists but --force specified. Regenerating.")

        try:
            # Load facet summaries and combine with topic-specific occurrence instructions
            from think.facets import facet_summaries

            facets_context = facet_summaries(detailed_entities=True)

            # Combine facet summaries with topic-specific instructions
            if extra_occ:
                combined_instructions = f"{facets_context}\n\n{extra_occ}"
            else:
                combined_instructions = facets_context

            occurrences = send_occurrence(
                result,
                occ_prompt,
                api_key,
                model,
                extra_instructions=combined_instructions,
            )
        except ValueError as e:
            print(f"Error: {e}")
            return

        full_occurrence_obj = {"day": day, "occurrences": occurrences}

        occ_result = json.dumps(full_occurrence_obj, indent=2)

        os.makedirs(occ_output_path.parent, exist_ok=True)
        with open(occ_output_path, "w") as f:
            f.write(occ_result)

        print(f"Results saved to: {occ_output_path}")

        occ_crumb_builder = (
            CrumbBuilder()
            .add_file(str(occ_prompt_content.path))
            .add_file(md_path)
            .add_model(model)
        )
        occ_crumb_path = occ_crumb_builder.commit(str(occ_output_path))
        print(f"Crumb saved to: {occ_crumb_path}")
        success = True

    finally:
        msg = f"summarize {topic_basename} {'ok' if success else 'failed'}"
        if args.force:
            msg += " --force"
        day_log(args.day, msg)


if __name__ == "__main__":
    main()
