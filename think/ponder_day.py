import argparse
import os
import sys
import threading
import time

# Add parent directory to path for module discovery
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from google import genai
from google.genai import types

from think.cluster_day import cluster_day

DEFAULT_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "ponder_day.txt")

FLASH_MODEL = "gemini-2.5-flash-preview-05-20"
PRO_MODEL = "gemini-2.5-pro-preview-05-06"


def count_tokens(markdown: str, prompt: str, api_key: str, model: str) -> None:
    client = genai.Client(api_key=api_key)

    total_tokens = client.models.count_tokens(
        model=model,
        contents=[markdown],
    )
    print(f"Token count: {total_tokens}")


def send_markdown(markdown: str, prompt: str, api_key: str, model: str) -> tuple[str, object]:
    client = genai.Client(api_key=api_key)

    done = threading.Event()

    def progress():
        elapsed = 0
        while not done.is_set():
            time.sleep(5)
            elapsed += 5
            if not done.is_set():
                print(f"... {elapsed}s elapsed")

    t = threading.Thread(target=progress, daemon=True)
    t.start()
    try:
        response = client.models.generate_content(
            model=model,
            contents=[markdown],
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=8192,
                system_instruction=prompt,
            ),
        )
        return response.text, response.usage_metadata
    finally:
        done.set()
        t.join()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send a day's clustered Markdown to Gemini for analysis."
    )
    parser.add_argument(
        "folder",
        help="Directory containing the day's folder",
    )
    parser.add_argument(
        "-f",
        "--prompt",
        default=DEFAULT_PROMPT_PATH,
        help="Prompt file to use",
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
    args = parser.parse_args()

    markdown, file_count = cluster_day(args.folder)

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        parser.error("GOOGLE_API_KEY not found in environment")

    try:
        with open(args.prompt, "r") as f:
            prompt = f.read().strip()
    except FileNotFoundError:
        parser.error(f"Prompt file not found: {args.prompt}")

    model = PRO_MODEL if args.pro else FLASH_MODEL
    day = os.path.basename(os.path.normpath(args.folder))
    size_kb = len(markdown.encode("utf-8")) / 1024
    
    print(
        f"Prompt: {args.prompt} | Model: {model} | Day: {day} | Files: {file_count} | Size: {size_kb:.1f}KB"
    )

    if args.count:
        count_tokens(markdown, prompt, api_key, model)
        return

    result, usage_metadata = send_markdown(markdown, prompt, api_key, model)
    print(f"Usage: {usage_metadata}")
    
    # Check if we got a valid response
    if result is None:
        print("Error: No text content in response")
        return
    
    # Create output filename and save result
    prompt_basename = os.path.splitext(os.path.basename(args.prompt))[0]
    output_filename = f"{prompt_basename}_{model}.md"
    output_path = os.path.join(args.folder, output_filename)
    
    os.makedirs(args.folder, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(result)
    
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
