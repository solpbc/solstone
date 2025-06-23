import argparse
import json
import os
import sys
import time

from PIL import Image

from think.border_detect import detect_border

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from see import gemini_look
from think.crumbs import CrumbBuilder


def find_missing(day_dir):
    if not os.path.isdir(day_dir):
        raise FileNotFoundError(f"Day directory not found: {day_dir}")
    missing = []
    for name in sorted(os.listdir(day_dir)):
        if name.endswith("_diff.png"):
            base = name[:-4]
            # Check for existing box JSON (from scan) and result JSON
            box_json_path = os.path.join(day_dir, base + "_box.json")
            result_json_path = os.path.join(day_dir, base + ".json")

            # Process if box JSON or result JSON doesn't exist
            if not os.path.exists(box_json_path) or not os.path.exists(result_json_path):
                png_path = os.path.join(day_dir, name)
                missing.append((png_path, box_json_path, result_json_path))
    return missing


def process_files(files, delay, models=None):
    if not gemini_look.initialize():
        print("Failed to initialize Gemini API")
        return
    for png_path, box_json_path, result_json_path in files:
        try:
            image = Image.open(png_path)
        except (OSError, Exception) as e:
            print(f"Could not open {png_path} (corrupted/truncated image): {e}")
            continue

        # Check if box JSON exists and use it, otherwise detect red box
        if os.path.exists(box_json_path):
            try:
                with open(box_json_path, "r") as f:
                    box_data = json.load(f)
                    box_coords = box_data.get("box_2d")
                print(f"Using existing box coordinates from {box_json_path}")
            except (OSError, json.JSONDecodeError) as e:
                print(f"Could not load box JSON {box_json_path}: {e}, detecting box")
                box_coords = None
        else:
            box_coords = None

        # If no box coords from JSON, detect red box
        if not box_coords:
            try:
                box_coords = detect_border(image, (255, 0, 0))
                # Save the detected box for future use
                box_data = {"box_2d": box_coords}
                with open(box_json_path, "w") as f:
                    json.dump(box_data, f, indent=2)
                print(f"Detected and saved box coordinates to {box_json_path}")
            except (ValueError, OSError) as e:
                print(f"Could not detect red box in {png_path}: {e}")
                continue

        # Only call Gemini API if result JSON doesn't exist
        if os.path.exists(result_json_path):
            print(f"Result JSON already exists: {result_json_path}")
            continue

        result = gemini_look.gemini_describe_region(image, {"box_2d": box_coords}, models)
        if result:
            with open(result_json_path, "w") as f:
                json.dump(result["result"], f, indent=2)
            print(f"Saved {result_json_path}")
            crumb_builder = (
                CrumbBuilder()
                .add_file(png_path)
                .add_file(box_json_path)
                .add_model(result["model_used"])
            )
            crumb_path = crumb_builder.commit(result_json_path)
            print(f"Crumb saved to: {crumb_path}")
        else:
            print(f"Gemini returned no result for {png_path}")
        time.sleep(delay)


def main():
    parser = argparse.ArgumentParser(description="Repair missing Gemini JSON for screenshot diffs")
    parser.add_argument("day_dir", help="Day directory path containing screenshot files")
    parser.add_argument(
        "--wait", type=float, default=0, help="Seconds to wait between API calls (default: 0)"
    )
    parser.add_argument(
        "-p", "--pro", action="store_true", help="Use pro models instead of default models"
    )
    args = parser.parse_args()

    try:
        missing = find_missing(args.day_dir)
    except FileNotFoundError as e:
        print(str(e))
        return

    if not missing:
        print(f"No missing JSON files found in {args.day_dir}.")
        return

    print(f"Found {len(missing)} missing JSON files.")

    models = ["gemini-2.5-pro", "gemini-2.5-flash"] if args.pro else None
    process_files(missing, args.wait, models)


if __name__ == "__main__":
    main()
