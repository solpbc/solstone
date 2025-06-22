import argparse
import os
import time
import json
from PIL import Image
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from see import gemini_look


def detect_red_box(image):
    """Return [y_min, x_min, y_max, x_max] of the red rectangle in image."""
    try:
        arr = np.array(image.convert("RGB"))
    except (OSError, Exception) as e:
        print(f"Error processing image data: {e}")
        return None
    mask = (arr[:, :, 0] > 200) & (arr[:, :, 1] < 80) & (arr[:, :, 2] < 80)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return [int(y_min), int(x_min), int(y_max), int(x_max)]


def find_missing(day_dir):
    if not os.path.isdir(day_dir):
        raise FileNotFoundError(f"Day directory not found: {day_dir}")
    missing = []
    for name in sorted(os.listdir(day_dir)):
        if name.endswith("_diff.png"):
            base = name[:-4]
            json_path = os.path.join(day_dir, base + ".json")
            if not os.path.exists(json_path):
                png_path = os.path.join(day_dir, name)
                missing.append((png_path, json_path))
    return missing


def process_files(files, delay, models=None):
    if not gemini_look.initialize():
        print("Failed to initialize Gemini API")
        return
    for png_path, json_path in files:
        try:
            image = Image.open(png_path)
        except (OSError, Exception) as e:
            print(f"Could not open {png_path} (corrupted/truncated image): {e}")
            continue
        box_coords = detect_red_box(image)
        if not box_coords:
            print(f"No red box found in {png_path}; skipping")
            continue
        result = gemini_look.gemini_describe_region(image, {"box_2d": box_coords}, models)
        if result:
            with open(json_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Saved {json_path}")
        else:
            print(f"Gemini returned no result for {png_path}")
        time.sleep(delay)


def main():
    parser = argparse.ArgumentParser(description="Repair missing Gemini JSON for screenshot diffs")
    parser.add_argument("day_dir", help="Day directory path containing screenshot files")
    parser.add_argument("--wait", type=float, default=0, help="Seconds to wait between API calls (default: 0)")
    parser.add_argument("-p", "--pro", action="store_true", help="Use pro models instead of default models")
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
