import argparse
import os
import time
import json
from PIL import Image
import numpy as np
from see import gemini_look


def detect_red_box(image):
    """Return [y_min, x_min, y_max, x_max] of the red rectangle in image."""
    arr = np.array(image.convert("RGB"))
    mask = (arr[:, :, 0] > 200) & (arr[:, :, 1] < 80) & (arr[:, :, 2] < 80)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return [int(y_min), int(x_min), int(y_max), int(x_max)]


def find_missing(folder, day):
    day_dir = os.path.join(folder, day)
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


def process_files(files, delay):
    if not gemini_look.initialize():
        print("Failed to initialize Gemini API")
        return
    for png_path, json_path in files:
        try:
            image = Image.open(png_path)
        except Exception as e:
            print(f"Could not open {png_path}: {e}")
            continue
        box_coords = detect_red_box(image)
        if not box_coords:
            print(f"No red box found in {png_path}; skipping")
            continue
        result = gemini_look.gemini_describe_region(image, {"box_2d": box_coords})
        if result:
            with open(json_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Saved {json_path}")
        else:
            print(f"Gemini returned no result for {png_path}")
        time.sleep(delay)


def main():
    parser = argparse.ArgumentParser(description="Repair missing Gemini JSON for screenshot diffs")
    parser.add_argument("folder", help="Base directory containing day folders")
    parser.add_argument("day", help="Day folder (YYYYMMDD)")
    args = parser.parse_args()

    try:
        missing = find_missing(args.folder, args.day)
    except FileNotFoundError as e:
        print(str(e))
        return

    if not missing:
        print("No missing JSON files found.")
        return

    print(f"Found {len(missing)} missing JSON files.")
    try:
        delay = float(input("Seconds to wait between API calls (0 to cancel): "))
    except ValueError:
        print("Invalid input; aborting")
        return
    if delay <= 0:
        print("Aborted")
        return

    process_files(missing, delay)


if __name__ == "__main__":
    main()
