#!/usr/bin/env python3
import argparse
import os
import time
import datetime
import json
import sys
from PIL import Image, ImageDraw
from screen_dbus import screen_snap, idle_time_ms
from screen_compare import compare_images
import gemini_look

GLOBAL_VERBOSE = False


def log(message, force=False):
    if GLOBAL_VERBOSE or force:
        print(message)


CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "sunstone", "see")


def load_cache():
    """Load previous monitor images and last run timestamp from the cache."""
    images = {}
    last_ts = 0
    if os.path.exists(CACHE_DIR):
        for name in os.listdir(CACHE_DIR):
            if name.startswith("monitor_") and name.endswith(".png"):
                try:
                    idx = int(name.split("_")[1].split(".")[0])
                    path = os.path.join(CACHE_DIR, name)
                    with Image.open(path) as im:
                        images[idx] = im.copy()
                except Exception:
                    pass
        last_path = os.path.join(CACHE_DIR, "last")
        if os.path.exists(last_path):
            try:
                with open(last_path, "r") as f:
                    last_ts = float(f.read().strip())
            except Exception:
                last_ts = 0
    else:
        os.makedirs(CACHE_DIR)
    return images, last_ts


def save_cache(images):
    """Persist monitor images and update the last run timestamp."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    for idx, img in images.items():
        img.save(os.path.join(CACHE_DIR, f"monitor_{idx}.png"))
    with open(os.path.join(CACHE_DIR, "last"), "w") as f:
        f.write(str(time.time()))


def process_once(output_dir, min_threshold, use_gemini):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    prev_images, last_ts = load_cache()

    idle_ms = idle_time_ms()
    if last_ts and idle_ms / 1000 >= (time.time() - last_ts):
        log("Desktop still idle; nothing to do.")
        return

    try:
        monitor_images = screen_snap()
    except Exception as e:
        log(f"Error taking screenshot: {e}", force=True)
        return

    for idx, pil_img in enumerate(monitor_images, start=1):
        prev_img = prev_images.get(idx)
        if not prev_img or prev_img.size != pil_img.size:
            prev_images[idx] = pil_img
            continue

        boxes = compare_images(prev_img, pil_img)
        if boxes:
            largest_box = max(
                boxes,
                key=lambda b: (b["box_2d"][3] - b["box_2d"][1]) * (b["box_2d"][2] - b["box_2d"][0]),
            )
            y_min, x_min, y_max, x_max = largest_box["box_2d"]
            box_width = x_max - x_min
            box_height = y_max - y_min
            if box_width > min_threshold and box_height > min_threshold:
                log(f"[Monitor {idx}] Detected significant difference: {largest_box}")
                annotated = pil_img.copy()
                draw = ImageDraw.Draw(annotated)
                draw.rectangle(((x_min, y_min), (x_max, y_max)), outline="red", width=3)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                date_part, time_part = timestamp.split("_", 1)
                day_dir = os.path.join(output_dir, date_part)
                os.makedirs(day_dir, exist_ok=True)
                base = os.path.join(day_dir, f"{time_part}_monitor_{idx}_diff")
                img_filename = base + ".png"
                annotated.save(img_filename)
                log(
                    f"[Monitor {idx}] Saved annotated diff image: {img_filename}",
                    force=True,
                )
                if use_gemini:
                    result = gemini_look.gemini_describe_region(pil_img, largest_box)
                    if result:
                        json_filename = base + ".json"
                        with open(json_filename, "w") as jf:
                            json.dump(result, jf, indent=2)
                        log(
                            f"[Monitor {idx}] Saved Gemini JSON result: {json_filename}",
                            force=True,
                        )
                prev_images[idx] = pil_img
            else:
                log(
                    f"[Monitor {idx}] Difference detected but largest box {box_width}x{box_height} is < {min_threshold}."
                )
        else:
            log(f"[Monitor {idx}] No significant change detected.")

    save_cache(prev_images)


def main():
    parser = argparse.ArgumentParser(
        description="Capture screenshots once and compare with cached versions."
    )
    parser.add_argument("directory", type=str, help="Directory to save screenshots")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--min", type=int, default=400, help="Minimum size threshold for a bounding box (pixels)"
    )
    parser.add_argument(
        "-g", "--gemini", action="store_true", help="Call Gemini API when differences are detected"
    )
    args = parser.parse_args()
    global GLOBAL_VERBOSE
    GLOBAL_VERBOSE = args.verbose

    if args.gemini:
        try:
            gemini_look.initialize()
            log("Gemini API client initialized successfully", force=True)
        except Exception as e:
            log(f"Failed to initialize Gemini API: {str(e)}", force=True)
            sys.exit(1)

    process_once(args.directory, args.min, args.gemini)


if __name__ == "__main__":
    main()
