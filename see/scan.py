#!/usr/bin/env python3
import argparse
import datetime
import json
import os
import sys
import time

from PIL import Image, ImageDraw

from see.screen_compare import compare_images
from see.screen_dbus import idle_time_ms, screen_snap
from think.border_detect import detect_border

GLOBAL_VERBOSE = False
BLUE_BORDER = (0, 0, 255)


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


def censor_border(img: Image.Image) -> Image.Image:
    """Black out the region inside a detected blue border."""
    try:
        y_min, x_min, y_max, x_max = detect_border(img, BLUE_BORDER)
    except ValueError as e:
        # Silently ignore when no border is detected or border not thick enough
        return img
    except Exception as e:
        # Log unexpected errors
        log(f"Unexpected error detecting border: {e}", force=True)
        return img
    log(f"Detected border at: {y_min}, {x_min}, {y_max}, {x_max}")
    censored = img.copy()
    draw = ImageDraw.Draw(censored)
    draw.rectangle(((x_min, y_min), (x_max, y_max)), fill="black")
    return censored


def process_once(journal, min_threshold):
    if not os.path.exists(journal):
        os.makedirs(journal)

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
        censored_img = censor_border(pil_img)
        prev_img = prev_images.get(idx)
        if not prev_img or prev_img.size != censored_img.size:
            prev_images[idx] = censored_img
            continue

        boxes = compare_images(prev_img, censored_img)
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
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                date_part, time_part = timestamp.split("_", 1)
                day_dir = os.path.join(journal, date_part)
                os.makedirs(day_dir, exist_ok=True)
                base = os.path.join(day_dir, f"{time_part}_monitor_{idx}_diff")
                img_filename = base + ".png"
                censored_img.save(img_filename)
                log(
                    f"[Monitor {idx}] Saved diff image: {img_filename}",
                    force=True,
                )
                box_filename = base + "_box.json"
                with open(box_filename, "w") as bf:
                    json.dump(largest_box, bf)
                log(
                    f"[Monitor {idx}] Saved bounding box JSON: {box_filename}",
                    force=True,
                )
                prev_images[idx] = censored_img
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
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--min", type=int, default=400, help="Minimum size threshold for a bounding box (pixels)"
    )
    args = parser.parse_args()

    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        parser.error("JOURNAL_PATH not set")
    global GLOBAL_VERBOSE
    GLOBAL_VERBOSE = args.verbose

    process_once(journal, args.min)


if __name__ == "__main__":
    main()
