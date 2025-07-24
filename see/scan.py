#!/usr/bin/env python3
import argparse
import datetime
import json
import logging
import os
import tempfile
import time

from PIL import Image, ImageDraw
from PIL.PngImagePlugin import PngInfo

from see.screen_compare import compare_images
from see.screen_dbus import idle_time_ms, screen_snap
from think.detect_border import detect_border
from think.utils import setup_cli, touch_health

BLUE_BORDER = (0, 0, 255)


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
    touch_health("see")


def recent_audio_activity(journal: str, window: int = 120) -> bool:
    """Return True if an *_audio.json file was modified in the last ``window`` seconds."""
    day_dir = os.path.join(journal, datetime.datetime.now().strftime("%Y%m%d"))
    if not os.path.isdir(day_dir):
        return False
    cutoff = time.time() - window
    for name in os.listdir(day_dir):
        if not name.endswith("_audio.json"):
            continue
        path = os.path.join(day_dir, name)
        try:
            if os.path.getmtime(path) >= cutoff:
                return True
        except OSError:
            continue
    return False


def censor_border(img: Image.Image) -> Image.Image:
    """Black out the region inside a detected blue border."""
    try:
        y_min, x_min, y_max, x_max = detect_border(img, BLUE_BORDER)
    except ValueError:
        # Silently ignore when no border is detected or border not thick enough
        return img
    except Exception as e:
        # Log unexpected errors
        logging.error("Unexpected error detecting border: %s", e)
        return img
    logging.debug("Detected border at: %s, %s, %s, %s", y_min, x_min, y_max, x_max)
    censored = img.copy()
    draw = ImageDraw.Draw(censored)
    draw.rectangle(((x_min, y_min), (x_max, y_max)), fill="black")
    return censored


def process_once(journal, min_threshold):
    if not os.path.exists(journal):
        os.makedirs(journal)

    prev_images, last_ts = load_cache()

    recent_audio = recent_audio_activity(journal)
    idle_ms = idle_time_ms()
    if not recent_audio and last_ts and idle_ms / 1000 >= (time.time() - last_ts):
        logging.debug("Desktop still idle; nothing to do.")
        return

    try:
        monitor_images = screen_snap()
    except Exception as e:
        logging.error("Error taking screenshot: %s", e)
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
                key=lambda b: (b["box_2d"][3] - b["box_2d"][1])
                * (b["box_2d"][2] - b["box_2d"][0]),
            )
            y_min, x_min, y_max, x_max = largest_box["box_2d"]
            box_width = x_max - x_min
            box_height = y_max - y_min
            if box_width > min_threshold and box_height > min_threshold:
                logging.debug(
                    "[Monitor %s] Detected significant difference: %s", idx, largest_box
                )
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                date_part, time_part = timestamp.split("_", 1)
                day_dir = os.path.join(journal, date_part)
                os.makedirs(day_dir, exist_ok=True)
                base = os.path.join(day_dir, f"{time_part}_monitor_{idx}_diff")
                img_filename = base + ".png"

                # Add box_2d to PNG metadata
                pnginfo = PngInfo()
                pnginfo.add_text("box_2d", json.dumps(largest_box["box_2d"]))

                # Atomically save the image
                with tempfile.NamedTemporaryFile(
                    dir=os.path.dirname(img_filename), suffix=".pngtmp", delete=False
                ) as tf:
                    censored_img.save(tf, format="PNG", pnginfo=pnginfo)
                os.replace(tf.name, img_filename)
                touch_health("see")

                logging.info("[Monitor %s] Saved diff image: %s", idx, img_filename)
                prev_images[idx] = censored_img
            else:
                logging.debug(
                    "[Monitor %s] Difference detected but largest box %s x %s is < %s.",
                    idx,
                    box_width,
                    box_height,
                    min_threshold,
                )
        else:
            logging.debug("[Monitor %s] No significant change detected.", idx)

    save_cache(prev_images)


def main():
    parser = argparse.ArgumentParser(
        description="Capture screenshots once and compare with cached versions."
    )
    parser.add_argument(
        "--min",
        type=int,
        default=400,
        help="Minimum size threshold for a bounding box (pixels)",
    )
    args = setup_cli(parser)
    journal = os.getenv("JOURNAL_PATH")
    process_once(journal, args.min)


if __name__ == "__main__":
    main()
