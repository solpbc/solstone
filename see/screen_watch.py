#!/usr/bin/env python3
import argparse
import os
import time
import datetime
import json
import sys  # for sys.exit
from PIL import ImageDraw
from screen_dbus import screen_snap
from screen_compare import compare_images
import gemini_look

GLOBAL_VERBOSE = False

# Top-level logging utility using global verbosity
def log(message, force=False):
    if GLOBAL_VERBOSE or force:
        print(message)

def process_screenshots(interval, output_dir, min_threshold, use_gemini):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Store previous PIL images for each monitor.
    prev_images = {}

    while True:
        try:
            monitor_images = screen_snap()
        except Exception as e:
            log(f"Error taking screenshot: {e}", force=True)
            time.sleep(interval)
            continue
        for idx, pil_img in enumerate(monitor_images, start=1):
            if idx not in prev_images:
                prev_images[idx] = pil_img
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(output_dir, f"monitor_{idx}_{timestamp}.png")
                pil_img.save(filename)
                log(f"[Monitor {idx}] Initial screenshot saved: {filename}")
            else:
                prev_img = prev_images[idx]
                if prev_img.size != pil_img.size:
                    log(f"[Monitor {idx}] Size mismatch; updating reference image.")
                    prev_images[idx] = pil_img
                    continue

                boxes = compare_images(prev_img, pil_img)
                if boxes:
                    # Select the largest bounding box.
                    largest_box = max(boxes, key=lambda b: (b["box_2d"][3] - b["box_2d"][1]) * (b["box_2d"][2] - b["box_2d"][0]))
                    y_min, x_min, y_max, x_max = largest_box["box_2d"]
                    box_width = x_max - x_min
                    box_height = y_max - y_min
                    if box_width > min_threshold and box_height > min_threshold:
                        log(f"[Monitor {idx}] Detected significant difference: {largest_box}")
                        annotated = pil_img.copy()
                        draw = ImageDraw.Draw(annotated)
                        draw.rectangle(((x_min, y_min), (x_max, y_max)), outline="red", width=3)
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = os.path.join(output_dir, f"monitor_{idx}_{timestamp}_diff.png")
                        annotated.save(filename)
                        log(f"[Monitor {idx}] Saved annotated diff image: {filename}", force=True)
                        if use_gemini:
                            result = gemini_look.gemini_describe_region(pil_img, largest_box)
                            if result:
                                json_filename = os.path.splitext(filename)[0] + ".json"
                                with open(json_filename, "w") as jf:
                                    json.dump(result, jf, indent=2)
                                log(f"[Monitor {idx}] Saved Gemini JSON result: {json_filename}", force=True)
                        prev_images[idx] = pil_img  # Update reference image.
                    else:
                        log(f"[Monitor {idx}] Difference detected but largest box {box_width}x{box_height} is < {min_threshold}.")
                else:
                    log(f"[Monitor {idx}] No significant change detected.")
        log(f"Sleeping for {interval:.2f} seconds before next cycle...\n")
        time.sleep(interval)

def main():
    parser = argparse.ArgumentParser(
        description="Periodically capture screenshots and save updated regions using a comparison approach."
    )
    parser.add_argument("interval", type=float, help="Seconds between screenshots")
    parser.add_argument("directory", type=str, help="Directory to save screenshots")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--min", type=int, default=400, help="Minimum size threshold for a bounding box (pixels)")
    parser.add_argument("-g", "--gemini", action="store_true", help="Call Gemini API when differences are detected")
    args = parser.parse_args()
    global GLOBAL_VERBOSE
    GLOBAL_VERBOSE = args.verbose
    
    # Initialize Gemini client if needed
    if args.gemini:
        try:
            gemini_look.initialize()
            log("Gemini API client initialized successfully", force=True)
        except Exception as e:
            log(f"Failed to initialize Gemini API: {str(e)}", force=True)
            sys.exit(1)
    
    process_screenshots(args.interval, args.directory, args.min, args.gemini)

if __name__ == '__main__':
    main()
