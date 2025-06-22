import argparse
import faulthandler
import json
import logging
import time
from pathlib import Path

import gemini_look
from dotenv import load_dotenv
from PIL import Image

from think.crumbs import CrumbBuilder


class Describer:
    def __init__(self, watch_dir: Path, poll_interval: int = 5, entities: Path | None = None):
        self.watch_dir = watch_dir
        self.poll_interval = poll_interval
        self.entities = entities
        self.processed: set[str] = set()

    def scan(self) -> list[tuple[Path, Path, Path]]:
        files = []
        for box_path in self.watch_dir.rglob("*_diff_box.json"):
            prefix = box_path.stem[:-4] if box_path.stem.endswith("_box") else box_path.stem
            img_path = box_path.with_name(prefix + ".png")
            json_path = box_path.with_name(prefix + ".json")
            if not img_path.exists():
                logging.info(f"Skipping {box_path}: no corresponding image file {img_path}")
                continue
            if json_path.name in self.processed:
                logging.info(f"Skipping {box_path}: already processed")
                continue
            files.append((img_path, box_path, json_path))

        if files:
            logging.info(f"Found {len(files)} files to process")
        else:
            logging.info("No new files found to process")

        return files

    def describe(self, img_path: Path, box_path: Path) -> dict | None:
        logging.info(f"Processing {img_path} with box {box_path}")
        box = json.loads(box_path.read_text())
        with Image.open(img_path) as im:
            return gemini_look.gemini_describe_region(
                im, box, entities=str(self.entities) if self.entities else None
            )

    def start(self):
        while True:
            for img_path, box_path, json_path in self.scan():
                if json_path.exists():
                    self.processed.add(json_path.name)
                    continue
                try:
                    result = self.describe(img_path, box_path)
                    if result:
                        json_path.write_text(json.dumps(result, indent=2))
                        logging.info(f"Described {img_path} -> {json_path}")
                        crumb_builder = CrumbBuilder().add_file(img_path).add_file(box_path)
                        if self.entities:
                            crumb_builder.add_file(self.entities)
                        crumb_builder.add_model("gemini-2.5-flash")
                        crumb_path = crumb_builder.commit(str(json_path))
                        logging.info(f"Crumb saved to {crumb_path}")
                        self.processed.add(json_path.name)
                except Exception as e:
                    logging.error(f"Failed to describe {img_path}: {e}")
            time.sleep(self.poll_interval)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Describe screenshot diffs using Gemini")
    parser.add_argument("watch_dir", type=Path, help="Directory containing screenshot diffs")
    parser.add_argument("-e", "--entities", type=Path, default=None, help="Optional entities file")
    parser.add_argument("-i", "--interval", type=int, default=5, help="Polling interval in seconds")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    faulthandler.enable()

    gemini_look.initialize()

    describer = Describer(args.watch_dir, args.interval, args.entities)
    describer.start()


if __name__ == "__main__":
    main()
