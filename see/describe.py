import argparse
import faulthandler
import json
import logging
import time
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

from . import gemini_look


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
                continue
            if json_path.name in self.processed:
                continue
            files.append((img_path, box_path, json_path))
        return files

    def describe(self, img_path: Path, box_path: Path) -> dict | None:
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    faulthandler.enable()

    gemini_look.initialize()

    describer = Describer(args.watch_dir, args.interval, args.entities)
    describer.start()


if __name__ == "__main__":
    main()
