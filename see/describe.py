import argparse
import faulthandler
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import gemini_look
from dotenv import load_dotenv
from PIL import Image
from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer

from think.crumbs import CrumbBuilder


class Describer:
    def __init__(self, watch_dir: Path, entities: Optional[Path] = None):
        self.watch_dir = watch_dir
        self.entities = entities
        self.processed: set[str] = set()
        self.observer: Optional[Observer] = None
        self.executor = ThreadPoolExecutor()
        self.attempts: dict[str, int] = {}

    def _process_once(self, img_path: Path, box_path: Path, json_path: Path) -> None:
        result = self.describe(img_path, box_path)
        if not result:
            raise RuntimeError("Gemini API returned no result")
        json_path.write_text(json.dumps(result["result"], indent=2))
        logging.info(f"Described {img_path} -> {json_path}")
        crumb_builder = CrumbBuilder().add_file(img_path).add_file(box_path)
        if self.entities:
            crumb_builder.add_file(self.entities)
        crumb_builder.add_model(result["model_used"])
        crumb_path = crumb_builder.commit(str(json_path))
        logging.info(f"Crumb saved to {crumb_path}")

    def _process(self, img_path: Path, box_path: Path, json_path: Path) -> None:
        if json_path.exists() or json_path.name in self.processed:
            self.processed.add(json_path.name)
            return
        attempts = 0
        while attempts < 2:
            try:
                self._process_once(img_path, box_path, json_path)
                break
            except Exception as e:
                attempts += 1
                if attempts < 2:
                    logging.warning(f"Retrying {img_path} due to error: {e}")
                else:
                    logging.error(f"Failed to describe {img_path}: {e}")
        self.processed.add(json_path.name)

    def describe(self, img_path: Path, box_path: Path) -> Optional[dict]:
        logging.info(f"Processing {img_path} with box {box_path}")
        box = json.loads(box_path.read_text())
        with Image.open(img_path) as im:
            return gemini_look.gemini_describe_region(
                im, box, entities=str(self.entities) if self.entities else None
            )

    def start(self):
        handler = PatternMatchingEventHandler(patterns=["*_diff_box.json"], ignore_directories=True)

        def on_created(event):
            box_path = Path(event.src_path)
            prefix = box_path.stem.replace("_box", "")
            img_path = box_path.with_name(prefix + ".png")
            json_path = box_path.with_name(prefix + ".json")
            
            if not img_path.exists():
                logging.warning(f"Skipping {box_path}: missing image {img_path}")
                return
            
            self.executor.submit(self._process, img_path, box_path, json_path)

        handler.on_created = on_created

        self.observer = Observer()
        self.observer.schedule(handler, str(self.watch_dir), recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(1)  # Short sleep for responsiveness
        except KeyboardInterrupt:
            pass
        finally:
            self.observer.stop()
            self.observer.join()
            self.executor.shutdown(wait=True)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Describe screenshot diffs using Gemini")
    parser.add_argument("watch_dir", type=Path, help="Directory containing screenshot diffs")
    parser.add_argument("-e", "--entities", type=Path, default=None, help="Optional entities file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    faulthandler.enable()

    gemini_look.initialize()

    describer = Describer(args.watch_dir, args.entities)
    describer.start()


if __name__ == "__main__":
    main()
