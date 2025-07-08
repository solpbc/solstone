import argparse
import datetime
import faulthandler
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from PIL import Image
from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer

from see import gemini_look
from think.crumbs import CrumbBuilder


class Describer:
    def __init__(self, journal_dir: Path):
        """Watch the journal and describe new screenshot diffs for the current day."""
        self.journal_dir = journal_dir
        self.watch_dir: Optional[Path] = None
        self.entities = journal_dir / "entities.md"
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
        crumb_builder = CrumbBuilder().add_file(img_path).add_file(box_path).add_file(self.entities)
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
            return gemini_look.gemini_describe_region(im, box, entities=str(self.entities))

    @staticmethod
    def scan_day(day_dir: Path) -> dict[str, list[str]]:
        """Return lists of raw, processed and repairable files within ``day_dir``."""
        raw = sorted(p.name for p in day_dir.glob("*_diff_box.json"))
        processed = sorted(p.name for p in day_dir.glob("*_diff.json"))

        repairable: list[str] = []
        for box_name in raw:
            box_path = day_dir / box_name
            prefix = box_path.stem.replace("_box", "")
            img_path = box_path.with_name(prefix + ".png")
            json_path = box_path.with_name(prefix + ".json")
            if not img_path.exists():
                continue
            if not json_path.exists():
                repairable.append(box_name)

        return {"raw": raw, "processed": processed, "repairable": sorted(repairable)}

    def repair_day(self, date_str: str, files: list[str], dry_run: bool = False) -> int:
        """Process ``files`` belonging to ``date_str`` and return the count."""
        day_dir = self.journal_dir / date_str
        if not day_dir.exists():
            logging.error(f"Day directory {day_dir} does not exist")
            return 0

        logging.info(f"Repairing day {date_str} in {day_dir}")

        if dry_run:
            return len(files)

        for box_name in files:
            box_path = day_dir / box_name
            prefix = box_path.stem.replace("_box", "")
            img_path = box_path.with_name(prefix + ".png")
            json_path = box_path.with_name(prefix + ".json")

            if not img_path.exists():
                logging.warning(f"Skipping {box_path}: missing image {img_path}")
                continue

            try:
                logging.info(f"Describing image: {img_path}")
                self._process_once(img_path, box_path, json_path)
            except Exception as e:
                logging.error(f"Failed to describe {img_path}: {e}")

        return len(files)

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

        self.observer = None
        current_day: Optional[str] = None
        try:
            while True:
                today_str = datetime.datetime.now().strftime("%Y%m%d")
                day_dir = self.journal_dir / today_str
                if day_dir.exists() and (current_day != today_str):
                    if self.observer:
                        self.observer.stop()
                        self.observer.join()
                    self.observer = Observer()
                    self.observer.schedule(handler, str(day_dir), recursive=True)
                    self.observer.start()
                    self.watch_dir = day_dir
                    self.processed.clear()
                    current_day = today_str
                    logging.info(f"Watching {day_dir}")
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            if self.observer:
                self.observer.stop()
                self.observer.join()
            self.executor.shutdown(wait=True)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Describe screenshot diffs using Gemini")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--repair",
        type=str,
        help="Repair mode: process incomplete files for specified day (YYYYMMDD format)",
    )
    args = parser.parse_args()

    journal = Path(os.getenv("JOURNAL_PATH", ""))
    if not journal.is_dir():
        parser.error("JOURNAL_PATH not set or invalid")

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    faulthandler.enable()

    gemini_look.initialize()

    ent_path = journal / "entities.md"
    if not ent_path.is_file():
        parser.error(f"entities file not found: {ent_path}")

    describer = Describer(journal)

    if args.repair:
        # Validate date format
        try:
            datetime.datetime.strptime(args.repair, "%Y%m%d")
        except ValueError:
            parser.error(f"Invalid date format: {args.repair}. Use YYYYMMDD format.")
        info = Describer.scan_day(journal / args.repair)
        describer.repair_day(args.repair, info["repairable"])
    else:
        describer.start()


if __name__ == "__main__":
    main()
