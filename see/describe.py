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

from PIL import Image
from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer

from see import gemini_look
from see.reduce import reduce_day
from think.crumbs import CrumbBuilder
from think.utils import day_log, setup_cli


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
        self._last_reduce: Optional[datetime.datetime] = None

    def _move_to_seen(self, img_path: Path) -> Path:
        seen_dir = img_path.parent / "seen"
        try:
            seen_dir.mkdir(exist_ok=True)
            new_img = seen_dir / img_path.name
            img_path.rename(new_img)
            logging.info("Moved %s to %s", img_path, seen_dir)
            return new_img
        except Exception as exc:  # pragma: no cover - filesystem errors
            logging.error(
                "Failed to move %s to seen: %s", img_path, exc
            )
            return img_path

    def _process_once(self, img_path: Path, json_path: Path) -> None:
        result = self.describe(img_path)
        if not result:
            raise RuntimeError("Gemini API returned no result")
        json_path.write_text(json.dumps(result["result"], indent=2))
        logging.info(f"Described {img_path} -> {json_path}")
        new_img = self._move_to_seen(img_path)
        crumb_builder = (
            CrumbBuilder().add_file(new_img).add_file(self.entities)
        )
        crumb_builder.add_model(result["model_used"])
        crumb_path = crumb_builder.commit(str(json_path))
        logging.info(f"Crumb saved to {crumb_path}")

    def _get_json_path(self, img_path: Path) -> Path:
        """Get the corresponding JSON path for a PNG file, ensuring *_diff.json naming."""
        if img_path.name.endswith("_diff.png"):
            return img_path.with_suffix(".json")
        else:
            # Convert *.png to *_diff.json
            return img_path.parent / f"{img_path.stem}_diff.json"

    def _process(self, img_path: Path, json_path: Path) -> None:
        if json_path.exists() or json_path.name in self.processed:
            if img_path.exists():
                self._move_to_seen(img_path)
            self.processed.add(json_path.name)
            return
        attempts = 0
        while attempts < 2:
            try:
                self._process_once(img_path, json_path)
                break
            except Exception as e:
                attempts += 1
                if attempts < 2:
                    logging.warning(f"Retrying {img_path} due to error: {e}")
                else:
                    logging.error(f"Failed to describe {img_path}: {e}")
        self.processed.add(json_path.name)

    def describe(self, img_path: Path) -> Optional[dict]:
        logging.info(f"Processing {img_path}")
        with Image.open(img_path) as im:
            # Read box_2d from PNG metadata,
            box_2d_str = im.text.get("box_2d")
            if box_2d_str:
                box = json.loads(box_2d_str)
            else:
                # Default to full image dimensions
                box = [0, 0, im.width, im.height]
                logging.warning(f"No box_2d metadata found in {img_path}, using full image dimensions: {box}")
            return gemini_look.gemini_describe_region(
                im, box, entities=str(self.entities)
            )

    @staticmethod
    def scan_day(day_dir: Path) -> dict[str, list[str]]:
        """Return lists of raw, processed and repairable files within ``day_dir``.

        Raw files are ``*.png`` files in the ``seen/`` directory.
        Processed files are ``*_diff.json`` files in ``day_dir``.
        Repairable files are ``*.png`` files in ``day_dir``.
        """

        seen_dir = day_dir / "seen"
        raw = (
            [f"seen/{p.name}" for p in sorted(seen_dir.glob("*.png"))]
            if seen_dir.is_dir()
            else []
        )

        processed = sorted(p.name for p in day_dir.glob("*_diff.json"))

        repairable = sorted(p.name for p in day_dir.glob("*.png"))

        return {"raw": raw, "processed": processed, "repairable": repairable}

    def repair_day(self, date_str: str, files: list[str], dry_run: bool = False) -> int:
        """Process ``files`` belonging to ``date_str`` and return the count."""
        day_dir = self.journal_dir / date_str
        if not day_dir.exists():
            logging.error(f"Day directory {day_dir} does not exist")
            return 0

        logging.info(f"Repairing day {date_str} in {day_dir}")

        if dry_run:
            return len(files)

        success = 0

        for img_name in files:
            img_path = day_dir / img_name
            json_path = self._get_json_path(img_path)

            if not img_path.exists():
                logging.warning(f"Skipping {img_path}: file does not exist")
                continue

            try:
                logging.info(f"Describing image: {img_path}")
                if json_path.exists():
                    logging.info(f"Already processed {img_path}")
                    self._move_to_seen(img_path)
                    success += 1
                else:
                    self._process_once(img_path, json_path)
                    if json_path.exists():
                        success += 1
            except Exception as e:
                logging.error(f"Failed to describe {img_path}: {e}")

        return success

    def _maybe_reduce(self) -> None:
        """Reduce the previous 5 minute window if enough time has passed."""
        now = datetime.datetime.now()
        prev_minute = now - datetime.timedelta(minutes=1)
        if prev_minute.minute % 5 != 0:
            return
        block_end = prev_minute.replace(
            minute=(prev_minute.minute // 5) * 5,
            second=0,
            microsecond=0,
        )
        if self._last_reduce == block_end:
            return
        block_start = block_end - datetime.timedelta(minutes=5)
        day_str = prev_minute.strftime("%Y%m%d")
        try:
            reduce_day(day_str, start=block_start, end=block_end)
            self._last_reduce = block_end
        except Exception as exc:
            logging.error(f"reduce_day failed: {exc}")

    def _handle_image_event(self, event_path: str, event_type: str = "detected") -> None:
        """Common handler for image file events (created/moved)."""
        img_path = Path(event_path)
        json_path = self._get_json_path(img_path)

        if not img_path.exists():
            logging.warning(f"Skipping {img_path}: file does not exist")
            return

        logging.info(f"Image {event_type}, processing: {img_path}")
        self.executor.submit(self._process, img_path, json_path)

    def start(self):
        handler = PatternMatchingEventHandler(
            patterns=["*.png"],
            ignore_directories=True,
            ignore_patterns=["*/seen/*"],
        )

        def on_created(event):
            self._handle_image_event(event.src_path, "created")

        def on_moved(event):
            self._handle_image_event(event.dest_path, "moved")

        handler.on_created = on_created
        handler.on_moved = on_moved

        self.observer = None
        current_day: Optional[str] = None
        try:
            while True:
                today_str = datetime.datetime.now().strftime("%Y%m%d")
                day_dir = self.journal_dir / today_str
                if day_dir.is_dir() and (current_day != today_str):
                    if self.observer:
                        self.observer.stop()
                        self.observer.join()
                    self.observer = Observer()
                    self.observer.schedule(handler, str(day_dir), recursive=False)
                    self.observer.start()
                    self.watch_dir = day_dir
                    self.processed.clear()
                    current_day = today_str
                    logging.info(f"Watching {day_dir}")
                self._maybe_reduce()
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            if self.observer:
                self.observer.stop()
                self.observer.join()
            self.executor.shutdown(wait=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Describe screenshot diffs using Gemini"
    )
    parser.add_argument(
        "--repair",
        type=str,
        help="Repair mode: process incomplete files for specified day (YYYYMMDD format)",
    )
    parser.add_argument(
        "--scan",
        type=str,
        help="Scan mode: show file counts for specified day (YYYYMMDD format)",
    )
    args = setup_cli(parser)

    journal = Path(os.getenv("JOURNAL_PATH", ""))
    faulthandler.enable()

    gemini_look.initialize()

    ent_path = journal / "entities.md"
    if not ent_path.is_file():
        parser.error(f"entities file not found: {ent_path}")

    describer = Describer(journal)

    if args.scan:
        # Validate date format
        try:
            datetime.datetime.strptime(args.scan, "%Y%m%d")
        except ValueError:
            parser.error(f"Invalid date format: {args.scan}. Use YYYYMMDD format.")

        day_dir = journal / args.scan
        if not day_dir.exists():
            print(f"Day directory {day_dir} does not exist")
            return

        info = Describer.scan_day(day_dir)
        print(f"Day {args.scan} scan results:")
        print(f"  Raw files: {len(info['raw'])}")
        print(f"  Processed files: {len(info['processed'])}")
        print(f"  Repairable files: {len(info['repairable'])}")

    elif args.repair:
        # Validate date format
        try:
            datetime.datetime.strptime(args.repair, "%Y%m%d")
        except ValueError:
            parser.error(f"Invalid date format: {args.repair}. Use YYYYMMDD format.")
        info = Describer.scan_day(journal / args.repair)
        repaired = describer.repair_day(args.repair, info["repairable"])
        failed = len(info["repairable"]) - repaired
        msg = f"see-describe repaired {repaired}"
        if failed:
            msg += f" failed {failed}"
        day_log(args.repair, msg)
    else:
        describer.start()


if __name__ == "__main__":
    main()
