import importlib
import os
import sys
import time
import types
from unittest.mock import Mock

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def set_test_journal_path(request, monkeypatch):
    """Set JOURNAL_PATH to fixtures/journal for all unit tests.

    This ensures all tests have a valid JOURNAL_PATH without needing
    to explicitly set it in each test. Integration tests are excluded.
    """
    # Skip for integration tests - they may have different requirements
    if "integration" in request.node.keywords:
        return

    # Set JOURNAL_PATH to fixtures/journal for all unit tests
    monkeypatch.setenv("JOURNAL_PATH", "fixtures/journal")


@pytest.fixture(autouse=True)
def add_module_stubs(request, monkeypatch):
    # Skip stubbing for integration tests
    if "integration" in request.node.keywords:
        return

    # stub heavy modules used by think.indexer
    if "usearch.index" not in sys.modules:
        usearch = types.ModuleType("usearch")
        index_mod = types.ModuleType("usearch.index")

        class DummyIndex:
            def __init__(self, *a, **k):
                pass

            def save(self, *a, **k):
                pass

            @classmethod
            def restore(cls, *a, **k):
                return cls()

            def remove(self, *a, **k):
                pass

            def add(self, *a, **k):
                pass

            def search(self, *a, **k):
                class Res:
                    keys = [1]
                    distances = [0.0]

                return Res()

        index_mod.Index = DummyIndex
        usearch.index = index_mod
        sys.modules["usearch"] = usearch
        sys.modules["usearch.index"] = index_mod
    if "sentence_transformers" not in sys.modules:
        st_mod = types.ModuleType("sentence_transformers")

        class DummyST:
            def __init__(self, *a, **k):
                pass

            def get_sentence_embedding_dimension(self):
                return 384

            def encode(self, texts):
                if isinstance(texts, str):
                    texts = [texts]
                return [([0.0] * 384) for _ in texts]

        st_mod.SentenceTransformer = DummyST
        sys.modules["sentence_transformers"] = st_mod
    if "sklearn.metrics.pairwise" not in sys.modules:
        pairwise = types.ModuleType("pairwise")

        def cosine_similarity(a, b):
            return [[1.0]]

        pairwise.cosine_similarity = cosine_similarity
        metrics = types.ModuleType("metrics")
        metrics.pairwise = pairwise
        sys.modules["sklearn"] = types.ModuleType("sklearn")
        sys.modules["sklearn.metrics"] = metrics
        sys.modules["sklearn.metrics.pairwise"] = pairwise
    if "dotenv" not in sys.modules:
        dotenv_mod = types.ModuleType("dotenv")

        def load_dotenv(*a, **k):
            return True

        def dotenv_values(*a, **k):
            return {}

        dotenv_mod.load_dotenv = load_dotenv
        dotenv_mod.dotenv_values = dotenv_values
        sys.modules["dotenv"] = dotenv_mod
    # Import real observe package first to avoid shadowing with stubs
    if "observe" not in sys.modules:
        importlib.import_module("observe")
    if "observe.detect" not in sys.modules:
        detect_mod = types.ModuleType("observe.detect")

        def input_detect():
            return None, None

        detect_mod.input_detect = input_detect
        sys.modules["observe.detect"] = detect_mod
        observe_pkg = sys.modules.get("observe")
        setattr(observe_pkg, "detect", detect_mod)
    if "observe.hear" not in sys.modules:
        # Import the real module for format_audio and load_transcript
        hear_mod = importlib.import_module("observe.hear")
        sys.modules["observe.hear"] = hear_mod
        observe_pkg = sys.modules.get("observe")
        setattr(observe_pkg, "hear", hear_mod)
    if "observe.sense" not in sys.modules:
        sense_mod = types.ModuleType("observe.sense")

        def scan_day(day_dir):
            # Stub matching real scan_day behavior:
            # - "raw": processed files in segments
            # - "processed": output JSON files in segments
            # - "repairable": source media files in day root without matching segment
            from pathlib import Path

            day_path = Path(day_dir)
            raw_files = []
            processed_files = []
            repairable_files = []

            if day_path.is_dir():
                # Find raw (processed) files in segments (HHMMSS/)
                for item in day_path.iterdir():
                    from think.utils import segment_key

                    if item.is_dir() and segment_key(item.name):
                        # Found segment
                        for p in item.glob("*.flac"):
                            raw_files.append(f"{item.name}/{p.name}")
                        for p in item.glob("*.m4a"):
                            raw_files.append(f"{item.name}/{p.name}")
                        for p in item.glob("*.webm"):
                            raw_files.append(f"{item.name}/{p.name}")
                        for p in item.glob("*.mp4"):
                            raw_files.append(f"{item.name}/{p.name}")

                # Find processed output files in segments
                for item in day_path.iterdir():
                    from think.utils import segment_key

                    if item.is_dir() and segment_key(item.name):
                        for p in item.glob("*audio.jsonl"):
                            processed_files.append(f"{item.name}/{p.name}")
                        for p in item.glob("*screen.jsonl"):
                            processed_files.append(f"{item.name}/{p.name}")

                # Find repairable files (source media in root without matching segment)
                for audio_ext in ["*.flac", "*.m4a"]:
                    for p in day_path.glob(audio_ext):
                        if "_" in p.stem:
                            segment_name = p.stem.split("_")[0]
                            segment_dir = day_path / segment_name
                            if not segment_dir.exists():
                                repairable_files.append(p.name)

                for video_ext in ["*.webm", "*.mp4"]:
                    for p in day_path.glob(video_ext):
                        if "_" in p.stem:
                            time_part = p.stem.split("_")[0]
                            ts_dir = day_path / time_part
                            if not ts_dir.exists():
                                repairable_files.append(p.name)

            return {
                "raw": raw_files,
                "processed": processed_files,
                "repairable": repairable_files,
            }

        sense_mod.scan_day = scan_day
        sys.modules["observe.sense"] = sense_mod
        observe_pkg = sys.modules.get("observe")
        setattr(observe_pkg, "sense", sense_mod)
    if "observe.utils" not in sys.modules:
        # Import the real module
        utils_mod = importlib.import_module("observe.utils")
        sys.modules["observe.utils"] = utils_mod
        observe_pkg = sys.modules.get("observe")
        setattr(observe_pkg, "utils", utils_mod)
    if "observe.screen" not in sys.modules:
        # Import the real module for format_screen
        screen_mod = importlib.import_module("observe.screen")
        sys.modules["observe.screen"] = screen_mod
        observe_pkg = sys.modules.get("observe")
        setattr(observe_pkg, "screen", screen_mod)
    if "gi" not in sys.modules:
        gi_mod = types.ModuleType("gi")
        gi_mod.require_version = lambda *a, **k: None

        class Dummy(types.ModuleType):
            pass

        repo = types.ModuleType("gi.repository")
        repo.Gdk = Dummy("Gdk")
        repo.Gtk = Dummy("Gtk")
        gi_mod.repository = repo
        sys.modules["gi"] = gi_mod
        sys.modules["gi.repository"] = repo
        sys.modules["Gdk"] = repo.Gdk
        sys.modules["Gtk"] = repo.Gtk
    google_mod = sys.modules.get("google", types.ModuleType("google"))
    genai_mod = types.ModuleType("google.genai")

    class DummyClient:
        def __init__(self, *a, **k):
            pass

    genai_mod.Client = DummyClient

    # Mock Content type for type hints
    class MockContent:
        pass

    genai_mod.types = types.SimpleNamespace(
        GenerateContentConfig=lambda **k: None, Content=MockContent
    )
    google_mod.genai = genai_mod
    sys.modules["google"] = google_mod
    sys.modules["google.genai"] = genai_mod
    if "skimage.metrics" not in sys.modules:
        metrics_mod = types.ModuleType("skimage.metrics")

        def structural_similarity(a, b, full=False):
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)
            diff = np.mean(np.abs(a - b)) / 255.0
            score = 1.0 - diff
            if full:
                return score, None
            return score

        metrics_mod.structural_similarity = structural_similarity
        skimage_mod = types.ModuleType("skimage")
        skimage_mod.metrics = metrics_mod
        sys.modules["skimage"] = skimage_mod
        sys.modules["skimage.metrics"] = metrics_mod
    if "cv2" not in sys.modules:
        cv2_mod = types.ModuleType("cv2")
        cv2_mod.COLOR_RGB2LAB = 0

        def cvtColor(arr, code):
            arr = np.asarray(arr)
            gray = arr.mean(axis=2)
            return np.stack([gray, gray, gray], axis=2)

        cv2_mod.cvtColor = cvtColor
        sys.modules["cv2"] = cv2_mod
    if "soundfile" not in sys.modules:
        sf_mod = types.ModuleType("soundfile")

        def write(buf, data, samplerate, format=None):
            buf.write(b"fLaCfake")

        sf_mod.write = write
        sys.modules["soundfile"] = sf_mod
    for name in [
        "noisereduce",
        "silero_vad",
        "watchdog.events",
        "watchdog.observers",
    ]:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            if name == "silero_vad":
                mod.load_silero_vad = lambda *a, **k: lambda data, sr: []
                mod.get_speech_timestamps = lambda *a, **k: []
            sys.modules[name] = mod
    if "watchdog.events" in sys.modules and not hasattr(
        sys.modules["watchdog.events"], "PatternMatchingEventHandler"
    ):

        class FileSystemEventHandler:
            pass

        class PatternMatchingEventHandler:
            pass

        sys.modules["watchdog.events"].FileSystemEventHandler = FileSystemEventHandler
        sys.modules["watchdog.events"].PatternMatchingEventHandler = (
            PatternMatchingEventHandler
        )
    if "watchdog.observers" in sys.modules and not hasattr(
        sys.modules["watchdog.observers"], "Observer"
    ):

        class Observer:
            def schedule(self, *a, **k):
                pass

            def start(self):
                pass

            def stop(self):
                pass

            def join(self, *a, **k):
                pass

            def is_alive(self):
                return False

        sys.modules["watchdog.observers"].Observer = Observer


@pytest.fixture
def mock_callosum(monkeypatch):
    """Mock Callosum connections to capture emitted events without real I/O.

    This fixture provides a MockCallosumConnection class that:
    - Enforces the start-before-emit requirement
    - Broadcasts events to all listeners (like the real Callosum)
    - Works without real socket connections

    Usage:
        def test_example(mock_callosum):
            from think.callosum import CallosumConnection

            received = []
            listener = CallosumConnection()
            listener.start(callback=lambda msg: received.append(msg))

            # Now emit events and they'll be captured in received
    """
    all_listeners = []

    class MockCallosumConnection:
        def __init__(self, socket_path=None):
            self.socket_path = socket_path
            self.callback = None
            self.thread = None

        def start(self, callback=None):
            """Simulate starting the background thread."""
            self.callback = callback
            self.thread = Mock()
            self.thread.is_alive.return_value = True
            if callback:
                all_listeners.append(self)

        def emit(self, tract, event, **kwargs):
            """Emit event and broadcast to all listeners."""
            # Return False if not started yet (matches real behavior)
            if self.thread is None or not self.thread.is_alive():
                return False

            # Build message
            msg = {"tract": tract, "event": event, **kwargs}
            if "ts" not in msg:
                msg["ts"] = int(time.time() * 1000)

            # Broadcast to all listeners
            for listener in all_listeners:
                if listener.callback:
                    listener.callback(msg)

            return True

        def stop(self):
            """Stop connection and remove from listeners."""
            if self in all_listeners:
                all_listeners.remove(self)
            self.thread = None
            self.callback = None

    # Patch both import locations
    monkeypatch.setattr("think.runner.CallosumConnection", MockCallosumConnection)
    monkeypatch.setattr("think.callosum.CallosumConnection", MockCallosumConnection)
    monkeypatch.setattr("think.supervisor.CallosumConnection", MockCallosumConnection)
