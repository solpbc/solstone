import importlib
import sys
import types

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def add_module_stubs(monkeypatch):
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

        dotenv_mod.load_dotenv = load_dotenv
        sys.modules["dotenv"] = dotenv_mod
    if "input_detect" not in sys.modules:
        input_detect_mod = types.ModuleType("input_detect")

        def input_detect():
            return None, None

        input_detect_mod.input_detect = input_detect
        sys.modules["input_detect"] = input_detect_mod
        sys.modules["hear.input_detect"] = input_detect_mod
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
    if "see.screen_dbus" not in sys.modules:
        screen_dbus = types.ModuleType("see.screen_dbus")
        screen_dbus.screen_snap = lambda: []
        screen_dbus.idle_time_ms = lambda: 0
        sys.modules["see.screen_dbus"] = screen_dbus
        sys.modules["screen_dbus"] = screen_dbus
    google_mod = sys.modules.get("google", types.ModuleType("google"))
    genai_mod = types.ModuleType("google.genai")

    class DummyClient:
        def __init__(self, *a, **k):
            pass

    genai_mod.Client = DummyClient
    genai_mod.types = types.SimpleNamespace(GenerateContentConfig=lambda **k: None)
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
    ws_mod = types.ModuleType("websockets")

    class DummyWS:
        async def send(self, data):
            return None

        async def wait_closed(self):
            return None

    class ConnectionClosed(Exception):
        pass

    class ClientConnection:
        def __init__(self, *a, **k):
            pass

    client_mod = types.ModuleType("websockets.client")
    client_mod.ClientConnection = ClientConnection

    async def connect(*a, **k):
        return ClientConnection()

    client_mod.connect = connect

    async def serve(handler, host, port):
        class Server:
            def __init__(self):
                self.ws = DummyWS()

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

        return Server()

    ws_mod.WebSocketServerProtocol = DummyWS
    ws_mod.serve = serve
    ws_mod.ConnectionClosed = ConnectionClosed
    ws_mod.client = client_mod
    sys.modules["websockets"] = ws_mod
    sys.modules["websockets.client"] = client_mod
    for name in ["librosa", "noisereduce", "silero_vad", "watchdog.events", "watchdog.observers"]:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            if name == "silero_vad":
                mod.load_silero_vad = lambda *a, **k: lambda data, sr: []
                mod.get_speech_timestamps = lambda *a, **k: []
            sys.modules[name] = mod
    if "watchdog.events" in sys.modules and not hasattr(
        sys.modules["watchdog.events"], "PatternMatchingEventHandler"
    ):

        class PatternMatchingEventHandler:
            pass

        sys.modules["watchdog.events"].PatternMatchingEventHandler = PatternMatchingEventHandler
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

        sys.modules["watchdog.observers"].Observer = Observer
    if "screen_compare" not in sys.modules:
        mod = importlib.import_module("see.screen_compare")
        sys.modules["screen_compare"] = mod
