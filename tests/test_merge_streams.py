import importlib
import sys
import types

import numpy as np

dummy_vad = types.ModuleType("silero_vad")
dummy_vad.get_speech_timestamps = lambda *a, **k: []
dummy_vad.load_silero_vad = lambda *a, **k: None
sys.modules.setdefault("silero_vad", dummy_vad)


def test_merge_streams_zero_sys():
    mod = importlib.import_module("hear.transcribe")
    sys_data = np.zeros(16000, dtype=np.float32)
    mic_data = np.ones(16000, dtype=np.float32) * 0.1
    merged, ranges = mod.merge_streams(sys_data, mic_data, mod.SAMPLE_RATE)
    assert np.allclose(merged, mic_data)
    assert ranges == [(0.0, 1.0)]
