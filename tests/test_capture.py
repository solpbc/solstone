import importlib

import numpy as np


def test_get_buffer_and_flac(tmp_path):
    cap = importlib.import_module("hear.capture")
    rec = cap.AudioRecorder(journal=str(tmp_path))
    rec.audio_queue.put(
        (np.array([0.1, -0.1], dtype=np.float32), np.array([0.2, 0.2], dtype=np.float32))
    )
    sys_buf, mic_buf = rec.get_buffers()
    assert len(mic_buf) == 2
    assert len(sys_buf) == 2
    data = rec.create_flac_bytes(mic_buf, sys_buf)
    assert data.startswith(b"fLaC")
    rec.save_flac(data, "t")
    files = list(tmp_path.rglob("*_t.flac"))
    assert files
