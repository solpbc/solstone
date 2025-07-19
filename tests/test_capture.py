import importlib

import numpy as np


def test_get_buffer_and_flac(tmp_path):
    cap = importlib.import_module("hear.capture")
    rec = cap.AudioRecorder(journal=str(tmp_path))
    stereo_chunk = np.column_stack(
        (
            np.array([0.1, -0.1], dtype=np.float32),
            np.array([0.2, 0.2], dtype=np.float32),
        )
    )
    rec.audio_queue.put(stereo_chunk)
    stereo_buf = rec.get_buffers()
    assert stereo_buf.shape == (2, 2)
    data = rec.create_flac_bytes(stereo_buf)
    assert data.startswith(b"fLaC")
    rec.save_flac(data, "t")
    files = list(tmp_path.rglob("*_t.flac"))
    assert files
