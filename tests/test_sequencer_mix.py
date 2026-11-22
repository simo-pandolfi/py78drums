#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
Test of the sequencer stems and of sequencer/mix.py (P4, V1).

wdf_rithm.synthesize() keeps the raw output of each voice and builds the
master bus with mix() (stem * GAIN * bus load gain, summed in the order of
VOICES, normalized to a peak of 0.5); mix.py re-mixes saved stems.

1. Short render ("Test cymb": every voice, 600 BPM, 1 cycle, seeded noise):
   the master returned by synthesize() equals mix() of its stems with GAIN,
   sample for sample, and the stems saved and loaded give the same master.
2. Linearity: doubling GAIN['bd'] doubles the bd contribution and leaves
   the others unchanged (before normalization).
3. mix.py writes the same 16-bit WAV as the sequencer with the GAIN of
   wdf_rithm.py.

Usage (from the py78drums root):
    python -m pytest tests/test_sequencer_mix.py
    python tests/test_sequencer_mix.py
"""

import os
import sys
import tempfile

import numpy as np
from scipy.io import wavfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sequencer"))
sys.path.insert(0, ROOT)
import wdf_rithm as W  # noqa: E402
import mix as M  # noqa: E402

NAME, BPM, BARS = "Test cymb", 600, 1


def _render():
    np.random.seed(1234)
    return W.synthesize(NAME, bpm=BPM, num_bars=BARS, return_stems=True)


def test_stems_mix_wav():
    master, stems, g_t, g_n = _render()
    assert set(stems) == set(W.VOICES)
    m2, _ = W.mix(stems, W.GAIN, g_t, g_n)
    assert np.array_equal(master, m2)

    # linearity, before normalization
    def bus(gain):
        return sum(stems[k] * gain[k] * (g_n if k == "noise" else g_t) for k in W.VOICES)
    g2 = dict(W.GAIN, bd=2 * W.GAIN["bd"])
    d = bus(g2) - bus(W.GAIN)
    ref = stems["bd"] * W.GAIN["bd"] * g_t
    assert np.max(np.abs(d - ref)) <= 1e-12 * np.max(np.abs(ref))

    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "s.npz")
        W.save_stems(p, stems, rhythm=NAME, bpm=BPM, bars=BARS, fs=W.FS, g_tonal=g_t, g_noise=g_n)
        st, meta = W.load_stems(p)
        m3, _ = W.mix(st, W.GAIN, meta["g_tonal"], meta["g_noise"])
        assert np.array_equal(master, m3)
        wav_seq, wav_mix = os.path.join(tmp, "seq.wav"), os.path.join(tmp, "mix.wav")
        W.save_wav(master, wav_seq)
        M.main([p, "-o", wav_mix])
        a, b = wavfile.read(wav_seq), wavfile.read(wav_mix)
        assert a[0] == b[0] == W.FS and np.array_equal(a[1], b[1])
    print(f"{len(master)} samples, voices {', '.join(sorted(stems))}: master = mix(stems), WAV identical")


if __name__ == "__main__":
    test_stems_mix_wav()
    print("OK")
