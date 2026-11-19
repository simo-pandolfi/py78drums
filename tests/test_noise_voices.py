#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
Regression test for C15: snare, cymbals and maracas as one circuit
(pe78/noise_voices.py) against tools/noise_voices_ref.py.

Before C15 the snare and the cymbals were separate models: each with its own
noise node (no C28/C29 at large signal, no mutual load), collectors driven
by the small-signal gm*v_b only (no envelope current through L1/L2), and each
with its own VR9 leg, R55, C31 and RLoad (VR9U and VR9L independent). Against
the whole-section reference the sum of the two outputs was +1.5 dB (snare
alone) and +8.5 dB (snare on a ringing long cymbal) too loud.

1. Without noise (deterministic): output, collectors, TR3 base and ENV of a
   snare hit, and of a snare hit on a ringing long cymbal, against the
   reference (envelope current through L1-C30 at 3.56 kHz, crosstalk of the
   snare hit into ENV through C28-N-C29, TR3 base delayed by C28).
2. With noise (same samples, first-order hold in the reference): output of
   the snare on a ringing long cymbal, below 4 kHz (bilinear warping makes
   the waveforms differ near fs/2): correlation and level.

Usage (from the py78drums root):
    python -m pytest tests/test_noise_voices.py
    python tests/test_noise_voices.py

pywdf is taken from the environment; set PYWDF_ROOT to the folder that
contains the pywdf package to use a specific copy (for example pywdf-ref).
"""

import os
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _nv_common import ref, run_model, edge_mask, max_err  # noqa: E402
from pe78.noise import noise_amp  # noqa: E402

FS = 96000


def test_snare_without_noise():
    g = {"sd": [(0.01, 0.01)]}
    t, o, _ = run_model(g, 0.15, FS)
    tr, r = ref.simulate(g, 0.15)
    err = max_err(t, o, tr, r, ("o", "c3", "b3", "env"), edge_mask(t, g))
    for k, tol in (("o", 1.5e-4), ("c3", 8e-4), ("b3", 2e-3), ("env", 8e-4)):
        assert err[k] < tol, (k, err[k])
    for k in ("env",):                                   # crosstalk into ENV
        assert abs(o[k].max() - r[k].max()) < 2e-3, (o[k].max(), r[k].max())
        assert abs(o[k].min() - r[k].min()) < 2e-3, (o[k].min(), r[k].min())
    on_m = t[np.argmax(o["b3"] > 0.5)]
    on_r = tr[np.argmax(r["b3"] > 0.5)]
    assert abs(on_m - on_r) < 5e-5, (on_m, on_r)        # TR3 base delayed by C28


def test_snare_on_cymbal_without_noise():
    g = {"lc": [(0.01, 0.02)], "sd": [(0.2, 0.01)]}
    t, o, _ = run_model(g, 0.32, FS)
    tr, r = ref.simulate(g, 0.32)
    err = max_err(t, o, tr, r, ("o", "c3", "c4", "env"), edge_mask(t, g))
    for k, tol in (("o", 2e-4), ("c3", 1.2e-3), ("c4", 8e-4), ("env", 1e-3)):
        assert err[k] < tol, (k, err[k])


def _lp(x, fc=4000.0):
    return sosfiltfilt(butter(6, fc, fs=FS, output="sos"), x)


def test_snare_on_cymbal_with_noise():
    g = {"lc": [(0.01, 0.02)], "sd": [(0.06, 0.01)]}
    t_end = 0.12
    noise = np.random.default_rng(1).standard_normal(int(t_end * FS) + 2)
    t, o, _ = run_model(g, t_end, FS, noise=noise)
    tr, r = ref.simulate(g, t_end, dt=1.0 / (4 * FS), noise_i=noise_amp(FS) * noise, fs_noise=FS)
    y = np.interp(t, tr, r["o"])
    w = (t > 0.062) & (t < 0.115)
    xm, xr = _lp(o["o"])[w], _lp(y)[w]
    corr = np.corrcoef(xm, xr)[0, 1]
    lvl = 20 * np.log10(np.sqrt(np.mean(xm ** 2)) / np.sqrt(np.mean(xr ** 2)))
    assert corr > 0.95, corr
    assert abs(lvl) < 0.5, lvl
    return corr, lvl


if __name__ == "__main__":
    test_snare_without_noise()
    test_snare_on_cymbal_without_noise()
    c, l = test_snare_on_cymbal_with_noise()
    print(f"with noise, below 4 kHz: correlation {c:.4f}, level {l:+.2f} dB")
    print("ok")
