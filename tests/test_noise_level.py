#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
Regression test for C3: level and coupling of the TR2/B1 noise (pe78/noise.py).

Reference: LTspice B1 = -white(2e6*time)*4e-6 A into node N, VR8U = 90 kOhm
from N to +12 V, N -> C28 -> TR3 base and N -> C29 -> TR4 base.

1. noise_amp(fs) gives the B1 one-sided density 4e-6/sqrt(12)*sqrt(2/2e6) A/rtHz
   at any fs.
2. Base noise voltage per unit B1 current, from the model step, against the
   analytic Norton transfer (bilinear frequency warping included):
       v_b/i = VR8U*r_base/(VR8U+r_base) * s*tau/(1+s*tau),
       tau = C*(VR8U+r_base)
   for the snare (C28, r_base = r_pi||R51) and the cymbals (C29,
   r_base = r_pi||R52). Before C3 the source was an ideal voltage (gain 1,
   tau = C*r_base): errors of several dB.
3. One definition of the level: drums.py wrappers and the models called
   directly give the same output with noise_sample=None (before C3: 0.8 mV
   vs 40 mV for the snare, no noise at all for the cymbals called directly).

Usage (from the py78drums root):
    python -m pytest tests/test_noise_level.py
    python tests/test_noise_level.py

pywdf is taken from the environment; set PYWDF_ROOT to the folder that
contains the pywdf package to use a specific copy (for example pywdf-ref).
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
if os.environ.get("PYWDF_ROOT"):
    sys.path.insert(0, os.path.abspath(os.environ["PYWDF_ROOT"]))

from pe78 import noise                       # noqa: E402
from pe78 import drums                       # noqa: E402
from pe78.snare import PE78_Snare            # noqa: E402
from pe78.cymb import PE78_Cymbals_Maracas   # noqa: E402

FREQS = (200.0, 1000.0, 3560.0, 10000.0)
R_PI = (10e3, 20e3, 1e6)          # TR on (peak), decaying, almost off
REL_TOL = 1e-9
N_SETTLE = 9600                   # 0.2 s at 48 kHz: tau <= 4.7 nF * 1.1 MOhm = 5.2 ms
N_WIN = 4800                      # integer number of periods for all FREQS at 48 kHz


def _bin(x, f, fs):
    n = np.arange(len(x))
    return np.sum(x * np.exp(-2j * np.pi * f * n / fs)) * 2.0 / len(x)


def _norton(c, vr8u, r_base, f, fs):
    s = 1j * 2.0 * fs * np.tan(np.pi * f / fs)    # bilinear warping
    tau = c * (vr8u + r_base)
    return vr8u * r_base / (vr8u + r_base) * s * tau / (1.0 + s * tau)


def _response(step, f, fs):
    n = np.arange(N_SETTLE + N_WIN)
    x = np.cos(2.0 * np.pi * f * n / fs)
    y = np.array([step(xi) for xi in x])
    return _bin(y[N_SETTLE:], f, fs) / _bin(x[N_SETTLE:], f, fs)


def _check(name, h, h_ref, f):
    err = abs(h - h_ref) / abs(h_ref)
    assert err < REL_TOL, (f"{name} f={f:g} Hz: |H|={abs(h):.6g} Ohm, "
                           f"|H_ref|={abs(h_ref):.6g} Ohm, rel err {err:.3g}")


def test_noise_amp():
    d = 4e-6 / np.sqrt(12.0) * np.sqrt(2.0 / 2e6)
    assert abs(noise.noise_density() - d) < 1e-12 * d
    for fs in (44100, 48000, 96000):
        assert abs(noise.noise_amp(fs) - d * np.sqrt(fs / 2.0)) < 1e-12 * d
    for cls in (PE78_Snare, PE78_Cymbals_Maracas):
        for fs in (48000, 96000):
            assert cls(fs).NOISE_AMP == noise.noise_amp(fs)


def test_snare_base_noise_transfer():
    fs = 48000
    for r_pi in R_PI:
        for f in FREQS:
            m = PE78_Snare(fs)
            gm = m.hFE / r_pi
            r_base = r_pi * m.R51 / (r_pi + m.R51)
            h = _response(lambda x: m._step_c28(x, gm), f, fs)
            _check(f"snare r_pi={r_pi:g}", h, _norton(m.C28, m.VR8U, r_base, f, fs), f)


def test_cymb_base_noise_transfer():
    fs = 48000
    for r_pi in R_PI:
        for f in FREQS:
            m = PE78_Cymbals_Maracas(fs)
            gm = m.hFE / r_pi
            r_base = r_pi * m.R52 / (r_pi + m.R52)

            def step(x):
                m._step_tr4(0.62, gm, x)
                return m._v_base

            h = _response(step, f, fs)
            _check(f"cymb r_pi={r_pi:g}", h, _norton(m.C29, m.VR8U, r_base, f, fs), f)


def _gate(n, fs):
    v = np.zeros(n)
    v[10:10 + int(0.010 * fs)] = 4.5
    return v


def test_single_level_definition():
    fs, n = 48000, 2400
    trig = _gate(n, fs)

    np.random.seed(1)
    d = drums.SnareDrum(fs)
    a = np.array([d.tick(v) for v in trig])
    np.random.seed(1)
    m = PE78_Snare(fs)
    b = np.array([m.process_sample(v, None) for v in trig])
    assert np.max(np.abs(a)) > 0.0
    assert np.array_equal(a, b), np.max(np.abs(a - b))

    np.random.seed(2)
    d = drums.CymbDrum(fs)
    a = np.array([d.tick(v, 0.0, 0.0) for v in trig])
    np.random.seed(2)
    m = PE78_Cymbals_Maracas(fs)
    b = np.array([m.process_sample(v, 0.0, 0.0) for v in trig])
    assert np.max(np.abs(a)) > 0.0
    assert np.array_equal(a, b), np.max(np.abs(a - b))


if __name__ == "__main__":
    test_noise_amp()
    test_snare_base_noise_transfer()
    test_cymb_base_noise_transfer()
    test_single_level_definition()
    print("ok")
