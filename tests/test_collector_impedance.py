#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
Regression test for C2: collector impedance of TR3 (snare) and TR4 (cymbals).

The transistor is a current source gm*v_b into the collector node. The node
must see exactly the physical load, each element once:
    snare : L1 || R53 || C30 || VR9U
    cymb  : L2 || R54 || VR9L
R53/R54 is the internal resistance of the Thevenin source, so it must not
appear again as a separate port.

The test drives the model's own step function with a sinusoid, takes the
ratio V_col/I_c at the drive frequency over a steady-state window, and
compares it with the analytic impedance under the bilinear transform
(s = j*2*fs*tan(w/(2*fs))), which the WDF reproduces to rounding error.

Usage (from the py78drums root):
    python -m pytest tests/test_collector_impedance.py
    python tests/test_collector_impedance.py

pywdf is taken from the environment; set PYWDF_ROOT to the folder that
contains the pywdf package to use a specific copy (for example pywdf-ref).
The test requires pywdf with the E1 correction (pywdf-ref). With the original
pywdf |Z| is right but V_col lags by one sample, so the complex comparison
fails (rel err = 2*sin(w/2), 1.3e-2 at 100 Hz).
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
if os.environ.get("PYWDF_ROOT"):
    sys.path.insert(0, os.path.abspath(os.environ["PYWDF_ROOT"]))

from pe78.cymb import PE78_Cymbals_Maracas  # noqa: E402
from pe78.snare import PE78_Snare  # noqa: E402

FS = 48000
N_WIN = 24000            # analysis window: bins of FS/N_WIN = 2 Hz
N_SETTLE = 24000         # samples discarded before the window
FREQS = (100.0, 1000.0, 3560.0, 15000.0)   # multiples of 2 Hz
GM = 1e-3                # [A/V] fixed transconductance
REL_TOL = 1e-9


def _s(f, fs):
    """Bilinear-transform image of s = j*2*pi*f."""
    return 1j * 2.0 * fs * np.tan(np.pi * f / fs)


def _bin(x, f, fs):
    """Complex amplitude of x at frequency f (integer number of periods)."""
    n = np.arange(len(x))
    return np.dot(x, np.exp(-2j * np.pi * f * n / fs))


def _ratio(step, f):
    """V_col / I_c at frequency f, with step(x) -> (v_col, i_c)."""
    n = np.arange(N_SETTLE + N_WIN)
    x = np.cos(2.0 * np.pi * f * n / FS)
    v = np.empty_like(x)
    i = np.empty_like(x)
    for k in range(len(x)):
        v[k], i[k] = step(x[k])
    return _bin(v[N_SETTLE:], f, FS) / _bin(i[N_SETTLE:], f, FS)


def _z_snare(m, f):
    s = _s(f, m.fs)
    y = 1.0 / (s * m.L1_val) + s * m.C30 + 1.0 / m.R53 + 1.0 / m.VR9U
    return 1.0 / y


def _z_cymb(m, f):
    s = _s(f, m.fs)
    y = 1.0 / (s * m.L2_val) + 1.0 / m.R54 + 1.0 / m.VR9L
    return 1.0 / y


def _check(name, z_wdf, z_ref, f):
    err = abs(z_wdf - z_ref) / abs(z_ref)
    assert err < REL_TOL, (
        f"{name} f={f:g} Hz: |Z_wdf|={abs(z_wdf):.6g} Ohm, "
        f"|Z_ref|={abs(z_ref):.6g} Ohm, rel err {err:.3g}")


def test_snare_collector_impedance():
    for vr9u in (25e3, 110e3):
        for f in FREQS:
            m = PE78_Snare(FS, vr9u=vr9u)

            def step(x):
                return m._step_tr3_collector(GM, x), GM * x

            _check(f"snare VR9U={vr9u:g}", _ratio(step, f), _z_snare(m, f), f)


def test_cymb_collector_impedance():
    for vr9l in (100e3, 195e3):
        for f in FREQS:
            m = PE78_Cymbals_Maracas(FS, vr9l=vr9l)

            def step(x):
                v = m._step_tr4(0.62, GM, x)
                return v, GM * m._v_base     # current after the C29 high-pass

            _check(f"cymb VR9L={vr9l:g}", _ratio(step, f), _z_cymb(m, f), f)


if __name__ == "__main__":
    test_snare_collector_impedance()
    test_cymb_collector_impedance()
    print("ok")
