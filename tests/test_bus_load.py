#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
Check of C16: mutual load of the tonal and noise sections on the output bus.

In the circuit the noise section (collectors, VR9, wiper W, R55, C31) and
the tonal section (five R41 joined, C21) drive the same node O, the pre-amp
input VR10 = RLOAD. TonalDrum and NoiseVoices terminate on RLOAD alone;
drums.bus_load_gains() (decision D1-B) corrects each output with a fixed
mid-band gain. This test compares those gains with the nodal AC solution of
the whole bus (C21, C31, (L1 + Rser) || R53 || C30, (L2 + Rser) || R54,
VR9), for both sources and four VR9 positions (the .asc, the default of
NoiseVoices, 0.5, 0.9):
  - 200 Hz to 5 kHz: within 0.06 dB;
  - 50 Hz: within 0.6 dB (C21 / C31 corners).

Usage (from the py78drums root):
    python -m pytest tests/test_bus_load.py
    python tests/test_bus_load.py

pywdf is taken from the environment; set PYWDF_ROOT to the folder that
contains the pywdf package to use a specific copy (for example pywdf-ref).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _nv_common import NoiseVoices  # noqa: E402
from pe78.drums import bus_load_gains, TonalDrum, BDO_PARAMS  # noqa: E402


def _v_o(f, pos, src, tonal_branch, noise_branch):
    nv = NoiseVoices
    s = 2j * np.pi * f
    u, l = pos * nv.VR9, (1 - pos) * nv.VR9
    r_t = BDO_PARAMS["kick_load"] / TonalDrum.N_TONAL
    Y = np.zeros((6, 6), dtype=complex)        # c3, c4, W, X, O, T

    def y(p, q, v):
        Y[p, p] += v
        if q is not None:
            Y[q, q] += v
            Y[p, q] -= v
            Y[q, p] -= v
    y(0, None, 1 / nv.R53 + 1 / (s * nv.L1 + nv.L_RSER) + s * nv.C30)
    y(1, None, 1 / nv.R54 + 1 / (s * nv.L2 + nv.L_RSER))
    y(0, 2, 1 / u)
    y(1, 2, 1 / l)
    y(4, None, 1 / TonalDrum.RLOAD)
    y(5, None, 1 / r_t)
    if noise_branch:
        y(2, 3, 1 / nv.R55)
        y(3, 4, s * nv.C31)
    else:
        y(3, None, 1.0)
    if tonal_branch:
        y(4, 5, s * TonalDrum.C21)
    I = np.zeros(6, dtype=complex)
    if src == "tonal":
        I[5] = 1 / r_t                          # Norton of the joined tonal node
    else:
        I[0 if src == "ic3" else 1] = -1.0
    return np.linalg.solve(Y, I)[4]


def _db(x):
    return 20 * np.log10(x)


POSITIONS = (25 / 220, NoiseVoices.VR9_POS_DEFAULT, 0.5, 0.9)


def test_bus_load_gains():
    for pos in POSITIONS:
        g_noise, g_tonal = bus_load_gains(pos)
        for f, tol in ((50, 0.6), (200, 0.06), (1000, 0.06), (5000, 0.06)):
            for src in ("ic3", "ic4"):
                exact = abs(_v_o(f, pos, src, True, True)) / abs(_v_o(f, pos, src, False, True))
                assert abs(_db(exact) - _db(g_noise)) < tol, (pos, f, src, _db(exact), _db(g_noise))
            exact = abs(_v_o(f, pos, "tonal", True, True)) / abs(_v_o(f, pos, "tonal", True, False))
            assert abs(_db(exact) - _db(g_tonal)) < tol, (pos, f, _db(exact), _db(g_tonal))


if __name__ == "__main__":
    test_bus_load_gains()
    for pos in POSITIONS:
        g = bus_load_gains(pos)
        print(f"VR9 pos {pos:.3f}: g_noise {_db(g[0]):+.2f} dB, g_tonal {_db(g[1]):+.2f} dB")
    print("ok")
