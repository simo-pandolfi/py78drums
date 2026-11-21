#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
Regression test for C2, C15 and D5: collectors and output network of the
noise voices (pe78/noise_voices.py).

C2: R53/R54 appear once (they were also a port: |Z| halved). C15: the two
collectors share the balance pot VR9 (one 220 kOhm linear pot, wiper W),
R55, C31 and RLoad (one output). The WDF of the collectors is driven with
sinusoidal collector currents Ic3, Ic4 and compared with the analytic nodal
solution (bilinear frequency warping included). D5 (P4): L1 and L2 carry the
series resistance of the .asc inductor model, 61.662 Ohm (its Rpar of 9.42
MOhm is left out); before D5 they were ideal (R54 || L2 at 200 Hz: 10 % lower).

    c3: R53 || (L1 + Rser) || C30, VR9U to W      c4: R54 || (L2 + Rser), VR9L to W
    W:  R55 to X;  X: C31 to O;  O: RLoad to GND   (+12 V = AC ground)

for V_col3, V_col4 and V_out, two VR9 positions, both sources.

Usage (from the py78drums root):
    python -m pytest tests/test_collector_impedance.py
    python tests/test_collector_impedance.py

pywdf is taken from the environment; set PYWDF_ROOT to the folder that
contains the pywdf package to use a specific copy (for example pywdf-ref).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _nv_common import NoiseVoices  # noqa: E402

FS = 96000
FREQS = (200.0, 1000.0, 3000.0, 12000.0)    # integer number of periods in N_WIN
N_SETTLE, N_WIN = 19200, 4800
REL_TOL = 1e-6
L_RSER = 61.662                             # [Ohm] .asc, Coilcraft PCH-45X-107


def _analytic(nv, f, src):
    s = 1j * 2.0 * FS * np.tan(np.pi * f / FS)
    u, l = nv._vr9u.Rp, nv._vr9l.Rp
    Y = np.zeros((5, 5), dtype=complex)        # c3, c4, W, X, O

    def y(p, q, val):
        Y[p, p] += val
        if q is not None:
            Y[q, q] += val
            Y[p, q] -= val
            Y[q, p] -= val
    y(0, None, 1 / nv.R53 + 1 / (s * nv.L1 + L_RSER) + s * nv.C30)
    y(1, None, 1 / nv.R54 + 1 / (s * nv.L2 + L_RSER))
    y(0, 2, 1 / u)
    y(1, 2, 1 / l)
    y(2, 3, 1 / nv.R55)
    y(3, 4, s * nv.C31)
    y(4, None, 1 / nv.Rload)
    I = np.zeros(5, dtype=complex)
    I[src] = -1.0                               # Ic leaves the collector node
    return np.linalg.solve(Y, I)


def _measured(nv, f, src):
    n = N_SETTLE + N_WIN
    k = np.arange(n)
    ic = np.sin(2 * np.pi * f * k / FS)
    v = np.zeros((n, 3))
    for i in range(n):
        nv._step_collectors(ic[i] if src == 0 else 0.0, ic[i] if src == 1 else 0.0)
        v[i] = (nv._V_col3, nv._V_col4, nv._V_out)
    w = v[N_SETTLE:]
    kk = k[N_SETTLE:]
    e = np.exp(-2j * np.pi * f * kk / FS)
    # sin = Im(e^{jwt}): phasor of the response relative to the source phasor -j
    return (w * e[:, None]).sum(axis=0) * 2.0 / N_WIN / (-1j)


def _check(pos):
    for src in (0, 1):
        for f in FREQS:
            nv = NoiseVoices(FS, vr9_pos=pos)
            got = _measured(nv, f, src)
            exp = _analytic(nv, f, src)[[0, 1, 4]]
            err = np.abs(got - exp) / np.max(np.abs(exp))
            assert np.all(err < REL_TOL), (pos, src, f, got, exp, err)


def test_inductor_rser():
    nv = NoiseVoices(FS)
    assert nv.L_RSER == L_RSER
    assert nv._rs1.Rp == L_RSER and nv._rs2.Rp == L_RSER


def test_default_vr9():
    _check(None)


def test_vr9_half():
    _check(0.5)


def test_vr9_one_pot():
    nv = NoiseVoices(FS)
    for p in (0.1, 0.5, 0.9):
        nv.set_vr9(p)
        assert abs(nv._vr9u.Rp + nv._vr9l.Rp - nv.VR9) < 1e-6
        assert abs(nv._vr9u.Rp - p * nv.VR9) < 1e-6


if __name__ == "__main__":
    test_inductor_rser()
    test_default_vr9()
    test_vr9_half()
    test_vr9_one_pot()
    print("ok")
