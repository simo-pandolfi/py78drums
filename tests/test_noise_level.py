#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
Regression test for C3 and C15: level and coupling of the TR2/B1 noise.

Reference: LTspice B1 = -white(2e6*time)*4e-6 A into node N, VR8U = 90 kOhm
from N to +12 V, N -> C28 -> TR3 base and N -> C29 -> TR4 base.

1. noise_amp(fs) gives the B1 one-sided density 4e-6/sqrt(12)*sqrt(2/2e6)
   A/rtHz at any fs; NoiseVoices uses it.
2. From C15 the node N is shared (one Norton source, VR8U) and the bases are
   Shockley ports of the bases root. At a DC operating point (TR3 off or on,
   TR4 off or on with the LC source held active) and with a small sinusoidal
   B1 current, V_b3 and V_env follow the analytic nodal solution of the
   linearised network (bilinear warping included): B3 (R50 || R51, g3, C28),
   N (VR8U, C28, C29), ENV (C29, R52, g4, R46 to the MR port, LC and SC
   branches). Before C3 the source was an ideal voltage; before C15 each
   voice had its own N and the other base did not load it.
3. tick(noise=None) draws N(0, 1) internally: same output as passing the
   same samples explicitly.

Usage (from the py78drums root):
    python -m pytest tests/test_noise_level.py
    python tests/test_noise_level.py

pywdf is taken from the environment; set PYWDF_ROOT to the folder that
contains the pywdf package to use a specific copy (for example pywdf-ref).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _nv_common import NoiseVoices  # noqa: E402
from pe78 import noise  # noqa: E402

FS = 96000
FREQS = (200.0, 1000.0, 3000.0, 12000.0)
N_SETTLE, N_WIN = 19200, 4800
I_AMP = 1e-10            # [A] small: linear around the operating point
REL_TOL = 2e-3


def test_noise_amp():
    d = 4e-6 / np.sqrt(12.0) * np.sqrt(2.0 / 2e6)
    assert abs(noise.noise_density() - d) < 1e-12 * d
    for fs in (44100, 48000, 96000):
        assert abs(noise.noise_amp(fs) - d * np.sqrt(fs / 2.0)) < 1e-12 * d
        assert NoiseVoices(fs).NOISE_AMP == noise.noise_amp(fs)


def _setup(v_c27, lc_on):
    nv = NoiseVoices(FS)
    nv._r50src.set_voltage(v_c27)
    nv._mr_port.set_voltage(0.0)
    if lc_on:
        nv._m253_lc.set_resistance(nv.R_ON)
        nv._m253_lc.set_voltage(nv.V_ON)
    nv._noise_src.set_voltage(0.0)
    for _ in range(N_SETTLE):
        nv._root_b.compute()
    return nv


def _analytic(nv, f, vb3, venv):
    s = 1j * 2.0 * FS * np.tan(np.pi * f / FS)
    g3 = nv.Is_b3 * np.exp(vb3 / nv.Vt) / nv.Vt
    g4 = nv.Is_b4 * np.exp(venv / nv.Vt) / nv.Vt
    rlc = nv._m253_lc.Rp
    rsc = nv._m253_sc.Rp
    y_lc = 1.0 / (nv.R47 + 1.0 / (s * nv.C23 + 1.0 / rlc))
    y_sc = 1.0 / (nv.R44 + 1.0 / (s * nv.C24 + 1.0 / (nv.R42 + rsc)))
    Y = np.array([
        [1 / nv.sd.R50 + 1 / nv.R51 + g3 + s * nv.C28, -s * nv.C28, 0],
        [-s * nv.C28, 1 / nv.VR8U + s * (nv.C28 + nv.C29), -s * nv.C29],
        [0, -s * nv.C29, s * nv.C29 + 1 / nv.R52 + g4 + 1 / nv.R46 + y_lc + y_sc]])
    return np.linalg.solve(Y, np.array([0, 1.0, 0]))     # per ampere into N


def _measured(nv, twin, f):
    """Response to a small B1 sine; twin is the same network without it
    (removes the slow residual drift of the operating point exactly)."""
    n = N_WIN * 2
    k = np.arange(n)
    i = I_AMP * np.sin(2 * np.pi * f * k / FS)
    v = np.zeros((n, 3))
    for j in range(n):
        nv._noise_src.set_voltage(i[j] * nv.VR8U)
        nv._root_b.compute()
        twin._root_b.compute()
        v[j] = nv._root_b.v_nodes - twin._root_b.v_nodes
    w = v[N_WIN:]
    e = np.exp(-2j * np.pi * f * k[N_WIN:] / FS)
    return (w * e[:, None]).sum(axis=0) * 2.0 / N_WIN / (-1j) / I_AMP, twin._root_b.v_nodes.copy()


def test_noise_transfer_shared_node():
    for v_c27, lc_on in ((0.0, False), (2.7, False), (0.0, True), (2.7, True)):
        nv = _setup(v_c27, lc_on)
        twin = _setup(v_c27, lc_on)
        for f in FREQS:
            got, dc = _measured(nv, twin, f)
            exp = _analytic(nv, f, dc[0], dc[2])
            err = np.abs(got[[0, 2]] - exp[[0, 2]]) / np.abs(exp[[0, 2]])
            assert np.all(err < REL_TOL), (v_c27, lc_on, f, got, exp, err)


def test_single_level_definition():
    fs, n = 48000, 2400
    a = NoiseVoices(fs)           # built first: pywdf Capacitor draws from np.random
    b = NoiseVoices(fs)
    np.random.seed(1)
    ya = np.array([a.tick(4.5 if 10 <= k < 490 else 0.0, 0, 0, 0) for k in range(n)])
    np.random.seed(1)
    x = [np.random.standard_normal() for _ in range(n)]   # same draws, one at a time
    yb = np.array([b.tick(4.5 if 10 <= k < 490 else 0.0, 0, 0, 0, x[k]) for k in range(n)])
    assert np.max(np.abs(ya)) > 0.0
    assert np.array_equal(ya, yb), np.max(np.abs(ya - yb))


if __name__ == "__main__":
    test_noise_amp()
    test_noise_transfer_shared_node()
    test_single_level_definition()
    print("ok")
