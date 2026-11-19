#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
Regression test for C11 and check of the linear region of TR3 and TR4
(from C15 in pe78/noise_voices.py).

Before C11 the collector current was clamped to (Vcc - 0.2 V)/R53 (snare)
and (Vcc - 0.2 V)/R54 (cymbals): not a physical mechanism (L1 and L2 short
R53 and R54 in DC) and never active. The clamp was removed.

The collectors are linear: no saturation (V_CE), no high injection (IKF).
This test runs the snare, a cymbal pattern (LC, SC, MR) and a snare hit on a
ringing long cymbal, with noise, at the levels of tools/baseline.py, and
fails with an explicit message when TR3 or TR4 leave the linear region: then
a saturation / high-injection model is needed before those levels can be
trusted. Run it again after every change that raises the levels at the bases.

1. Ic = hFE*Ib with no hidden limit (also with hFE raised so that Ic exceeds
   the old clamp).
2. V_CE >= V_CE_MIN and Ic <= IC_MAX at every sample of the three runs.

Usage (from the py78drums root):
    python -m pytest tests/test_linear_region.py
    python tests/test_linear_region.py

pywdf is taken from the environment; set PYWDF_ROOT to the folder that
contains the pywdf package to use a specific copy (for example pywdf-ref).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _nv_common import NoiseVoices  # noqa: E402

V_CE_MIN = 1.0          # [V]  well above V_CE(sat) ~ 0.2 V
IKF = 0.1357            # [A]  BC547B card (LTspice): beta drops near IKF
SEED = 42
FS = 96000


def _ic_max(nv):
    return min(0.1 * IKF, nv.hFE * nv.Vt / nv.R_PI_MIN)


def _run(name, gates, t_end, vr9_pos=None):
    nv = NoiseVoices(FS, vr9_pos=vr9_pos)
    n = int(round(t_end * FS))
    noise = np.random.default_rng(SEED).standard_normal(n)
    t = np.arange(1, n + 1) / FS
    ic = np.zeros((n, 2))
    vce = np.zeros((n, 2))

    def v(k, tt):
        return nv.V_ON if any(a <= tt < a + g for a, g in gates.get(k, [])) else 0.0

    for k in range(n):
        nv.tick(v("sd", t[k]), v("lc", t[k]), v("sc", t[k]), v("mr", t[k]), noise[k])
        ic[k] = nv._Ic3, nv._Ic4
        vce[k] = nv.Vcc + nv._V_col3, nv.Vcc + nv._V_col4
    for j, tr in enumerate(("TR3", "TR4")):
        k = int(np.argmin(vce[:, j]))
        assert vce[k, j] >= V_CE_MIN, (
            f"{name}: {tr} outside the linear region at t = {t[k]:.4f} s: V_CE = "
            f"{vce[k, j]:.3f} V < {V_CE_MIN} V. The collector model has no saturation "
            f"(C11): add a saturation model before using these levels.")
        k = int(np.argmax(ic[:, j]))
        assert ic[k, j] <= _ic_max(nv), (
            f"{name}: {tr} Ic = {ic[k, j]*1e3:.2f} mA at t = {t[k]:.4f} s > "
            f"{_ic_max(nv)*1e3:.2f} mA. The model has no high-injection limit and r_pi is "
            f"clipped at R_PI_MIN (C11): add a high-current model before using these levels.")
    return vce.min(axis=0), ic.max(axis=0)


def test_no_hidden_clamp():
    nv = NoiseVoices(FS)
    nv.hFE = 2000.0                         # Ic above the old clamp (2.5 mA)
    big = 0.0
    for k in range(int(0.03 * FS)):
        nv.tick(nv.V_ON if 10 <= k < 10 + int(0.01 * FS) else 0.0, 0, 0, 0, 0.0)
        ib3, ib4 = nv._root_b.i_ports
        assert nv._Ic3 == nv.hFE * ib3 and nv._Ic4 == nv.hFE * ib4
        big = max(big, nv._Ic3)
    assert big > (nv.Vcc - 0.2) / nv.R53, big


RUNS = {
    "snare": ({"sd": [(10 / FS, 0.01)]}, 0.6, None),
    "snare VR9 0.5": ({"sd": [(10 / FS, 0.01)]}, 0.6, 0.5),
    "cymbal pattern": ({"lc": [(0.05, 0.1)], "sc": [(0.55, 0.1)], "mr": [(1.05, 0.1)]}, 2.0, None),
    "LC + snare": ({"lc": [(0.05, 0.02)], "sd": [(0.25, 0.01)]}, 0.6, None),
}


def test_linear_region():
    for name, (g, te, pos) in RUNS.items():
        _run(name, g, te, pos)


if __name__ == "__main__":
    test_no_hidden_clamp()
    for name, (g, te, pos) in RUNS.items():
        vmin, imax = _run(name, g, te, pos)
        print(f"{name:15s} V_CE min TR3 {vmin[0]:.3f} TR4 {vmin[1]:.3f} V; "
              f"Ic max TR3 {imax[0]*1e3:.3f} TR4 {imax[1]*1e3:.3f} mA")
    print("ok")
