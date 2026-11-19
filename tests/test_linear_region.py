#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
Regression test for C11 and check of the linear region of TR3 and TR4.

Before C11 the collector current was clamped to (Vcc - 0.2 V)/R53 (snare)
and (Vcc - 0.2 V)/R54 (cymbals): not a physical mechanism (L1 and L2 short
R53 and R54 in DC, the collector sits at +12 V whatever Ic) and never active
(max Ic 1.1 mA against 2.5 mA). The clamp is removed.

The models of the collectors are linear: no saturation (V_CE), no high
injection (IKF), r_pi limited below by R_PI_MIN. This test runs the snare
and a cymbal pattern (LC, SC, MR, noise) at the levels of tools/baseline.py
and fails with an explicit message when TR3 or TR4 leave the linear region:
then the models need a saturation / high-injection model before those
levels can be trusted. Run it again after every change that raises the
levels at the bases (C8, C9, C15, C16).

1. Ic follows the Shockley law above the old clamp (no hidden limit).
2. Snare (gate 10 ms, VR9U 25k and 110k) and cymbal pattern (gate 100 ms,
   VR9L 195k and 100k): V_CE >= V_CE_MIN and Ic <= IC_MAX at every sample.

Usage (from the py78drums root):
    python -m pytest tests/test_linear_region.py
    python tests/test_linear_region.py

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

from pe78.snare import PE78_Snare  # noqa: E402
from pe78.cymb import PE78_Cymbals_Maracas  # noqa: E402

V_CE_MIN = 1.0          # [V]  well above V_CE(sat) ~ 0.2 V
IKF = 0.1357            # [A]  BC547B card (LTspice): beta drops near IKF
SEED = 42


def _ic_max(model):
    """Largest Ic of the linear model: 0.1*IKF and the r_pi floor."""
    return min(0.1 * IKF, model.hFE * model.Vt / model.R_PI_MIN)


def _check(name, t, ic, v_col, vcc, ic_max):
    vce = vcc + v_col                       # emitter at GND, V_col around +12 V (AC GND)
    k = int(np.argmin(vce))
    assert vce[k] >= V_CE_MIN, (
        f"{name}: TR collector outside the linear region at t = {t[k]:.4f} s: "
        f"V_CE = {vce[k]:.3f} V < {V_CE_MIN} V. The model has no saturation "
        f"(C11): add a saturation model before using these levels.")
    k = int(np.argmax(ic))
    assert ic[k] <= ic_max, (
        f"{name}: Ic = {ic[k]*1e3:.2f} mA at t = {t[k]:.4f} s > {ic_max*1e3:.2f} mA. "
        f"The model has no high-injection limit and r_pi is clipped at R_PI_MIN "
        f"(C11): add a high-current model before using these levels.")
    return float(vce.min()), float(ic.max())


def test_no_hidden_clamp():
    s = PE78_Snare(48000)
    s._V_base = 0.7                     # warm start (cold start: C1b)
    G = 1.0 / s.R50 + 1.0 / s.R51
    V, gm, _ = s._solve_base_ebers_moll(0.7 + s.Is_b * (np.exp(0.7 / s.Vt) - 1.0) / G)
    assert abs(V - 0.7) < 1e-9, V
    ic = s.hFE * s.Is_b * np.exp(V / s.Vt)
    assert ic > (s.Vcc - 0.2) / s.R53, ic                  # above the old clamp
    assert abs(gm * s.Vt - ic) < 1e-12 * ic, (gm * s.Vt, ic)
    c = PE78_Cymbals_Maracas(96000)
    c._V_env = 0.7
    G = 1e-6
    v_j = c._solve_env_ebers_moll(G, -0.7 * G - c.Is_b * (np.exp(0.7 / c.Vt) - 1.0))
    V, gm, _ = c._env_solution
    assert abs(V - 0.7) < 1e-9 and abs(v_j + V) < 1e-15, (V, v_j)
    ic = c.hFE * c.Is_b * np.exp(V / c.Vt)
    assert ic > (c.Vcc - 0.2) / c.R54, ic
    assert abs(gm * c.Vt - ic) < 1e-12 * ic, (gm * c.Vt, ic)


def _snare(vr9u):
    fs, n = 48000, int(0.6 * 48000)
    m = PE78_Snare(fs)
    if vr9u is not None:
        m.set_vr9u(vr9u)
    noise = np.random.default_rng(SEED).standard_normal(n)
    t = np.arange(n) / fs
    ic, v_col = np.zeros(n), np.zeros(n)
    for k in range(n):
        m.process_sample(m.V_ON if 10 <= k < 10 + int(0.01 * fs) else 0.0, noise[k])
        ic[k] = m.hFE * m.Is_b * np.exp(m._V_base / m.Vt)
        v_col[k] = m._V_col
    return _check(f"snare VR9U {vr9u}", t, ic, v_col, m.Vcc, _ic_max(m))


def _cymb(vr9l):
    fs, n = 96000, int(2.0 * 96000)
    m = PE78_Cymbals_Maracas(fs, vr9l=vr9l)
    noise = np.random.default_rng(SEED).standard_normal(n)
    t = np.arange(n) / fs
    g = int(0.1 * fs)
    starts = {"lc": int(0.05 * fs), "sc": int(0.55 * fs), "mr": int(1.05 * fs)}
    ic, v_col = np.zeros(n), np.zeros(n)
    for k in range(n):
        v = {x: (m.V_ON if a <= k < a + g else 0.0) for x, a in starts.items()}
        m.process_sample(v["lc"], v["sc"], v["mr"], noise[k])
        ic[k] = m.hFE * m.Is_b * np.exp(m._V_env / m.Vt)
        v_col[k] = m._V_col
    return _check(f"cymbals VR9L {vr9l}", t, ic, v_col, m.Vcc, _ic_max(m))


def test_snare_linear_region():
    for vr9u in (None, 110e3):
        _snare(vr9u)


def test_cymbals_linear_region():
    for vr9l in (None, 100e3):
        _cymb(vr9l)


if __name__ == "__main__":
    test_no_hidden_clamp()
    for vr9u in (None, 110e3):
        print("snare  VR9U %-8s V_CE min %.3f V, Ic max %.3f mA" % ((vr9u,) + tuple(
            x * (1e3 if i else 1) for i, x in enumerate(_snare(vr9u)))))
    for vr9l in (None, 100e3):
        print("cymbal VR9L %-8s V_CE min %.3f V, Ic max %.3f mA" % ((vr9l,) + tuple(
            x * (1e3 if i else 1) for i, x in enumerate(_cymb(vr9l)))))
    print("ok")
