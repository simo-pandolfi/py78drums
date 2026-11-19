#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
Check of the ENV <-> maracas coupling of cymb.py (C9).

The ENV root and the maracas root see each other through Thevenin sources of
the previous sample (one-sample delay both ways, R46 in both roots, static
R_th). C9 kept this structure (decision D1-A) after measuring its error
against tools/cymb_env_ref.py, which solves the whole section at once.
This test bounds that error, at 96 kHz (rate of tools/baseline.py) and at
48 kHz, so that later changes (C15, C16) cannot make it grow unnoticed.

Measured at C9 (edges +-1 ms excluded): maracas alone V_ENV 1.10 / 1.84 mV,
V_C26 0.90 / 2.65 mV (96 / 48 kHz); LC, SC and MR overlapping V_C26
0.78 mV at 96 kHz.

Usage (from the py78drums root):
    python -m pytest tests/test_cymb_coupling.py
    python tests/test_cymb_coupling.py

pywdf is taken from the environment; set PYWDF_ROOT to the folder that
contains the pywdf package to use a specific copy (for example pywdf-ref).
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
if os.environ.get("PYWDF_ROOT"):
    sys.path.insert(0, os.path.abspath(os.environ["PYWDF_ROOT"]))

from pe78.cymb import PE78_Cymbals_Maracas  # noqa: E402
import cymb_env_ref as ref  # noqa: E402

CASES = {
    "mr_alone": ([], [], [(0.05, 0.015)], 0.4),
    "overlap": ([(0.05, 0.1)], [(0.08, 0.1)], [(0.11, 0.1)], 0.6),
}
TOL = {96000: 2e-3, 48000: 4e-3}        # [V] on V_ENV, V_C26, V_C23


def _errors(fs, glc, gsc, gmr, t_end, r_ref):
    tr, r = r_ref
    m = PE78_Cymbals_Maracas(fs)
    n = int(round(t_end * fs))
    t = np.arange(n) / fs
    o = {k: np.zeros(n) for k in ("env", "v_mr", "v_lc")}

    def on(gs, tt):
        return any(a <= tt < a + g for a, g in gs)

    for k in range(n):
        m.process_sample(m.V_ON * on(glc, t[k]), m.V_ON * on(gsc, t[k]),
                         m.V_ON * on(gmr, t[k]), 0.0)
        o["env"][k] = m._V_env
        o["v_mr"][k] = m._V_c26
        o["v_lc"][k] = m._V_c23
    keep = np.ones(n, dtype=bool)
    for a, g in glc + gsc + gmr:
        for e in (a, a + g):
            keep &= np.abs(t - e) > 1e-3
    return {k: float(np.abs(o[k] - np.interp(t, tr, r[k]))[keep].max()) for k in o}


def _case(name):
    glc, gsc, gmr, t_end = CASES[name]
    r_ref = ref.simulate(glc, gsc, t_end, gates_mr=gmr)
    out = {}
    for fs, tol in TOL.items():
        err = _errors(fs, glc, gsc, gmr, t_end, r_ref)
        for k, d in err.items():
            assert d < tol, (name, fs, k, d)
        out[fs] = err
    return out


def test_mr_alone():
    _case("mr_alone")


def test_overlap():
    _case("overlap")


if __name__ == "__main__":
    for name in CASES:
        for fs, err in _case(name).items():
            print(f"{name:9s} {fs:6d} Hz: " + ", ".join(f"{k} {v*1e3:.2f} mV" for k, v in err.items()))
    print("ok")
