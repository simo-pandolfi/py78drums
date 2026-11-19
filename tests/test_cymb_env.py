#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
Regression test for C7: TR4 base load on the cymbal envelope node ENV.

Before C7 the TR4 base was a linear port r_pi (previous sample) inside the
ENV scatter, and the Newton-Raphson on the Shockley law ran apart: the
subtrees saw a node at about 68 mV while the NR gave 620 mV, so C23, C24 and
C26 discharged into the base with about 2.6 times the real base current
(LC: V_ENV > 0.5 V until 0.648 s instead of 0.804 s). The D9-branch
Thevenin of ENV used V_env (already loaded by the R46 current) in series
with R_env_th.

1. ENV is a nonlinear root: the junction voltage seen by the subtrees is
   the NR solution, and the current into the base equals the Shockley law.
2. Against tools/cymb_env_ref.py (nodal, trapezoidal, all simultaneous):
   V_ENV, V_C23, V_C24 for an LC hit and an SC hit, and the times below
   0.55 V and 0.5 V.

Usage (from the py78drums root):
    python -m pytest tests/test_cymb_env.py
    python tests/test_cymb_env.py

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

FS = 96000


def _run(gates_lc, gates_sc, t_end, check=False):
    m = PE78_Cymbals_Maracas(FS)
    n = int(round(t_end * FS))
    t = np.arange(n) / FS
    out = {k: np.zeros(n) for k in ("env", "v_lc", "v_sc")}
    for k in range(n):
        on_lc = any(a <= t[k] < a + g for a, g in gates_lc)
        on_sc = any(a <= t[k] < a + g for a, g in gates_sc)
        m.process_sample(m.V_ON if on_lc else 0.0, m.V_ON if on_sc else 0.0, 0.0, 0.0)
        out["env"][k] = m._V_env
        out["v_lc"][k] = m._V_c23
        out["v_sc"][k] = m._V_c24
        if check:
            root = m._root_env
            v_j = -m._V_env                                   # junction orientation
            for p in root.down_ports:                         # every port at -V_env
                assert abs(p.wave_to_voltage() - v_j) < 1e-12
            G = np.array([1.0 / p.Rp for p in root.down_ports])
            i_base = float(np.sum(G * (root.a_vals - v_j)))   # current into the root
            ib = m.Is_b * (np.exp(m._V_env / m.Vt) - 1.0)
            assert abs(i_base + ib) <= 1e-9 * max(abs(ib), 1e-12) + 1e-15, (k, i_base, ib)
    return t, out


def _edge_mask(t, gates, win=1e-3):
    keep = np.ones(len(t), dtype=bool)
    for a, g in gates:
        for e in (a, a + g):
            keep &= np.abs(t - e) > win
    return keep


def _compare(gates_lc, gates_sc, t_end, keys, tol):
    t, o = _run(gates_lc, gates_sc, t_end)
    tr, r = ref.simulate(gates_lc, gates_sc, t_end, dt=1e-5)
    keep = _edge_mask(t, gates_lc + gates_sc)
    for key in keys:
        d = np.abs(o[key] - np.interp(t, tr, r[key]))[keep].max()
        assert d < tol, (key, d)
    for thr in (0.55, 0.5):
        tm, trf = ref.last_above(t, o["env"], thr), ref.last_above(tr, r["env"], thr)
        assert abs(tm - trf) < 2e-3, (thr, tm, trf)


def test_root_coherence():
    _run([(0.01, 0.015)], [(0.05, 0.015)], 0.12, check=True)


def test_lc_against_reference():
    _compare([(0.05, 0.015)], [], 0.95, ("env", "v_lc", "v_sc"), 3e-3)


def test_sc_against_reference():
    _compare([], [(0.05, 0.015)], 0.4, ("env", "v_lc", "v_sc"), 3e-3)


if __name__ == "__main__":
    test_root_coherence()
    test_lc_against_reference()
    test_sc_against_reference()
    print("ok")
