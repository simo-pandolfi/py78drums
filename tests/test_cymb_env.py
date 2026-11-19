#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
Regression test for C7: TR4 base load on the cymbal envelope node ENV (from
C15 a port of the bases root of pe78/noise_voices.py).

Before C7 the TR4 base was a linear port r_pi (previous sample) inside the
ENV scatter, and the Newton-Raphson on the Shockley law ran apart: the
subtrees saw a node at about 68 mV while the NR gave 620 mV, so C23, C24 and
C26 discharged into the base with about 2.6 times the real base current
(LC: V_ENV > 0.5 V until 0.648 s instead of 0.804 s).

1. The bases root is coherent at every sample: KCL at B3, N, ENV and the
   Shockley laws of both bases (RTypeNLRoot.residuals()).
2. Against tools/noise_voices_ref.py (whole section, all simultaneous):
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _nv_common import NoiseVoices, ref, run_model, edge_mask, max_err  # noqa: E402

FS = 96000


def test_root_coherence():
    nv = NoiseVoices(FS)
    n = int(0.12 * FS)
    for k in range(n):
        t = k / FS
        nv.tick(4.5 if 0.06 <= t < 0.07 else 0.0, 4.5 if 0.01 <= t < 0.025 else 0.0,
                4.5 if 0.05 <= t < 0.065 else 0.0, 0.0, 0.0)
        kcl, dv, di = nv._root_b.residuals()
        assert kcl < 1e-15 and dv < 1e-12 and di < 1e-15, (k, kcl, dv, di)


def _compare(g, t_end, tol):
    t, o, _ = run_model(g, t_end, FS)
    tr, r = ref.simulate(g, t_end, dt=1e-5)
    err = max_err(t, o, tr, r, ("env", "lc", "sc"), edge_mask(t, g))
    for k, d in err.items():
        assert d < tol[k], (k, d)
    for thr in (0.55, 0.5):
        tm, trf = ref.last_above(t, o["env"], thr), ref.last_above(tr, r["env"], thr)
        assert abs(tm - trf) < 2e-3, (thr, tm, trf)


def test_lc_against_reference():
    _compare({"lc": [(0.05, 0.015)]}, 0.95, {"env": 1.5e-3, "lc": 3e-4, "sc": 3e-4})


def test_sc_against_reference():
    _compare({"sc": [(0.05, 0.015)]}, 0.4, {"env": 2e-3, "lc": 3e-4, "sc": 8e-3})


if __name__ == "__main__":
    test_root_coherence()
    test_lc_against_reference()
    test_sc_against_reference()
    print("ok")
