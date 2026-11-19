#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
Regression test for C8: maracas branch with C25 in series, D8 and D9.

Before C8, C25 was a shunt low-pass IIR followed as a Thevenin by the D9
anode: a DC path from the M253 output to C26 that the schematic does not
have. V_C26 rose to 3.00 V with a 15 ms gate and stayed at 3.60 V as long
as a 100 ms gate (plateau); D8 was not modelled. In the circuit the charge
is shared between C25 and C26: V_C26 peaks (2.36 V at 24 ms) and decays
while the gate is still high; after the gate D8 clamps node A.

1. Against tools/cymb_env_ref.py (nodal, trapezoidal, all simultaneous):
   V_C26, V_ENV and node A for gates of 15 ms and 100 ms (edges +-1 ms
   excluded), V_C26 at the end of each gate.
2. No plateau: with a 100 ms gate V_C26 at the end of the gate is below
   0.85 times its peak.
3. D8: node A stays above -0.7 V and goes below -0.3 V after the gate.
4. Root of the branch (pe78.wdf_nlroot): KCL and diode laws at every sample.

Usage (from the py78drums root):
    python -m pytest tests/test_cymb_mr.py
    python tests/test_cymb_mr.py

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
T0 = 0.05


def _run(gate_s, t_end=0.4, check=False):
    m = PE78_Cymbals_Maracas(FS)
    n = int(round(t_end * FS))
    t = np.arange(n) / FS
    out = {k: np.zeros(n) for k in ("v_mr", "env", "v_a")}
    for k in range(n):
        m.process_sample(0.0, 0.0, m.V_ON if T0 <= t[k] < T0 + gate_s else 0.0, 0.0)
        out["v_mr"][k] = m._V_c26
        out["env"][k] = m._V_env
        out["v_a"][k] = m._V_a
        if check:
            kcl, dv, di = m._root_mr.residuals()
            assert kcl < 1e-12 and dv < 1e-12 and di < 1e-15, (k, kcl, dv, di)
    return t, out


def _gate_case(gate_s, check=False):
    t, o = _run(gate_s, check=check)
    tr, r = ref.simulate([], [], 0.4, gates_mr=[(T0, gate_s)])
    keep = (np.abs(t - T0) > 1e-3) & (np.abs(t - T0 - gate_s) > 1e-3)
    for key, tol in (("v_mr", 3e-3), ("env", 3e-3), ("v_a", 3e-3)):
        d = np.abs(o[key] - np.interp(t, tr, r[key]))[keep].max()
        assert d < tol, (gate_s, key, d)
    k = np.searchsorted(t, T0 + gate_s) - 1
    v_end, v_end_ref = o["v_mr"][k], np.interp(t[k], tr, r["v_mr"])
    assert abs(v_end - v_end_ref) < 3e-3, (gate_s, v_end, v_end_ref)
    after = t > T0 + gate_s + 1e-3
    assert o["v_a"].min() > -0.7, o["v_a"].min()
    assert o["v_a"][after].min() < -0.3, o["v_a"][after].min()
    return o["v_mr"].max(), v_end


def test_gate_15ms():
    _gate_case(0.015, check=True)


def test_gate_100ms_no_plateau():
    peak, v_end = _gate_case(0.1)
    assert v_end < 0.85 * peak, (peak, v_end)


if __name__ == "__main__":
    test_gate_15ms()
    test_gate_100ms_no_plateau()
    print("ok")
