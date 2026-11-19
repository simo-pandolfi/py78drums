#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
Regression test for C8: maracas branch with C25 in series, D8 and D9 (from
C15 part of pe78/noise_voices.py).

Before C8, C25 was a shunt low-pass IIR followed as a Thevenin by the D9
anode: a DC path from the M253 output to C26 that the schematic does not
have. V_C26 rose to 3.00 V with a 15 ms gate and stayed at 3.60 V as long
as a 100 ms gate (plateau); D8 was not modelled. In the circuit the charge
is shared between C25 and C26: V_C26 peaks (2.36 V at 24 ms) and decays
while the gate is still high; after the gate D8 clamps node A.

1. Against tools/noise_voices_ref.py: V_C26, V_ENV and node A for gates of
   15 ms and 100 ms (edges +-1 ms excluded), V_C26 at the end of each gate.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _nv_common import NoiseVoices, ref, run_model, edge_mask, max_err  # noqa: E402

FS = 96000
T0 = 0.05


def _gate_case(gate_s):
    g = {"mr": [(T0, gate_s)]}
    t, o, _ = run_model(g, 0.4, FS)
    tr, r = ref.simulate(g, 0.4)
    err = max_err(t, o, tr, r, ("mr", "env", "a"), edge_mask(t, g))
    for k, tol in (("mr", 2e-3), ("env", 2e-3), ("a", 1e-3)):
        assert err[k] < tol, (gate_s, k, err[k])
    k = np.searchsorted(t, T0 + gate_s) - 1
    v_end, v_end_ref = o["mr"][k], np.interp(t[k], tr, r["mr"])
    assert abs(v_end - v_end_ref) < 3e-3, (gate_s, v_end, v_end_ref)
    after = t > T0 + gate_s + 1e-3
    assert o["a"].min() > -0.7, o["a"].min()
    assert o["a"][after].min() < -0.3, o["a"][after].min()
    return o["mr"].max(), v_end


def test_gate_15ms():
    _gate_case(0.015)


def test_gate_100ms_no_plateau():
    peak, v_end = _gate_case(0.1)
    assert v_end < 0.85 * peak, (peak, v_end)


def test_root_residuals():
    nv = NoiseVoices(FS)
    for k in range(int(0.1 * FS)):
        nv.tick(0, 0, 0, nv.V_ON if T0 * FS <= k < (T0 + 0.015) * FS else 0.0, 0.0)
        kcl, dv, di = nv._root_mr.residuals()
        assert kcl < 1e-12 and dv < 1e-12 and di < 1e-15, (k, kcl, dv, di)


if __name__ == "__main__":
    test_gate_15ms()
    test_gate_100ms_no_plateau()
    test_root_residuals()
    print("ok")
