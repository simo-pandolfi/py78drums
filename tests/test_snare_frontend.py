#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
Regression test for C6: snare trigger front end and C27 (from C15 part of
pe78/noise_voices.py, _SnareFrontEnd).

Before C6 the TR5 emitter followed the gate level (no C22/R48), D7 had no
drop and C27 decayed with a fixed tau (R50+R51)*C27, without the TR3 base
current: V_C27 peak 3.85 V, base above 0.5 V for 134 ms (gate 1 ms) to
233 ms (gate 100 ms).

1. The decay does not depend on the gate length (C22/R48 differentiator).
2. Against tools/noise_voices_ref.py (whole section, nodal, trapezoidal):
   V_C27 peak, V_C27, TR3 base and TR5 base after the attack, time of the
   TR3 base below 0.5 V (C15: C28 and node N included).
3. Retrigger after 62.5 ms: C22 discharges through D6 and R21; without R21
   the second hit does not recharge C27.
4. Junction and Newton agree: KCL at every node of the front end, port
   voltages equal to the nonlinear laws of TR5+D7 and D6.
5. C27 gives the TR3 base current: with TR3 off (Is_b3 = 0) the half-time
   of C27 is ln2*(R50+R51)*C27 = 69.3 ms; with TR3 on it is shorter.

Usage (from the py78drums root):
    python -m pytest tests/test_snare_frontend.py
    python tests/test_snare_frontend.py

pywdf is taken from the environment; set PYWDF_ROOT to the folder that
contains the pywdf package to use a specific copy (for example pywdf-ref).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _nv_common import NoiseVoices, ref, run_model, edge_mask, max_err  # noqa: E402

FS = 96000
T0 = 0.01


def test_gate_independence():
    peaks, ends = [], []
    for g_ms in (1, 10, 100):
        t, o, m = run_model({"sd": [(T0, g_ms * 1e-3)]}, 0.2, FS)
        peaks.append(o["c27"].max())
        ends.append(1e3 * (ref.last_above(t, o["b3"], 0.5) - T0))
        assert m.sd.iter_max <= 30, m.sd.iter_max
    assert max(peaks) - min(peaks) < 1e-6 * max(peaks), peaks
    assert max(ends) - min(ends) < 0.1, ends          # ms


def test_against_reference():
    g = {"sd": [(T0, 0.01)]}
    t, o, _ = run_model(g, 0.15, FS)
    tr, r = ref.simulate(g, 0.15)
    pk_m, pk_r = o["c27"].max(), r["c27"].max()
    assert abs(pk_m - pk_r) < 2e-3 * pk_r, (pk_m, pk_r)
    late = (t > T0 + 2e-3) & edge_mask(t, g)
    err = max_err(t, o, tr, r, ("c27", "b3", "vb5"), late)
    for k, tol in (("c27", 2e-3), ("b3", 2e-3), ("vb5", 7e-3)):
        assert err[k] < tol, (k, err[k])
    e_m, e_r = ref.last_above(t, o["b3"], 0.5), ref.last_above(tr, r["b3"], 0.5)
    assert abs(e_m - e_r) < 5e-4, (e_m, e_r)


def test_retrigger():
    g = {"sd": [(T0, 0.01), (T0 + 0.0625, 0.01)]}
    m2 = T0 + 0.0625
    t, o, _ = run_model(g, 0.12, FS)
    k = t >= m2
    ratio = o["c27"][k].max() / o["c27"][~k].max()
    assert ratio > 0.95, ratio
    nv = NoiseVoices(FS)
    nv.sd.R21 = 1e12                       # no load on the trigger line
    nv.sd.build()
    t, o, _ = run_model(g, 0.12, FS, nv=nv)
    v_before = o["c27"][~k][-1]
    assert o["c27"][k].max() < v_before + 0.05, (o["c27"][k].max(), v_before)


def test_junction_kcl_and_ports():
    nv = NoiseVoices(FS)
    fe = nv.sd
    vt_t, vt_d = fe.TR5_NF * fe.VT_27C, fe.D_N * fe.VT_27C
    checked = 0
    for k in range(int(0.03 * FS)):
        nv.tick(nv.V_ON if T0 * FS <= k < (T0 + 0.01) * FS else 0.0, 0, 0, 0, 0.0)
        leaves = [leaf for leaf, _, _ in fe.leaves]
        vL = np.array([leaf.wave_to_voltage() for leaf in leaves])
        iL = np.array([(leaf.a - leaf.b) / (2.0 * leaf.Rp) for leaf in leaves])
        v = np.linalg.lstsq(fe.AL.T, vL, rcond=None)[0]
        assert np.max(np.abs(fe.AL.T @ v - vL)) < 1e-9
        ie = fe.TR5_IS * (1 + 1 / fe.TR5_BF) * (np.exp(fe.w / vt_t) - 1)
        ib = ie / (fe.TR5_BF + 1)
        i6 = fe.D_IS * (np.exp(fe.v6 / vt_d) - 1)
        iN = np.array([ib, -(ie - ib), i6])
        kcl = fe.AL @ iL + fe.AN @ iN
        assert np.max(np.abs(kcl)) < 1e-12, (k, kcl)
        vN = fe.AN.T @ v
        assert abs(vN[0] - (fe.w + vt_d * np.log1p(ie / fe.D_IS))) < 1e-8, k
        assert abs(vN[2] - fe.v6) < 1e-8, k
        if ie > 1e-6 or i6 > 1e-6:
            checked += 1
    assert checked > 10, checked


def _half_time_ms(t, v):
    k = int(np.argmax(v))
    i = np.nonzero(v[k:] > 0.5 * v[k])[0][-1] + k
    return 1e3 * (t[i] - t[k])


def test_c27_gives_base_current():
    g = {"sd": [(T0, 0.01)]}
    nv = NoiseVoices(FS)
    nv.Is_b3 = 0.0
    t, o, _ = run_model(g, 0.25, FS, nv=nv)
    th_off = _half_time_ms(t, o["c27"])
    expected = np.log(2) * (nv.sd.R50 + nv.R51) * nv.sd.C27 * 1e3
    assert abs(th_off - expected) < 0.01 * expected, (th_off, expected)
    t, o, _ = run_model(g, 0.25, FS)
    th_on = _half_time_ms(t, o["c27"])
    assert th_on < 0.85 * th_off, (th_on, th_off)


if __name__ == "__main__":
    test_gate_independence()
    test_against_reference()
    test_retrigger()
    test_junction_kcl_and_ports()
    test_c27_gives_base_current()
    print("ok")
