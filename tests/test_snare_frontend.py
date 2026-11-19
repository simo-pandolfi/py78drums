#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
Regression test for C6: snare trigger front end and C27 (pe78/snare.py).

Before C6 the TR5 emitter followed the gate level (no C22/R48), D7 had no
drop and C27 decayed with a fixed tau (R50+R51)*C27, without the TR3 base
current: V_C27 peak 3.85 V, base above 0.5 V for 134 ms (gate 1 ms) to
233 ms (gate 100 ms).

1. The decay does not depend on the gate length (C22/R48 differentiator).
2. Against tools/snare_frontend_ref.py (nodal analysis, trapezoidal, 5 us
   step, TR3 base solved together with the rest): V_C27 peak, V_C27 and
   TR3 base voltage after the attack, time of the base below 0.5 V.
3. Retrigger after 62.5 ms: C22 discharges through D6 and R21; without R21
   the second hit does not recharge C27.
4. Junction and Newton agree: KCL at every node of the front end, port
   voltages equal to the nonlinear laws of TR5+D7 and D6.
5. C27 gives the TR3 base current: with TR3 off (Is_b = 0) the half-time of
   C27 is ln2*(R50+R51)*C27 = 69.3 ms; with TR3 on it is shorter.

Usage (from the py78drums root):
    python -m pytest tests/test_snare_frontend.py
    python tests/test_snare_frontend.py

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

from pe78.snare import PE78_Snare  # noqa: E402
import snare_frontend_ref as ref  # noqa: E402

FS = 48000
T0 = 0.01


def _run(gates, t_end, cls=PE78_Snare, is_b=None):
    m = cls(FS)
    if is_b is not None:
        m.Is_b = is_b
    n = int(round(t_end * FS))
    t = np.arange(n) / FS
    out = {k: np.zeros(n) for k in ("vc27", "vbase3", "vb5")}
    for k in range(n):
        on = any(a <= t[k] < a + g for a, g in gates)
        m.process_sample(m.V_ON if on else 0.0, 0.0)
        out["vc27"][k] = m._V_c27
        out["vbase3"][k] = m._V_base
        out["vb5"][k] = m._V_b5
    return t, out, m


def test_gate_independence():
    peaks, ends = [], []
    for g_ms in (1, 10, 100):
        t, o, m = _run([(T0, g_ms * 1e-3)], 0.25)
        peaks.append(o["vc27"].max())
        ends.append(ref.base_end_ms(t, o["vbase3"], T0))
        assert m.fe_iter_max <= 30, m.fe_iter_max
    assert max(peaks) - min(peaks) < 1e-6 * max(peaks), peaks
    assert max(ends) - min(ends) < 0.1, ends          # ms


def test_against_reference():
    t, o, _ = _run([(T0, 0.01)], 0.15)
    tr, r = ref.simulate([(T0, 0.01)], 0.15, dt=5e-6)
    pk_m, pk_r = o["vc27"].max(), r["vc27"].max()
    assert abs(pk_m - pk_r) < 2e-3 * pk_r, (pk_m, pk_r)
    late = t > T0 + 2e-3                                # after the attack
    for k, tol in (("vc27", 2e-3), ("vbase3", 5e-4)):
        d = np.abs(o[k] - np.interp(t, tr, r[k]))[late].max()
        assert d < tol, (k, d)
    e_m = ref.base_end_ms(t, o["vbase3"], T0)
    e_r = ref.base_end_ms(tr, r["vbase3"], T0)
    assert abs(e_m - e_r) < 0.5, (e_m, e_r)


class _NoR21(PE78_Snare):
    R21 = 1e12          # no load on the trigger line


def test_retrigger():
    gates = [(T0, 0.01), (T0 + 0.0625, 0.01)]
    m2 = T0 + 0.0625
    t, o, _ = _run(gates, 0.12)
    k = t >= m2
    ratio = o["vc27"][k].max() / o["vc27"][~k].max()
    assert ratio > 0.95, ratio
    t, o, _ = _run(gates, 0.12, cls=_NoR21)
    k = t >= m2
    v_before = o["vc27"][~k][-1]
    assert o["vc27"][k].max() < v_before + 0.05, (o["vc27"][k].max(), v_before)


def test_junction_kcl_and_ports():
    m = PE78_Snare(FS)
    AL, AN = m._fe_AL, m._fe_AN
    vt_t, vt_d = m.TR5_NF * m.VT_27C, m.D_N * m.VT_27C
    checked = 0
    for k in range(int(0.03 * FS)):
        m.process_sample(m.V_ON if T0 * FS <= k < (T0 + 0.01) * FS else 0.0, 0.0)
        leaves = [leaf for leaf, _, _ in m._fe_leaves]
        vL = np.array([leaf.wave_to_voltage() for leaf in leaves])
        iL = np.array([(leaf.a - leaf.b) / (2.0 * leaf.Rp) for leaf in leaves])
        v = np.linalg.lstsq(AL.T, vL, rcond=None)[0]          # node voltages
        assert np.max(np.abs(AL.T @ v - vL)) < 1e-9
        ie = m.TR5_IS * (1 + 1 / m.TR5_BF) * (np.exp(m._fe_w / vt_t) - 1)
        ib = ie / (m.TR5_BF + 1)
        i6 = m.D_IS * (np.exp(m._fe_v6 / vt_d) - 1)
        iN = np.array([ib, -(ie - ib), i6])
        kcl = AL @ iL + AN @ iN
        assert np.max(np.abs(kcl)) < 1e-12, (k, kcl)
        vN = AN.T @ v
        vA = m._fe_w + vt_d * np.log1p(ie / m.D_IS)
        assert abs(vN[0] - vA) < 1e-8, (k, vN[0], vA)
        assert abs(vN[2] - m._fe_v6) < 1e-8, (k, vN[2], m._fe_v6)
        if ie > 1e-6 or i6 > 1e-6:
            checked += 1
    assert checked > 10, checked      # TR5 and D6 conducting in the test window


def _half_time_ms(t, v):
    k = int(np.argmax(v))
    i = np.nonzero(v[k:] > 0.5 * v[k])[0][-1] + k
    return 1e3 * (t[i] - t[k])


def test_c27_gives_base_current():
    t, o, _ = _run([(T0, 0.01)], 0.25, is_b=0.0)
    th_off = _half_time_ms(t, o["vc27"])
    expected = np.log(2) * (PE78_Snare.R50 + PE78_Snare.R51) * PE78_Snare.C27 * 1e3
    assert abs(th_off - expected) < 0.01 * expected, (th_off, expected)
    t, o, _ = _run([(T0, 0.01)], 0.25)
    th_on = _half_time_ms(t, o["vc27"])
    assert th_on < 0.85 * th_off, (th_on, th_off)


if __name__ == "__main__":
    test_gate_independence()
    test_against_reference()
    test_retrigger()
    test_junction_kcl_and_ports()
    test_c27_gives_base_current()
    print("ok")
