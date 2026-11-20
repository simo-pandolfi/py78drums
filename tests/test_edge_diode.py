#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
Regression test for C14: D2 of the EdgeDetector as the LTspice 1N4148.

Before C14, D2 was wdf.Diode(Is = 2.52 nA) with the default n_diodes = 1,
i.e. emission coefficient N = 1: drop about 0.33 V at 1 mA. The LTspice
standard.dio model (used for D6-D9 since C6/C8) has N = 1.752: drops
0.38/0.48/0.58 V at 0.01/0.1/1 mA (user's data).

1. EdgeDetector constants and pywdf Diode thermal voltage: Vt * n_diodes =
   N * Vt with N = 1.752.
2. Closed-form drops of the law at 0.01/0.1/1 mA (+-10 mV).
3. EdgeDetector against a nodal reference of the same circuit (trapezoidal
   C8 at the same fs, exact Newton on the diode): voltage on R13 for 10 ms
   and 100 ms gates at 4.5 V and 5 V and a retrigger.
   a. With the exact Wright omega in place of pywdf's omega4 the model equals
      the reference (< 1e-9 V): same circuit, same discretisation.
   b. With omega4 (as used) the model follows the reference with N = 1.752
      within 2 mV (peak 3.99 V) and is >= 0.2 V away from N = 1.
4. At every conducting sample (i > 0.1 uA) the diode port voltage is the
   Shockley drop with N = 1.752 within 3 mV.

pywdf's Diode.omega4 (4th-order approximation of the Wright omega function)
has a relative error up to about 3.5 % (x near -3) and 0.9 % (x near 6),
which here is up to 2.4 mV on the diode voltage (about 5 % of the current at
10 uA). This is a property of pywdf, measured, not changed here; D2 moves to
the Newton root of the twin-T in C13.

Usage (from the py78drums root):
    python -m pytest tests/test_edge_diode.py
    python tests/test_edge_diode.py

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

from pe78 import drums                    # noqa: E402
from pe78.twint import EdgeDetector       # noqa: E402

FS = 48000
P = drums.BDO_PARAMS
ED = EdgeDetector


def drop(i, n=ED.D_N):
    return n * ED.VT_27C * np.log1p(i / ED.D_IS)


def gates(n):
    """(name, trigger) cases; 1 s at FS."""
    def g(start, length, level):
        v = np.zeros(n)
        a = int(round(start * FS))
        v[a:a + int(round(length * FS))] = level
        return v
    return [("4.5V_10ms", g(0.01, 0.010, 4.5)),
            ("4.5V_100ms", g(0.01, 0.100, 4.5)),
            ("5V_10ms", g(0.01, 0.010, 5.0)),
            ("retrigger", g(0.01, 0.010, 4.5) + g(0.06, 0.010, 4.5))]


def reference(trig, n_em, r_s=1e-3):
    """Nodal model of the EdgeDetector: nodes T1, T2, A (anode of D2).

    C8 trapezoidal at FS (as the WDF capacitor); the linear part is solved
    exactly, Newton on the diode voltage against its Thevenin.
    Returns the voltage on R13 (T2 - A) and the diode current.
    """
    r14, r15, r13 = (float(x) for x in P["edge_r"][:3])
    gc = 2.0 * float(P["edge_c"]) * FS
    nvt = n_em * ED.VT_27C
    G = np.array([[1 / r_s + 1 / r14 + gc, -gc, 0.0],
                  [-gc, gc + 1 / r15 + 1 / r13, -1 / r13],
                  [0.0, -1 / r13, 1 / r13]])
    Gi = np.linalg.inv(G)
    z = Gi[:, 2]                              # node voltages per unit diode current
    r_th = z[2]
    u_prev, ic_prev = 0.0, 0.0
    out = np.empty(len(trig))
    idd = np.empty(len(trig))
    for k, vs in enumerate(trig):
        hist = gc * u_prev + ic_prev          # i_c = gc*u - hist
        x0 = Gi @ np.array([vs / r_s + hist, -hist, 0.0])
        v_th = x0[2]
        # f(vd) = vd + r_th*I(vd) - v_th is increasing and convex: Newton from
        # a point right of the root converges monotonically.
        vd = 0.0 if v_th <= 0 else min(v_th, nvt * np.log1p(v_th / (r_th * ED.D_IS)))
        for _ in range(200):
            e = np.exp(vd / nvt)
            f = vd + r_th * ED.D_IS * (e - 1.0) - v_th
            dv = -f / (1.0 + r_th * ED.D_IS * e / nvt)
            vd += dv
            if abs(dv) < 1e-14:
                break
        else:
            raise RuntimeError("reference Newton did not converge")
        i_d = ED.D_IS * (np.exp(vd / nvt) - 1.0)
        x = x0 - z * i_d
        u = x[0] - x[1]
        ic_prev = gc * u - hist
        u_prev = u
        out[k] = x[1] - x[2]
        idd[k] = i_d
    return out, idd


def run_model(trig, exact_omega=False):
    ed = ED(FS, P["edge_r"], P["edge_c"])
    if exact_omega:
        from scipy.special import wrightomega
        ed.diode.omega4 = lambda x: float(np.real(wrightomega(x)))
    out = np.empty(len(trig))
    a = np.empty(len(trig))
    b = np.empty(len(trig))
    for k, vs in enumerate(trig):
        out[k] = ed.process_sample(vs)
        a[k], b[k] = ed.diode.a, ed.diode.b
    return out, a, b, ed.s_r13.Rp


def test_parameters():
    d = ED(FS, P["edge_r"], P["edge_c"]).diode
    assert d.Is == 2.52e-9
    assert abs(d.Vt - 1.752 * 25.852e-3) < 1e-15, d.Vt


def test_drops():
    for i, ref in ((1e-5, 0.38), (1e-4, 0.48), (1e-3, 0.58)):
        assert abs(drop(i) - ref) < 0.01, (i, drop(i))
    # N = 1 (before C14) is far from the LTspice model at 1 mA
    assert drop(1e-3) - drop(1e-3, n=1.0) > 0.24


def test_against_reference(verbose=False):
    n = FS
    for name, trig in gates(n):
        y, _, _, _ = run_model(trig)
        y_x, _, _, _ = run_model(trig, exact_omega=True)
        r_ok, i_ok = reference(trig, ED.D_N)
        r_old, _ = reference(trig, 1.0)
        e_x = np.max(np.abs(y_x - r_ok))
        e_ok = np.max(np.abs(y - r_ok))
        e_old = np.max(np.abs(y - r_old))
        if verbose:
            print(f"  {name:11s} peak {r_ok.max():.4f} V (N=1: {r_old.max():.4f} V), "
                  f"I_D max {1e3 * i_ok.max():.3f} mA; max|model-ref|: exact omega "
                  f"{e_x:.1e} V, omega4 {e_ok:.2e} V, vs N=1 {e_old:.3f} V")
        assert e_x < 1e-9, (name, e_x)
        assert e_ok < 2e-3, (name, e_ok)
        assert e_old > 0.2, (name, e_old)


def test_port_law(verbose=False):
    _, trig = gates(FS)[0]
    _, a, b, rp = run_model(trig)
    v = 0.5 * (a + b)
    i = (a - b) / (2.0 * rp)
    on = i > 1e-7
    dv = np.max(np.abs(v[on] - drop(i[on])))
    if verbose:
        print(f"  conducting samples {on.sum()}, V_D max {v.max():.4f} V, "
              f"max |V_D - drop(I_D)| {1e3 * dv:.2f} mV")
    assert on.sum() > 10
    assert dv < 3e-3, dv


if __name__ == "__main__":
    test_parameters()
    test_drops()
    print("drops 0.01/0.1/1 mA: " + ", ".join(f"{drop(i):.4f}" for i in (1e-5, 1e-4, 1e-3))
          + " V (N = 1: " + ", ".join(f"{drop(i, 1.0):.4f}" for i in (1e-5, 1e-4, 1e-3)) + " V)")
    test_against_reference(verbose=True)
    test_port_law(verbose=True)
    print("OK")
