#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
Regression test for C12: CD4011A gate of the twin-T as a nonlinear element.

Before C12 the gate was a linear inverter (a = -40, r_o = 1 kOhm) inside the
R-type scattering matrix: no DC operating point, and above the VR4 threshold
(bass drum about 189.5 kOhm) the output grew without limit (2.3e18 V after
2 s at 300 kOhm). Already at the default VR4 = 150 kOhm the linear V_n3
reached +3.5/-3.4 V around the operating point, beyond the 0-5 V swing.

1. Gate law: unloaded switching voltage = vm (1.5 ... 3.5 V); stack current
   0.5 mA at the switching point (vm = 2.5 V); gm = 2.0 mS, A0 = 109.9;
   Jacobian of the multiport law against finite differences (saturation,
   triode, negative V_DS).
2. DC operating point: at rest V_n3 stays at V_dc within 1 uV (rounding
   residues of the start, about 1e-7 V) and the TonalDrum output within
   10 nV (no power-on transient).
3. Small signal: threshold of the linearised circuit 189.5 kOhm (as the
   former linear model, decision D4-A, tuned on the circuit of C13: n1
   loaded by R16 and D2 at 0 V, decision E2; 164.7 kOhm with the 27 kOhm
   load of before C13), also with vm = 1.5 and 3.5 V; simulated
   frequency and decay after a 1 mV step on C7 against the dominant pole of
   the linearised circuit (bass drum, VR4 = 150 kOhm).
4. Above the threshold (VR4 = 220 and 470 kOhm): bounded oscillation inside
   0 ... 5 V with a steady amplitude.
5. Newton at the root: at most 20 iterations (claves up to 10, the others
   4; step limit 0.2 V), KCL and port laws satisfied,
   for every tonal voice at VR4 = 0, 150 and 470 kOhm with a full trigger.

Usage (from the py78drums root):
    python -m pytest tests/test_cd4011.py
    python tests/test_cd4011.py           (about 1-2 minutes)

pywdf is taken from the environment; set PYWDF_ROOT to the folder that
contains the pywdf package to use a specific copy (for example pywdf-ref).
"""

import os
import sys

import numpy as np
import scipy.linalg as sl
from scipy.optimize import brentq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
if os.environ.get("PYWDF_ROOT"):
    sys.path.insert(0, os.path.abspath(os.environ["PYWDF_ROOT"]))

from pe78 import drums                       # noqa: E402
from pe78.twint import CD4011AGate           # noqa: E402

FS = 48000
VOICES = ("BDO_PARAMS", "CONGA_PARAMS", "HBONGO_PARAMS", "LBONGO_PARAMS", "CLAVES_PARAMS")
THR_LINEAR = 189.54e3       # VR4 threshold of the former linear model (bass drum)
THR_PRE_C13 = 164.65e3      # same gate (lambda of E2) with the 27 kOhm load on n1 of before C13
G_PRE_C13 = 1 / 27e3        # load of n1 before C13: trigger source behind R13


def _drum(name="BDO_PARAMS", vr4=None, vm=None):
    d = drums.TonalDrum(FS, **getattr(drums, name), vm=vm)
    if vr4 is not None:
        d.set_decay(vr4)
    return d


def _run(d, n, level=4.5, gate_samples=480):
    """Returns V_n3, TonalDrum output."""
    vo, y = np.empty(n), np.empty(n)
    for k in range(n):
        y[k] = d.tick(level if k < gate_samples else 0.0)
        vo[k] = d.voice.v_out
    return vo, y


# -----------------------------------------------------------------------------
# Linearised circuit (nodal, continuous time)
# -----------------------------------------------------------------------------
def _poles(voice, vr4, g_d2=None):
    """Poles of the twin-T linearised at the DC point. At rest the M253 output
    is off and D2 is at 0 V: the detector loads n1 through the D2
    conductance Is/(N Vt) (18 MOhm) in series with R13 and R15 || (R14 + C8);
    it is included as D2 alone (the rest is negligible against it).
    g_d2 overrides that conductance (e.g. 1/27 kOhm for the circuit before C13)."""
    v = voice.v_dc
    _, gm, go = voice.gate.current(v, v)
    r11, r12, r16, rb, r41 = (c.Rp for c in (voice.r11, voice.r12, voice.r16,
                                              voice.r_bias, voice.r_out))
    c5, c6, c7 = voice.c5.C, voice.c6.C, voice.c7.C
    if g_d2 is None:
        g_d2 = voice.D_IS / (voice.D_N * voice.VT_27C)
    G = np.zeros((4, 4))
    C = np.zeros((4, 4))

    def add(M, p, q, val):
        for a, b, s in ((p, p, 1), (q, q, 1), (p, q, -1), (q, p, -1)):
            if a >= 0 and b >= 0:
                M[a, b] += s * val
    add(G, 2, 0, 1 / r11)
    add(G, 0, 3, 1 / r12)
    add(G, 1, -1, 1 / r16 + g_d2)
    add(G, 2, 3, 1 / (voice.R10 + vr4))
    add(G, 3, -1, 1 / r41)
    add(C, 0, -1, c5)
    add(C, 2, 1, c6)
    add(C, 1, 3, c7)
    G[3, 2] += gm
    G[3, 3] += go
    w = sl.eigvals(-G, C)
    return w[np.isfinite(w)]


def _dominant(voice, vr4, g_d2=None):
    w = _poles(voice, vr4, g_d2)
    return w[np.argmax(w.real)]


def _threshold(voice, g_d2=None):
    return brentq(lambda r: _dominant(voice, r, g_d2).real, 20e3, 1e6, xtol=1.0)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_gate_law(verbose=False):
    for vm in (1.5, 2.0, 2.5, 3.0, 3.5):
        g = CD4011AGate(vm)
        assert abs(g.current(vm, vm)[0]) < 1e-15, vm
    g = CD4011AGate()
    i_n = 0.5 * g.K * (2.5 - g.vtn) ** 2 * (1 + g.LAM * 2.5)
    assert abs(i_n - 0.5e-3) < 1e-12, i_n
    _, gm, go = g.current(2.5, 2.5)
    assert abs(gm - 2.0e-3) < 1e-6 and abs(gm / go - 109.9) < 0.1, (gm, gm / go)
    if verbose:
        print(f"  K {g.K * 1e3:.4f} mA/V^2, lambda {g.LAM} 1/V, gm {gm * 1e3:.4f} mS, "
              f"A0 {gm / go:.2f}, r_out {1e-3 / go:.2f} kOhm")
    rng = np.random.default_rng(1)
    h = 1e-7
    for vm in (1.5, 2.5, 3.5):
        g = CD4011AGate(vm)
        for _ in range(200):
            v = np.array([rng.uniform(-0.5, 5.5), rng.uniform(-0.5, 5.5)])
            i, J = g.law(v)
            for c in range(2):
                dv = np.zeros(2)
                dv[c] = h
                fd = (g.law(v + dv)[0] - g.law(v - dv)[0]) / (2 * h)
                assert np.allclose(fd, J[:, c], rtol=1e-5, atol=1e-10), (vm, v, c, fd, J)


def test_dc_point():
    for vm in (None, 1.5, 3.5):
        d = _drum(vm=vm)
        v_dc = d.voice.v_dc
        ref = 2.5 if vm is None else vm
        assert -5e-3 < v_dc - ref < 0, (vm, v_dc)
        vo, y = _run(d, FS // 2, level=0.0)
        assert np.max(np.abs(vo - v_dc)) < 1e-6, (vm, np.max(np.abs(vo - v_dc)))
        assert np.max(np.abs(y)) < 1e-8, (vm, np.max(np.abs(y)))


def test_small_signal(verbose=False):
    for vm in (None, 1.5, 3.5):
        v = _drum(vm=vm).voice
        thr_pre, thr = _threshold(v, G_PRE_C13), _threshold(v)
        if verbose:
            print(f"  vm {2.5 if vm is None else vm}: VR4 threshold {thr / 1e3:.2f} kOhm "
                  f"(n1 loaded by 27 kOhm as before C13: {thr_pre / 1e3:.2f} kOhm)")
        assert abs(thr - THR_LINEAR) < 0.5e3, (vm, thr)
        assert abs(thr_pre - THR_PRE_C13) < 0.5e3, (vm, thr_pre)
    # simulated small-signal response, bass drum, VR4 = 150 kOhm, 1 mV trigger
    d = _drum(vr4=150e3)
    p = _dominant(d.voice, 150e3)
    f_ref, tau_ref = abs(p.imag) / (2 * np.pi), -1 / p.real
    f_ref_w = FS / np.pi * np.arctan(np.pi * f_ref / FS)        # bilinear warping
    d.voice.c7.z -= 1e-3          # 1 mV step on C7 (the M253 input switches above 0.5 V)
    vo, _ = _run(d, FS, level=0.0)
    x = vo - d.voice.v_dc
    t = np.arange(len(x)) / FS
    w = slice(int(0.05 * FS), int(0.6 * FS))
    zc = np.nonzero(np.diff(np.sign(x[w])) != 0)[0]
    # linear interpolation of the zero crossings
    tz = t[w][zc] - x[w][zc] * (t[w][zc + 1] - t[w][zc]) / (x[w][zc + 1] - x[w][zc])
    f_sim = (len(tz) - 1) / (2 * (tz[-1] - tz[0]))
    peaks = [np.max(np.abs(x[w][a:b])) for a, b in zip(zc[:-1], zc[1:])]
    tp = 0.5 * (tz[:-1] + tz[1:])
    slope = np.polyfit(tp, np.log(peaks), 1)[0]
    tau_sim = -1 / slope
    if verbose:
        print(f"  BD 150k, 1 mV on C7: f {f_sim:.3f} Hz (pole {f_ref:.3f}, warped "
              f"{f_ref_w:.3f}), tau {1e3 * tau_sim:.2f} ms (pole {1e3 * tau_ref:.2f}); "
              f"peak |V_n3 - V_dc| {np.max(np.abs(x)) * 1e3:.3f} mV")
    assert abs(f_sim / f_ref_w - 1) < 2e-3, (f_sim, f_ref_w)
    assert abs(tau_sim / tau_ref - 1) < 2e-2, (tau_sim, tau_ref)


def test_large_signal(verbose=False):
    for vr4 in (220e3, 470e3):
        d = _drum(vr4=vr4)
        vo, y = _run(d, int(1.5 * FS))
        a, b = vo[-int(0.5 * FS):-int(0.25 * FS)], vo[-int(0.25 * FS):]
        pp_a, pp_b = np.ptp(a), np.ptp(b)
        if verbose:
            print(f"  BD VR4 {vr4 / 1e3:.0f}k: V_n3 {vo.min():.3f} ... {vo.max():.3f} V, "
                  f"last 0.25 s {b.min():.3f} ... {b.max():.3f} V, output peak "
                  f"{np.max(np.abs(y)) * 1e3:.2f} mV, iterations <= {d.voice.root.iter_max}")
        assert np.all(np.isfinite(vo))
        assert vo.min() > 0.0 and vo.max() < CD4011AGate.VDD, (vr4, vo.min(), vo.max())
        assert pp_b > 2.0, (vr4, pp_b)
        assert abs(pp_b / pp_a - 1) < 5e-3, (vr4, pp_a, pp_b)


def test_newton(verbose=False):
    for name in VOICES:
        for vr4 in (0.0, 150e3, 470e3):
            d = _drum(name, vr4=vr4)
            root = d.voice.root
            worst = np.zeros(3)
            for k in range(int(0.3 * FS)):
                d.tick(4.5 if k < 480 else 0.0)
                worst = np.maximum(worst, root.residuals())
            if verbose:
                print(f"  {name:14s} VR4 {vr4 / 1e3:3.0f}k: iterations <= {root.iter_max}, "
                      f"KCL {worst[0]:.1e} A, port V {worst[1]:.1e} V, law {worst[2]:.1e} A")
            assert root.iter_max <= 20, (name, vr4, root.iter_max)
            assert worst[0] < 1e-12 and worst[1] < 1e-9 and worst[2] < 1e-12, (name, vr4, worst)


if __name__ == "__main__":
    test_gate_law(verbose=True)
    test_dc_point()
    test_small_signal(verbose=True)
    test_large_signal(verbose=True)
    test_newton(verbose=True)
    print("OK")
