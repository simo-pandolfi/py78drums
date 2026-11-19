#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
Reference for the snare trigger front end (C6): nodal analysis, small step.

Independent of pe78/snare.py (no pywdf, no wave variables): modified nodal
analysis with the emitter of TR5 as an explicit node, trapezoidal companion
models for C22 and C27, Newton on all nodes at every step, and the TR3 base
solved together with the rest (no one-sample decoupling).

Circuit (PE 1978 Fig. 6, schematic page 24; trigger line from the user):
    M253 pin 5 (V_ON = 4.5 V, R_ON = 350 Ohm; R_OFF = 10 MOhm, 0 V)
      -> v1 -> C22 -> vb (TR5 base, R48 to GND)
      v1 -> D6 -> h (HB trigger line, R21 = 12 kOhm to GND; C12 and the rest
                     of the HB edge detector are neglected)
    TR5 (collector at the +5 V rail): ve -> D7 -> vd -> R49 -> vc (C27 to GND)
    vc -> R50 -> v3 (TR3 base, R51 to GND, Ib3 = Is_b*(exp(v3/Vt3) - 1))
Models: 1N4148 as LTspice standard.dio (Is = 2.52 nA, N = 1.752, Rs
neglected); TR5 Ebers-Moll, forward active only, with IS, NF, BF of the
LTspice BC547B card (the BC junction does not conduct: base <= 4.7 V).
gmin = 1e-12 S on every node, as in SPICE.

Usage (from the py78drums root):
    python tools/snare_frontend_ref.py
prints V_C27 peak, base duration and retrigger for a few gates; tests import
simulate() from here.
"""

import numpy as np

VT = 25.852e-3                       # kT/q at 27 C (LTspice default)
D_IS, D_N = 2.52e-9, 1.752           # 1N4148, standard.dio
T_IS, T_NF, T_BF = 2.39e-14, 1.008, 294.3   # BC547B card (TR5)
V_ON, R_ON, R_OFF = 4.5, 350.0, 10e6
C22, R48, R21, R49, C27, R50, R51 = 0.1e-6, 22e3, 12e3, 2200.0, 0.05e-6, 1e6, 1e6
IS_B3, VT3 = 7.035e-17, 26e-3        # TR3 base, as pe78/snare.py
GMIN = 1e-12

# node indices
V1, VB, VE, VD, VC, H, V3 = range(7)
NN = 7


def _lexp(x, xm=80.0):
    if x <= xm:
        e = np.exp(x)
        return e, e
    e = np.exp(xm)
    return e * (1.0 + x - xm), e


def simulate(gates, t_end, dt=2e-6):
    """gates: list of (start_s, length_s). Returns t and a dict of node voltages."""
    n = int(round(t_end / dt))
    t = np.arange(n + 1) * dt
    out = np.zeros((n + 1, NN))
    x = np.zeros(NN)
    i22 = 0.0                       # trapezoidal history currents
    i27 = 0.0
    g22, g27 = 2.0 * C22 / dt, 2.0 * C27 / dt
    for k in range(1, n + 1):
        on = any(s <= t[k] < s + g for s, g in gates)
        vs, rs = (V_ON, R_ON) if on else (0.0, R_OFF)
        xp = x.copy()
        # companion: i_C = g*(v - v_prev) - i_prev  (trapezoidal)
        ieq22 = g22 * (xp[V1] - xp[VB]) + i22
        ieq27 = g27 * xp[VC] + i27
        for _ in range(100):
            F = np.zeros(NN)
            J = np.zeros((NN, NN))

            def cond(p, q, g):
                vq = x[q] if q >= 0 else 0.0
                F[p] += g * (x[p] - vq)
                J[p, p] += g
                if q >= 0:
                    F[q] -= g * (x[p] - vq)
                    J[q, q] += g
                    J[p, q] -= g
                    J[q, p] -= g

            def branch(p, q, i, di):        # current i(v_p - v_q) from p to q
                F[p] += i
                J[p, p] += di
                J[p, q] -= di
                F[q] -= i
                J[q, q] += di
                J[q, p] -= di

            for m in range(NN):
                cond(m, -1, GMIN)
            cond(V1, -1, 1.0 / rs)
            F[V1] -= vs / rs
            cond(V1, VB, g22)
            F[V1] -= ieq22
            F[VB] += ieq22
            cond(VB, -1, 1.0 / R48)
            e, de = _lexp((x[V1] - x[H]) / (D_N * VT))           # D6
            branch(V1, H, D_IS * (e - 1.0), D_IS * de / (D_N * VT))
            cond(H, -1, 1.0 / R21)
            e, de = _lexp((x[VB] - x[VE]) / (T_NF * VT))         # TR5
            ib, dib = T_IS / T_BF * (e - 1.0), T_IS / T_BF * de / (T_NF * VT)
            branch(VB, VE, ib, dib)
            ic, dic = T_IS * (e - 1.0), T_IS * de / (T_NF * VT)  # from the rail into ve
            F[VE] -= ic
            J[VE, VE] += dic
            J[VE, VB] -= dic
            e, de = _lexp((x[VE] - x[VD]) / (D_N * VT))          # D7
            branch(VE, VD, D_IS * (e - 1.0), D_IS * de / (D_N * VT))
            cond(VD, VC, 1.0 / R49)
            cond(VC, -1, g27)
            F[VC] -= ieq27
            cond(VC, V3, 1.0 / R50)
            cond(V3, -1, 1.0 / R51)
            e, de = _lexp(x[V3] / VT3)                           # TR3 base
            F[V3] += IS_B3 * (e - 1.0)
            J[V3, V3] += IS_B3 * de / VT3
            dx = np.clip(np.linalg.solve(J, -F), -0.2, 0.2)
            x = x + dx
            if np.max(np.abs(dx)) < 1e-11:
                break
        else:
            raise RuntimeError(f"Newton did not converge at t = {t[k]:.6f} s")
        i22 = g22 * ((x[V1] - x[VB]) - (xp[V1] - xp[VB])) - i22
        i27 = g27 * (x[VC] - xp[VC]) - i27
        out[k] = x
    names = ("v1", "vb", "ve", "vd", "vc27", "h", "vbase3")
    return t, {nm: out[:, i] for i, nm in enumerate(names)}


def base_end_ms(t, v3, t0, thr=0.5):
    idx = np.nonzero(v3 > thr)[0]
    return 1e3 * (t[idx[-1]] - t0) if len(idx) else float("nan")


if __name__ == "__main__":
    for g_ms in (1, 10, 100):
        t, v = simulate([(0.01, g_ms * 1e-3)], 0.25, dt=5e-6)
        print(f"gate {g_ms:3d} ms: V_C27 peak {v['vc27'].max():.4f} V, "
              f"V_base TR3 > 0.5 V until {base_end_ms(t, v['vbase3'], 0.01):.1f} ms")
    t, v = simulate([(0.01, 0.01), (0.0725, 0.01)], 0.2, dt=5e-6)
    m = t >= 0.0725
    print(f"retrigger after 62.5 ms: V_C27 peak {v['vc27'][~m].max():.4f} -> {v['vc27'][m].max():.4f} V")
