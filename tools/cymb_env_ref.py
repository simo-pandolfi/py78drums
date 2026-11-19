#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
Reference for the cymbal envelope section (C7, C8): nodal analysis, small step.

Independent of pe78/cymb.py (no pywdf, no wave variables): modified nodal
analysis, trapezoidal companion models for C23, C24, C25 and C26, Newton on
all nodes at every step (ENV, TR4 base, D8 and D9 solved together).

Circuit (PE 1978 Fig. 6, bottom section):
    LC: M253 pin 21 (V_ON = 4.5 V, R_ON = 350 Ohm; R_OFF = 10 MOhm, 0 V)
        -> n_lc (C23 to GND) -> R47 -> ENV
    SC: M253 pin 20 (same source) -> R42 -> n_sc (C24 to GND) -> R44 -> ENV
    MR: M253 pin 19 (same source) -> S (R43 to GND) -> C25 -> A;
        D8 from GND (anode) to A; D9 from A to K; K -> R45 -> n_mr
        (C26 to GND) -> R46 -> ENV
    ENV: R52 to GND, TR4 base Ib = Is_b*(exp(V/Vt) - 1)
Is_b and Vt as pe78/cymb.py (calibrated against LTspice, V_ENV ~ 620 mV);
1N4148 as LTspice standard.dio (Is 2.52 nA, N 1.752, Rs neglected).
With the .asc source (5.2 V through a diode) this circuit matches the
user's LTspice traces of C23 and C26 (C7, C8; not part of this tool).
C29 and the noise node are not included (as in cymb.py after C7).

Usage (from the py78drums root):
    python tools/cymb_env_ref.py
prints V_ENV at the end of an LC gate and the times below 0.55 / 0.5 V;
tests import simulate() from here.
"""

import numpy as np

V_ON, R_ON, R_OFF = 4.5, 350.0, 10e6
C23, R47 = 0.33e-6, 1e6
R42, C24, R44 = 10e3, 68e-9, 1e6
R43, C25, R45 = 22e3, 0.22e-6, 100e3
R46, C26 = 470e3, 0.1e-6
VT_D, D_IS, D_N = 25.852e-3, 2.52e-9, 1.752
R52 = 1e6
IS_B, VT = 7.035e-17, 26e-3

N_LC, ENV, N_SC, N_SRC_SC, N_MR, S_MR, A_MR, K_MR = range(8)
NN = 8


def _lexp(x, xm=80.0):
    if x <= xm:
        e = np.exp(x)
        return e, e
    e = np.exp(xm)
    return e * (1.0 + x - xm), e


def simulate(gates_lc, gates_sc, t_end, dt=5e-6, gates_mr=()):
    """gates_*: lists of (start_s, length_s). Returns t and a dict of node voltages."""
    n = int(round(t_end / dt))
    t = np.arange(n + 1) * dt
    out = np.zeros((n + 1, NN))
    x = np.zeros(NN)
    caps = [(N_LC, -1, C23), (N_SC, -1, C24), (N_MR, -1, C26), (S_MR, A_MR, C25)]
    nvt = D_N * VT_D
    hist = np.zeros(len(caps))
    for k in range(1, n + 1):
        on_lc = any(s <= t[k] < s + g for s, g in gates_lc)
        on_sc = any(s <= t[k] < s + g for s, g in gates_sc)
        on_mr = any(s <= t[k] < s + g for s, g in gates_mr)
        G = np.zeros((NN, NN))
        I = np.zeros(NN)

        def cond(p, q, g):
            G[p, p] += g
            if q >= 0:
                G[q, q] += g
                G[p, q] -= g
                G[q, p] -= g

        r_lc = R_ON if on_lc else R_OFF
        cond(N_LC, -1, 1.0 / r_lc)
        I[N_LC] += (V_ON if on_lc else 0.0) / r_lc
        cond(N_LC, ENV, 1.0 / R47)
        r_sc = R_ON if on_sc else R_OFF
        cond(N_SRC_SC, -1, 1.0 / r_sc)
        I[N_SRC_SC] += (V_ON if on_sc else 0.0) / r_sc
        cond(N_SRC_SC, N_SC, 1.0 / R42)
        cond(N_SC, ENV, 1.0 / R44)
        r_mr = R_ON if on_mr else R_OFF
        cond(S_MR, -1, 1.0 / r_mr)
        I[S_MR] += (V_ON if on_mr else 0.0) / r_mr
        cond(S_MR, -1, 1.0 / R43)
        cond(K_MR, N_MR, 1.0 / R45)
        cond(ENV, N_MR, 1.0 / R46)
        cond(ENV, -1, 1.0 / R52)
        xp = x.copy()
        for j, (p, q, C) in enumerate(caps):
            gc = 2.0 * C / dt
            cond(p, q, gc)
            ieq = gc * (xp[p] - (xp[q] if q >= 0 else 0.0)) + hist[j]
            I[p] += ieq
            if q >= 0:
                I[q] -= ieq
        for _ in range(200):
            e = np.exp(min(x[ENV] / VT, 80.0))
            F = G @ x - I
            F[ENV] += IS_B * (e - 1.0)
            J = G.copy()
            J[ENV, ENV] += IS_B * e / VT
            e8, de8 = _lexp(-x[A_MR] / nvt)                  # D8: GND -> A
            F[A_MR] -= D_IS * (e8 - 1.0)
            J[A_MR, A_MR] += D_IS * de8 / nvt
            e9, de9 = _lexp((x[A_MR] - x[K_MR]) / nvt)       # D9: A -> K
            i9, g9 = D_IS * (e9 - 1.0), D_IS * de9 / nvt
            F[A_MR] += i9
            F[K_MR] -= i9
            J[A_MR, A_MR] += g9
            J[K_MR, K_MR] += g9
            J[A_MR, K_MR] -= g9
            J[K_MR, A_MR] -= g9
            dx = np.clip(np.linalg.solve(J, -F), -0.2, 0.2)
            x = x + dx
            if np.max(np.abs(dx)) < 1e-12:
                break
        else:
            raise RuntimeError(f"Newton did not converge at t = {t[k]:.6f} s")
        for j, (p, q, C) in enumerate(caps):
            dv = (x[p] - (x[q] if q >= 0 else 0.0)) - (xp[p] - (xp[q] if q >= 0 else 0.0))
            hist[j] = 2.0 * C / dt * dv - hist[j]
        out[k] = x
    names = ("v_lc", "env", "v_sc", "v_src_sc", "v_mr", "v_s_mr", "v_a", "v_k")
    return t, {nm: out[:, i] for i, nm in enumerate(names)}


def last_above(t, v, thr):
    idx = np.nonzero(v > thr)[0]
    return t[idx[-1]] if len(idx) else float("nan")


if __name__ == "__main__":
    t, v = simulate([(0.15, 0.015)], [], 1.1)
    k = np.searchsorted(t, 0.165) - 1
    print(f"LC gate 15 ms at 0.15 s: V_ENV at the end of the gate {v['env'][k]*1e3:.1f} mV; "
          f"V_ENV > 0.55 V until {last_above(t, v['env'], 0.55):.4f} s, "
          f"> 0.5 V until {last_above(t, v['env'], 0.5):.4f} s")
    for g in (0.015, 0.1):
        t, v = simulate([], [], 0.3, gates_mr=[(0.05, g)])
        k = np.searchsorted(t, 0.05 + g) - 1
        print(f"MR gate {g*1e3:.0f} ms at 0.05 s: V_C26 peak {v['v_mr'].max():.3f} V, "
              f"at the end of the gate {v['v_mr'][k]:.3f} V, node A min {v['v_a'].min():.3f} V")
