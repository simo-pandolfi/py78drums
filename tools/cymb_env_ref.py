#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
Reference for the cymbal envelope node ENV (C7): nodal analysis, small step.

Independent of pe78/cymb.py (no pywdf, no wave variables): modified nodal
analysis, trapezoidal companion models for C23, C24 and C26, Newton on all
nodes at every step.

Circuit (PE 1978 Fig. 6, bottom section), maracas branch idle (D9 off):
    LC: M253 pin 21 (V_ON = 4.5 V, R_ON = 350 Ohm; R_OFF = 10 MOhm, 0 V)
        -> n_lc (C23 to GND) -> R47 -> ENV
    SC: M253 pin 20 (same source) -> R42 -> n_sc (C24 to GND) -> R44 -> ENV
    MR: ENV -> R46 -> n_mr (C26 to GND); D9 off, so nothing else on n_mr
    ENV: R52 to GND, TR4 base Ib = Is_b*(exp(V/Vt) - 1)
Is_b and Vt as pe78/cymb.py (calibrated against LTspice, V_ENV ~ 620 mV).
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
R46, C26 = 470e3, 0.1e-6
R52 = 1e6
IS_B, VT = 7.035e-17, 26e-3

N_LC, ENV, N_SC, N_SRC_SC, N_MR = range(5)
NN = 5


def simulate(gates_lc, gates_sc, t_end, dt=5e-6):
    """gates_*: lists of (start_s, length_s). Returns t and a dict of node voltages."""
    n = int(round(t_end / dt))
    t = np.arange(n + 1) * dt
    out = np.zeros((n + 1, NN))
    x = np.zeros(NN)
    caps = [(N_LC, C23), (N_SC, C24), (N_MR, C26)]
    hist = np.zeros(len(caps))
    for k in range(1, n + 1):
        on_lc = any(s <= t[k] < s + g for s, g in gates_lc)
        on_sc = any(s <= t[k] < s + g for s, g in gates_sc)
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
        cond(ENV, N_MR, 1.0 / R46)
        cond(ENV, -1, 1.0 / R52)
        xp = x.copy()
        for j, (node, C) in enumerate(caps):
            gc = 2.0 * C / dt
            cond(node, -1, gc)
            I[node] += gc * xp[node] + hist[j]
        for _ in range(100):
            e = np.exp(min(x[ENV] / VT, 80.0))
            F = G @ x - I
            F[ENV] += IS_B * (e - 1.0)
            J = G.copy()
            J[ENV, ENV] += IS_B * e / VT
            dx = np.clip(np.linalg.solve(J, -F), -0.2, 0.2)
            x = x + dx
            if np.max(np.abs(dx)) < 1e-12:
                break
        else:
            raise RuntimeError(f"Newton did not converge at t = {t[k]:.6f} s")
        for j, (node, C) in enumerate(caps):
            hist[j] = 2.0 * C / dt * (x[node] - xp[node]) - hist[j]
        out[k] = x
    names = ("v_lc", "env", "v_sc", "v_src_sc", "v_mr")
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
