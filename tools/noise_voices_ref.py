#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
Reference for the noise voices (pe78/noise_voices.py, C15): the whole Fig. 6
section in one modified nodal analysis, trapezoidal rule, small step, Newton
on all nodes at every step (no WDF, no one-sample couplings).

Snare front end (M253 pin 5: 4.85 V / 350 Ohm, open when off, as
pe78/m253.py; C22, R48, D6, R21, TR5, D7, R49, C27, R50), TR3 base (R51, Shockley), noise node N (VR8U to
+12 V, B1 current, C28 to the TR3 base, C29 to the TR4 base), ENV (R52, TR4
base Shockley, LC: C23 + R47, SC: R42 + C24 + R44, maracas: R43, C25, D8,
D9, R45, C26, R46), collectors ((L1 + Rser) || R53 || C30, (L2 + Rser) || R54,
Rser = 61.662 Ohm, Ic = hFE*Ib),
VR9U/VR9L with the common wiper W, R55, C31, RLoad (output O).
+12 V rails are AC ground: collector nodes and N are deviations from +12 V.
Models as pe78/noise_voices.py: 1N4148 LTspice standard.dio, TR5 BC547B card
(forward active), TR3/TR4 base Is_b = 7.035e-17 A, Vt = 26 mV, hFE = 200.

Noise: B1 current per model sample (noise_amp(fs)*sample), first-order hold
between samples (the trapezoidal WDF sees the samples the same way). Compare
noise-driven signals below a few kHz (bilinear warping near fs/2).

Validated pieces: C23 and C26 against the user's LTspice traces (C7, C8),
TR3 base delay by C28 and ENV crosstalk of a snare hit (C15).

Usage (from the py78drums root):
    python tools/noise_voices_ref.py
"""
import numpy as np

VT = 25.852e-3
D_IS, D_N = 2.52e-9, 1.752
T5_IS, T5_NF, T5_BF = 2.39e-14, 1.008, 294.3
IS_B, VTB, HFE = 7.035e-17, 26e-3, 200.0
V_ON, R_ON, R_OFF = 4.85, 350.0, 1e12     # pe78/m253.py (D11)
L_RSER = 61.662                           # series resistance of L1, L2 (.asc; D5)


def lexp(x, xm=80.0):
    if x <= xm:
        e = np.exp(x)
        return e, e
    e = np.exp(xm)
    return e * (1 + x - xm), e


class NoiseVoicesRef:
    def __init__(self, dt, vr9_pos=25e3 / 220e3, vr9=220e3):
        vr9u, vr9l = max(vr9_pos * vr9, 1.0), max((1 - vr9_pos) * vr9, 1.0)
        self.dt = dt
        names = ["v1", "vb5", "ve5", "vd7", "c27", "h", "b3", "n", "env", "lc", "sc", "ssc",
                 "s", "a", "k", "mr", "c3", "c4", "w", "x31", "o", "l3", "l4"]
        self.ix = {nm: i for i, nm in enumerate(names)}
        self.N = len(names)
        self.R = [("vb5", None, 22e3), ("h", None, 12e3), ("vd7", "c27", 2200.0),
                  ("c27", "b3", 1e6), ("b3", None, 1e6),
                  ("n", None, 90e3),
                  ("lc", "env", 1e6), ("ssc", "sc", 10e3), ("sc", "env", 1e6),
                  ("s", None, 22e3), ("k", "mr", 100e3), ("mr", "env", 470e3), ("env", None, 1e6),
                  ("c3", None, 4700.0), ("c4", None, 4700.0),
                  ("o", None, 10e3)]
        self.C = [("v1", "vb5", 0.1e-6), ("c27", None, 0.05e-6), ("b3", "n", 4.7e-9),
                  ("lc", None, 0.33e-6), ("sc", None, 68e-9), ("s", "a", 0.22e-6), ("mr", None, 0.1e-6),
                  ("c3", None, 0.02e-6)]
        self.L = [("c3", "l3", 0.1), ("c4", "l4", 0.1)]
        self.R += [("l3", None, L_RSER), ("l4", None, L_RSER)]
        self.C.append(("n", "env", 4.7e-9))
        self.R += [("c3", "w", vr9u), ("c4", "w", vr9l), ("w", "x31", 22e3)]
        self.C.append(("x31", "o", 0.1e-6))
        self.src = {"sd": "v1", "lc": "lc", "sc": "ssc", "mr": "s"}
        self.x = np.zeros(self.N)
        self.hc = np.zeros(len(self.C))
        self.il = np.zeros(len(self.L))
        self._G_cache = {}

    def _Glin(self, on):
        key = tuple(on[s] for s in ("sd", "lc", "sc", "mr"))
        G = self._G_cache.get(key)
        if G is not None:
            return G
        N, X, dt = self.N, self.ix, self.dt
        G = np.zeros((N, N))

        def cond(p, q, gv):
            p = X[p]
            G[p, p] += gv
            if q is not None:
                q = X[q]
                G[q, q] += gv
                G[p, q] -= gv
                G[q, p] -= gv
        for p, q, r in self.R:
            cond(p, q, 1.0 / r)
        for p, q, c in self.C:
            cond(p, q, 2 * c / dt)
        for p, q, l in self.L:
            cond(p, q, dt / (2 * l))
        for s, nd in self.src.items():
            cond(nd, None, 1.0 / (R_ON if on[s] else R_OFF))
        for i in range(N):
            G[i, i] += 1e-12
        self._G_cache[key] = G
        return G

    def step(self, on, i_noise):
        X, dt = self.ix, self.dt
        G = self._Glin(on)
        xp = self.x
        I = np.zeros(self.N)
        for s, nd in self.src.items():
            if on[s]:
                I[X[nd]] += V_ON / R_ON
        I[X["n"]] += i_noise

        def vd(p, q, x):
            return x[X[p]] - (x[X[q]] if q is not None else 0.0)
        for j, (p, q, c) in enumerate(self.C):
            ieq = 2 * c / dt * vd(p, q, xp) + self.hc[j]
            I[X[p]] += ieq
            if q is not None:
                I[X[q]] -= ieq
        for j, (p, q, l) in enumerate(self.L):
            ieq = self.il[j] + dt / (2 * l) * vd(p, q, xp)       # flows p -> q
            I[X[p]] -= ieq
            if q is not None:
                I[X[q]] += ieq
        x = xp.copy()
        nvt = D_N * VT
        for _ in range(200):
            F = G @ x - I
            J = G.copy()

            def branch(p, q, i, di):
                p = X[p]
                F[p] += i
                J[p, p] += di
                if q is not None:
                    q = X[q]
                    F[q] -= i
                    J[q, q] += di
                    J[p, q] -= di
                    J[q, p] -= di
            # D6 v1 -> h, D7 ve5 -> vd7, D9 a -> k, D8 GND -> a
            for p, q in (("v1", "h"), ("ve5", "vd7"), ("a", "k")):
                e, de = lexp((x[X[p]] - x[X[q]]) / nvt)
                branch(p, q, D_IS * (e - 1), D_IS * de / nvt)
            e, de = lexp(-x[X["a"]] / nvt)
            F[X["a"]] -= D_IS * (e - 1)
            J[X["a"], X["a"]] += D_IS * de / nvt
            # TR5: base vb5 -> ve5, collector current from rail into ve5
            e, de = lexp((x[X["vb5"]] - x[X["ve5"]]) / (T5_NF * VT))
            ib, dib = T5_IS / T5_BF * (e - 1), T5_IS / T5_BF * de / (T5_NF * VT)
            branch("vb5", "ve5", ib, dib)
            ic, dic = T5_IS * (e - 1), T5_IS * de / (T5_NF * VT)
            F[X["ve5"]] -= ic
            J[X["ve5"], X["ve5"]] += dic
            J[X["ve5"], X["vb5"]] -= dic
            # TR3, TR4: base Shockley; Ic = hFE*Ib out of the collector node
            for b, c in (("b3", "c3"), ("env", "c4")):
                e, de = lexp(x[X[b]] / VTB)
                ib, dib = IS_B * (e - 1), IS_B * de / VTB
                F[X[b]] += ib
                J[X[b], X[b]] += dib
                F[X[c]] += HFE * ib
                J[X[c], X[b]] += HFE * dib
            dx = np.clip(np.linalg.solve(J, -F), -0.2, 0.2)
            x = x + dx
            if np.max(np.abs(dx)) < 1e-11:
                break
        else:
            raise RuntimeError("ref Newton")
        for j, (p, q, c) in enumerate(self.C):
            self.hc[j] = 2 * c / dt * (vd(p, q, x) - vd(p, q, xp)) - self.hc[j]
        for j, (p, q, l) in enumerate(self.L):
            self.il[j] = self.il[j] + dt / (2 * l) * (vd(p, q, x) + vd(p, q, xp))
        self.x = x
        return x


def simulate(gates, t_end, dt=5e-6, noise_i=None, fs_noise=None, **kw):
    """gates: dict voice -> list of (start, length). noise_i: B1 current per model
    sample (array at fs_noise), held (ZOH). Returns t and node dict."""
    r = NoiseVoicesRef(dt, **kw)
    n = int(round(t_end / dt))
    t = np.arange(1, n + 1) * dt
    out = np.zeros((n, r.N))
    for k in range(n):
        on = {v: any(a <= t[k] < a + g for a, g in gates.get(v, [])) for v in ("sd", "lc", "sc", "mr")}
        if noise_i is not None:
            u = t[k] * fs_noise                       # first-order hold between samples
            j = min(int(u), len(noise_i) - 2)
            fr = u - j
            i_n = (1 - fr) * noise_i[j] + fr * noise_i[j + 1]
        else:
            i_n = 0.0
        out[k] = r.step(on, i_n)
    return t, {nm: out[:, i] for nm, i in r.ix.items()}


def last_above(t, v, thr):
    idx = np.nonzero(v > thr)[0]
    return t[idx[-1]] if len(idx) else float("nan")


if __name__ == "__main__":
    t, v = simulate({"sd": [(0.01, 0.01)]}, 0.12)
    k = np.argmax(v["b3"] > 0.5)
    print(f"SD hit: V_C27 peak {v['c27'].max():.3f} V, TR3 base > 0.5 V from {1e3*(t[k]-0.01):.2f} ms; "
          f"ENV {1e3*v['env'].max():.0f} / {1e3*v['env'].min():.0f} mV; output peak {1e3*np.abs(v['o']).max():.2f} mV")
