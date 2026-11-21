#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
Reference for the HB line (C13): the noise voices of Fig. 6
(tools/noise_voices_ref.py) and the Hi Bongo voice of Fig. 7
(tools/tonal_voice_ref.py) in one nodal analysis, trapezoidal rule, Newton on
all nodes at every step.

The HB trigger line is node h of the noise reference: M253 pin 6 (350 Ohm on
/ open off, pe78/m253.py), D6 from the snare's pin 5 node v1, R21 = the R14 of the HB
edge detector (user, 2026-11-20, counted once), and the rest of the HB voice:
C8 (h-t2), R15, R13, D2 (on the C6/C7 node), twin-T, CD4011A gate (law of
pe78/twint.py), R41 to GND. The HB voice starts at its DC operating point.

Usage (from the py78drums root):
    python tools/hb_line_ref.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import noise_voices_ref as nvr                  # noqa: E402
from noise_voices_ref import NoiseVoicesRef, lexp, D_IS, D_N, VT, V_ON, R_ON, R_OFF   # noqa: E402
from pe78.twint import CD4011AGate, TwinTDrum   # noqa: E402
from pe78 import drums                          # noqa: E402

HB_NODES = ["t2h", "ah", "n0h", "n1h", "n2h", "n3h"]


class HBLineRef(NoiseVoicesRef):
    def __init__(self, dt, vr4=None, vm=CD4011AGate.VM_DEFAULT, hb_detector=True, **kw):
        NoiseVoicesRef.__init__(self, dt, **kw)
        self.hb_detector = hb_detector
        if not hb_detector:             # plain noise reference (HB line = R21 only)
            return
        P = drums.HBONGO_PARAMS
        r14, r15, r13 = (float(x) for x in P["edge_r"][:3])
        assert r14 == 12e3              # R21 of the noise reference is the HB R14
        r11, r12, r16 = (float(x) for x in P["kick_r"])
        c5, c6, c7 = (float(x) for x in P["kick_c"])
        vr4 = float(P["kick_vr4"] if vr4 is None else vr4)
        for nm in HB_NODES:
            self.ix[nm] = self.N
            self.N += 1
        self.R += [("t2h", None, r15), ("t2h", "ah", r13), ("n2h", "n0h", r11), ("n0h", "n3h", r12),
                   ("n1h", None, r16), ("n2h", "n3h", TwinTDrum.R10 + vr4),
                   ("n3h", None, float(P["kick_load"]))]
        self.C += [("h", "t2h", float(P["edge_c"])), ("n0h", None, c5), ("n2h", "n1h", c6),
                   ("n1h", "n3h", c7)]
        self.hc = np.zeros(len(self.C))
        self.src["hb"] = "h"
        self.gate = CD4011AGate(vm)
        v = self.gate.dc_point(float(P["kick_load"]))
        self.x = np.zeros(self.N)
        for nm in ("n0h", "n2h", "n3h"):
            self.x[self.ix[nm]] = v
        self._G_cache = {}

    def _Glin(self, on):
        key = tuple(sorted(on.items()))
        G = self._G_cache.get(key)
        if G is None:
            saved, self._G_cache = self._G_cache, {}
            G = NoiseVoicesRef._Glin(self, on)
            self._G_cache = saved
            self._G_cache[key] = G
        return G

    def step(self, on, i_noise):
        if not self.hb_detector:
            on = {k: v for k, v in on.items() if k != "hb"}
            return NoiseVoicesRef.step(self, on, i_noise)
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
            ieq = self.il[j] + dt / (2 * l) * vd(p, q, xp)
            I[X[p]] -= ieq
            if q is not None:
                I[X[q]] += ieq
        x = xp.copy()
        nvt = D_N * VT
        for _ in range(300):
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
            # D6, D7, D9, D2 of HB; D8 GND -> a
            for p, q in (("v1", "h"), ("ve5", "vd7"), ("a", "k"), ("ah", "n1h")):
                e, de = lexp((x[X[p]] - x[X[q]]) / nvt)
                branch(p, q, D_IS * (e - 1), D_IS * de / nvt)
            e, de = lexp(-x[X["a"]] / nvt)
            F[X["a"]] -= D_IS * (e - 1)
            J[X["a"], X["a"]] += D_IS * de / nvt
            e, de = lexp((x[X["vb5"]] - x[X["ve5"]]) / (nvr.T5_NF * VT))
            ib, dib = nvr.T5_IS / nvr.T5_BF * (e - 1), nvr.T5_IS / nvr.T5_BF * de / (nvr.T5_NF * VT)
            branch("vb5", "ve5", ib, dib)
            ic, dic = nvr.T5_IS * (e - 1), nvr.T5_IS * de / (nvr.T5_NF * VT)
            F[X["ve5"]] -= ic
            J[X["ve5"], X["ve5"]] += dic
            J[X["ve5"], X["vb5"]] -= dic
            for b, c in (("b3", "c3"), ("env", "c4")):
                e, de = lexp(x[X[b]] / nvr.VTB)
                ib, dib = nvr.IS_B * (e - 1), nvr.IS_B * de / nvr.VTB
                F[X[b]] += ib
                J[X[b], X[b]] += dib
                F[X[c]] += nvr.HFE * ib
                J[X[c], X[b]] += nvr.HFE * dib
            # CD4011A gate of HB: current out of n3h
            i_g, g_in, g_out = self.gate.current(x[X["n2h"]], x[X["n3h"]])
            F[X["n3h"]] += i_g
            J[X["n3h"], X["n2h"]] += g_in
            J[X["n3h"], X["n3h"]] += g_out
            dx = np.clip(np.linalg.solve(J, -F), -0.2, 0.2)
            x = x + dx
            if np.max(np.abs(dx)) < 1e-11:
                break
        else:
            raise RuntimeError("HB line reference: Newton did not converge")
        for j, (p, q, c) in enumerate(self.C):
            self.hc[j] = 2 * c / dt * (vd(p, q, x) - vd(p, q, xp)) - self.hc[j]
        for j, (p, q, l) in enumerate(self.L):
            self.il[j] = self.il[j] + dt / (2 * l) * (vd(p, q, x) + vd(p, q, xp))
        self.x = x
        return x


def simulate(gates, t_end, dt, **kw):
    """gates: dict voice ("sd", "hb", "lc", "sc", "mr") -> list of (start,
    length). No noise. Returns t and node dict."""
    r = HBLineRef(dt, **kw)
    n = int(round(t_end / dt))
    t = np.arange(1, n + 1) * dt
    out = np.zeros((n, r.N))
    for k in range(n):
        on = {v: any(a <= t[k] < a + g for a, g in gates.get(v, []))
              for v in ("sd", "lc", "sc", "mr", "hb")}
        out[k] = r.step(on, 0.0)
    return t, {nm: out[:, i] for nm, i in r.ix.items()}


if __name__ == "__main__":
    t, v = simulate({"sd": [(0.01, 0.01)]}, 0.1, 1 / 192000)
    print(f"SD hit: HB line peak {v['h'].max():.3f} V; HB V_n3 {v['n3h'].min():.3f} ... "
          f"{v['n3h'].max():.3f} V; C27 peak {v['c27'].max():.4f} V")
