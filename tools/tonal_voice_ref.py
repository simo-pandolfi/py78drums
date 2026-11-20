#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
Reference for one tonal voice (pe78/twint.py, TwinTDrum, C12-C14): the whole
Fig. 7 stage in one nodal analysis, trapezoidal rule, Newton on all nodes at
every step (no WDF, no R-type junction).

Nodes: n0 (R11-R12, C5), n1 (C6-C7, R16, cathode of D2), n2 (gate input),
n3 (gate output, R41 to GND), t1 (M253 output, R14), t2 (C8-R15-R13),
a (anode of D2). M253 output 350 Ohm on / 10 MOhm off, D2 1N4148 LTspice
standard.dio (Is 2.52 nA, N 1.752, Vt 25.852 mV), CD4011A gate law of
pe78/twint.py (CD4011AGate, checked on its own in tests/test_cd4011.py).
Start at the DC operating point of the model.

With dt = 1/fs and the gate switching on the same samples the result must
equal the WDF model up to rounding (same circuit, same discretisation); with
a smaller dt it shows the discretisation error of the model.

Usage (from the py78drums root):
    python tools/tonal_voice_ref.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pe78.twint import CD4011AGate, TwinTDrum   # noqa: E402  (constants and gate law)

N0, N1, N2, N3, T1, T2, A = range(7)


class TonalVoiceRef:
    def __init__(self, params, dt, vr4=None, vm=CD4011AGate.VM_DEFAULT):
        r14, r15, r13 = (float(x) for x in params["edge_r"][:3])
        r11, r12, r16 = (float(x) for x in params["kick_r"])
        c5, c6, c7 = (float(x) for x in params["kick_c"])
        vr4 = float(params["kick_vr4"] if vr4 is None else vr4)
        r41 = float(params["kick_load"])
        self.dt = dt
        self.gate = CD4011AGate(vm)
        self.R = [(N2, N0, r11), (N0, N3, r12), (N1, -1, r16), (N2, N3, TwinTDrum.R10 + vr4),
                  (N3, -1, r41), (T1, -1, r14), (T2, -1, r15), (T2, A, r13)]
        self.C = [(N0, -1, c5), (N2, N1, c6), (N1, N3, c7), (T1, T2, float(params["edge_c"]))]
        G = np.zeros((7, 7))
        for p, q, r in self.R:
            self._stamp(G, p, q, 1.0 / r)
        for p, q, c in self.C:
            self._stamp(G, p, q, 2.0 * c / dt)
        self.G = G
        self.nvt = TwinTDrum.D_N * TwinTDrum.VT_27C
        v = self.gate.dc_point(r41)
        self.v = np.array([v, 0.0, v, v, 0.0, 0.0, 0.0])
        self.uc = np.array([self.v[p] - (self.v[q] if q >= 0 else 0.0) for p, q, _ in self.C])
        self.ic = np.zeros(len(self.C))
        self.iter_max = 0

    @staticmethod
    def _stamp(M, p, q, g):
        for a, b, s in ((p, p, 1), (q, q, 1), (p, q, -1), (q, p, -1)):
            if a >= 0 and b >= 0:
                M[a, b] += s * g

    def step(self, v_trigger):
        on = v_trigger > 0.5
        g_s = 1.0 / (TwinTDrum.R_ON if on else TwinTDrum.R_OFF)
        G = self.G.copy()
        G[T1, T1] += g_s
        rhs = np.zeros(7)
        rhs[T1] += (v_trigger if on else 0.0) * g_s
        hist = []
        for (p, q, c), u, i in zip(self.C, self.uc, self.ic):
            h = 2.0 * c / self.dt * u + i                 # i_c = gc*u - h
            hist.append(h)
            rhs[p] += h
            if q >= 0:
                rhs[q] -= h
        x = self.v.copy()
        for it in range(1, 201):
            F = G @ x - rhs
            J = G.copy()
            vd = x[A] - x[N1]
            e = np.exp(min(vd / self.nvt, 80.0))
            i_d, g_d = TwinTDrum.D_IS * (e - 1.0), TwinTDrum.D_IS * e / self.nvt
            F[A] += i_d
            F[N1] -= i_d
            J[A, A] += g_d
            J[A, N1] -= g_d
            J[N1, A] -= g_d
            J[N1, N1] += g_d
            i_g, g_in, g_out = self.gate.current(x[N2], x[N3])
            F[N3] += i_g
            J[N3, N2] += g_in
            J[N3, N3] += g_out
            dx = np.linalg.solve(J, -F)
            m = np.max(np.abs(dx))
            if m > 0.2:
                dx *= 0.2 / m
            x += dx
            if m < 1e-10:            # rounding floor about 1e-12 V
                break
        else:
            raise RuntimeError("tonal voice reference: Newton did not converge")
        self.iter_max = max(self.iter_max, it)
        self.v = x
        for k, ((p, q, c), h) in enumerate(zip(self.C, hist)):
            u = x[p] - (x[q] if q >= 0 else 0.0)
            self.ic[k] = 2.0 * c / self.dt * u - h
            self.uc[k] = u
        self.i_d2 = TwinTDrum.D_IS * (np.exp((x[A] - x[N1]) / self.nvt) - 1.0)
        return x[N3]


def run(params, fs, n, gate, sub=1, vr4=None, vm=CD4011AGate.VM_DEFAULT):
    """gate(k) -> M253 voltage at model sample k; sub steps per sample.
    Returns V_n3 and i_D2 at the model samples."""
    ref = TonalVoiceRef(params, 1.0 / (fs * sub), vr4=vr4, vm=vm)
    vo, i2 = np.empty(n), np.empty(n)
    for k in range(n):
        g = gate(k)
        for _ in range(sub):
            vo[k] = ref.step(g)
        i2[k] = ref.i_d2
    return vo, i2


if __name__ == "__main__":
    from pe78 import drums
    fs = 48000
    vo, i2 = run(drums.BDO_PARAMS, fs, fs // 2, lambda k: 4.5 if k < 480 else 0.0)
    print(f"BD, VR4 150 kOhm, 10 ms gate: V_n3 {vo.min():.4f} ... {vo.max():.4f} V, "
          f"D2 peak {1e3 * i2.max():.4f} mA")
