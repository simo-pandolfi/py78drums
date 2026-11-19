#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
PE78 noise voices — snare drum, long and short cymbal, maracas
==============================================================
Source: Practical Electronics, Jan 1978, Fig. 6 (page 24); LTspice reference
ltspice/Practical-Electronics-1978-01_Pagina_24_Fig6.asc.

The three voices share the noise of TR2 (B1, pe78/noise.py) through node N,
and the two collectors share the balance pot VR9 and the output R55/C31:
they are one circuit and are modelled as one object (C15).

    NoiseVoices(fs, vr9_pos=None)
    out = nv.tick(v_sd, v_lc, v_sc, v_mr, noise=None)

Inputs are the M253 trigger outputs (0 V idle, 4.5 V active); noise is a
unit-variance sample (None: an internal N(0,1) sample). The output is the
voltage on RLoad (10 kOhm, the pre-amp input VR10; the tonal voices on the
same bus come with C16).

Blocks per sample (WDF; nonlinear roots solved by Newton-Raphson)
-----------------------------------------------------------------
A  Snare front end (C6): M253 pin 5, C22/R48, D6/R21 (HB line), TR5, D7,
   R49, C27; R-type root with a 2-D Newton (_SnareFrontEnd). The R50 branch
   is a resistive source R50 with V = V_base_TR3 of the previous sample.
M  Maracas branch (C8): M253 pin 19, R43, C25 in series, D8, D9, R45, C26 and
   the ENV branch (R46 + R_th, open-circuit V_th, previous sample: C7 D4-B);
   pe78.wdf_nlroot.RTypeNLRoot with ports D8, D9.
B  Bases (C15): nodes B3 (TR3 base), N (noise node), ENV (TR4 base).
   Leaves: R50 source (V = V_C27 of this sample), R51, C28 (B3-N), the noise
   Norton (VR8U, V = i_B1*VR8U) at N, C29 (N-ENV), the LC and SC subtrees,
   R52, the MR port (R46, V = V_C26 of the previous sample). Ports: TR3 and
   TR4 base-emitter (Shockley, Is_b, Vt), RTypeNLRoot with a 2-D Newton.
   The bases are coupled without delay through C28-N-C29; the noise goes
   through the Shockley laws (no small-signal model: C10).
C  Collectors and output (C15, D3-A): one linear WDF, root RTypeAdaptor at the
   wiper W of VR9 with three branches:
     VR9U + (L1 || C30 || R53 with Ic3)      TR3 collector
     VR9L + (L2 || R54 with Ic4)             TR4 collector
     R55 + C31 + RLoad                       output
   Ic = hFE*Ib (total, D2-A): the envelope current through L1/L2 (the "thump"
   and the L1-C30 ring at 3.56 kHz) and the noise are both in it. The +12 V
   rails are AC ground (collector voltages are deviations from +12 V).

Coupling: A -> B through V_C27 (same sample), B -> A through V_base_TR3
(one sample, measured 3e-4 V on V_C27, C6); M <-> ENV one sample (C9).
TR3/TR4 are linear in the collector (no saturation, no high injection):
tests/test_linear_region.py.

VR9: one 220 kOhm linear pot (user, 2026-11-20); vr9_pos = VR9U/VR9, default
25/220 (VR9U 25 kOhm, VR9L 195 kOhm as in the .asc).
"""

import numpy as np
import pywdf.core.wdf as _wdf
import pywdf.core.rtype as _rtype

from pe78.noise import noise_amp, VR8U as _VR8U
from pe78.wdf_nlroot import RTypeNLRoot, diode_law, lexp

Resistor               = _wdf.Resistor
Capacitor              = _wdf.Capacitor
Inductor               = _wdf.Inductor
ResistiveVoltageSource = _wdf.ResistiveVoltageSource
SeriesAdaptor          = _wdf.SeriesAdaptor
ParallelAdaptor        = _wdf.ParallelAdaptor
RootRTypeAdaptor       = _rtype.RootRTypeAdaptor


def _par_scatter(adaptor):
    """S[i][j] = 2*G[j]/Gt - delta(i,j) for a parallel junction; Rp = 1/Gt."""
    G = [1.0 / p.Rp for p in adaptor.down_ports]
    Gt = sum(G)
    n = len(G)
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            S[i][j] = 2.0 * G[j] / Gt - (1.0 if i == j else 0.0)
    adaptor.set_S_matrix(S)
    adaptor.Rp = 1.0 / Gt


# ---------------------------------------------------------------------------
# A: snare front end (C6)
# ---------------------------------------------------------------------------
class _SnareFrontEnd:
    """
    M253 pin 5 -> C22 -> TR5 base (R48 to GND); D6 into the HB line (R21);
    TR5 emitter -> D7 -> R49 -> C27 (to GND) -> R50 branch.

    Leaves (pywdf) joined by an R-type junction (nodes v1, vb, vd, vc, h).
    Nonlinear ports: A = TR5 base-emitter in series with D7 (current Ib),
    B = TR5 collector current into vd (-Ic), D6. Newton on (w, v6), w = V_BE
    of TR5 (current-controlled composite port, Werner 2017); the emitter node
    is eliminated. TR5: Ebers-Moll forward active with IS, NF, BF of the
    LTspice BC547B card (BC108B in the schematic; collector on the +5 V rail,
    7805, 4.7 V on load: forward active only, the rail does not enter).
    1N4148: LTspice standard.dio.
    """

    C22, R48, R21 = 0.1e-6, 22e3, 12e3
    R49, C27, R50 = 2200.0, 0.05e-6, 1e6
    VT_27C = 25.852e-3
    D_IS, D_N = 2.52e-9, 1.752
    TR5_IS, TR5_NF, TR5_BF = 2.39e-14, 1.008, 294.3
    FE_MAXIT, FE_TOL = 100, 1e-10

    def __init__(self, fs, r_on, r_off):
        self.fs = fs
        self.R_ON, self.R_OFF = r_on, r_off
        self._on = False
        self.V_c27 = 0.0
        self.V_b5 = 0.0
        self.w = 0.0
        self.v6 = 0.0
        self.iter_max = 0
        self.build()

    def build(self):
        self.m253 = ResistiveVoltageSource(self.R_OFF)
        self.c22 = Capacitor(self.C22, self.fs)
        self.r48 = Resistor(self.R48)
        self.r21 = Resistor(self.R21)
        self.r49 = Resistor(self.R49)
        self.c27 = Capacitor(self.C27, self.fs)
        self.r50 = ResistiveVoltageSource(self.R50)      # V = V_base_TR3[n-1]
        self._on = False
        V1, VB, VD, VC, H = range(5)
        self.leaves = [(self.m253, V1, -1), (self.c22, V1, VB), (self.r48, VB, -1),
                       (self.r21, H, -1), (self.r49, VD, VC), (self.c27, VC, -1),
                       (self.r50, VC, -1)]
        ports = [(VB, VD), (VD, -1), (V1, H)]                # A, B, D6

        def inc(p, q):
            col = np.zeros(5)
            col[p] = 1.0
            if q >= 0:
                col[q] = -1.0
            return col
        self.AL = np.array([inc(p, q) for _, p, q in self.leaves]).T
        self.AN = np.array([inc(p, q) for p, q in ports]).T
        self._cache = {}

    def _matrices(self):
        R = np.array([leaf.Rp for leaf, _, _ in self.leaves])
        key = tuple(R)
        m = self._cache.get(key)
        if m is None:
            Ginv = np.linalg.inv(self.AL @ np.diag(1.0 / R) @ self.AL.T)
            B = self.AL @ np.diag(1.0 / R)
            m = (Ginv @ B, Ginv @ self.AN, self.AN.T @ Ginv @ B, self.AN.T @ Ginv @ self.AN)
            self._cache[key] = m
        return m

    def step(self, on, v_on, v_base3_prev):
        """Advance one sample; returns V_C27."""
        if on != self._on:
            self.m253.set_resistance(self.R_ON if on else self.R_OFF)
            self._on = on
        self.m253.set_voltage(v_on if on else 0.0)
        self.r50.set_voltage(v_base3_prev)
        b = np.array([leaf.propagate_reflected_wave() for leaf, _, _ in self.leaves])
        Kb, Ki, H, Z = self._matrices()
        h = H @ b
        vt_t = self.TR5_NF * self.VT_27C
        vt_d = self.D_N * self.VT_27C
        k_e = self.TR5_IS * (1.0 + 1.0 / self.TR5_BF)
        k_b = 1.0 / (self.TR5_BF + 1.0)
        w, v6 = self.w, self.v6
        for it in range(1, self.FE_MAXIT + 1):
            ex, dex = lexp(w / vt_t)
            ie, die = k_e * (ex - 1.0), k_e * dex / vt_t
            ib, dib = k_b * ie, k_b * die
            ic, dic = ie - ib, die - dib
            u = 1.0 + ie / self.D_IS
            vA = w + vt_d * np.log(u)
            dvA = 1.0 + vt_d * die / (self.D_IS * u)
            e6, de6 = lexp(v6 / vt_d)
            i6, di6 = self.D_IS * (e6 - 1.0), self.D_IS * de6 / vt_d
            iN = (ib, -ic, i6)
            fA = vA - (h[0] - Z[0, 0] * iN[0] - Z[0, 1] * iN[1] - Z[0, 2] * iN[2])
            f6 = v6 - (h[2] - Z[2, 0] * iN[0] - Z[2, 1] * iN[1] - Z[2, 2] * iN[2])
            jAw = dvA + Z[0, 0] * dib - Z[0, 1] * dic
            jA6 = Z[0, 2] * di6
            j6w = Z[2, 0] * dib - Z[2, 1] * dic
            j66 = 1.0 + Z[2, 2] * di6
            det = jAw * j66 - jA6 * j6w
            dw = -(fA * j66 - jA6 * f6) / det
            d6 = -(jAw * f6 - j6w * fA) / det
            dw = min(max(dw, -0.2), 0.2)
            d6 = min(max(d6, -0.2), 0.2)
            w += dw
            v6 += d6
            if abs(dw) < self.FE_TOL and abs(d6) < self.FE_TOL:
                break
        else:
            raise RuntimeError("snare front end: Newton did not converge")
        self.iter_max = max(self.iter_max, it)
        self.w, self.v6 = w, v6
        ex, _ = lexp(w / vt_t)
        ie = k_e * (ex - 1.0)
        ib = k_b * ie
        e6, _ = lexp(v6 / vt_d)
        self.i_ports = np.array([ib, -(ie - ib), self.D_IS * (e6 - 1.0)])
        self.v_nodes = Kb @ b - Ki @ self.i_ports
        for (leaf, _, _), vk, bk in zip(self.leaves, self.AL.T @ self.v_nodes, b):
            leaf.accept_incident_wave(2.0 * vk - bk)
        self.V_b5 = self.v_nodes[1]
        self.V_c27 = self.c27.wave_to_voltage()
        return self.V_c27


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class NoiseVoices:
    """
    Snare drum, long cymbal, short cymbal and maracas of PE78 Fig. 6.

    Observable state (after each tick):
        _V_c27, _V_b5            snare front end [V]
        _V_b3, _V_n, _V_env      TR3 base, noise node N (deviation from
                                 +12 V), TR4 base [V]
        _V_c23, _V_c24, _V_c25, _V_c26, _V_a    cymbal/maracas capacitors, node A
        _Ic3, _Ic4               collector currents [A]
        _V_col3, _V_col4         collector voltages (deviation from +12 V) [V]
        _V_out                   output on RLoad [V]
    """

    V_ON, R_ON, R_OFF = 4.5, 350.0, 10e6      # M253 outputs

    # Bases
    R51, R52 = 1e6, 1e6
    C28, C29 = 4.7e-9, 4.7e-9
    VR8U = _VR8U
    hFE, Vt = 200.0, 26e-3
    Is_b3 = 7.035e-17       # [A] TR3 base (BC108B; IS/BF order of the BC547B card)
    Is_b4 = 7.035e-17       # [A] TR4 base, calibrated: V_ENV ~ 620 mV with LC (LTspice)
    R_PI_MIN, R_PI_MAX = 500.0, 10e6

    # Cymbal branches (C7)
    R47, C23 = 1e6, 0.33e-6
    R42, C24, R44 = 10e3, 68e-9, 1e6
    # Maracas (C8)
    R43, C25, R45, C26, R46 = 22e3, 0.22e-6, 100e3, 0.1e-6, 470e3
    VT_27C, D_IS, D_N = 25.852e-3, 2.52e-9, 1.752

    # Collectors and output
    Vcc = 12.0
    L1, R53, C30 = 100e-3, 4700.0, 0.02e-6
    L2, R54 = 100e-3, 4700.0
    VR9 = 220e3             # [Ohm] balance pot, linear (user)
    VR9_POS_DEFAULT = 25e3 / 220e3
    R55, C31, Rload = 22e3, 0.1e-6, 10e3

    def __init__(self, fs, vr9_pos=None):
        self.fs = int(fs)
        self.NOISE_AMP = noise_amp(self.fs)
        self.vr9_pos = self.VR9_POS_DEFAULT if vr9_pos is None else float(vr9_pos)
        self._on = {"lc": False, "sc": False, "mr": False}
        self._V_b3 = self._V_n = self._V_env = 0.0
        self._V_c23 = self._V_c24 = self._V_c25 = self._V_c26 = self._V_a = 0.0
        self._Ic3 = self._Ic4 = 0.0
        self._V_col3 = self._V_col4 = self._V_out = 0.0
        self._r_pi4 = self.R_PI_MAX
        self.sd = _SnareFrontEnd(self.fs, self.R_ON, self.R_OFF)
        self._build_cymbal_branches()
        self._build_mr()
        self._build_bases()
        self._build_collectors()

    # ------------------------------------------------------------------
    @property
    def _V_c27(self):
        return self.sd.V_c27

    @property
    def _V_b5(self):
        return self.sd.V_b5

    def set_vr9(self, pos):
        """VR9 wiper position 0..1 (VR9U = pos*VR9, VR9L = (1-pos)*VR9)."""
        self.vr9_pos = float(pos)
        self._vr9u.set_resistance(max(self.vr9_pos * self.VR9, 1.0))
        self._vr9l.set_resistance(max((1.0 - self.vr9_pos) * self.VR9, 1.0))

    # ------------------------------------------------------------------
    def tick(self, v_sd, v_lc, v_sc, v_mr, noise=None):
        if noise is None:
            noise = np.random.standard_normal()
        on = {"lc": v_lc > 0.5, "sc": v_sc > 0.5, "mr": v_mr > 0.5}
        for k, src in (("lc", self._m253_lc), ("sc", self._m253_sc), ("mr", self._m253_mr)):
            if on[k] != self._on[k]:
                src.set_resistance(self.R_ON if on[k] else self.R_OFF)
                self._on[k] = on[k]
        self._m253_lc.set_voltage(self.V_ON if on["lc"] else 0.0)
        self._m253_sc.set_voltage(-(self.V_ON if on["sc"] else 0.0))   # source on the GND side
        self._m253_mr.set_voltage(self.V_ON if on["mr"] else 0.0)

        # A: snare front end (TR3 base of the previous sample)
        v_c27 = self.sd.step(v_sd > 0.5, self.V_ON, self._V_b3)

        # MR port of the bases root: C26 of the previous sample
        self._mr_port.set_voltage(self._V_c26)

        # M: maracas branch, ENV seen as a Thevenin of the previous sample
        R_th = 1.0 / (1.0 / (self.R47 + self._m253_lc.Rval)
                      + 1.0 / (self.R42 + self.R44 + self._m253_sc.Rval)
                      + 1.0 / self.R52 + 1.0 / self._r_pi4)
        i_r46 = (self._V_c26 - self._V_env) / self.R46
        self._env_mr.set_resistance(self.R46 + R_th)
        self._env_mr.set_voltage(self._V_env - R_th * i_r46)
        self._root_mr.compute()
        self._V_c26 = self._c26.wave_to_voltage()
        self._V_c25 = self._c25.wave_to_voltage()
        self._V_a = self._root_mr.v_nodes[1]

        # B: bases (TR3, TR4) with the shared noise node
        self._r50src.set_voltage(v_c27)
        self._noise_src.set_voltage(noise * self.NOISE_AMP * self.VR8U)
        self._root_b.compute()
        vb3, vn, venv = self._root_b.v_nodes
        self._V_b3, self._V_n, self._V_env = vb3, vn, venv
        self._V_c23 = self._c23.wave_to_voltage()
        self._V_c24 = self._c24.wave_to_voltage()
        ib3 = self._root_b.i_ports[0]
        ib4 = self._root_b.i_ports[1]
        self._Ic3 = self.hFE * ib3
        self._Ic4 = self.hFE * ib4
        ic4 = max(self._Ic4, 1e-15)
        self._r_pi4 = float(np.clip(self.hFE * self.Vt / ic4, self.R_PI_MIN, self.R_PI_MAX))

        # C: collectors and output
        return self._step_collectors(self._Ic3, self._Ic4)

    def _step_collectors(self, ic3, ic4):
        # Norton -> Thevenin: Ic flows out of the collector node (into TR),
        # with R53 (R54) in parallel: V_th = -Ic*R53.
        self._tr3_src.set_voltage(-ic3 * self.R53)
        self._tr4_src.set_voltage(-ic4 * self.R54)
        self._root_c.compute()
        self._V_col3 = self._c30.wave_to_voltage()
        self._V_col4 = self._l2.wave_to_voltage()
        # RLoad sits inside two series adaptors: its port voltage is the
        # opposite of the physical O node voltage (checked against
        # tools/noise_voices_ref.py).
        self._V_out = -self._rload.wave_to_voltage()
        return self._V_out

    # ------------------------------------------------------------------
    def _shockley(self, v, Is):
        e, de = lexp(v / self.Vt)
        return Is * (e - 1.0), Is * de / self.Vt

    def _build_cymbal_branches(self):
        """LC and SC subtrees (as cymb.py, C7): pywdf series/parallel adaptors."""
        self._m253_lc = ResistiveVoltageSource(self.R_OFF)
        self._c23 = Capacitor(self.C23, self.fs)
        self._r47 = Resistor(self.R47)
        self._ser_lc = SeriesAdaptor(ParallelAdaptor(self._m253_lc, self._c23), self._r47)
        self._m253_sc = ResistiveVoltageSource(self.R_OFF)
        self._r42 = Resistor(self.R42)
        self._c24 = Capacitor(self.C24, self.fs)
        self._r44 = Resistor(self.R44)
        self._ser_sc = SeriesAdaptor(
            self._r44, ParallelAdaptor(self._c24, SeriesAdaptor(self._r42, self._m253_sc)))
        self._r52 = Resistor(self.R52)
        self._mr_port = ResistiveVoltageSource(self.R46)    # V = V_C26[n-1]

    def _build_mr(self):
        """Maracas branch (C8): RTypeNLRoot with D8, D9."""
        self._m253_mr = ResistiveVoltageSource(self.R_OFF)
        self._r43 = Resistor(self.R43)
        self._c25 = Capacitor(self.C25, self.fs)
        self._r45 = Resistor(self.R45)
        self._c26 = Capacitor(self.C26, self.fs)
        self._env_mr = ResistiveVoltageSource(self.R46 + 1.0 / (1.0 / self.R47 + 1.0 / self.R52))
        S, A, K, M = range(4)
        nvt = self.D_N * self.VT_27C
        self._root_mr = RTypeNLRoot(
            4,
            [(self._m253_mr, S, -1), (self._r43, S, -1), (self._c25, S, A),
             (self._r45, K, M), (self._c26, M, -1), (self._env_mr, M, -1)],
            [(A, -1, diode_law(self.D_IS, nvt, reverse=True)),    # D8
             (A, K, diode_law(self.D_IS, nvt))])                  # D9

    def _build_bases(self):
        """Bases root: nodes B3, N, ENV; ports TR3 and TR4 base (C15)."""
        self._r50src = ResistiveVoltageSource(self.sd.R50)      # V = V_C27[n]
        self._r51 = Resistor(self.R51)
        self._c28 = Capacitor(self.C28, self.fs)
        self._noise_src = ResistiveVoltageSource(self.VR8U)    # Norton of B1 with VR8U
        self._c29 = Capacitor(self.C29, self.fs)
        B3, N, E = range(3)
        # The LC/SC subtrees have the voltage -V_env on their port (series
        # adaptor convention, as in cymb.py): connected from GND to ENV.
        leaves = [(self._r50src, B3, -1), (self._r51, B3, -1), (self._c28, B3, N),
                  (self._noise_src, N, -1), (self._c29, N, E),
                  (self._ser_lc, -1, E), (self._ser_sc, -1, E),
                  (self._r52, E, -1), (self._mr_port, E, -1)]
        ports = [(B3, -1, lambda v: self._shockley(v, self.Is_b3)),
                 (E, -1, lambda v: self._shockley(v, self.Is_b4))]
        self._root_b = RTypeNLRoot(3, leaves, ports)

    def _build_collectors(self):
        """Collectors, VR9, R55, C31, RLoad: one linear WDF rooted at the wiper W."""
        self._tr3_src = ResistiveVoltageSource(self.R53)    # Ic3 || R53
        self._l1 = Inductor(self.L1, self.fs)
        self._c30 = Capacitor(self.C30, self.fs)
        self._tr4_src = ResistiveVoltageSource(self.R54)    # Ic4 || R54
        self._l2 = Inductor(self.L2, self.fs)
        self._vr9u = Resistor(max(self.vr9_pos * self.VR9, 1.0))
        self._vr9l = Resistor(max((1.0 - self.vr9_pos) * self.VR9, 1.0))
        self._r55 = Resistor(self.R55)
        self._c31 = Capacitor(self.C31, self.fs)
        self._rload = Resistor(self.Rload)
        col3 = ParallelAdaptor(ParallelAdaptor(self._tr3_src, self._l1), self._c30)
        col4 = ParallelAdaptor(self._tr4_src, self._l2)
        br3 = SeriesAdaptor(self._vr9u, col3)
        br4 = SeriesAdaptor(self._vr9l, col4)
        brout = SeriesAdaptor(self._r55, SeriesAdaptor(self._c31, self._rload))
        self._root_c = RootRTypeAdaptor([br3, br4, brout], _par_scatter)
